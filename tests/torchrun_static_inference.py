"""Static inference parity and workspace reuse, on one or more torchrun nodes.

The reference uses only this rank's input and all expert weights, so increasing
the EP world does not replicate the global activation workload in the test.
"""

import argparse
from datetime import timedelta
import json
import os

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime


def reference(x, routes, gates, w1, w2):
    tokens, top_k = routes.shape
    contributions = torch.empty(
        (tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device
    )
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(routes == expert)
        if not rows.numel():
            continue
        ab = x[rows] @ w1[expert].T
        a, b = ab.chunk(2, dim=-1)
        activation = (
            torch.nn.functional.silu(a.float())
            * b.float()
            * gates[rows, slots, None]
        ).to(x.dtype)
        contributions[rows, slots] = activation @ w2[expert].T
    order = routes.argsort(dim=1, stable=True)
    result = torch.zeros_like(x, dtype=torch.float32)
    rows = torch.arange(tokens, device=x.device)
    for slot in range(top_k):
        result.add_(contributions[rows, order[:, slot]].float())
    return result.to(x.dtype)


def check(actual, expected, label):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=2e-5)
    delta = (actual.float() - expected.float()).abs()
    print(json.dumps({
        "test": "STATIC_INFERENCE_PASS", "rank": dist.get_rank(),
        "case": label, "tokens": actual.shape[0],
        "max_abs": delta.max().item() if delta.numel() else 0.0,
    }), flush=True)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64, 8192, 20480])
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--edge-cases", action="store_true")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    hidden, intermediate = 2048, 512
    generator = torch.Generator(device="cuda").manual_seed(123)

    def rand(*shape):
        return torch.randn(
            *shape, device="cuda", dtype=torch.bfloat16, generator=generator
        ) * 0.03

    all_w1 = rand(args.experts, 2 * intermediate, hidden)
    all_w2 = rand(args.experts, hidden, intermediate)
    w1 = all_w1.chunk(world)[rank].contiguous()
    w2 = all_w2.chunk(world)[rank].contiguous()
    generator.manual_seed(900 + rank)
    max_tokens = max(*args.tokens, 257 if args.edge_cases else 1)
    max_rows = max(2 * max_tokens * args.top_k + (args.experts // world) * 128,
                   world * 257 * args.top_k)
    runtime = TensorRuntime(
        group=dist.group.WORLD, experts=args.experts, top_k=args.top_k,
        hidden=hidden, intermediate=intermediate, max_tokens=max_tokens,
        max_rows=max_rows,
        training=False,
    )

    def inputs(tokens, skew=False):
        x = rand(tokens, hidden)
        routes = torch.arange(tokens * args.top_k, device="cuda").reshape(
            tokens, args.top_k
        ).add_(rank * 7).remainder_(args.experts).int()
        if skew:
            # Duplicate slots plus zero-incoming destinations exercise compact
            # Return indexing and deterministic (expert, slot) reduction.
            routes.fill_(args.experts - 1)
        gates = torch.rand(tokens, args.top_k, device="cuda", generator=generator)
        gates.div_(gates.sum(dim=1, keepdim=True))
        return x, routes, gates

    for tokens in args.tokens:
        x, routes, gates = inputs(tokens)
        expected = reference(x, routes, gates, all_w1, all_w2)
        actual = runtime.inference_eager(x, routes, gates, w1, w2)
        check(actual, expected, "balanced-eager")
        actual = runtime.inference(0, x, routes, gates, w1, w2)
        check(actual, expected, "balanced-graph")

    if args.edge_cases:
        if max_tokens >= 8192:
            # Different ranks may select different GEMM entries while sharing
            # the same invocation. Empty sources are covered by eager below.
            x, routes, gates = inputs(8192 if rank % 2 else 64)
            expected = reference(x, routes, gates, all_w1, all_w2)
            check(runtime.inference_eager(x, routes, gates, w1, w2), expected,
                  "mixed-prefill-eager")
            check(runtime.inference(3, x, routes, gates, w1, w2), expected,
                  "mixed-prefill-graph")
        # Different actual lengths, including a source with no routes. Pool
        # capacity stays collective and every rank still participates.
        x, routes, gates = inputs(0 if rank == 0 else 17 + rank, skew=True)
        actual = runtime.inference_eager(x, routes, gates, w1, w2)
        check(actual, reference(x, routes, gates, all_w1, all_w2), "unequal-empty")

        x, routes, gates = inputs(257, skew=True)
        first = runtime.inference(1, x, routes, gates, w1, w2)
        second = runtime.inference(2, first, routes, gates, w1, w2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            first = runtime.inference(1, x, routes, gates, w1, w2)
            second = runtime.inference(2, first, routes, gates, w1, w2)
        for iteration in range(4):
            x.mul_(0.9)
            # Alternate a single hot expert on the last node with routes on
            # every node, keeping all graph input addresses stable.
            if iteration % 2:
                routes.copy_(torch.arange(routes.numel(), device="cuda").reshape_as(
                    routes).add_(rank).remainder_(args.experts))
            else:
                routes.fill_(args.experts - 1)
            gates.mul_(0.9)
            probe = runtime.inference_eager(x[:3], routes[:3], gates[:3], w1, w2)
            check(probe, reference(x[:3], routes[:3], gates[:3], all_w1, all_w2),
                  "interleaved-eager")
            expected_first = reference(x, routes, gates, all_w1, all_w2)
            graph.replay()
            check(first, expected_first, f"mutating-graph-{iteration}")
            expected_second = reference(first, routes, gates, all_w1, all_w2)
            check(second, expected_second, f"shared-workspace-{iteration}")
        del graph
        # Every warmed bucket of the updated layer must release its packed W1.
        # All four layers share these weights; their outer graph is gone above.
        for layer in range(4):
            runtime.invalidate_inference_layer(layer)
        all_w1.mul_(0.75)
        all_w2.mul_(0.875)
        for tokens in args.tokens:
            x, routes, gates = inputs(tokens)
            check(runtime.inference(0, x, routes, gates, w1, w2),
                  reference(x, routes, gates, all_w1, all_w2),
                  "weight-refresh-graph")
        if world * max_tokens * args.top_k > max_rows:
            x, routes, gates = inputs(max_tokens, skew=True)
            ring = int(os.getenv("WEFTMOE_STATIC_RING_TOKENS", "0")) != 0
            if ring:
                expected = reference(x, routes, gates, all_w1, all_w2)
                check(runtime.inference_eager(x, routes, gates, w1, w2), expected,
                      "hotspot-no-drop-eager")
                check(runtime.inference(4, x, routes, gates, w1, w2), expected,
                      "hotspot-no-drop-graph")
            else:
                rejected = 0
                try:
                    runtime.inference_eager(x, routes, gates, w1, w2)
                except RuntimeError as error:
                    assert "production-terminal-agreement" in str(error), str(error)
                    rejected = 1
                # The old arena rejects overflow collectively (GIN) or on the
                # overflowing destination (local). Ring must instead finish.
                failures = torch.tensor(rejected, device="cuda")
                dist.all_reduce(failures)
                network = world > int(os.environ["LOCAL_WORLD_SIZE"])
                assert failures.item() == (world if network else 1), \
                    "overflow must be reported, not silently drop routes"
            actual = runtime.inference_eager(x[:3], routes[:3], gates[:3], w1, w2)
            check(actual, reference(x[:3], routes[:3], gates[:3], all_w1, all_w2),
                  "post-hotspot-reuse" if ring else "capacity-rejection-recovery")
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
