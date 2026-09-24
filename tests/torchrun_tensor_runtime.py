"""Distributed application-tensor forward/backward parity; run with torchrun."""

import os
import gc
import weakref

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime, moe


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    experts = int(os.environ.get("TEST_EXPERTS", max(16, world * 2)))
    tokens = int(os.environ.get("TEST_TOKENS", 8))
    top_k = int(os.environ.get("TEST_TOP_K", 4))
    max_rows = int(os.environ.get(
        "TEST_MAX_ROWS", max(2048, 2 * tokens * top_k + (experts // world) * 128)
    ))
    hidden, intermediate = 2048, 512
    generator = torch.Generator(device="cuda").manual_seed(123)

    def rand(*shape):
        return (
            torch.randn(
                *shape, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            * 0.03
        ).requires_grad_(True)

    all_x = rand(world * tokens, hidden)
    all_w1 = rand(experts, 2 * intermediate, hidden)
    all_w2 = rand(experts, hidden, intermediate)
    route = (
        torch.arange(world * tokens * top_k, device="cuda").reshape(-1, top_k) % experts
    ).int()
    if os.environ.get("TEST_ROUTE_SKEW") == "1":
        # Concentrate every route on the first rank while retaining distinct
        # top-k experts.  This covers zero-incoming-row ranks, multi-M-block
        # experts, and source-major Return intervals in the static inference
        # scheduler.
        route = torch.arange(top_k, device="cuda", dtype=torch.int32).expand(
            world * tokens, top_k
        ).contiguous()
    gate = torch.full(
        (world * tokens, top_k), 1 / top_k, device="cuda", requires_grad=True
    )
    dy = rand(world * tokens, hidden).detach()
    x = all_x.detach()[rank * tokens : (rank + 1) * tokens].clone().requires_grad_()
    w1 = all_w1.detach().chunk(world)[rank].clone().requires_grad_()
    w2 = all_w2.detach().chunk(world)[rank].clone().requires_grad_()
    g = gate.detach().chunk(world)[rank].clone().requires_grad_()
    runtime = TensorRuntime(
        group=dist.group.WORLD,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=tokens,
        max_rows=max_rows,
        training=True,
        max_saved=4,
    )
    output = moe(runtime, x, route.chunk(world)[rank].contiguous(), g, w1, w2)
    (output.float() * dy.chunk(world)[rank].float()).sum().backward()
    reference = torch.zeros_like(all_x)
    for e in range(experts):
        rows, slots = torch.where(route == e)
        ab = all_x[rows] @ all_w1[e].T
        a, b = ab.chunk(2, dim=-1)
        u = (
            torch.nn.functional.silu(a.float()) * b.float() * gate[rows, slots, None]
        ).to(torch.bfloat16)
        y = u @ all_w2[e].T
        reference.index_add_(0, rows, y)
    (reference.float() * dy.float()).sum().backward()
    checks = dict(
        output=(output, reference.chunk(world)[rank]),
        dx=(x.grad, all_x.grad.chunk(world)[rank]),
        dgate=(g.grad, gate.grad.chunk(world)[rank]),
        dw1=(w1.grad, all_w1.grad.chunk(world)[rank]),
        dw2=(w2.grad, all_w2.grad.chunk(world)[rank]),
    )
    for name, (actual, expected) in checks.items():
        delta = (actual.float() - expected.float()).abs()
        print(
            rank,
            name,
            "max_abs",
            delta.max().item(),
            "rms",
            delta.square().mean().sqrt().item(),
            flush=True,
        )
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.00002
        )
    print(rank, "TENSOR_PARITY_PASS", runtime.native.usage(), flush=True)
    del output, runtime
    # AReaL trains across EP16 but serves with independent node-local EP4
    # engines. Allow the test to exercise those distinct process groups.
    inference_world = int(os.environ.get("TEST_INFERENCE_EP_SIZE", world))
    assert world % inference_world == 0
    inference_group = dist.group.WORLD
    if inference_world != world:
        for start in range(0, world, inference_world):
            group = dist.new_group(list(range(start, start + inference_world)))
            if start <= rank < start + inference_world:
                inference_group = group
        inference_rank = dist.get_rank(inference_group)
        w1 = all_w1.detach().chunk(inference_world)[inference_rank].contiguous()
        w2 = all_w2.detach().chunk(inference_world)[inference_rank].contiguous()
    inference = TensorRuntime(
        group=inference_group,
        experts=experts,
        top_k=top_k,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=tokens,
        max_rows=max(max_rows, 2 * tokens * top_k + (experts // inference_world) * 128),
        training=False,
    )
    ids = route.chunk(world)[rank].contiguous()
    with torch.no_grad():
        # External producer provenance must not create a token/backend cycle
        # that retains application tensors and the NCCL session after completion.
        probe = x.detach().clone()
        probe_ref = weakref.ref(probe)
        inference.inference_eager(probe, ids, g, w1, w2)
        del probe
        gc.collect()
        assert probe_ref() is None, "External CUDA producer retained its input"
        # Long prompts use eager prefill. The application's largest default
        # decode bucket is 512 TP-replicated rows, or 128 per EP4 rank.
        graph_tokens = min(tokens, 128)
        x = x[:graph_tokens]
        ids = ids[:graph_tokens]
        g = g[:graph_tokens]
        # Two layers share the communication workspace and bucket staging.
        first = inference.inference(0, x, ids, g, w1, w2)
        inference.inference(1, first, ids, g, w1, w2)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.graph(graph)
        with capture:
            graph_first = inference.inference(0, x, ids, g, w1, w2)
            graph_second = inference.inference(1, graph_first, ids, g, w1, w2)
        for i in range(4):
            x.mul_(0.9)
            ids.add_(1).remainder_(experts)
            # Other buckets and eager prefill must not invalidate captured
            # metadata or retain an exclusive lease on the shared workspace.
            inference.inference(0, x[:3], ids[:3], g[:3], w1, w2)
            inference.inference_eager(x[:5], ids[:5], g[:5], w1, w2)
            expected_first = inference.inference(0, x, ids, g, w1, w2)
            expected_second = inference.inference(
                1, expected_first, ids, g, w1, w2
            ).clone()
            expected_first = expected_first.clone()
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(graph_first, expected_first, rtol=0, atol=0)
            torch.testing.assert_close(graph_second, expected_second, rtol=0, atol=0)
        benchmark_replays = int(os.environ.get("TEST_GRAPH_BENCH_REPLAYS", "0"))
        if benchmark_replays:
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier(group=inference_group)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(benchmark_replays):
                graph.replay()
            end.record()
            end.synchronize()
            per_layer_us = start.elapsed_time(end) * 1000 / benchmark_replays / 2
            timings = [None] * inference_world
            dist.all_gather_object(timings, per_layer_us, group=inference_group)
            if dist.get_rank(inference_group) == 0:
                print(
                    "TWO_LAYER_GRAPH_PER_LAYER_US",
                    max(timings),
                    "replays",
                    benchmark_replays,
                    flush=True,
                )
        print(rank, "MULTILAYER_GRAPH_PASS", inference.native.usage(), flush=True)
        del graph
    inference.close()
    del inference, graph_first, graph_second, capture
    torch.cuda.empty_cache()
    print(rank, "GRAPH_DESTROY_AND_OFFLOAD_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
