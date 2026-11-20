"""Paired B300 tile measurements for inference, CUDA Graph decode and training."""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.models import add_model_arguments, resolve_model
from benchmarks.sm100.hardware import check_blackwell
from weftmoe.blackwell_tuning import STRICT_BLOCK_M, SUPPORTED_BLOCK_M, source_digest
from weftmoe.nvfp4 import prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime


def tile_policies(args):
    policies = {} if args.strict_only else {"dynamic": (0, False)}
    for m in args.fixed_m:
        policies[f"fixed-{m}"] = (m, False)
    for m in args.full_m:
        policies[f"full-{m}"] = (m, True)
    return policies


@torch.no_grad()
def measure(args, group):
    import deep_gemm as dg

    ep, rank = group.size(), group.rank()
    h, i, e, k = args.hidden, args.intermediate, args.experts, args.top_k
    cap = max(args.tokens)
    torch.manual_seed(args.seed + rank)
    device = torch.device("cuda", torch.cuda.current_device())
    rt = TensorRuntime(
        group=group,
        experts=e,
        top_k=k,
        hidden=h,
        intermediate=i,
        max_tokens=cap,
        max_rows=cap * ep * k,
        training=args.mode == "training",
        decode_only=args.mode == "decode",
        precision=args.precision,
        batch_invariant=args.strict_only,
        blackwell_block_m=STRICT_BLOCK_M,
    )
    windows = getattr(rt.native, "windows", (rt.native,))
    w1 = torch.randn(e // ep, 2 * i, h, device=device, dtype=torch.bfloat16) * 0.03
    w2 = torch.randn(e // ep, h, i, device=device, dtype=torch.bfloat16) * 0.03
    if args.precision == "nvfp4" and not rt.training:
        packed = prepare_nvfp4_weights(w1, w2)
    x = torch.empty((cap, h), device=device, dtype=torch.bfloat16)
    ids = torch.empty((cap, k), device=device, dtype=torch.int32)
    gates = torch.empty((cap, k), device=device, dtype=torch.float)
    bias = (
        torch.randn(
            e,
            device=device,
            generator=torch.Generator(device=device).manual_seed(args.seed),
        )
        * args.route_bias_std
    )
    dy = torch.empty_like(x) if rt.training else None
    policies = tile_policies(args)

    def select(policy):
        # Diagnostic override between synchronized launches. Reuse the same registered
        # addresses, weight tensors and GIN windows for every member of the pair.
        for window in windows:
            window.buffer.fixed_block_m, window.buffer.full_mma_tiles = policy

    def fill(n):
        x[:n].normal_(0, 0.03)
        if rt.training:
            dy[:n].normal_(0, 0.03)
        scores, indices = (torch.randn(n, e, device=device) + bias).topk(k)
        ids[:n].copy_(indices)
        gates[:n].copy_(scores.softmax(-1))

    def run(n):
        if rt.training:
            tape = rt.forward_async(x[:n], ids[:n], gates[:n], w1, w2)
            return (tape.output, *rt.backward_async(tape, dy[:n]))
        if args.precision == "nvfp4":
            return (rt.inference_nvfp4(0, x[:n], ids[:n], gates[:n], packed),)
        return (rt.inference(0, x[:n], ids[:n], gates[:n], w1, w2),)

    def use_graph(n):
        return args.mode == "decode" or (args.mode == "mixed" and n <= 512)

    timings = {n: {p: [] for p in policies} for n in args.tokens}
    graphs = {}
    for n in args.tokens:
        fill(n)
        for name in policies:
            select(policies[name])
            out = run(n)
            torch.cuda.synchronize()
            if use_graph(n):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = run(n)
                graphs[n, name] = graph, out
        if rank == 0:
            print(f"Warmed {args.model} {args.precision} EP{ep} tokens={n}", flush=True)

    names = list(policies)
    checks = {n: {p: 0 for p in policies} for n in args.tokens}
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    for step in range(args.warmups + args.samples):
        for n in args.tokens:
            fill(n)
            select((STRICT_BLOCK_M, True) if args.canonical_reference else (0, False))
            expected = tuple(t.clone() for t in run(n))
            assert all(torch.isfinite(t).all() for t in expected)
            if args.strict_only and not rt.training:
                prefix = run(n // 2)[0]
                assert torch.equal(
                    prefix.view(torch.int16),
                    expected[0][: len(prefix)].view(torch.int16),
                )
            # A balanced rotation avoids giving one variant a fixed clock/cache advantage.
            shift = step % len(names)
            order = names[shift:] + names[:shift]
            if (step // len(names)) % 2:
                order.reverse()
            for name in order:
                select(policies[name])
                dist.barrier()
                torch.cuda.synchronize()
                if use_graph(n):
                    graph, out = graphs[n, name]
                    start.record()
                    for _ in range(args.replays):
                        graph.replay()
                    stop.record()
                    stop.synchronize()
                    us = start.elapsed_time(stop) * 1000 / args.replays
                else:
                    begin = time.perf_counter_ns()
                    out = run(n)
                    torch.cuda.synchronize()
                    us = (time.perf_counter_ns() - begin) / 1000
                if step >= args.warmups:
                    timings[n][name].append(us)
                for value, ref in zip(out, expected, strict=True):
                    bits = torch.int16 if value.element_size() == 2 else torch.int32
                    checks[n][name] += int(
                        torch.count_nonzero(
                            value.contiguous().view(bits) != ref.contiguous().view(bits)
                        )
                    )
    gathered = [None] * ep
    dist.all_gather_object(gathered, (timings, checks))
    source = None
    if args.canonical_reference:
        sources = [None] * ep
        dist.all_gather_object(sources, source_digest())
        if len(set(sources)) != 1:
            raise RuntimeError("EP ranks use different Blackwell kernel sources")
        source = sources[0]
    if rank == 0:
        for n in args.tokens:
            buf = windows[0].buffer
            launched = len(windows[0].dummy) if hasattr(windows[0], "dummy") else n
            dynamic_m = dg.get_block_m_for_mega_moe(
                buf.group.size(),
                buf.num_experts,
                buf.num_max_tokens_per_rank,
                launched,
                k,
                buf.mma_type,
            )
            for name, (m, full) in policies.items():
                samples = list(map(max, zip(*(t[n][name] for t, _ in gathered))))
                print(
                    "TILE_M_JSON "
                    + json.dumps(
                        dict(
                            model=args.model,
                            profile_key=rt.profile_key,
                            source=source,
                            precision=args.precision,
                            ep=ep,
                            capacity=cap,
                            tokens=n,
                            policy=name,
                            block_m=m or dynamic_m,
                            full_mma_tiles=full,
                            n_tile=128,
                            k_tile=(256 if i % 256 == 0 else 128)
                            if args.precision == "nvfp4"
                            else 64,
                            seed=args.seed,
                            warmups=args.warmups,
                            replays=args.replays if use_graph(n) else 1,
                            samples_us=samples,
                            p50_us=statistics.median(samples),
                            mismatched_elements=sum(c[n][name] for _, c in gathered),
                        )
                    ),
                    flush=True,
                )
    graphs.clear()
    rt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument(
        "--mode", choices=("mixed", "inference", "decode", "training"), default="mixed"
    )
    parser.add_argument("--canonical-reference", action="store_true")
    parser.add_argument("--strict-only", action="store_true")
    parser.add_argument("--precision", choices=("bf16", "nvfp4"), default="nvfp4")
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[8192, 20480, 1024, 64, 512]
    )
    parser.add_argument(
        "--fixed-m",
        type=int,
        choices=SUPPORTED_BLOCK_M,
        nargs="*",
        default=SUPPORTED_BLOCK_M,
    )
    parser.add_argument(
        "--full-m",
        type=int,
        choices=SUPPORTED_BLOCK_M,
        nargs="*",
        default=[STRICT_BLOCK_M],
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--route-bias-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1049)
    args = parser.parse_args()
    resolve_model(args)
    if args.strict_only and (args.fixed_m or args.full_m != [STRICT_BLOCK_M]):
        parser.error("Strict mode requires only the canonical full-width M tile")
    if (
        min(*args.tokens, args.samples, args.replays) < 1
        or args.warmups < 0
        or len(set(args.tokens)) != len(args.tokens)
    ):
        parser.error(
            "Use distinct positive token counts and positive sample/replay counts."
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    check_blackwell()
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    try:
        measure(args, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
