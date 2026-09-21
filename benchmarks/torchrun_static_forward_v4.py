#!/usr/bin/env python3
"""Process-level A/B benchmark for the single-node inference scheduler.

Use --compare-v3 for the preferred same-process, alternating-order A/B.  A
single scheduler can also be selected with WEFTMOE_DISABLE_STATIC_FORWARD_V4.
Timings come from backend CUDA events; every reported sample is the slowest EP
rank for that collective invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256])
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--route-skew", action="store_true")
    parser.add_argument(
        "--compare-v3",
        action="store_true",
        help="interleave V4/V3 in one process and reverse order every sample",
    )
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(round((len(ordered) - 1) * fraction), len(ordered) - 1)]


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.experts % world:
        raise ValueError("experts must be divisible by the EP world size")

    hidden, intermediate = 2048, 512
    max_tokens = max(args.tokens)
    local_experts = args.experts // world
    # The skew case sends every source route to rank zero.  Capacity is sized
    # for that exact no-drop upper bound even when this run uses balanced routes.
    max_rows = max(
        2048,
        world * max_tokens * args.top_k + local_experts * 64,
    )
    generator = torch.Generator(device="cuda").manual_seed(0x57454654 + rank)
    w1 = torch.randn(
        local_experts,
        2 * intermediate,
        hidden,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    w2 = torch.randn(
        local_experts,
        hidden,
        intermediate,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    runtime = TensorRuntime(
        group=dist.group.WORLD,
        experts=args.experts,
        top_k=args.top_k,
        hidden=hidden,
        intermediate=intermediate,
        max_tokens=max_tokens,
        max_rows=max_rows,
        training=False,
    )

    selected_scheduler = (
        "v3"
        if os.environ.get("WEFTMOE_DISABLE_STATIC_FORWARD_V4")
        else "static_v4"
    )
    schedulers = ("static_v4", "v3") if args.compare_v3 else (selected_scheduler,)
    records = {scheduler: [] for scheduler in schedulers}

    def select_scheduler(scheduler):
        if scheduler == "v3":
            os.environ["WEFTMOE_DISABLE_STATIC_FORWARD_V4"] = "1"
        else:
            os.environ.pop("WEFTMOE_DISABLE_STATIC_FORWARD_V4", None)

    for tokens in args.tokens:
        x = torch.randn(
            tokens,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        if args.route_skew:
            if args.top_k > local_experts:
                raise ValueError("skew benchmark needs top-k <= local experts")
            routes = torch.arange(
                args.top_k, device="cuda", dtype=torch.int32
            ).expand(tokens, args.top_k).contiguous()
        else:
            route_offset = rank * tokens * args.top_k
            routes = (
                torch.arange(
                    route_offset,
                    route_offset + tokens * args.top_k,
                    device="cuda",
                    dtype=torch.int32,
                ).reshape(tokens, args.top_k)
                % args.experts
            ).contiguous()
        gates = torch.full(
            (tokens, args.top_k),
            1.0 / args.top_k,
            device="cuda",
            dtype=torch.float32,
        )

        for warmup in range(args.warmups):
            order = schedulers if warmup % 2 == 0 else tuple(reversed(schedulers))
            for scheduler in order:
                select_scheduler(scheduler)
                runtime.native.profile_inference(
                    x,
                    routes,
                    gates,
                    w1,
                    w2,
                    torch.cuda.current_stream().cuda_stream,
                )
        timing_names = (
            "total_us",
            "dispatch_us",
            "fc1_swiglu_us",
            "fc2_us",
            "return_us",
            "reduce_us",
        )
        samples = {scheduler: [] for scheduler in schedulers}
        for iteration in range(args.iterations):
            order = schedulers if iteration % 2 == 0 else tuple(reversed(schedulers))
            for scheduler in order:
                select_scheduler(scheduler)
                profile = runtime.native.profile_inference(
                    x,
                    routes,
                    gates,
                    w1,
                    w2,
                    torch.cuda.current_stream().cuda_stream,
                )
                samples[scheduler].append(
                    {name: float(profile[name]) for name in timing_names}
                )
        for scheduler in schedulers:
            all_samples = [None] * world
            dist.all_gather_object(all_samples, samples[scheduler])
            if rank == 0:
                global_samples = {
                    name: [
                        max(per_rank[i][name] for per_rank in all_samples)
                        for i in range(args.iterations)
                    ]
                    for name in timing_names
                }
                total = global_samples["total_us"]
                record = {
                    "tokens_per_rank": tokens,
                    "median_us": statistics.median(total),
                    "p10_us": percentile(total, 0.10),
                    "p90_us": percentile(total, 0.90),
                    "min_us": min(total),
                }
                record.update(
                    {
                        "median_" + name: statistics.median(values)
                        for name, values in global_samples.items()
                        if name != "total_us"
                    }
                )
                records[scheduler].append(record)

    runtime.close()
    if rank == 0:
        result = {
            "scheduler": "interleaved_ab" if args.compare_v3 else selected_scheduler,
            "world": world,
            "experts": args.experts,
            "top_k": args.top_k,
            "route_skew": args.route_skew,
        }
        if args.compare_v3:
            result["results_by_scheduler"] = records
        else:
            result["results"] = records[selected_scheduler]
        print(
            "STATIC_FORWARD_BENCH "
            + json.dumps(result, sort_keys=True),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
