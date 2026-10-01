#!/usr/bin/env python3
"""Eager Forward+Backward wall time with application-owned tensors (torchrun)."""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weftmoe.runtime import TensorRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pf", "v3"), required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 20480])
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--ring-tokens", type=int, default=512)
    parser.add_argument("--ring-depth", type=int, default=8)
    args = parser.parse_args()
    if args.backend == "v3":
        os.environ["WEFTMOE_DISABLE_STATIC_FORWARD_V4"] = "1"
        os.environ.pop("WEFTMOE_STATIC_RING_TOKENS", None)
        os.environ.pop("WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS", None)
    else:
        os.environ.pop("WEFTMOE_DISABLE_STATIC_FORWARD_V4", None)
        os.environ["WEFTMOE_STATIC_RING_TOKENS"] = str(args.ring_tokens)
        os.environ["WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS"] = str(args.ring_depth)

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    generator = torch.Generator(device="cuda").manual_seed(123 + rank)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                           generator=generator) * 0.03

    owned = args.experts // world
    w1 = rand(owned, 2 * args.intermediate, args.hidden)
    w2 = rand(owned, args.hidden, args.intermediate)
    for tokens in args.tokens:
        x, dy = rand(tokens, args.hidden), rand(tokens, args.hidden)
        routes = ((torch.arange(tokens, device="cuda")[:, None] * 17 + rank * 31
                   + torch.arange(args.top_k, device="cuda")[None, :] * 53)
                  % args.experts).int()
        gates = torch.rand(tokens, args.top_k, device="cuda", generator=generator)
        runtime = TensorRuntime(
            group=dist.group.WORLD, experts=args.experts, top_k=args.top_k,
            hidden=args.hidden, intermediate=args.intermediate, max_tokens=tokens,
            max_rows=2 * tokens * args.top_k + owned * 128, training=True, max_saved=2)
        samples = []
        forward_samples = []
        for iteration in range(args.warmups + args.samples):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            tape = runtime.forward(x, routes, gates, w1, w2)
            forward_end = time.perf_counter_ns()
            gradients = runtime.backward(tape, dy)
            del tape, gradients
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1000
            if iteration >= args.warmups:
                samples.append(elapsed)
                forward_samples.append((forward_end - start) / 1000)
        per_rank = [None] * world
        dist.all_gather_object(per_rank, samples)
        forward_per_rank = [None] * world
        dist.all_gather_object(forward_per_rank, forward_samples)
        if rank == 0:
            slowest = [max(row[i] for row in per_rank) for i in range(args.samples)]
            print("TRAINING_CYCLE_JSON " + json.dumps(dict(
                backend=args.backend, world=world, tokens_per_rank=tokens,
                experts=args.experts, top_k=args.top_k, hidden=args.hidden,
                intermediate=args.intermediate, warmups=args.warmups,
                samples=args.samples, wall_slowest_rank_p50_us=statistics.median(slowest),
                forward_slowest_rank_p50_us=statistics.median(
                    max(row[i] for row in forward_per_rank) for i in range(args.samples)),
                backward_slowest_rank_p50_us=statistics.median(
                    max(per_rank[r][i] - forward_per_rank[r][i] for r in range(world))
                    for i in range(args.samples)),
                wall_slowest_rank_samples_us=slowest, per_rank_wall_us=per_rank,
                extension=os.environ["WEFTMOE_TENSOR_EXTENSION"])), flush=True)
        runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
