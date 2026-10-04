#!/usr/bin/env python3
"""Eager FWD+BWD with varying per-rank tokens and synthetic routes (torchrun).

Input preparation is untimed. This exercises changing communication sizes, not
a full training step: expert weights are fixed and there is no optimizer.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weftmoe.runtime import TensorRuntime


def balanced_routes(tokens, experts, top_k, device, generator):
    """Randomize complete balanced periods; only the final period is partial."""
    period = experts // math.gcd(experts, top_k)
    blocks = math.ceil(tokens / period)
    permutations = torch.rand(blocks, experts, device=device, generator=generator).argsort(dim=-1)
    slots = torch.arange(period * top_k, device=device) % experts
    routes = permutations[:, slots].reshape(-1, top_k)[:tokens]
    return routes[torch.randperm(tokens, device=device, generator=generator)]


def make_workload(args, nominal_tokens, rank, device, route_distribution="random"):
    """Freeze a reproducible CPU route trace, without keeping activations live."""
    seed = args.seed + nominal_tokens * 1000003
    lengths = random.Random(seed + rank)
    routes_rng = torch.Generator(device=device).manual_seed(seed + rank)
    # A common per-step bias creates expert hotspots shared across source ranks.
    # A separate stream keeps it identical despite unequal token counts.
    bias_rng = torch.Generator(device=device).manual_seed(seed + 1000000007)
    balanced_rng = torch.Generator(device=device).manual_seed(seed + rank + 1000000009)
    sigma = nominal_tokens * args.token_std_fraction
    low = max(1, math.ceil(nominal_tokens - 3 * sigma))
    high = math.floor(nominal_tokens + 3 * sigma)
    batches = []
    for _ in range(args.warmups + args.samples):
        tokens = round(lengths.gauss(nominal_tokens, sigma))
        while not low <= tokens <= high:
            tokens = round(lengths.gauss(nominal_tokens, sigma))
        logits = torch.randn(tokens, args.experts, device=device, generator=routes_rng)
        logits.add_(torch.randn(args.experts, device=device, generator=bias_rng),
                    alpha=args.route_bias_std)
        scores, routes = logits.topk(args.top_k, dim=-1)
        if route_distribution == "balanced":
            # Keep lengths, selected gate scores and the input RNG unchanged.
            routes = balanced_routes(tokens, args.experts, args.top_k, device, balanced_rng)
        batches.append((routes.int().cpu(), scores.softmax(dim=-1).cpu()))
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pf", "v3", "megatron"), required=True)
    parser.add_argument("--dispatcher", choices=("alltoall", "flex"), default="flex",
                        help="Megatron communication backend: NCCL all-to-all or DeepEP")
    parser.add_argument("--deepep-num-sms", type=int, default=20)
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 20480])
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--prime-trace", action="store_true",
                        help="Replay the trace once untimed to warm dynamic compiler/allocator variants")
    parser.add_argument("--ring-tokens", type=int, default=512)
    parser.add_argument("--ring-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--token-std-fraction", type=float, default=0.1,
                        help="Per-rank normal token std / nominal tokens; truncate at +/-3 std")
    parser.add_argument("--route-bias-std", type=float, default=0.5,
                        help="Std of per-step shared expert-logit bias; 0 gives uniform routing in expectation")
    parser.add_argument("--route-distribution", choices=("random", "balanced"), default="random",
                        help="Balance expert counts in complete periods, keeping token lengths and gate scores")
    args = parser.parse_args()
    if min(args.tokens) < 1 or args.samples < 1 or args.warmups < 0:
        parser.error("tokens and samples must be positive; warmups must be nonnegative")
    if not 1 <= args.top_k <= args.experts:
        parser.error("top-k must be between 1 and experts")
    if any(not math.isfinite(value) or value < 0
           for value in (args.token_std_fraction, args.route_bias_std)):
        parser.error("distribution standard deviations must be finite and nonnegative")
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
    if args.experts % world:
        parser.error("experts must be divisible by the EP world size")
    generator = torch.Generator(device="cuda").manual_seed(args.seed + rank)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                           generator=generator) * 0.03

    owned = args.experts // world
    w1 = rand(owned, 2 * args.intermediate, args.hidden)
    w2 = rand(owned, args.hidden, args.intermediate)
    for tokens in args.tokens:
        batches = make_workload(args, tokens, rank, "cuda", args.route_distribution)
        digest = hashlib.sha256()
        for routes, gates in batches:
            digest.update(routes.shape[0].to_bytes(8, "little"))
            digest.update(routes.numpy().tobytes())
            digest.update(gates.numpy().tobytes())
        workload = dict(tokens=[routes.shape[0] for routes, _ in batches],
                        sha256=digest.hexdigest())
        workloads = [None] * world
        dist.all_gather_object(workloads, workload)
        # Size both backends for the exact same trace, including skewed routes.
        # Do not assume received rows <= 2 * nominal tokens * top-k.
        incoming = torch.stack([
            torch.bincount(routes.flatten().long(), minlength=args.experts)
            for routes, _ in batches
        ]).cuda()
        dist.all_reduce(incoming, op=dist.ReduceOp.SUM)
        expert_rows = incoming.cpu()
        incoming = expert_rows.reshape(len(batches), world, owned).sum(dim=-1)
        max_tokens = max(max(workload["tokens"]) for workload in workloads)
        max_rows = int(incoming.max()) + owned * 128
        if args.backend == "megatron":
            from benchmarks.megatron_training_backend import MegatronTrainingBackend
            runtime = MegatronTrainingBackend(args, w1, w2)
        else:
            runtime = TensorRuntime(
                group=dist.group.WORLD, experts=args.experts, top_k=args.top_k,
                hidden=args.hidden, intermediate=args.intermediate, max_tokens=max_tokens,
                max_rows=max_rows, training=True, max_saved=2)
        samples = []
        forward_samples = []
        priming_steps = len(batches) if args.prime_trace else 0
        input_rng_state = generator.get_state()
        compile_graphs_before = 0
        for iteration, (cpu_routes, cpu_gates) in enumerate(batches * (2 if args.prime_trace else 1)):
            if args.prime_trace and iteration == priming_steps:
                generator.set_state(input_rng_state)
            if args.backend == "megatron" and iteration == priming_steps + args.warmups:
                compile_graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            routes, gates = cpu_routes.cuda(), cpu_gates.cuda()
            x, dy = rand(routes.shape[0], args.hidden), rand(routes.shape[0], args.hidden)
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            tape = runtime.forward(x, routes, gates, w1, w2)
            forward_end = time.perf_counter_ns()
            gradients = runtime.backward(tape, dy)
            del tape, gradients
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1000
            if iteration >= priming_steps + args.warmups:
                samples.append(elapsed)
                forward_samples.append((forward_end - start) / 1000)
            del x, dy, routes, gates
        per_rank = [None] * world
        dist.all_gather_object(per_rank, samples)
        forward_per_rank = [None] * world
        dist.all_gather_object(forward_per_rank, forward_samples)
        compile_graphs = None
        if args.backend == "megatron":
            compile_graphs = [None] * world
            dist.all_gather_object(compile_graphs,
                torch._dynamo.utils.counters["stats"]["unique_graphs"] - compile_graphs_before)
        if rank == 0:
            slowest = [max(row[i] for row in per_rank) for i in range(args.samples)]
            measured_tokens = [workload["tokens"][args.warmups:] for workload in workloads]
            flat_tokens = [count for counts in measured_tokens for count in counts]
            print("TRAINING_CYCLE_JSON " + json.dumps(dict(
                schema_version=2, workload=f"variable_tokens_{args.route_distribution}_routes",
                backend=args.backend, world=world, nominal_tokens_per_rank=tokens,
                seed=args.seed, token_std_fraction=args.token_std_fraction,
                token_truncation_std=3, route_bias_std=args.route_bias_std,
                route_distribution=args.route_distribution,
                workload_sha256_per_rank=[workload["sha256"] for workload in workloads],
                per_rank_tokens=measured_tokens,
                per_rank_warmup_tokens=[workload["tokens"][:args.warmups]
                                       for workload in workloads],
                per_rank_received_rows=incoming[args.warmups:].T.tolist(),
                expert_rows_min_per_step=expert_rows[args.warmups:].min(dim=-1).values.tolist(),
                expert_rows_max_per_step=expert_rows[args.warmups:].max(dim=-1).values.tolist(),
                actual_tokens_min=min(flat_tokens), actual_tokens_max=max(flat_tokens),
                actual_tokens_mean=statistics.mean(flat_tokens),
                actual_tokens_std=statistics.pstdev(flat_tokens),
                max_tokens_capacity=max_tokens, max_rows_capacity=max_rows,
                input_tokens_per_second=sum(flat_tokens) * 1e6 / sum(slowest),
                experts=args.experts, top_k=args.top_k, hidden=args.hidden,
                intermediate=args.intermediate, warmups=args.warmups,
                priming_steps=priming_steps, compiled_graphs_during_measurement=compile_graphs,
                samples=args.samples, wall_slowest_rank_p50_us=statistics.median(slowest),
                forward_slowest_rank_p50_us=statistics.median(
                    max(row[i] for row in forward_per_rank) for i in range(args.samples)),
                backward_slowest_rank_p50_us=statistics.median(
                    max(per_rank[r][i] - forward_per_rank[r][i] for r in range(world))
                    for i in range(args.samples)),
                wall_slowest_rank_samples_us=slowest, per_rank_wall_us=per_rank,
                measurement_scope="eager forward+backward+release+device synchronize; input preparation excluded",
                forward_timing_scope="host API boundary (not synchronized CUDA phase timing)",
                checkpoint="routed X + FC1 gate/up; recompute gated SwiGLU in backward",
                implementation=getattr(runtime, "metadata", {}),
                extension=os.environ.get("WEFTMOE_TENSOR_EXTENSION")
                          if args.backend != "megatron" else None)), flush=True)
        runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
