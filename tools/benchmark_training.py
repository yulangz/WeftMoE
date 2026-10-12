#!/usr/bin/env python3
"""Eager training FWD(+BWD) with varying tokens and synthetic routes (torchrun).

Input preparation is untimed. This exercises changing communication sizes, not
a full training step: expert weights are fixed and there is no optimizer.
Forward-only retains X/GU until device completion and reports both cache-ready
and release-inclusive wall time. Default PF timing includes weight packing.
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
    parser.add_argument("--backend", choices=("pf", "megatron"), required=True)
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
    parser.add_argument("--verify", action="store_true",
                        help="Check output/dX/dGate/dW1/dW2 against Torch instead of timing")
    parser.add_argument("--backward-identity", action="store_true",
                        help="Verify the communication-only diagnostic, not a training kernel")
    parser.add_argument("--forward-only", action="store_true",
                        help="Time training forward with saved X/GU, tape release and synchronize, without backward")
    parser.add_argument("--prepared-weights", action="store_true",
                        help="Reuse immutable PF weights; default includes per-call packing as in training")
    parser.add_argument("--trace-kernels", action="store_true",
                        help="Record CUDA entries during the first numerical check, outside performance runs")
    parser.add_argument("--profile-stages", action="store_true",
                        help="Collect cached-FWD stage counters from the diagnostic tensor adapter")
    parser.add_argument("--trace-stage-kernels", action="store_true",
                        help="Separate CUPTI diagnostic run for notify/repack/main GPU durations")
    parser.add_argument("--report-window-padding", action="store_true",
                        help="Estimate PF M128 padding for the default node-group windows, outside timing")
    parser.add_argument("--report-communication", action="store_true",
                        help="Count route-derived dispatch/return payloads outside timing")
    parser.add_argument("--timeline-dir", type=Path,
                        help="Download the final FWD timeline, outside timing; requires a timeline build")
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
    if args.backward_identity and (args.backend != "pf" or not args.verify):
        parser.error("--backward-identity requires PF --verify and an identity diagnostic build")
    if args.prepared_weights and (args.backend != "pf" or not args.forward_only):
        parser.error("--prepared-weights requires --backend pf --forward-only")
    if args.trace_kernels and not args.verify:
        parser.error("--trace-kernels requires --verify")
    if args.profile_stages and (args.backend != "pf" or not args.forward_only):
        parser.error("--profile-stages requires --backend pf --forward-only")
    if args.trace_stage_kernels and (args.backend != "pf" or not args.forward_only):
        parser.error("--trace-stage-kernels requires --backend pf --forward-only")
    if args.timeline_dir and (args.backend != "pf" or not args.forward_only or args.verify):
        parser.error("--timeline-dir requires PF forward-only diagnostics without --verify")
    if min(args.tokens) < 1 or args.samples < 1 or args.warmups < 0:
        parser.error("tokens and samples must be positive; warmups must be nonnegative")
    if not 1 <= args.top_k <= args.experts:
        parser.error("top-k must be between 1 and experts")
    if any(not math.isfinite(value) or value < 0
           for value in (args.token_std_fraction, args.route_bias_std)):
        parser.error("distribution standard deviations must be finite and nonnegative")
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
        communication = None
        if args.report_communication:
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
            # [trace step, sending rail/expert rank, dispatch rows/return rows/peer rows].
            traffic = torch.zeros((len(batches), world, 3), dtype=torch.int64)
            for step, (routes, _) in enumerate(batches):
                destination = routes.long() // owned
                nodes = destination // local_world
                for node in range(world // local_world):
                    if node != rank // local_world:
                        traffic[step, rank, 0] += (nodes == node).any(dim=1).sum()
                remote = destination[nodes != rank // local_world]
                traffic[step, :, 1] = torch.bincount(remote, minlength=world)
                peers = destination[destination != rank]
                traffic[step, :, 2] = torch.bincount(peers, minlength=world)
            traffic = traffic.cuda()
            dist.all_reduce(traffic)
            communication = dict(
                rows_per_step_rank=traffic[args.warmups:].cpu().tolist(),
                columns=["dispatch_rdma_rows", "return_rdma_rows", "off_rank_expert_rows"],
                dispatch_bytes_per_row=2 * args.hidden,
                backward_return_bytes_per_row=2 * args.hidden + 4)
        # Size both backends for the exact same trace, including skewed routes.
        # Do not assume received rows <= 2 * nominal tokens * top-k.
        incoming = torch.stack([
            torch.bincount(routes.flatten().long(), minlength=args.experts)
            for routes, _ in batches
        ]).cuda()
        dist.all_reduce(incoming, op=dist.ReduceOp.SUM)
        expert_rows = incoming.cpu()
        if communication is not None:
            communication["expert_rows_per_step"] = expert_rows[args.warmups:].tolist()
        incoming = expert_rows.reshape(len(batches), world, owned).sum(dim=-1)
        max_tokens = max(max(workload["tokens"]) for workload in workloads)
        window_padding = None
        if args.report_window_padding:
            local_world = int(os.environ.get("LOCAL_WORLD_SIZE", world))
            nodes = world // local_world
            if nodes == 1:
                padded = (expert_rows + 127) // 128 * 128
                padded = padded.reshape(len(batches), world, owned).sum(-1)
            else:
                windows = math.ceil(max_tokens / args.ring_tokens)
                counts = torch.zeros(len(batches), nodes, windows, args.experts, dtype=torch.int64)
                for step, (routes, _) in enumerate(batches):
                    for window, chunk in enumerate(routes.split(args.ring_tokens)):
                        counts[step, rank // local_world, window] = torch.bincount(
                            chunk.flatten().long(), minlength=args.experts)
                counts = counts.cuda()
                dist.all_reduce(counts)
                padded = ((counts + 127) // 128 * 128).reshape(
                    len(batches), nodes, windows, world, owned).sum((1, 2, 4))
                assert torch.equal(counts.sum((1, 2)).cpu(), expert_rows)
            window_padding = dict(actual_rows=incoming[args.warmups:].tolist(),
                                  m128_rows=padded[args.warmups:].tolist(),
                                  canonical_m128_rows=(
                                      ((expert_rows + 127) // 128 * 128)
                                      .reshape(len(batches), world, owned)
                                      .sum(-1)[args.warmups:].tolist()))
        max_rows = int(incoming.max()) + owned * 128
        if args.backend == "megatron":
            from benchmarks.megatron_training_backend import MegatronTrainingBackend
            runtime = MegatronTrainingBackend(args, w1, w2)
        else:
            runtime = TensorRuntime(
                group=dist.group.WORLD, experts=args.experts, top_k=args.top_k,
                hidden=args.hidden, intermediate=args.intermediate, max_tokens=max_tokens,
                max_rows=max_rows, training=True, max_saved=2)
        if args.verify:
            from benchmarks.training_reference import verify_step
            for step, (cpu_routes, cpu_gates) in enumerate(batches):
                x, dy = rand(cpu_routes.shape[0], args.hidden), rand(cpu_routes.shape[0], args.hidden)
                probe = None
                if args.trace_kernels and step == 0:
                    probe = torch.profiler.profile(activities=[
                        torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                    probe.start()
                if args.backward_identity:
                    tape = runtime.forward(x, cpu_routes.cuda(), cpu_gates.cuda(), w1, w2)
                    dx, dg, dw1, dw2 = runtime.backward(tape, dy)
                    expected = (dy.float() * args.top_k).bfloat16()
                    torch.testing.assert_close(dx, expected, rtol=0, atol=0)
                    torch.testing.assert_close(dg, torch.zeros_like(dg), rtol=0, atol=0)
                    errors = {"identity_dx_exact": True, "identity_dgate_zero": True}
                    del tape, dx, dg, dw1, dw2
                else:
                    errors = verify_step(runtime, x, cpu_routes.cuda(), cpu_gates.cuda(), w1, w2, dy)
                if probe is not None:
                    probe.stop()
                    kernels = sorted({event.name for event in probe.events()
                                      if event.device_type == torch.autograd.DeviceType.CUDA})
                    print("TRAINING_KERNEL_JSON " + json.dumps(dict(
                        rank=rank, tokens=tokens, kernels=kernels)), flush=True)
                per_rank_errors = [None] * world
                dist.all_gather_object(per_rank_errors, errors)
                if rank == 0:
                    print("TRAINING_PARITY_JSON " + json.dumps(dict(
                        nominal_tokens_per_rank=tokens, step=step,
                        errors=per_rank_errors)), flush=True)
            runtime.close()
            continue
        samples = []
        forward_samples = []
        forward_ready_samples = []
        stage_profiles = []
        measurement_intervals_ns = []
        priming_steps = len(batches) if args.prime_trace else 0
        input_rng_state = generator.get_state()
        compile_graphs_before = 0
        allocator_before = None
        kernel_probe = None
        for iteration, (cpu_routes, cpu_gates) in enumerate(batches * (2 if args.prime_trace else 1)):
            if args.prime_trace and iteration == priming_steps:
                generator.set_state(input_rng_state)
            if args.backend == "megatron" and iteration == priming_steps + args.warmups:
                compile_graphs_before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
            if (args.profile_stages or args.trace_stage_kernels) and iteration == priming_steps + args.warmups:
                allocator_before = torch.cuda.memory_stats()
                if args.trace_stage_kernels:
                    kernel_probe = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CUDA])
                    kernel_probe.start()
            routes, gates = cpu_routes.cuda(), cpu_gates.cuda()
            x, dy = rand(routes.shape[0], args.hidden), rand(routes.shape[0], args.hidden)
            dist.barrier()
            torch.cuda.synchronize()
            interval_start = time.time_ns()
            start = time.perf_counter_ns()
            if args.prepared_weights:
                runtime._check(x, routes, gates, w1, w2)
                tape = runtime.native.forward(x, routes, gates, w1, w2, layer=0)
            else:
                tape = runtime.forward(x, routes, gates, w1, w2)
            forward_end = time.perf_counter_ns()
            if args.forward_only:
                # Keep the checkpoint alive until it is usable by Backward.
                torch.cuda.synchronize()
                forward_ready_us = (time.perf_counter_ns() - start) / 1000
            else:
                gradients = runtime.backward(tape, dy)
                del gradients
            profile = tape.profile if args.profile_stages else None
            if not args.timeline_dir:
                del tape
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 1000
            if iteration >= priming_steps + args.warmups:
                samples.append(elapsed)
                forward_samples.append((forward_end - start) / 1000)
                if args.forward_only:
                    forward_ready_samples.append(forward_ready_us)
                if args.profile_stages:
                    stage_profiles.append(profile)
                measurement_intervals_ns.append((interval_start, time.time_ns()))
            if args.timeline_dir:
                if iteration == len(batches) * (2 if args.prime_trace else 1) - 1:
                    args.timeline_dir.mkdir(parents=True, exist_ok=True)
                    snapshot = tape.timeline
                    (args.timeline_dir / f"t{tokens}.rank{rank}.json").write_text(json.dumps(snapshot))
                del tape
            del x, dy, routes, gates
        allocator_delta = None
        stage_kernels = None
        if kernel_probe is not None:
            kernel_probe.stop()
            stage_kernels = {}
            local = world == int(os.environ.get("LOCAL_WORLD_SIZE", world))
            split_notify = runtime.native.usage().get("separate_forward_notify",
                bool(os.environ.get("WEFTMOE_PF_SPLIT_NOTIFY")))
            for label, name in (("mini_notify", "training_count_notify_kernel"),
                                ("standalone_notify", "weftmoe_pf::notify_kernel"),
                                ("weight_repack", "interleave_forward_w1_kernel"),
                                ("main", "weftmoe_pf::local::kernel" if local
                                 else "weftmoe_pf::network::kernel")):
                events = sorted((event for event in kernel_probe.events()
                                 if event.device_type == torch.autograd.DeviceType.CUDA
                                 and name in event.name), key=lambda event: event.time_range.start)
                expected = 0 if label == "weight_repack" and args.prepared_weights else args.samples
                if ((label == "mini_notify" and split_notify) or
                        (label == "standalone_notify" and not split_notify)):
                    expected = 0
                assert len(events) == expected, (label, len(events), expected)
                stage_kernels[label] = [event.time_range.elapsed_us() for event in events]
        if args.profile_stages:
            allocator_after = torch.cuda.memory_stats()
            allocator_delta = {key: allocator_after[key] - allocator_before[key]
                               for key in ("num_device_alloc", "num_device_free",
                                           "num_alloc_retries", "num_ooms")}
        per_rank = [None] * world
        dist.all_gather_object(per_rank, samples)
        forward_per_rank = [None] * world
        dist.all_gather_object(forward_per_rank, forward_samples)
        ready_per_rank = [None] * world
        dist.all_gather_object(ready_per_rank, forward_ready_samples)
        profiles_per_rank = [None] * world if args.profile_stages else None
        if args.profile_stages:
            dist.all_gather_object(profiles_per_rank, stage_profiles)
        allocator_per_rank = [None] * world if args.profile_stages else None
        if args.profile_stages:
            dist.all_gather_object(allocator_per_rank, allocator_delta)
        kernels_per_rank = [None] * world if args.trace_stage_kernels else None
        if args.trace_stage_kernels:
            dist.all_gather_object(kernels_per_rank, stage_kernels)
        intervals_per_rank = [None] * world
        dist.all_gather_object(intervals_per_rank, measurement_intervals_ns)
        compile_graphs = None
        if args.backend == "megatron":
            compile_graphs = [None] * world
            dist.all_gather_object(compile_graphs,
                torch._dynamo.utils.counters["stats"]["unique_graphs"] - compile_graphs_before)
            assert not any(compile_graphs), compile_graphs
        workspace = [None] * world
        dist.all_gather_object(workspace, runtime.native.usage() if args.backend != "megatron" else None)
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
                forward_only=args.forward_only, weights_prepared_once=args.prepared_weights,
                samples=args.samples, wall_slowest_rank_p50_us=statistics.median(slowest),
                forward_ready_slowest_rank_p50_us=(statistics.median(
                    map(max, zip(*ready_per_rank))) if args.forward_only else None),
                per_rank_forward_ready_us=ready_per_rank,
                per_rank_stage_profiles=profiles_per_rank,
                window_padding=window_padding,
                communication=communication,
                per_rank_allocator_delta=allocator_per_rank,
                per_rank_stage_kernel_us=kernels_per_rank,
                forward_ready_timing_scope=("eager training forward + device synchronize; checkpoint still live"
                                            if args.forward_only else None),
                forward_slowest_rank_p50_us=statistics.median(
                    max(row[i] for row in forward_per_rank) for i in range(args.samples)),
                backward_slowest_rank_p50_us=None if args.forward_only else statistics.median(
                    max(per_rank[r][i] - forward_per_rank[r][i] for r in range(world))
                    for i in range(args.samples)),
                wall_slowest_rank_samples_us=slowest, per_rank_wall_us=per_rank,
                measurement_intervals_ns_per_rank=intervals_per_rank,
                measurement_scope=("eager training forward+release+device synchronize; input preparation excluded"
                                   if args.forward_only else
                                   "eager forward+backward+release+device synchronize; input preparation excluded"),
                forward_timing_scope="host API boundary (not synchronized CUDA phase timing)",
                checkpoint="routed X + FC1 gate/up; recompute gated SwiGLU in backward",
                implementation=getattr(runtime, "metadata", {}),
                gpu=torch.cuda.get_device_name(), per_rank_workspace=workspace,
                extension=os.environ.get("WEFTMOE_TENSOR_EXTENSION")
                          if args.backend != "megatron" else None)), flush=True)
        runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
