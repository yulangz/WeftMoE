#!/usr/bin/env python3
"""Megatron-Core EP all-to-all dispatch/combine baseline.

Run with ``torchrun --standalone --nproc-per-node=8``.  The benchmark uses
Megatron-Core's production ``MoEAlltoAllTokenDispatcher`` rather than a raw
NCCL all-to-all stand-in.  The identity expert body intentionally isolates the
routing/transport slice that is comparable to MegaMOE's current synthetic
activation transport; ``intermediate`` is recorded for the future FC baseline.
"""

import argparse
import importlib.metadata
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def stats(values):
    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def measure(function, warmups, iterations):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmups):
        dist.barrier()
        function()
        torch.cuda.synchronize()
    cuda_samples = []
    wall_samples = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
        wall_begin = time.perf_counter_ns()
        begin.record()
        function()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_begin) / 1.0e6
        cuda_elapsed = torch.tensor(begin.elapsed_time(end),
                                    dtype=torch.float64, device="cuda")
        wall_elapsed = torch.tensor(wall_ms, dtype=torch.float64,
                                    device="cuda")
        dist.all_reduce(cuda_elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(wall_elapsed, op=dist.ReduceOp.MAX)
        cuda_samples.append(cuda_elapsed.item())
        wall_samples.append(wall_elapsed.item())
    return {"cuda_event": stats(cuda_samples), "host_wall": stats(wall_samples)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--save-output-dir")
    parser.add_argument("--trace-dir")
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    if args.experts % world:
        raise ValueError("experts must be divisible by EP world")
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )
    config = TransformerConfig(
        num_layers=1,
        hidden_size=args.hidden,
        num_attention_heads=8,
        ffn_hidden_size=args.intermediate,
        moe_ffn_hidden_size=args.intermediate,
        num_moe_experts=args.experts,
        moe_router_topk=args.top_k,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        moe_router_pre_softmax=args.top_k == 1,
        moe_permute_fusion=False,
        deterministic_mode=True,
    )
    local_experts = args.experts // world
    dispatcher = MoEAlltoAllTokenDispatcher(
        local_experts,
        list(range(rank * local_experts, (rank + 1) * local_experts)),
        config,
        get_default_pg_collection(),
    )

    # Bit-for-bit the same deterministic BF16 input used by MegaMOE's C++
    # integrated runner: flattened element i is (rank*101 + i%97)/32.
    flat = torch.arange(args.tokens * args.hidden, dtype=torch.int32,
                        device="cuda")
    hidden = ((rank * 101 + torch.remainder(flat, 97)).to(torch.float32) /
              32.0).to(torch.bfloat16).reshape(args.tokens, args.hidden)
    del flat
    token = torch.arange(args.tokens, device="cuda", dtype=torch.long)[:, None]
    slot = torch.arange(args.top_k, device="cuda", dtype=torch.long)[None, :]
    selected = (token * 17 + rank * 31 + slot * 53) % args.experts
    routing = torch.zeros((args.tokens, args.experts), dtype=torch.bool,
                          device="cuda")
    routing.scatter_(1, selected, True)
    probabilities = torch.zeros((args.tokens, args.experts), dtype=torch.float32,
                                device="cuda")
    # Megatron's dispatcher transports probabilities but does not apply them;
    # its expert implementation normally consumes them. For an actual identity
    # expert and byte-identical common path, every selected route has gate 1.
    probabilities.scatter_(1, selected, 1.0)

    def prepare_dispatch():
        return dispatcher.dispatch_preprocess(hidden, routing, probabilities)

    prepared_hidden, prepared_probabilities = prepare_dispatch()
    dispatched_hidden, dispatched_probabilities = dispatcher.token_dispatch(
        prepared_hidden, prepared_probabilities)
    expert_hidden, _, _ = dispatcher.dispatch_postprocess(
        dispatched_hidden, dispatched_probabilities)

    def dispatch_collective():
        return dispatcher.token_dispatch(prepared_hidden, prepared_probabilities)

    def combine_roundtrip():
        local = dispatcher.combine_preprocess(expert_hidden)
        returned = dispatcher.token_combine(local)
        return dispatcher.combine_postprocess(returned)

    def full_roundtrip():
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            hidden, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, _, _ = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        combine_input = dispatcher.combine_preprocess(expert_input)
        returned = dispatcher.token_combine(combine_input)
        return dispatcher.combine_postprocess(returned)

    def common_semantics_roundtrip():
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            hidden, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, _, _ = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        combine_input = dispatcher.combine_preprocess(expert_input)
        returned = dispatcher.token_combine(combine_input)
        # Megatron's native combine accumulates in BF16. This explicit common
        # boundary keeps Megatron's real dispatcher/collectives and replaces
        # only that final local reduction with fixed expert-major FP32 order.
        common = torch.zeros(dispatcher.hidden_shape_before_permute,
                             dtype=torch.float32, device="cuda")
        common.index_add_(
            0, dispatcher.reversed_local_input_permutation_mapping,
            returned.float())
        return common.to(torch.bfloat16).view(dispatcher.hidden_shape)

    measured_runs = []
    for run in range(args.runs):
        measured_runs.append({
            "run": run,
            "routing_metadata_and_local_permute": measure(
                prepare_dispatch, args.warmups, args.iterations),
            "dispatch_alltoall": measure(
                dispatch_collective, args.warmups, args.iterations),
            "combine": measure(
                combine_roundtrip, args.warmups, args.iterations),
            "native_identity_e2e": measure(
                full_roundtrip, args.warmups, args.iterations),
            "common_semantics_identity_e2e": measure(
                common_semantics_roundtrip, args.warmups, args.iterations),
        })

    native_output = full_roundtrip()
    common_output = common_semantics_roundtrip()
    torch.cuda.synchronize()
    expected_fp32 = torch.zeros_like(hidden, dtype=torch.float32)
    for _slot in range(args.top_k):
        expected_fp32.add_(hidden.float())
    expected_output = expected_fp32.to(torch.bfloat16)
    native_absolute = (native_output.float() - expected_output.float()).abs()
    native_relative = (native_absolute /
                       expected_output.float().abs().clamp_min(1.0e-12))
    common_absolute = (common_output.float() - expected_output.float()).abs()
    common_relative = (common_absolute /
                       expected_output.float().abs().clamp_min(1.0e-12))
    output_validation = {
        "native": {
            "byte_identical_to_common_reference": bool(
                torch.equal(native_output, expected_output)),
            "max_abs_error": native_absolute.max().item(),
            "max_relative_error": native_relative.max().item(),
        },
        "common_semantics": {
            "byte_identical_to_common_reference": bool(
                torch.equal(common_output, expected_output)),
            "max_abs_error": common_absolute.max().item(),
            "max_relative_error": common_relative.max().item(),
        },
    }
    if args.save_output_dir:
        os.makedirs(args.save_output_dir, mode=0o700, exist_ok=True)
        torch.save({"native": native_output.cpu(),
                    "common": common_output.cpu()}, os.path.join(
                        args.save_output_dir, f"megatron-rank-{rank}.pt"))

    if args.trace_dir:
        os.makedirs(args.trace_dir, mode=0o700, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU,
                      torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            dist.barrier()
            full_roundtrip()
            torch.cuda.synchronize()
        profile.export_chrome_trace(os.path.join(
            args.trace_dir, f"megatron-rank-{rank}.json"))

    activation_peer_bytes = int(args.tokens * args.top_k * args.hidden * 2 *
                                (world - 1) / world)
    probability_peer_bytes = int(args.tokens * args.top_k * 4 *
                                 (world - 1) / world)
    return_peer_bytes = activation_peer_bytes
    native_wall_means = [
        run["native_identity_e2e"]["host_wall"]["mean_ms"]
        for run in measured_runs]
    common_wall_means = [
        run["common_semantics_identity_e2e"]["host_wall"]["mean_ms"]
        for run in measured_runs]
    output = {
        "schema": 2,
        "implementation": "Megatron-Core MoEAlltoAllTokenDispatcher",
        "megatron_core_version": importlib.metadata.version("megatron-core"),
        "torch_version": torch.__version__,
        "torch_nccl_version": ".".join(
            str(part) for part in torch.cuda.nccl.version()),
        "world": world,
        "shape": {
            "tokens": args.tokens, "hidden": args.hidden,
            "intermediate": args.intermediate, "experts": args.experts,
            "top_k": args.top_k, "warmups": args.warmups,
            "iterations": args.iterations, "runs": args.runs,
        },
        "dtype": "BF16 input/output; FP32 routing probabilities",
        "expert_ownership": "contiguous rank-major",
        "expert_body": "identity",
        "combine_reference": (
            "unit FP32 gate; fixed expert-ID FP32 sum; final BF16 cast"),
        "payload_bytes_per_rank": {
            "dispatch_activation_peer": activation_peer_bytes,
            "dispatch_probability_peer": probability_peer_bytes,
            "combine_activation_peer": return_peer_bytes,
            "total_peer": (activation_peer_bytes + probability_peer_bytes +
                           return_peer_bytes),
        },
        "output_validation": output_validation,
        "runs": measured_runs,
        "native_identity_e2e_host_wall_cross_run": {
            "mean_ms": statistics.fmean(native_wall_means),
            "variance_ms2": statistics.pvariance(native_wall_means),
            "per_run_mean_ms": native_wall_means,
        },
        "common_identity_e2e_host_wall_cross_run": {
            "mean_ms": statistics.fmean(common_wall_means),
            "variance_ms2": statistics.pvariance(common_wall_means),
            "per_run_mean_ms": common_wall_means,
        },
    }
    if rank == 0:
        print("MEGATRON_BASELINE_JSON " + json.dumps(output, sort_keys=True),
              flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
