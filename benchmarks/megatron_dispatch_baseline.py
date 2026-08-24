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
    for _ in range(warmups):
        dist.barrier()
        function()
        torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        dist.barrier()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        function()
        end.record()
        end.synchronize()
        elapsed = torch.tensor(begin.elapsed_time(end), dtype=torch.float64,
                               device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        samples.append(elapsed.item())
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
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

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260824 + rank)
    hidden = torch.randn((args.tokens, args.hidden), dtype=torch.bfloat16,
                         device="cuda", generator=generator)
    token = torch.arange(args.tokens, device="cuda", dtype=torch.long)[:, None]
    slot = torch.arange(args.top_k, device="cuda", dtype=torch.long)[None, :]
    selected = (token * 17 + rank * 31 + slot * 53) % args.experts
    routing = torch.zeros((args.tokens, args.experts), dtype=torch.bool,
                          device="cuda")
    routing.scatter_(1, selected, True)
    probabilities = torch.zeros((args.tokens, args.experts), dtype=torch.float32,
                                device="cuda")
    probabilities.scatter_(1, selected, 1.0 / args.top_k)

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

    results = {
        "routing_metadata_and_local_permute": stats(
            measure(prepare_dispatch, args.warmups, args.iterations)),
        "dispatch_alltoall": stats(
            measure(dispatch_collective, args.warmups, args.iterations)),
        "combine": stats(
            measure(combine_roundtrip, args.warmups, args.iterations)),
        "identity_expert_transport_roundtrip": stats(
            measure(full_roundtrip, args.warmups, args.iterations)),
    }
    peer_payload_bytes = (args.tokens * args.top_k * args.hidden * 2 *
                          (world - 1) / world)
    results["dispatch_alltoall"]["effective_peer_GBps"] = (
        peer_payload_bytes / (results["dispatch_alltoall"]["mean_ms"] * 1e6))
    results["combine"]["effective_peer_GBps"] = (
        peer_payload_bytes / (results["combine"]["mean_ms"] * 1e6))
    output = {
        "schema": 1,
        "implementation": "Megatron-Core MoEAlltoAllTokenDispatcher",
        "megatron_core_version": importlib.metadata.version("megatron-core"),
        "torch_version": torch.__version__,
        "torch_nccl_version": ".".join(
            str(part) for part in torch.cuda.nccl.version()),
        "world": world,
        "shape": vars(args),
        "dtype": "BF16 input/output; FP32 routing probabilities",
        "expert_body": "identity (transport-isolation baseline)",
        "estimated_peer_payload_bytes_per_direction_per_rank": int(peer_payload_bytes),
        "results": results,
    }
    if rank == 0:
        print("MEGATRON_BASELINE_JSON " + json.dumps(output, sort_keys=True),
              flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
