#!/usr/bin/env python3
"""Megatron-Core EP all-to-all dispatch/combine baseline.

Run with ``torchrun --standalone --nproc-per-node=8``.  The benchmark uses
Megatron-Core's production ``MoEAlltoAllTokenDispatcher`` rather than a raw
NCCL all-to-all stand-in.  The identity expert body intentionally isolates the
routing/transport slice that is comparable to MegaMOE's current synthetic
activation transport; ``intermediate`` is recorded for the future FC baseline.
"""

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
import statistics
import struct
import time

# Load the requested NCCL runtime before importing PyTorch. This makes the
# executable baseline independent of LD_PRELOAD and lets the reported version
# come from the exact handle that established the process runtime.
_NCCL_LIBRARY_REQUEST = os.environ.get("MEGAMOE_NCCL_LIBRARY",
                                       "libnccl.so.2")
_NCCL_RUNTIME = ctypes.CDLL(_NCCL_LIBRARY_REQUEST,
                           mode=ctypes.RTLD_GLOBAL)

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)


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


def measure_backward(setup, backward, warmups, iterations):
    """Time native backward only; graph construction is outside the boundary."""
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(warmups):
        output, gradient = setup()
        torch.cuda.synchronize()
        dist.barrier()
        backward(output, gradient)
        torch.cuda.synchronize()
    cuda_samples = []
    wall_samples = []
    for _ in range(iterations):
        output, gradient = setup()
        torch.cuda.synchronize()
        dist.barrier()
        wall_begin = time.perf_counter_ns()
        begin.record()
        backward(output, gradient)
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


def fixed_random_selected_gates(tokens, top_k, rank,
                                seed=0x4D4547414D4F4547):
    values = []
    for token in range(tokens):
        scores = [1 + ((seed + token * 1315423911 + slot * 2654435761 +
                        rank * 97531) % 65521) for slot in range(top_k)]
        total = sum(scores)
        values.extend(float(score / total) for score in scores)
    return torch.tensor(values, dtype=torch.float32).reshape(tokens, top_k)


def fp32_hash(tensor):
    value = 1469598103934665603
    for gate in tensor.contiguous().view(-1).tolist():
        for byte in struct.pack("<f", gate):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def i32_hash(tensor):
    value = 1469598103934665603
    for item in tensor.contiguous().view(-1).cpu().tolist():
        bits = int(item) & 0xffffffff
        for byte in struct.pack("<I", bits):
            value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def bf16_indexed_hash(tensor):
    bits = tensor.contiguous().view(torch.uint16).view(-1)
    result = 0
    chunk = 1_000_000
    for offset in range(0, bits.numel(), chunk):
        end = min(bits.numel(), offset + chunk)
        indices = torch.arange(offset + 1, end + 1, dtype=torch.int64,
                               device=bits.device)
        subtotal = (bits[offset:end].to(torch.int64) * indices).sum().item()
        result = (result + int(subtotal)) & ((1 << 64) - 1)
    return result


def fixture_mix32(value):
    value = torch.bitwise_and(value ^ torch.bitwise_right_shift(value, 16),
                              0xffffffff)
    value = torch.bitwise_and(value * 0x7feb352d, 0xffffffff)
    value = torch.bitwise_and(value ^ torch.bitwise_right_shift(value, 15),
                              0xffffffff)
    value = torch.bitwise_and(value * 0x846ca68b, 0xffffffff)
    return torch.bitwise_and(value ^ torch.bitwise_right_shift(value, 16),
                             0xffffffff)


WEIGHT_HASH_CHUNK_ELEMENTS = 1 << 22
WEIGHT_HASH_SEED_LOW = 0x243F6A8885A308D3
WEIGHT_HASH_SEED_HIGH = 0x13198A2E03707344
WEIGHT_HASH_MULTIPLIER_LOW = 0x9E3779B185EBCA87
WEIGHT_HASH_MULTIPLIER_HIGH = 0xC2B2AE3D27D4EB4F


def _signed_u64(value):
    value &= (1 << 64) - 1
    return value if value < (1 << 63) else value - (1 << 64)


@torch.no_grad()
def actual_device_bf16_hash(tensors):
    """Hash complete device BF16 storage with MegaMOE's fixed definition."""
    tensors = [tensor.detach().contiguous().view(torch.uint16).view(-1)
               for tensor in tensors]
    elements = sum(tensor.numel() for tensor in tensors)
    low = (WEIGHT_HASH_SEED_LOW ^
           (elements * 0x100000001B3)) & ((1 << 64) - 1)
    high = (WEIGHT_HASH_SEED_HIGH ^
            (elements * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
    logical_offset = 0
    for bits in tensors:
        for offset in range(0, bits.numel(), WEIGHT_HASH_CHUNK_ELEMENTS):
            end = min(bits.numel(), offset + WEIGHT_HASH_CHUNK_ELEMENTS)
            index = torch.arange(logical_offset + offset,
                                 logical_offset + end,
                                 dtype=torch.int64, device=bits.device)
            values = bits[offset:end].to(torch.int64)
            low_key = (values ^ torch.bitwise_and(
                index * 0x9E3779B1, 0xffffffff) ^ 0xA5A5F00D)
            high_key = ((values << 16) ^ torch.bitwise_and(
                index * 0x85EBCA77, 0xffffffff) ^ 0x5A5A0FF1)
            low_term = fixture_mix32(low_key) * _signed_u64(
                WEIGHT_HASH_MULTIPLIER_LOW)
            high_term = fixture_mix32(high_key) * _signed_u64(
                WEIGHT_HASH_MULTIPLIER_HIGH)
            low = (low + int(low_term.sum(dtype=torch.int64).item())) & (
                (1 << 64) - 1)
            high = (high + int(high_term.sum(dtype=torch.int64).item())) & (
                (1 << 64) - 1)
        logical_offset += bits.numel()
    return [low, high]


def forward_fixture_values(global_expert, tensor_kind, row, column):
    key = fixture_mix32(torch.full_like(
        row, global_expert ^ 0x9e3779b9, dtype=torch.int64))
    key = key ^ fixture_mix32(torch.full_like(
        row, tensor_kind ^ 0x85ebca6b, dtype=torch.int64))
    key = key ^ fixture_mix32(torch.bitwise_xor(row, 0xc2b2ae35))
    key = key ^ fixture_mix32(torch.bitwise_xor(column, 0x27d4eb2f))
    signed = (torch.bitwise_and(fixture_mix32(key), 1).to(torch.int32) * 2 - 1)
    return (signed.float() / 256.0).to(torch.bfloat16)


def forward_weight_sample_hash(expert_begin, expert_count, hidden,
                               intermediate):
    value = 1469598103934665603
    for expert in range(expert_begin, expert_begin + expert_count):
        coordinates_by_tensor = (
            (1, (((sample * 7919 + 5) % (2 * intermediate),
                  (sample * 104729 + 17) % hidden)
                 for sample in range(32))),
            (2, (((sample * 65537 + 11) % hidden,
                  (sample * 8191 + 3) % intermediate)
                 for sample in range(32))),
        )
        for tensor_kind, coordinates in coordinates_by_tensor:
            for row, column in coordinates:
                bits = int(forward_fixture_values(
                    expert, tensor_kind, torch.tensor([row], dtype=torch.int64),
                    torch.tensor([column], dtype=torch.int64)).view(
                        torch.uint16).item())
                for byte in struct.pack("<H", bits):
                    value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


@torch.no_grad()
def initialize_megamoe_forward_weights(module, global_expert_begin,
                                        hidden, intermediate):
    """Load the exact BF16 fixture used by MegaMOE's device initializer."""
    chunk = 4 * 1024 * 1024
    actual_parameters = {"w1": [], "w2": []}
    for name, parameter in module.named_parameters():
        if "bias" in name:
            parameter.zero_()
            continue
        suffix = name.rsplit("weight", 1)[-1]
        local_expert = int(suffix)
        global_expert = global_expert_begin + local_expert
        is_fc1 = "linear_fc1" in name
        if is_fc1:
            expected = 2 * intermediate * hidden
            tensor_kind = 1
            row_width = hidden
            tensor_name = "w1"
        elif "linear_fc2" in name:
            expected = hidden * intermediate
            tensor_kind = 2
            row_width = intermediate
            tensor_name = "w2"
        else:
            raise RuntimeError(f"unexpected expert parameter: {name}")
        if parameter.numel() != expected:
            raise RuntimeError(
                f"parameter {name} shape mismatch: expected {expected}, "
                f"actual {parameter.numel()}")
        flat = parameter.view(-1)
        for offset in range(0, expected, chunk):
            end = min(expected, offset + chunk)
            index = torch.arange(offset, end,
                                 dtype=torch.int64, device=parameter.device)
            flat[offset:end].copy_(forward_fixture_values(
                global_expert, tensor_kind,
                torch.div(index, row_width, rounding_mode="floor"),
                torch.remainder(index, row_width)))
        actual_parameters[tensor_name].append((local_expert, parameter))
    for tensor_name in actual_parameters:
        actual_parameters[tensor_name].sort(key=lambda item: item[0])
    return actual_parameters


def loaded_nccl_libraries():
    paths = set()
    with open("/proc/self/maps", "r", encoding="utf-8") as mappings:
        for line in mappings:
            path = line.rsplit(" ", 1)[-1].strip()
            if "libnccl.so" in path:
                paths.add(path)
    return sorted(paths)


def nccl_runtime_version():
    version = ctypes.c_int()
    result = _NCCL_RUNTIME.ncclGetVersion(ctypes.byref(version))
    if result != 0:
        raise RuntimeError(f"ncclGetVersion failed: {result}")
    value = version.value
    return f"{value // 10000}.{(value % 10000) // 100}.{value % 100}"


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
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    parser.add_argument("--expert-body", choices=("identity", "te-grouped-mlp"),
                        default="identity")
    parser.add_argument("--benchmark-backward", action="store_true")
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
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_grouped_gemm=args.expert_body == "te-grouped-mlp",
    )
    local_experts = args.experts // world
    dispatcher = MoEAlltoAllTokenDispatcher(
        local_experts,
        list(range(rank * local_experts, (rank + 1) * local_experts)),
        config,
        get_default_pg_collection(),
    )
    experts_module = None
    actual_weight_parameters = None
    if args.expert_body == "te-grouped-mlp":
        layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.experts, moe_grouped_gemm=True,
            use_te_activation_func=True)
        experts_builder = layer_spec.submodules.mlp.submodules.experts
        experts_module = experts_builder(
            local_experts, config, pg_collection=get_default_pg_collection())
        actual_weight_parameters = initialize_megamoe_forward_weights(
            experts_module, rank * local_experts, args.hidden,
            args.intermediate)

    # Shared MegaMOE/Megatron BF16 fixture v3. Rank and flat index enter the
    # same expression on both implementations before the BF16 cast.
    flat = torch.arange(args.tokens * args.hidden, dtype=torch.int32,
                        device="cuda")
    hidden = ((torch.remainder(flat + rank * 13, 127) - 63).to(torch.float32) /
              64.0).to(torch.bfloat16).reshape(args.tokens, args.hidden)
    del flat
    activation_hash = bf16_indexed_hash(hidden)
    token = torch.arange(args.tokens, device="cuda", dtype=torch.long)[:, None]
    slot = torch.arange(args.top_k, device="cuda", dtype=torch.long)[None, :]
    selected = (token * 17 + rank * 31 + slot * 53) % args.experts
    routing = torch.zeros((args.tokens, args.experts), dtype=torch.bool,
                          device="cuda")
    routing.scatter_(1, selected, True)
    probabilities = torch.zeros((args.tokens, args.experts), dtype=torch.float32,
                                device="cuda")
    selected_gates_cpu = (torch.ones((args.tokens, args.top_k),
                                     dtype=torch.float32)
                          if args.gate_mode == "unit" else
                          fixed_random_selected_gates(
                              args.tokens, args.top_k, rank))
    gate_hash = fp32_hash(selected_gates_cpu)
    route_hash = i32_hash(selected.to(torch.int32))
    weight_hash = forward_weight_sample_hash(
        rank * local_experts, local_experts, args.hidden, args.intermediate)
    actual_w1_hash = None
    actual_w2_hash = None
    if actual_weight_parameters is not None:
        w1_parameters = [item[1]
                         for item in actual_weight_parameters["w1"]]
        w2_parameters = [item[1]
                         for item in actual_weight_parameters["w2"]]
        actual_w1_hash = actual_device_bf16_hash(w1_parameters)
        actual_w2_hash = actual_device_bf16_hash(w2_parameters)
        sampled_indices = (0, len(w1_parameters) // 2,
                           len(w1_parameters) - 1)
        sampled_w1 = [actual_device_bf16_hash([w1_parameters[index]])
                      for index in sampled_indices]
        sampled_w2 = [actual_device_bf16_hash([w2_parameters[index]])
                      for index in sampled_indices]
        if len({tuple(value) for value in sampled_w1}) != len(sampled_w1):
            raise RuntimeError("actual W1 device storage is not expert-distinct")
        if len({tuple(value) for value in sampled_w2}) != len(sampled_w2):
            raise RuntimeError("actual W2 device storage is not expert-distinct")
        first_w2 = w2_parameters[0].view(-1)
        sampled_rows = (0, 1, args.hidden // 2, args.hidden - 1)
        row_hashes = [actual_device_bf16_hash([
            first_w2[row * args.intermediate:(row + 1) * args.intermediate]])
            for row in sampled_rows]
        if len({tuple(value) for value in row_hashes}) != len(row_hashes):
            raise RuntimeError("actual W2 output rows are not distinct")
    activation_hashes = [None] * world
    route_hashes = [None] * world
    gate_hashes = [None] * world
    weight_hashes = [None] * world
    actual_w1_hashes = [None] * world
    actual_w2_hashes = [None] * world
    dist.all_gather_object(activation_hashes, activation_hash)
    dist.all_gather_object(route_hashes, route_hash)
    dist.all_gather_object(gate_hashes, gate_hash)
    dist.all_gather_object(weight_hashes, weight_hash)
    dist.all_gather_object(actual_w1_hashes, actual_w1_hash)
    dist.all_gather_object(actual_w2_hashes, actual_w2_hash)
    selected_gates = selected_gates_cpu.to(device="cuda")
    probabilities.scatter_(1, selected, selected_gates)

    def prepare_dispatch():
        return dispatcher.dispatch_preprocess(hidden, routing, probabilities)

    prepared_hidden, prepared_probabilities = prepare_dispatch()
    dispatched_hidden, dispatched_probabilities = dispatcher.token_dispatch(
        prepared_hidden, prepared_probabilities)
    expert_hidden, tokens_per_expert, expert_probabilities = dispatcher.dispatch_postprocess(
        dispatched_hidden, dispatched_probabilities)

    def dispatch_collective():
        return dispatcher.token_dispatch(prepared_hidden, prepared_probabilities)

    def combine_roundtrip():
        weighted = (expert_hidden.float() *
                    expert_probabilities.float().unsqueeze(-1)).to(
                        torch.bfloat16)
        local = dispatcher.combine_preprocess(weighted)
        returned = dispatcher.token_combine(local)
        return dispatcher.combine_postprocess(returned)

    def full_roundtrip():
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            hidden, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, _, remote_gate = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        weighted = (expert_input.float() * remote_gate.float().unsqueeze(-1)).to(
            torch.bfloat16)
        combine_input = dispatcher.combine_preprocess(weighted)
        returned = dispatcher.token_combine(combine_input)
        return dispatcher.combine_postprocess(returned)

    def full_forward_roundtrip():
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            hidden, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, local_counts, remote_gate = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        expert_output, _ = experts_module(
            expert_input, local_counts, remote_gate)
        combine_input = dispatcher.combine_preprocess(expert_output)
        returned = dispatcher.token_combine(combine_input)
        return dispatcher.combine_postprocess(returned)

    def common_forward_roundtrip():
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            hidden, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, local_counts, _ = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        # Disable Megatron's expert-side gate-before for this reference-oriented
        # path. The returned per-route BF16 expert output is reduced at source
        # with the same fixed expert-ID FP32 gate-after semantics as MegaMOE.
        unit_remote_gate = torch.ones(
            expert_input.shape[0], dtype=torch.float32, device="cuda")
        expert_output, _ = experts_module(
            expert_input, local_counts, unit_remote_gate)
        combine_input = dispatcher.combine_preprocess(expert_output)
        returned = dispatcher.token_combine(combine_input)
        route_count = args.tokens * args.top_k
        flat_expert = selected.reshape(-1)
        route_index = torch.arange(route_count, device="cuda",
                                   dtype=torch.long)
        expert_order = torch.argsort(flat_expert * route_count + route_index)
        route_output = torch.empty_like(returned)
        route_output[expert_order] = returned
        slot_order = torch.argsort(selected, dim=1, stable=True)
        token_base = (torch.arange(args.tokens, device="cuda",
                                   dtype=torch.long) * args.top_k)
        common = torch.zeros_like(hidden, dtype=torch.float32)
        flat_gate = selected_gates.reshape(-1)
        for position in range(args.top_k):
            routes = token_base + slot_order[:, position]
            common = common + (route_output[routes].float() *
                               flat_gate[routes].unsqueeze(-1))
        return common.to(torch.bfloat16)

    gradient_flat = torch.arange(args.tokens * args.hidden,
                                 dtype=torch.int32, device="cuda")
    output_gradient = ((torch.remainder(gradient_flat * 7 + rank * 19, 113) -
                        56).to(torch.float32) / 128.0).to(
                            torch.bfloat16).reshape(args.tokens, args.hidden)
    del gradient_flat
    output_gradient_hash = bf16_indexed_hash(output_gradient)
    output_gradient_hashes = [None] * world
    dist.all_gather_object(output_gradient_hashes, output_gradient_hash)

    def native_training_graph():
        if experts_module is None:
            raise RuntimeError("Backward baseline requires te-grouped-mlp")
        experts_module.zero_grad(set_to_none=False)
        input_with_grad = hidden.detach().requires_grad_(True)
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            input_with_grad, routing, probabilities)
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs)
        expert_input, local_counts, remote_gate = dispatcher.dispatch_postprocess(
            remote_hidden, remote_probs)
        expert_output, _ = experts_module(
            expert_input, local_counts, remote_gate)
        combine_input = dispatcher.combine_preprocess(expert_output)
        returned = dispatcher.token_combine(combine_input)
        output = dispatcher.combine_postprocess(returned)
        return output, output_gradient

    def native_backward(output, gradient):
        output.backward(gradient)

    def native_training_step():
        output, gradient = native_training_graph()
        native_backward(output, gradient)
        return output

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
        route_count = args.tokens * args.top_k
        flat_expert = selected.reshape(-1)
        route_index = torch.arange(route_count, device="cuda",
                                   dtype=torch.long)
        expert_order = torch.argsort(flat_expert * route_count + route_index)
        route_output = torch.empty_like(returned)
        route_output[expert_order] = returned
        slot_order = torch.argsort(selected, dim=1, stable=True)
        token_base = (torch.arange(args.tokens, device="cuda",
                                   dtype=torch.long) * args.top_k)
        common = torch.zeros_like(hidden, dtype=torch.float32)
        flat_gate = selected_gates.reshape(-1)
        for position in range(args.top_k):
            routes = token_base + slot_order[:, position]
            product = (route_output[routes].float() *
                       flat_gate[routes].unsqueeze(-1))
            common = common + product
        return common.to(torch.bfloat16)

    measured_runs = []
    for run in range(args.runs):
        run_result = {
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
        }
        if experts_module is not None:
            run_result["native_forward_e2e"] = measure(
                full_forward_roundtrip, args.warmups, args.iterations)
            run_result["common_forward_e2e"] = measure(
                common_forward_roundtrip, args.warmups, args.iterations)
            if args.benchmark_backward:
                run_result["native_backward_only"] = measure_backward(
                    native_training_graph, native_backward,
                    args.warmups, args.iterations)
                run_result["native_forward_backward_e2e"] = measure(
                    native_training_step, args.warmups, args.iterations)
        measured_runs.append(run_result)

    native_output = full_roundtrip()
    common_output = common_semantics_roundtrip()
    native_forward_output = (full_forward_roundtrip()
                             if experts_module is not None else None)
    common_forward_output = (common_forward_roundtrip()
                             if experts_module is not None else None)
    torch.cuda.synchronize()
    slot_order = torch.argsort(selected, dim=1, stable=True)
    expected_fp32 = torch.zeros_like(hidden, dtype=torch.float32)
    for position in range(args.top_k):
        gates = torch.gather(selected_gates, 1,
                             slot_order[:, position:position + 1])
        product = gates * hidden.float()
        expected_fp32 = expected_fp32 + product
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
                    "common": common_output.cpu(),
                    "native_forward": (native_forward_output.cpu()
                                       if native_forward_output is not None
                                       else None),
                    "common_forward": (common_forward_output.cpu()
                                       if common_forward_output is not None
                                       else None)}, os.path.join(
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
    forward_wall_means = ([
        run["native_forward_e2e"]["host_wall"]["mean_ms"]
        for run in measured_runs] if experts_module is not None else [])
    weight_gradient_dtypes = []
    if actual_weight_parameters is not None:
        weight_gradient_dtypes = sorted({
            str(parameter.grad.dtype)
            for kind in ("w1", "w2")
            for _, parameter in actual_weight_parameters[kind]
            if parameter.grad is not None
        })
    output = {
        "schema": 2,
        "implementation": "Megatron-Core MoEAlltoAllTokenDispatcher",
        "megatron_core_version": importlib.metadata.version("megatron-core"),
        "torch_version": torch.__version__,
        "torch_nccl_version": ".".join(
            str(part) for part in torch.cuda.nccl.version()),
        "nccl_runtime_version": nccl_runtime_version(),
        "explicit_nccl_library_request": _NCCL_LIBRARY_REQUEST,
        "loaded_nccl_libraries": loaded_nccl_libraries(),
        "world": world,
        "shape": {
            "tokens": args.tokens, "hidden": args.hidden,
            "intermediate": args.intermediate, "experts": args.experts,
            "top_k": args.top_k, "warmups": args.warmups,
            "iterations": args.iterations, "runs": args.runs,
        },
        "dtype": "BF16 input/output; FP32 routing probabilities",
        "expert_ownership": "contiguous rank-major",
        "expert_body": args.expert_body,
        "gemm_backend": ("TransformerEngine TEGroupedMLP"
                         if experts_module is not None else "none"),
        "weight_fixture": ("MegaMOE four-coordinate PRNG BF16 v3"
                           if experts_module is not None else "none"),
        "weight_generator_version": 3 if experts_module is not None else None,
        "activation_fixture": "MegaMOE rank-flat BF16 v2",
        "per_rank_activation_hash": activation_hashes,
        "per_rank_output_gradient_hash": output_gradient_hashes,
        "per_rank_route_hash": route_hashes,
        "gate_mode": args.gate_mode,
        "gate_seed": "0x4d4547414d4f4547",
        "per_rank_gate_hash": gate_hashes,
        "per_rank_weight_sample_hash": weight_hashes,
        "per_rank_actual_w1_device_hash": actual_w1_hashes,
        "per_rank_actual_w2_device_hash": actual_w2_hashes,
        "weight_gradient_dtypes": weight_gradient_dtypes,
        "combine_reference": (
            "source-side per-route FP32 gate-after; fixed expert-ID FP32 "
            "sum; final BF16 cast"),
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
    if forward_wall_means:
        output["native_forward_e2e_host_wall_cross_run"] = {
            "mean_ms": statistics.fmean(forward_wall_means),
            "variance_ms2": statistics.pvariance(forward_wall_means),
            "per_run_mean_ms": forward_wall_means,
        }
        output["common_forward_e2e_host_wall_cross_run"] = {
            "mean_ms": statistics.fmean([
                run["common_forward_e2e"]["host_wall"]["mean_ms"]
                for run in measured_runs]),
            "variance_ms2": statistics.pvariance([
                run["common_forward_e2e"]["host_wall"]["mean_ms"]
                for run in measured_runs]),
            "per_run_mean_ms": [
                run["common_forward_e2e"]["host_wall"]["mean_ms"]
                for run in measured_runs],
        }
        if args.benchmark_backward:
            backward_means = [
                run["native_backward_only"]["host_wall"]["mean_ms"]
                for run in measured_runs]
            training_means = [
                run["native_forward_backward_e2e"]["host_wall"]["mean_ms"]
                for run in measured_runs]
            output["native_backward_host_wall_cross_run"] = {
                "mean_ms": statistics.fmean(backward_means),
                "variance_ms2": statistics.pvariance(backward_means),
                "per_run_mean_ms": backward_means,
            }
            output["native_forward_backward_host_wall_cross_run"] = {
                "mean_ms": statistics.fmean(training_means),
                "variance_ms2": statistics.pvariance(training_means),
                "per_run_mean_ms": training_means,
            }
    if rank == 0:
        print("MEGATRON_BASELINE_JSON " + json.dumps(output, sort_keys=True),
              flush=True)
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
