#!/usr/bin/env python3
"""Megatron EP Backward baseline with only routed X and FC1 A/B retained.

The forward graph is built before each timed region. Megatron's selective
``moe_act`` checkpoint keeps the routed FC1 input and BF16 FC1 output, releases
the gated SwiGLU output consumed by FC2, and recomputes that output in
Backward. The timed region therefore starts with the same X/A/B checkpoint
contract used by WeftMoE and includes both reverse collectives, dX, dGate,
dW1, and dW2.
"""

import argparse
import importlib.metadata
import json
import os
import time

# This helper loads the explicitly requested NCCL runtime before importing
# PyTorch. Keep it ahead of every direct torch/Megatron import in this file.
from megatron_dispatch_baseline import (
    add_dispatcher_arguments,
    create_dispatcher,
    fixed_random_selected_gates,
    initialize_weftmoe_forward_weights,
    loaded_nccl_libraries,
    nccl_runtime_version,
    stats,
    measure,
    validate_dispatcher_arguments,
    validate_dispatcher_runtime,
)

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.transformer_config import TransformerConfig


def deterministic_gradient(shape, rank):
    elements = torch.arange(
        shape.numel(), dtype=torch.int32, device="cuda"
    ).reshape(shape)
    return (
        (torch.remainder(torch.remainder(elements, 113) * 29 + rank * 17, 113) - 56).float()
        / 64.0
    ).to(torch.bfloat16)


def clear_gradients(hidden, selected_gates, parameters):
    hidden.grad = None
    selected_gates.grad = None
    for parameter in parameters:
        parameter.grad = None


def measure_backward(build_graph, output_gradient, clear, warmups, iterations,
                     allocate_tensors=False):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    cuda_samples = []
    wall_samples = []

    for iteration in range(warmups + iterations):
        clear()
        output = build_graph()
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()

        wall_begin = time.perf_counter_ns()
        begin.record()
        gradient = output_gradient.clone() if allocate_tensors else output_gradient
        torch.autograd.backward(output, gradient)
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_begin) / 1.0e6

        if iteration >= warmups:
            cuda_elapsed = torch.tensor(
                begin.elapsed_time(end), dtype=torch.float64, device="cuda"
            )
            wall_elapsed = torch.tensor(
                wall_ms, dtype=torch.float64, device="cuda"
            )
            dist.all_reduce(cuda_elapsed, op=dist.ReduceOp.MAX)
            dist.all_reduce(wall_elapsed, op=dist.ReduceOp.MAX)
            cuda_samples.append(cuda_elapsed.item())
            wall_samples.append(wall_elapsed.item())
        del output
        del gradient

    return {
        "cuda_event": stats(cuda_samples),
        "host_wall": stats(wall_samples),
        "cuda_samples_ms": cuda_samples,
        "wall_samples_ms": wall_samples,
    }


def gradient_snapshot(hidden, selected_gates, named_parameters):
    if hidden.grad is None or selected_gates.grad is None:
        raise RuntimeError("Backward did not produce dX and dGate")
    result = {
        "dx": [hidden.grad.detach().clone()],
        "dgate": [selected_gates.grad.detach().clone()],
        "dw1": [],
        "dw2": [],
    }
    for name, parameter in named_parameters:
        if parameter.grad is None:
            raise RuntimeError(f"Backward did not produce {name} gradient")
        category = "dw1" if "linear_fc1" in name else "dw2"
        result[category].append(parameter.grad.detach().clone())
    return result


def compare_gradients(reference, actual):
    result = {}
    for category in ("dx", "dgate", "dw1", "dw2"):
        expected = reference[category]
        observed = actual[category]
        if len(expected) != len(observed):
            raise RuntimeError(f"{category} tensor-count mismatch")
        exact = True
        maximum = torch.zeros((), dtype=torch.float32, device="cuda")
        for expected_tensor, observed_tensor in zip(expected, observed):
            exact = exact and torch.equal(expected_tensor, observed_tensor)
            difference = (
                expected_tensor.float() - observed_tensor.float()
            ).abs().max()
            maximum = torch.maximum(maximum, difference)
        exact_across_ranks = torch.tensor(
            int(exact), dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(exact_across_ranks, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        result[category] = {
            "all_ranks_byte_identical": bool(exact_across_ranks.item()),
            "max_abs_error": maximum.item(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--validation-repeats", type=int, default=2)
    parser.add_argument("--allocate-tensors", action="store_true",
                        help="include per-call input clone and output/gradient allocations")
    parser.add_argument("--measure-forward", action="store_true",
                        help="also time the training Forward with the same checkpoint contract")
    parser.add_argument("--save-output-dir",
                        help="dump one Forward and its input fixture after timing")
    parser.add_argument(
        "--gate-mode", choices=("fixed-random", "unit"),
        default="fixed-random"
    )
    add_dispatcher_arguments(parser)
    args = parser.parse_args()
    validate_dispatcher_arguments(parser, args)
    if args.validation_repeats not in (0, 2):
        parser.error("--validation-repeats must be 0 or 2")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    validate_dispatcher_runtime(args, world)
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
        moe_token_dispatcher_type=args.dispatcher,
        moe_flex_dispatcher_backend="deepep",
        moe_deepep_num_sms=args.deepep_num_sms,
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
        moe_grouped_gemm=True,
        use_te_activation_func=True,
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    local_experts = args.experts // world
    dispatcher = create_dispatcher(
        args,
        local_experts,
        list(range(rank * local_experts, (rank + 1) * local_experts)),
        config,
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.experts,
        moe_grouped_gemm=True,
        use_te_activation_func=True,
    )
    experts_builder = layer_spec.submodules.mlp.submodules.experts
    experts_module = experts_builder(
        local_experts, config, pg_collection=get_default_pg_collection()
    )
    initialize_weftmoe_forward_weights(
        experts_module,
        rank * local_experts,
        args.hidden,
        args.intermediate,
    )

    flat = torch.arange(
        args.tokens * args.hidden, dtype=torch.int32, device="cuda"
    )
    hidden = (
        (torch.remainder(flat + rank * 13, 127) - 63).float() / 64.0
    ).to(torch.bfloat16).reshape(args.tokens, args.hidden)
    hidden.requires_grad_(True)
    del flat

    token = torch.arange(args.tokens, device="cuda", dtype=torch.long)[:, None]
    slot = torch.arange(args.top_k, device="cuda", dtype=torch.long)[None, :]
    selected = (token * 17 + rank * 31 + slot * 53) % args.experts
    routing = torch.zeros(
        (args.tokens, args.experts), dtype=torch.bool, device="cuda"
    )
    routing.scatter_(1, selected, True)
    selected_gates = (
        torch.ones((args.tokens, args.top_k), dtype=torch.float32)
        if args.gate_mode == "unit"
        else fixed_random_selected_gates(args.tokens, args.top_k, rank)
    ).to(device="cuda").requires_grad_(True)
    output_gradient = deterministic_gradient(hidden.shape, rank)

    named_parameters = list(experts_module.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]

    def build_graph(fresh_inputs=False):
        call_hidden = hidden.detach().clone().requires_grad_(True) if fresh_inputs else hidden
        call_selected = selected.clone() if fresh_inputs else selected
        call_gates = selected_gates.detach().clone().requires_grad_(True) if fresh_inputs else selected_gates
        call_routing = routing
        if fresh_inputs:
            call_routing = torch.zeros_like(routing).scatter_(1, call_selected, True)
        probabilities = torch.zeros(
            (args.tokens, args.experts), dtype=torch.float32, device="cuda"
        ).scatter(1, call_selected, call_gates)
        local_hidden, local_probs = dispatcher.dispatch_preprocess(
            call_hidden, call_routing, probabilities
        )
        remote_hidden, remote_probs = dispatcher.token_dispatch(
            local_hidden, local_probs
        )
        expert_input, local_counts, remote_gate = (
            dispatcher.dispatch_postprocess(remote_hidden, remote_probs)
        )
        expert_output, _ = experts_module(
            expert_input, local_counts, remote_gate
        )
        combine_input = dispatcher.combine_preprocess(expert_output)
        returned = dispatcher.token_combine(combine_input)
        return dispatcher.combine_postprocess(returned)

    def clear():
        clear_gradients(hidden, selected_gates, parameters)

    validation = None
    if args.validation_repeats:
        experts_module.activation_recompute = False
        clear()
        torch.autograd.backward(build_graph(), output_gradient)
        torch.cuda.synchronize()
        standard = gradient_snapshot(hidden, selected_gates, named_parameters)

        experts_module.activation_recompute = True
        clear()
        torch.autograd.backward(build_graph(), output_gradient)
        torch.cuda.synchronize()
        cached = gradient_snapshot(hidden, selected_gates, named_parameters)
        accuracy = compare_gradients(standard, cached)
        del standard

        clear()
        torch.autograd.backward(build_graph(), output_gradient)
        torch.cuda.synchronize()
        repeated = gradient_snapshot(hidden, selected_gates, named_parameters)
        stability = compare_gradients(cached, repeated)
        del cached, repeated
        torch.cuda.empty_cache()
        validation = {
            "against_non_recomputed_megatron": accuracy,
            "cached_x_ab_repeat_stability": stability,
        }

    experts_module.activation_recompute = True
    measured_runs = []
    for run in range(args.runs):
        measured_runs.append({
            "run": run,
            "backward_only": measure_backward(
                build_graph,
                output_gradient,
                clear,
                args.warmups,
                args.iterations,
                args.allocate_tensors,
            ),
        })
        if args.measure_forward:
            clear()
            measured_runs[-1]["forward_e2e"] = measure(
                lambda: build_graph(args.allocate_tensors), args.warmups, args.iterations)

    implementation = (
        "Megatron-Core MoEFlexTokenDispatcher + DeepEP + TEGroupedMLP"
        if args.dispatcher == "flex"
        else "Megatron-Core MoEAlltoAllTokenDispatcher + TEGroupedMLP"
    )
    if args.save_output_dir:
        clear()
        saved_output = build_graph(args.allocate_tensors)
        torch.cuda.synchronize()
        os.makedirs(args.save_output_dir, mode=0o700, exist_ok=True)
        torch.save({"native_forward": saved_output.detach().cpu(),
                    "input": hidden.detach().cpu(),
                    "selected": selected.detach().cpu(),
                    "gates": selected_gates.detach().cpu()},
                   os.path.join(args.save_output_dir, f"megatron-rank-{rank}.pt"))
        del saved_output
    output = {
        "schema": 1,
        "implementation": implementation,
        "token_dispatcher": args.dispatcher,
        "flex_backend": "deepep" if args.dispatcher == "flex" else None,
        "deepep_version": (
            importlib.metadata.version("deep-ep")
            if args.dispatcher == "flex" else None
        ),
        "deepep_num_sms": (
            args.deepep_num_sms if args.dispatcher == "flex" else None
        ),
        "communication_backend": (
            "DeepEP NVLink high-throughput"
            + (" + NVSHMEM/IBGDA internode"
               if world > int(os.environ.get("LOCAL_WORLD_SIZE", world))
               else "")
            if args.dispatcher == "flex"
            else "Megatron permute + NCCL all-to-all"
        ),
        "megatron_core_version": importlib.metadata.version("megatron-core"),
        "transformer_engine_version": importlib.metadata.version(
            "transformer-engine"
        ),
        "torch_version": torch.__version__,
        "torch_nccl_version": ".".join(
            str(part) for part in torch.cuda.nccl.version()
        ),
        "nccl_runtime_version": nccl_runtime_version(),
        "loaded_nccl_libraries": loaded_nccl_libraries(),
        "world": world,
        "shape": {
            "tokens": args.tokens,
            "hidden": args.hidden,
            "intermediate": args.intermediate,
            "experts": args.experts,
            "top_k": args.top_k,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "runs": args.runs,
        },
        "dtype": "BF16 activation/weights/grads; FP32 gate",
        "checkpoint_contract": {
            "retained": ["canonical routed X", "BF16 FC1 A/B"],
            "released": ["gated BF16 SwiGLU output U"],
            "recomputed_in_backward": ["SwiGLU", "gate multiply", "U"],
            "megatron_config": {
                "recompute_granularity": "selective",
                "recompute_modules": ["moe_act"],
            },
        },
        "measurement_scope": (
            "backward-only from cached X/A/B: dY dispatch, dAB, dX, "
            "dGate, dW1, dW2, and dX combine"
        ),
        "forward_setup_in_timed_region": False,
        "allocate_tensors": args.allocate_tensors,
        "synchronize_ranks": True,
        "input_initialization": "timed device-to-device clone" if args.allocate_tensors else "reused",
        "allocator": "PyTorch caching allocator (warmed)",
        "validation": validation,
        "runs": measured_runs,
    }
    if rank == 0:
        print(
            "MEGATRON_BASELINE_JSON " + json.dumps(output, sort_keys=True),
            flush=True,
        )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
