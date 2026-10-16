"""Compare SGLang MoE paths in interleaved CUDA graphs.

Run with torchrun, the desired TP/EP size, and WEFTMOE_TENSOR_EXTENSION set.
Weights and random routes are shared between backends. By default, inputs are
replicated and the model's outer TP all-reduce is included, as in Qwen's
routed-plus-shared-expert block. Pass --deepgemm-package to include MegaMoE
with the same fused input staging and layout conversion as WeftMoE. Weight
packing and graph/JIT preparation are excluded for all backends.
With --dp-attention, --tokens is per DP rank: baseline includes SGLang's
all-gather/reduce-scatter, while WeftMoE/MegaMoE return DP-local rows directly.
--weft-backend pf/dt uses the explicitly compiled native profile with the same
fused staging, and rejects fallback by profiling its captured kernel. Native
capacity is the largest requested local bucket. --fresh-routes changes routing
and X between samples outside timing; all backends share each sample's inputs.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import pynvml
import torch
import torch.distributed as dist
import triton
import weftmoe.sglang as adapter
from weftmoe._sglang_kernels import scatter_output, stage_inputs
from weftmoe.runtime import TensorRuntime


def make_staged_forward(storage, kernel, tokens, rank, world, dp_attention):
    """Use the same fused input staging and output layout for native backends."""
    local_tokens = tokens if dp_attention else (tokens + world - 1) // world
    start = 0 if dp_attention else rank * local_tokens
    valid = max(0, min(local_tokens, tokens - start))
    staged_x, staged_ids, staged_weights = storage
    hidden, top_k = staged_x.shape[1], staged_ids.shape[1]
    result = None if dp_attention else staged_x.new_empty((tokens, hidden))

    def forward(x, topk):
        stage_inputs[(local_tokens,)](
            x, topk.topk_ids, topk.topk_weights,
            staged_x, staged_ids, staged_weights, *x.stride(),
            *topk.topk_ids.stride(), *topk.topk_weights.stride(),
            start, valid, hidden, top_k,
            triton.next_power_of_2(hidden), triton.next_power_of_2(top_k),
        )
        output = kernel()
        if dp_attention:
            return output
        scatter_output[(triton.cdiv(tokens * hidden, 4096),)](
            output, result, start, valid, hidden, tokens * hidden, 4096,
        )
        return result

    return forward


def make_mega_forward(deep_gemm, buffer, weights, tokens, rank, world, dp_attention):
    local_tokens = tokens if dp_attention else (tokens + world - 1) // world
    output = buffer.x.new_empty((local_tokens, buffer.x.shape[1]))

    def kernel():
        deep_gemm.bf16_mega_moe(
            output, *weights, buffer, activation_clamp=None, fast_math=True
        )
        return output

    return make_staged_forward(
        (buffer.x, buffer.topk_idx, buffer.topk_weights), kernel,
        tokens, rank, world, dp_attention,
    )


def make_dp_baseline(baseline, global_topk, x, world):
    from sglang.srt.layers.dp_attention import (
        DpPaddingMode,
        dp_gather_partial,
        dp_reduce_scatter_tensor,
    )

    gathered = x.new_empty((x.shape[0] * world, x.shape[1]))
    output = torch.empty_like(x)
    batch = SimpleNamespace(dp_padding_mode=DpPaddingMode.MAX_LEN)

    def forward(local_x, local_topk):
        dp_gather_partial(gathered, local_x, batch)
        # Stock SGLang routes after gathering X. Router/top-k is precomputed
        # outside timing for every backend, so no route all-gather is needed.
        dp_reduce_scatter_tensor(output, baseline(gathered, global_topk))
        return output

    return forward


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 32, 48, 64, 96, 128, 256, 512]
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument(
        "--profile-ncu",
        action="store_true",
        help="Bracket one warmed Weft graph for NCU application replay",
    )
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--weft-backend", choices=("auto", "pf", "dt"), default="auto",
                        help="Use an explicit native profile and verify its captured kernel")
    parser.add_argument("--fresh-routes", action="store_true",
                        help="Change inputs and routes between timed samples, outside timing")
    parser.add_argument("--route-bias-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--deepgemm-package", type=Path)
    parser.add_argument(
        "--dp-attention", action="store_true",
        help="Use DP=EP with distinct tokens per rank; --tokens is per DP rank",
    )
    args = parser.parse_args()
    # SGLang imports deep_gemm itself; select the frozen package first.
    if args.deepgemm_package:
        sys.path.insert(0, str(args.deepgemm_package.resolve()))
        import deep_gemm

    from sglang.srt.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    pynvml.nvmlInit()
    card = pynvml.nvmlDeviceGetHandleByUUID(
        "GPU-" + str(torch.cuda.get_device_properties(local_rank).uuid)
    )
    server = ServerArgs(
        model_path="/storage/openpsi/models/Qwen__Qwen3.5-35B-A3B-Base",
        tp_size=world,
        ep_size=world,
        dp_size=world if args.dp_attention else 1,
        enable_dp_attention=args.dp_attention,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        chunked_prefill_size=2048,
        max_prefill_tokens=2048,
    )
    set_global_server_args_for_scheduler(server)
    initialize_moe_config(server)
    init_distributed_environment(world, rank, local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world, expert_model_parallel_size=world,
        attention_data_parallel_size=world if args.dp_attention else 1,
    )
    if args.dp_attention:
        from sglang.srt.layers.dp_attention import (
            initialize_dp_attention,
            set_dp_buffer_len,
        )

        initialize_dp_attention(
            server, SimpleNamespace(hidden_size=2048, dtype=torch.bfloat16)
        )
        set_dp_buffer_len(0, 0, True)
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

    layer = FusedMoE(
        256, 2048, 512, layer_id=0, top_k=8, params_dtype=torch.bfloat16, inplace=False
    ).cuda()
    torch.manual_seed(37 + rank)
    layer.w13_weight.normal_(0, 0.03)
    layer.w2_weight.normal_(0, 0.03)
    baseline = layer.forward
    adapter.install()
    group = get_tp_group()
    max_local_tokens = (
        max(args.tokens) if args.dp_attention
        else (max(args.tokens) + world - 1) // world
    )
    runtime = None
    weft_name = "weftmoe" if args.weft_backend == "auto" else args.weft_backend
    if args.weft_backend != "auto":
        runtime = TensorRuntime(
            group=group.device_group, experts=256, top_k=8, hidden=2048,
            intermediate=512, max_tokens=max_local_tokens,
            max_rows=max_local_tokens * world * 8 + (256 // world) * 128,
            training=False,
        )
    mega_buffer = None
    if args.deepgemm_package:
        deep_gemm.set_num_sms(
            torch.cuda.get_device_properties(local_rank).multi_processor_count
        )
        mega_buffer = deep_gemm.get_symm_buffer_for_mega_moe(
            group.device_group, 256, max(64, max_local_tokens),
            8, 2048, 512, mma_type="bf16xbf16",
        )
        mega_weights = deep_gemm.transform_weights_for_mega_moe(
            layer.w13_weight, layer.w2_weight
        )
    for tokens in args.tokens:
        global_tokens = tokens * world if args.dp_attention else tokens
        torch.manual_seed(args.seed + global_tokens)
        x = torch.randn(global_tokens, 2048, device="cuda", dtype=torch.bfloat16) * 0.03
        bias = (torch.randn(256, device="cuda") * args.route_bias_std
                if args.route_bias_std else torch.zeros(256, device="cuda"))
        logits = torch.randn(global_tokens, 256, device="cuda") + bias
        scores, ids = logits.topk(8, dim=-1)
        topk = StandardTopKOutput(scores.softmax(-1), ids.int(), logits)
        global_topk = topk
        baseline_call = baseline
        if args.dp_attention:
            set_dp_buffer_len(global_tokens, tokens, True, [tokens] * world)
            source = slice(rank * tokens, (rank + 1) * tokens)
            x = x[source].clone()
            baseline_call = make_dp_baseline(baseline, topk, x, world)
            topk = StandardTopKOutput(
                topk.topk_weights[source], topk.topk_ids[source], logits[source]
            )
        weft_call = layer.forward
        if runtime is not None:
            local_tokens = tokens if args.dp_attention else (tokens + world - 1) // world
            storage = (
                x.new_empty((local_tokens, 2048)),
                torch.empty((local_tokens, 8), dtype=torch.int32, device=x.device),
                torch.empty((local_tokens, 8), dtype=torch.float32, device=x.device),
            )

            def native_kernel():
                return runtime.inference(0, *storage, layer.w13_weight, layer.w2_weight)

            weft_call = make_staged_forward(
                storage, native_kernel, tokens, rank, world, args.dp_attention,
            )
        calls = {"baseline": baseline_call, weft_name: weft_call}
        if mega_buffer is not None:
            calls["megamoe"] = make_mega_forward(
                deep_gemm, mega_buffer, mega_weights, tokens, rank, world,
                args.dp_attention,
            )
        if not args.dp_attention:
            calls = {
                name: (lambda x, topk, call=call: group.all_reduce(call(x, topk)))
                for name, call in calls.items()
            }
        graphs, outputs = {}, {}
        for name, call in calls.items():
            with model_capture_mode(), group.graph_capture() as context:
                for _ in range(3):
                    call(x, topk)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    outputs[name] = call(x, topk)
            graphs[name] = graph
            graph.replay()
            torch.cuda.synchronize()
        max_abs = {}
        for name, output in outputs.items():
            torch.testing.assert_close(
                output, outputs["baseline"], rtol=0.025, atol=2e-5
            )
            max_abs[name] = (
                (output.float() - outputs["baseline"].float()).abs().max().item()
            )
        kernel_names = []
        if runtime is not None:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as probe:
                graphs[weft_name].replay()
                torch.cuda.synchronize()
            kernel_names = sorted({event.name for event in probe.events()
                                   if event.device_type == torch.autograd.DeviceType.CUDA})
            expected = ("weftmoe_pf::local::kernel" if args.weft_backend == "pf"
                        else "weftmoe_dt::kernel")
            assert any(expected in name for name in kernel_names), kernel_names
        samples = {name: [] for name in calls}
        workload = hashlib.sha256()
        clocks = []
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for iteration in range(args.samples + 5):
            if args.fresh_routes:
                # Each rank constructs the same global routing table, while X
                # is DP-local. Captured views retain their original addresses.
                torch.manual_seed(args.seed + global_tokens + 1009 * (iteration + 1))
                logits.copy_(torch.randn_like(logits) + bias)
                scores, ids = logits.topk(8, dim=-1)
                global_topk.topk_ids.copy_(ids)
                global_topk.topk_weights.copy_(scores.softmax(-1))
                torch.manual_seed(args.seed + global_tokens + 9173 * (iteration + 1) + rank)
                x.normal_(0, 0.03)
            workload.update(global_topk.topk_ids.cpu().numpy().tobytes())
            workload.update(global_topk.topk_weights.cpu().numpy().tobytes())
            names = list(calls)
            offset = iteration % len(names)
            names = names[offset:] + names[:offset]
            for name in names:
                dist.barrier()
                torch.cuda.synchronize()
                before = pynvml.nvmlDeviceGetClockInfo(card, pynvml.NVML_CLOCK_SM)
                begin.record()
                for _ in range(50):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                if iteration >= 5:
                    samples[name].append(begin.elapsed_time(end) * 1000 / 50)
                    after = pynvml.nvmlDeviceGetClockInfo(card, pynvml.NVML_CLOCK_SM)
                    clocks.extend((before, after))
            for name, output in outputs.items():
                torch.testing.assert_close(output, outputs["baseline"], rtol=0.025, atol=2e-5)
                max_abs[name] = max(max_abs[name],
                    (output.float() - outputs["baseline"].float()).abs().max().item())
        gathered = [None] * world
        dist.all_gather_object(gathered, samples)
        all_errors = [None] * world
        dist.all_gather_object(all_errors, max_abs)
        all_clocks = [None] * world
        dist.all_gather_object(all_clocks, clocks)
        if rank == 0:
            maximum = {
                name: [max(r[name][i] for r in gathered) for i in range(args.samples)]
                for name in calls
            }
            print(
                "SGLANG_DECODE_COMPARE "
                + json.dumps(
                    {
                        "world": world,
                        "tokens": tokens,
                        "tokens_scope": (
                            "dp_local" if args.dp_attention else "global_replicated"
                        ),
                        "global_tokens": global_tokens,
                        "dp_size": world if args.dp_attention else 1,
                        "baseline_communication": (
                            "all_gather+reduce_scatter" if args.dp_attention
                            else "all_reduce"
                        ),
                        "experts": 256, "top_k": 8,
                        "hidden": 2048, "intermediate": 512,
                        "weft_backend": args.weft_backend,
                        "native_capacity": max_local_tokens if runtime is not None else None,
                        "kernel_names": kernel_names,
                        "fresh_routes": args.fresh_routes,
                        "route_bias_std": args.route_bias_std,
                        "seed": args.seed, "routing_sha256": workload.hexdigest(),
                        "scope": (
                            "CUDA graph; includes staging, dispatch, FC1/FC2, return/reduce; "
                            "excludes router, initialization and weight packing"
                        ),
                        "deepgemm_package": (
                            deep_gemm.__file__ if mega_buffer is not None else None
                        ),
                        "mega_capacity": (
                            mega_buffer.num_max_tokens_per_rank
                            if mega_buffer is not None else None
                        ),
                        "warmup": 5, "measured": args.samples, "replays": 50,
                        "rank_max_abs_vs_baseline": {
                            name: max(row[name] for row in all_errors) for name in calls
                        },
                        "rank_max_p50_us": {
                            name: statistics.median(values)
                            for name, values in maximum.items()
                        },
                        "rank_max_samples_us": maximum,
                        "observed_sm_clocks_mhz": sorted(
                            {c for row in all_clocks for c in row}
                        ),
                    }
                ),
                flush=True,
            )
        if args.profile_ncu:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            graphs[weft_name].replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        if args.trace_dir:
            if rank == 0:
                args.trace_dir.mkdir(parents=True, exist_ok=True)
            for name, graph in graphs.items():
                dist.barrier()
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                )
                with profiler if rank == 0 else nullcontext():
                    for _ in range(5):
                        graph.replay()
                    torch.cuda.synchronize()
                if rank == 0:
                    profiler.export_chrome_trace(
                        str(args.trace_dir / f"{name}_t{tokens}.json")
                    )
                    print(
                        name,
                        tokens,
                        profiler.key_averages().table(
                            sort_by="self_cuda_time_total", row_limit=12
                        ),
                        flush=True,
                    )
        graphs.clear()
        del graph, outputs
    if runtime is not None:
        runtime.close()
    adapter.suspend(SimpleNamespace(graph_runner=None))
    if mega_buffer is not None:
        mega_buffer.destroy()
    dist.destroy_process_group()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
