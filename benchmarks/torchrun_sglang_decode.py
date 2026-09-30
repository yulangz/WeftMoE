"""Compare SGLang MoE paths in interleaved CUDA graphs.

Run with torchrun, the desired TP/EP size, and WEFTMOE_TENSOR_EXTENSION set.
Weights and random routes are shared between backends. By default, inputs are
replicated and the model's outer TP all-reduce is included, as in Qwen's
routed-plus-shared-expert block. Pass --deepgemm-package to include MegaMoE
with the same fused input staging and layout conversion as WeftMoE. Weight
packing and graph/JIT preparation are excluded for all backends.
With --dp-attention, --tokens is per DP rank: baseline includes SGLang's
all-gather/reduce-scatter, while WeftMoE/MegaMoE return DP-local rows directly.
"""

import argparse
from contextlib import nullcontext
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


def make_mega_forward(deep_gemm, buffer, weights, tokens, rank, world, dp_attention):
    local_tokens = tokens if dp_attention else (tokens + world - 1) // world
    start = 0 if dp_attention else rank * local_tokens
    valid = max(0, min(local_tokens, tokens - start))
    hidden, top_k = buffer.x.shape[1], buffer.topk_idx.shape[1]
    output = buffer.x.new_empty((local_tokens, hidden))
    result = None if dp_attention else buffer.x.new_empty((tokens, hidden))

    def forward(x, topk):
        stage_inputs[(local_tokens,)](
            x, topk.topk_ids, topk.topk_weights,
            buffer.x, buffer.topk_idx, buffer.topk_weights, *x.stride(),
            *topk.topk_ids.stride(), *topk.topk_weights.stride(),
            start, valid, hidden, top_k,
            triton.next_power_of_2(hidden), triton.next_power_of_2(top_k),
        )
        deep_gemm.bf16_mega_moe(
            output, *weights, buffer, activation_clamp=None, fast_math=True
        )
        if dp_attention:
            return output
        scatter_output[(triton.cdiv(tokens * hidden, 4096),)](
            output, result, start, valid, hidden, tokens * hidden, 4096,
        )
        return result

    return forward


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
    mega_buffer = None
    if args.deepgemm_package:
        deep_gemm.set_num_sms(
            torch.cuda.get_device_properties(local_rank).multi_processor_count
        )
        max_local_tokens = (
            max(args.tokens) if args.dp_attention
            else (max(args.tokens) + world - 1) // world
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
        torch.manual_seed(100 + global_tokens)
        x = torch.randn(global_tokens, 2048, device="cuda", dtype=torch.bfloat16) * 0.03
        logits = torch.randn(global_tokens, 256, device="cuda")
        scores, ids = logits.topk(8, dim=-1)
        topk = StandardTopKOutput(scores.softmax(-1), ids.int(), logits)
        baseline_call = baseline
        if args.dp_attention:
            set_dp_buffer_len(global_tokens, tokens, True, [tokens] * world)
            source = slice(rank * tokens, (rank + 1) * tokens)
            x = x[source].clone()
            baseline_call = make_dp_baseline(baseline, topk, x, world)
            topk = StandardTopKOutput(
                topk.topk_weights[source], topk.topk_ids[source], logits[source]
            )
        calls = {"baseline": baseline_call, "weftmoe": layer.forward}
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
        samples = {name: [] for name in calls}
        clocks = []
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for iteration in range(args.samples + 5):
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
            graphs["weftmoe"].replay()
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
    adapter.suspend(SimpleNamespace(graph_runner=None))
    if mega_buffer is not None:
        mega_buffer.destroy()
    dist.destroy_process_group()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
