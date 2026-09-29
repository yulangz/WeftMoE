"""Compare the real replicated-TP SGLang MoE paths in interleaved CUDA graphs.

Run with torchrun, the desired TP/EP size, and WEFTMOE_TENSOR_EXTENSION set.
Weights and random routes are shared between backends. The model's outer TP
all-reduce is included, as in Qwen's routed-plus-shared-expert block.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import statistics
from types import SimpleNamespace

import pynvml
import torch
import torch.distributed as dist
from sglang.srt.distributed import (
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

import weftmoe.sglang as adapter


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 16, 32, 48, 64, 96, 128, 256, 512]
    )
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
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
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        chunked_prefill_size=2048,
        max_prefill_tokens=2048,
    )
    set_global_server_args_for_scheduler(server)
    initialize_moe_config(server)
    init_distributed_environment(world, rank, local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world, expert_model_parallel_size=world
    )
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
    for tokens in args.tokens:
        torch.manual_seed(100 + tokens)
        x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16) * 0.03
        logits = torch.randn(tokens, 256, device="cuda")
        scores, ids = logits.topk(8, dim=-1)
        topk = StandardTopKOutput(scores.softmax(-1), ids.int(), logits)
        calls = {"baseline": baseline, "weftmoe": layer.forward}
        graphs, outputs = {}, {}
        for name, call in calls.items():
            with model_capture_mode(), group.graph_capture() as context:
                for _ in range(3):
                    group.all_reduce(call(x, topk))
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=context.stream):
                    outputs[name] = group.all_reduce(call(x, topk))
            graphs[name] = graph
            graph.replay()
            torch.cuda.synchronize()
        torch.testing.assert_close(
            outputs["weftmoe"], outputs["baseline"], rtol=0.025, atol=2e-5
        )
        samples = {name: [] for name in calls}
        clocks = []
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        for iteration in range(args.samples + 5):
            names = list(calls) if iteration % 2 == 0 else list(reversed(calls))
            for name in names:
                dist.barrier()
                torch.cuda.synchronize()
                begin.record()
                for _ in range(50):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                if iteration >= 5:
                    samples[name].append(begin.elapsed_time(end) * 1000 / 50)
                    clocks.append(
                        pynvml.nvmlDeviceGetClockInfo(card, pynvml.NVML_CLOCK_SM)
                    )
        gathered = [None] * world
        dist.all_gather_object(gathered, samples)
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
    dist.destroy_process_group()
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
