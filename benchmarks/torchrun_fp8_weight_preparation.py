"""Isolated FP8 W/W.T preparation cost; never part of training-cycle timing.

torchrun --standalone --nproc-per-node=2 benchmarks/torchrun_fp8_weight_preparation.py
requires WEFTMOE_TENSOR_EXTENSION pointing at the freshly built adapter.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.models import MODELS
from weftmoe.fp8 import _storage
from weftmoe.runtime import _extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.samples < 1:
        parser.error("warmups must be nonnegative and samples positive")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    torch.manual_seed(20261029 + rank)
    extension = _extension()
    records = []
    for profile, model in MODELS.items():
        assert model.experts % world == 0
        tensors = []
        for m, k in ((2 * model.intermediate, model.hidden), (model.hidden, model.intermediate)):
            shape = (model.experts // world, m, k)
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            tensors.append((x, _storage(shape, 128, x.device),
                            _storage((shape[0], k, m), 128, x.device)))

        def launch(fused):
            for x, q, qt in tensors:
                if fused:
                    extension.fp8_quantize_weight(x, q.data, q.scales, qt.data, qt.scales,
                                                  stream.cuda_stream)
                else:
                    extension.fp8_quantize(x, q.data, q.scales, 128, False, stream.cuda_stream)
                    extension.fp8_transpose_weight(q.data, q.scales, qt.data, qt.scales,
                                                   x.shape[-1], stream.cuda_stream)

        graphs = {}
        for name in ("separate", "fused"):
            launch(name == "fused")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch(name == "fused")
            graphs[name] = graph
        events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        samples = {name: [] for name in graphs}
        for iteration in range(args.warmups + args.samples):
            names = list(graphs)
            if iteration % 2:
                names.reverse()
            for name in names:
                torch.cuda.synchronize()
                dist.barrier()
                events[0].record(stream)
                graphs[name].replay()
                events[1].record(stream)
                events[1].synchronize()
                if iteration >= args.warmups:
                    samples[name].append(events[0].elapsed_time(events[1]))
        gathered = [None] * world
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            p50 = {name: max(statistics.median(r[name]) for r in gathered) for name in graphs}
            record = dict(profile=profile, shapes=[list(t[0].shape) for t in tensors],
                          max_rank_p50_ms=p50, speedup=p50["separate"] / p50["fused"],
                          rank_samples_ms=gathered)
            records.append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "rank_samples_ms"}), flush=True)
        del tensors, graphs, graph, x
        torch.cuda.empty_cache()
    if rank == 0:
        library = Path(os.environ["WEFTMOE_TENSOR_EXTENSION"]).resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(dict(
            gpu=torch.cuda.get_device_name(), world_size=world, input_dtype="bfloat16",
            output_dtype="float8_e4m3fn", scale_dtype="float32", scale_block=[128, 128],
            method="CUDA graph replay, CUDA events, max rank p50; W1 and W2 serial; "
                   "preallocated outputs; excludes allocation and Python validation. "
                   "This standalone diagnostic is excluded from training benchmark totals.",
            warmups=args.warmups, samples=args.samples,
            torch=torch.__version__, cuda=torch.version.cuda, library=str(library),
            native_sha256=hashlib.sha256((library.parent / "libweftmoe.so").read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), results=records,
        ), indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
