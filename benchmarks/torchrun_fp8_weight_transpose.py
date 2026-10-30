"""Time the production FP8 weight transpose, excluding allocation/quantization.

Run with torchrun and WEFTMOE_TENSOR_EXTENSION pointing at a built adapter.
Each rank owns experts/world_size matrices. No communication is timed.
"""
from __future__ import annotations

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
from weftmoe.runtime import _extension


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.warmups < 0:
        parser.error("samples must be positive and warmups nonnegative")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(20261029 + rank)
    extension = _extension()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    results = []

    for profile, model in MODELS.items():
        assert model.experts % world == 0
        experts = model.experts // world
        tensors = []
        for rows, columns in ((2 * model.intermediate, model.hidden),
                              (model.hidden, model.intermediate)):
            # Finite, varying FP8 bit patterns; scales transpose without arithmetic.
            q = torch.randint(0, 127, (experts, rows, columns),
                              dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
            s = torch.rand((experts, rows // 128, columns // 128), device="cuda")
            qt = torch.empty((experts, columns, rows), dtype=q.dtype, device="cuda")
            st = torch.empty((experts, columns // 128, rows // 128), device="cuda")
            tensors.append((q, s, qt, st))

        def launch(indices):
            for index in indices:
                q, s, qt, st = tensors[index]
                extension.fp8_transpose_weight(q, s, qt, st, q.shape[-1], stream.cuda_stream)

        launch((0, 1))
        for q, s, qt, st in tensors:
            assert torch.equal(qt.view(torch.uint8), q.view(torch.uint8).transpose(-1, -2))
            assert torch.equal(st, s.transpose(-1, -2))
        shapes = [list(t[0].shape) for t in tensors]
        input_bytes = [q.numel() + s.numel() * s.element_size() for q, s, _, _ in tensors]
        graphs = {}
        groups = {"w1": (0,), "w2": (1,), "both": (0, 1)}
        for name, indices in groups.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                launch(indices)
            graphs[name] = graph
        for mode in ("eager", "graph"):
            samples = {name: [] for name in groups}
            events = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            for iteration in range(args.warmups + args.samples):
                # Rotate the ordering to avoid always timing one operation first.
                names = list(groups)
                shift = iteration % len(names)
                for name in names[shift:] + names[:shift]:
                    torch.cuda.synchronize()
                    dist.barrier()  # Outside the measured CUDA interval.
                    events[0].record(stream)
                    if mode == "eager":
                        launch(groups[name])
                    else:
                        graphs[name].replay()
                    events[1].record(stream)
                    events[1].synchronize()
                    if iteration >= args.warmups:
                        samples[name].append(events[0].elapsed_time(events[1]))
            gathered = [None] * world
            dist.all_gather_object(gathered, samples)
            if rank == 0:
                summary = {}
                for name in groups:
                    rank_p50 = [statistics.median(item[name]) for item in gathered]
                    p50 = max(rank_p50)
                    byte_count = sum(input_bytes[index] for index in groups[name])
                    summary[name] = {
                        "max_rank_p50_ms": p50, "rank_p50_ms": rank_p50,
                        "read_write_GB_per_s": 2 * byte_count / (p50 * 1e6),
                    }
                record = {"profile": profile, "mode": mode, "shapes": shapes,
                          "input_bytes": input_bytes, "summary": summary,
                          "rank_samples_ms": gathered, "bit_exact": True}
                results.append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "rank_samples_ms"}), flush=True)
        del graphs, graph, tensors, q, s, qt, st
        torch.cuda.empty_cache()

    if rank == 0:
        library = Path(os.environ["WEFTMOE_TENSOR_EXTENSION"]).resolve()
        metadata = {
            "gpu": torch.cuda.get_device_name(), "world_size": world,
            "dtype": "float8_e4m3fn with FP32 scales per 128x128 block",
            "warmups": args.warmups, "samples": args.samples,
            "method": "CUDA events; max rank p50; preallocated; scale transpose included; "
                      "allocation, quantization and communication excluded; both means two serial launches. "
                      "Eager includes host dispatch gaps; graph measures captured launches.",
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "library": str(library),
            "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            "native_sha256": hashlib.sha256((library.parent / "libweftmoe.so").read_bytes()).hexdigest(),
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
