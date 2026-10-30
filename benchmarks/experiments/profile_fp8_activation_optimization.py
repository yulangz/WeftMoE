"""Compare native builds on identical variable-token EP workloads using CUPTI.

Example: python benchmarks/experiments/profile_fp8_activation_optimization.py \
    --candidate build/fp8-activation-opt-final-{model} --worlds 2 8
Weight preparation is excluded. CUDA launch correlations attribute GPU events
to phases; asynchronous CPU call durations are not GPU execution times.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.analyze_training_trace import analyze


def category(event):
    name = event["name"]
    for needle, label in (
        ("::local::kernel", "main"),
        ("::backward_stream::kernel", "main"),
        ("prepare_fp8_x_columns", "x_columns"),
        ("prepare_source_x", "source_quantization"),
        ("notify_kernel", "notify"),
        ("::backward_stream::prepare", "prepare"),
    ):
        if needle in name:
            return label
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="build/fp8-dual-weight-{model}")
    parser.add_argument("--models", nargs="+", default=["qwen35b", "qwen122b", "flash"])
    parser.add_argument("--worlds", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 20480])
    parser.add_argument("--variants", nargs="+", choices=["baseline", "candidate", "bf16"],
                        default=["baseline", "candidate", "bf16"])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for world in args.worlds:
        for model in args.models:
            for tokens in args.tokens:
                hashes = None
                for variant in args.variants:
                    directory = args.output / f"ep{world}-{model}-t{tokens}-{variant}"
                    directory.mkdir(parents=True, exist_ok=True)
                    pattern = args.candidate if variant == "candidate" else args.baseline
                    native = (ROOT / pattern.format(model=model)).resolve()
                    extension = next(native.glob("_runtime*.so"))
                    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                           f"--nproc-per-node={world}", "benchmarks/torchrun_training.py",
                           "--backend", "pf", "--model", model, "--tokens", str(tokens),
                           "--warmups", str(args.warmups), "--samples", str(args.samples),
                           "--stream-ordered", "--cuda-trace-dir", str(directory.resolve())]
                    if variant != "bf16":
                        cmd += ["--fp8", "--weight-grad-dtype", "bf16"]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, range(world))),
                               WEFTMOE_TENSOR_EXTENSION=str(extension), NCCL_DEBUG="WARN")
                    with (directory / "run.log").open("w") as log:
                        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=600, check=True)
                    log = (directory / "run.log").read_text()
                    metadata = json.loads(next(line.removeprefix("TRAINING_CYCLE_JSON ")
                                               for line in log.splitlines()
                                               if line.startswith("TRAINING_CYCLE_JSON ")))
                    route_hashes = metadata["workload_sha256_per_rank"]
                    assert hashes is None or hashes == route_hashes
                    hashes = route_hashes
                    ranks = [analyze(directory / f"t{tokens}.rank{rank}.json")
                             for rank in range(world)]
                    compact, summary = [], {}
                    for rank in ranks:
                        samples = {}
                        for phase, rows in rank.items():
                            assert len(rows) == args.samples
                            samples[phase] = []
                            for row in rows:
                                events = {}
                                for event in row["events"]:
                                    label = category(event)
                                    events[label] = events.get(label, 0.0) + event["duration_us"]
                                samples[phase].append(dict(events_us=events,
                                    gpu_span_us=row["gpu_span_us"],
                                    gpu_busy_union_us=row["gpu_busy_union_us"]))
                        compact.append(samples)
                    for phase in ranks[0]:
                        def metric(fn):
                            return statistics.median(max(fn(rank[phase][i]) for rank in compact)
                                                     for i in range(args.samples))
                        summary[phase] = dict(
                            main_us=metric(lambda row: row["events_us"]["main"]),
                            main_plus_x_columns_us=metric(lambda row:
                                row["events_us"]["main"] + row["events_us"].get("x_columns", 0)),
                            x_columns_us=metric(lambda row: row["events_us"].get("x_columns", 0)),
                            gpu_span_us=metric(lambda row: row["gpu_span_us"]),
                            gpu_busy_union_us=metric(lambda row: row["gpu_busy_union_us"]))
                    records.append(dict(world=world, model=model, tokens=tokens, variant=variant,
                        command=cmd, library=str(native / "libweftmoe.so"),
                        native_sha256=hashlib.sha256((native / "libweftmoe.so").read_bytes()).hexdigest(),
                        workload_sha256_per_rank=hashes, summary=summary, rank_samples=compact,
                        wall_slowest_rank_p50_us=metadata["wall_slowest_rank_p50_us"],
                        trace_directory=str(directory.resolve())))
                    (args.output / "results.json").write_text(json.dumps(dict(
                        source_id="weftmoe-fp8-activation-optimization-20261029",
                        gpu="NVIDIA L20X SM90, NVLink", warmups=args.warmups, samples=args.samples,
                        method="CUPTI duration; median of per-sample maximum across ranks. "
                               "FP8 dW is BF16; weights prepared outside timing. "
                               "main_plus_x_columns adds serial kernels per rank/sample before reduction. "
                               "Phase spans may overlap; do not add phase medians.",
                        records=records), indent=2) + "\n")
                    print(directory.name, summary, flush=True)


if __name__ == "__main__":
    main()
