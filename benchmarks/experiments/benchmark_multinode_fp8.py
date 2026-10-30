"""Paired EP16 BF16/FP8 runs; optional separate CUPTI kernel measurements."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.analyze_training_trace import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--ssh-port", default=2222, type=int)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--native", default="build/fp8-network-production-{model}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["qwen35b", "qwen122b", "flash"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 20480])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--prime-trace", action="store_true",
                        help="Replay the workload once untimed before warmup to warm allocator variants")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for model in args.models:
        workload_hashes = {}
        for dtype in ("bf16", "fp8"):
            native = (ROOT / args.native.format(model=model)).resolve()
            directory = (args.output / f"{model}-{dtype}").resolve()
            command = [sys.executable, "benchmarks/experiments/run_multinode_training.py",
                       "--remote", args.remote, "--ssh-port", str(args.ssh_port),
                       "--master-addr", args.master_addr, "--native", str(native),
                       "--output", str(directory), "--timeout", "360", "--",
                       "--backend", "pf", "--model", model,
                       "--tokens", *map(str, args.tokens), "--stream-ordered",
                       "--warmups", str(args.warmups), "--samples", str(args.samples)]
            if dtype == "fp8":
                command += ["--fp8", "--weight-grad-dtype", "bf16"]
            if args.prime_trace:
                command += ["--prime-trace"]
            if args.trace:
                command += ["--cuda-trace-dir", str(directory)]
            subprocess.run(command, cwd=ROOT, check=True)
            metadata = [json.loads(line.removeprefix("TRAINING_CYCLE_JSON "))
                        for line in (directory / "node0.log").read_text().splitlines()
                        if line.startswith("TRAINING_CYCLE_JSON ")]
            assert len(metadata) == len(args.tokens)
            for record in metadata:
                tokens = record["nominal_tokens_per_rank"]
                hashes = record["workload_sha256_per_rank"]
                if tokens in workload_hashes:
                    assert hashes == workload_hashes[tokens]
                workload_hashes[tokens] = hashes
                record.update(model=model, transport_dtype="fp8_e4m3fn" if dtype == "fp8" else "bf16",
                              return_dtype="bf16", checkpoint_dtype="fp8_e4m3fn" if dtype == "fp8" else "bf16",
                              source_fp8=dtype == "fp8", compact_workspace=dtype == "fp8",
                              native_sha256=hashlib.sha256((native / "libweftmoe.so").read_bytes()).hexdigest(),
                              log_directory=str(directory), command=command)
                if args.trace:
                    ranks = [analyze(directory / f"t{tokens}.rank{rank}.json")
                             for rank in range(record["world"])]
                    kernel_summary = {}
                    for phase in ("weftmoe.forward", "weftmoe.backward"):
                        names = ("::network::kernel", "::backward_stream::kernel")
                        values = []
                        for rank in ranks:
                            assert len(rank[phase]) == args.samples
                            row = []
                            for sample in rank[phase]:
                                events = [e for e in sample["events"]
                                          if any(n in e["name"] for n in names)]
                                assert len(events) == 1, [e["name"] for e in sample["events"]]
                                row.append(events[0]["duration_us"])
                            values.append(row)
                        kernel_summary[phase] = dict(
                            main_p50_us=statistics.median(map(max, zip(*values))),
                            per_rank_main_us=values)
                    record["kernel_summary"] = kernel_summary
                records.append(record)
                print(model, dtype, tokens, "wall_p50_ms",
                      round(record["wall_slowest_rank_p50_us"] / 1000, 3),
                      "kernel_ms", {k: round(v["main_p50_us"] / 1000, 3)
                                    for k, v in record.get("kernel_summary", {}).items()}, flush=True)
            (args.output / "results.json").write_text(json.dumps(dict(
                method="Median of per-step maximum across 16 ranks; two nodes, eight L20X each. "
                       "Identical workloads across dtypes. FP8 weight preparation excluded; BF16 dW. "
                       "FP8 GIN dispatches source FP8 X/dY with scales; returns BF16 and saves FP8 X/AB. "
                       "CUPTI runs are diagnostic and separate from unprofiled wall timing.",
                traced=args.trace, prime_trace=args.prime_trace,
                records=records), indent=2) + "\n")


if __name__ == "__main__":
    main()
