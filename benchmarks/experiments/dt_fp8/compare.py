"""Paired latency measurements of default and previous winner, without NCU.

These are diagnostic until the candidate passes the full autotune correctness
suite: a benchmark's random inputs do not replace mutation/tail stress cases.
"""
import argparse
import os
from pathlib import Path
import sys

from autotune import ROOT, MODELS, SIZES, atomic_json, idle_gpus, parse_measurements, run_command
from benchmarks.fp8_validation import POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-root", type=Path,
                        default=ROOT / "build/dt-fp8-autotune-m64/jobs")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for model, (h, i, topk) in MODELS.items():
        for candidate in ("default", "m64-s3-r1-c132"):
            native = args.native_root.resolve() / model / candidate / "native"
            env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                OMP_NUM_THREADS="1", NCCL_DEBUG="WARN",
                LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64",
                WEFTMOE_TENSOR_EXTENSION=str(next(native.glob("_runtime*.so"))),
                TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i), BENCH_TOPK=str(topk), BENCH_EXPERTS="256",
                BENCH_SIZES=" ".join(map(str, SIZES)), BENCH_REPEATS="100", BENCH_SWEEPS="11", BENCH_SEED="1700")
            env.pop("BENCH_PROFILE", None)
            env.pop("WEFTMOE_AUTOTUNE_TABLE", None)
            env.pop("WEFTMOE_STATIC_RING_TOKENS", None)
            log = output / f"{model}-{candidate}.log"
            rec = dict(model=model, candidate=candidate, native=str(native))
            print(f"BENCH {model} {candidate}", flush=True)
            try:
                idle_gpus()
                run_command([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                    "benchmarks/experiments/dt_fp8/benchmark.py"], cwd=ROOT, env=env, log=log, timeout=600)
                rec.update(status="passed", rows=parse_measurements(log))
            except RuntimeError as error:
                rec.update(status="failed", error=str(error))
            results.append(rec)
            atomic_json(output / "summary.json", dict(validation=POLICY,
                eligibility="diagnostic timing only; benchmark checks do not replace full correctness suite",
                gpu="8 x NVIDIA L20X", ep=8, results=results))
    idle_gpus()
    if any(rec["status"] != "passed" for rec in results):
        raise RuntimeError("comparison failed; inspect summary.json")


if __name__ == "__main__":
    main()
