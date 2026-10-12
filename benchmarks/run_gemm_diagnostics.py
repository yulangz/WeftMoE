#!/usr/bin/env python3
"""Build a GEMM-only diagnostic from the current PF backward loops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("backward",),
                        default="backward")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--define", action="append", default=[],
                        help="Compile-time ablation, recorded in provenance")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = args.source.resolve()
    phase = "backward"
    source = root / "src/cuda/backward/compute_kernels.cuh"
    shutil.copyfile(args.profile, directory / "tuning.hpp")
    benchmark = root / f"benchmarks/pure_{args.phase.replace('-', '_')}_gemm.cu"
    command = [os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"), "-O2",
               "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_90a",
               "-lineinfo", "-Xptxas=-v,--warn-on-spills",
               *("-D" + value for value in args.define),
               "--threads", str(args.jobs), "-I" + str(root / "src"),
               "-I" + str(args.profile.resolve().parent),
               "-include", str(directory / "tuning.hpp"),
               "-I" + os.environ.get("CUTLASS_INCLUDE_DIR", "/opt/cutlass/include"),
               "-I" + str(directory), str(benchmark), "-lcuda", "-lcublas",
               "-l:libnvidia-ml.so.1", "-Xcompiler=-pthread",
               "-o", str(directory / "pure_gemm")]
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    dependencies = [source, benchmark, root / "benchmarks/run_gemm_diagnostics.py",
                    root / "benchmarks/gemm_benchmark.cuh",
                    root / "src/cuda/primitives/hopper.cuh",
                    root / f"src/cuda/{phase}/tuning_config.hpp"]
    manifest = {"source": str(root), "phase": args.phase,
                "source_hashes": {str(p.relative_to(root)):
                                  hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in dependencies},
                "profile": str(args.profile.resolve()),
                "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256((directory / "pure_gemm").read_bytes()).hexdigest(),
                "command": command,
                "scope": "Static paired tasks, balanced or rows-file expert extents, preallocated tensors; no routing, activation or communication",
                "timing": "CUDA Graph replay; 2 s spaced joint warmup; 30 alternating event samples, one conditioning call and 50 ms cooldown; NVML clocks before/after each sample",
                "cublas": "balanced: strided batched; rows-file: graph of per-expert GEMMs; BF16/FP32/BF16, reduced-precision reduction disabled"}
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory / "pure_gemm")


if __name__ == "__main__":
    main()
