#!/usr/bin/env python3
"""Build a static-task microbenchmark from production FWD or BWD GEMMs.

This is a GEMM diagnostic, not an end-to-end comparison. Allocations, routing,
SwiGLU and communication are excluded. The generated binary compares against
cuBLAS on identical inputs. Flex's actual TE GEMMs are measured separately by
forward_diagnostics.py, including a CUDA Graph replay without Python launch gaps.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("forward", "backward"), default="forward")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = args.source.resolve()
    source = root / ("src/cuda/persistent_v3/forward/compute_kernel.cuh"
                     if args.phase == "forward" else
                     "src/cuda/persistent_v3/backward/compute_kernels.cuh")
    code = source.read_text()
    if args.phase == "forward":
        extract_forward(code, directory)
    shutil.copyfile(args.profile, directory / "tuning.hpp")
    benchmark = root / f"benchmarks/pure_{args.phase}_gemm.cu"
    command = [os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"), "-O2",
               "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_90a",
               "--threads", str(args.jobs), "-I" + str(root / "src"),
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
                    root / f"src/cuda/{args.phase}/tuning_config.hpp"]
    manifest = {"source": str(root), "phase": args.phase,
                "source_hashes": {str(p.relative_to(root)):
                                  hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in dependencies},
                "profile": str(args.profile.resolve()),
                "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256((directory / "pure_gemm").read_bytes()).hexdigest(),
                "command": command,
                "scope": "Static paired tasks, balanced M, preallocated tensors; no routing, activation or communication",
                "timing": "CUDA Graph replay; 2 s spaced joint warmup; 30 alternating event samples, one conditioning call and 50 ms cooldown; NVML clocks before/after each sample",
                "cublas": "strided batched BF16/FP32/BF16; reduced-precision reduction disabled"}
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(directory / "pure_gemm")


def omit_checkpoints(code):
    """Remove checkpoint-only blocks from the pure FC1 math diagnostic."""
    start = "if constexpr (RetainBackwardState) {"
    # Ignore braces in comments and literals when matching a C++ block.
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    while start in code:
        begin = code.index(start)
        opening = begin + len(start) - 1
        depth = 0
        for token in tokens.finditer(code, opening):
            if token.group() == "{":
                depth += 1
            elif token.group() == "}":
                depth -= 1
                if depth == 0:
                    code = code[:begin] + code[token.end():]
                    break
        else:
            raise ValueError("Unterminated FC1 checkpoint block")
    return code


def extract_forward(code, directory):
    # The diagnostic producer must use the same gate/up stage layout as the
    # extracted consumer, including when comparing an older source snapshot.
    packed = "constexpr bool kPackedFc1 =" in code
    (directory / "fc1_features.cuh").write_text(
        "#define WEFTMOE_DIAGNOSTIC_PACKED_FC1 " + str(int(packed)) + "\n")
    for name, start, end in (
        ("fc1", "using Fc1TileMma =", "fc1_pipeline += iterations;"),
        ("fc2", "using Fc2TileMma =", "fc2_pipeline += kFc2Iterations;"),
    ):
        begin = code.index(start)
        stop = code.index(end, begin) + len(end)
        math = code[begin:stop]
        if name == "fc1":
            math = omit_checkpoints(math)
        (directory / f"{name}_math.cuh").write_text(math + "\n")
    helper_begin = code.index("// A 64-column atom uses")
    helper_end = code.index("// FC1/FC2 lends", helper_begin)
    (directory / "epilogue_helpers.cuh").write_text(code[helper_begin:helper_end])
    marker = "GEMM_DIAGNOSTIC_FC2_EPILOGUE_BEGIN"
    begin = code.index("\n", code.index(marker)) + 1
    stop = code.rfind("\n", begin, code.index("GEMM_DIAGNOSTIC_FC2_EPILOGUE_END", begin))
    epilogue = code[begin:stop].replace(
        "static_cast<std::uint32_t>(row)",
        "static_cast<std::uint32_t>(expert * M + row)")
    (directory / "fc2_store.cuh").write_text(epilogue + "\n")


if __name__ == "__main__":
    main()
