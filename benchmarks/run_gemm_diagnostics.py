#!/usr/bin/env python3
"""Build a static-task microbenchmark from production FWD or BWD GEMMs.

This is a GEMM diagnostic, not an end-to-end comparison. Allocations, routing
and communication are excluded. Static-forward includes SwiGLU and actual
expert tails; legacy forward/backward modes measure GEMMs only. The generated
binary compares against cuBLAS on identical inputs. Flex's TE GEMMs are measured
separately by forward_diagnostics.py, including a CUDA Graph replay.
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
    parser.add_argument("--phase", choices=("forward", "backward", "static-forward"),
                        default="forward")
    parser.add_argument("--logical-n", type=int, choices=(32, 64, 128), default=64,
                        help="Static FC1 SwiGLU output columns per task")
    parser.add_argument("--stages", type=int, choices=(2, 3, 4, 5, 6), default=2)
    parser.add_argument("--tile-k", type=int, choices=(64, 128), default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    root = args.source.resolve()
    phase = "forward" if args.phase == "static-forward" else args.phase
    source = root / ("src/cuda/persistent_v3/forward/compute_kernel.cuh"
                     if phase == "forward" else
                     "src/cuda/persistent_v3/backward/compute_kernels.cuh")
    code = source.read_text()
    if args.phase == "static-forward":
        extract_static_forward(code, directory, root)
    elif args.phase == "forward":
        extract_forward(code, directory)
    if args.phase == "static-forward":
        # Preserve relative includes in the selected frozen profile.
        (directory / "tuning.hpp").write_text(
            '#include "' + str(args.profile.resolve()) + '"\n')
    else:
        shutil.copyfile(args.profile, directory / "tuning.hpp")
    benchmark = root / f"benchmarks/pure_{args.phase.replace('-', '_')}_gemm.cu"
    command = [os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"), "-O2",
               "-std=c++17", "--expt-relaxed-constexpr", "-arch=sm_90a",
               "--threads", str(args.jobs), "-I" + str(root / "src"),
               "-include", str(directory / "tuning.hpp"),
               "-I" + os.environ.get("CUTLASS_INCLUDE_DIR", "/opt/cutlass/include"),
               "-I" + str(directory), str(benchmark), "-lcuda", "-lcublas",
               "-l:libnvidia-ml.so.1", "-Xcompiler=-pthread",
               "-o", str(directory / "pure_gemm")]
    if args.phase == "static-forward":
        command += [f"-DWEFTMOE_GEMM_LOGICAL_N={args.logical_n}",
                    f"-DWEFTMOE_GEMM_STAGES={args.stages}",
                    f"-DWEFTMOE_GEMM_TILE_K={args.tile_k}"]
    with (directory / "build.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    dependencies = [source, benchmark, root / "benchmarks/run_gemm_diagnostics.py",
                    root / "benchmarks/gemm_benchmark.cuh",
                    root / "src/cuda/primitives/hopper.cuh",
                    root / f"src/cuda/{phase}/tuning_config.hpp"]
    if args.phase == "static-forward":
        dependencies += [root / "src/cuda/static_v4/fc1_epilogue.cuh",
                         root / "src/cuda/forward/support/fallback_kernels.cuh"]
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
    if args.phase == "static-forward":
        manifest["scope"] = (
            "Canonical local X; actual balanced/hotspot expert tails; "
            "FC1+SwiGLU or FC1+SwiGLU followed by fixed FC2; no communication")
        manifest["logical_n"] = args.logical_n
        manifest["stages"] = args.stages
        manifest["tile_k"] = args.tile_k
        manifest["cublas"] = "Per-expert BF16/FP32/BF16 plus identical rounded SwiGLU"
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


def extract_static_forward(code, directory, root):
    """Reuse production instruction loops, not a separately maintained GEMM."""
    begin = 0
    for name in ("packed", "separated"):
        begin = code.index("using Fc1TileMma =", begin)
        end = code.index("fc1_pipeline += iterations;", begin)
        end += len("fc1_pipeline += iterations;")
        (directory / f"fc1_{name}_math.cuh").write_text(
            omit_checkpoints(code[begin:end]) + "\n")
        begin = end
    begin = code.index("using Fc2TileMma =")
    end = code.index("fc2_pipeline += kFc2Iterations;", begin)
    end += len("fc2_pipeline += kFc2Iterations;")
    (directory / "fc2_math.cuh").write_text(code[begin:end] + "\n")
    helpers = []
    for first, last in (
        ("// FC1's WGMMA accumulator", "#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)"),
        ("// A 64-column atom uses", "// FC1/FC2 lends"),
        ("// TMA lays a K128 stage", "template <bool StaticLocal, bool RetainBackwardState,"),
    ):
        begin = code.index(first)
        helpers.append(code[begin:code.index(last, begin)])
    conversions = (root / "src/cuda/forward/support/fallback_kernels.cuh").read_text()
    helpers.insert(0, conversions[:conversions.index("__device__ float gate_after_accumulate")])
    (directory / "static_gemm_helpers.cuh").write_text("\n".join(helpers))


if __name__ == "__main__":
    main()
