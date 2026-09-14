#!/usr/bin/env python3
"""Compare static expert GEMMs against cuBLAS using prebuilt diagnostics.

Build <model>_forward/pure_gemm and <model>_backward/pure_gemm with
run_gemm_diagnostics.py first. This measures CUDA Graph replay with preallocated
tensors; use run_performance_matrix.py for allocation-inclusive operator time.
Clocks are observed, never changed by this script. Tokens are per EP rank.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

MODELS = {"qwen": (2048, 512, 256, 8), "flash": (4096, 2048, 256, 6)}


def parse_output(output):
    if "VALIDATION full_output=1 repeat_byte_identical=1" not in output:
        raise RuntimeError("Full-output validation is missing")
    result = json.loads(next(line for line in output.splitlines() if line.startswith("{")))
    for index, prefix in enumerate(("native", "cublas")):
        for tag, key in (("SAMPLES", "samples_ms"), ("CLOCKS", "clocks_mhz"),
                         ("WINDOWS", "windows_ns")):
            values = next(line for line in output.splitlines()
                          if line.startswith(f"{tag} {index} ")).split()[2:]
            result[f"{prefix}_{key}"] = (
                [float(value) for value in values] if tag == "SAMPLES" else
                [[int(x) for x in value.split(":")] for value in values])
        samples = result[f"{prefix}_samples_ms"]
        if len(samples) != 30 or any(value <= 0 for value in samples):
            raise RuntimeError("Expected 30 positive timing samples")
        median = statistics.median(samples)
        result[f"{prefix}_p50_ms"] = median
        result[f"{prefix}_p50_tflops"] = result["flops"] / median / 1e9
    result["fraction_of_cublas"] = result["cublas_p50_ms"] / result["native_p50_ms"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=list(MODELS))
    parser.add_argument("--eps", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--ops", choices=["fc1", "fc2", "du", "dx", "dw1", "dw2"],
                        nargs="+", default=["fc2", "du", "dx", "dw1", "dw2"])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ctas", type=int, default=132)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cooldown-ms", type=int, default=50)
    parser.add_argument("--clock-mhz", type=int, help="Report whether every sample endpoint matches this target")
    args = parser.parse_args()
    if args.repeats < 1 or args.ctas < 2 or args.ctas % 2 or not 0 <= args.cooldown_ms <= 2000:
        parser.error("Invalid repetitions, CTA count or cooldown")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, WEFTMOE_GEMM_COOLDOWN_MS=str(args.cooldown_ms))
    for repeat in range(args.repeats):
        for model in args.models:
            hidden, intermediate, experts, top_k = MODELS[model]
            for ep in args.eps:
                if ep < 1 or experts % ep:
                    parser.error("EP must divide the expert count")
                groups = experts // ep
                for tokens in args.tokens:
                    routes = tokens * top_k
                    if tokens < 1 or routes % groups or (routes // groups) % 128:
                        parser.error("The static diagnostic requires balanced M divisible by 128")
                    rows = routes // groups
                    shapes = dict(fc1=(rows, 2 * intermediate, hidden),
                                  fc2=(rows, hidden, intermediate), du=(rows, intermediate, hidden),
                                  dx=(rows, hidden, 2 * intermediate), dw1=(2 * intermediate, hidden, rows),
                                  dw2=(hidden, intermediate, rows))
                    for op in args.ops:
                        phase = "forward" if op.startswith("fc") else "backward"
                        binary = args.binaries.resolve() / f"{model}_{phase}/pure_gemm"
                        command = [str(binary), op, *map(str, shapes[op]), str(groups),
                                   str(args.device), str(args.ctas), str(repeat % 2), "0"]
                        path = args.output / f"{model}_ep{ep}_t{tokens}_{op}_r{repeat}"
                        process = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                                 stderr=subprocess.STDOUT, timeout=300)
                        path.with_suffix(".log").write_text(process.stdout)
                        if process.returncode:
                            raise RuntimeError(f"GEMM failed: {path}.log\n{process.stdout[-2000:]}")
                        result = parse_output(process.stdout)
                        result.update(model=model, ep=ep, tokens=tokens, repeat=repeat, command=command)
                        if args.clock_mhz:
                            result["all_endpoints_at_target"] = all(
                                clock == args.clock_mhz for prefix in ("native", "cublas")
                                for pair in result[f"{prefix}_clocks_mhz"] for clock in pair)
                        path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
                        print(f"{path.name}: {result['native_p50_tflops']:.1f} / "
                              f"{result['cublas_p50_tflops']:.1f} TFLOPS", flush=True)


if __name__ == "__main__":
    main()
