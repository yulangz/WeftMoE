#!/usr/bin/env python3
"""Run independent MegaMOE/Megatron identity-EP baselines and compare E2E.

Only comparable host-wall, slowest-rank integrated measurements enter the
reported speedup. Component means are retained for diagnosis, never summed to
manufacture an end-to-end result.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys


def extract(output, marker):
    lines = [line for line in output.splitlines() if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(f"expected one {marker.strip()} record, got {len(lines)}")
    return json.loads(lines[0][len(marker):])


def run(command, environment, timeout):
    completed = subprocess.run(
        command, env=environment, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed exit={completed.returncode}: {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}")
    return completed.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--measurements", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--nccl-lib-dir")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.runs < 3 or args.warmups < 10 or args.measurements < 100:
        parser.error("fair comparison requires >=3 runs, 10 warmups, 100 measurements")

    environment = os.environ.copy()
    environment["NCCL_DEBUG"] = "WARN"
    if args.nccl_lib_dir:
        current = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            args.nccl_lib_dir + (":" + current if current else ""))

    mega_runs = []
    for _run_index in range(args.runs):
        command = [
            sys.executable, "tests/run_local_activation_p2p.py",
            "--binary", args.binary, "--world", "8",
            "--integrated-benchmark", "--top-k", "6",
            "--rounds", str(args.warmups + args.measurements),
            "--timeout", str(args.timeout),
        ]
        mega_runs.append(extract(run(command, environment, args.timeout),
                                 "INTEGRATED_BENCH_JSON "))

    megatron_launches = []
    for _run_index in range(args.runs):
        megatron_command = [
            "torchrun", "--standalone", "--nproc-per-node=8",
            "benchmarks/megatron_dispatch_baseline.py",
            "--warmups", str(args.warmups),
            "--iterations", str(args.measurements),
            "--runs", "1",
        ]
        megatron_launches.append(extract(
            run(megatron_command, environment, args.timeout),
            "MEGATRON_BASELINE_JSON "))

    # Preserve one self-contained Megatron record while making the aggregate
    # explicitly span separate process/NCCL-session launches, just as MegaMOE.
    megatron = dict(megatron_launches[0])
    megatron["runs"] = [launch["runs"][0]
                         for launch in megatron_launches]

    mega_means_ms = [
        item["wall_slowest_rank_us"]["mean"] / 1000.0
        for item in mega_runs]
    megatron_common_means = [
        launch["common_identity_e2e_host_wall_cross_run"]["mean_ms"]
        for launch in megatron_launches]
    megatron_native_means = [
        launch["native_identity_e2e_host_wall_cross_run"]["mean_ms"]
        for launch in megatron_launches]
    megatron_nccl = megatron_launches[0]["torch_nccl_version"]
    mega_nccl = {
        version for item in mega_runs
        for version in item["per_rank_nccl_runtime"]}
    if mega_nccl != {megatron_nccl}:
        raise RuntimeError(
            "NCCL runtime mismatch: MegaMOE=" + str(sorted(mega_nccl)) +
            " Megatron=" + megatron_nccl)
    megatron["independent_process_launches"] = args.runs
    megatron["common_identity_e2e_host_wall_cross_run"] = {
        "mean_ms": statistics.fmean(megatron_common_means),
        "variance_ms2": statistics.pvariance(megatron_common_means),
        "per_run_mean_ms": megatron_common_means,
    }
    megatron["native_identity_e2e_host_wall_cross_run"] = {
        "mean_ms": statistics.fmean(megatron_native_means),
        "variance_ms2": statistics.pvariance(megatron_native_means),
        "per_run_mean_ms": megatron_native_means,
    }
    mega_mean = statistics.fmean(mega_means_ms)
    common_mean = statistics.fmean(megatron_common_means)
    speedup = (common_mean - mega_mean) / common_mean * 100.0
    result = {
        "schema": 1,
        "comparison": "single-node-8-rank-identity-EP-integrated-host-wall",
        "shape": {
            "tokens_per_rank": 8192, "hidden": 7168,
            "intermediate": 3072, "experts": 384, "top_k": 6,
            "dtype": "BF16 activation/output; FP32 gate/accumulate",
        },
        "measurement": {
            "independent_runs": args.runs,
            "warmups_per_run": args.warmups,
            "measurements_per_run": args.measurements,
            "rank_aggregation": "MAX",
            "clock": "host wall through output-ready and required drain",
        },
        "megamoe_integrated_e2e_ms": {
            "per_run_mean": mega_means_ms,
            "mean": mega_mean,
            "variance": statistics.pvariance(mega_means_ms),
        },
        "megatron_common_semantics_e2e_ms": {
            "per_run_mean": megatron_common_means,
            "mean": common_mean,
            "variance": statistics.pvariance(megatron_common_means),
        },
        "megatron_native_e2e_ms": {
            "per_run_mean": megatron_native_means,
            "mean": statistics.fmean(megatron_native_means),
            "variance": statistics.pvariance(megatron_native_means),
        },
        "formal_common_semantics_speedup_percent": speedup,
        "target_at_least_15_percent": speedup >= 15.0,
        "cross_implementation_output_equivalence": {
            "method": "both outputs compared elementwise to one shared reference",
            "byte_identical": all(
                launch["output_validation"]["common_semantics"]
                      ["byte_identical_to_common_reference"]
                for launch in megatron_launches),
        },
        "megamoe_runs": mega_runs,
        "megatron": megatron,
        "environment_overrides": {
            key: environment[key] for key in (
                "CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCL_PROTO",
                "NCCL_P2P_LEVEL", "NCCL_DEBUG") if key in environment
        },
        "runtime_parity": {
            "nccl_version": megatron_nccl,
            "megamoe_loaded_libraries": sorted({
                library for item in mega_runs
                for library in item["per_rank_nccl_library"]}),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "cuda_visible_devices": environment.get(
                "CUDA_VISIBLE_DEVICES", "<unset: all eight visible>"),
        },
    }
    encoded = json.dumps(result, sort_keys=True)
    print("FAIR_BASELINE_JSON " + encoded)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as destination:
            destination.write(encoded + "\n")
    if speedup < 15.0:
        raise RuntimeError(f"integrated speedup {speedup:.3f}% is below 15%")


if __name__ == "__main__":
    main()
