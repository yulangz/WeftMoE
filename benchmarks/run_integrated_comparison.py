#!/usr/bin/env python3
"""Run independent WeftMoE/Megatron full Forward baselines and compare E2E.

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

try:
    from .nccl_library import configure_nccl_library
except ImportError:
    from nccl_library import configure_nccl_library


FIXTURE_FIELDS = (
    "weight_generator_version",
    "per_rank_activation_hash", "per_rank_route_hash",
    "per_rank_gate_hash", "per_rank_weight_sample_hash",
    "per_rank_actual_w1_device_hash", "per_rank_actual_w2_device_hash")


def verify_fixture_agreement(mega_runs, megatron_launches):
    agreement = {}
    for field in FIXTURE_FIELDS:
        mega_values = mega_runs[0].get(field)
        for run_index, run_record in enumerate(mega_runs):
            if run_record.get(field) != mega_values:
                raise RuntimeError(
                    f"WeftMoE fixture changed field={field} run={run_index}")
        for run_index, launch in enumerate(megatron_launches):
            actual = launch.get(field)
            if actual != mega_values:
                raise RuntimeError(
                    f"fixture mismatch field={field} run={run_index} "
                    f"WeftMoE={mega_values} Megatron={actual}")
        agreement[field] = mega_values
    return agreement


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
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    args = parser.parse_args()
    if args.runs < 3 or args.warmups < 10 or args.measurements < 100:
        parser.error("fair comparison requires >=3 runs, 10 warmups, 100 measurements")

    environment = os.environ.copy()
    environment["NCCL_DEBUG"] = "WARN"
    if args.nccl_lib_dir:
        configure_nccl_library(environment, args.nccl_lib_dir)

    mega_runs = []
    for _run_index in range(args.runs):
        command = [
            sys.executable, "tests/run_local_activation_p2p.py",
            "--binary", args.binary, "--world", "8",
            "--forward-benchmark", "--top-k", "6",
            "--rounds", str(args.warmups + args.measurements),
            "--gate-mode", args.gate_mode,
            "--timeout", str(args.timeout),
        ]
        mega_runs.append(extract(run(command, environment, args.timeout),
                                 "FORWARD_BENCH_JSON "))

    megatron_launches = []
    for _run_index in range(args.runs):
        megatron_command = [
            "torchrun", "--standalone", "--nproc-per-node=8",
            "benchmarks/megatron_dispatch_baseline.py",
            "--warmups", str(args.warmups),
            "--iterations", str(args.measurements),
            "--runs", "1",
            "--gate-mode", args.gate_mode,
            "--expert-body", "te-grouped-mlp",
        ]
        megatron_launches.append(extract(
            run(megatron_command, environment, args.timeout),
            "MEGATRON_BASELINE_JSON "))

    # Preserve one self-contained Megatron record while making the aggregate
    # explicitly span separate process/NCCL-session launches, just as WeftMoE.
    megatron = dict(megatron_launches[0])
    megatron["runs"] = [launch["runs"][0]
                         for launch in megatron_launches]

    mega_means_ms = [
        item["wall_slowest_rank_us"]["mean"] / 1000.0
        for item in mega_runs]
    megatron_common_means = [
        launch["common_forward_e2e_host_wall_cross_run"]["mean_ms"]
        for launch in megatron_launches]
    megatron_native_means = [
        launch["native_forward_e2e_host_wall_cross_run"]["mean_ms"]
        for launch in megatron_launches]
    megatron_nccl = megatron_launches[0]["nccl_runtime_version"]
    mega_nccl = {
        version for item in mega_runs
        for version in item["per_rank_nccl_runtime"]}
    megatron["independent_process_launches"] = args.runs
    megatron["common_forward_e2e_host_wall_cross_run"] = {
        "mean_ms": statistics.fmean(megatron_common_means),
        "variance_ms2": statistics.pvariance(megatron_common_means),
        "per_run_mean_ms": megatron_common_means,
    }
    megatron["native_forward_e2e_host_wall_cross_run"] = {
        "mean_ms": statistics.fmean(megatron_native_means),
        "variance_ms2": statistics.pvariance(megatron_native_means),
        "per_run_mean_ms": megatron_native_means,
    }
    mega_mean = statistics.fmean(mega_means_ms)
    common_mean = statistics.fmean(megatron_common_means)
    speedup = (common_mean - mega_mean) / common_mean * 100.0
    native_mean = statistics.fmean(megatron_native_means)
    native_speedup = (native_mean - mega_mean) / native_mean * 100.0
    per_run_native_speedup = [
        (native - mega) / native * 100.0
        for native, mega in zip(megatron_native_means, mega_means_ms)]
    fixture_agreement = verify_fixture_agreement(
        mega_runs, megatron_launches)
    result = {
        "schema": 1,
        "comparison": "single-node-8-rank-full-forward-integrated-host-wall",
        "gate_mode": args.gate_mode,
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
        "weftmoe_integrated_e2e_ms": {
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
        "speedup_vs_megatron_common_percent": speedup,
        "speedup_vs_megatron_native_percent": native_speedup,
        "per_run_speedup_vs_megatron_native_percent":
            per_run_native_speedup,
        "formal_target_at_least_15_percent":
            all(value >= 15.0 for value in per_run_native_speedup),
        "fixture_hash_agreement": fixture_agreement,
        "cross_implementation_output_equivalence": {
            "method": "explicit output artifacts; see --save-output-dir runs",
            "byte_identical": False,
            "status": "not asserted by timing-only invocation",
        },
        "weftmoe_runs": mega_runs,
        "megatron": megatron,
        "environment_overrides": {
            key: environment[key] for key in (
                "CUDA_VISIBLE_DEVICES", "NCCL_ALGO", "NCCL_PROTO",
                "NCCL_P2P_LEVEL", "NCCL_DEBUG") if key in environment
        },
        "runtime_parity": {
            "weftmoe_nccl_runtime_versions": sorted(mega_nccl),
            "megatron_torch_nccl_version": megatron_nccl,
            "megatron_torch_build_nccl_version":
                megatron_launches[0]["torch_nccl_version"],
            "nccl_versions_identical": mega_nccl == {megatron_nccl},
            "weftmoe_loaded_libraries": sorted({
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
    if any(value < 15.0 for value in per_run_native_speedup):
        raise RuntimeError(
            "at least one independent integrated native speedup is below "
            f"15%: {per_run_native_speedup}")


if __name__ == "__main__":
    main()
