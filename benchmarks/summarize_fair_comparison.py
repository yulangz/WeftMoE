#!/usr/bin/env python3
"""Pool every formal wall-time sample; never select the fastest repetition.

The output retains the raw per-iteration rank maxima, per-pass means, and the
allocation/checkpoint scope alongside the aggregate statistics. Native Forward
CUDA events exclude Notify, so only host-wall ratios compare implementations.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text())


def stats(samples):
    if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(samples)
    mean = statistics.mean(samples)
    return dict(count=len(samples), mean_ms=mean, p50_ms=statistics.median(samples),
                p95_ms=ordered[math.ceil(0.95 * len(samples)) - 1],
                min_ms=min(samples), max_ms=max(samples),
                stddev_ms=statistics.pstdev(samples),
                cv=statistics.pstdev(samples) / mean)


def summarize(root):
    manifest = read(root / "manifest.json")
    if manifest.get("schema") != 2:
        raise ValueError("unsupported manifest schema; use the matching benchmark revision")
    if not manifest.get("completed"):
        raise ValueError("formal comparison is incomplete")
    iterations, passes = manifest["iterations"], manifest["passes"]
    cases, flex_validation = [], {}
    for model in ("qwen", "flash"):
        for ep in (8, 16):
            for tokens in (8192, 65536):
                case = f"{model}_ep{ep}_t{tokens}"
                sms = manifest["flex_selection"][case]
                profile = "short" if tokens == 8192 else "long"
                row = dict(case=case, model=model, ep=ep, tokens_per_rank=tokens,
                           flex_sms=sms, phases={})
                native_path = root / f"after_{profile}" / f"{model}_ep{ep}_forward_t{tokens}_pass1.json"
                row["shape"] = read(native_path)["record"]["shape"]
                if row["shape"]["tokens_per_rank"] != tokens:
                    raise ValueError(f"token count mismatch: {native_path}")
                for phase in ("forward", "backward"):
                    measurements = {}
                    for implementation in ("before", "after", "flex"):
                        directory = implementation if implementation != "after" else f"after_{profile}"
                        wall, gpu, pass_means, pass_medians, files = [], [], [], [], []
                        for repeat in range(1, passes + 1):
                            if implementation == "flex":
                                path = root / directory / f"{model}_ep{ep}_flex_t{tokens}_sm{sms}_pass{repeat}.json"
                                data = read(path)
                                if not (data["allocate_tensors"] and data["synchronize_ranks"]):
                                    raise ValueError(f"allocation scope mismatch: {path}")
                                measured = data["runs"][0]["forward_e2e" if phase == "forward" else "backward_only"]
                                shape = dict(data["shape"], tokens_per_rank=data["shape"]["tokens"])
                                world = data["world"]
                                sample = measured["wall_samples_ms"]
                                cuda = measured["cuda_samples_ms"]
                                if repeat == 1:
                                    flex_validation[case] = data["validation"]
                            else:
                                path = root / directory / f"{model}_ep{ep}_{phase}_t{tokens}_pass{repeat}.json"
                                data = read(path)
                                measured = data["record"]
                                shape, world = measured["shape"], measured["world"]
                                if (data["returncode"] or not measured["allocate_tensors"] or
                                        not measured["synchronize_ranks"]):
                                    raise ValueError(f"native result/scope mismatch: {path}")
                                if any(measured["per_rank_checked_repetitions"]):
                                    raise ValueError(f"correctness checks enabled during formal timing: {path}")
                                if min(measured["per_rank_tensor_allocation_requests"]) <= iterations:
                                    raise ValueError(f"missing per-call allocation: {path}")
                                sample = [x / 1000 for x in measured["wall_slowest_rank_samples_us"]]
                                key = "forward" if phase == "forward" else "total"
                                cuda = [x / 1000 for x in measured[key + "_slowest_rank_samples_us"]]
                            if world != ep or any(shape.get(k) != v for k, v in row["shape"].items()):
                                raise ValueError(f"workload mismatch: {path}")
                            if len(sample) != iterations or len(cuda) != iterations:
                                raise ValueError(f"sample count mismatch: {path}")
                            wall.extend(sample)
                            gpu.extend(cuda)
                            pass_means.append(statistics.mean(sample))
                            pass_medians.append(statistics.median(sample))
                            files.append(str(path.relative_to(root)))
                        measurements[implementation] = dict(
                            wall=stats(wall), wall_samples_ms=wall,
                            pass_wall_means_ms=pass_means,
                            pass_wall_p50_ms=pass_medians,
                            cuda=stats(gpu), cuda_samples_ms=gpu,
                            cuda_scope=("complete phase GPU event interval" if implementation == "flex" else
                                        "Backward kernels; excludes fixed input/output preparation" if phase == "backward" else
                                        "Forward kernels; excludes Notify and fixed input/output preparation"),
                            source_files=files)
                    after = measurements["after"]["wall"]["mean_ms"]
                    after_p50 = measurements["after"]["wall"]["p50_ms"]
                    row["phases"][phase] = dict(
                        measurements=measurements,
                        before_over_after_mean=measurements["before"]["wall"]["mean_ms"] / after,
                        flex_over_after_mean=measurements["flex"]["wall"]["mean_ms"] / after,
                        before_over_after_p50=measurements["before"]["wall"]["p50_ms"] / after_p50,
                        flex_over_after_p50=measurements["flex"]["wall"]["p50_ms"] / after_p50)
                cases.append(row)
    return dict(schema=2, manifest=manifest, metric="per-iteration slowest-rank host wall time",
                primary_statistic="P50; mean, P95, dispersion, and all samples are retained",
                tokens="per rank", allocation="fresh fixed tensor ownership and D2D input initialization inside each call; warmed native/PyTorch caches",
                excluded="one-time weights, workspace, peer mappings, RDMA/DeepEP contexts; prior-output release; warmups and correctness",
                checkpoint="retain routed X and BF16 A/B; regenerate gated SwiGLU U in Backward",
                cases=cases, flex_validation=flex_validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for phase in ("forward", "backward"):
        print(phase)
        for case in result["cases"]:
            row = case["phases"][phase]
            values = [row["measurements"][name]["wall"]["p50_ms"] for name in ("before", "after", "flex")]
            print(case["case"], *(f"{x:.3f}" for x in values),
                  f"P50 before/after={row['before_over_after_p50']:.3f}",
                  f"P50 flex/after={row['flex_over_after_p50']:.3f}",
                  f"mean before/after={row['before_over_after_mean']:.3f}",
                  f"mean flex/after={row['flex_over_after_mean']:.3f}")


if __name__ == "__main__":
    main()
