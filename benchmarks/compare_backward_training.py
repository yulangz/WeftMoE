#!/usr/bin/env python3
"""Fail-closed comparison for the formal Notify-inclusive training boundary."""

import argparse
import json
from pathlib import Path


def load_marker(path: str, marker: str):
    text = Path(path).read_text(encoding="utf-8")
    if marker in text:
        return json.loads(text.rsplit(marker, 1)[1].splitlines()[0])
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--megamoe", nargs="+", required=True,
                        help="Three BACKWARD_BENCH_JSON logs/JSON files")
    parser.add_argument("--megatron", required=True,
                        help="MEGATRON_BASELINE_JSON log/JSON file")
    parser.add_argument("--minimum-percent", type=float, default=15.0)
    args = parser.parse_args()

    megamoe = [load_marker(path, "BACKWARD_BENCH_JSON ")
               for path in args.megamoe]
    megatron = load_marker(args.megatron, "MEGATRON_BASELINE_JSON ")
    if len(megamoe) < 3:
        raise SystemExit("at least three independent MegaMOE sessions required")
    native_backward = megatron["native_backward_host_wall_cross_run"][
        "per_run_mean_ms"]
    native_step = megatron["native_forward_backward_host_wall_cross_run"][
        "per_run_mean_ms"]
    if len(native_backward) < len(megamoe) or len(native_step) < len(megamoe):
        raise SystemExit("Megatron does not contain matching independent runs")

    rows = []
    for index, result in enumerate(megamoe):
        if result.get("world") != 8 or result.get("warmup", 0) < 10 or \
                result.get("measured", 0) < 100:
            raise SystemExit(f"run {index}: world/warmup/measurement mismatch")
        audits = result.get("per_rank_notify_collective_audit", [])
        if len(audits) != 8 or any(
                audit != {"fused": 1, "control": 0, "config": 0,
                          "count": 0, "summary": 0}
                for audit in audits):
            raise SystemExit(f"run {index}: release Notify was not fused")
        if result.get("per_rank_dispatch_consumer_acquires") != [1] * 8 or \
                result.get("per_rank_dispatch_consumer_releases") != [1] * 8:
            raise SystemExit(f"run {index}: consumer handoff was not one-shot")
        backward_ms = result["backward_wall_slowest_rank_us"]["mean"] / 1000.0
        # This exact field is the only accepted formal F+B boundary. The
        # post-notify diagnostic is deliberately never read here.
        step_ms = result[
            "notify_inclusive_training_step_wall_slowest_rank_us"]["mean"] / 1000.0
        backward_gain = (native_backward[index] - backward_ms) / \
            native_backward[index] * 100.0
        step_gain = (native_step[index] - step_ms) / native_step[index] * 100.0
        rows.append({"run": index + 1, "megamoe_backward_ms": backward_ms,
                     "megatron_backward_ms": native_backward[index],
                     "backward_improvement_percent": backward_gain,
                     "megamoe_notify_inclusive_training_step_ms": step_ms,
                     "megatron_training_step_ms": native_step[index],
                     "training_step_improvement_percent": step_gain})
        if backward_gain < args.minimum_percent or step_gain < args.minimum_percent:
            raise SystemExit(
                f"run {index + 1}: backward={backward_gain:.3f}% "
                f"notify-inclusive-step={step_gain:.3f}% below "
                f"{args.minimum_percent:.3f}%")
    print("FAIR_BACKWARD_TRAINING_JSON " + json.dumps(
        {"schema": 1,
         "formal_boundary": "route/gate-ready-through-final-gradient-drain",
         "minimum_percent": args.minimum_percent, "runs": rows},
        sort_keys=True))


if __name__ == "__main__":
    main()
