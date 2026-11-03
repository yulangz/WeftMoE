"""Summarize local AReaL StatsLogger records without uploading artifacts."""

import argparse
import json
import math
from pathlib import Path
import re
import statistics


def summarize(run):
    steps = []
    current = None
    for line in (run / "rl.log").read_text(errors="replace").splitlines():
        match = re.search(r"Train step (\d+)/\d+ done", line)
        if match:
            current = {"step": int(match[1])}
            steps.append(current)
        if current is not None and "│" in line:
            cells = line.split("│")[1:-1]
            for key, value in zip(cells[::2], cells[1::2]):
                key = key.strip()
                if key.startswith(("timeperf/", "train_perf/", "ppo_actor/update/",
                                   "ppo_actor/task_reward/")):
                    try:
                        current[key] = float(value.strip())
                    except ValueError:
                        pass
    keys = ("timeperf/train_step", "timeperf/rollout", "timeperf/update_weights",
            "train_perf/tokens_per_second", "ppo_actor/update/grad_norm",
            "ppo_actor/task_reward/avg", "ppo_actor/update/logp_abs_diff/avg")
    means = {}
    for key in keys:
        values = [s[key] for s in steps[1:] if key in s and math.isfinite(s[key])]
        means[key] = statistics.mean(values) if values else None
    status_path = run / "status.json"
    return dict(run=str(run.resolve()), completed_steps=len(steps),
                status=json.loads(status_path.read_text()) if status_path.exists() else {},
                successful_updates=sum(s.get("ppo_actor/update/update_successful") == 1 for s in steps),
                nonfinite_grad_steps=[s["step"] for s in steps
                                      if not math.isfinite(s.get("ppo_actor/update/grad_norm", float("nan")))],
                means_excluding_first_step=means, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--label", default="fp8")
    args = parser.parse_args()
    result = {args.label: summarize(args.run),
              "scope": "Historical BF16 comparison; sampled token counts may differ. "
                       "Quantization and graph rebuilding contribute to end-to-end runtime; "
                       "update_weights excludes decode graph recapture. "
                       "StatsLogger table values are rounded; retain W&B for full precision."}
    if args.baseline and (args.baseline / "rl.log").exists():
        result["historical_bf16"] = summarize(args.baseline)
    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    output = args.run / "report.json"
    output.write_text(json.dumps(finite_json(result), indent=2, allow_nan=False))
    print(output)


if __name__ == "__main__":
    main()
