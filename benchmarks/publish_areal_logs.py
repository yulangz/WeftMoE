"""Mirror completed AReaL console metrics to W&B without restarting training.

Pass --log name=path for each run and a JSON file of non-secret run metadata.
Only complete metric tables are uploaded. Values retain the console's printed
precision; no raw logs, environment variables, or system metrics are sent.
The API key is read from WANDB_API_KEY or an echo-free terminal prompt.
"""

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import re
import time

import wandb


def read_updates(path):
    raw = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    rows, current = [], None
    for line in raw.splitlines():
        match = re.search(
            r"(\d{8}-\d\d:\d\d:\d\d\.\d+).*Train step (\d+)/\d+ done\.", line
        )
        if match:
            current = {
                "global_step": int(match[2]) - 1,
                "source/completed_at_unix_s": datetime.strptime(
                    match[1], "%Y%m%d-%H:%M:%S.%f"
                ).replace(tzinfo=timezone.utc).timestamp(),
            }
        elif current is not None and line.startswith("│"):
            cells = [cell.strip() for cell in line.strip("│").split("│")]
            for key, value in zip(cells[::2], cells[1::2]):
                try:
                    current[key] = float(value)
                except ValueError:
                    pass
        elif current is not None and line.startswith("╘"):
            rows.append(current)
            current = None
    return rows, "Training failed with exception:" in raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dir", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("WANDB_API_KEY") or getpass.getpass("W&B API key: ")
    metadata = json.loads(args.config.read_text())
    args.dir.mkdir(parents=True, exist_ok=True)
    active = {}
    for spec in args.log:
        name, path = spec.split("=", 1)
        run = wandb.init(
            project=args.project,
            group=args.group,
            name=f"{args.group}-{name}",
            id=f"{args.group}-{name}",
            resume="allow",
            reinit="create_new",
            mode="online",
            save_code=False,
            dir=str(args.dir),
            config={
                **metadata,
                "rollout_backend": name,
                "metric_source": "completed_console_tables",
                "metric_precision": "console_printed",
            },
            settings=wandb.Settings(
                api_key=key,
                base_url=args.base_url,
                disable_git=True,
                disable_code=True,
                x_disable_meta=True,
                x_disable_stats=True,
                x_save_requirements=False,
                console="off",
                x_log_level=40,
            ),
        )
        active[name] = [run, Path(path), run.step - 1]
        print(f"WANDB_RUN {name} {run.url}", flush=True)
    while active:
        for name, state in list(active.items()):
            run, path, last = state
            rows, failed = read_updates(path)
            for row in rows:
                step = row["global_step"]
                if step <= last:
                    continue
                run.log(row, step=step, commit=True)
                run.summary["completed_updates"] = step + 1
                state[2] = last = step
                print(f"WANDB_UPLOADED {name} update={step + 1}", flush=True)
            if failed or last + 1 >= args.steps:
                run.finish(exit_code=int(failed))
                del active[name]
        if active:
            time.sleep(15)


if __name__ == "__main__":
    main()
