"""Reclaim processes belonging to one explicit local AReaL trial."""

import argparse
import json
import os
from pathlib import Path
import time

import psutil


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def cleanup(experiment, trial, run_root, *, apply=False):
    targets = {}
    protected = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
    run_root = str(Path(run_root).resolve())
    for process in psutil.process_iter():
        if process.pid in protected or not alive(process):
            continue
        try:
            argv = process.cmdline()
            fields = dict(zip(argv, argv[1:]))
            owned_rpc = "areal.infra.rpc.rpc_server" in argv and (
                fields.get("--experiment-name"), fields.get("--trial-name")
            ) == (experiment, trial)
            owned_env = process.environ().get("AREAL_EXPERIMENT_ROOT") == run_root
            if owned_rpc or owned_env:
                targets[process.pid] = process
                for child in process.children(recursive=True):
                    if child.pid not in protected:
                        targets[child.pid] = child
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    report = dict(experiment=experiment, trial=trial, apply=apply,
                  processes=len(targets), pids=sorted(targets))
    if not apply:
        return report
    for action, seconds in (("terminate", 10), ("kill", 30)):
        for process in targets.values():
            try:
                if alive(process):
                    getattr(process, action)()
            except psutil.NoSuchProcess:
                pass
        # pidfd_open is unsupported in this container; avoid psutil.wait_procs.
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline and any(alive(p) for p in targets.values()):
            time.sleep(0.5)
    report["remaining"] = [p.pid for p in targets.values() if alive(p)]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = cleanup(args.experiment, args.trial, args.run_root, apply=args.apply)
    print(json.dumps(report), flush=True)
    if report.get("remaining"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
