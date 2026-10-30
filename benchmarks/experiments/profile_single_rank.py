#!/usr/bin/env python3
"""Run training under torchrun, injecting NCU into one local rank only.

Diagnostic recipe validated on SM90 with NCU 2025.2.1. The fixed hardware
metrics below took ONE pass. Do not expand this to multi-pass kernel replay:
peer kernels and their communication state would not be replayed together.
Inspect the NCU log for `1 pass`; use CUPTI for undisturbed latency.

Set WEFTMOE_NCU_OUTPUT, WEFTMOE_NCU_KERNEL, and optionally WEFTMOE_NCU_RANK.
Pass training benchmark arguments, including --ncu-range --samples 1.
"""

import os
from pathlib import Path
import shutil
import sys


METRICS = (
    "gpu__time_duration.sum,"
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
    "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
    "smsp__issue_active.avg.pct_of_peak_sustained_active,"
    "smsp__warps_eligible.avg.per_cycle_active"
)


def main():
    rank = int(os.environ["LOCAL_RANK"])
    selected = int(os.environ.get("WEFTMOE_NCU_RANK", "0"))
    if not 0 <= selected < int(os.environ["LOCAL_WORLD_SIZE"]):
        raise ValueError("WEFTMOE_NCU_RANK must identify a local rank")
    if "--ncu-range" not in sys.argv[1:]:
        raise ValueError("Pass --ncu-range --samples 1 to mark the measured iteration")
    benchmark = Path(__file__).resolve().parents[1] / "torchrun_training.py"
    command = [sys.executable, str(benchmark), *sys.argv[1:]]
    if rank == selected:
        ncu = shutil.which("ncu")
        if ncu is None:
            raise RuntimeError("ncu is not available on PATH")
        command = [
            ncu, "--target-processes", "application-only",
            "--replay-mode", "kernel", "--profile-from-start", "off",
            "--clock-control", "none", "--cache-control", "none",
            "--kernel-name-base", "demangled",
            "--kernel-name", os.environ["WEFTMOE_NCU_KERNEL"],
            "--launch-count", "1", "--metrics", METRICS,
            "--export", os.environ["WEFTMOE_NCU_OUTPUT"], "--force-overwrite",
            *command,
        ]
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
