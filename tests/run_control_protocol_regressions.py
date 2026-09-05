#!/usr/bin/env python3
"""Run real two-GPU invalid-rank protocol regressions with a hard timeout."""

import argparse
import os
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    completed = subprocess.run(
        [args.binary, "--protocol-regressions"],
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.timeout,
        check=False,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    required = (
        "scenario=out_of_range",
        "scenario=duplicate",
        "scenario=max_mismatch",
        "scenario=over_capacity",
        "scenario=generation_mismatch",
        "scenario=session_mismatch",
        "scenario=mtile_mismatch",
        "scenario=two_failures",
        "scenario=multi_field_mismatch",
        "scenario=topology_mismatch",
        "scenario=expert_capacity_mismatch",
        "scenario=workspace_capacity_mismatch",
        "same ncclComm_t 5 categories x 4 invalid->valid recovery rounds",
    )
    missing = [marker for marker in required if marker not in completed.stdout]
    if missing:
        raise RuntimeError(f"protocol regressions omitted scenarios: {missing}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as error:
        print(
            f"[FAIL] protocol regression timed out after {error.timeout}s; "
            "a rank likely diverged at a collective boundary",
            file=sys.stderr,
        )
        raise SystemExit(124)
