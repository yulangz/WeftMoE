#!/usr/bin/env python3
"""Interleave frozen native before/after binaries and tuned Flex Megatron.

Two passes reverse implementation and case order. Each measurement has ten
warmups and independent per-rank maximum wall-time samples. Both native phases
use one combined binary. Pre-combine Forward receives an additional first-pass
measurement for both before and after. Run only after all builds have finished.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import run_performance_matrix as matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-builds", type=Path, required=True)
    parser.add_argument("--after-short", type=Path, required=True)
    parser.add_argument("--after-long", type=Path, required=True)
    parser.add_argument("--flex-selection", type=Path, required=True,
                        help="JSON mapping model_epN_tTOKENS to a selected SM count")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--remote", default="root@10.10.19.233")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--socket-ifname", default="bond0")
    args = parser.parse_args()
    if args.iterations <= 0 or args.passes <= 0:
        parser.error("iterations and passes must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    selection = json.loads(args.flex_selection.read_text())
    binaries = {}
    for name, source in (("before", args.before_builds),
                         ("after_short", args.after_short),
                         ("after_long", args.after_long)):
        destination = args.output / name
        destination.mkdir(exist_ok=True)
        for model in matrix.MODELS:
            for ep in (8, 16):
                for phase in ("forward", "backward"):
                    filename = f"{model}_ep{ep}_{phase}.build.json"
                    path = source / filename
                    if name == "after_long" and not path.exists():
                        path = args.after_short / filename
                    record = json.loads(path.read_text())
                    binary = Path(record["build"]["binary"])
                    binaries[str(binary)] = hashlib.sha256(binary.read_bytes()).hexdigest()
                    shutil.copy2(path, destination / filename)
    manifest = dict(schema=1, iterations=args.iterations, passes=args.passes,
                    warmups=10, tokens="per rank", binary_sha256=binaries,
                    rank_synchronization="untimed NCCL collective before every native/Flex iteration",
                    flex_selection=selection, execution_order=[])
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        for key in ("binary_sha256", "flex_selection", "iterations", "passes", "rank_synchronization"):
            if old.get(key) != manifest[key]:
                raise ValueError(f"cannot resume with changed {key}")
        manifest = old
    cases = [(m, e, t) for m in matrix.MODELS for e in (8, 16) for t in (8192, 65536)]
    for repetition in range(args.passes):
        for model, ep, tokens in (cases if repetition % 2 == 0 else reversed(cases)):
            case = f"{model}_ep{ep}_t{tokens}"
            order = ("before", "after", "flex") if repetition % 2 == 0 else ("flex", "after", "before")
            for implementation in order:
                profile = "short" if tokens == 8192 else "long"
                directory = implementation if implementation != "after" else f"after_{profile}"
                output = args.output / directory
                output.mkdir(exist_ok=True)
                call = SimpleNamespace(
                    output=output, source=matrix.ROOT, config=None,
                    remote=args.remote, port=args.port, socket_ifname=args.socket_ifname,
                    models=[model], eps=[ep], tokens=[tokens],
                    phases=["forward", "backward"], iterations=args.iterations,
                    label=f"pass{repetition + 1}_pc0", check_repeats=False,
                    pre_combine=False, dump_output=None, dump_tokens=8192,
                    flex_sms=int(selection[case]),
                    validation_repeats=2 if repetition == 0 else 0)
                entry = dict(case=case, implementation=implementation,
                             label=call.label, start=datetime.now(timezone.utc).isoformat())
                if any(old.get("complete") and all(old.get(key) == entry[key]
                                                  for key in ("case", "implementation", "label"))
                       for old in manifest["execution_order"]):
                    continue
                manifest["execution_order"].append(entry)
                matrix.save(manifest_path, manifest)
                if implementation == "flex":
                    if repetition == 0 and tokens == 8192:
                        call.dump_output = args.output / "flex_forward"
                    matrix.flex(call)
                else:
                    matrix.weft(call)
                    if repetition == 0:
                        call.pre_combine = True
                        call.phases = ["forward"]
                        call.label = "pass1_pc1"
                        matrix.weft(call)
                entry["complete"] = datetime.now(timezone.utc).isoformat()
                matrix.save(manifest_path, manifest)
    manifest["completed"] = True
    matrix.save(manifest_path, manifest)


if __name__ == "__main__":
    main()
