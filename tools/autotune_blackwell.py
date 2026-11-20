"""Measure Blackwell tile candidates and publish an offline capacity-bucket policy."""

import json
import os
from pathlib import Path
from statistics import geometric_mean, mean

from autotune_common import (
    choose_confirmed_candidate,
    confirmation_finalists,
    require,
    write_result,
)
from weftmoe.blackwell_tuning import STRICT_BLOCK_M, SUPPORTED_BLOCK_M
from weftmoe.tuning import publish_entry

ROOT = Path(__file__).resolve().parents[1]


def run_blackwell_search(args, spec, launch_workers):
    strict = args.batch_invariant
    baseline = f"full-{STRICT_BLOCK_M}" if strict else "dynamic"
    candidates = (
        [baseline] if strict else [baseline, *(f"fixed-{m}" for m in SUPPORTED_BLOCK_M)]
    )
    require(
        args.gpus_per_node > 0
        and args.experts % ((1 + len(args.remotes)) * args.gpus_per_node) == 0,
        "experts must divide the EP group",
    )
    require(
        not args.remotes or args.master_addr != "127.0.0.1",
        "multi-node requires --master-addr",
    )
    require(
        spec["confirmation_candidates"] > 0 and spec["confirmation_repeats"] > 0,
        "confirmation counts must be positive",
    )
    plan = dict(
        arch="sm100",
        mode=args.mode,
        precision=args.precision,
        tokens=args.token_interval,
        representatives=args.tokens,
        batch_invariant=strict,
        candidates=candidates,
    )
    write_result(args.output / "plan.json", plan)
    if args.dry_run:
        print(args.output / "plan.json")
        return

    script = ROOT / "benchmarks/sm100/torchrun_tile_m.py"
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("CUDA_", "NCCL_", "UCX_", "DG_"))
        or k in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PATH", "TORCH_EXTENSIONS_DIR")
    }
    env.update(spec.get("environment", {}))
    env.update(
        PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), os.getenv("PYTHONPATH")))),
        OMP_NUM_THREADS="1",
    )

    def measure(stage, names, settings, seed):
        cli = [
            "--mode",
            args.mode,
            "--precision",
            args.precision,
            "--model",
            args.model,
            "--hidden",
            str(args.hidden),
            "--intermediate",
            str(args.intermediate),
            "--experts",
            str(args.experts),
            "--top-k",
            str(args.top_k),
            "--tokens",
            *map(str, args.tokens),
            "--canonical-reference",
            "--route-bias-std",
            str(spec["route_bias_std"]),
            "--seed",
            str(seed),
            "--samples",
            str(settings["samples"]),
            "--warmups",
            str(settings["warmups"]),
            "--replays",
            str(spec.get("graph_replays", 20)),
            "--fixed-m",
            *(n.split("-")[1] for n in names if n.startswith("fixed-")),
            "--full-m",
        ]
        if strict:
            cli.extend([str(STRICT_BLOCK_M), "--strict-only"])
        logs = launch_workers(
            args, args.output / stage, script, cli, env, spec["timeout_seconds"]
        )
        records = [
            json.loads(line.removeprefix("TILE_M_JSON "))
            for line in logs[0].read_text().splitlines()
            if line.startswith("TILE_M_JSON ")
        ]
        expected = {(n, name) for name in names for n in args.tokens}
        require(
            len(records) == len(expected)
            and {(r["tokens"], r["policy"]) for r in records} == expected,
            f"incomplete benchmark: {logs[0]}",
        )
        scores = {}
        for name in names:
            rows = [r for r in records if r["policy"] == name]
            if all(
                r["mismatched_elements"] == 0 and 0 < r["p50_us"] < float("inf")
                for r in rows
            ):
                scores[name] = geometric_mean(r["p50_us"] for r in rows)
        require(
            baseline in scores, f"baseline failed numerical or timing checks: {logs[0]}"
        )
        return scores, records

    scores, screen = measure("screen", candidates, spec["screen"], spec["seed"])
    finalists = confirmation_finalists(
        [dict(name=name, selection_metric_us=us) for name, us in scores.items()],
        baseline,
        spec["confirmation_candidates"],
    )
    confirmations = []
    for repeat in range(spec["confirmation_repeats"]):
        names = [item["name"] for item in finalists]
        confirmation, rows = measure(
            f"confirm{repeat}", names, spec["final"], spec["seed"] + repeat + 1
        )
        confirmations.extend(rows)
        for item in finalists:
            if item["name"] in confirmation:
                item.setdefault("metrics", []).append(confirmation[item["name"]])
    for item in finalists:
        if len(item.get("metrics", [])) == spec["confirmation_repeats"]:
            item["aggregate_metric_us"] = mean(item["metrics"])
    selected, decision = choose_confirmed_candidate(
        finalists, baseline, spec["minimum_improvement_percent"]
    )
    winner = selected["name"]
    # Confirmation uses fresh traces; only publish when all launches used the same kernel build.
    require(
        len({r["source"] for r in [*screen, *confirmations]}) == 1,
        "Blackwell kernel sources changed during tuning",
    )
    record = next(r for r in confirmations if r["policy"] == winner)
    entry = dict(
        key=record["profile_key"],
        tokens=args.token_interval,
        source=record["source"],
        validated=True,
        batch_invariant=strict,
        fixed_block_m=0 if winner == "dynamic" else record["block_m"],
    )
    write_result(
        args.output / "result.json",
        dict(
            plan=plan,
            screen=screen,
            confirmations=confirmations,
            selected=winner,
            decision=decision,
        ),
    )
    write_result(args.output / "table.json", dict(schema=2, entries=[entry]))
    publish_entry(args.table, entry)
    write_result(
        args.output / "runtime.json",
        dict(
            environment={"WEFTMOE_AUTOTUNE_TABLE": str(args.table.resolve())},
            tokens=args.token_interval,
            batch_invariant=strict,
        ),
    )
    print(f"Selected {winner} for {args.token_interval}: {args.table}", flush=True)
