#!/usr/bin/env python3
"""Small target-independent helpers for MegaMOE kernel autotuners."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


class PlanError(ValueError):
    """An explicit tuning plan is incomplete or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def require_choice(config: Mapping[str, Any], key: str,
                   choices: Iterable[int]) -> None:
    choices = tuple(choices)
    require(config[key] in choices,
            f"{key}={config[key]!r}, expected one of {choices}")


def validate_selection(selection: Mapping[str, Any],
                       candidate_count: int) -> None:
    require(isinstance(selection, dict), "selection must be an object")
    expected = {
        "confirmation_candidates",
        "confirmation_repeats",
        "confirmation_rounds",
        "minimum_improvement_percent",
    }
    require(set(selection) == expected,
            "selection must contain exactly " + ", ".join(sorted(expected)))
    count = selection["confirmation_candidates"]
    repeats = selection["confirmation_repeats"]
    rounds = selection["confirmation_rounds"]
    threshold = selection["minimum_improvement_percent"]
    require(type(count) is int and 1 <= count <= candidate_count,
            "selection.confirmation_candidates must be in "
            f"[1, {candidate_count}]")
    require(type(rounds) is int and rounds > 10,
            "selection.confirmation_rounds must be an integer >10")
    require(type(repeats) is int and 1 <= repeats <= 4,
            "selection.confirmation_repeats must be in [1, 4]")
    require(type(threshold) in (int, float) and 0.0 <= threshold <= 10.0,
            "selection.minimum_improvement_percent must be in [0, 10]")


def candidate_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def compile_in_parallel(candidates: list[dict[str, Any]], compiler: Any,
                        log_dir: Path, compile_one: Any
                        ) -> dict[str, dict[str, Any]]:
    """Compile specializations concurrently within one global job budget."""

    workers = min(len(candidates), max(1, compiler.jobs // 2))
    command_jobs = max(1, compiler.jobs // workers)
    print(f"compile scheduling: candidates={len(candidates)} "
          f"workers={workers} commands_per_worker={command_jobs}", flush=True)
    results: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                compile_one, candidate, compiler, log_dir,
                command_jobs=command_jobs,
            ): candidate
            for candidate in candidates
        }
        for completed, future in enumerate(
                concurrent.futures.as_completed(futures), 1):
            candidate = futures[future]
            try:
                build_result = future.result()
            except Exception as error:
                build_result = {
                    "backend": "python-content-addressed-jit",
                    "returncode": 1,
                    "error": f"{type(error).__name__}: {error}",
                    "elapsed_seconds": 0.0,
                    "log": str(log_dir / f"{candidate['name']}.build.log"),
                }
            results[candidate["name"]] = build_result
            outcome = (
                "cache hit" if build_result.get("cache_hit") else
                "built" if build_result.get("returncode") == 0 else
                "failed"
            )
            print(f"[compile {completed}/{len(candidates)}] "
                  f"{candidate['name']}: {outcome}", flush=True)
    return results


def metric_value(record: Mapping[str, Any], dotted_name: str) -> float:
    value: Any = record
    for component in dotted_name.split("."):
        if not isinstance(value, dict) or component not in value:
            raise PlanError(f"metric {dotted_name} is absent from benchmark JSON")
        value = value[component]
    if not isinstance(value, (int, float)):
        raise PlanError(f"metric {dotted_name} is not numeric")
    return float(value)


def weighted_metric(records: Iterable[Mapping[str, Any]],
                    dotted_name: str) -> float:
    weighted_sum = 0.0
    measured_sum = 0
    for record in records:
        measured = record.get("measured")
        require(type(measured) is int and measured > 0,
                "benchmark record measured count must be a positive integer")
        weighted_sum += metric_value(record, dotted_name) * measured
        measured_sum += measured
    require(measured_sum > 0, "cannot aggregate an empty benchmark record set")
    return weighted_sum / measured_sum


def confirmation_finalists(valid: Iterable[dict[str, Any]],
                           incumbent_name: str,
                           count: int) -> list[dict[str, Any]]:
    ranked = sorted(valid, key=lambda item: item["selection_metric_us"])
    incumbent = next(
        (item for item in ranked if item["name"] == incumbent_name), None)
    finalists = [incumbent] if incumbent is not None else []
    finalists.extend(item for item in ranked if item is not incumbent)
    return finalists[:count]


def choose_confirmed_candidate(
        finalists: Iterable[dict[str, Any]], incumbent_name: str,
        minimum_improvement_percent: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    confirmed = [item for item in finalists
                 if "aggregate_metric_us" in item]
    require(bool(confirmed), "no candidate completed confirmation")
    best = min(confirmed, key=lambda item: item["aggregate_metric_us"])
    incumbent = next(
        (item for item in confirmed if item["name"] == incumbent_name), None)
    decision: dict[str, Any] = {
        "incumbent": incumbent_name,
        "best_confirmed": best["name"],
        "minimum_improvement_percent": minimum_improvement_percent,
    }
    if incumbent is None:
        decision.update({
            "selected": best["name"],
            "reason": "incumbent_not_confirmed",
            "challenger_improvement_percent": None,
        })
        return best, decision
    if best is incumbent:
        decision.update({
            "selected": incumbent["name"],
            "reason": "incumbent_fastest",
            "challenger_improvement_percent": 0.0,
        })
        return incumbent, decision
    improvement = 100.0 * (
        incumbent["aggregate_metric_us"] - best["aggregate_metric_us"]
    ) / incumbent["aggregate_metric_us"]
    if improvement >= minimum_improvement_percent:
        winner = best
        reason = "challenger_exceeds_threshold"
    else:
        winner = incumbent
        reason = "challenger_below_threshold"
    decision.update({
        "selected": winner["name"],
        "reason": reason,
        "challenger_improvement_percent": improvement,
    })
    return winner, decision


def git_revision(repo_root: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False).stdout.strip()
    return {"commit": revision, "dirty": bool(dirty)}


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def timing_summary(result: Mapping[str, Any], elapsed: float) -> dict[str, float]:
    candidates = result.get("candidates", [])
    selected = result.get("selected", {})
    phase_wall = result.get("phase_wall_seconds", {})
    return {
        "build_seconds": phase_wall.get("compile", sum(
            item.get("build", {}).get("elapsed_seconds", 0.0)
            for item in candidates)),
        "preflight_seconds": phase_wall.get("preflight", sum(
            item.get("preflight", {}).get("elapsed_seconds", 0.0)
            for item in candidates)),
        "tune_benchmark_seconds": phase_wall.get("tune", sum(
            item.get("tune", {}).get("elapsed_seconds", 0.0)
            for item in candidates)),
        "confirmation_benchmark_seconds": phase_wall.get(
            "confirmation", sum(
                sum(run.get("elapsed_seconds", 0.0)
                    for run in item.get("confirmations", []))
                for item in candidates)),
        "final_benchmark_seconds": phase_wall.get("final", selected.get(
            "final", {}).get("elapsed_seconds", 0.0)),
        "total_seconds": elapsed,
    }
