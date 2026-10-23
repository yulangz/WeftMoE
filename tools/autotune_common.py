"""Shared compilation, selection and result I/O for offline tuning."""

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
