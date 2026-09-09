#!/usr/bin/env python3
"""Small target-independent helpers for MegaMOE kernel autotuners."""

from __future__ import annotations

import copy
import concurrent.futures
import hashlib
import json
import subprocess
import sys
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


def resolve_parallel_settings(system: Mapping[str, Any] | None = None) -> dict[str, int]:
    """Resolve topology from caller settings only, never tuning axes or host names."""
    settings = dict(system or {})
    gpus = settings.get("gpus_per_node", 8)
    nodes = settings.get("nodes")
    ep = settings.get("ep")
    for name, value in (("gpus_per_node", gpus), ("nodes", nodes), ("ep", ep)):
        require(value is None or (type(value) is int and value > 0),
                f"system.{name} must be a positive integer")
    if nodes is None:
        require(ep is None or ep % gpus == 0, "ep must be divisible by gpus_per_node")
        nodes = ep // gpus if ep is not None else 1
    require(ep is None or ep == nodes * gpus, "ep must equal nodes * gpus_per_node")
    require("world" not in settings or (nodes == 1 and settings["world"] == gpus),
            "system.world is a legacy local setting; use ep or nodes/gpus_per_node")
    return dict(nodes=nodes, gpus_per_node=gpus, ep=nodes * gpus)


def constrain_transport_space(space: Mapping[str, Any], topology: Mapping[str, Any]):
    """Copy before constraining axes, so caller-owned grids stay unchanged."""
    require(isinstance(space, Mapping), "hyperparameters must be an object")
    result = copy.deepcopy(space)
    transport = result.get("2_transport", {})
    require(isinstance(transport, dict), "2_transport must be an object")
    require("rdma_ctas" in transport, "hyperparameters omit rdma_ctas")
    if topology["nodes"] == 1:
        transport["rdma_ctas"] = [0]
    else:
        # Eight logical GIN owners are a kernel constraint, not the EP size.
        require(transport["rdma_ctas"] == [8],
                "multinode currently requires rdma_ctas=[8] (eight GIN owners)")
    return result


def validate_execution_topology(benchmark: Mapping[str, Any]) -> None:
    """Keep unsupported plans away from the current two-host GIN protocol."""
    if benchmark.get("launcher", "multinode") == "multinode":
        require(benchmark.get("nodes", 2) == 2 and benchmark["gpus_per_node"] == 8,
                "runtime currently supports two nodes with eight GPUs each; larger topology "
                "plans can be generated, but require generalizing the FWD/BWD GIN protocol "
                "and launcher before benchmarking")


def resolve_runtime_settings(*, model: Mapping[str, Any] | None,
                             system: Mapping[str, Any] | None, repo_root: Path):
    """Resolve caller-owned model/topology and optional runtime libraries."""
    require(isinstance(model, Mapping) and
             set(model) == {"hidden", "intermediate", "experts", "top_k"},
             "model must supply hidden, intermediate, experts and top_k")
    require(all(type(v) is int and v > 0 for v in model.values()),
             "model values must be positive integers")
    parallel = resolve_parallel_settings(system)
    local = parallel["nodes"] == 1
    settings = dict(system or {})
    allowed = {"ep", "nodes", "remote", "port", "gpus_per_node", "world", "nccl_lib_dir",
               "runtime_library_path", "socket_ifname", "disable_ib", "enable_ib", "jit"}
    require(not (set(settings) - allowed), "unknown system settings")
    jit_overrides = settings.pop("jit", {})
    require(isinstance(jit_overrides, dict), "system.jit must be an object")
    require(not local or ("remote" not in settings and "enable_ib" not in settings),
            "single-node topology cannot use remote/enable_ib; specify --nodes or --ep explicitly")
    require(local or "disable_ib" not in settings, "multinode uses enable_ib, not disable_ib")
    topology = {"launcher": "local", "world": parallel["ep"], "disable_ib": True} if local else {
        "launcher": "multinode", "port": 2222, "enable_ib": True}
    topology.update(settings)
    topology.update(parallel)
    world = parallel["ep"]
    require(model["experts"] % world == 0 and model["top_k"] <= model["experts"],
             "experts must divide evenly across ranks and be >= top_k")
    private_nccl = repo_root / "artifacts/gin_port_fix/build/lib"
    if "nccl_lib_dir" not in topology and (private_nccl / "libnccl.so").exists():
        topology["nccl_lib_dir"] = str(private_nccl)
    if "runtime_library_path" not in topology:
        paths = [Path("/usr/local/cuda/compat")]
        if topology.get("nccl_lib_dir"):
            paths.append(Path(topology["nccl_lib_dir"]))
        paths.append(Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" /
                     "site-packages/nvidia/cuda_runtime/lib")
        libraries = [str(p) for p in paths if p.is_dir()]
        if libraries:
            topology["runtime_library_path"] = ":".join(libraries)
    return topology, jit_overrides
