#!/usr/bin/env python3
"""JIT-compile and tune explicit WeftMoE Forward specializations.

The search plan is JSON rather than a collection of shell environment
variables.  Python generates each specialization, compiles it into an atomic
content-addressed cache without Make, and benchmarks the cached executable.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from autotune_common import (
    DEFAULT_COMPILE_JOBS, DEFAULT_TUNE_ROUNDS, DEFAULT_FINAL_ROUNDS,
    configure_preparation, environment_autotune_config,
    resolve_runtime_settings,
    resolve_parallel_settings,
    constrain_transport_space,
    validate_execution_topology,
    PlanError,
    candidate_digest,
    choose_confirmed_candidate,
    compile_in_parallel,
    confirmation_finalists,
    git_revision as _common_git_revision,
    metric_value,
    require as _require,
    require_choice as _require_choice,
    timing_summary as _common_timing_summary,
    validate_selection,
    weighted_metric,
    write_result as _common_write_result,
)
from forward_jit import ForwardJitCompiler, JitError, validate_jit_config


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = (
    "1_gemm",
    "2_transport",
    "3_scheduler",
    "4_completion_reduce",
)

# This is deliberately exhaustive.  Adding a C++ tuning macro without adding
# it here makes plan validation fail instead of silently falling back to an
# unreported binary default.
PARAMETER_MACROS = {
    "main_hidden": "WEFTMOE_FWD_MAIN_HIDDEN",
    "main_intermediate": "WEFTMOE_FWD_MAIN_INTERMEDIATE",
    "tile_m": "WEFTMOE_FWD_TILE_M",
    "fc1_tile_n": "WEFTMOE_FWD_FC1_TILE_N",
    "fc2_tile_n": "WEFTMOE_FWD_FC2_TILE_N",
    "tile_k": "WEFTMOE_FWD_TILE_K",
    "fc1_stages": "WEFTMOE_FWD_FC1_STAGES",
    "fc2_stages": "WEFTMOE_FWD_FC2_STAGES",
    "compute_threads": "WEFTMOE_FWD_COMPUTE_THREADS",
    "math_warpgroups": "WEFTMOE_FWD_MATH_WARPGROUPS",
    "producer_warpgroups": "WEFTMOE_FWD_PRODUCER_WARPGROUPS",
    "math_registers": "WEFTMOE_FWD_MATH_REGISTERS",
    "producer_registers": "WEFTMOE_FWD_PRODUCER_REGISTERS",
    "wgmma_wait_group": "WEFTMOE_FWD_WGMMA_WAIT_GROUP",
    "fc2_store_atom_n": "WEFTMOE_FWD_FC2_STORE_ATOM_N",
    "cluster_size": "WEFTMOE_FWD_CLUSTER_SIZE",
    "fc2_tma_store": "WEFTMOE_FWD_FC2_TMA_STORE",
    "rdma_ctas": "WEFTMOE_FWD_RDMA_CTAS",
    "relay_ctas": "WEFTMOE_FWD_RELAY_CTAS",
    "relay_tma_warps": "WEFTMOE_FWD_RELAY_TMA_WARPS",
    "local_dispatch_tma_warps": "WEFTMOE_FWD_LOCAL_DISPATCH_TMA_WARPS",
    "relay_scatter_warps": "WEFTMOE_FWD_RELAY_SCATTER_WARPS",
    "local_return_batch_rows":
        "WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS",
    "dispatch_publish_batch_rows":
        "WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS",
    "relay_local_return": "WEFTMOE_FWD_RELAY_LOCAL_RETURN",
    "local_route_batch_rows": "WEFTMOE_FWD_LOCAL_ROUTE_BATCH_ROWS",
    "startup_mblocks": "WEFTMOE_FWD_STARTUP_MBLOCKS",
    "fc1_locality_group": "WEFTMOE_FWD_FC1_LOCALITY_GROUP",
    "fc1_locality_group4_begin":
        "WEFTMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN",
    "fc1_locality_group4_end": "WEFTMOE_FWD_FC1_LOCALITY_GROUP4_END",
    "fc2_priority_burst": "WEFTMOE_FWD_FC2_PRIORITY_BURST",
    "task_claim_chunk": "WEFTMOE_FWD_TASK_CLAIM_CHUNK",
    "task_idle_nanosleep": "WEFTMOE_FWD_TASK_IDLE_NANOSLEEP",
    "transport_idle_nanosleep": "WEFTMOE_FWD_TRANSPORT_IDLE_NANOSLEEP",
    "setup_nanosleep": "WEFTMOE_FWD_SETUP_NANOSLEEP",
    "completion_peer_probes":
        "WEFTMOE_FWD_COMPLETION_PEER_PROBES",
    "reduce_scheduler": "WEFTMOE_FWD_REDUCE_SCHEDULER",
    "reduce_warps": "WEFTMOE_FWD_REDUCE_WARPS",
    "reduce_tokens_per_claim": "WEFTMOE_FWD_REDUCE_TOKENS_PER_CLAIM",
    "reduce_registers": "WEFTMOE_FWD_REDUCE_REGISTERS",
    "return_idle_nanosleep": "WEFTMOE_FWD_RETURN_IDLE_NANOSLEEP",
    "reduce_queue_nanosleep": "WEFTMOE_FWD_REDUCE_QUEUE_NANOSLEEP",
    "finalize_nanosleep": "WEFTMOE_FWD_FINALIZE_NANOSLEEP",
}

# Preserve internal keys/macros for running autotunes and cached candidates.
PARAMETER_ALIASES = {"local_return_tma_warps": "relay_tma_warps"}

BOOLEAN_PARAMETERS = {
    "fc2_tma_store",
    "relay_local_return",
}

ENUM_PARAMETERS = {
    "reduce_scheduler": {
        "queue": 0,
        "scan": 1,
        "static": 2,
    },
}

CORRECTNESS_KEYS = (
    "shape",
    "per_rank_activation_hash",
    "per_rank_route_hash",
    "per_rank_gate_hash",
    "per_rank_actual_w1_device_hash",
    "per_rank_actual_w2_device_hash",
    "per_rank_output_hash",
)


def flatten_groups(groups: Mapping[str, Any], *, require_all: bool) -> dict[str, Any]:
    _require(isinstance(groups, dict), "tuning groups must be a JSON object")
    unknown_groups = set(groups) - set(GROUP_NAMES)
    _require(not unknown_groups,
             f"unknown tuning groups: {sorted(unknown_groups)}")
    if require_all:
        missing_groups = set(GROUP_NAMES) - set(groups)
        _require(not missing_groups,
                 f"missing tuning groups: {sorted(missing_groups)}")
    flat: dict[str, Any] = {}
    for group_name in GROUP_NAMES:
        group = groups.get(group_name, {})
        _require(isinstance(group, dict), f"{group_name} must be an object")
        for key, value in group.items():
            key = PARAMETER_ALIASES.get(key, key)
            _require(key in PARAMETER_MACROS,
                     f"unknown Forward parameter {group_name}.{key}")
            _require(key not in flat,
                     f"Forward parameter {key} appears in more than one group")
            flat[key] = value
    if require_all:
        missing = set(PARAMETER_MACROS) - set(flat)
        extra = set(flat) - set(PARAMETER_MACROS)
        _require(not missing, f"fixed config omits parameters: {sorted(missing)}")
        _require(not extra, f"fixed config has extra parameters: {sorted(extra)}")
    return flat


def validate_tuning(config: Mapping[str, Any]) -> None:
    _require(set(config) == set(PARAMETER_MACROS),
             "expanded tuning config must define every parameter exactly once")
    for key, value in config.items():
        if key in BOOLEAN_PARAMETERS:
            _require(type(value) is bool, f"{key} must be true or false")
        elif key in ENUM_PARAMETERS:
            _require(value in ENUM_PARAMETERS[key],
                     f"{key}={value!r}, expected one of "
                     f"{tuple(ENUM_PARAMETERS[key])}")
        else:
            _require(type(value) is int, f"{key} must be an integer")
            _require(value >= 0, f"{key} must be non-negative")

    _require(config["main_hidden"] > 0 and config["main_hidden"] % 256 == 0,
             "main_hidden must be a positive multiple of 256")
    _require(config["main_intermediate"] > 0 and config["main_intermediate"] % 256 == 0,
             "main_intermediate must be a positive multiple of 256")
    _require(config["tile_m"] == 128,
             "cluster multicast requires tile_m=128")
    _require_choice(config, "fc1_tile_n", (64, 128, 256))
    _require_choice(config, "fc2_tile_n", (64, 128, 256))
    _require(config["tile_k"] == 64,
             "the Hopper 128-byte-swizzle TMA descriptor requires tile_k=64")
    _require_choice(config, "fc1_stages", (2, 3, 4))
    _require_choice(config, "fc2_stages", (2, 3, 4))
    _require(config["compute_threads"] == 384,
             "the current producer + two-math-WG kernel requires 384 threads")
    _require(config["math_warpgroups"] == 2,
             "the current WGMMA mapping requires two math warpgroups")
    _require(config["producer_warpgroups"] == 1,
             "the current TMA producer requires one producer warpgroup")
    for key in ("math_registers", "producer_registers", "reduce_registers"):
        _require(1 <= config[key] <= 255, f"{key} must be in [1, 255]")
    compute_registers = 128 * (
        config["math_warpgroups"] * config["math_registers"] +
        config["producer_warpgroups"] * config["producer_registers"])
    _require(compute_registers <= 65536,
             f"compute register allocation {compute_registers} exceeds 65536")
    reduce_registers = config["compute_threads"] * config["reduce_registers"]
    _require(reduce_registers <= 65536,
             f"reduce register allocation {reduce_registers} exceeds 65536")
    _require_choice(config, "wgmma_wait_group", (0, 1))
    _require(config["fc2_store_atom_n"] == 64,
             "the current STSM/TMA epilogue store atom is N64")
    _require(config["cluster_size"] == 2,
             "Forward requires cluster_size=2 for TMA multicast")
    _require(config["fc2_tma_store"],
             "Forward requires the FC2 TMA-store epilogue")
    _require((config["main_intermediate"] // config["tile_k"]) %
             config["fc2_stages"] == 0,
             "FC2 reduction iterations must be divisible by fc2_stages")
    _require((config["main_intermediate"] // config["fc1_tile_n"]) % 2 == 0
             and (config["main_hidden"] // config["fc2_tile_n"]) % 2 == 0,
             "cluster multicast requires even FC1 and FC2 N-tile counts")

    # Two-host GIN owns eight physical contexts. One-node NVLink has no RDMA
    # owner and tunes Relay independently in small four-CTA increments.
    _require_choice(config, "rdma_ctas", (0, 8))
    if config["rdma_ctas"] == 0:
        _require(config["relay_ctas"] >= 4 and config["relay_ctas"] % 4 == 0,
                 "relay_ctas must be a positive multiple of 4")
        _require(config["local_dispatch_tma_warps"] < 512 // 32,
                 "local split Relay reserves one FC1-ready publisher warp")
    else:
        _require(config["relay_ctas"] >= config["rdma_ctas"],
                 "relay_ctas must cover every RDMA context")
        _require(config["relay_ctas"] % config["rdma_ctas"] == 0,
                 "relay_ctas must be a multiple of rdma_ctas")
    _require(config["relay_local_return"], "local Return requires relay_local_return=true")
    _require_choice(config, "relay_tma_warps", (0, 4, 8, 12))
    _require_choice(config, "local_dispatch_tma_warps", (4, 8, 12))
    _require_choice(config, "relay_scatter_warps", (4, 8, 12))
    _require_choice(config, "local_return_batch_rows", (32, 64, 128, 256, 512, 1024))
    _require_choice(config, "dispatch_publish_batch_rows",
                    (4, 8, 16, 32, 64, 128))

    _require_choice(config, "local_route_batch_rows", (32, 64, 128, 256))
    _require_choice(config, "startup_mblocks", (1, 2, 4))
    _require_choice(config, "fc1_locality_group", (1, 2, 4))
    begin = config["fc1_locality_group4_begin"]
    end = config["fc1_locality_group4_end"]
    _require(begin >= 2 and begin % 2 == 0 and end >= begin and end <= 255 and
             (end - begin) % 4 == 0,
             "FC1 locality window must be even, ordered, <=255, and span % 4 == 0")
    _require_choice(config, "fc2_priority_burst", (0, 2, 4, 8))
    _require(config["task_claim_chunk"] == 1,
             "only the proven one-task work-queue claim is implemented")

    _require_choice(config, "completion_peer_probes", (1, 2, 4))
    _require_choice(config, "reduce_warps", (4, 8, 12))
    _require(1 <= config["reduce_tokens_per_claim"] <= config["reduce_warps"],
             "reduce_tokens_per_claim must be in [1, reduce_warps]")
    for key in (
        "task_idle_nanosleep",
        "transport_idle_nanosleep",
        "setup_nanosleep",
        "return_idle_nanosleep",
        "reduce_queue_nanosleep",
        "finalize_nanosleep",
    ):
        _require_choice(config, key, (0, 16, 32, 64, 128, 256, 512))


def expand_search_space(document: dict[str, Any]) -> None:
    """Normalize a complete schema-3 Cartesian grid to explicit candidates."""
    _require("fixed" not in document and "candidates" not in document,
             "schema 3 uses search_space, not fixed/candidates")
    _require("search_space" in document, "plan omits search_space")
    model = document.get("model")
    model_keys = {"main_hidden", "main_intermediate"}
    _require(isinstance(model, dict) and set(model) == model_keys,
             "schema 3 model must define main_hidden and main_intermediate")
    for key, value in model.items():
        _require(type(value) is int and value > 0 and value % 256 == 0,
                 f"model.{key} must be a positive multiple of 256")
    space = copy.deepcopy(document["search_space"])
    provided = flatten_groups(space, require_all=False)
    # Normalize without reordering axes so candidate names and overrides stay
    # identical when a configuration adopts the new public spelling.
    space = {group: {PARAMETER_ALIASES.get(key, key): value
                     for key, value in values.items()}
             for group, values in space.items()}
    _require(not (model_keys & set(provided)),
             "model dimensions belong in model, not search_space")
    _require("1_gemm" in space, "search_space omits 1_gemm")
    space["1_gemm"].update({key: [value] for key, value in model.items()})
    flat = flatten_groups(space, require_all=True)
    count = 1
    for key, values in flat.items():
        _require(isinstance(values, list) and values,
                 f"search_space.{key} must be a non-empty array")
        encoded = [json.dumps(v, sort_keys=True) for v in values]
        _require(len(set(encoded)) == len(values),
                 f"search_space.{key} has duplicate values")
        count *= len(values)
    maximum = document.get("max_candidates")
    _require(type(maximum) is int and 1 <= maximum <= 4096,
             "schema 3 max_candidates must be in [1, 4096]")
    _require(count <= maximum,
             f"Cartesian candidate count {count} exceeds max_candidates={maximum}")
    fixed = {group: {key: values[0] for key, values in space[group].items()}
             for group in GROUP_NAMES}
    axes = [(group, key, values) for group in GROUP_NAMES
            for key, values in space[group].items() if len(values) > 1]
    candidates = []
    for index, values in enumerate(itertools.product(*(a[2] for a in axes))):
        override = {}
        for (group, key, _), value in zip(axes, values):
            if value != fixed[group][key]:
                override.setdefault(group, {})[key] = value
        candidates.append({"name": "incumbent" if index == 0 else f"grid_{index:04d}",
                           "override": override})
    document["fixed"] = fixed
    document["candidates"] = candidates


def make_runtime_plan(space: dict[str, Any], *, model: Mapping[str, Any] | None,
                      benchmark: Mapping[str, Any] | None = None,
                      system: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Combine hyperparameters, caller-owned workload and optional host settings."""
    topology, jit_overrides = resolve_runtime_settings(
        model=model, system=system, repo_root=REPO_ROOT)
    space = constrain_transport_space(space, topology)
    flat = flatten_groups(space, require_all=False)
    workload = dict(benchmark or {})
    _require(not (set(workload) - {"tokens_per_rank", "seed", "generation_base",
        "tune_rounds", "final_rounds", "timeout_seconds", "result_dir", "pre_combine"}),
        "benchmark contains model/system fields or unknown settings")
    count = 1
    for values in flat.values():
        _require(isinstance(values, list) and values,
                 "hyperparameters must be non-empty arrays")
        count *= len(values)
    return {
        "schema": 3, "search_space": copy.deepcopy(space), "max_candidates": 4096,
        "model": {"main_hidden": model["hidden"], "main_intermediate": model["intermediate"]},
        "jit": dict(cache_root="artifacts/forward-jit", jobs=DEFAULT_COMPILE_JOBS, architecture="sm_90a",
                    timeout_seconds=1800) | jit_overrides,
        "selection": dict(confirmation_candidates=min(2, count), confirmation_repeats=2,
                          confirmation_rounds=DEFAULT_FINAL_ROUNDS, minimum_improvement_percent=0.5),
        "benchmark": dict(seed=0x4D554C54, tokens_per_rank=8192, top_k=model["top_k"],
            experts=model["experts"], gate_mode="fixed-random", generation_base=1000,
            tune_rounds=DEFAULT_TUNE_ROUNDS, final_rounds=DEFAULT_FINAL_ROUNDS, timeout_seconds=600,
            selection_metric="forward_slowest_rank_us.mean",
            result_dir="artifacts/forward-autotune") | topology | workload,
    }


def load_plan(path: Path, *, model: Mapping[str, Any] | None = None,
              benchmark: Mapping[str, Any] | None = None,
              system: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load {path}: {error}") from error
    _require(isinstance(document, dict), "top-level plan must be an object")
    if "forward" in document:
        document = document["forward"]
        _require(isinstance(document, dict), "phase search space must be an object")
    if "schema" in document and any(k in (system or {}) for k in ("ep", "nodes", "gpus_per_node")):
        requested = resolve_parallel_settings(system)
        frozen = document.get("benchmark", {})
        actual_nodes = frozen.get("nodes", 1 if frozen.get("launcher") == "local" else 2)
        actual_gpus = frozen.get("gpus_per_node", frozen.get("world", 8))
        _require(requested["nodes"] == actual_nodes and requested["gpus_per_node"] == actual_gpus,
                "explicit parallel settings conflict with frozen plan topology")
    if "schema" not in document:
        document = make_runtime_plan(document, model=model, benchmark=benchmark, system=system)
    _require(document.get("schema") in (2, 3), "plan schema must be 2 or 3")
    if document["schema"] == 2:
        # Historical frozen plans retain this removed tail-only parameter.
        # New pure hyperparameter files reject it as unknown.
        groups = [document.get("fixed", {})] + [c.get("override", {}) for c in document.get("candidates", [])]
        for group in groups:
            group.get("4_completion_reduce", {}).pop("reduce_cta_divisor", None)
    if document["schema"] == 3:
        expand_search_space(document)
    for section in ("jit", "benchmark", "selection", "fixed", "candidates"):
        _require(section in document, f"plan omits {section}")
    _require(isinstance(document["jit"], dict), "jit must be an object")
    try:
        validate_jit_config(document["jit"])
    except JitError as error:
        raise PlanError(str(error)) from error
    _require(isinstance(document["benchmark"], dict),
             "benchmark must be an object")
    benchmark = document["benchmark"]
    launcher = benchmark.get("launcher", "multinode")
    _require(type(benchmark.get("pre_combine", False)) is bool,
             "benchmark.pre_combine must be boolean")
    _require(launcher in ("multinode", "local"),
             "benchmark.launcher must be multinode or local")
    common_benchmark_keys = (
        "seed",
        "tokens_per_rank",
        "top_k",
        "generation_base",
        "tune_rounds",
        "final_rounds",
        "timeout_seconds",
        "selection_metric",
        "result_dir",
    )
    for key in common_benchmark_keys:
        _require(key in benchmark, f"benchmark omits {key}")
    if launcher == "multinode":
        for key in ("port", "gpus_per_node"):
            _require(key in benchmark, f"benchmark omits {key}")
    else:
        _require("world" in benchmark, "local benchmark omits world")
    for key in (
        "seed",
        "tokens_per_rank",
        "top_k",
        "generation_base",
        "tune_rounds",
        "final_rounds",
        "timeout_seconds",
    ):
        _require(type(benchmark[key]) is int and benchmark[key] > 0,
                 f"benchmark.{key} must be a positive integer")
    topology_integer_keys = (("port", "gpus_per_node")
                             if launcher == "multinode" else ("world",))
    for key in topology_integer_keys:
        _require(type(benchmark[key]) is int and benchmark[key] > 0,
                 f"benchmark.{key} must be a positive integer")
    if launcher == "local":
        _require(benchmark["world"] in (1, 2, 8),
                 "local benchmark.world must be 1, 2, or 8")
    _require(benchmark["top_k"] in (1, 6, 8),
             "benchmark.top_k must be 1, 6, or 8")
    _require(benchmark["tune_rounds"] > 10 and
             benchmark["final_rounds"] > 10,
             "Forward benchmark rounds must be >10")
    _require(isinstance(document["candidates"], list) and
             document["candidates"], "candidates must be a non-empty array")
    maximum = document.get("max_candidates", 8)
    limit = 4096 if document["schema"] == 3 else 16
    _require(type(maximum) is int and 1 <= maximum <= limit,
             f"max_candidates must be in [1, {limit}]")
    _require(len(document["candidates"]) <= maximum,
             f"candidate count exceeds max_candidates={maximum}")
    validate_selection(document["selection"], len(document["candidates"]))
    flatten_groups(document["fixed"], require_all=True)
    return document


def expand_candidates(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixed = flatten_groups(plan["fixed"], require_all=True)
    expanded = []
    names: set[str] = set()
    fingerprints: set[str] = set()
    for candidate in plan["candidates"]:
        _require(isinstance(candidate, dict), "each candidate must be an object")
        name = candidate.get("name")
        _require(isinstance(name, str) and
                 re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is not None,
                 f"invalid candidate name {name!r}")
        _require(name not in names, f"duplicate candidate name {name}")
        names.add(name)
        override = flatten_groups(candidate.get("override", {}),
                                  require_all=False)
        values = dict(fixed)
        values.update(override)
        validate_tuning(values)
        fingerprint = json.dumps(values, sort_keys=True, separators=(",", ":"))
        _require(fingerprint not in fingerprints,
                 f"candidate {name} duplicates another expanded config")
        fingerprints.add(fingerprint)
        expanded.append({
            "name": name,
            "description": candidate.get("description", ""),
            "override": copy.deepcopy(candidate.get("override", {})),
            "config": values,
        })
    return expanded


def generated_header(_name: str, config: Mapping[str, Any]) -> str:
    validate_tuning(config)
    digest = candidate_digest(config)
    lines = [
        "#pragma once",
        "",
        "// Generated by tools/autotune_forward.py.",
        f"// config_sha256_prefix={digest}",
    ]
    for key, macro in PARAMETER_MACROS.items():
        value = config[key]
        if key in BOOLEAN_PARAMETERS:
            value = int(value)
        elif key in ENUM_PARAMETERS:
            value = ENUM_PARAMETERS[key][value]
        lines.append(f"#define {macro} {value}")
    lines.append("")
    return "\n".join(lines)


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def materialize_candidate(candidate: Mapping[str, Any],
                          build_root: Path) -> dict[str, Path]:
    digest = candidate_digest(candidate["config"])
    build_dir = build_root / f"{candidate['name']}-{digest}"
    include_dir = build_dir / "generated"
    include_dir.mkdir(parents=True, exist_ok=True)
    header = include_dir / "weftmoe_forward_tuning_generated.hpp"
    content = generated_header(candidate["name"], candidate["config"])
    if not header.exists() or header.read_text(encoding="utf-8") != content:
        header.write_text(content, encoding="utf-8")
    return {
        "build_dir": build_dir,
        "include_dir": include_dir,
        "header": header,
    }


def dynamic_shared_memory(config: Mapping[str, Any], world: int) -> dict[str, int]:
    a_stage = config["tile_m"] * config["tile_k"]
    fc1_b_stage = config["fc1_tile_n"] * config["tile_k"]
    fc2_b_stage = config["fc2_tile_n"] * config["tile_k"]
    a_storage = max(config["fc1_stages"], config["fc2_stages"]) * a_stage
    b_storage = max(2 * config["fc1_stages"] * fc1_b_stage,
                    config["fc2_stages"] * fc2_b_stage)
    barriers = 3 * config["fc1_stages"] + 2 * config["fc2_stages"]
    gemm_bytes = 2 * (a_storage + b_storage)
    control_bytes = barriers * 8 + world * 4
    row_streams = max(config["local_dispatch_tma_warps"],
                      config["relay_tma_warps"])
    transport_rows = row_streams * config["main_hidden"] * 2
    return {
        "compute_bytes": gemm_bytes + control_bytes,
        "transport_bytes": max(gemm_bytes, transport_rows) + control_bytes,
    }


def inspect_forward_kernel(binary: Path, log_path: Path,
                           config: Mapping[str, Any]) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump not found"}
    completed = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(binary)],
        cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8")
    functions = []
    lines = completed.stdout.splitlines()
    kernel_names = (
        "local_p2p_forward_tma_store_kernel",
        "local_p2p_forward_compute_kernel",
    )
    resource_pattern = re.compile(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
    for index, line in enumerate(lines[:-1]):
        if not line.startswith(" Function ") or not any(
                name in line for name in kernel_names):
            continue
        resources = resource_pattern.search(lines[index + 1])
        if resources is None:
            continue
        functions.append({
            "mangled_name": line[len(" Function "):-1],
            "registers_per_thread": int(resources.group(1)),
            "stack_bytes_per_thread": int(resources.group(2)),
            "static_shared_bytes": int(resources.group(3)),
            "local_bytes": int(resources.group(4)),
        })
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "single_compiled_split_kernel": len(functions) == 1,
        "compiled_split_kernel_count": len(functions),
        "kernels": functions,
        "dynamic_shared_memory_world16": dynamic_shared_memory(config, 16),
        "log": str(log_path),
    }


def _sanitized_environment() -> dict[str, str]:
    environment = os.environ.copy()
    explicit_removals = {
        "NCCL_IB_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "WEFTMOE_INTEGRATED_TOP_K",
        "WEFTMOE_P2P_TOP_K",
        "WEFTMOE_P2P_BENCH_TOKENS",
        "WEFTMOE_FORWARD_BENCH_ROUNDS",
        "WEFTMOE_FORWARD_BENCH_GENERATION_BASE",
    }
    for key in tuple(environment):
        if (key.startswith("WEFTMOE_FORWARD_") or
                key.startswith("WEFTMOE_FWD_") or key in explicit_removals):
            environment.pop(key, None)
    return environment


def compile_candidate(candidate: Mapping[str, Any],
                      compiler: ForwardJitCompiler,
                      log_dir: Path, *,
                      command_jobs: int | None = None) -> dict[str, Any]:
    log_path = log_dir / f"{candidate['name']}.build.log"
    result = compiler.build(
        candidate_name=str(candidate["name"]),
        config_digest=candidate_digest(candidate["config"]),
        generated_header=generated_header(
            str(candidate["name"]), candidate["config"]),
        log_path=log_path,
        command_jobs=command_jobs,
    )
    if result["returncode"] == 0:
        resources = inspect_forward_kernel(
            Path(result["binary"]),
            log_dir / f"{candidate['name']}.resources.log",
            candidate["config"])
        result["resources"] = resources
        if (resources.get("available") and
                not resources.get("single_compiled_split_kernel")):
            result["returncode"] = 125
            result["resource_validation_error"] = (
                "expected exactly one compiled split Forward kernel")
    return result


def compile_candidates(candidates: list[dict[str, Any]],
                       compiler: ForwardJitCompiler,
                       log_dir: Path) -> dict[str, dict[str, Any]]:
    return compile_in_parallel(
        candidates, compiler, log_dir, compile_candidate
    )


def launcher_command(binary: Path, benchmark: Mapping[str, Any],
                     *, rounds: int) -> list[str]:
    if benchmark.get("launcher", "multinode") == "local":
        return [
            sys.executable,
            str(REPO_ROOT / "tests" / "run_local_activation_p2p.py"),
            "--binary", str(binary),
            "--world", str(int(benchmark["world"])),
            "--seed", str(int(benchmark["seed"])),
            "--timeout", str(int(benchmark["timeout_seconds"])),
            "--rounds", str(rounds),
            "--forward-benchmark",
            "--top-k", str(int(benchmark["top_k"])),
            "--gate-mode", str(benchmark.get("gate_mode", "fixed-random")),
        ]
    validate_execution_topology(benchmark)
    command = [
        sys.executable,
        str(REPO_ROOT / "tests" / "run_multinode_nccl.py"),
        "--binary", str(binary),
        "--remote", str(benchmark["remote"]),
        "--port", str(int(benchmark["port"])),
        "--remote-binary", str(binary),
        "--gpus-per-node", str(int(benchmark["gpus_per_node"])),
        "--seed", str(int(benchmark["seed"])),
        "--scenario", "gin_forward_bench",
        "--timeout", str(int(benchmark["timeout_seconds"])),
    ]
    nccl_lib_dir = benchmark.get("runtime_library_path", benchmark.get("nccl_lib_dir"))
    if nccl_lib_dir:
        command += ["--nccl-lib-dir", str(nccl_lib_dir)]
    socket_ifname = benchmark.get("socket_ifname")
    if socket_ifname:
        command += ["--socket-ifname", str(socket_ifname)]
    if benchmark.get("enable_ib", False):
        command.append("--enable-ib")
    elif benchmark.get("disable_ib", False):
        command.append("--disable-ib")
    # Rounds are carried in the explicit subprocess environment because the
    # existing benchmark ABI uses this non-kernel test control.  No kernel
    # tuning decision is passed through the environment.
    _require(rounds > 10, "Forward benchmark rounds must be >10")
    return command


def extract_benchmark_json(output: str,
                           benchmark: Mapping[str, Any]) -> dict[str, Any]:
    local = benchmark.get("launcher", "multinode") == "local"
    prefix = "FORWARD_BENCH_JSON " if local else "GIN_BENCH_JSON "
    payloads = [line[len(prefix):] for line in output.splitlines()
                if line.startswith(prefix)]
    if len(payloads) != 1:
        raise RuntimeError(
            f"launcher emitted {len(payloads)} {prefix.strip()} records")
    result = json.loads(payloads[0])
    expected = ("single-persistent-forward-main-shape" if local else
                "multinode-gin-forward-main-shape")
    if result.get("benchmark") != expected:
        raise RuntimeError(f"unexpected benchmark record: {result.get('benchmark')}")
    return result


def benchmark_candidate(candidate: Mapping[str, Any], binary: Path,
                        benchmark: Mapping[str, Any], log_dir: Path,
                        environment: Mapping[str, str], *, rounds: int,
                        suffix: str = "tune") -> dict[str, Any]:
    command = launcher_command(binary, benchmark, rounds=rounds)
    run_environment = dict(environment)
    run_environment["WEFTMOE_P2P_BENCH_TOKENS"] = str(
        int(benchmark["tokens_per_rank"]))
    run_environment["WEFTMOE_P2P_TOP_K"] = str(int(benchmark["top_k"]))
    run_environment["WEFTMOE_FORWARD_BENCH_ROUNDS"] = str(rounds)
    run_environment["WEFTMOE_FORWARD_PRE_COMBINE"] = str(
        int(benchmark.get("pre_combine", False)))
    run_environment["WEFTMOE_FORWARD_BENCH_GENERATION_BASE"] = str(
        int(benchmark["generation_base"]))
    if "experts" in benchmark:
        run_environment["WEFTMOE_BENCH_EXPERTS"] = str(benchmark["experts"])
    if benchmark.get("launcher", "multinode") == "local":
        nccl_lib_dir = benchmark.get("runtime_library_path", benchmark.get("nccl_lib_dir"))
        if nccl_lib_dir:
            prior = run_environment.get("LD_LIBRARY_PATH", "")
            run_environment["LD_LIBRARY_PATH"] = (
                str(nccl_lib_dir) + ((":" + prior) if prior else ""))
        if benchmark.get("socket_ifname"):
            run_environment["NCCL_SOCKET_IFNAME"] = str(
                benchmark["socket_ifname"])
        if benchmark.get("disable_ib", True):
            run_environment["NCCL_IB_DISABLE"] = "1"
    log_path = log_dir / f"{candidate['name']}.{suffix}.log"
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=run_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(benchmark["timeout_seconds"]) + 60,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output += f"\nTIMEOUT: {error}\n"
        returncode = 124
    elapsed = time.monotonic() - start
    log_path.write_text("COMMAND " + json.dumps(command) + "\n" + output,
                        encoding="utf-8")
    record = None
    parse_error = None
    if returncode == 0:
        try:
            record = extract_benchmark_json(output, benchmark)
        except (RuntimeError, json.JSONDecodeError) as error:
            parse_error = str(error)
    return {
        "command": command,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "log": str(log_path),
        "parse_error": parse_error,
        "record": record,
    }


def correctness_fingerprint(record: Mapping[str, Any]) -> str:
    missing = [key for key in CORRECTNESS_KEYS if key not in record]
    if missing:
        raise RuntimeError(f"benchmark JSON omits correctness keys: {missing}")
    selected = {key: record[key] for key in CORRECTNESS_KEYS}
    return hashlib.sha256(json.dumps(
        selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_revision() -> dict[str, Any]:
    return _common_git_revision(REPO_ROOT)


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    _common_write_result(path, result)


def _timing_summary(result: Mapping[str, Any], elapsed: float) -> dict[str, float]:
    return _common_timing_summary(result, elapsed)


def _configure_from_cli(plan: dict[str, Any], args: argparse.Namespace) -> None:
    jit = plan["jit"]
    benchmark = plan["benchmark"]
    selection = plan["selection"]
    if args.jobs is not None:
        jit["jobs"] = args.jobs
    if args.cache_root is not None:
        jit["cache_root"] = args.cache_root
    if args.nvcc is not None:
        jit["nvcc"] = args.nvcc
    if args.cxx is not None:
        jit["cxx"] = args.cxx
    if args.remote is not None:
        benchmark["remote"] = args.remote
    if args.port is not None:
        benchmark["port"] = args.port
    if args.nccl_lib_dir is not None:
        benchmark["nccl_lib_dir"] = args.nccl_lib_dir
    if args.pre_combine is not None:
        benchmark["pre_combine"] = args.pre_combine
    if args.tune_rounds is not None:
        benchmark["tune_rounds"] = args.tune_rounds
    if args.final_rounds is not None:
        benchmark["final_rounds"] = args.final_rounds
    if args.confirmation_rounds is not None:
        selection["confirmation_rounds"] = args.confirmation_rounds
    if args.confirmation_repeats is not None:
        selection["confirmation_repeats"] = args.confirmation_repeats
    if args.minimum_improvement_percent is not None:
        selection["minimum_improvement_percent"] = (
            args.minimum_improvement_percent)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-tokens", action="store_true", help="tune the exact token count instead of the 64/8192/65536 bucket representative")
    parser.add_argument("--config", type=Path, help="hyperparameter file; default chosen from topology")
    parser.add_argument("--ep", type=int, help="EP world size; must equal nodes * gpus-per-node")
    parser.add_argument("--nodes", type=int, help="node count (default: 1, or ep / gpus-per-node)")
    parser.add_argument("--gpus-per-node", type=int, help="GPUs per node (default: 8)")
    parser.add_argument("--system-config", type=Path)
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--intermediate", type=int)
    parser.add_argument("--experts", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--pre-combine", action=argparse.BooleanOptionalAction,
                        default=None, help="sum per-GPU contributions before Return; changes cross-EP rounding")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--candidate", action="append", default=[],
                        help="run only this named candidate (repeatable)")
    parser.add_argument("--generate-only", action="store_true",
                        help="validate and generate headers without building")
    parser.add_argument("--build-only", action="store_true",
                        help="dynamically compile requested candidates without benchmarking")
    parser.add_argument("--skip-final", action="store_true",
                        help="do not rerun the winner with final_rounds")
    parser.add_argument("--output", type=Path,
                        help="explicit result JSON path")
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--cache-root")
    parser.add_argument("--nvcc")
    parser.add_argument("--cxx")
    parser.add_argument("--remote")
    parser.add_argument("--port", type=int)
    parser.add_argument("--nccl-lib-dir")
    parser.add_argument("--tune-rounds", type=int)
    parser.add_argument("--final-rounds", type=int)
    parser.add_argument("--confirmation-rounds", type=int)
    parser.add_argument("--confirmation-repeats", type=int)
    parser.add_argument("--minimum-improvement-percent", type=float)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not (args.generate_only and args.build_only),
             "--generate-only and --build-only are mutually exclusive")
    _require(not args.build_only or args.candidate,
             "--build-only requires at least one explicit --candidate")
    model = {key: getattr(args, key) for key in ("hidden", "intermediate", "experts", "top_k")}
    system = None
    if args.system_config:
        try:
            system = json.loads(args.system_config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise PlanError(f"cannot read system config: {error}") from error
        _require(isinstance(system, dict), "system config must be an object")
    system = dict(system or {})
    for key in ("ep", "nodes", "gpus_per_node", "remote", "port", "nccl_lib_dir"):
        if getattr(args, key) is not None:
            system[key] = getattr(args, key)
    parallel = resolve_parallel_settings(system)
    topology_name = "single_node" if parallel["nodes"] == 1 else "multi_node"
    config_path = (args.config or environment_autotune_config() or REPO_ROOT / "configs" /
                   f"forward_h200_{topology_name}_autotune.json").resolve()
    plan = load_plan(config_path, model=model,
                     benchmark={"tokens_per_rank": args.tokens} if args.tokens is not None else None,
                     system=system)
    configure_preparation(plan, "forward", exact_tokens=args.exact_tokens)
    _configure_from_cli(plan, args)
    if not args.generate_only and not args.build_only and plan["benchmark"].get("launcher", "multinode") == "multinode":
        _require(bool(plan["benchmark"].get("remote")),
                 "multinode benchmark requires --remote or system.remote")
    if not args.generate_only and not args.build_only:
        validate_execution_topology(plan["benchmark"])
    validate_jit_config(plan["jit"])
    candidates = expand_candidates(plan)
    requested = set(args.candidate)
    if requested:
        known = {candidate["name"] for candidate in candidates}
        unknown = requested - known
        _require(not unknown, f"unknown requested candidates: {sorted(unknown)}")
        candidates = [candidate for candidate in candidates
                      if candidate["name"] in requested]
    _require(candidates, "candidate filter selected no configurations")
    validate_selection(plan["selection"], len(plan["candidates"]))

    jit = plan["jit"]
    benchmark = plan["benchmark"]
    selection = plan["selection"]
    _require(type(benchmark["tune_rounds"]) is int and
             benchmark["tune_rounds"] > 10,
             "benchmark.tune_rounds must be an integer >10")
    _require(type(benchmark["final_rounds"]) is int and
             benchmark["final_rounds"] > 10,
             "benchmark.final_rounds must be an integer >10")
    cache_root = _resolve_repo_path(str(jit["cache_root"]))
    result_dir = _resolve_repo_path(str(benchmark["result_dir"]))
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (args.output.resolve() if args.output else
                   result_dir / f"forward_h200_{timestamp}.json")
    log_dir = output_path.parent / (output_path.stem + "-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    materialized = {
        candidate["name"]: materialize_candidate(
            candidate, cache_root / "generated")
        for candidate in candidates
    } if args.generate_only else {}
    if args.generate_only:
        manifest = {
            "schema": 1,
            "mode": "generate-only",
            "benchmark": copy.deepcopy(plan["benchmark"]),
            "config_path": str(config_path),
            "jit": copy.deepcopy(jit),
            "candidates": [
                {
                    **candidate,
                    "digest": candidate_digest(candidate["config"]),
                    "generated_header": str(materialized[candidate["name"]]["header"]),
                }
                for candidate in candidates
            ],
        }
        _write_result(output_path, manifest)
        print(f"generated {len(candidates)} specializations: {output_path}")
        return 0

    if args.build_only:
        environment = _sanitized_environment()
        compiler = ForwardJitCompiler(
            REPO_ROOT, jit, environment,
            nccl_lib_dir=benchmark.get("nccl_lib_dir"),
        )
        start_build_only = time.monotonic()
        manifest: dict[str, Any] = {
            "schema": 1,
            "mode": "dynamic-compile",
            "config_path": str(config_path),
            "git": _git_revision(),
            "jit": copy.deepcopy(jit),
            "candidates": [],
        }
        all_valid = True
        start_compile = time.monotonic()
        build_results = compile_candidates(candidates, compiler, log_dir)
        manifest["compile_wall_seconds"] = time.monotonic() - start_compile
        for candidate in candidates:
            build_result = build_results[candidate["name"]]
            valid = build_result["returncode"] == 0
            all_valid = all_valid and valid
            manifest["candidates"].append({
                **candidate,
                "digest": candidate_digest(candidate["config"]),
                "generated_header": build_result.get("generated_header"),
                "status": "built" if valid else "build_failed",
                "build": build_result,
            })
        manifest["status"] = "complete" if all_valid else "build_failed"
        manifest["elapsed_seconds"] = time.monotonic() - start_build_only
        _write_result(output_path, manifest)
        print(f"dynamic compile result: {output_path}")
        return 0 if all_valid else 1

    start_total = time.monotonic()
    environment = _sanitized_environment()
    compiler = ForwardJitCompiler(
        REPO_ROOT, jit, environment,
        nccl_lib_dir=benchmark.get("nccl_lib_dir"),
    )
    result: dict[str, Any] = {
        "schema": 1,
        "mode": "compile-and-tune",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git": _git_revision(),
        "selection_metric": benchmark["selection_metric"],
        "selection_policy": copy.deepcopy(selection),
        "jit": copy.deepcopy(jit),
        "benchmark": copy.deepcopy(benchmark),
        "phase_wall_seconds": {},
        "candidates": [],
    }

    # Finish compilation before touching the benchmark fabric. This keeps
    # candidate measurements under the same thermal/runtime conditions and
    # makes a cold cache affect wall time, never the ordering of measurements.
    for candidate in candidates:
        candidate_result: dict[str, Any] = {
            **candidate,
            "digest": candidate_digest(candidate["config"]),
            "status": "building",
        }
        result["candidates"].append(candidate_result)
    _write_result(output_path, result)
    start_compile = time.monotonic()
    build_results = compile_candidates(candidates, compiler, log_dir)
    result["phase_wall_seconds"]["compile"] = (
        time.monotonic() - start_compile)
    for candidate_result in result["candidates"]:
        build_result = build_results[candidate_result["name"]]
        candidate_result["build"] = build_result
        candidate_result["generated_header"] = build_result.get(
            "generated_header")
        if build_result["returncode"] != 0:
            candidate_result["status"] = "build_failed"
            print(f"  build failed; see {build_result['log']}", flush=True)
            continue
        candidate_result["status"] = "built"
    _write_result(output_path, result)

    reference_fingerprint = None
    candidate_by_name = {candidate["name"]: candidate
                         for candidate in candidates}
    benchmarkable = [item for item in result["candidates"]
                     if item["status"] == "built"]
    start_tune = time.monotonic()
    for ordinal, candidate_result in enumerate(benchmarkable, 1):
        candidate = candidate_by_name[candidate_result["name"]]
        print(f"[tune {ordinal}/{len(benchmarkable)}] {candidate['name']}: "
              f"{int(benchmark['tune_rounds'])} rounds (including 10 warmup)", flush=True)
        bench_result = benchmark_candidate(
            candidate, Path(candidate_result["build"]["binary"]), benchmark,
            log_dir,
            environment, rounds=int(benchmark["tune_rounds"]))
        candidate_result["tune"] = bench_result
        if bench_result["returncode"] != 0 or bench_result["record"] is None:
            candidate_result["status"] = "benchmark_failed"
            print(f"  benchmark failed; see {bench_result['log']}", flush=True)
            _write_result(output_path, result)
            continue
        fingerprint = correctness_fingerprint(bench_result["record"])
        candidate_result["correctness_fingerprint"] = fingerprint
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        if fingerprint != reference_fingerprint:
            candidate_result["status"] = "correctness_mismatch"
            print("  correctness fingerprint differs from first valid candidate",
                  flush=True)
            _write_result(output_path, result)
            continue
        metric = metric_value(bench_result["record"],
                              str(benchmark["selection_metric"]))
        candidate_result["selection_metric_us"] = metric
        candidate_result["status"] = "valid"
        forward_stats = bench_result["record"]["forward_slowest_rank_us"]
        print(f"  forward mean={forward_stats['mean']:.3f} us "
              f"p95={forward_stats['p95']:.3f} us", flush=True)
        _write_result(output_path, result)
    result["phase_wall_seconds"]["tune"] = time.monotonic() - start_tune

    valid = [item for item in result["candidates"]
             if item["status"] == "valid"]
    if not valid:
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["timing"] = _timing_summary(
            result, result["elapsed_seconds"])
        result["status"] = "no_valid_candidate"
        _write_result(output_path, result)
        print(f"no valid candidate; partial result: {output_path}", file=sys.stderr)
        return 1
    incumbent_name = candidates[0]["name"]
    finalist_count = min(int(selection["confirmation_candidates"]), len(valid))
    finalists = confirmation_finalists(
        valid, incumbent_name, finalist_count)
    result["selection_finalists"] = [item["name"] for item in finalists]
    confirmation_repeats = int(selection["confirmation_repeats"])
    for candidate_result in finalists:
        candidate_result["confirmations"] = []
        candidate_result["confirmation_status"] = "running"
    start_confirmation = time.monotonic()
    total_confirmation_runs = len(finalists) * confirmation_repeats
    completed_confirmation_runs = 0
    for repeat in range(confirmation_repeats):
        ordered = finalists if repeat % 2 == 0 else list(reversed(finalists))
        for candidate_result in ordered:
            completed_confirmation_runs += 1
            if candidate_result["confirmation_status"] != "running":
                continue
            candidate = candidate_by_name[candidate_result["name"]]
            print(f"[confirm {completed_confirmation_runs}/"
                  f"{total_confirmation_runs}] {candidate['name']} "
                  f"pass={repeat + 1}: "
                  f"{int(selection['confirmation_rounds'])} rounds (including 10 warmup)", flush=True)
            confirmation = benchmark_candidate(
                candidate, Path(candidate_result["build"]["binary"]),
                benchmark, log_dir, environment,
                rounds=int(selection["confirmation_rounds"]),
                suffix=f"confirm{repeat + 1}")
            candidate_result["confirmations"].append(confirmation)
            if (confirmation["returncode"] != 0 or
                    confirmation["record"] is None):
                candidate_result["confirmation_status"] = "benchmark_failed"
                print(f"  confirmation failed; see {confirmation['log']}",
                      flush=True)
                _write_result(output_path, result)
                continue
            fingerprint = correctness_fingerprint(confirmation["record"])
            confirmation["correctness_fingerprint"] = fingerprint
            if fingerprint != reference_fingerprint:
                candidate_result["confirmation_status"] = (
                    "correctness_mismatch")
                print("  confirmation correctness fingerprint changed",
                      flush=True)
                _write_result(output_path, result)
                continue
            confirmation_metric = metric_value(
                confirmation["record"], str(benchmark["selection_metric"]))
            confirmation["metric_us"] = confirmation_metric
            print(f"  confirmation mean={confirmation_metric:.3f} us",
                  flush=True)
            _write_result(output_path, result)

    for candidate_result in finalists:
        if candidate_result["confirmation_status"] != "running":
            continue
        confirmation_records = [
            confirmation["record"]
            for confirmation in candidate_result["confirmations"]
        ]
        if len(confirmation_records) != confirmation_repeats:
            candidate_result["confirmation_status"] = "incomplete"
            continue
        candidate_result["confirmation_metrics_us"] = [
            confirmation["metric_us"]
            for confirmation in candidate_result["confirmations"]
        ]
        candidate_result["confirmation_metric_us"] = weighted_metric(
            confirmation_records, str(benchmark["selection_metric"]))
        candidate_result["aggregate_metric_us"] = weighted_metric(
            [candidate_result["tune"]["record"], *confirmation_records],
            str(benchmark["selection_metric"]))
        candidate_result["confirmation_status"] = "confirmed"
        print(f"[confirm aggregate] {candidate_result['name']}: "
              f"{candidate_result['aggregate_metric_us']:.3f} us", flush=True)
    _write_result(output_path, result)
    result["phase_wall_seconds"]["confirmation"] = (
        time.monotonic() - start_confirmation)

    confirmed = [item for item in finalists
                 if item.get("confirmation_status") == "confirmed"]
    if not confirmed:
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["status"] = "no_confirmed_candidate"
        result["timing"] = _timing_summary(
            result, result["elapsed_seconds"])
        _write_result(output_path, result)
        print(f"no candidate completed confirmation; partial result: "
              f"{output_path}", file=sys.stderr)
        return 1

    winner, decision = choose_confirmed_candidate(
        confirmed, incumbent_name,
        float(selection["minimum_improvement_percent"]))
    result["selection_decision"] = decision
    result["selected"] = {
        "name": winner["name"],
        "digest": winner["digest"],
        "config": winner["config"],
        "initial_metric_us": winner["selection_metric_us"],
        "selection_metric_us": winner["aggregate_metric_us"],
        "aggregate_metric_us": winner["aggregate_metric_us"],
        "binary": winner["build"]["binary"],
    }

    if not args.skip_final:
        print(f"final validation {winner['name']}: "
              f"{int(benchmark['final_rounds'])} rounds (including 10 warmup)", flush=True)
        winner_candidate = next(candidate for candidate in candidates
                                if candidate["name"] == winner["name"])
        start_final = time.monotonic()
        final_result = benchmark_candidate(
            winner_candidate, Path(winner["build"]["binary"]), benchmark,
            log_dir, environment, rounds=int(benchmark["final_rounds"]),
            suffix="final")
        result["phase_wall_seconds"]["final"] = time.monotonic() - start_final
        result["selected"]["final"] = final_result
        if final_result["returncode"] != 0 or final_result["record"] is None:
            result["status"] = "final_validation_failed"
            result["elapsed_seconds"] = time.monotonic() - start_total
            result["timing"] = _timing_summary(
                result, result["elapsed_seconds"])
            _write_result(output_path, result)
            print(f"final validation failed; see {final_result['log']}",
                  file=sys.stderr)
            return 1
        if correctness_fingerprint(final_result["record"]) != reference_fingerprint:
            result["status"] = "final_correctness_mismatch"
            result["elapsed_seconds"] = time.monotonic() - start_total
            result["timing"] = _timing_summary(
                result, result["elapsed_seconds"])
            _write_result(output_path, result)
            print("final validation correctness fingerprint changed",
                  file=sys.stderr)
            return 1
        result["selected"]["final_metric_us"] = metric_value(
            final_result["record"], str(benchmark["selection_metric"]))

    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - start_total
    result["timing"] = _timing_summary(result, result["elapsed_seconds"])
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_result(output_path, result)
    print(f"selected {winner['name']} at "
          f"{winner['aggregate_metric_us']:.3f} us "
          f"({decision['reason']}); result {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlanError, JitError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)
