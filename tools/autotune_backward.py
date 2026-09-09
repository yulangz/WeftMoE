#!/usr/bin/env python3
"""JIT-compile and tune explicit WeftMoE Backward specializations."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

from autotune_common import (
    resolve_runtime_settings,
    resolve_parallel_settings,
    constrain_transport_space,
    validate_execution_topology,
    PlanError,
    candidate_digest,
    choose_confirmed_candidate,
    compile_in_parallel,
    confirmation_finalists,
    git_revision,
    metric_value,
    require,
    require_choice,
    timing_summary,
    validate_selection,
    weighted_metric,
    write_result,
)
from forward_jit import BackwardJitCompiler, JitError, validate_jit_config


REPO_ROOT = Path(__file__).resolve().parents[1]
GROUP_NAMES = (
    "1_gemm",
    "2_transport",
    "3_scheduler",
    "4_completion_reduce",
)

# Keep this surface exhaustive. A checked-in C++ default that is absent here
# must fail the source/config parity test instead of becoming an invisible
# tuning input.
PARAMETER_MACROS = {
    "main_hidden": "WEFTMOE_BWD_MAIN_HIDDEN",
    "main_intermediate": "WEFTMOE_BWD_MAIN_INTERMEDIATE",
    "tile_m": "WEFTMOE_BWD_TILE_M",
    "dab_tile_n": "WEFTMOE_BWD_DAB_TILE_N",
    "dx_tile_n": "WEFTMOE_BWD_DX_TILE_N",
    "dw_tile_m": "WEFTMOE_BWD_DW_TILE_M",
    "dw_tile_n": "WEFTMOE_BWD_DW_TILE_N",
    "tile_k": "WEFTMOE_BWD_TILE_K",
    "dab_stages": "WEFTMOE_BWD_DAB_STAGES",
    "dx_stages": "WEFTMOE_BWD_DX_STAGES",
    "dw1_stages": "WEFTMOE_BWD_DW1_STAGES",
    "dw2_stages": "WEFTMOE_BWD_DW2_STAGES",
    "compute_threads": "WEFTMOE_BWD_COMPUTE_THREADS",
    "math_warpgroups": "WEFTMOE_BWD_MATH_WARPGROUPS",
    "producer_warpgroups": "WEFTMOE_BWD_PRODUCER_WARPGROUPS",
    "math_registers": "WEFTMOE_BWD_MATH_REGISTERS",
    "producer_registers": "WEFTMOE_BWD_PRODUCER_REGISTERS",
    "rdma_ctas": "WEFTMOE_BWD_RDMA_CTAS",
    "relay_ctas": "WEFTMOE_BWD_RELAY_CTAS",
    "dispatch_batch_rows": "WEFTMOE_BWD_DISPATCH_BATCH_ROWS",
    "gin_max_inflight": "WEFTMOE_BWD_GIN_MAX_INFLIGHT",
    "setup_threads": "WEFTMOE_BWD_SETUP_THREADS",
    "setup_idle_nanosleep": "WEFTMOE_BWD_SETUP_IDLE_NANOSLEEP",
    "queue_idle_nanosleep": "WEFTMOE_BWD_QUEUE_IDLE_NANOSLEEP",
    "ready_word_stagger": "WEFTMOE_BWD_READY_WORD_STAGGER",
    "activation_owner_group": "WEFTMOE_BWD_ACTIVATION_OWNER_GROUP",
    "epilogue_rows_per_task": "WEFTMOE_BWD_EPILOGUE_ROWS_PER_TASK",
    "dw2_claim_tiles": "WEFTMOE_BWD_DW2_CLAIM_TILES",
    "dw1_claim_tiles": "WEFTMOE_BWD_DW1_CLAIM_TILES",
    "reduce_threads": "WEFTMOE_BWD_REDUCE_THREADS",
    "reduce_idle_nanosleep": "WEFTMOE_BWD_REDUCE_IDLE_NANOSLEEP",
}

EXPECTED_INVARIANTS = {
    "queue_priority": ["dx", "dab", "dw2", "dw1"],
    "linear_dw_claim": True,
    "second_dx_probe": False,
    "cohort_count": 1,
    "forward_return_chunk_rows": 256,
}


def flatten_groups(groups: Mapping[str, Any], *,
                   require_all: bool) -> dict[str, Any]:
    require(isinstance(groups, dict), "tuning groups must be a JSON object")
    unknown_groups = set(groups) - set(GROUP_NAMES)
    require(not unknown_groups,
            f"unknown tuning groups: {sorted(unknown_groups)}")
    if require_all:
        missing_groups = set(GROUP_NAMES) - set(groups)
        require(not missing_groups,
                f"missing tuning groups: {sorted(missing_groups)}")
    flat: dict[str, Any] = {}
    for group_name in GROUP_NAMES:
        group = groups.get(group_name, {})
        require(isinstance(group, dict), f"{group_name} must be an object")
        for key, value in group.items():
            require(key in PARAMETER_MACROS,
                    f"unknown Backward parameter {group_name}.{key}")
            require(key not in flat,
                    f"Backward parameter {key} appears in multiple groups")
            flat[key] = value
    if require_all:
        missing = set(PARAMETER_MACROS) - set(flat)
        require(not missing, f"fixed config omits parameters: {sorted(missing)}")
    return flat


def validate_tuning(config: Mapping[str, Any]) -> None:
    require(set(config) == set(PARAMETER_MACROS),
            "expanded tuning config must define every parameter exactly once")
    for key, value in config.items():
        require(type(value) is int, f"{key} must be an integer")
        require(value >= 0, f"{key} must be non-negative")

    require(config["main_hidden"] > 0 and config["main_hidden"] % 256 == 0,
             "main_hidden must be a positive multiple of 256")
    require(config["main_intermediate"] > 0 and config["main_intermediate"] % 256 == 0,
             "main_intermediate must be a positive multiple of 256")
    require(config["tile_m"] == 128,
            "the activation dependency block is fixed at M128")
    require_choice(config, "dab_tile_n", (128, 256))
    require_choice(config, "dx_tile_n", (128, 256))
    require_choice(config, "dw_tile_m", (128, 256))
    require_choice(config, "dw_tile_n", (128, 256))
    require(config["main_hidden"] % config["dw_tile_m"] == 0,
            "dW2 M tile must divide hidden")
    require(config["main_intermediate"] % config["dw_tile_n"] == 0,
            "dW2 N tile must divide intermediate")
    require(2 * config["main_intermediate"] % config["dw_tile_m"] == 0,
            "dW1 M tile must divide twice intermediate")
    require(config["main_hidden"] % config["dw_tile_n"] == 0,
            "dW1 N tile must divide hidden")
    require(config["tile_k"] == 64,
            "the Hopper 128-byte-swizzle descriptor requires tile_k=64")
    for key in ("dab_stages", "dx_stages", "dw1_stages", "dw2_stages"):
        require_choice(config, key, (2, 3, 4))
    require(config["dab_stages"] == config["dx_stages"],
            "dAB and dX share one activation pipeline stage count")
    require(config["dw1_stages"] == config["dw2_stages"],
            "dW1 and dW2 share one weight-gradient pipeline stage count")
    require(config["compute_threads"] == 384,
            "the current producer plus two math-WG kernel requires 384 threads")
    require(config["math_warpgroups"] == 2 and
            config["producer_warpgroups"] == 1,
            "the current WGMMA mapping requires two math and one producer WG")
    for key in ("math_registers", "producer_registers"):
        require(1 <= config[key] <= 255, f"{key} must be in [1, 255]")
    registers = 128 * (
        config["math_warpgroups"] * config["math_registers"] +
        config["producer_warpgroups"] * config["producer_registers"]
    )
    require(registers <= 65536,
            f"compute register allocation {registers} exceeds 65536")

    require(config["rdma_ctas"] in (0, 8),
            "RDMA CTAs must be zero for local or eight for GIN")
    require(config["relay_ctas"] >= 8 and config["relay_ctas"] % 4 == 0,
            "relay_ctas must be >=8 and a multiple of 4")
    require_choice(config, "dispatch_batch_rows", (12, 16, 32, 64, 128))
    require_choice(config, "gin_max_inflight", (8, 16, 24, 32, 64))
    for key in ("setup_threads", "reduce_threads"):
        require_choice(config, key, (128, 256, 384, 512))
    require_choice(config, "ready_word_stagger", (0, 1))
    require_choice(config, "activation_owner_group", (1, 2, 4, 8, 16, 32))
    require_choice(config, "epilogue_rows_per_task",
                   (12, 16, 24, 32, 48, 64, 96, 128))
    require_choice(config, "dw2_claim_tiles", (1, 4, 8, 16))
    require_choice(config, "dw1_claim_tiles", (1, 4, 8, 16))
    for key in (
        "setup_idle_nanosleep",
        "queue_idle_nanosleep",
        "reduce_idle_nanosleep",
    ):
        require_choice(config, key, (0, 16, 32, 64, 128, 256, 512))


def make_runtime_plan(space: Mapping[str, Any], *, model: Mapping[str, Any] | None,
                      benchmark: Mapping[str, Any] | None = None,
                      system: Mapping[str, Any] | None = None) -> dict[str, Any]:
    topology, jit_overrides = resolve_runtime_settings(
        model=model, system=system, repo_root=REPO_ROOT)
    space = constrain_transport_space(space, topology)
    flat = flatten_groups(space, require_all=False)
    require(set(flat) == set(PARAMETER_MACROS) - {"main_hidden", "main_intermediate"},
            "hyperparameters must cover every tuning parameter and exclude model dimensions")
    count = 1
    for key, values in flat.items():
        require(isinstance(values, list) and values, f"{key} must be a non-empty array")
        require(len({json.dumps(v, sort_keys=True) for v in values}) == len(values),
                f"{key} has duplicate values")
        count *= len(values)
    require(count <= 4096, "Cartesian candidate count exceeds 4096")
    fixed = {g: {k: vs[0] for k, vs in space.get(g, {}).items()} for g in GROUP_NAMES}
    fixed["1_gemm"].update(main_hidden=model["hidden"], main_intermediate=model["intermediate"])
    axes = [(g, k, vs) for g in GROUP_NAMES for k, vs in space.get(g, {}).items() if len(vs) > 1]
    candidates = []
    for index, values in enumerate(itertools.product(*(axis[2] for axis in axes))):
        override = {}
        for (g, k, _), value in zip(axes, values):
            if value != fixed[g][k]:
                override.setdefault(g, {})[k] = value
        candidates.append(dict(name="incumbent" if index == 0 else f"grid_{index:04d}", override=override))
    workload = dict(benchmark or {})
    require(not (set(workload) - {"tokens_per_rank", "seed", "tune_rounds", "final_rounds",
        "timeout_seconds", "result_dir", "preflight_world", "preflight_tokens"}),
        "benchmark contains model/system fields or unknown settings")
    return dict(schema=1, max_candidates=4096, fixed=fixed, candidates=candidates,
        invariants=copy.deepcopy(EXPECTED_INVARIANTS),
        jit=dict(cache_root="artifacts/backward-jit", jobs=8, architecture="sm_90a",
                 timeout_seconds=1800) | jit_overrides,
        selection=dict(confirmation_candidates=min(2, count), confirmation_repeats=2,
                       confirmation_rounds=60, minimum_improvement_percent=0.5),
        benchmark=dict(seed=0x4D554C54, tokens_per_rank=8192, top_k=model["top_k"],
            experts=model["experts"], tune_rounds=30, final_rounds=110, timeout_seconds=600,
            preflight_world=8, preflight_tokens=8,
            selection_metric="total_slowest_rank_us.mean", result_dir="artifacts/backward-autotune")
            | topology | workload)


def load_plan(path: Path, *, model: Mapping[str, Any] | None = None,
              benchmark: Mapping[str, Any] | None = None,
              system: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load {path}: {error}") from error
    require(isinstance(document, dict), "top-level plan must be an object")
    if "schema" in document and any(k in (system or {}) for k in ("ep", "nodes", "gpus_per_node")):
        requested = resolve_parallel_settings(system)
        frozen = document.get("benchmark", {})
        actual_nodes = frozen.get("nodes", 1 if frozen.get("launcher") == "local" else 2)
        actual_gpus = frozen.get("gpus_per_node", frozen.get("world", 8))
        require(requested["nodes"] == actual_nodes and requested["gpus_per_node"] == actual_gpus,
                "explicit parallel settings conflict with frozen plan topology")
    if "schema" not in document:
        document = make_runtime_plan(document, model=model, benchmark=benchmark, system=system)
    require(document.get("schema") == 1, "Backward plan schema must be 1")
    required_sections = {
        "jit", "benchmark", "selection", "invariants", "fixed", "candidates"
    }
    require(required_sections <= set(document),
            f"plan omits sections: {sorted(required_sections - set(document))}")
    require(document["invariants"] == EXPECTED_INVARIANTS,
            "Backward invariants do not match the implemented scheduler contract")
    try:
        validate_jit_config(document["jit"])
    except JitError as error:
        raise PlanError(str(error)) from error

    benchmark = document["benchmark"]
    require(isinstance(benchmark, dict), "benchmark must be an object")
    launcher = benchmark.get("launcher", "multinode")
    require(launcher in ("multinode", "local"),
            "benchmark.launcher must be multinode or local")
    common_keys = (
        "seed", "tokens_per_rank", "top_k",
        "tune_rounds", "final_rounds", "timeout_seconds",
        "selection_metric", "result_dir", "preflight_world",
        "preflight_tokens",
    )
    for key in common_keys:
        require(key in benchmark, f"benchmark omits {key}")
    if launcher == "multinode":
        for key in ("port", "gpus_per_node"):
            require(key in benchmark, f"benchmark omits {key}")
    else:
        require("world" in benchmark, "local benchmark omits world")
    for key in (
        "seed", "tokens_per_rank", "top_k",
        "tune_rounds", "final_rounds", "timeout_seconds",
        "preflight_world", "preflight_tokens",
    ):
        require(type(benchmark[key]) is int and benchmark[key] > 0,
                f"benchmark.{key} must be a positive integer")
    require(benchmark["top_k"] in (1, 6, 8),
            "benchmark.top_k must be 1, 6, or 8")
    require(benchmark["tune_rounds"] > 10 and
            benchmark["final_rounds"] > 10,
            "Backward benchmark rounds must be >10")
    require(benchmark["preflight_world"] in (1, 2, 8),
            "benchmark.preflight_world must be 1, 2, or 8")
    if launcher == "local":
        require(benchmark["world"] in (1, 2, 8),
                "local benchmark.world must be 1, 2, or 8")
    require(isinstance(document["candidates"], list) and
            document["candidates"], "candidates must be a non-empty array")
    maximum = document.get("max_candidates", 8)
    require(type(maximum) is int and 1 <= maximum <= 4096,
            "max_candidates must be in [1, 4096]")
    require(len(document["candidates"]) <= maximum,
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
        require(isinstance(candidate, dict), "each candidate must be an object")
        name = candidate.get("name")
        require(isinstance(name, str) and
                re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is not None,
                f"invalid candidate name {name!r}")
        require(name not in names, f"duplicate candidate name {name}")
        names.add(name)
        override = flatten_groups(candidate.get("override", {}),
                                  require_all=False)
        values = dict(fixed)
        values.update(override)
        validate_tuning(values)
        fingerprint = json.dumps(values, sort_keys=True, separators=(",", ":"))
        require(fingerprint not in fingerprints,
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
        "// Generated by tools/autotune_backward.py.",
        f"// config_sha256_prefix={digest}",
    ]
    for key, macro in PARAMETER_MACROS.items():
        lines.append(f"#define {macro} {config[key]}")
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
    header = include_dir / "weftmoe_backward_tuning_generated.hpp"
    content = generated_header(candidate["name"], candidate["config"])
    if not header.exists() or header.read_text(encoding="utf-8") != content:
        header.write_text(content, encoding="utf-8")
    return {"build_dir": build_dir, "include_dir": include_dir,
            "header": header}


def dynamic_shared_memory(config: Mapping[str, Any]) -> int:
    storage_tile_n = max(config["dab_tile_n"], config["dx_tile_n"],
                         config["dw_tile_n"])
    activation_elements = (
        config["tile_m"] + storage_tile_n
    ) * config["tile_k"]
    dw_elements = (
        config["dw_tile_m"] + config["dw_tile_n"]
    ) * config["tile_k"]
    stage_elements = max(activation_elements, dw_elements)
    activation_arena = config["dab_stages"] * stage_elements
    dw_arena = (config["dw1_stages"] * stage_elements +
                config["dw_tile_m"] * config["dw_tile_n"])
    barriers = 2 * (config["dab_stages"] + config["dw1_stages"])
    return 2 * max(activation_arena, dw_arena) + 8 * barriers


def inspect_backward_kernels(binary: Path, log_path: Path,
                             config: Mapping[str, Any]) -> dict[str, Any]:
    cuobjdump = shutil.which("cuobjdump")
    if cuobjdump is None:
        return {"available": False, "reason": "cuobjdump not found"}
    completed = subprocess.run(
        [cuobjdump, "--dump-resource-usage", str(binary)], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    resource_pattern = re.compile(
        r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
    kernels: dict[str, list[dict[str, Any]]] = {
        "compute": [], "transport": []
    }
    for index, line in enumerate(completed.stdout.splitlines()[:-1]):
        kind = next((name for name in kernels
                     if f"backward_local_{name}_kernel" in line), None)
        if kind is None or not line.startswith(" Function "):
            continue
        resources = resource_pattern.search(
            completed.stdout.splitlines()[index + 1])
        if resources is None:
            continue
        kernels[kind].append({
            "mangled_name": line[len(" Function "):-1],
            "registers_per_thread": int(resources.group(1)),
            "stack_bytes_per_thread": int(resources.group(2)),
            "static_shared_bytes": int(resources.group(3)),
            "local_bytes": int(resources.group(4)),
        })
    return {
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "single_compute_kernel": len(kernels["compute"]) == 1,
        "single_transport_kernel": len(kernels["transport"]) == 1,
        "kernels": kernels,
        "dynamic_shared_memory_bytes": dynamic_shared_memory(config),
        "log": str(log_path),
    }


def sanitized_environment() -> dict[str, str]:
    environment = os.environ.copy()
    explicit = {
        "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME", "WEFTMOE_P2P_TOP_K",
        "WEFTMOE_P2P_BENCH_TOKENS", "WEFTMOE_BACKWARD_ROUNDS",
        "WEFTMOE_BACKWARD_WARMUPS", "WEFTMOE_BACKWARD_TOKENS",
        "WEFTMOE_BACKWARD_OUTPUT_DIR",
    }
    for key in tuple(environment):
        if key.startswith("WEFTMOE_BWD_") or key in explicit:
            environment.pop(key, None)
    return environment


def compile_candidate(candidate: Mapping[str, Any],
                      compiler: BackwardJitCompiler, log_dir: Path, *,
                      command_jobs: int | None = None) -> dict[str, Any]:
    result = compiler.build(
        candidate_name=str(candidate["name"]),
        config_digest=candidate_digest(candidate["config"]),
        generated_header=generated_header(
            str(candidate["name"]), candidate["config"]),
        log_path=log_dir / f"{candidate['name']}.build.log",
        command_jobs=command_jobs,
    )
    if result["returncode"] == 0:
        resources = inspect_backward_kernels(
            Path(result["binary"]),
            log_dir / f"{candidate['name']}.resources.log",
            candidate["config"],
        )
        result["resources"] = resources
        if (resources.get("available") and
                (not resources.get("single_compute_kernel") or
                 not resources.get("single_transport_kernel"))):
            result["returncode"] = 125
            result["resource_validation_error"] = (
                "expected one Backward compute and one transport kernel"
            )
    return result


def launcher_command(binary: Path, benchmark: Mapping[str, Any], *,
                     rounds: int) -> list[str]:
    require(rounds > 10, "Backward benchmark rounds must be >10")
    if benchmark.get("launcher", "multinode") == "local":
        return [
            sys.executable,
            str(REPO_ROOT / "tests" / "run_local_activation_p2p.py"),
            "--binary", str(binary),
            "--world", str(int(benchmark["world"])),
            "--seed", str(int(benchmark["seed"])),
            "--timeout", str(int(benchmark["timeout_seconds"])),
            "--rounds", str(rounds),
            "--backward-benchmark",
            "--backward-tokens", str(int(benchmark["tokens_per_rank"])),
            "--top-k", str(int(benchmark["top_k"])),
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
        "--scenario", "gin_backward_bench",
        "--timeout", str(int(benchmark["timeout_seconds"])),
    ]
    library_path = benchmark.get("runtime_library_path", benchmark.get("nccl_lib_dir"))
    if library_path:
        command += ["--nccl-lib-dir", str(library_path)]
    if benchmark.get("socket_ifname"):
        command += ["--socket-ifname", str(benchmark["socket_ifname"])]
    command.append("--enable-ib" if benchmark.get("enable_ib", False)
                   else "--disable-ib")
    return command


def _run(command: list[str], environment: Mapping[str, str], log_path: Path,
         timeout: int) -> dict[str, Any]:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=dict(environment),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=timeout + 60, check=False,
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
    return {"command": command, "elapsed_seconds": elapsed,
            "returncode": returncode, "log": str(log_path),
            "output": output}


def benchmark_candidate(candidate: Mapping[str, Any], binary: Path,
                        benchmark: Mapping[str, Any], log_dir: Path,
                        environment: Mapping[str, str], *, rounds: int,
                        suffix: str = "tune") -> dict[str, Any]:
    run_environment = dict(environment)
    if "experts" in benchmark:
        run_environment["WEFTMOE_BENCH_EXPERTS"] = str(benchmark["experts"])
    run_environment["WEFTMOE_BACKWARD_ROUNDS"] = str(rounds)
    run_environment["WEFTMOE_BACKWARD_WARMUPS"] = "10"
    run_environment["WEFTMOE_P2P_BENCH_TOKENS"] = str(
        int(benchmark["tokens_per_rank"]))
    run_environment["WEFTMOE_P2P_TOP_K"] = str(int(benchmark["top_k"]))
    if benchmark.get("launcher", "multinode") == "local":
        nccl_lib_dir = benchmark.get("runtime_library_path", benchmark.get("nccl_lib_dir"))
        if nccl_lib_dir:
            prior = run_environment.get("LD_LIBRARY_PATH", "")
            run_environment["LD_LIBRARY_PATH"] = (
                str(nccl_lib_dir) + ((":" + prior) if prior else "")
            )
        if benchmark.get("socket_ifname"):
            run_environment["NCCL_SOCKET_IFNAME"] = str(
                benchmark["socket_ifname"])
        run_environment["NCCL_IB_DISABLE"] = "1"
    run = _run(
        launcher_command(binary, benchmark, rounds=rounds), run_environment,
        log_dir / f"{candidate['name']}.{suffix}.log",
        int(benchmark["timeout_seconds"]),
    )
    prefix = ("BACKWARD_BENCH_JSON "
              if benchmark.get("launcher", "multinode") == "local"
              else "GIN_BACKWARD_BENCH_JSON ")
    payloads = [line[len(prefix):] for line in run.pop("output").splitlines()
                if line.startswith(prefix)]
    run["parse_error"] = None
    run["record"] = None
    if run["returncode"] == 0:
        if len(payloads) != 1:
            run["parse_error"] = (
                f"launcher emitted {len(payloads)} {prefix.strip()} records"
            )
        else:
            try:
                run["record"] = json.loads(payloads[0])
            except json.JSONDecodeError as error:
                run["parse_error"] = str(error)
    return run


def preflight_candidate(candidate: Mapping[str, Any], binary: Path,
                        benchmark: Mapping[str, Any], log_dir: Path,
                        environment: Mapping[str, str]) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tests" / "run_local_activation_p2p.py"),
        "--binary", str(binary),
        "--world", str(int(benchmark["preflight_world"])),
        "--seed", str(int(benchmark["seed"])),
        "--timeout", str(int(benchmark["timeout_seconds"])),
        "--rounds", "2",
        "--backward-validation",
        "--backward-tokens", str(int(benchmark["preflight_tokens"])),
        "--top-k", str(int(benchmark["top_k"])),
    ]
    run_environment = dict(environment)
    nccl_lib_dir = benchmark.get("runtime_library_path", benchmark.get("nccl_lib_dir"))
    if nccl_lib_dir:
        prior = run_environment.get("LD_LIBRARY_PATH", "")
        run_environment["LD_LIBRARY_PATH"] = (
            str(nccl_lib_dir) + ((":" + prior) if prior else "")
        )
    run_environment["NCCL_IB_DISABLE"] = "1"
    run = _run(
        command, run_environment,
        log_dir / f"{candidate['name']}.preflight.log",
        int(benchmark["timeout_seconds"]),
    )
    prefix = "BACKWARD_VALIDATION_JSON "
    payloads = [line[len(prefix):] for line in run.pop("output").splitlines()
                if line.startswith(prefix)]
    run["parse_error"] = None
    run["record"] = None
    if run["returncode"] == 0:
        if len(payloads) != 1:
            run["parse_error"] = (
                f"launcher emitted {len(payloads)} validation records"
            )
        else:
            try:
                run["record"] = json.loads(payloads[0])
            except json.JSONDecodeError as error:
                run["parse_error"] = str(error)
    return run


def correctness_fingerprint(record: Mapping[str, Any]) -> str:
    keys = (
        "shape", "per_rank_dx_hash", "per_rank_dgate_hash",
        "per_rank_dw1_hash", "per_rank_dw2_hash", "repeat_equal",
    )
    missing = [key for key in keys if key not in record]
    if missing:
        raise RuntimeError(f"validation JSON omits correctness keys: {missing}")
    selected = {key: record[key] for key in keys}
    return hashlib.sha256(json.dumps(
        selected, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def _configure_from_cli(plan: dict[str, Any], args: argparse.Namespace) -> None:
    if args.jobs is not None:
        plan["jit"]["jobs"] = args.jobs
    if args.cache_root is not None:
        plan["jit"]["cache_root"] = args.cache_root
    if args.remote is not None:
        plan["benchmark"]["remote"] = args.remote
    if args.port is not None:
        plan["benchmark"]["port"] = args.port
    if args.nccl_lib_dir is not None:
        plan["benchmark"]["nccl_lib_dir"] = args.nccl_lib_dir
    if args.tune_rounds is not None:
        plan["benchmark"]["tune_rounds"] = args.tune_rounds
    if args.final_rounds is not None:
        plan["benchmark"]["final_rounds"] = args.final_rounds
    if args.confirmation_rounds is not None:
        plan["selection"]["confirmation_rounds"] = args.confirmation_rounds
    if args.confirmation_repeats is not None:
        plan["selection"]["confirmation_repeats"] = args.confirmation_repeats
    if args.minimum_improvement_percent is not None:
        plan["selection"]["minimum_improvement_percent"] = args.minimum_improvement_percent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="hyperparameter file; default chosen from topology")
    parser.add_argument("--ep", type=int, help="EP world size; must equal nodes * gpus-per-node")
    parser.add_argument("--nodes", type=int, help="node count (default: 1, or ep / gpus-per-node)")
    parser.add_argument("--gpus-per-node", type=int, help="GPUs per node (default: 8)")
    parser.add_argument("--system-config", type=Path)
    for key in ("hidden", "intermediate", "experts", "top-k", "tokens"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--skip-final", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--jobs", type=int)
    parser.add_argument("--cache-root")
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
    require(not (args.generate_only and args.build_only),
            "--generate-only and --build-only are mutually exclusive")
    require(not args.build_only or args.candidate,
            "--build-only requires at least one explicit --candidate")
    model = {key: getattr(args, key) for key in ("hidden", "intermediate", "experts", "top_k")}
    system = {}
    if args.system_config:
        try:
            system = json.loads(args.system_config.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise PlanError(f"cannot read system config: {error}") from error
        require(isinstance(system, dict), "system config must be an object")
    for key in ("ep", "nodes", "gpus_per_node", "remote", "port", "nccl_lib_dir"):
        if getattr(args, key) is not None:
            system[key] = getattr(args, key)
    parallel = resolve_parallel_settings(system)
    topology_name = "single_node" if parallel["nodes"] == 1 else "multi_node"
    config_path = (args.config or REPO_ROOT / "configs" /
                   f"backward_h200_{topology_name}_autotune.json").resolve()
    plan = load_plan(config_path, model=model, system=system,
                     benchmark={"tokens_per_rank": args.tokens} if args.tokens is not None else None)
    if not args.generate_only and not args.build_only and plan["benchmark"].get("launcher") == "multinode":
        require(bool(plan["benchmark"].get("remote")), "multinode benchmark requires --remote or system.remote")
    _configure_from_cli(plan, args)
    if not args.generate_only and not args.build_only:
        validate_execution_topology(plan["benchmark"])
    validate_jit_config(plan["jit"])
    candidates = expand_candidates(plan)
    requested = set(args.candidate)
    if requested:
        known = {candidate["name"] for candidate in candidates}
        require(not requested - known,
                f"unknown requested candidates: {sorted(requested - known)}")
        candidates = [candidate for candidate in candidates
                      if candidate["name"] in requested]
    require(bool(candidates), "candidate filter selected no configurations")

    benchmark = plan["benchmark"]
    cache_root = _resolve_repo_path(str(plan["jit"]["cache_root"]))
    result_dir = _resolve_repo_path(str(benchmark["result_dir"]))
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (args.output.resolve() if args.output else
                   result_dir / f"backward_h200_{timestamp}.json")
    log_dir = output_path.parent / (output_path.stem + "-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.generate_only:
        generated = [
            {
                **candidate,
                "digest": candidate_digest(candidate["config"]),
                "generated_header": str(materialize_candidate(
                    candidate, cache_root / "generated")["header"]),
            }
            for candidate in candidates
        ]
        write_result(output_path, {
            "schema": 1, "mode": "generate-only",
            "benchmark": copy.deepcopy(plan["benchmark"]),
            "config_path": str(config_path), "candidates": generated,
        })
        print(f"generated {len(generated)} specializations: {output_path}")
        return 0

    environment = sanitized_environment()
    compiler = BackwardJitCompiler(
        REPO_ROOT, plan["jit"], environment,
        nccl_lib_dir=benchmark.get("nccl_lib_dir"),
    )
    start_total = time.monotonic()
    result: dict[str, Any] = {
        "schema": 1,
        "mode": "dynamic-compile" if args.build_only else "compile-and-tune",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git": git_revision(REPO_ROOT),
        "selection_metric": benchmark["selection_metric"],
        "selection_policy": copy.deepcopy(plan["selection"]),
        "invariants": copy.deepcopy(plan["invariants"]),
        "jit": copy.deepcopy(plan["jit"]),
        "benchmark": copy.deepcopy(benchmark),
        "phase_wall_seconds": {},
        "candidates": [
            {**candidate, "digest": candidate_digest(candidate["config"]),
             "status": "building"}
            for candidate in candidates
        ],
    }
    write_result(output_path, result)
    start_compile = time.monotonic()
    builds = compile_in_parallel(
        candidates, compiler, log_dir, compile_candidate
    )
    result["phase_wall_seconds"]["compile"] = time.monotonic() - start_compile
    for item in result["candidates"]:
        item["build"] = builds[item["name"]]
        item["generated_header"] = item["build"].get("generated_header")
        item["status"] = ("built" if item["build"]["returncode"] == 0
                          else "build_failed")
    if args.build_only:
        result["status"] = (
            "complete" if all(item["status"] == "built"
                              for item in result["candidates"])
            else "build_failed"
        )
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["timing"] = timing_summary(result, result["elapsed_seconds"])
        write_result(output_path, result)
        print(f"dynamic compile result: {output_path}")
        return 0 if result["status"] == "complete" else 1

    candidate_by_name = {item["name"]: item for item in candidates}
    reference_fingerprint = None
    start_preflight = time.monotonic()
    for ordinal, item in enumerate(result["candidates"], 1):
        if item["status"] != "built":
            continue
        candidate = candidate_by_name[item["name"]]
        print(f"[preflight {ordinal}/{len(candidates)}] {item['name']}",
              flush=True)
        preflight = preflight_candidate(
            candidate, Path(item["build"]["binary"]), benchmark, log_dir,
            environment,
        )
        item["preflight"] = preflight
        if preflight["returncode"] != 0 or preflight["record"] is None:
            item["status"] = "preflight_failed"
            write_result(output_path, result)
            continue
        fingerprint = correctness_fingerprint(preflight["record"])
        item["correctness_fingerprint"] = fingerprint
        if reference_fingerprint is None:
            reference_fingerprint = fingerprint
        if fingerprint != reference_fingerprint:
            item["status"] = "correctness_mismatch"
        else:
            item["status"] = "preflight_valid"
        write_result(output_path, result)
    result["phase_wall_seconds"]["preflight"] = (
        time.monotonic() - start_preflight)

    start_tune = time.monotonic()
    benchmarkable = [item for item in result["candidates"]
                     if item["status"] == "preflight_valid"]
    for ordinal, item in enumerate(benchmarkable, 1):
        candidate = candidate_by_name[item["name"]]
        print(f"[tune {ordinal}/{len(benchmarkable)}] {item['name']}: "
              f"{benchmark['tune_rounds']} rounds", flush=True)
        tune = benchmark_candidate(
            candidate, Path(item["build"]["binary"]), benchmark, log_dir,
            environment, rounds=int(benchmark["tune_rounds"]),
        )
        item["tune"] = tune
        if tune["returncode"] != 0 or tune["record"] is None:
            item["status"] = "benchmark_failed"
            write_result(output_path, result)
            continue
        metric = metric_value(tune["record"], benchmark["selection_metric"])
        item["selection_metric_us"] = metric
        item["status"] = "valid"
        total = tune["record"]["total_slowest_rank_us"]
        compute = tune["record"]["compute_slowest_rank_us"]
        print(f"  total mean={total['mean']:.3f} us "
              f"p95={total['p95']:.3f} us; "
              f"compute mean={compute['mean']:.3f} us", flush=True)
        write_result(output_path, result)
    result["phase_wall_seconds"]["tune"] = time.monotonic() - start_tune

    valid = [item for item in result["candidates"]
             if item["status"] == "valid"]
    if not valid:
        result["status"] = "no_valid_candidate"
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["timing"] = timing_summary(result, result["elapsed_seconds"])
        write_result(output_path, result)
        return 1

    selection = plan["selection"]
    incumbent_name = candidates[0]["name"]
    finalists = confirmation_finalists(
        valid, incumbent_name,
        min(int(selection["confirmation_candidates"]), len(valid)),
    )
    result["selection_finalists"] = [item["name"] for item in finalists]
    for item in finalists:
        item["confirmations"] = []
        item["confirmation_status"] = "running"
    start_confirmation = time.monotonic()
    repeats = int(selection["confirmation_repeats"])
    for repeat in range(repeats):
        ordered = finalists if repeat % 2 == 0 else list(reversed(finalists))
        for item in ordered:
            if item["confirmation_status"] != "running":
                continue
            candidate = candidate_by_name[item["name"]]
            print(f"[confirm {repeat + 1}/{repeats}] {item['name']}: "
                  f"{selection['confirmation_rounds']} rounds", flush=True)
            confirmation = benchmark_candidate(
                candidate, Path(item["build"]["binary"]), benchmark, log_dir,
                environment, rounds=int(selection["confirmation_rounds"]),
                suffix=f"confirm{repeat + 1}",
            )
            item["confirmations"].append(confirmation)
            if (confirmation["returncode"] != 0 or
                    confirmation["record"] is None):
                item["confirmation_status"] = "benchmark_failed"
            else:
                confirmation["metric_us"] = metric_value(
                    confirmation["record"], benchmark["selection_metric"])
                print(f"  mean={confirmation['metric_us']:.3f} us",
                      flush=True)
            write_result(output_path, result)
    for item in finalists:
        if (item["confirmation_status"] != "running" or
                len(item["confirmations"]) != repeats):
            continue
        records = [run["record"] for run in item["confirmations"]]
        item["confirmation_metric_us"] = weighted_metric(
            records, benchmark["selection_metric"])
        item["aggregate_metric_us"] = weighted_metric(
            [item["tune"]["record"], *records],
            benchmark["selection_metric"],
        )
        item["confirmation_status"] = "confirmed"
    result["phase_wall_seconds"]["confirmation"] = (
        time.monotonic() - start_confirmation)
    confirmed = [item for item in finalists
                 if item.get("confirmation_status") == "confirmed"]
    if not confirmed:
        result["status"] = "no_confirmed_candidate"
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["timing"] = timing_summary(result, result["elapsed_seconds"])
        write_result(output_path, result)
        return 1

    winner, decision = choose_confirmed_candidate(
        confirmed, incumbent_name,
        float(selection["minimum_improvement_percent"]),
    )
    result["selection_decision"] = decision
    result["selected"] = {
        "name": winner["name"], "digest": winner["digest"],
        "config": winner["config"],
        "selection_metric_us": winner["aggregate_metric_us"],
        "binary": winner["build"]["binary"],
    }
    if not args.skip_final:
        print(f"[final] {winner['name']}: {benchmark['final_rounds']} rounds",
              flush=True)
        start_final = time.monotonic()
        final = benchmark_candidate(
            candidate_by_name[winner["name"]], Path(winner["build"]["binary"]),
            benchmark, log_dir, environment,
            rounds=int(benchmark["final_rounds"]), suffix="final",
        )
        result["phase_wall_seconds"]["final"] = time.monotonic() - start_final
        result["selected"]["final"] = final
        if final["returncode"] != 0 or final["record"] is None:
            result["status"] = "final_validation_failed"
            result["elapsed_seconds"] = time.monotonic() - start_total
            result["timing"] = timing_summary(result, result["elapsed_seconds"])
            write_result(output_path, result)
            return 1
        result["selected"]["final_metric_us"] = metric_value(
            final["record"], benchmark["selection_metric"])

    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - start_total
    result["timing"] = timing_summary(result, result["elapsed_seconds"])
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_result(output_path, result)
    print(f"selected {winner['name']} at "
          f"{winner['aggregate_metric_us']:.3f} us ({decision['reason']}); "
          f"result {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlanError, JitError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)
