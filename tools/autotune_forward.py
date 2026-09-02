#!/usr/bin/env python3
"""Compile and tune explicit MegaMOE Forward specializations.

The search plan is JSON rather than a collection of shell environment
variables.  Each candidate gets a generated header and a distinct build
directory, so its hot-path decisions are compile-time constants and its binary
can be reproduced directly from the result artifact.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "forward_h200_autotune.json"
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
    "main_hidden": "MEGAMOE_FWD_MAIN_HIDDEN",
    "main_intermediate": "MEGAMOE_FWD_MAIN_INTERMEDIATE",
    "tile_m": "MEGAMOE_FWD_TILE_M",
    "fc1_tile_n": "MEGAMOE_FWD_FC1_TILE_N",
    "fc2_tile_n": "MEGAMOE_FWD_FC2_TILE_N",
    "tile_k": "MEGAMOE_FWD_TILE_K",
    "fc1_stages": "MEGAMOE_FWD_FC1_STAGES",
    "fc2_stages": "MEGAMOE_FWD_FC2_STAGES",
    "compute_threads": "MEGAMOE_FWD_COMPUTE_THREADS",
    "math_warpgroups": "MEGAMOE_FWD_MATH_WARPGROUPS",
    "producer_warpgroups": "MEGAMOE_FWD_PRODUCER_WARPGROUPS",
    "math_registers": "MEGAMOE_FWD_MATH_REGISTERS",
    "producer_registers": "MEGAMOE_FWD_PRODUCER_REGISTERS",
    "wgmma_wait_group": "MEGAMOE_FWD_WGMMA_WAIT_GROUP",
    "fc2_store_atom_n": "MEGAMOE_FWD_FC2_STORE_ATOM_N",
    "cluster_size": "MEGAMOE_FWD_CLUSTER_SIZE",
    "fc2_tma_store": "MEGAMOE_FWD_FC2_TMA_STORE",
    "rdma_ctas": "MEGAMOE_FWD_RDMA_CTAS",
    "relay_ctas": "MEGAMOE_FWD_RELAY_CTAS",
    "rdma_control_warps": "MEGAMOE_FWD_RDMA_CONTROL_WARPS",
    "relay_tma_warps": "MEGAMOE_FWD_RELAY_TMA_WARPS",
    "local_dispatch_tma_warps": "MEGAMOE_FWD_LOCAL_DISPATCH_TMA_WARPS",
    "relay_scatter_warps": "MEGAMOE_FWD_RELAY_SCATTER_WARPS",
    "local_return_batch_mblocks":
        "MEGAMOE_FWD_LOCAL_RETURN_BATCH_MBLOCKS",
    "dispatch_publish_batch_rows":
        "MEGAMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS",
    "return_chunk_rows": "MEGAMOE_FWD_RETURN_CHUNK_ROWS",
    "return_max_inflight": "MEGAMOE_FWD_RETURN_MAX_INFLIGHT",
    "registered_output": "MEGAMOE_FWD_REGISTERED_OUTPUT",
    "relay_local_return": "MEGAMOE_FWD_RELAY_LOCAL_RETURN",
    "local_route_batch_rows": "MEGAMOE_FWD_LOCAL_ROUTE_BATCH_ROWS",
    "startup_mblocks": "MEGAMOE_FWD_STARTUP_MBLOCKS",
    "fc1_locality_group": "MEGAMOE_FWD_FC1_LOCALITY_GROUP",
    "fc1_locality_group4_begin":
        "MEGAMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN",
    "fc1_locality_group4_end": "MEGAMOE_FWD_FC1_LOCALITY_GROUP4_END",
    "fc2_priority_burst": "MEGAMOE_FWD_FC2_PRIORITY_BURST",
    "task_claim_chunk": "MEGAMOE_FWD_TASK_CLAIM_CHUNK",
    "task_idle_nanosleep": "MEGAMOE_FWD_TASK_IDLE_NANOSLEEP",
    "transport_idle_nanosleep": "MEGAMOE_FWD_TRANSPORT_IDLE_NANOSLEEP",
    "setup_nanosleep": "MEGAMOE_FWD_SETUP_NANOSLEEP",
    "completion_chunks_per_pass":
        "MEGAMOE_FWD_COMPLETION_CHUNKS_PER_PASS",
    "completion_marker_warps": "MEGAMOE_FWD_COMPLETION_MARKER_WARPS",
    "reduce_cta_divisor": "MEGAMOE_FWD_REDUCE_CTA_DIVISOR",
    "reduce_scheduler": "MEGAMOE_FWD_REDUCE_SCHEDULER",
    "reduce_warps": "MEGAMOE_FWD_REDUCE_WARPS",
    "reduce_tokens_per_claim": "MEGAMOE_FWD_REDUCE_TOKENS_PER_CLAIM",
    "reduce_registers": "MEGAMOE_FWD_REDUCE_REGISTERS",
    "return_idle_nanosleep": "MEGAMOE_FWD_RETURN_IDLE_NANOSLEEP",
    "reduce_queue_nanosleep": "MEGAMOE_FWD_REDUCE_QUEUE_NANOSLEEP",
    "finalize_nanosleep": "MEGAMOE_FWD_FINALIZE_NANOSLEEP",
}

BOOLEAN_PARAMETERS = {
    "fc2_tma_store",
    "registered_output",
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


class PlanError(ValueError):
    """An explicit tuning plan is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def _require_choice(config: Mapping[str, Any], key: str,
                    choices: Iterable[int]) -> None:
    choices = tuple(choices)
    _require(config[key] in choices,
             f"{key}={config[key]!r}, expected one of {choices}")


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

    _require(config["main_hidden"] == 7168,
             "this Forward specialization requires main_hidden=7168")
    _require(config["main_intermediate"] == 3072,
             "this Forward specialization requires main_intermediate=3072")
    _require_choice(config, "tile_m", (64, 128))
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
    _require_choice(config, "cluster_size", (1, 2))
    if config["fc2_tma_store"]:
        _require(config["tile_m"] == 128,
                 "FC2 TMA store currently requires tile_m=128")
        _require(config["fc1_stages"] == 4 and config["fc2_stages"] == 4,
                 "FC2 TMA store currently requires the validated 4x4 stages")
    if config["cluster_size"] == 2:
        _require(config["fc2_tma_store"],
                 "cluster_size=2 requires the paired FC2 TMA-store kernel")

    # RDMA is an explicit singleton for this 8-rail topology; Relay remains a
    # narrow performance axis.
    _require(config["rdma_ctas"] == 8,
             "the current two-host topology contract requires rdma_ctas=8")
    _require_choice(config, "relay_ctas", (8, 16, 24))
    _require(config["relay_ctas"] % config["rdma_ctas"] == 0,
             "relay_ctas must be a multiple of rdma_ctas")
    _require(config["rdma_control_warps"] == 2,
             "the current RDMA protocol has exactly issue and completion warps")
    _require_choice(config, "relay_tma_warps", (0, 4, 8, 12))
    _require_choice(config, "local_dispatch_tma_warps", (4, 8, 12))
    _require_choice(config, "relay_scatter_warps", (4, 8, 12))
    _require_choice(config, "local_return_batch_mblocks", (1, 2, 4, 8))
    _require_choice(config, "dispatch_publish_batch_rows",
                    (4, 8, 16, 32, 64, 128))
    _require_choice(config, "return_chunk_rows", (128, 256, 512))
    _require(config["return_chunk_rows"] % 128 == 0,
             "return chunks must preserve the padded M128 contract")
    _require_choice(config, "return_max_inflight", (0, 4, 8, 16, 32, 64))
    _require(not config["fc2_tma_store"] or config["registered_output"],
             "GIN FC2 TMA store requires registered_output=true")

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

    _require_choice(config, "completion_chunks_per_pass", (1, 2, 4))
    _require_choice(config, "completion_marker_warps", (1, 2, 4))
    _require_choice(config, "reduce_cta_divisor", (1, 2, 4))
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


def load_plan(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlanError(f"cannot load {path}: {error}") from error
    _require(isinstance(document, dict), "top-level plan must be an object")
    _require(document.get("schema") == 1, "plan schema must be 1")
    for section in ("build", "benchmark", "fixed", "candidates"):
        _require(section in document, f"plan omits {section}")
    _require(isinstance(document["build"], dict), "build must be an object")
    _require(isinstance(document["benchmark"], dict),
             "benchmark must be an object")
    _require(isinstance(document["candidates"], list) and
             document["candidates"], "candidates must be a non-empty array")
    maximum = document.get("max_candidates", 8)
    _require(type(maximum) is int and 1 <= maximum <= 16,
             "max_candidates must be in [1, 16]")
    _require(len(document["candidates"]) <= maximum,
             f"candidate count exceeds max_candidates={maximum}")
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


def candidate_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def generated_header(name: str, config: Mapping[str, Any]) -> str:
    validate_tuning(config)
    digest = candidate_digest(config)
    lines = [
        "#pragma once",
        "",
        "// Generated by tools/autotune_forward.py.",
        f"// candidate={name} config_sha256_prefix={digest}",
        "#define MEGAMOE_FWD_COMPILED_TUNING 1",
        "#define MEGAMOE_FWD_ENABLE_LEGACY_ENV 0",
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
    header = include_dir / "megamoe_forward_tuning_generated.hpp"
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
        "MEGAMOE_INTEGRATED_TOP_K",
        "MEGAMOE_P2P_TOP_K",
        "MEGAMOE_P2P_BENCH_TOKENS",
        "MEGAMOE_FORWARD_BENCH_ROUNDS",
        "MEGAMOE_FORWARD_BENCH_GENERATION_BASE",
    }
    for key in tuple(environment):
        if (key.startswith("MEGAMOE_FORWARD_") or
                key.startswith("MEGAMOE_FWD_") or key in explicit_removals):
            environment.pop(key, None)
    return environment


def _run_to_log(command: list[str], log_path: Path, *, timeout: int,
                environment: Mapping[str, str]) -> tuple[float, int]:
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND " + json.dumps(command) + "\n")
        log.flush()
        try:
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=dict(environment),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            return time.monotonic() - start, completed.returncode
        except subprocess.TimeoutExpired:
            log.write(f"\nTIMEOUT after {timeout}s\n")
            return time.monotonic() - start, 124


def compile_candidate(candidate: Mapping[str, Any], paths: Mapping[str, Path],
                      build: Mapping[str, Any], log_dir: Path,
                      environment: Mapping[str, str]) -> dict[str, Any]:
    binary = paths["build_dir"] / str(build["target"])
    existed = binary.exists()
    command = [
        "make",
        f"-j{int(build['jobs'])}",
        f"BUILD_DIR={paths['build_dir']}",
        f"FORWARD_TUNING_INCLUDE_DIR={paths['include_dir']}",
        f"CUDA_ARCH_FLAGS={build['cuda_arch_flags']}",
        str(binary),
    ]
    log_path = log_dir / f"{candidate['name']}.build.log"
    elapsed, returncode = _run_to_log(
        command, log_path, timeout=int(build["timeout_seconds"]),
        environment=environment)
    result = {
        "command": command,
        "elapsed_seconds": elapsed,
        "returncode": returncode,
        "binary": str(binary),
        "binary_preexisted": existed,
        "log": str(log_path),
    }
    if returncode == 0:
        resources = inspect_forward_kernel(
            binary, log_dir / f"{candidate['name']}.resources.log",
            candidate["config"])
        result["resources"] = resources
        if (resources.get("available") and
                not resources.get("single_compiled_split_kernel")):
            result["returncode"] = 125
            result["resource_validation_error"] = (
                "expected exactly one compiled split Forward kernel")
    return result


def launcher_command(binary: Path, benchmark: Mapping[str, Any],
                     *, rounds: int) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "tests" / "run_multinode_nccl.py"),
        "--binary", str(binary),
        "--remote", str(benchmark["remote"]),
        "--port", str(int(benchmark["port"])),
        "--remote-binary", str(binary),
        "--gpus-per-node", str(int(benchmark["gpus_per_node"])),
        "--scenario", "gin_forward_bench",
        "--timeout", str(int(benchmark["timeout_seconds"])),
    ]
    nccl_lib_dir = benchmark.get("nccl_lib_dir")
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


def extract_benchmark_json(output: str) -> dict[str, Any]:
    prefix = "GIN_BENCH_JSON "
    payloads = [line[len(prefix):] for line in output.splitlines()
                if line.startswith(prefix)]
    if len(payloads) != 1:
        raise RuntimeError(
            f"launcher emitted {len(payloads)} GIN_BENCH_JSON records")
    result = json.loads(payloads[0])
    if result.get("benchmark") != "multinode-gin-forward-main-shape":
        raise RuntimeError(f"unexpected benchmark record: {result.get('benchmark')}")
    return result


def benchmark_candidate(candidate: Mapping[str, Any], binary: Path,
                        benchmark: Mapping[str, Any], log_dir: Path,
                        environment: Mapping[str, str], *, rounds: int,
                        suffix: str = "tune") -> dict[str, Any]:
    command = launcher_command(binary, benchmark, rounds=rounds)
    run_environment = dict(environment)
    run_environment["MEGAMOE_P2P_BENCH_TOKENS"] = str(
        int(benchmark["tokens_per_rank"]))
    run_environment["MEGAMOE_FORWARD_BENCH_ROUNDS"] = str(rounds)
    run_environment["MEGAMOE_FORWARD_BENCH_GENERATION_BASE"] = "1000"
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
        output = (error.stdout or "") + f"\nTIMEOUT: {error}\n"
        returncode = 124
    elapsed = time.monotonic() - start
    log_path.write_text("COMMAND " + json.dumps(command) + "\n" + output,
                        encoding="utf-8")
    record = None
    parse_error = None
    if returncode == 0:
        try:
            record = extract_benchmark_json(output)
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


def metric_value(record: Mapping[str, Any], dotted_name: str) -> float:
    value: Any = record
    for component in dotted_name.split("."):
        if not isinstance(value, dict) or component not in value:
            raise PlanError(f"metric {dotted_name} is absent from benchmark JSON")
        value = value[component]
    if not isinstance(value, (int, float)):
        raise PlanError(f"metric {dotted_name} is not numeric")
    return float(value)


def correctness_fingerprint(record: Mapping[str, Any]) -> str:
    missing = [key for key in CORRECTNESS_KEYS if key not in record]
    if missing:
        raise RuntimeError(f"benchmark JSON omits correctness keys: {missing}")
    selected = {key: record[key] for key in CORRECTNESS_KEYS}
    return hashlib.sha256(json.dumps(
        selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_revision() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        check=False).stdout.strip()
    return {"commit": revision, "dirty": bool(dirty)}


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _configure_from_cli(plan: dict[str, Any], args: argparse.Namespace) -> None:
    build = plan["build"]
    benchmark = plan["benchmark"]
    if args.jobs is not None:
        build["jobs"] = args.jobs
    if args.remote is not None:
        benchmark["remote"] = args.remote
    if args.port is not None:
        benchmark["port"] = args.port
    if args.nccl_lib_dir is not None:
        benchmark["nccl_lib_dir"] = args.nccl_lib_dir
    if args.tune_rounds is not None:
        benchmark["tune_rounds"] = args.tune_rounds
    if args.final_rounds is not None:
        benchmark["final_rounds"] = args.final_rounds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    parser.add_argument("--remote")
    parser.add_argument("--port", type=int)
    parser.add_argument("--nccl-lib-dir")
    parser.add_argument("--tune-rounds", type=int)
    parser.add_argument("--final-rounds", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require(not (args.generate_only and args.build_only),
             "--generate-only and --build-only are mutually exclusive")
    _require(not args.build_only or args.candidate,
             "--build-only requires at least one explicit --candidate")
    config_path = args.config.resolve()
    plan = load_plan(config_path)
    _configure_from_cli(plan, args)
    candidates = expand_candidates(plan)
    requested = set(args.candidate)
    if requested:
        known = {candidate["name"] for candidate in candidates}
        unknown = requested - known
        _require(not unknown, f"unknown requested candidates: {sorted(unknown)}")
        candidates = [candidate for candidate in candidates
                      if candidate["name"] in requested]
    _require(candidates, "candidate filter selected no configurations")

    build = plan["build"]
    benchmark = plan["benchmark"]
    build_root = _resolve_repo_path(str(build["build_root"]))
    result_dir = _resolve_repo_path(str(benchmark["result_dir"]))
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = (args.output.resolve() if args.output else
                   result_dir / f"forward_h200_{timestamp}.json")
    log_dir = output_path.parent / (output_path.stem + "-logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    materialized = {
        candidate["name"]: materialize_candidate(candidate, build_root)
        for candidate in candidates
    }
    if args.generate_only:
        manifest = {
            "schema": 1,
            "mode": "generate-only",
            "config_path": str(config_path),
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
        start_build_only = time.monotonic()
        manifest: dict[str, Any] = {
            "schema": 1,
            "mode": "dynamic-compile",
            "config_path": str(config_path),
            "git": _git_revision(),
            "candidates": [],
        }
        all_valid = True
        for ordinal, candidate in enumerate(candidates, 1):
            print(f"[{ordinal}/{len(candidates)}] compile {candidate['name']}",
                  flush=True)
            paths = materialized[candidate["name"]]
            build_result = compile_candidate(candidate, paths, build, log_dir,
                                             environment)
            valid = build_result["returncode"] == 0
            all_valid = all_valid and valid
            manifest["candidates"].append({
                **candidate,
                "digest": candidate_digest(candidate["config"]),
                "generated_header": str(paths["header"]),
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
    result: dict[str, Any] = {
        "schema": 1,
        "mode": "compile-and-tune",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config_path": str(config_path),
        "git": _git_revision(),
        "selection_metric": benchmark["selection_metric"],
        "build": copy.deepcopy(build),
        "benchmark": copy.deepcopy(benchmark),
        "candidates": [],
    }
    reference_fingerprint = None
    for ordinal, candidate in enumerate(candidates, 1):
        print(f"[{ordinal}/{len(candidates)}] compile {candidate['name']}",
              flush=True)
        paths = materialized[candidate["name"]]
        candidate_result: dict[str, Any] = {
            **candidate,
            "digest": candidate_digest(candidate["config"]),
            "generated_header": str(paths["header"]),
            "status": "building",
        }
        result["candidates"].append(candidate_result)
        build_result = compile_candidate(candidate, paths, build, log_dir,
                                         environment)
        candidate_result["build"] = build_result
        if build_result["returncode"] != 0:
            candidate_result["status"] = "build_failed"
            print(f"  build failed; see {build_result['log']}", flush=True)
            _write_result(output_path, result)
            continue
        print(f"  benchmark {int(benchmark['tune_rounds'])} rounds", flush=True)
        bench_result = benchmark_candidate(
            candidate, Path(build_result["binary"]), benchmark, log_dir,
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

    valid = [item for item in result["candidates"]
             if item["status"] == "valid"]
    if not valid:
        result["elapsed_seconds"] = time.monotonic() - start_total
        result["timing"] = {
            "build_seconds": sum(item["build"]["elapsed_seconds"]
                                 for item in result["candidates"]),
            "tune_benchmark_seconds": sum(
                item.get("tune", {}).get("elapsed_seconds", 0.0)
                for item in result["candidates"]),
            "final_benchmark_seconds": 0.0,
            "total_seconds": result["elapsed_seconds"],
        }
        result["status"] = "no_valid_candidate"
        _write_result(output_path, result)
        print(f"no valid candidate; partial result: {output_path}", file=sys.stderr)
        return 1
    winner = min(valid, key=lambda item: item["selection_metric_us"])
    result["selected"] = {
        "name": winner["name"],
        "digest": winner["digest"],
        "config": winner["config"],
        "selection_metric_us": winner["selection_metric_us"],
        "binary": winner["build"]["binary"],
    }

    if not args.skip_final:
        print(f"final validation {winner['name']}: "
              f"{int(benchmark['final_rounds'])} rounds", flush=True)
        winner_candidate = next(candidate for candidate in candidates
                                if candidate["name"] == winner["name"])
        final_result = benchmark_candidate(
            winner_candidate, Path(winner["build"]["binary"]), benchmark,
            log_dir, environment, rounds=int(benchmark["final_rounds"]),
            suffix="final")
        result["selected"]["final"] = final_result
        if final_result["returncode"] != 0 or final_result["record"] is None:
            result["status"] = "final_validation_failed"
            result["elapsed_seconds"] = time.monotonic() - start_total
            _write_result(output_path, result)
            print(f"final validation failed; see {final_result['log']}",
                  file=sys.stderr)
            return 1
        if correctness_fingerprint(final_result["record"]) != reference_fingerprint:
            result["status"] = "final_correctness_mismatch"
            result["elapsed_seconds"] = time.monotonic() - start_total
            _write_result(output_path, result)
            print("final validation correctness fingerprint changed",
                  file=sys.stderr)
            return 1
        result["selected"]["final_metric_us"] = metric_value(
            final_result["record"], str(benchmark["selection_metric"]))

    result["status"] = "complete"
    result["elapsed_seconds"] = time.monotonic() - start_total
    result["timing"] = {
        "build_seconds": sum(item["build"]["elapsed_seconds"]
                             for item in result["candidates"]),
        "tune_benchmark_seconds": sum(
            item.get("tune", {}).get("elapsed_seconds", 0.0)
            for item in result["candidates"]),
        "final_benchmark_seconds": result["selected"].get(
            "final", {}).get("elapsed_seconds", 0.0),
        "total_seconds": result["elapsed_seconds"],
    }
    result["completed_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_result(output_path, result)
    print(f"selected {winner['name']} at "
          f"{winner['selection_metric_us']:.3f} us; result {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlanError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        raise SystemExit(2)
