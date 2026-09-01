#!/usr/bin/env python3
"""Launch a two-host NCCL protocol test with explicit remote/network options."""

import argparse
import base64
import json
import os
import re
import secrets
import select
import shlex
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceTopology,
    build_reference_all,
)


MASK64 = (1 << 64) - 1
BOOTSTRAP_BYTES = 128
_active_bootstrap = b""


@dataclass(frozen=True)
class RankOutput:
    rank: int
    returncode: int
    stdout: str
    stderr: str


def parse_fields(stdout, marker):
    line = next((line for line in stdout.splitlines()
                 if line.startswith(marker)), None)
    if line is None:
        raise RuntimeError(f"rank output has no {marker.strip()}")
    fields = {}
    for item in line[len(marker):].split(" "):
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def sample_stats(values):
    if not values:
        raise RuntimeError("benchmark contains no measured samples")
    ordered = sorted(values)

    def percentile(fraction):
        ordinal = max(1, int(len(ordered) * fraction + 0.999999))
        return ordered[ordinal - 1]

    return {
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def aggregate_gin_benchmark(outputs, scenario):
    forward = scenario == "gin_forward_bench"
    marker = "FORWARD_BENCH " if forward else "BACKWARD_BENCH "
    records = [parse_fields(item.stdout, marker) for item in outputs]
    if forward:
        schedulers = {item.get("reduce_scheduler") for item in records}
        if len(schedulers) != 1 or None in schedulers:
            raise RuntimeError(
                "Forward ranks disagree on reduce scheduler: " +
                ",".join(sorted(str(item) for item in schedulers)))
    stage_keys = (("wall", "notify", "forward", "dispatch", "fc1", "fc2",
                   "return", "reduce", "fc1_fc2_overlap",
                   "fc2_return_overlap", "return_reduce_overlap",
                   "terminal_status_agreement") if forward
                  else
                  ("notify_inclusive_training_step_wall",
                   "post_notify_training_step_wall", "notify_wall",
                   "forward_wall", "backward_wall", "dy_dispatch", "dv",
                   "dgate", "transpose", "dx", "dw1", "dw2",
                   "v3_local_delivery_complete",
                   "v3_outbound_pack_complete", "v3_rdma_complete",
                   "v3_inbound_relay_complete", "v3_dv_first_start",
                   "v3_dv_complete", "v3_dw2_first_ready",
                   "v3_dw2_first_start", "v3_dw2_last_complete",
                   "v3_terminal_complete",
                   "dw1_ready_wait", "dw2_ready_wait",
                   "dw1_max_expert_wait", "dw2_max_expert_wait",
                   "dw1_tail", "dw2_tail", "dw1_overlap_dv_dx",
                   "dw2_overlap_dv_dx", "reduce", "forward_call_host",
                   "dy_ready_host", "backward_call_host",
                   "integration_overhead", "handoff", "backward_setup_host",
                   "forward_consumer_drain", "final_drain",
                   "forward_terminal_status_agreement",
                   "backward_terminal_status_agreement",
                   "notify_pool_acquire", "notify_local_histogram_wall",
                   "notify_local_mapping_device",
                   "notify_collective_enqueue", "notify_collective_wait",
                   "notify_materialization", "notify_publish"))
    measured = None
    for rank, record in enumerate(records):
        if record.get("backend") != "gin-rdma":
            raise RuntimeError(
                f"rank={rank} benchmark did not run GinRdma backend")
        for key in stage_keys:
            samples = [float(value) for value in record[key].split(",")]
            record[key] = samples
            if measured is None:
                measured = len(samples)
            elif len(samples) != measured:
                raise RuntimeError(
                    f"rank={rank} stage={key} sample count={len(samples)} "
                    f"expected={measured}")
    summary = {}
    for key in stage_keys:
        slowest = [max(record[key][iteration] for record in records)
                   for iteration in range(measured)]
        summary[key + "_slowest_rank_us"] = sample_stats(slowest)
    result = {
        "schema": 1,
        "benchmark": ("multinode-gin-forward-main-shape" if forward else
                      "multinode-gin-forward-backward-main-shape"),
        "backend": "gin-rdma-device-persistent",
        "world": len(records),
        "warmup": int(records[0].get("warmup", "10")),
        "measured": measured,
        "shape": {
            "tokens_per_rank": int(records[0]["tokens"]),
            "hidden": int(records[0]["hidden"]),
            "intermediate": int(records[0]["intermediate"]),
            "experts": int(records[0]["experts"]),
            "top_k": int(records[0]["top_k"]),
        },
        "per_rank_activation_hash": [int(item["activation_hash"])
                                     for item in records],
        "per_rank_route_hash": [int(item["route_hash"])
                                for item in records],
        "per_rank_gate_hash": [int(item["gate_hash"])
                               for item in records],
        "per_rank_actual_w1_device_hash": [
            [int(item["actual_w1_hash_low"]),
             int(item["actual_w1_hash_high"])] for item in records],
        "per_rank_actual_w2_device_hash": [
            [int(item["actual_w2_hash_low"]),
             int(item["actual_w2_hash_high"])] for item in records],
        **summary,
    }
    if forward:
        result.update({
            "reduce_scheduler": records[0]["reduce_scheduler"],
            "per_rank_device_pool_bytes": [int(item["device_pool_bytes"])
                                           for item in records],
            "per_rank_compute_pool_bytes": [int(item["compute_pool_bytes"])
                                            for item in records],
            "per_rank_gin_registered_bytes": [
                int(item["gin_registered_bytes"]) for item in records],
            "per_rank_pinned_pool_bytes": [int(item["pinned_pool_bytes"])
                                           for item in records],
            "per_rank_pool_events": [int(item["pool_events"])
                                     for item in records],
            "per_rank_weight_bytes": [int(item["weight_bytes"])
                                      for item in records],
            "per_rank_dispatch_peer_bytes": [
                int(item["dispatch_peer_bytes"]) for item in records],
            "per_rank_return_peer_bytes": [
                int(item["return_peer_bytes"]) for item in records],
            "per_rank_gin_dispatch_puts": [int(item["gin_dispatch_puts"])
                                           for item in records],
            "per_rank_raw_remote_route_bytes": [
                int(item["raw_remote_route_bytes"]) for item in records],
            "per_rank_gin_dispatch_payload_bytes": [
                int(item["gin_dispatch_payload_bytes"]) for item in records],
            "per_rank_gin_return_payload_bytes": [
                int(item["gin_return_payload_bytes"]) for item in records],
            "per_rank_gin_return_puts": [int(item["gin_return_puts"])
                                         for item in records],
            "per_rank_gin_signals": [int(item["gin_signals"])
                                     for item in records],
            "per_rank_gin_metadata_bytes": [
                int(item["gin_metadata_bytes"]) for item in records],
            "per_rank_output_hash": [int(item["output_hash"])
                                     for item in records],
        })
    else:
        result.update({
            "per_rank_workspace_bytes": [int(item["workspace"])
                                         for item in records],
            "per_rank_compute_pool_bytes": [int(item["compute_pool_bytes"])
                                            for item in records],
            "per_rank_gin_registered_bytes": [
                int(item["gin_registered_bytes"]) for item in records],
            "per_rank_pinned_pool_bytes": [int(item["pinned_pool_bytes"])
                                           for item in records],
            "per_rank_pool_events": [int(item["pool_events"])
                                     for item in records],
            "per_rank_forward_dispatch_peer_bytes": [
                int(item["forward_dispatch_peer_bytes"])
                for item in records],
            "per_rank_forward_return_peer_bytes": [
                int(item["forward_return_peer_bytes"]) for item in records],
            "per_rank_forward_gin_dispatch_puts": [
                int(item["forward_gin_dispatch_puts"]) for item in records],
            "per_rank_forward_raw_remote_route_bytes": [
                int(item["forward_raw_remote_route_bytes"])
                for item in records],
            "per_rank_forward_gin_dispatch_payload_bytes": [
                int(item["forward_gin_dispatch_payload_bytes"])
                for item in records],
            "per_rank_forward_gin_return_payload_bytes": [
                int(item["forward_gin_return_payload_bytes"])
                for item in records],
            "per_rank_forward_gin_return_puts": [
                int(item["forward_gin_return_puts"]) for item in records],
            "per_rank_forward_gin_signals": [
                int(item["forward_gin_signals"]) for item in records],
            "per_rank_forward_gin_metadata_bytes": [
                int(item["forward_gin_metadata_bytes"]) for item in records],
            "per_rank_backward_dy_peer_bytes": [
                int(item["backward_dy_peer_bytes"]) for item in records],
            "per_rank_backward_dx_peer_bytes": [
                int(item["backward_dx_peer_bytes"]) for item in records],
            "per_rank_backward_gin_dispatch_puts": [
                int(item["backward_gin_dispatch_puts"])
                for item in records],
            "per_rank_backward_raw_remote_dy_bytes": [
                int(item["backward_raw_remote_dy_bytes"])
                for item in records],
            "per_rank_backward_gin_dy_payload_bytes": [
                int(item["backward_gin_dy_payload_bytes"])
                for item in records],
            "per_rank_backward_gin_dx_payload_bytes": [
                int(item["backward_gin_dx_payload_bytes"])
                for item in records],
            "per_rank_backward_gin_return_puts": [
                int(item["backward_gin_return_puts"]) for item in records],
            "per_rank_backward_gin_signals": [
                int(item["backward_gin_signals"]) for item in records],
            "per_rank_backward_gin_metadata_bytes": [
                int(item["backward_gin_metadata_bytes"])
                for item in records],
            "per_rank_output_gradient_hash": [
                int(item["output_gradient_hash"]) for item in records],
            "per_rank_actual_dx_device_hash": [
                [int(item["actual_dx_hash_low"]),
                 int(item["actual_dx_hash_high"])] for item in records],
            "per_rank_actual_dw1_device_hash": [
                [int(item["actual_dw1_hash_low"]),
                 int(item["actual_dw1_hash_high"])] for item in records],
            "per_rank_actual_dw2_device_hash": [
                [int(item["actual_dw2_hash_low"]),
                 int(item["actual_dw2_hash_high"])] for item in records],
            "per_rank_actual_dgate_device_hash": [
                [int(item["actual_dgate_hash_low"]),
                 int(item["actual_dgate_hash_high"])] for item in records],
            "per_rank_notify_collective_audit": [{
                "fused": int(item["notify_fused_enqueues"]),
                "control": int(item["notify_control_enqueues"]),
                "config": int(item["notify_config_enqueues"]),
                "count": int(item["notify_count_enqueues"]),
                "summary": int(item["notify_summary_enqueues"]),
            } for item in records],
        })
    return result


def scrub_output(value, bootstrap=None):
    """Remove bootstrap binary/text encodings from any aggregated diagnostic."""
    secret = _active_bootstrap if bootstrap is None else bytes(bootstrap)
    if isinstance(value, bytes):
        raw = value
        if secret:
            raw = raw.replace(secret, b"[REDACTED_SECRET]")
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(value)
    # Defense in depth for the retired stdout protocol, even before enough
    # bytes are known to derive its exact representation.
    text = re.sub(
        r"(?im)^BOOTSTRAP[ \t]+(?:[0-9a-f]{2})+[ \t]*$",
        "[REDACTED_SECRET]",
        text,
    )
    if secret:
        # Match the actual hexadecimal value case-insensitively per character.
        # Escaping the value keeps secret-derived text out of regex syntax.
        text = re.sub(
            re.escape(secret.hex()),
            "[REDACTED_SECRET]",
            text,
            flags=re.IGNORECASE,
        )
        representations = {
            base64.b64encode(secret).decode("ascii"),
            base64.urlsafe_b64encode(secret).decode("ascii"),
        }
        representations.update(item.rstrip("=") for item in tuple(representations))
        for representation in sorted(representations, key=len, reverse=True):
            if representation:
                text = text.replace(representation, "[REDACTED_SECRET]")
    return text


def seeded_inputs(world, experts, top_k, seed, max_tokens=10):
    def next_value():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        return seed

    routes = []
    for _rank in range(world):
        rank_routes = []
        for _token in range(next_value() % (max_tokens + 1)):
            candidates = list(range(experts))
            selected = []
            for slot in range(top_k):
                index = slot + next_value() % (experts - slot)
                candidates[slot], candidates[index] = candidates[index], candidates[slot]
                selected.append(candidates[slot])
            rank_routes.append(selected)
        routes.append(rank_routes)
    return routes


def config_for(gpus_per_node):
    nodes = 2
    world = nodes * gpus_per_node
    experts = max(16, world * 2)
    top_k = min(6, experts)
    return ReferenceConfig(
        experts,
        top_k,
        7,
        ReferenceTopology(
            world,
            nodes,
            [node for node in range(nodes) for _ in range(gpus_per_node)],
            [local for _node in range(nodes) for local in range(gpus_per_node)],
            [expert % world for expert in range(experts)],
        ),
    )


def runner_args(binary, rank, gpus_per_node, seed, scenario, launch_id,
                bootstrap_fd, produces_bootstrap=False):
    result = [
        binary,
        "--rank-runner",
        "--rank", str(rank),
        "--nodes", "2",
        "--gpus-per-node", str(gpus_per_node),
        "--local-rank", str(rank % gpus_per_node),
        "--seed", str(seed),
        "--scenario", scenario,
        "--launch-id", launch_id,
    ]
    result.extend([
        "--bootstrap-output-fd" if produces_bootstrap
        else "--bootstrap-input-fd",
        str(bootstrap_fd),
    ])
    return result


def environment_overrides(args):
    overrides = {"NCCL_DEBUG": args.nccl_debug}
    if args.socket_ifname:
        overrides["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    if args.disable_ib:
        overrides["NCCL_IB_DISABLE"] = "1"
    elif args.enable_ib:
        overrides["NCCL_IB_DISABLE"] = "0"
    # These test/benchmark controls are ordinary non-secret configuration.
    # Forward them explicitly so local and remote ranks cannot silently run a
    # different generation count, shape, or gate mode.
    for key in (
        "MEGAMOE_P2P_ROUNDS",
        "MEGAMOE_P2P_BENCH_ROUNDS",
        "MEGAMOE_P2P_BENCH_TOKENS",
        "MEGAMOE_FORWARD_BENCH_ROUNDS",
        "MEGAMOE_FORWARD_BENCH_GENERATION_BASE",
        "MEGAMOE_FORWARD_PROGRESS",
        "MEGAMOE_FORWARD_PROGRESS_DEADLINE_SECONDS",
        "MEGAMOE_BACKWARD_BENCH_ROUNDS",
        "MEGAMOE_BACKWARD_BENCH_TRACE",
        "MEGAMOE_BACKWARD_HASH_REPEATS",
        "MEGAMOE_BACKWARD_ROUTE_CASE",
        "MEGAMOE_BACKWARD_SMOKE_REPEATS",
        "MEGAMOE_FORWARD_SMOKE_REPEATS",
        "MEGAMOE_BACKWARD_SMOKE_TRACE",
        "MEGAMOE_BACKWARD_POOL2_VALIDATE",
        "MEGAMOE_BACKWARD_CAPTURE_HASH",
        "MEGAMOE_BACKWARD_STAGE_HASHES",
        "MEGAMOE_BACKWARD_PROGRESS_DUMP",
        "MEGAMOE_BACKWARD_SPLIT_DW1",
        "MEGAMOE_BACKWARD_DIAG_DW2",
        "MEGAMOE_BACKWARD_DIAG_DW2_128",
        "MEGAMOE_BACKWARD_DW2_128_STAGES",
        "MEGAMOE_BACKWARD_DIAG_V3_DV_DW2",
        "MEGAMOE_BACKWARD_DIAG_V3_VERIFY_ORACLE",
        "MEGAMOE_BACKWARD_PERSISTENT_V3",
        "MEGAMOE_BACKWARD_V3_RETURN_GROUP_ROWS",
        "MEGAMOE_BACKWARD_V3_WEIGHT_CTA_DIVISOR",
        "MEGAMOE_BACKWARD_V3_DV_PHASE_SLOTS",
        "MEGAMOE_BACKWARD_V3_DX_PHASE_SLOTS",
        "MEGAMOE_BACKWARD_V3_ACTIVATION_CTAS",
        "MEGAMOE_BACKWARD_V3_DW1_CTAS",
        "MEGAMOE_BACKWARD_V3_DW_TILE_N",
        "MEGAMOE_BACKWARD_V3_WEIGHT_CLUSTER2",
        "MEGAMOE_BACKWARD_V3_SPLIT_COMPUTE",
        "MEGAMOE_BACKWARD_V3_ELASTIC",
        "MEGAMOE_BACKWARD_V3_CLUSTER_COMPUTE",
        "MEGAMOE_BACKWARD_V3_ACTIVATION_MULTICAST",
        "MEGAMOE_BACKWARD_V3_CLUSTER_REDUCE_CTAS",
        "MEGAMOE_BACKWARD_V3_DIRECT_DY_INPUT",
        "MEGAMOE_BACKWARD_V3_REDUCE_CTAS",
        "MEGAMOE_BACKWARD_V3_DIRECT_RETURN_STAGING",
        "MEGAMOE_BACKWARD_V3_DIRECT_RETURN_CONSUME",
        "MEGAMOE_BACKWARD_V3_GENERIC_DY_COPY",
        "MEGAMOE_BACKWARD_V3_WAIT_FULL_DY_DISPATCH",
        "MEGAMOE_BACKWARD_V3_CLEAR_CANONICAL_DY",
        "MEGAMOE_BACKWARD_V3_SNAPSHOT_DY",
        "MEGAMOE_BACKWARD_V3_LOCALIZE_DY",
        "MEGAMOE_FORWARD_RETURN_CHUNK_ROWS",
        "MEGAMOE_FORWARD_DYNAMIC_REDUCE_QUEUE",
        "MEGAMOE_FORWARD_DYNAMIC_REDUCE_SCAN",
        "MEGAMOE_FORWARD_LEGACY_REDUCE",
        "MEGAMOE_FORWARD_FC1_CTAS",
        "MEGAMOE_FORWARD_M64",
        "MEGAMOE_FORWARD_M64_HEAD",
        "MEGAMOE_FORWARD_M64_HEAD_CTAS",
        "MEGAMOE_FORWARD_M64_HEAD_EXPERTS",
        "MEGAMOE_FORWARD_SPECIALIZE_STAGES",
        "MEGAMOE_FORWARD_FC2_TMA_STORE",
        "MEGAMOE_FORWARD_COMPUTE_LOCAL_RETURN",
        "MEGAMOE_FORWARD_RELAY_CTAS",
        "MEGAMOE_FORWARD_RELAY_TMA_WARPS",
        "MEGAMOE_FORWARD_LOCAL_RETURN_BATCH_MBLOCKS",
        "MEGAMOE_FORWARD_DISPATCH_PUBLISH_BATCH_ROWS",
        "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP",
        "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP4_BEGIN",
        "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP4_END",
        "MEGAMOE_FORWARD_RETURN_MAX_INFLIGHT",
        "MEGAMOE_FORWARD_REGISTERED_OUTPUT",
        "MEGAMOE_FORWARD_FC1_STAGES",
        "MEGAMOE_FORWARD_FC2_STAGES",
        "MEGAMOE_GIN_BENCH_MODE",
        "MEGAMOE_GIN_BENCH_SHIFT",
        "MEGAMOE_GIN_BENCH_FANOUT",
        "MEGAMOE_GIN_BENCH_WARMUP",
        "MEGAMOE_GIN_BENCH_MEASURED",
        "MEGAMOE_GIN_BENCH_BATCH",
        "MEGAMOE_BACKWARD_INJECT_HASH_MISMATCH",
        "MEGAMOE_BACKWARD_CACHE_FIRST",
        "MEGAMOE_PRODUCTION_STATUS_FAULT_PHASE",
        "MEGAMOE_FORWARD_CACHE",
        "MEGAMOE_BACKWARD_CACHE",
        "MEGAMOE_P2P_TOP_K",
        "MEGAMOE_P2P_RING_SLOTS",
        "MEGAMOE_P2P_NO_WRAP",
        "MEGAMOE_GATE_MODE",
        "MEGAMOE_RUNTIME_HIDDEN",
        "MEGAMOE_RUNTIME_INTERMEDIATE",
        "MEGAMOE_DEVICE_TIMELINE_PREFIX",
        "NCCL_IB_HCA",
        "NCCL_IB_GID_INDEX",
        "NCCL_IB_TC",
        "NCCL_IB_QPS_PER_CONNECTION",
        "NCCL_IB_SPLIT_DATA_ON_QPS",
        "NCCL_IB_PCI_RELAXED_ORDERING",
        "NCCL_NET_GDR_LEVEL",
        "NCCL_CROSS_NIC",
        "NCCL_SOCKET_FAMILY",
    ):
        if key in os.environ:
            overrides[key] = os.environ[key]
    return overrides


def local_environment(args):
    environment = os.environ.copy()
    environment.update(environment_overrides(args))
    if args.nccl_lib_dir:
        prior = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = args.nccl_lib_dir + (
            (":" + prior) if prior else ""
        )
    return environment


def remote_command(args, runner):
    overrides = environment_overrides(args)
    if args.nccl_lib_dir:
        overrides["LD_LIBRARY_PATH"] = args.nccl_lib_dir
    command = ["env", *(f"{key}={value}" for key, value in overrides.items()), *runner]
    return [
        "ssh", "-o", "BatchMode=yes", "-p", str(args.port), args.remote,
        "exec " + shlex.join(command),
    ]


def read_bootstrap(fd, timeout):
    global _active_bootstrap
    deadline = time.monotonic() + timeout
    received = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("bootstrap secret pipe", timeout)
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise subprocess.TimeoutExpired("bootstrap secret pipe", timeout)
        chunk = os.read(fd, BOOTSTRAP_BYTES + 1 - len(received))
        if not chunk:
            break
        received.extend(chunk)
        _active_bootstrap = bytes(received[:BOOTSTRAP_BYTES])
        if len(received) > BOOTSTRAP_BYTES:
            raise RuntimeError(
                "bootstrap secret channel malformed: expected=128, actual=>128"
            )
    if len(received) != BOOTSTRAP_BYTES:
        raise RuntimeError(
            "bootstrap secret channel short read: expected=128, "
            f"actual={len(received)}"
        )
    return bytes(received)


def send_bootstrap(process, bootstrap):
    stream = process.stdin
    primary_error = None
    try:
        if stream is None:
            raise RuntimeError(
                "bootstrap secret channel has no writable stdin"
            )
        try:
            frame = bytes(bootstrap)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "bootstrap secret channel requires exactly 128 bytes"
            ) from error
        if len(frame) != BOOTSTRAP_BYTES:
            raise RuntimeError(
                "bootstrap secret channel requires exactly 128 bytes; "
                f"actual={len(frame)}"
            )
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError) as error:
            raise RuntimeError(
                "bootstrap secret channel has no valid descriptor"
            ) from error
        offset = 0
        while offset < BOOTSTRAP_BYTES:
            try:
                written = os.write(descriptor, frame[offset:])
            except InterruptedError:
                continue
            except BrokenPipeError as error:
                raise RuntimeError(
                    "bootstrap secret channel closed before 128 bytes"
                ) from error
            except OSError as error:
                raise RuntimeError(
                    "bootstrap secret channel write failed before 128 bytes"
                ) from error
            if written is None or not isinstance(written, int):
                raise RuntimeError(
                    "bootstrap secret channel write returned no byte count"
                )
            if written <= 0:
                raise RuntimeError(
                    "bootstrap secret channel write made zero progress"
                )
            if written > BOOTSTRAP_BYTES - offset:
                raise RuntimeError(
                    "bootstrap secret channel write exceeded remaining frame"
                )
            offset += written
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError) as error:
                if primary_error is None:
                    raise RuntimeError(
                        "bootstrap secret channel close failed"
                    ) from error
            finally:
                # communicate() otherwise attempts to flush a closed stream.
                process.stdin = None


def remote_cleanup(args, launch_id, signal):
    # The unique argv marker lets cleanup target only this launcher even when
    # terminating the local ssh client did not propagate a signal remotely.
    pattern = f"--launch-id {launch_id}"
    command = (
        f"pkill -{signal} -f -- " + shlex.quote(pattern)
    )
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-p", str(args.port),
             args.remote, command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Cleanup is best-effort here; the primary test error is preserved.
        pass


def read_rank_output(rank, process, captures, bootstrap=None):
    stdout_capture, stderr_capture = captures[rank]
    stdout_capture.seek(0)
    stderr_capture.seek(0)
    return RankOutput(
        rank,
        process.returncode if process.returncode is not None else -1,
        scrub_output(stdout_capture.read(), bootstrap),
        scrub_output(stderr_capture.read(), bootstrap),
    )


def close_captures(captures):
    for stdout_capture, stderr_capture in captures.values():
        stdout_capture.close()
        stderr_capture.close()


def stop_processes(processes, captures, args, launch_id):
    for _rank, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    drained = []
    for rank, process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        drained.append(read_rank_output(rank, process, captures))
    remote_cleanup(args, launch_id, "TERM")
    time.sleep(0.25)
    remote_cleanup(args, launch_id, "KILL")
    return drained


def cleanup_diagnostic(outputs):
    details = []
    for result in outputs:
        combined = result.stdout + "\n" + result.stderr
        lines = [line for line in combined.splitlines() if line.strip()]
        # Persistent-kernel debug dumps can be tens of kilobytes per rank.
        # Keep the last milestone from each useful class so a 16-rank timeout
        # remains readable and still identifies the causal protocol phase.
        markers = (
            "FORWARD_PROGRESS_COARSE",
            "FORWARD_HOST_PROGRESS",
            "FORWARD_RUN_HOST",
            "BACKWARD_PROGRESS rank=",
            "[FAIL]",
            "terminate called",
            "what():",
        )
        selected = []
        for marker in markers:
            match = next((line for line in reversed(lines) if marker in line),
                         None)
            if match is not None:
                # The detailed generation-completion record is useful, but
                # cap a malformed/expanded record without dropping the coarse
                # milestone that precedes it.
                selected.append(match[-10000:] if
                                "BACKWARD_PROGRESS rank=" in match else
                                match[-2000:])
        detailed = next(
            (line for line in reversed(lines)
             if "FORWARD_PROGRESS rank=" in line),
            None,
        )
        if detailed is not None:
            prefix = detailed.split(" forward_completion.", 1)[0]
            fields = [prefix]
            for name in (
                "dispatch", "relay", "fc1_or_dv", "fc2_or_dx", "return",
                "arrival", "reduce", "directory",
                "bulk_return_barrier", "bulk_return_stage", "fc_scheduler",
                "tile_wait", "fc2_stalled_ctas",
            ):
                match = re.search(
                    rf"forward_completion\.{name}=(\{{[^}}]*\}}|[^ ]+)",
                    detailed,
                )
                if match is not None:
                    fields.append(f"{name}={match.group(1)}")
            for context in range(8):
                match = re.search(
                    rf"forward_completion\.context\[{context}\]="
                    rf"(\{{[^}}]*\}})",
                    detailed,
                )
                if match is not None:
                    fields.append(f"context[{context}]={match.group(1)}")
            for name in ("kernel_complete_epoch", "terminal_publish_count"):
                match = re.search(
                    rf"forward_completion\.{name}=([^ ]+)", detailed
                )
                if match is not None:
                    fields.append(f"{name}={match.group(1)}")
            selected.append(" ".join(fields))
        if not selected and lines:
            selected.append("\n".join(lines[-8:])[-2000:])
        if selected:
            details.append(
                f"rank={result.rank}, exit={result.returncode}, "
                + " | ".join(selected)
            )
    return scrub_output("; ".join(details))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--remote", required=True,
                        help="explicit SSH destination, for example user@host")
    parser.add_argument("--remote-binary", required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4D554C54)
    parser.add_argument("--nccl-debug", default="WARN")
    parser.add_argument("--nccl-lib-dir",
                        help="optional libnccl directory on both hosts")
    parser.add_argument("--socket-ifname",
                        help="optional NCCL_SOCKET_IFNAME override")
    ib_group = parser.add_mutually_exclusive_group()
    ib_group.add_argument("--disable-ib", action="store_true",
                          help="set NCCL_IB_DISABLE=1 explicitly")
    ib_group.add_argument("--enable-ib", action="store_true",
                          help="set NCCL_IB_DISABLE=0 explicitly")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--scenario",
        choices=("valid", "out_of_range", "duplicate", "max_mismatch",
                 "over_capacity", "generation_mismatch", "session_mismatch",
                 "mtile_mismatch", "topology_mismatch",
                 "expert_capacity_mismatch", "workspace_capacity_mismatch",
                 "gin_probe", "gin_put_probe", "gin_link_bench",
                 "gin_team_pool",
                 "gin_forward_smoke",
                 "gin_backward_smoke",
                 "socket_backward_smoke",
                 "gin_forward_bench",
                 "gin_backward_bench",
                 "socket_forward_bench",
                 "socket_backward_bench",
                 "activation_socket",
                 "activation_gin"),
        default="valid",
    )
    args = parser.parse_args(argv)
    if args.gpus_per_node <= 0:
        raise SystemExit("gpus-per-node must be positive")
    return args


def launch_rank_processes(args):
    global _active_bootstrap
    _active_bootstrap = b""

    environment = local_environment(args)
    world = 2 * args.gpus_per_node
    launch_id = secrets.token_hex(16)
    processes = []
    captures = {}
    outputs = []
    bootstrap_read_fd = -1
    bootstrap_write_fd = -1
    pending_error = None
    try:
        bootstrap_read_fd, bootstrap_write_fd = os.pipe()
        captures[0] = (tempfile.TemporaryFile(), tempfile.TemporaryFile())
        rank0 = subprocess.Popen(
            runner_args(args.binary, 0, args.gpus_per_node, args.seed,
                        args.scenario, launch_id, bootstrap_write_fd, True),
            stdout=captures[0][0],
            stderr=captures[0][1],
            stdin=subprocess.DEVNULL,
            env=environment,
            pass_fds=(bootstrap_write_fd,),
        )
        processes.append((0, rank0))
        os.close(bootstrap_write_fd)
        bootstrap_write_fd = -1
        bootstrap = read_bootstrap(
            bootstrap_read_fd, min(args.timeout, 60.0)
        )
        os.close(bootstrap_read_fd)
        bootstrap_read_fd = -1

        for rank in range(1, world):
            runner = runner_args(
                args.binary if rank < args.gpus_per_node else args.remote_binary,
                rank, args.gpus_per_node, args.seed, args.scenario, launch_id,
                0,
            )
            command = runner if rank < args.gpus_per_node else remote_command(args, runner)
            captures[rank] = (tempfile.TemporaryFile(),
                              tempfile.TemporaryFile())
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=captures[rank][0],
                stderr=captures[rank][1],
                env=environment,
            )
            processes.append((rank, process))
            send_bootstrap(process, bootstrap)

        deadline = time.monotonic() + args.timeout
        for rank, process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, args.timeout)
            process.wait(timeout=remaining)
        outputs = [
            read_rank_output(rank, process, captures, bootstrap)
            for rank, process in processes
        ]
    except BaseException as error:
        pending_error = error
    finally:
        if bootstrap_read_fd >= 0:
            os.close(bootstrap_read_fd)
        if bootstrap_write_fd >= 0:
            os.close(bootstrap_write_fd)
        drained = stop_processes(processes, captures, args, launch_id)
    if pending_error is not None:
        # Preserve any output captured before cleanup together with the final
        # snapshots; the first failing rank usually explains why a peer waited.
        diagnostic = cleanup_diagnostic(outputs + drained)
        if isinstance(pending_error, subprocess.TimeoutExpired):
            # `drained` already contains every process.  Attaching only the
            # compact summary avoids duplicating an unbounded rank dump.
            pending_error.output = diagnostic
            pending_error.stderr = ""
            close_captures(captures)
            raise pending_error
        message = scrub_output(str(pending_error))
        if diagnostic:
            message += "; cleanup=" + diagnostic
        # Do not retain an unsanitized exception as __cause__/debug repr.
        close_captures(captures)
        raise RuntimeError(message) from None
    close_captures(captures)
    return outputs


def main(argv=None):
    args = parse_args(argv)
    world = 2 * args.gpus_per_node
    launch_environment = environment_overrides(args)
    print(
        "LAUNCH_CONFIG "
        f"route_case={launch_environment.get('MEGAMOE_BACKWARD_ROUTE_CASE', 'regular')} "
        f"repeats={launch_environment.get('MEGAMOE_BACKWARD_SMOKE_REPEATS', '1')} "
        f"pool2={int(launch_environment.get('MEGAMOE_BACKWARD_POOL2_VALIDATE') == '1')} "
        f"cache_first={int('MEGAMOE_BACKWARD_CACHE_FIRST' in launch_environment)} "
        f"NCCL_IB_DISABLE={launch_environment.get('NCCL_IB_DISABLE', 'inherit')}"
    )
    outputs = launch_rank_processes(args)

    runtime_lines = []
    for result in outputs:
        evidence = [line for line in result.stdout.splitlines()
                    if line.startswith("NCCL_RUNTIME ")]
        if len(evidence) != 1:
            raise RuntimeError(
                f"rank={result.rank} missing unique NCCL runtime evidence; "
                f"exit={result.returncode}, stdout={result.stdout[-2000:]}, "
                f"stderr={result.stderr[-2000:]}"
            )
        runtime_lines.append(evidence[0])
    print("\n".join(runtime_lines))

    if args.scenario == "gin_probe":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_DISTRIBUTED_PROBE ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"evidence={len(lines)}; stdout={result.stdout[-4000:]}; "
                    f"stderr={result.stderr[-4000:]}"
                )
            else:
                evidence.append(lines[0])
        if failures:
            raise RuntimeError("distributed GIN probe failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        initializable = all("runtime_initializable=1" in line
                            for line in evidence)
        print(f"[PASS] distributed GIN lifecycle probe world={world} "
              f"runtime_initializable={int(initializable)}")
        return

    if args.scenario == "gin_put_probe":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_PUT_SIGNAL ")]
            if (result.returncode != 0 or len(lines) != 1 or
                    "match=1" not in (lines[0] if lines else "")):
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"evidence={len(lines)}; stdout={result.stdout[-4000:]}; "
                    f"stderr={result.stderr[-4000:]}"
                )
            else:
                evidence.append(lines[0])
        if failures:
            raise RuntimeError("distributed GIN put/signal failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        print(f"[PASS] real GIN rail put/signal/local-completion world={world}")
        return

    if args.scenario == "gin_link_bench":
        failures = []
        evidence = []
        samples = {}
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_LINK_BENCH ")]
            if (result.returncode != 0 or not lines or
                    any("match=1" not in line for line in lines)):
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"evidence={len(lines)}; stdout={result.stdout[-6000:]}; "
                    f"stderr={result.stderr[-6000:]}"
                )
            else:
                evidence.extend(lines)
                for line in lines:
                    fields = {
                        key: value
                        for key, value in (
                            item.split("=", 1)
                            for item in line[len("GIN_LINK_BENCH "):].split()
                        )
                    }
                    key = (fields["mode"], int(fields["shift"]),
                           int(fields["payload"]), int(fields["puts"]))
                    durations = [int(value) for value in
                                 fields["durations_ns"].split(",")]
                    samples.setdefault(key, {})[result.rank] = durations
        if failures:
            raise RuntimeError("distributed GIN link benchmark failed:\n" +
                               "\n".join(failures))
        if os.environ.get("MEGAMOE_GIN_BENCH_VERBOSE") == "1":
            print("\n".join(evidence))
        for (mode, shift, payload, puts), rank_samples in sorted(
                samples.items()):
            if len(rank_samples) != world:
                raise RuntimeError(
                    f"GIN link benchmark is missing ranks: mode={mode} "
                    f"shift={shift} payload={payload} puts={puts} "
                    f"observed={len(rank_samples)} expected={world}")
            sample_counts = {len(values)
                             for values in rank_samples.values()}
            if len(sample_counts) != 1:
                raise RuntimeError(
                    f"GIN link benchmark sample count mismatch: "
                    f"mode={mode} shift={shift} payload={payload} puts={puts}")
            # Every rank runs concurrently, so the slowest rank for each
            # iteration determines the distributed critical path.
            critical_ns = [
                max(values[index] for values in rank_samples.values())
                for index in range(next(iter(sample_counts)))
            ]
            stats = sample_stats(critical_ns)
            gbps = payload * puts / stats["mean"]
            print(
                f"GIN_LINK_SUMMARY mode={mode} shift={shift} "
                f"payload={payload} puts={puts} "
                f"mean_us={stats['mean'] / 1.0e3:.3f} "
                f"p50_us={stats['p50'] / 1.0e3:.3f} "
                f"p95_us={stats['p95'] / 1.0e3:.3f} "
                f"critical_gbps={gbps:.3f} "
                f"aggregate_gbps={gbps * args.gpus_per_node:.3f}")
        print(f"[PASS] GIN link benchmark world={world}")
        return

    if args.scenario == "gin_team_pool":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_TEAM_POOL ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}; "
                    f"stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
        if failures:
            raise RuntimeError("distributed GIN team/pool failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        print(f"[PASS] runtime GIN rail/local-team/pool world={world}")
        return

    if args.scenario in ("gin_forward_smoke", "gin_backward_smoke",
                         "socket_backward_smoke"):
        status_fault_phase = os.environ.get(
            "MEGAMOE_PRODUCTION_STATUS_FAULT_PHASE")
        if status_fault_phase:
            failures = []
            evidence = []
            error_hashes = set()
            expected_field = "field=120" if status_fault_phase == "forward" \
                else "field=220"
            for result in outputs:
                lines = [line for line in result.stdout.splitlines()
                         if line.startswith("PRODUCTION_STATUS_FAULT ")]
                if result.returncode != 0 or len(lines) != 1:
                    failures.append(
                        f"rank={result.rank} exit={result.returncode}; "
                        f"stdout={result.stdout[-5000:]}; "
                        f"stderr={result.stderr[-5000:]}"
                    )
                    continue
                line = lines[0]
                fields = dict(item.split("=", 1) for item in line.split()[1:])
                if (fields.get("origin_rank") != "1" or
                        expected_field not in line or
                        fields.get("recovered") != "1"):
                    failures.append(f"rank={result.rank} malformed={line}")
                    continue
                error_hashes.add(fields.get("error_hash"))
                evidence.append(line)
            if len(error_hashes) != 1:
                failures.append(
                    "production status errors were not byte-identical: " +
                    repr(sorted(error_hashes)))
            if failures:
                raise RuntimeError(
                    "production terminal status agreement failed:\n" +
                    "\n".join(failures))
            print("\n".join(evidence))
            print(f"[PASS] production {status_fault_phase} status agreement "
                  f"and same-communicator recovery world={world}")
            return
        failures = []
        evidence = []
        for result in outputs:
            marker = ("FORWARD_SMOKE " if args.scenario == "gin_forward_smoke"
                      else "BACKWARD_SMOKE ")
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith(marker)]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}; "
                    f"stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
        if failures:
            raise RuntimeError("distributed GIN MegaKernel failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        if args.scenario in ("gin_backward_smoke", "socket_backward_smoke"):
            stage_evidence = []
            for result in outputs:
                stage_evidence.extend(
                    line for line in result.stdout.splitlines()
                    if line.startswith("BACKWARD_HASH_RECOVERY ")
                )
                stage_evidence.extend(
                    line for line in result.stderr.splitlines()
                    if line.startswith("BACKWARD_STAGE_HASH ") or
                    line.startswith("BACKWARD_STAGE_FIRST_MISMATCH ")
                )
            if stage_evidence:
                print("\n".join(stage_evidence))
        stage = "Forward" if args.scenario == "gin_forward_smoke" else "Backward"
        backend = ("host-enqueued Socket fallback"
                   if args.scenario == "socket_backward_smoke"
                   else "same-rail GIN")
        print(f"[PASS] {backend} {stage} persistent kernel world={world}")
        return

    if args.scenario in ("gin_forward_bench", "gin_backward_bench"):
        failures = []
        evidence = []
        pool2 = (args.scenario == "gin_backward_bench" and
                 os.environ.get("MEGAMOE_BACKWARD_POOL2_VALIDATE") == "1")
        marker = ("FORWARD_BENCH " if args.scenario == "gin_forward_bench"
                  else ("BACKWARD_POOL2 " if pool2 else "BACKWARD_BENCH "))
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith(marker)]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}; "
                    f"stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
                if not pool2 and "backend=gin-rdma" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} wrong benchmark backend: "
                        f"{lines[0]}"
                    )
        if failures:
            raise RuntimeError("distributed GIN benchmark failed:\n" +
                               "\n".join(failures))
        if (args.scenario == "gin_backward_bench" and
                os.environ.get("MEGAMOE_BACKWARD_STAGE_HASHES")):
            for result in outputs:
                evidence.extend(
                    line for line in result.stdout.splitlines()
                    if line.startswith("BACKWARD_V3_DX_HASH ")
                )
        print("\n".join(evidence))
        if pool2:
            print(f"[PASS] same-rail GIN Backward pool=2 lifecycle world={world}")
        else:
            print("GIN_BENCH_JSON " + json.dumps(
                aggregate_gin_benchmark(outputs, args.scenario),
                sort_keys=True))
            print(f"[PASS] same-rail GIN benchmark world={world}")
        return

    if args.scenario == "socket_forward_bench":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("FORWARD_BENCH ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}; "
                    f"stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
                if "backend=nccl-socket-fallback" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} wrong Socket Forward backend: "
                        f"{lines[0]}"
                    )
        if failures:
            raise RuntimeError("distributed Socket Forward failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        print(f"[PASS] host-enqueued Socket full Forward world={world}")
        return

    if args.scenario == "socket_backward_bench":
        failures = []
        evidence = []
        pool2 = os.environ.get("MEGAMOE_BACKWARD_POOL2_VALIDATE") == "1"
        marker = "BACKWARD_POOL2 " if pool2 else "BACKWARD_BENCH "
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith(marker)]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}; "
                    f"stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
                if not pool2 and "backend=nccl-socket-fallback" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} wrong Socket Backward backend: "
                        f"{lines[0]}"
                    )
        if failures:
            raise RuntimeError("distributed Socket Backward failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        suffix = "pool=2 lifecycle" if pool2 else "full Backward"
        print(f"[PASS] host-enqueued Socket {suffix} world={world}")
        return

    if args.scenario == "activation_socket":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("SOCKET_RESULT ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"results={len(lines)}; stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
                if "backend=nccl-socket-host-enqueued-fallback" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} wrong backend audit: {lines[0]}"
                    )
                if "oracle=1" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} missing Socket oracle proof: "
                        f"{lines[0]}"
                    )
        if failures:
            raise RuntimeError("NCCL Socket activation failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        print(f"[PASS] real cross-host NCCL Socket activation world={world}")
        return

    if args.scenario == "activation_gin":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_RESULT ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"results={len(lines)}; stdout={result.stdout[-5000:]}; "
                    f"stderr={result.stderr[-5000:]}"
                )
            else:
                evidence.append(lines[0])
                if "backend=nccl-gin-device" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} wrong backend audit: {lines[0]}"
                    )
                if "oracle=1" not in lines[0]:
                    failures.append(
                        f"rank={result.rank} missing GIN oracle proof: "
                        f"{lines[0]}"
                    )
        if failures:
            raise RuntimeError("NCCL GIN activation failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        print(f"[PASS] real device-issued GIN activation world={world}")
        return

    if args.scenario != "valid":
        errors = []
        failures = []
        for result in outputs:
            markers = [line.removeprefix("CONTROL_ERROR ")
                       for line in result.stdout.splitlines()
                       if line.startswith("CONTROL_ERROR ")]
            recoveries = [line for line in result.stdout.splitlines()
                          if line.startswith(
                              f"RECOVERY_OK rank={result.rank} round=")]
            if (result.returncode == 0 or len(markers) != 20 or
                    len(recoveries) != 20):
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"control_errors={len(markers)}, recoveries={len(recoveries)}; "
                    f"stdout={result.stdout[-4000:]}; stderr={result.stderr[-4000:]}"
                )
            elif any("origin_rank=1" not in marker for marker in markers):
                failures.append(
                    f"rank={result.rank} error lacks origin_rank=1: {markers[-1]}"
                )
            errors.append(markers)
        if failures:
            raise RuntimeError("invalid protocol launch failed:\n" + "\n".join(failures))
        for round_index in range(20):
            round_errors = [rank_errors[round_index] for rank_errors in errors]
            if len(set(round_errors)) != 1:
                raise AssertionError(
                    f"ranks selected different errors in round={round_index}: {round_errors}"
                )
        print(
            f"[PASS] real cross-host 20-round control rejection/recovery: "
            f"scenario={args.scenario}, world={world}, error={errors[0][0]}"
        )
        return

    actual = [None] * world
    failures = []
    for result in outputs:
        if result.returncode:
            failures.append(
                f"rank={result.rank} exit={result.returncode}; "
                f"stdout={result.stdout[-10000:]}; "
                f"stderr={result.stderr[-10000:]}"
            )
            continue
        lines = [line for line in result.stdout.splitlines()
                 if line.startswith("{")]
        if len(lines) != 1:
            failures.append(
                f"rank={result.rank} expected one metadata JSON line, "
                f"got {len(lines)}: {result.stdout[-4000:]}"
            )
            continue
        actual[result.rank] = json.loads(lines[0])
    if failures:
        raise RuntimeError("multinode NCCL launch failed:\n" + "\n".join(failures))

    config = config_for(args.gpus_per_node)
    routes = seeded_inputs(world, config.num_experts, config.top_k, args.seed)
    expected = build_reference_all(config, routes, "cuda-multinode", 701)
    if actual != expected:
        mismatch = next(rank for rank, pair in enumerate(zip(actual, expected))
                        if pair[0] != pair[1])
        raise AssertionError(f"multinode full-field mismatch at rank={mismatch}")
    if not any(routes[rank] for rank in range(args.gpus_per_node, world)):
        raise AssertionError("fixture did not contain any remote-node routes")
    print(
        f"[PASS] real cross-host NCCL control/config/count/summary: nodes=2, "
        f"gpus_per_node={args.gpus_per_node}, world={world}, full-field match"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as error:
        diagnostic = scrub_output(error.output or "")
        print(scrub_output(
            f"[FAIL] rank launch timed out after {error.timeout}s; "
            f"cleanup={diagnostic}"
        ), file=sys.stderr)
        raise SystemExit(124)
    except Exception as error:
        print(scrub_output(f"[FAIL] {error}"), file=sys.stderr)
        raise SystemExit(1)
    finally:
        _active_bootstrap = b""
