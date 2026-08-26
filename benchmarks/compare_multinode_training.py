#!/usr/bin/env python3
"""Compare three world16 GinRdma runs with Megatron NCCL Socket runs."""

import argparse
import json
import statistics
from pathlib import Path


FIXTURE_FIELDS = (
    "per_rank_activation_hash",
    "per_rank_route_hash",
    "per_rank_gate_hash",
    "per_rank_actual_w1_device_hash",
    "per_rank_actual_w2_device_hash",
)

FORWARD_AUDIT_FIELDS = (
    "per_rank_compute_pool_bytes", "per_rank_gin_registered_bytes",
    "per_rank_pinned_pool_bytes", "per_rank_pool_events",
    "per_rank_raw_remote_route_bytes",
    "per_rank_gin_dispatch_payload_bytes",
    "per_rank_gin_return_payload_bytes", "per_rank_gin_metadata_bytes",
    "per_rank_gin_dispatch_puts", "per_rank_gin_return_puts",
    "per_rank_gin_signals", "per_rank_dispatch_peer_bytes",
    "per_rank_return_peer_bytes",
)

BACKWARD_AUDIT_FIELDS = (
    "per_rank_backward_raw_remote_dy_bytes",
    "per_rank_backward_gin_dy_payload_bytes",
    "per_rank_backward_gin_dx_payload_bytes",
    "per_rank_backward_gin_metadata_bytes",
    "per_rank_backward_gin_dispatch_puts",
    "per_rank_backward_gin_return_puts",
    "per_rank_backward_gin_signals",
)


def load_marker(path, marker):
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(
            f"{path}: expected one {marker.strip()}, actual={len(lines)}")
    return json.loads(lines[0].split(" ", 1)[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin-forward", nargs=3, required=True)
    parser.add_argument("--gin-backward", nargs=3, required=True)
    parser.add_argument("--megatron", nargs=3, required=True)
    parser.add_argument("--minimum-percent", type=float, default=15.0)
    parser.add_argument("--output")
    args = parser.parse_args()

    forwards = [load_marker(path, "GIN_BENCH_JSON ")
                for path in args.gin_forward]
    backwards = [load_marker(path, "GIN_BENCH_JSON ")
                 for path in args.gin_backward]
    megatron = [load_marker(path, "MULTINODE_MEGATRON_JSON ")
                for path in args.megatron]
    rows = []
    for run, (forward, backward, baseline) in enumerate(
            zip(forwards, backwards, megatron), start=1):
        if forward.get("backend") != "gin-rdma-device-persistent" or \
                backward.get("backend") != "gin-rdma-device-persistent":
            raise RuntimeError(f"run={run}: MegaMOE did not use GinRdma")
        if baseline.get("network_backend") != "nccl-socket":
            raise RuntimeError(f"run={run}: Megatron did not use NCCL Socket")
        if any(item.get("world") != 16
               for item in (forward, backward, baseline)):
            raise RuntimeError(f"run={run}: expected world=16")
        if forward.get("measured") < 100 or backward.get("measured") < 100:
            raise RuntimeError(f"run={run}: MegaMOE requires 100 measurements")
        if baseline["shape"]["iterations"] < 100:
            raise RuntimeError(f"run={run}: Megatron requires 100 measurements")
        for field in FIXTURE_FIELDS:
            expected = baseline[field]
            if forward[field] != expected or backward[field] != expected:
                raise RuntimeError(
                    f"run={run}: fixture mismatch field={field}")
        if (backward["per_rank_output_gradient_hash"] !=
                baseline["per_rank_output_gradient_hash"]):
            raise RuntimeError(f"run={run}: output gradient fixture mismatch")

        mega_forward = forward["wall_slowest_rank_us"]["mean"] / 1000.0
        mega_backward = backward["backward_wall_slowest_rank_us"]["mean"] / 1000.0
        mega_step = backward[
            "notify_inclusive_training_step_wall_slowest_rank_us"]["mean"] / 1000.0
        base_forward = baseline["runs"][0][
            "native_forward_e2e"]["host_wall"]["mean_ms"]
        base_backward = baseline["runs"][0][
            "native_backward_only"]["host_wall"]["mean_ms"]
        base_step = baseline["runs"][0][
            "native_forward_backward_e2e"]["host_wall"]["mean_ms"]
        improvements = {
            "forward": (base_forward - mega_forward) / base_forward * 100.0,
            "backward": (base_backward - mega_backward) / base_backward * 100.0,
            "forward_backward": (base_step - mega_step) / base_step * 100.0,
        }
        for field, value in improvements.items():
            if value < args.minimum_percent:
                raise RuntimeError(
                    f"run={run}: {field} improvement={value:.3f}% "
                    f"minimum={args.minimum_percent:.3f}%")
        rows.append({
            "run": run,
            "megamoe_gin_ms": {
                "forward": mega_forward,
                "backward": mega_backward,
                "notify_inclusive_forward_backward": mega_step,
            },
            "megatron_socket_ms": {
                "forward": base_forward,
                "backward": base_backward,
                "forward_backward": base_step,
            },
            "improvement_percent": improvements,
            "megamoe_forward_p95_ms":
                forward["wall_slowest_rank_us"]["p95"] / 1000.0,
            "megamoe_backward_p95_ms":
                backward["backward_wall_slowest_rank_us"]["p95"] / 1000.0,
            "megamoe_step_p95_ms": backward[
                "notify_inclusive_training_step_wall_slowest_rank_us"]["p95"] /
                1000.0,
            "megatron_forward_p95_ms": baseline["runs"][0][
                "native_forward_e2e"]["host_wall"]["p95_ms"],
            "megatron_backward_p95_ms": baseline["runs"][0][
                "native_backward_only"]["host_wall"]["p95_ms"],
            "megatron_step_p95_ms": baseline["runs"][0][
                "native_forward_backward_e2e"]["host_wall"]["p95_ms"],
        })

    for field in FORWARD_AUDIT_FIELDS:
        if any(run[field] != forwards[0][field] for run in forwards[1:]):
            raise RuntimeError(f"Forward runtime audit changed field={field}")
    for field in BACKWARD_AUDIT_FIELDS:
        if any(run[field] != backwards[0][field] for run in backwards[1:]):
            raise RuntimeError(f"Backward runtime audit changed field={field}")
    forward_audit = {field: forwards[0][field]
                     for field in FORWARD_AUDIT_FIELDS}
    backward_audit = {field: backwards[0][field]
                      for field in BACKWARD_AUDIT_FIELDS}
    forward_audit["derived_per_rank_local_dispatch_fanout_bytes"] = [
        total - remote for total, remote in zip(
            forward_audit["per_rank_dispatch_peer_bytes"],
            forward_audit["per_rank_gin_dispatch_payload_bytes"])]
    forward_audit["derived_per_rank_local_return_fanout_bytes"] = [
        total - remote for total, remote in zip(
            forward_audit["per_rank_return_peer_bytes"],
            forward_audit["per_rank_gin_return_payload_bytes"])]
    dispatch_us = statistics.fmean(
        run["dispatch_slowest_rank_us"]["mean"] for run in forwards)
    return_us = statistics.fmean(
        run["return_slowest_rank_us"]["mean"] for run in forwards)
    forward_audit["derived_per_rail_dispatch_payload_gbytes_per_second_range"] = [
        min(forward_audit["per_rank_gin_dispatch_payload_bytes"]) /
        dispatch_us / 1000.0,
        max(forward_audit["per_rank_gin_dispatch_payload_bytes"]) /
        dispatch_us / 1000.0,
    ]
    forward_audit["derived_per_rail_return_payload_gbytes_per_second_range"] = [
        min(forward_audit["per_rank_gin_return_payload_bytes"]) /
        return_us / 1000.0,
        max(forward_audit["per_rank_gin_return_payload_bytes"]) /
        return_us / 1000.0,
    ]

    result = {
        "schema": 1,
        "comparison": "world16-gin-vs-megatron-nccl-socket-training",
        "shape": forwards[0]["shape"],
        "measurement": {
            "independent_sessions": 3,
            "warmups_per_session": 10,
            "measurements_per_session": 100,
            "rank_aggregation": "slowest-rank/MAX",
            "minimum_improvement_percent": args.minimum_percent,
        },
        "fixture_agreement": {
            "fields": list(FIXTURE_FIELDS) +
                      ["per_rank_output_gradient_hash"],
            "all_ranks_all_runs": True,
        },
        "runs": rows,
        "cross_run_mean_ms": {
            "megamoe_gin_forward": statistics.fmean(
                row["megamoe_gin_ms"]["forward"] for row in rows),
            "megamoe_gin_backward": statistics.fmean(
                row["megamoe_gin_ms"]["backward"] for row in rows),
            "megamoe_gin_forward_backward": statistics.fmean(
                row["megamoe_gin_ms"]["notify_inclusive_forward_backward"]
                for row in rows),
            "megatron_socket_forward": statistics.fmean(
                row["megatron_socket_ms"]["forward"] for row in rows),
            "megatron_socket_backward": statistics.fmean(
                row["megatron_socket_ms"]["backward"] for row in rows),
            "megatron_socket_forward_backward": statistics.fmean(
                row["megatron_socket_ms"]["forward_backward"] for row in rows),
        },
        "network": {
            "megamoe": "NCCL GIN/RDMA same-rail device put/signal",
            "megatron": "NCCL Socket host-enqueued all-to-all",
            "megatron_socket_ifname": "eth0",
            "nccl_runtime": "2.29.7",
        },
        "gin_forward_runtime_audit": forward_audit,
        "gin_backward_runtime_audit": backward_audit,
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print("MULTINODE_TRAINING_COMPARISON_JSON " + json.dumps(
        result, sort_keys=True))


if __name__ == "__main__":
    main()
