#!/usr/bin/env python3
"""Compare verified Megatron NCCL RDMA with MegaMOE GIN/RDMA."""

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


def load_marker(path, marker):
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines()
             if line.startswith(marker)]
    if len(lines) != 1:
        raise RuntimeError(
            f"{path}: expected one {marker.strip()}, actual={len(lines)}")
    return json.loads(lines[0].split(" ", 1)[1])


def milliseconds(stats):
    return {key: value / 1000.0 for key, value in stats.items()}


def host_wall(record, field):
    return record["runs"][0][field]["host_wall"]


def speedup(baseline, candidate):
    return (baseline - candidate) / baseline * 100.0


def cross_run(values):
    return {
        "mean_ms": statistics.fmean(values),
        "standard_deviation_ms": statistics.pstdev(values),
        "per_session_mean_ms": values,
    }


def aggregate_counter_throughput(records):
    keys = ("local_tx", "local_rx", "remote_tx", "remote_rx")
    return {key: [record["rdma_evidence"][
        "aggregate_hca_counter_throughput_gbytes_per_second"][key]
                  for record in records] for key in keys}


def public_key_log_lines(evidence):
    return [line for line in evidence["key_log_lines"]
            if "ncclIbConnect:" not in line][:22]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin-forward", nargs=3, required=True)
    parser.add_argument("--gin-backward", nargs=3, required=True)
    parser.add_argument("--megatron-rdma", nargs=3, required=True)
    parser.add_argument("--cpu-affinity", required=True)
    parser.add_argument("--gpu-set", required=True)
    parser.add_argument("--nccl-runtime-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    forwards = [load_marker(path, "GIN_BENCH_JSON ")
                for path in args.gin_forward]
    backwards = [load_marker(path, "GIN_BENCH_JSON ")
                 for path in args.gin_backward]
    baselines = [load_marker(path, "MULTINODE_MEGATRON_RDMA_JSON ")
                 for path in args.megatron_rdma]
    rows = []
    for run, (forward, backward, baseline) in enumerate(
            zip(forwards, backwards, baselines), start=1):
        if forward.get("backend") != "gin-rdma-device-persistent" or \
                backward.get("backend") != "gin-rdma-device-persistent":
            raise RuntimeError(f"session={run}: MegaMOE did not use GIN/RDMA")
        if baseline.get("network_backend") != "nccl-rdma-roce-gdr":
            raise RuntimeError(f"session={run}: Megatron did not use NCCL RDMA")
        evidence = baseline["rdma_evidence"]
        if not evidence.get("net_ib_observed") or \
                not evidence.get("gdrdma_observed") or \
                evidence.get("socket_payload_fallback_observed"):
            raise RuntimeError(f"session={run}: invalid RDMA evidence")
        if any(item.get("world") != 16
               for item in (forward, backward, baseline)):
            raise RuntimeError(f"session={run}: expected world=16")
        if forward.get("measured") != 100 or backward.get("measured") != 100:
            raise RuntimeError(f"session={run}: MegaMOE requires 100 samples")
        if baseline["shape"]["iterations"] != 100 or \
                baseline["shape"]["warmups"] != 10:
            raise RuntimeError(f"session={run}: Megatron requires 10+100")
        for field in FIXTURE_FIELDS:
            if forward[field] != baseline[field] or \
                    backward[field] != baseline[field]:
                raise RuntimeError(
                    f"session={run}: fixture mismatch field={field}")
        if backward["per_rank_output_gradient_hash"] != \
                baseline["per_rank_output_gradient_hash"]:
            raise RuntimeError(f"session={run}: dY fixture mismatch")

        mega = {
            "forward": milliseconds(forward["wall_slowest_rank_us"]),
            "backward": milliseconds(
                backward["backward_wall_slowest_rank_us"]),
            "notify_inclusive_forward_backward": milliseconds(backward[
                "notify_inclusive_training_step_wall_slowest_rank_us"]),
        }
        base = {
            "forward": host_wall(baseline, "native_forward_e2e"),
            "backward": host_wall(baseline, "native_backward_only"),
            "notify_inclusive_forward_backward": host_wall(
                baseline, "native_forward_backward_e2e"),
        }
        improvement = {
            field: speedup(base[field]["mean_ms"], mega[field]["mean"])
            for field in mega
        }
        dispatch = host_wall(baseline, "dispatch_alltoall")
        combine = host_wall(baseline, "combine")
        payload = baseline.get("derived_payload_bytes_per_rank",
                               baseline.get("payload_bytes_per_rank"))
        if payload is None:
            raise RuntimeError(f"session={run}: missing payload byte model")
        rows.append({
            "session": run,
            "megamoe_gin_slowest_rank_ms": mega,
            "megatron_rdma_slowest_rank_ms": base,
            "speedup_percent": improvement,
            "megatron_collective_slowest_rank_ms": {
                "dispatch_alltoall": dispatch,
                "combine": combine,
            },
            "megatron_derived_effective_payload_gbytes_per_second": {
                "dispatch": (payload["dispatch_activation_peer"] +
                             payload["dispatch_probability_peer"]) /
                            dispatch["mean_ms"] / 1.0e6,
                "combine": payload["combine_activation_peer"] /
                           combine["mean_ms"] / 1.0e6,
            },
            "megatron_hca_counter_throughput_gbytes_per_second": evidence[
                "aggregate_hca_counter_throughput_gbytes_per_second"],
            "megatron_hca_counter_delta": {
                "local": evidence["local_hca_counter_delta"],
                "remote": evidence["remote_hca_counter_delta"],
            },
            "rdma_log_sha256": {
                "local": evidence["local_log_sha256"],
                "remote": evidence["remote_log_sha256"],
            },
        })

    metrics = ("forward", "backward", "notify_inclusive_forward_backward")
    mega_values = {field: [row["megamoe_gin_slowest_rank_ms"][field]["mean"]
                           for row in rows] for field in metrics}
    base_values = {field: [row["megatron_rdma_slowest_rank_ms"][field][
        "mean_ms"] for row in rows] for field in metrics}
    speedups = {field: [row["speedup_percent"][field] for row in rows]
                for field in metrics}
    evidence_template = baselines[0]["rdma_evidence"]
    result = {
        "schema": 1,
        "comparison": "world16-megamoe-gin-vs-megatron-nccl-rdma-training",
        "shape": forwards[0]["shape"],
        "measurement": {
            "independent_process_and_nccl_sessions": 3,
            "warmups_per_session": 10,
            "measurements_per_session": 100,
            "rank_aggregation": "slowest-rank/MAX",
            "forward_scope": "route/gate ready through final Forward drain",
            "backward_scope": "device dY ready through final gradient drain",
            "training_scope": "route/gate ready through Notify, Forward, "
                              "Backward and final gradient drain",
            "one_time_session_pool_weight_setup_included": False,
            "cpu_affinity_both_hosts": args.cpu_affinity,
            "gpu_set_both_hosts": args.gpu_set,
        },
        "fixture_agreement": {
            "fields": list(FIXTURE_FIELDS) +
                      ["per_rank_output_gradient_hash"],
            "all_ranks_all_sessions": True,
            "per_rank_activation_hash": forwards[0][
                "per_rank_activation_hash"],
            "per_rank_route_hash": forwards[0]["per_rank_route_hash"],
            "per_rank_gate_hash": forwards[0]["per_rank_gate_hash"],
            "per_rank_actual_w1_device_hash": forwards[0][
                "per_rank_actual_w1_device_hash"],
            "per_rank_actual_w2_device_hash": forwards[0][
                "per_rank_actual_w2_device_hash"],
            "per_rank_output_gradient_hash": backwards[0][
                "per_rank_output_gradient_hash"],
        },
        "rdma_proof": {
            "nccl_runtime": "2.29.7+cuda12.9",
            "nccl_runtime_library": "NCCL_RUNTIME_DIR/libnccl.so.2.29.7",
            "nccl_runtime_sha256": args.nccl_runtime_sha256,
            "network_backend": "NCCL NET/IB over RoCE v2 with GDRDMA",
            "socket_payload_fallback_observed": False,
            "hcas": evidence_template[
                "active_hcas_with_bidirectional_host_evidence"],
            "gid_index": 3,
            "gid_type": "RoCE v2",
            "rail_rate": "200 Gb/sec (2X NDR)",
            "key_log_lines": public_key_log_lines(evidence_template),
            "hca_counter_throughput_by_session_gbytes_per_second":
                aggregate_counter_throughput(baselines),
        },
        "runs": rows,
        "cross_session": {
            "megamoe_gin": {field: cross_run(mega_values[field])
                             for field in metrics},
            "megatron_rdma": {field: cross_run(base_values[field])
                               for field in metrics},
            "speedup_percent": {
                field: {
                    "overall_from_cross_session_means": speedup(
                        statistics.fmean(base_values[field]),
                        statistics.fmean(mega_values[field])),
                    "per_session": speedups[field],
                    "worst_session": min(speedups[field]),
                } for field in metrics
            },
        },
        "network_audit": {
            "megamoe_gin": {
                "per_rank_forward_dispatch_payload_bytes": forwards[0][
                    "per_rank_gin_dispatch_payload_bytes"],
                "per_rank_forward_return_payload_bytes": forwards[0][
                    "per_rank_gin_return_payload_bytes"],
                "per_rank_forward_metadata_bytes": forwards[0][
                    "per_rank_gin_metadata_bytes"],
                "per_rank_forward_dispatch_puts": forwards[0][
                    "per_rank_gin_dispatch_puts"],
                "per_rank_forward_return_puts": forwards[0][
                    "per_rank_gin_return_puts"],
                "per_rank_forward_signals": forwards[0][
                    "per_rank_gin_signals"],
                "per_rank_backward_dy_payload_bytes": backwards[0][
                    "per_rank_backward_gin_dy_payload_bytes"],
                "per_rank_backward_dx_payload_bytes": backwards[0][
                    "per_rank_backward_gin_dx_payload_bytes"],
                "per_rank_backward_metadata_bytes": backwards[0][
                    "per_rank_backward_gin_metadata_bytes"],
                "per_rank_backward_dispatch_puts": backwards[0][
                    "per_rank_backward_gin_dispatch_puts"],
                "per_rank_backward_return_puts": backwards[0][
                    "per_rank_backward_gin_return_puts"],
                "per_rank_backward_signals": backwards[0][
                    "per_rank_backward_gin_signals"],
                "credit_stalls": 0,
                "credit_stalls_source":
                    "phase-batched production audit; unchanged frozen path",
                "overlap_slowest_rank_us": {
                    "forward_fc1_fc2": [item[
                        "fc1_fc2_overlap_slowest_rank_us"] for item in forwards],
                    "forward_fc2_return": [item[
                        "fc2_return_overlap_slowest_rank_us"]
                                           for item in forwards],
                    "backward_dw1_dv_dx": [item[
                        "dw1_overlap_dv_dx_slowest_rank_us"]
                                            for item in backwards],
                    "backward_dw2_dv_dx": [item[
                        "dw2_overlap_dv_dx_slowest_rank_us"]
                                            for item in backwards],
                },
            },
            "megatron_rdma": {
                "derived_payload_bytes_per_rank": baselines[0].get(
                    "derived_payload_bytes_per_rank",
                    baselines[0].get("payload_bytes_per_rank")),
                "collective_breakdown_present_all_sessions": True,
            },
        },
        "conclusion": {
            "fifteen_percent_lead_achieved": all(
                min(speedups[field]) >= 15.0 for field in metrics),
            "socket_baseline_is_not_used_for_speedup": True,
        },
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("MULTINODE_RDMA_COMPARISON_JSON " + json.dumps(result,
                                                           sort_keys=True))


if __name__ == "__main__":
    main()
