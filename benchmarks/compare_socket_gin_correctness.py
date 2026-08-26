#!/usr/bin/env python3
"""Compare actual per-rank Forward/Backward storage hashes across backends."""

import argparse
import json
from pathlib import Path


def records(path, marker):
    result = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.startswith(marker + " "):
            continue
        fields = {}
        for item in line.split()[1:]:
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        rank = int(fields["rank"])
        if rank in result:
            raise RuntimeError(f"{path}: duplicate {marker} rank {rank}")
        result[rank] = fields
    if not result:
        raise RuntimeError(f"{path}: no {marker} records")
    return result


def compare(left, right, fields, phase):
    if set(left) != set(right):
        raise RuntimeError(
            f"{phase}: rank sets differ: {sorted(left)} != {sorted(right)}")
    mismatches = []
    for rank in sorted(left):
        for field in fields:
            expected = left[rank].get(field)
            actual = right[rank].get(field)
            if expected != actual:
                mismatches.append({
                    "rank": rank, "field": field,
                    "gin": expected, "socket": actual,
                })
    if mismatches:
        raise RuntimeError(
            f"{phase}: backend hash mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    return len(left) * len(fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin-forward", required=True)
    parser.add_argument("--socket-forward", required=True)
    parser.add_argument("--gin-backward", required=True)
    parser.add_argument("--socket-backward", required=True)
    args = parser.parse_args()

    fixture_fields = [
        "world", "tokens", "experts", "top_k", "hidden", "intermediate",
        "activation_hash", "route_hash", "gate_hash",
        "actual_w1_hash_low", "actual_w1_hash_high",
        "actual_w2_hash_low", "actual_w2_hash_high",
    ]
    forward_fields = fixture_fields + ["output_hash"]
    backward_fields = fixture_fields + [
        "output_gradient_hash",
        "actual_dx_hash_low", "actual_dx_hash_high",
        "actual_dw1_hash_low", "actual_dw1_hash_high",
        "actual_dw2_hash_low", "actual_dw2_hash_high",
        "actual_dgate_hash_low", "actual_dgate_hash_high",
    ]
    gin_forward = records(args.gin_forward, "FORWARD_BENCH")
    socket_forward = records(args.socket_forward, "FORWARD_BENCH")
    gin_backward = records(args.gin_backward, "BACKWARD_BENCH")
    socket_backward = records(args.socket_backward, "BACKWARD_BENCH")
    forward_checks = compare(
        gin_forward, socket_forward, forward_fields, "Forward")
    backward_checks = compare(
        gin_backward, socket_backward, backward_fields, "Backward")
    result = {
        "schema": 1,
        "world": len(gin_forward),
        "forward_rank_field_checks": forward_checks,
        "backward_rank_field_checks": backward_checks,
        "socket_backend": "nccl-socket-fallback",
        "gin_backend": "gin-rdma",
        "byte_identical": True,
    }
    print("SOCKET_GIN_CORRECTNESS_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
