#!/usr/bin/env python3
"""Summarize wire-only GIN and real PF copy-substitution diagnostic logs.

Bandwidth is useful payload GB/s, not FLOPS or summed TX+RX on one link.
PF accounts each Get at its receiver and each Return Put at its sender;
sum over all ranks / two nodes is the mean one-direction node payload.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics


def stats(values):
    return dict(p50=statistics.median(values), min=min(values), max=max(values),
                samples=values)


def summarize(path):
    raw = path.read_bytes()
    result = dict(log=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest())
    gin, pf = [], []
    for line in raw.decode().splitlines():
        if line.startswith("GIN_PACK_JSON "):
            record = json.loads(line.removeprefix("GIN_PACK_JSON "))
            gin = record["summaries"]
            result["clocks"] = record["clocks"]
            result["runtime"] = record["runtime"]
            result["validated_rank_samples"] = sum(
                r["validated_samples"] for r in record["records"])
            for case in gin:
                case["node_one_direction_GBps_p50"] = (
                    case["network_GBps_per_rank"]["p50"] * record["world"] / 2)
                case["line_rate_fraction"] = case["network_GBps_per_rank"]["p50"] / 50
        elif line.startswith("MEGAMOE_PREFILL_JSON "):
            record = json.loads(line.removeprefix("MEGAMOE_PREFILL_JSON "))
            assert record["communication_copy"]
            ranks = record["per_rank"]
            if not ranks[0]["device_phases"]:
                continue
            measured = len(ranks[0]["device_phases"])
            times, total, plans, tails, bandwidth, full_bandwidth = [], [], [], [], [], []
            payload, dispatch, returning = [], [], []
            directional = [[], []]
            for i in range(measured):
                phases = [r["device_phases"][i] for r in ranks]
                get_bytes = sum(p["dispatch_payload_bytes"] for p in phases)
                put_bytes = sum(p["return_payload_bytes"] for p in phases)
                assert get_bytes == sum(p["expected_dispatch_bytes"] for p in phases)
                assert put_bytes == sum(p["expected_return_bytes"] for p in phases)
                elapsed = max(p["return_us"] for p in phases)
                complete = max(p["total_us"] for p in phases)
                times.append(elapsed)
                total.append(complete)
                plans.append(max(p["route_exchange_us"] for p in phases))
                tails.append(max(p["completion_tail_us"] for p in phases))
                dispatch.append(get_bytes)
                returning.append(put_bytes)
                payload.append((get_bytes + put_bytes) / 2)
                bandwidth.append(payload[-1] / elapsed / 1000)
                full_bandwidth.append(payload[-1] / complete / 1000)
                local = len(phases) // 2
                for node in range(2):
                    outbound = sum(p["return_payload_bytes"] for p in phases[node*local:(node+1)*local])
                    peer = 1 - node
                    outbound += sum(p["dispatch_payload_bytes"] for p in phases[peer*local:(peer+1)*local])
                    directional[node].append(outbound / elapsed / 1000)
            pf.append(dict(
                nominal_tokens=record["nominal_tokens_per_rank"],
                world=record["world"], workload=record["workload"],
                workspace_rank0=record["workspace_rank0"],
                communication_us=stats(times), kernel_us=stats(total),
                prepare_us=stats(plans), completion_tail_us=stats(tails),
                dispatch_global_bytes=stats(dispatch), return_global_bytes=stats(returning),
                payload_bytes_per_direction=stats(payload),
                node_one_direction_GBps=stats(bandwidth),
                full_kernel_node_one_direction_GBps=stats(full_bandwidth),
                node0_tx_GBps=stats(directional[0]), node1_tx_GBps=stats(directional[1]),
                per_gpu_one_direction_GBps=stats([v / (record["world"] / 2) for v in bandwidth]),
                line_rate_fraction=statistics.median(bandwidth) / 400,
            ))
    result.update(gin=gin, pf=pf)
    if pf:
        prefix = re.sub(r"_ep\d+_node\d+\.log$", "", path.name)
        result["clocks"] = json.loads(path.with_name(prefix + ".clocks.json").read_text())
        manifest = path.with_name(prefix + ".launch.json")
        if not manifest.exists():
            manifest = path.with_name(prefix + ".manifest.json")
        result["launch"] = json.loads(manifest.read_text())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(schema=1, hardware="2 nodes x 8 L20X; each GPU has a 2x200Gb/s bonded rail",
                  gpu_SM_MHz=1620, scope="Bidirectional concurrent traffic, report one-direction payload GB/s",
                  runs=[summarize(path) for path in args.logs])
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
