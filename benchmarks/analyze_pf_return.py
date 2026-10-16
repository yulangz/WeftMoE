"""Summarize paired Return measurements and diagnostic-only credit traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics


def distribution(values):
    return dict(min=min(values), p50=statistics.median(values), max=max(values))


def paired_latency_change(baseline, candidate):
    a = list(map(max, zip(*baseline["raw"]["per_rank_forward_ready_us"])))
    b = list(map(max, zip(*candidate["raw"]["per_rank_forward_ready_us"])))
    rng = random.Random(123)
    changes = []
    for _ in range(4000):
        indices = rng.choices(range(len(a)), k=len(a))
        changes.append(100 * (1 - statistics.median(b[i] for i in indices) /
                              statistics.median(a[i] for i in indices)))
    changes.sort()
    return dict(latency_reduction_percent=100 * (1 - candidate["ms"] / baseline["ms"]),
                paired_bootstrap_95_percent=[changes[100], changes[3899]])


def analyze_trace(path):
    trace = json.loads(path.read_text())
    if trace["overflow_events"]:
        raise ValueError(f"Incomplete timeline: {path}")
    events = defaultdict(list)
    for event in trace["events"]:
        events[event["event"]].append(event)
    origin = min(e["timestamp_ns"] for e in events[108])

    def boundary(event, last=True):
        clocks = [e["timestamp_ns"] for e in events[event]]
        return ((max(clocks) if last else min(clocks)) - origin) / 1e6

    waits = {e["channel"]: e["arg0"] / 1e6 for e in events[119]}
    credit = [e["arg0"] / 1e6 for e in events[120]]
    workers = events[121]
    puts = events[103]
    return dict(
        rank=trace["rank"], generation=trace["generation"],
        notify_ms=boundary(101) - boundary(96, False),
        main_ms=boundary(22), last_admit_ms=boundary(117),
        last_fc2_ms=boundary(87), reduce_start_ms=boundary(20, False),
        fc2_to_reduce_ms=boundary(20, False) - boundary(87),
        admit_wait_by_mask_ms=waits, admit_return_only_ms=waits[4],
        admit_any_return_ms=sum(value for mask, value in waits.items() if mask & 4),
        tx_credit_cta_ms=distribution(credit), tx_credit_cta_sum_ms=sum(credit),
        tx_credit_blocked_tiles=sum(e["arg1"] for e in events[120]),
        nic_wait_worker_ms=distribution([e["arg0"] / 1e6 for e in events[122]]),
        return_bytes=sum(e["arg0"] for e in workers),
        return_puts=sum(e["arg1"] for e in workers),
        return_first_put_ms=boundary(103, False) if puts else None,
        return_workers_done_ms=boundary(121),
        return_worker_bytes=[e["arg0"] for e in sorted(workers, key=lambda x: x["channel"])])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = dict(measurements=[], inference=[], diagnostics={}, runs={}, parity={}, comparisons=[])
    workload_hashes = {}
    workspaces = {}
    for log in sorted(args.directory.glob("*_ep16_node0.log")):
        label = log.name.removesuffix("_ep16_node0.log")
        manifest = json.loads((args.directory / f"{label}.manifest.json").read_text())
        clock_file = args.directory / f"{label}.clocks.json"
        result["runs"][label] = dict(manifest=manifest,
            clocks=json.loads(clock_file.read_text()) if clock_file.exists() else [])
        for line in log.read_text().splitlines():
            if line.startswith("TRAINING_PARITY_JSON "):
                result["parity"].setdefault(label, []).append(json.loads(line.split(" ", 1)[1]))
            if line.startswith("MEGAMOE_PREFILL_JSON "):
                result["inference"].append(dict(label=label, raw=json.loads(line.split(" ", 1)[1])))
            if not line.startswith("TRAINING_CYCLE_JSON "):
                continue
            raw = json.loads(line.split(" ", 1)[1])
            shape = "qwen" if "_qwen_" in label else "flash"
            variant = next(v for v in ("bound", "split", "weighted", "rail") if f"_{v}_" in label)
            route, tokens = raw["route_distribution"], raw["nominal_tokens_per_rank"]
            mode = label.rsplit("_", 1)[1]
            key = (mode, shape, route, tokens, raw["warmups"], raw["samples"])
            hashes = raw["workload_sha256_per_rank"]
            if key in workload_hashes and hashes != workload_hashes[key]:
                raise ValueError(f"Unmatched workload: {label}, {tokens}")
            workload_hashes[key] = hashes
            if key in workspaces and workspaces[key] != raw["per_rank_workspace"]:
                raise ValueError(f"Unmatched persistent capacity: {label}, {tokens}")
            workspaces[key] = raw["per_rank_workspace"]
            if mode == "trace":
                result["runs"][label].setdefault("workloads", []).append(raw)
                continue
            result["measurements"].append(dict(label=label, shape=shape, variant=variant,
                route=route, tokens=tokens,
                ms=raw["forward_ready_slowest_rank_p50_us"] / 1000,
                manifest=manifest, raw=raw))
    for row in result["measurements"]:
        if row["variant"] == "bound":
            continue
        baseline = next((other for other in result["measurements"]
                         if other["variant"] == "bound" and
                         all(other[key] == row[key] for key in ("shape", "route", "tokens"))
                         and other["label"].split("_", 1)[0] == row["label"].split("_", 1)[0]), None)
        if baseline:
            result["comparisons"].append(dict(label=row["label"], tokens=row["tokens"],
                baseline=baseline["label"], **paired_latency_change(baseline, row)))
    for directory in sorted(args.directory.glob("*_trace")):
        if not directory.is_dir() or not (args.directory / f"{directory.name}.clocks.json").exists():
            continue
        cases = defaultdict(list)
        for path in sorted(directory.glob("t*.rank*.json")):
            cases[path.name.split(".")[0]].append(analyze_trace(path))
        diagnostics = {}
        for case, ranks in cases.items():
            if len(ranks) != 16:
                raise ValueError(f"Missing ranks: {directory}/{case}")
            ranks.sort(key=lambda row: row["rank"])
            nodes = []
            for node in range(2):
                payloads = [r["return_bytes"] for r in ranks[node * 8:(node + 1) * 8]]
                mean = statistics.mean(payloads)
                nodes.append(dict(bytes=payloads, max_over_mean=max(payloads) / mean,
                                  cv=statistics.pstdev(payloads) / mean))
            summary = {key: distribution([r[key] for r in ranks]) for key in (
                "notify_ms", "main_ms", "last_admit_ms", "last_fc2_ms", "reduce_start_ms",
                "fc2_to_reduce_ms", "admit_return_only_ms", "admit_any_return_ms")}
            diagnostics[case] = dict(ranks=ranks, nodes=nodes, summary=summary)
        result["diagnostics"][directory.name] = diagnostics
    traffic = {}
    for label, cases in result["diagnostics"].items():
        shape = "qwen" if "_qwen_" in label else "flash"
        route = "balanced" if "_balanced_" in label else "random"
        for case, data in cases.items():
            key = (shape, route, case)
            totals = [sum(n["bytes"]) for n in data["nodes"]]
            if key in traffic and totals != traffic[key]:
                raise ValueError(f"Unmatched Return bytes: {label}/{case}")
            traffic[key] = totals
    output = args.output or args.directory / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["measurements"]:
        print(row["label"], row["tokens"], round(row["ms"], 4))
    for label, cases in result["diagnostics"].items():
        for case, data in cases.items():
            print(label, case, json.dumps(data["summary"], sort_keys=True),
                  "TX max/mean", [round(n["max_over_mean"], 3) for n in data["nodes"]])


if __name__ == "__main__":
    main()
