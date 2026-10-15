"""Summarize paired Return measurements and diagnostic-only credit traces."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def distribution(values):
    return dict(min=min(values), p50=statistics.median(values), max=max(values))


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
    result = dict(measurements=[], diagnostics={})
    workload_hashes = {}
    for log in sorted(args.directory.glob("*_ep16_node0.log")):
        label = log.name.removesuffix("_ep16_node0.log")
        if not label.endswith("_measure"):
            continue
        manifest = json.loads((args.directory / f"{label}.manifest.json").read_text())
        for line in log.read_text().splitlines():
            if not line.startswith("TRAINING_CYCLE_JSON "):
                continue
            raw = json.loads(line.split(" ", 1)[1])
            shape = "qwen" if "_qwen_" in label else "flash"
            variant = next(v for v in ("bound", "split", "weighted", "rail") if f"_{v}_" in label)
            route, tokens = raw["route_distribution"], raw["nominal_tokens_per_rank"]
            key = (shape, route, tokens)
            hashes = raw["workload_sha256_per_rank"]
            if key in workload_hashes and hashes != workload_hashes[key]:
                raise ValueError(f"Unmatched workload: {label}, {tokens}")
            workload_hashes[key] = hashes
            result["measurements"].append(dict(label=label, shape=shape, variant=variant,
                route=route, tokens=tokens,
                ms=raw["forward_ready_slowest_rank_p50_us"] / 1000,
                manifest=manifest, raw=raw))
    for directory in sorted(args.directory.glob("*_trace")):
        if not directory.is_dir():
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
