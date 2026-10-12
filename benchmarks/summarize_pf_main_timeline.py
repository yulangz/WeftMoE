"""Summarize paired PF runs and audit their measurement windows at 100 ms."""
from collections import Counter, defaultdict
import datetime
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "build/pf_main_timeline"
records, checks, groups, spans = [], [], defaultdict(list), []
matched = {}
for path in sorted((BASE / "results").glob("*_node0.log")):
    label = path.name.rsplit("_ep", 1)[0]
    for line in path.read_text().splitlines():
        prefix, _, payload = line.partition(" ")
        if prefix == "TRAINING_PARITY_JSON":
            checks.append(dict(label=label, result=json.loads(payload)))
            continue
        if prefix not in ("TRAINING_CYCLE_JSON", "MEGAMOE_PREFILL_JSON"):
            continue
        row = json.loads(payload)
        if not label.startswith(("compare_", "final_", "inference_", "stages_", "local_")) or "verify" in label or "trace" in label or "skew" in label:
            if prefix == "MEGAMOE_PREFILL_JSON":
                checks.append(dict(label=label, result=row))
            continue
        phase, _, variant = label.split("_", 2)
        size = row["nominal_tokens_per_rank"]
        world = row.get("world", len(row.get("per_rank", [])))
        hashes = row.get("workload_sha256_per_rank") or [w["sha256"] for w in row["workload"]]
        assert matched.setdefault((phase, world, size), hashes) == hashes, (label, size)
        intervals = row.get("measurement_intervals_ns_per_rank") or [r["measurement_intervals_ns"] for r in row["per_rank"]]
        stamps = [t for rank in intervals for pair in rank for t in pair]
        spans.append((min(stamps), max(stamps), label, size, world, intervals))
        if prefix == "TRAINING_CYCLE_JSON":
            samples = list(map(max, zip(*row["per_rank_forward_ready_us"])))
        else:
            samples = list(map(max, zip(*(r["device_samples_us"] for r in row["per_rank"]))))
        item = dict(label=label, phase=phase, world=world, tokens=size, variant=variant,
                    p50_us=statistics.median(samples), mean_us=statistics.mean(samples),
                    slowest_rank_samples_us=samples, workload_sha256_per_rank=hashes,
                    log=str(path.relative_to(ROOT)))
        item["workspace"] = row.get("per_rank_workspace") or row.get("workspace_rank0")
        if row.get("per_rank_stage_kernel_us"):
            values = row["per_rank_stage_kernel_us"]
            item["kernels_us"] = {
                k: statistics.median(map(max, zip(*(r[k] for r in values)))) if values[0][k] else 0
                for k in values[0]}
        if prefix == "MEGAMOE_PREFILL_JSON":
            values = row["per_rank"]
            item["notify_us"] = statistics.median(map(max, zip(*(
                [p["route_exchange_us"] for p in r["device_phases"]] for r in values))))
            item["main_span_us"] = statistics.median(map(max, zip(*(
                [p["total_us"] - p["route_exchange_us"] for p in r["device_phases"]] for r in values))))
        records.append(item)
        groups[phase, world, size, variant].append(item)

summary = []
for (phase, world, size, variant), rows in sorted(groups.items()):
    item = dict(phase=phase, world=world, tokens=size, variant=variant,
                rounds=len(rows), p50_us=statistics.mean(r["p50_us"] for r in rows),
                mean_us=statistics.mean(r["mean_us"] for r in rows))
    summary.append(item)
    print("PERF", json.dumps(item))

clocks = []
for node in (0, 1):
    counts = {(label, size): Counter() for _, _, label, size, world, _ in spans if node * 8 < world}
    deviations = defaultdict(list)
    for line in (BASE / f"clocks/node{node}.csv").read_text().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:
            continue
        stamp = int(datetime.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9)
        for begin, end, label, size, world, intervals in spans:
            if node * 8 < world and begin <= stamp <= end:
                frequency = int(fields[2].split()[0])
                counts[label, size][frequency] += 1
                if frequency != 1620:
                    rank = node * 8 + int(fields[1])
                    deviations[label, size].append(dict(sample=line, rank=rank,
                        inside_call=any(a <= stamp <= b for a, b in intervals[rank])))
    for (label, size), frequencies in counts.items():
        item = dict(node=node, label=label, tokens=size, frequencies=dict(frequencies),
                    deviations=deviations[label, size])
        clocks.append(item)
        if set(frequencies) != {1620}:
            print("CLOCK_REVIEW", json.dumps(item))

traces = {p.parent.name: json.loads(p.read_text()) for p in BASE.glob("*/summary.json")}
manifests = {p.stem: json.loads(p.read_text()) for p in (BASE / "results").glob("*.manifest.json")}
training = [e for c in checks if c["label"] in
            ("fc2_register_verify", "fc2_register_skew", "local_verify")
            for e in c["result"]["errors"]]
inference = [e for c in checks if c["label"] in
             ("retained_verify_inference", "retained_verify_graph", "local_verify_inference")
             for rank in c["result"]["per_rank"] for e in rank["errors"]]
validation = dict(training_rank_steps=len(training), inference_rank_steps=len(inference),
                  training_max_relative_l2=max(v["relative_l2"] for e in training for v in e.values()),
                  inference_max_relative_l2=max(e["relative_l2"] for e in inference))
print("VALIDATION", json.dumps(validation))
result = dict(retained_source_commit="269f63a", early_reduce_experiment_commit="a2a3a0a",
              summary=summary, records=records, clocks=clocks, checks=checks, validation=validation,
              timelines=traces, manifests=manifests)
(ROOT / "benchmarks/pf_main_timeline_20261012.json").write_text(json.dumps(result, indent=2))
