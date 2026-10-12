"""Audit the standalone-notify measurements, including padding and GPU clocks."""
from collections import Counter, defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "build/pf_notify_production"
records, checks, failed = [], [], []
for path in sorted((BASE / "results").glob("*_node0.log")):
    label = path.name.rsplit("_ep", 1)[0]
    parsed = 0
    for line in path.read_text().splitlines():
        if line.startswith(("TRAINING_CYCLE_JSON ", "MEGAMOE_PREFILL_JSON ", "TRAINING_PARITY_JSON ")):
            row = json.loads(line.partition(" ")[2])
            if line.startswith("TRAINING_PARITY_JSON "):
                checks.append(dict(label=label, result=row))
            else:
                records.append(dict(label=label, result=row))
            parsed += 1
    if not parsed:
        failed.append(str(path.relative_to(ROOT)))

groups = defaultdict(list)
matched = {}
times = []
measurement_spans = []
stages, inference = [], []
for entry in records:
    label, row = entry["label"], entry["result"]
    phase, node, _, variant, *_ = label.split("_")
    if phase.startswith("verify"):
        continue
    size, world = row["nominal_tokens_per_rank"], row["world"] if "world" in row else int(node[1:]) * 8
    hashes = row.get("workload_sha256_per_rank") or [w["sha256"] for w in row["workload"]]
    assert matched.setdefault((world, size), hashes) == hashes, (label, size)
    intervals = row.get("measurement_intervals_ns_per_rank") or [r["measurement_intervals_ns"] for r in row["per_rank"]]
    times.extend(t for rank in intervals for pair in rank for t in pair)
    stamps = [t for rank in intervals for pair in rank for t in pair]
    measurement_spans.append((min(stamps), max(stamps), label, size, world, intervals))
    if phase in ("performance", "cycle", "sweep"):
        value = row["forward_ready_slowest_rank_p50_us"] if row["forward_only"] else row["wall_slowest_rank_p50_us"]
        padding = row["window_padding"]
        groups[phase, world, size, variant].append(dict(
            us=value, workspace=row["per_rank_workspace"],
            m_padding_fraction=statistics.median(sum(p) / sum(a) - 1 for p, a in
                                                zip(padding["m128_rows"], padding["actual_rows"]))))
    if phase == "trace":
        values = row["per_rank_stage_kernel_us"]
        item = dict(world=world, tokens=size,
                    **{k: statistics.median(map(max, zip(*(r[k] for r in values)))) if values[0][k] else 0
                       for k in values[0]})
        stages.append(item)
        print("STAGES", item)
    if phase == "inference":
        values = row["per_rank"]
        item = dict(world=world, tokens=size, total_us=row["device_slowest_rank_p50_us"],
                    wall_us=row["wall_slowest_rank_p50_us"],
                    notify_us=statistics.median(map(max, zip(*(
                        [p["route_exchange_us"] for p in r["device_phases"]] for r in values)))),
                    main_span_us=statistics.median(map(max, zip(*(
                        [p["total_us"] - p["route_exchange_us"] for p in r["device_phases"]] for r in values)))))
        inference.append(item)
        print("INFERENCE", item)

summary = []
for (phase, world, size, variant), values in sorted(groups.items()):
    assert len(values) == 2, (phase, world, size, variant, len(values))
    item = dict(phase=phase, world=world, tokens=size, variant=variant,
                us=statistics.mean(v["us"] for v in values),
                m_padding_fraction=values[0]["m_padding_fraction"],
                workspace_gib_rank0=sum(v for k, v in values[0]["workspace"][0].items()
                                       if k.endswith("_bytes")) / 2**30,
                rounds=values)
    summary.append(item)
    print("PERF", phase, world, size, variant, item["us"], "padding", item["m_padding_fraction"],
          "workspace", values[0]["workspace"][0])

clocks, clock_cases = {}, []
for path in (BASE / "continuous_clocks").glob("*.csv"):
    counts, outside = Counter(), []
    by_case = defaultdict(Counter)
    off_clock = defaultdict(list)
    node = int(path.stem[-1])
    for _, _, label, size, world, _ in measurement_spans:
        if node * 8 < world:
            by_case[label, size] = Counter()
    start, end = min(times), max(times)
    for line in path.read_text().splitlines():
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 5:
            continue
        stamp = int(datetime.datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f")
                    .replace(tzinfo=datetime.timezone.utc).timestamp() * 1e9)
        frequency = int(fields[2].split()[0])
        if start <= stamp <= end:
            counts[frequency] += 1
        elif frequency != 1620:
            outside.append(line)
        for begin, finish, label, size, world, intervals in measurement_spans:
            if node * 8 >= world or not begin <= stamp <= finish:
                continue
            key = (label, size)
            by_case[key][frequency] += 1
            if frequency != 1620:
                rank = node * 8 + int(fields[1])
                off_clock[key].append(dict(sample=line, inside_call=any(
                    a <= stamp <= b for a, b in intervals[rank])))
    clocks[path.name] = dict(frequencies=dict(counts), off_clock_outside_measurements=outside)
    for (label, size), frequencies in sorted(by_case.items()):
        item = dict(node=node, label=label, tokens=size, frequencies=dict(frequencies),
                    off_clock=off_clock[label, size])
        clock_cases.append(item)
        if frequencies and set(frequencies) != {1620}:
            print("CLOCK_DEVIATION", label, size, "node", node, dict(frequencies))
        if frequencies and not label.startswith("cycle"):
            assert set(frequencies) == {1620}, item

parity = [e for c in checks for e in c["result"]["errors"]]
inference_errors = [e for r in records if r["label"].startswith("verify")
                    for rank in r["result"]["per_rank"] for e in rank["errors"]]
validation = dict(training_rank_steps=len(parity),
                  training_max_relative_l2=max(e[k]["relative_l2"] for e in parity for k in e),
                  inference_rank_steps=len(inference_errors),
                  inference_max_relative_l2=max(e["relative_l2"] for e in inference_errors))
print("VALIDATION", validation)
payload = dict(native_commit="19ab646", summary=summary, stages=stages, inference=inference,
               records=records, training_checks=checks, failed_attempts=failed, clocks=clocks,
               clocks_per_case=clock_cases,
               validation=validation,
               measurement_window_ns=[min(times), max(times)],
               library_sha256={v: hashlib.file_digest((BASE / v / "libweftmoe.so").open("rb"), "sha256").hexdigest()
                               for v in ("local", "network")},
               manifests={p.stem: json.loads(p.read_text()) for p in (BASE / "results").glob("*.manifest.json")})
(ROOT / "benchmarks/pf_notify_production_20261012.json").write_text(json.dumps(payload, indent=2))
