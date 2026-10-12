"""PF pipeline diagnostics; all subtractions stay in one GPU clock domain."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


def stats(values):
    values = sorted(values)
    if not values:
        return {}
    return dict(count=len(values), min=min(values), p50=statistics.median(values),
                p95=values[min(len(values) - 1, int(len(values) * .95))], max=max(values))


def analyze(path):
    trace = json.loads(path.read_text())
    assert trace["overflow_events"] == 0, (path, trace["overflow_events"])
    events = defaultdict(list)
    for e in trace["events"]:
        events[e["event"]].append(e)
    start = min(e["timestamp_ns"] for e in events[108])
    extents = {k: dict(first_us=(min(e["timestamp_ns"] for e in v) - start) / 1000,
                       last_us=(max(e["timestamp_ns"] for e in v) - start) / 1000,
                       count=len(v)) for k, v in events.items()}

    def pairs(begin, end):
        first = {(e["cta"], e["channel"], e["arg0"]): e["timestamp_ns"] for e in events[begin]}
        return [(e["timestamp_ns"] - first[key]) / 1000 for e in events[end]
                if (key := (e["cta"], e["channel"], e["arg0"])) in first]

    tiles = {}
    for name, a, b, c, d in (("fc1", 82, 113, 83, 84), ("fc2", 85, 114, 86, 87)):
        tiles[name] = dict(first_input_wait_us=stats(pairs(a, b)),
                           math_span_us=stats(pairs(b, c)),
                           epilogue_us=stats(pairs(c, d)), total_us=stats(pairs(a, d)))
    producer = defaultdict(dict)
    for event in (111, 112):
        for e in events[event]:
            producer[e["cta"], e["channel"], e["arg0"], e["arg1"]][event] = e["timestamp_ns"]
    producer_wait = {str(fc): stats([(v[112] - v[111]) / 1000
                                    for k, v in producer.items() if k[1] == fc and len(v) == 2])
                     for fc in (0, 1)}
    compute_end = defaultdict(int)
    for e in events[86]:
        compute_end[e["arg0"]] = max(compute_end[e["arg0"]], e["timestamp_ns"])
    publish = [(e["timestamp_ns"] - compute_end[e["arg0"]]) / 1000 for e in events[79]
               if e["arg0"] in compute_end]
    tma = defaultdict(float)
    for e in events[118]:
        tma[e["cta"], e["channel"]] += e["arg0"] / 1000
    workers = defaultdict(list)
    for e in events[109]:
        workers[e["channel"]].append((e["timestamp_ns"] - start) / 1000)
    main_us = extents[22]["last_us"]
    return dict(file=path.name, rank=trace["rank"], generation=trace["generation"],
                events=extents, main_us=main_us, tiles=tiles,
                producer_wait_us=producer_wait,
                fc2_math_to_block_publication_us=stats(publish),
                tma_wait_per_cta_us={str(fc): stats([v for k, v in tma.items() if k[1] == fc])
                                     for fc in (0, 1)},
                workers_done_us={k: stats(v) for k, v in workers.items()},
                dispatch_get_submit_us=stats(pairs(115, 116)),
                dispatch_get_complete_us=stats(pairs(115, 4)),
                remote_pull_packet_us=stats(pairs(6, 7)),
                return_submit_us=stats(pairs(103, 14)),
                return_complete_us=stats(pairs(103, 15)),
                main_to_first_mma_us=extents[113]["first_us"],
                last_fc2_to_reduce_us=extents[20]["first_us"] - extents[87]["last_us"],
                last_rx_to_reduce_us=extents[20]["first_us"] - extents[16]["last_us"],
                join_us=extents[20]["first_us"] - extents[109]["last_us"],
                reduce_span_us=extents[21]["last_us"] - extents[20]["first_us"],
                terminal_span_us=main_us - extents[21]["last_us"])


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
args = parser.parse_args()
groups = defaultdict(list)
for path in sorted(args.directory.glob("t*.rank*.json")):
    groups[path.name.split(".")[0]].append(analyze(path))
result = {}
for size, ranks in groups.items():
    keys = ("main_us", "main_to_first_mma_us", "last_fc2_to_reduce_us", "last_rx_to_reduce_us",
            "join_us", "reduce_span_us", "terminal_span_us")
    totals = {k: stats([r[k] for r in ranks]) for k in keys}
    result[size] = dict(ranks=ranks, summary=totals)
    print(size, json.dumps(totals))
(args.directory / "summary.json").write_text(json.dumps(result, indent=2))
