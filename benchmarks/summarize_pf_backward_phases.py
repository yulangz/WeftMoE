"""Summarize device-local BWD clocks; phase envelopes may overlap.

Use --env WEFTMOE_BACKWARD_PROGRESS=1 with run_pf_network.py. Input is one
node-0 log; the matching node-1 log is included when present. Only timed trace
positions are summarized; no timestamps from different devices are subtracted.
"""
import argparse
import json
from pathlib import Path
import re
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    paths = [args.log]
    peer = args.log.with_name(args.log.name.replace("node0", "node1"))
    if peer != args.log and peer.exists():
        paths.append(peer)
    records, reports = {}, []
    for path in paths:
        for line in path.read_text().splitlines():
            if line.startswith("TRAINING_CYCLE_JSON "):
                reports.append(json.loads(line.split(" ", 1)[1]))
            if "BACKWARD_STATUS rank=" not in line:
                continue
            r = {key: int(value) for key, value in re.findall(r"(\w+)=(\d+)", line)}
            if r["pending"] == 0:
                records.setdefault(r["rank"], []).append(r)
    offset = 0
    for report in reports:
        begin = offset + report["priming_steps"] + report["warmups"]
        end = begin + report["samples"]
        phases = {}
        for index in range(begin, end):
            step = {}
            for rank in range(report["world"]):
                r = records[rank][index]
                dw_start = min(r["dw1_start"], r["dw2_start"])
                dw_end = max(r["dw1_complete"], r["dw2_complete"])
                spans = dict(first_gemm=(r["start"], r["du_start"]),
                             dispatch=(r["start"], r["dispatch_complete"]),
                             activation=(r["du_start"], r["dx_complete"]),
                             weights=(dw_start, dw_end),
                             reduce=(r["reduce_start"], r["reduce_complete"]),
                             return_tail=(r["dx_complete"], r["return_complete"]),
                             main_envelope=(r["start"], max(dw_end, r["reduce_complete"])))
                for name, (lo, hi) in spans.items():
                    if 0 < lo <= hi < 2**64 - 1:
                        step[name] = max(step.get(name, 0), (hi - lo) / 1e6)
            for name, value in step.items():
                phases.setdefault(name, []).append(value)
        print(json.dumps(dict(world=report["world"], tokens=report["nominal_tokens_per_rank"],
                              rank_max_p50_ms={k: statistics.median(v) for k, v in phases.items()},
                              samples_ms=phases)))
        offset = end


if __name__ == "__main__":
    main()
