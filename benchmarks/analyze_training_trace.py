"""Attribute CUPTI GPU events to benchmark FWD/BWD CPU launch ranges.

Use launch correlation, not GPU/CPU timestamp containment: async kernels may
execute after the CPU range ends. Report overlap as an interval union.
"""
import argparse
import collections
import json
from pathlib import Path


def analyze(path):
    trace = json.loads(Path(path).read_text())["traceEvents"]
    ranges = sorted((e for e in trace if e.get("cat") == "user_annotation"
                     and e["name"] in ("weftmoe.forward", "weftmoe.backward")),
                    key=lambda e: e["ts"])
    launches = {e["args"]["correlation"]: e for e in trace
                if e.get("cat") in ("cuda_runtime", "cuda_driver")
                and "correlation" in e.get("args", {})}
    attached = [[] for _ in ranges]
    unmatched = []
    for event in trace:
        if event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        launch = launches.get(event.get("args", {}).get("correlation"))
        owners = [i for i, r in enumerate(ranges) if launch
                  and r["pid"] == launch["pid"] and r["tid"] == launch["tid"]
                  and r["ts"] <= launch["ts"] < r["ts"] + r["dur"]]
        if owners:
            assert len(owners) == 1
            attached[owners[0]].append(event)
        elif "weftmoe" in event.get("name", ""):
            unmatched.append(event)
    assert not unmatched, f"Unattributed WeftMoE GPU launches in {path}: {unmatched}"
    result = collections.defaultdict(list)
    for region, events in zip(ranges, attached):
        assert events, (path, region)
        begin = min(e["ts"] for e in events)
        end = max(e["ts"] + e["dur"] for e in events)
        intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in events)
        busy, until = 0.0, begin
        for start, stop in intervals:
            busy += max(0.0, stop - max(start, until))
            until = max(until, stop)
        result[region["name"]].append(dict(
            gpu_span_us=end - begin, gpu_busy_union_us=busy,
            cpu_api_us=region["dur"],
            events=[dict(name=e["name"], category=e["cat"], duration_us=e["dur"],
                         offset_us=e["ts"] - begin, args=e.get("args", {}))
                    for e in sorted(events, key=lambda e: e["ts"])],
        ))
    return dict(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.trace), indent=2) + "\n")
