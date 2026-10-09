#!/usr/bin/env python3
"""Run gin_pack.cu using the existing two-host bootstrap and cleanup machinery."""

import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from run_multinode_nccl import launch_rank_processes, parse_args, sample_stats


def monitor_clocks(args, stop, samples):
    query = ["nvidia-smi", "--query-gpu=index,clocks.current.sm,clocks.current.memory,utilization.gpu",
             "--format=csv,noheader,nounits"]
    commands = [query, ["ssh", "-o", "BatchMode=yes", "-p", str(args.port),
                        args.remote, shlex.join(query)]]
    while not stop.is_set():
        for node, command in enumerate(commands):
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            samples.append({"node": node, "time": time.time(), "returncode": result.returncode,
                            "csv": result.stdout.strip(), "error": result.stderr.strip()})
        stop.wait(1)


def main():
    args = parse_args()
    stop = threading.Event()
    clocks = []
    monitor = threading.Thread(target=monitor_clocks, args=(args, stop, clocks))
    monitor.start()
    try:
        outputs = launch_rank_processes(args)
    finally:
        stop.set()
        monitor.join()
    records = []
    runtime = []
    for output in outputs:
        if output.returncode:
            raise RuntimeError(f"rank {output.rank}: {output.stdout[-3000:]}\n{output.stderr[-3000:]}")
        for line in output.stdout.splitlines():
            if line.startswith("GIN_PACK_CASE "):
                records.append(json.loads(line.removeprefix("GIN_PACK_CASE ")))
            elif line.startswith("GIN_PACK_RUNTIME "):
                runtime.append(line)
    if len(runtime) != 2 * args.gpus_per_node or not records:
        raise RuntimeError("incomplete GIN benchmark output")
    cases = {}
    for record in records:
        key = (record["nominal_tokens"], record["operation"], record.get("mode", "burst"),
               record.get("depth", 1), record["packet_tokens"], record["repeat"])
        cases.setdefault(key, []).append(record)
    summaries = []
    for (nominal, operation, mode, depth, packet, repeat), ranks in sorted(cases.items()):
        if len(ranks) != len(runtime):
            raise RuntimeError("missing rank in case")
        counts = [sample["tokens"] for sample in ranks[0]["samples"]]
        if any([s["tokens"] for s in r["samples"]] != counts for r in ranks):
            raise RuntimeError("rank token counts differ")
        summary = {
            "nominal_tokens": nominal, "operation": operation,
            "packet_tokens": packet, "repeat": repeat,
            "mode": mode, "depth": depth,
            "contexts": ranks[0]["contexts"], "batch_tokens": ranks[0]["batch_tokens"],
            "mean_tokens": statistics.fmean(counts),
        }
        for metric in ("network_us", "pack_us", "ready_us", "total_us", "pack_tail_after_issue_us"):
            if metric not in ranks[0]["samples"][0]:
                continue
            values = [max(r["samples"][i][metric] for r in ranks) for i in range(len(counts))]
            summary[metric] = sample_stats(values)
            if metric in ("network_us", "total_us"):
                summary[metric.removesuffix("_us") + "_GBps_per_rank"] = sample_stats([
                    count * ranks[0]["row_bytes"] / elapsed / 1000
                    for count, elapsed in zip(counts, values)])
        summaries.append(summary)
        print(f"GIN_PACK_SUMMARY T={nominal} {operation} mode={mode} D={depth} Q={packet} repeat={repeat} "
              f"net_us={summary['network_us']['p50']:.3f} "
              f"total_us={summary['total_us']['p50']:.3f} "
              f"net_GBps={summary['network_GBps_per_rank']['p50']:.3f}", flush=True)
    print("GIN_PACK_JSON " + json.dumps({
        "schema": 2, "world": len(runtime), "runtime": runtime, "clocks": clocks,
        "scope": "bidirectional same-rail; full dense RX/TX; bounded request window; no GEMM; "
                 "stream network_us includes concurrent pack; pack_tail refers to owner/producer 0",
        "summaries": summaries, "records": records,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
