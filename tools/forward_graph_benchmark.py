"""Common single-node and GIN CUDA Graph benchmark aggregation."""

import statistics


def aggregate_forward_graph(records):
    samples = [[float(value) for value in item["samples"].split(",")]
               for item in records]
    measured = len(samples[0])
    if not measured or any(len(row) != measured for row in samples):
        raise ValueError("incomplete Forward graph benchmark")
    slowest = [max(row[i] for row in samples) for i in range(measured)]
    ordered = sorted(slowest)
    result = {
        "schema": 1,
        "benchmark": "forward-graph-replay",
        "backend": records[0].get("backend", "local-cuda-p2p"),
        "world": len(records),
        "warmup": 10,
        "measured": measured,
        "batch": int(records[0]["batch"]),
        "shape": {key: int(records[0][field]) for key, field in (
            ("tokens_per_rank", "tokens"), ("hidden", "hidden"),
            ("intermediate", "intermediate"), ("experts", "experts"),
            ("top_k", "top_k"))},
        "per_rank_output_hash": [item["output_hash"] for item in records],
        "per_rank_device_kernel_us": [float(item["device_kernel_us"])
                                      for item in records],
        "slowest_rank_us": {
            "mean": statistics.fmean(slowest),
            "p50": ordered[(measured - 1) // 2],
            "min": ordered[0], "max": ordered[-1],
        },
        "slowest_rank_samples_us": slowest,
        "per_rank_samples_us": samples,
    }
    for name in ("nodes", "kernel_nodes", "memcpy_nodes",
                 "event_record_nodes", "event_wait_nodes"):
        result["per_rank_graph_" + name] = [
            int(item["graph_" + name]) for item in records]
    for name in ("device_bytes", "compute_device_bytes", "gin_registered_bytes"):
        if all(name in item for item in records):
            result["per_rank_" + name] = [int(item[name]) for item in records]
    for name in ("static_schedule", "ring_window_tokens", "ring_contribution_windows"):
        if name in records[0]:
            result[name] = int(records[0][name])
    for name in ("route_exchange_us", "fc1_swiglu_us", "fc2_us", "after_combine_us"):
        if all(name in item for item in records):
            result["per_rank_" + name] = [float(item[name]) for item in records]
    return result
