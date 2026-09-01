#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize per-GPU MegaMOE %globaltimer traces.

Timestamps are never compared across ranks. Every duration is formed within
one rank/CTA/channel and aggregate percentiles are computed only afterwards.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EVENTS = {
    1: "kernel_enter",
    2: "dispatch_packet_ready",
    3: "gin_put_issue",
    4: "dispatch_local_completion",
    5: "remote_signal_observed",
    6: "proxy_relay_start",
    7: "proxy_relay_end",
    8: "x_mtile_ready",
    9: "fc1_first",
    10: "fc1_last",
    11: "fc2_tile_ready",
    12: "return_pack_start",
    13: "return_pack_end",
    14: "return_gin_put_issue",
    15: "return_local_completion",
    16: "source_signal_observed",
    17: "source_relay_start",
    18: "source_relay_end",
    19: "reduce_ready",
    20: "reduce_start",
    21: "reduce_end",
    22: "terminal",
    23: "dy_packet_ready",
    24: "dy_gin_put_issue",
    25: "dy_local_completion",
    26: "dy_relay_start",
    27: "dy_relay_end",
    28: "dv_first",
    29: "dv_last",
    30: "dx_tile_ready",
    31: "dw1_ready",
    32: "dw1_start",
    33: "dw1_end",
    34: "dw2_ready",
    35: "dw2_start",
    36: "dw2_end",
    37: "dx_return_pack_start",
    38: "dx_return_gin_put_issue",
    39: "dx_return_local_completion",
    40: "dx_source_relay",
    41: "dgate_return",
    42: "x_first_local_row_visible",
    43: "x_first_remote_signal",
    44: "x_first_proxy_relay",
    45: "x_first_mtile_ready",
    46: "fc1_claim_start",
    47: "x_last_dispatch_put",
    48: "x_last_relay",
    49: "x_dispatch_complete",
    50: "dy_first_local_row_visible",
    51: "dy_first_remote_signal",
    52: "dy_first_mtile_ready",
    53: "dv_claim_start",
    54: "dy_last_dispatch_put",
    55: "dy_last_relay",
    56: "dy_dispatch_complete",
    57: "return_pack_cta_bytes",
    58: "return_directory_start",
    59: "return_directory_end",
    60: "dispatch_pack_start",
    61: "dispatch_pack_end",
    62: "dispatch_pack_cta_bytes",
    63: "return_gather_start",
    64: "return_gather_end",
    65: "return_gather_cta_bytes",
    66: "dv_math_end",
    67: "dv_tile_end",
    68: "dx_first",
    69: "dx_math_end",
    70: "dx_tile_end",
    71: "fc2_first",
    72: "fc2_last",
    73: "reduce_token_ready_observed",
    74: "local_return_batch_start",
    75: "local_return_batch_end",
    76: "local_return_terminal",
    77: "full_return_chunk_ready",
    78: "full_return_chunk_published",
    79: "fc2_mblock_return_ready",
}

PAIRS = {
    "kernel": (1, 22),
    "dispatch_issue_to_completion": (3, 4),
    "dispatch_signal_to_relay": (5, 6),
    "proxy_relay": (6, 7),
    "fc1": (9, 10),
    "return_pack": (12, 13),
    "return_issue_to_completion": (14, 15),
    "source_relay": (17, 18),
    "reduce": (20, 21),
    "dy_issue_to_completion": (24, 25),
    "dy_relay": (26, 27),
    "dv": (28, 29),
    "dw1_ready_to_start": (31, 32),
    "dw1": (32, 33),
    "dw2_ready_to_start": (34, 35),
    "dw2": (35, 36),
    "dx_return_issue_to_completion": (38, 39),
    "x_first_local_to_first_mtile": (42, 45),
    "x_first_signal_to_first_mtile": (43, 45),
    "x_first_mtile_to_fc1_claim": (45, 46),
    "x_last_put_to_dispatch_complete": (47, 49),
    "x_last_relay_to_dispatch_complete": (48, 49),
    "dy_first_local_to_first_mtile": (50, 52),
    "dy_first_signal_to_first_mtile": (51, 52),
    "dy_first_mtile_to_dv_claim": (52, 53),
    "dy_last_put_to_dispatch_complete": (54, 56),
    "dy_last_relay_to_dispatch_complete": (55, 56),
    "return_directory": (58, 59),
    "dispatch_pack": (60, 61),
    "return_gather": (63, 64),
    "dv_math": (28, 66),
    "dv_tile": (28, 67),
    "dv_epilogue": (66, 67),
    "dx_math": (68, 69),
    "dx_tile": (68, 70),
    "dx_epilogue": (69, 70),
    "fc2": (71, 72),
    "fc1_end_to_fc2_start": (10, 71),
    "local_return_batch": (74, 75),
    "full_return_chunk_publish": (77, 78),
}


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(path: Path) -> tuple[dict[str, object], dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload["events"]
    by_key: dict[tuple[int, int], dict[int, list[int]]] = {}
    event_extents: dict[int, list[int]] = {}
    for item in events:
        event = int(item["event"])
        timestamp = int(item["timestamp_ns"])
        key = (int(item["cta"]), int(item["channel"]))
        by_key.setdefault(key, {}).setdefault(event, []).append(timestamp)
        event_extents.setdefault(event, []).append(timestamp)
    durations: dict[str, list[float]] = {}
    for name, (begin_event, end_event) in PAIRS.items():
        samples: list[float] = []
        for observed in by_key.values():
            begins = observed.get(begin_event, [])
            ends = observed.get(end_event, [])
            for begin, end in zip(begins, ends):
                if end >= begin:
                    samples.append((end - begin) / 1_000_000.0)
        durations[name] = samples
    rank_summary: dict[str, object] = {
        "path": str(path),
        "rank": payload["rank"],
        "phase": payload["phase"],
        "generation": payload["generation"],
        "event_count": len(events),
        "overflow_events": payload["overflow_events"],
        "events": {
            EVENTS.get(event, f"event_{event}"): {
                "first_ns": min(values),
                "last_ns": max(values),
                "count": len(values),
            }
            for event, values in sorted(event_extents.items())
        },
        "return_pack_cta_bytes": {
            str(int(item["cta"])): int(item["arg0"])
            for item in events if int(item["event"]) == 57
        },
        "durations_ms": {
            name: {
                "count": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for name, values in durations.items()
        },
    }
    return rank_summary, durations


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--redact-paths", action="store_true",
        help="store trace basenames instead of machine-local paths")
    args = parser.parse_args()
    rank_summaries = []
    aggregate: dict[str, list[float]] = {}
    for path in sorted(args.traces):
        summary, durations = summarize(path)
        if args.redact_paths:
            summary["path"] = path.name
        rank_summaries.append(summary)
        for name, values in durations.items():
            aggregate.setdefault(name, []).extend(values)
    result = {
        "schema": 2,
        "clock_contract": "durations within one GPU only",
        "ranks": rank_summaries,
        "aggregate_duration_ms": {
            name: {
                "count": len(values),
                "p50": percentile(values, 0.50),
                "p95": percentile(values, 0.95),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
            for name, values in sorted(aggregate.items())
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
