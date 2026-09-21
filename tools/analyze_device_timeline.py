#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Summarize per-GPU WeftMoE %globaltimer traces.

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
    42: "x_first_local_row_visible",
    43: "x_first_remote_signal",
    44: "x_first_proxy_relay",
    45: "x_first_mtile_ready",
    46: "fc1_claim_start",
    47: "x_last_dispatch_put",
    48: "x_last_relay",
    49: "x_dispatch_complete",
    57: "return_pack_cta_bytes",
    58: "return_directory_start",
    59: "return_directory_end",
    60: "dispatch_pack_start",
    61: "dispatch_pack_end",
    62: "dispatch_pack_cta_bytes",
    63: "return_gather_start",
    64: "return_gather_end",
    65: "return_gather_cta_bytes",
    71: "fc2_first",
    72: "fc2_last",
    73: "reduce_token_ready_observed",
    74: "local_return_batch_start",
    75: "local_return_batch_end",
    76: "local_return_terminal",
    77: "full_return_chunk_ready",
    78: "full_return_chunk_published",
    79: "fc2_mblock_return_ready",
    80: "compute_select_start",
    81: "compute_select_end",
    82: "fc1_tile_start",
    83: "fc1_math_end",
    84: "fc1_tile_end",
    85: "fc2_tile_start",
    86: "fc2_math_end",
    87: "fc2_tile_end",
    88: "compute_pair_join_end",
    89: "transport_role_done",
    90: "gin_sender_select",
    91: "gin_sender_submit",
    92: "gin_sender_progress",
    93: "source_signal_ready",
    94: "source_packet_claimed",
    95: "setup_done",
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
    "x_first_local_to_first_mtile": (42, 45),
    "x_first_signal_to_first_mtile": (43, 45),
    "x_first_mtile_to_fc1_claim": (45, 46),
    "x_last_put_to_dispatch_complete": (47, 49),
    "x_last_relay_to_dispatch_complete": (48, 49),
    "return_directory": (58, 59),
    "dispatch_pack": (60, 61),
    "return_gather": (63, 64),
    "fc2": (71, 72),
    "fc1_end_to_fc2_start": (10, 71),
    "local_return_batch": (74, 75),
    "full_return_chunk_publish": (77, 78),
    "compute_select": (80, 81),
    "compute_pair_join": (80, 88),
    "fc1_tile": (82, 84),
    "fc1_math": (82, 83),
    "fc1_epilogue": (83, 84),
    "fc2_tile": (85, 87),
    "fc2_math": (85, 86),
    "fc2_epilogue": (86, 87),
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
    selection_by_cta: dict[int, dict[int, list[int]]] = {}
    event_extents: dict[int, list[int]] = {}
    for item in events:
        event = int(item["event"])
        timestamp = int(item["timestamp_ns"])
        key = (int(item["cta"]), int(item["channel"]))
        by_key.setdefault(key, {}).setdefault(event, []).append(timestamp)
        if event in (80, 81):
            # SelectEnd's channel records the selected task kind; SelectStart
            # necessarily precedes that choice. Each CTA has one selector.
            selection_by_cta.setdefault(key[0], {}).setdefault(event, []).append(timestamp)
        event_extents.setdefault(event, []).append(timestamp)
    durations: dict[str, list[float]] = {}
    for name, (begin_event, end_event) in PAIRS.items():
        samples: list[float] = []
        groups = selection_by_cta if name == "compute_select" else by_key
        for observed in groups.values():
            begins = observed.get(begin_event, [])
            ends = observed.get(end_event, [])
            for begin, end in zip(begins, ends):
                if end >= begin:
                    samples.append((end - begin) / 1_000_000.0)
        durations[name] = samples
    for event in (90, 91, 92):
        durations[EVENTS[event]] = [int(item["arg0"]) / 1_000_000.0
                                    for item in events if int(item["event"]) == event]
    # The independent observer may miss a packet claimed between its polls.
    # Match identities, not event positions; report only observed pairs.
    ready = {(int(item["channel"]), int(item["arg0"])): int(item["timestamp_ns"])
             for item in events if int(item["event"]) == 93}
    durations["return_ready_to_claim"] = [
        (int(item["timestamp_ns"]) - ready[key]) / 1_000_000.0
        for item in events if int(item["event"]) == 94
        if (key := (int(item["channel"]), int(item["arg0"]))) in ready
        and int(item["timestamp_ns"]) >= ready[key]]
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
