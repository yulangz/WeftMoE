"""Offline V4 results: exact workload lookup, without compiling or benchmarking.

A table selects a whole prepared pool. Loading another communication profile
requires rebuilding/loading its native library and preparing fresh graphs.
"""

import json
from pathlib import Path


def select_profile(path, key, compiled_id):
    table = json.loads(Path(path).read_text())
    if table.get("schema") != 1:
        raise ValueError("unsupported WeftMoE autotune table schema")
    matches = [entry for entry in table["entries"] if entry["key"] == key]
    if not matches:
        raise ValueError(f"no validated autotune entry for {key}")
    if len(matches) != 1:
        raise ValueError("ambiguous autotune entries for the same workload")
    entry = matches[0]
    if entry["compiled_digest"] != compiled_id:
        raise ValueError(
            f"autotune profile {entry['compiled_digest']} requires a native rebuild "
            f"using {entry['header']}; loaded profile is {compiled_id}"
        )
    return {name: entry[name] for name in
            ("config_digest", "window_tokens", "contribution_windows")}
