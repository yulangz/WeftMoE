"""V4 search axes and graph fixtures, shared by the Forward autotuner.

Physical M stays 128. Decode M only slices arithmetic inside that directory;
network geometry remains a collective, prepare-time choice.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import subprocess
from pathlib import Path

from autotune_common import candidate_digest, require
import unified_tuning


# Runtime capacities are kept in the candidate digest but not in CUDA headers.
DEFAULTS = dict(decode_math_m=0, decode_stages=3, decode_cta_multiplier=0,
                cta_count=0, network_lanes=2, packet_rows=32,
                return_packet_rows=32, network_min_depth=8,
                network_max_workers=32, source_group=0,
                network_prefill_min_tokens=0,
                window_tokens=256, contribution_windows=2,
                local_prefill_streaming=0, local_prefill_fast_math=0)
MACROS = {
    "local_prefill_streaming": "WEFTMOE_FWD_LOCAL_PREFILL_STREAMING",
    "local_prefill_fast_math": "WEFTMOE_FWD_LOCAL_PREFILL_FAST_MATH",
    "decode_math_m": "WEFTMOE_FWD_LOCAL_DECODE_M",
    "decode_stages": "WEFTMOE_FWD_LOCAL_DECODE_STAGES",
    "decode_cta_multiplier": "WEFTMOE_FWD_LOCAL_DECODE_CTA_MULTIPLIER",
    "cta_count": "WEFTMOE_FWD_STATIC_V4_CTA_COUNT",
    "network_lanes": "WEFTMOE_FWD_STATIC_V4_NETWORK_LANES",
    "packet_rows": "WEFTMOE_FWD_STATIC_V4_PACKET_ROWS",
    "return_packet_rows": "WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS",
    "network_min_depth": "WEFTMOE_FWD_STATIC_V4_NETWORK_MIN_DEPTH",
    "network_max_workers": "WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS",
    "source_group": "WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP",
    "network_prefill_min_tokens": "WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS",
}


def local_streaming_shared_bytes(config):
    """Mirror local::Shared's independently aligned stages and output arena."""
    n, stages = config["fc1_tile_n"], config["static_v4_fc1_stages"]
    payload = stages * (128 + n) * 64 * 2 + 128 * n * 2 + 3 * 4096
    return (payload + (2 * stages + 3) * 8 + 1023) // 1024 * 1024


def validate(config):
    v = config["v4"]
    require(set(v) == set(DEFAULTS), "incomplete V4 candidate")
    require(all(type(x) is int and x >= 0 for x in v.values()),
            "V4 parameters must be nonnegative integers")
    choices = dict(local_prefill_streaming=(0, 1), local_prefill_fast_math=(0, 1),
                   decode_math_m=(0, 16, 64), decode_stages=(2, 3),
                   decode_cta_multiplier=(0, 1, 2, 3),
                   network_lanes=(1, 2, 4, 8, 16, 32, 64, 132), packet_rows=(8, 16, 32, 64, 128),
                   return_packet_rows=(16, 32, 64, 128),
                   network_min_depth=(8, 16, 32),
                   network_max_workers=(1, 2, 4, 8, 16, 32),
                   source_group=(0, 1, 2, 4, 8, 16))
    for key, allowed in choices.items():
        require(v[key] in allowed, f"V4 {key} must be one of {allowed}")
    require(v["cta_count"] % 2 == 0, "V4 cta_count must be even")
    require(v["window_tokens"] > 0 and v["contribution_windows"] > 0,
            "V4 window and contribution capacity must be positive")
    require(config["tile_m"] == 128 and config["unified_workers"] == 1 and
            (config["unified_threads"], config["unified_comm_base"],
             config["unified_comm_threads"]) == (384, 288, 96),
            "V4 ring requires physical M128 and the 256/32/96 partition")
    if v["local_prefill_streaming"]:
        tile_n = config["fc1_tile_n"]
        require(tile_n in (128, 256) and config["fc2_tile_n"] == tile_n and
                (config["tile_k"], config["cluster_size"]) == (64, 2) and
                config["static_v4_prefill_min_tokens"] == 0 and
                v["network_prefill_min_tokens"] == 0,
                "local streaming requires equal N128/256, K64, cluster2 without a companion entry")
        require(config["static_v4_fc1_stages"] == config["static_v4_fc2_stages"],
                "local streaming FC1/FC2 share one stage ring")
        # Include the compiler's 1 KiB static shared reservation.
        unified_tuning.require_resources(
            local_streaming_shared_bytes(config) + 1024 <= 227 * 1024,
            "local streaming shared memory exceeds the SM90 CTA budget")
    if v["decode_math_m"] or v["network_prefill_min_tokens"]:
        require((config["fc1_tile_n"], config["fc2_tile_n"], config["tile_k"],
                 config["static_v4_fc1_stages"], config["static_v4_fc2_stages"],
                 config["fc2_claim_tiles"]) == (64, 128, 128, 3, 3, 1),
                "decode/prefill companion requires the narrow 3/3-stage profile")
    if v["decode_math_m"]:
        require(config["static_v4_prefill_min_tokens"] > 0,
                "decode M requires the local decode entry")


def expand(candidates, spec, benchmark):
    spec = copy.deepcopy(spec)
    space = spec.get("search_space", {})
    require(not (set(space) - set(DEFAULTS)), "unknown V4 search axis")
    axes = {k: space.get(k, [value]) for k, value in DEFAULTS.items()}
    for key, values in axes.items():
        require(isinstance(values, list) and values and
                len(set(values)) == len(values), f"invalid V4 axis {key}")
    local = benchmark["nodes"] == 1
    if spec.get("interconnect") == "auto":
        spec["interconnect"] = "nvlink" if local else "nvlink+gin"
    if local:
        # Network geometry has no effect on the NVLink-only dataflow.
        for key in ("network_lanes", "packet_rows", "return_packet_rows",
                    "network_min_depth", "network_max_workers",
                    "network_prefill_min_tokens"):
            axes[key] = axes[key][:1]
    require(len(candidates) * math.prod(map(len, axes.values())) <= 4096,
            "combined Forward/V4 grid exceeds 4096 candidates")
    for key in ("gpu", "interconnect"):
        require(isinstance(spec.get(key), str) and spec[key], f"V4 omits {key}")
    require(spec.get("input_layout", "dp") == "dp",
            "native graph fixtures measure DP-local inputs, not the TP adapter")
    require(type(spec.get("memory_limit_bytes")) is int and spec["memory_limit_bytes"] > 0,
            "V4 requires an explicit per-rank workspace memory_limit_bytes")
    require(type(spec.get("sm_clock_mhz")) is int and spec["sm_clock_mhz"] > 0,
            "V4 requires sm_clock_mhz")
    cases = spec.get("cases")
    require(isinstance(cases, list) and cases, "V4 requires route cases")
    for case in cases:
        require(set(case) <= {"route", "seed", "tokens", "validate_only"}, "unknown V4 case field")
        require(case.get("route") in ("balanced", "hotspot", "uneven"),
                "V4 route must be balanced, hotspot or uneven")
        require(type(case.get("seed")) is int and case["seed"] >= 0,
                "V4 case seed must be nonnegative")
        require(type(case.get("tokens", 1)) is int and case.get("tokens", 1) > 0,
                "V4 case tokens must be positive")
        require(case.get("tokens", benchmark["tokens_per_rank"]) <= benchmark["tokens_per_rank"],
                "V4 validation tokens exceed selected capacity")
    require(any(not c.get("validate_only", False) for c in cases),
            "V4 needs a measured selection case")
    if benchmark["tokens_per_rank"] > 1:
        for case in [c for c in cases if not c.get("validate_only", False)]:
            require(any(other["route"] == case["route"] and other["seed"] == case["seed"]
                        and other.get("tokens") == 1 for other in cases),
                    "each measured route/seed needs a tokens=1 batch-invariance fixture")
    else:
        spec["cases"] = [c for c in cases if not c.get("validate_only", False)]
    result = []
    for candidate, values in itertools.product(candidates, itertools.product(*axes.values())):
        config = dict(candidate["config"], v4=dict(zip(axes, values)))
        validate(config)
        v = config["v4"]
        group = v["source_group"]
        require(not group or benchmark["gpus_per_node"] % group == 0,
                "source_group must divide GPUs per node")
        if not local:
            require(v["decode_math_m"] == 0 and v["decode_stages"] == 3 and
                    v["decode_cta_multiplier"] == 0,
                    "network ring implements M128, not local decode arithmetic")
        decode_entry = (local and benchmark["tokens_per_rank"] <= 64 and
                        benchmark["experts"] <= 256 and
                        (config["static_v4_prefill_min_tokens"] or
                         v["network_prefill_min_tokens"]) and
                        config["fc2_tile_n"] == 128 and
                        v["window_tokens"] >= benchmark["tokens_per_rank"] and
                        (not group or group == benchmark["gpus_per_node"]))
        if not decode_entry:
            require((v["decode_math_m"], v["decode_stages"], v["decode_cta_multiplier"])
                    == (0, 3, 0), "decode axes require the single-window decode entry")
        name = "incumbent" if not result else f"v4_{len(result):04d}"
        result.append(dict(candidate, name=name, config=config))
    benchmark.update(graph_replay=True, selection_metric="forward_slowest_rank_us.mean",
                     v4=spec)
    return result


def header(config):
    v = config["v4"]
    return [f"#define {macro} {v[key]}" for key, macro in MACROS.items()]


def compiled_digest(config):
    if "v4" not in config:
        return candidate_digest(config)
    return candidate_digest(dict(config, v4={k: config["v4"][k] for k in MACROS}))


def environment(config):
    v = config["v4"]
    return dict(WEFTMOE_FORWARD_BENCH_GRAPH_REPLAY="1", WEFTMOE_FORWARD_GRAPH_BATCH="10",
                WEFTMOE_STATIC_RING_TOKENS=str(v["window_tokens"]),
                WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS=str(v["contribution_windows"]))


def aggregate(runs, spec):
    """All fixtures must pass; equal-weight case means avoid favoring long runs."""
    records = []
    for run in runs:
        require(run["returncode"] == 0 and run["record"] is not None,
                f"V4 fixture failed: {run['log']}")
        record = run["record"]
        require(record.get("static_schedule") == 1 and record.get("ring_window_tokens", 0) > 0,
                "V4 fixture did not enter static ring inference")
        require(max(record["per_rank_device_bytes"]) <= spec["memory_limit_bytes"],
                "candidate exceeds workspace memory budget")
        records.append(record)
    prefixes = {}
    for case, record in zip(spec["cases"], records):
        key = (case["route"], case["seed"])
        prefix = record["per_rank_prefix_hash"]
        require(prefixes.setdefault(key, prefix) == prefix,
                "candidate violates batch invariance across validation shapes")
    latencies = [r["slowest_rank_us"]["mean"] for c, r in zip(spec["cases"], records)
                 if not c.get("validate_only", False)]
    require(all(math.isfinite(x) and x > 0 for x in latencies),
            "invalid V4 timing sample")
    samples = sorted(value for case, record in zip(spec["cases"], records)
                     if not case.get("validate_only", False)
                     for value in record["slowest_rank_samples_us"])
    return dict(benchmark="static-v4-route-suite", measured=records[0]["measured"],
                forward_slowest_rank_us=dict(mean=sum(latencies) / len(latencies),
                                             p95=samples[math.ceil(0.95 * len(samples)) - 1]),
                max_case_mean_us=max(latencies), cases=records)


def check_clocks(benchmark):
    spec = benchmark["v4"]
    query = ["nvidia-smi", "--query-gpu=name,clocks.sm", "--format=csv,noheader,nounits"]
    commands = [query]
    if benchmark["nodes"] > 1:
        commands.append(["ssh", "-p", str(benchmark["port"]),
                         benchmark["remote"], *query])
    for command in commands:
        output = subprocess.check_output(command, text=True, timeout=20)
        devices = [line.rsplit(",", 1) for line in output.strip().splitlines()]
        require(len(devices) >= benchmark["gpus_per_node"], "insufficient GPUs for autotune")
        for name, clock in devices[:benchmark["gpus_per_node"]]:
            require(name.strip() == spec["gpu"] and int(clock) == spec["sm_clock_mhz"],
                    "GPU/SM clock does not match V4 key; lock clocks before tuning")


def fingerprint(record):
    return [dict(shape=r["shape"], inputs=r["per_rank_input_hash"],
                 outputs=r["per_rank_output_hash"], tokens=r["per_rank_tokens"])
            for r in record["cases"]]


def export_table(path: Path, result, header_path: Path):
    """Publish a measured, exact-capacity entry, never a guessed bucket range."""
    b, selected = result["benchmark"], result["selected"]
    spec = b["v4"]
    config = selected["config"]
    key = dict(gpu=spec["gpu"], interconnect=spec["interconnect"], dtype="bfloat16",
               hidden=config["main_hidden"], intermediate=config["main_intermediate"],
               experts=b["experts"], top_k=b["top_k"], nodes=b["nodes"],
               gpus_per_node=b["gpus_per_node"], max_tokens=b["tokens_per_rank"],
               input_layout="dp", training=False)
    entry = dict(key=key, config_digest=candidate_digest(config),
                 compiled_digest=compiled_digest(config),
                 header=str(header_path), window_tokens=config["v4"]["window_tokens"],
                 contribution_windows=config["v4"]["contribution_windows"],
                 sm_clock_mhz=spec["sm_clock_mhz"],
                 metric_us=selected["final_metric_us"],
                 cases=spec["cases"])
    table = json.loads(path.read_text()) if path.exists() else dict(schema=1, entries=[])
    require(table.get("schema") == 1, "unsupported V4 table schema")
    table["entries"] = [e for e in table["entries"] if e["key"] != key] + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(table, indent=2) + "\n")
    temporary.replace(path)
