"""PF/DT ring geometry and kernel resource constraints."""

from autotune_common import require
import unified_tuning


DEFAULTS = dict(decode_math_m=0, decode_stages=3, decode_cta_multiplier=0,
                decode_slice_schedule=0,
                decode_m64_min_rows=16,
                decode_narrow_cta_multiplier=0,
                cta_count=0, network_lanes=2, packet_rows=32,
                return_packet_rows=32, network_min_depth=8,
                network_max_workers=32, source_group=0,
                network_prefill_min_tokens=0,
                window_tokens=256, contribution_windows=2,
                local_prefill_streaming=0, local_prefill_fast_math=0,
                local_pull_warps=2, local_compact_return=0,
                local_compact_return_max_tokens=0,
                ordered_dispatch=0, coalesce_get=0, window_ordered_tape=0,
                backward_return_pending=1, backward_wave_blocks=64,
                backward_dynamic_dw=0,
                network_credit_windows=1, network_get_workers=64,
                network_workers=112, network_progress_workers=19,
                backward_return_spans=0, backward_striped_signals=0,
                network_private_dr=0, network_vector_pull=0, return_rails=1)


MACROS = {
    "return_rails": "WEFTMOE_PF_RETURN_RAILS",
    "network_private_dr": "WEFTMOE_PF_NETWORK_PRIVATE_DR",
    "network_vector_pull": "WEFTMOE_PF_NETWORK_VECTOR_PULL",
    "backward_return_spans": "WEFTMOE_PF_BWD_RETURN_SPANS",
    "backward_striped_signals": "WEFTMOE_PF_BWD_STRIPED_SIGNALS",
    "network_get_workers": "WEFTMOE_PF_NETWORK_GET_WORKERS",
    "network_workers": "WEFTMOE_PF_NETWORK_WORKERS",
    "network_progress_workers": "WEFTMOE_PF_NETWORK_PROGRESS_WORKERS",
    "network_credit_windows": "WEFTMOE_PF_CREDIT_WINDOWS",
    "ordered_dispatch": "WEFTMOE_PF_ORDERED_DISPATCH",
    "coalesce_get": "WEFTMOE_PF_COALESCE_GET",
    "window_ordered_tape": "WEFTMOE_PF_WINDOW_ORDERED_TAPE",
    "backward_return_pending": "WEFTMOE_PF_BWD_RETURN_PENDING",
    "backward_wave_blocks": "WEFTMOE_PF_BWD_WAVE_BLOCKS",
    "backward_dynamic_dw": "WEFTMOE_PF_BWD_DYNAMIC_DW",
    "local_pull_warps": "WEFTMOE_PF_LOCAL_PULL_WARPS",
    "local_compact_return": "WEFTMOE_PF_LOCAL_COMPACT_RETURN",
    "local_compact_return_max_tokens": "WEFTMOE_PF_LOCAL_COMPACT_RETURN_MAX_TOKENS",
    "local_prefill_streaming": "WEFTMOE_FWD_LOCAL_PREFILL_STREAMING",
    "local_prefill_fast_math": "WEFTMOE_FWD_LOCAL_PREFILL_FAST_MATH",
    "decode_math_m": "WEFTMOE_FWD_LOCAL_DECODE_M",
    "decode_slice_schedule": "WEFTMOE_DT_SLICE_SCHEDULE",
    "decode_m64_min_rows": "WEFTMOE_FWD_LOCAL_DECODE_M64_MIN_ROWS",
    "decode_narrow_cta_multiplier": "WEFTMOE_FWD_LOCAL_DECODE_NARROW_CTA_MULTIPLIER",
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
                   return_rails=(0, 1, 2, 4, 8),
                   local_pull_warps=(1, 2, 3),
                   local_compact_return=(0, 1),
                   backward_dynamic_dw=(0, 1),
                   ordered_dispatch=(0, 1), coalesce_get=(0, 1),
                   window_ordered_tape=(0, 1), backward_return_pending=(1, 2, 4, 8),
                   backward_return_spans=(0, 1), backward_striped_signals=(0, 1),
                   network_private_dr=(0, 1), network_vector_pull=(0, 1),
                   decode_math_m=(0, 16, 64), decode_stages=(2, 3),
                   decode_slice_schedule=(0, 1, 2, 3),
                   decode_cta_multiplier=(0, 1, 2, 3),
                   decode_narrow_cta_multiplier=(0, 1, 2, 3),
                   packet_rows=(8, 16, 32, 64, 128),
                   return_packet_rows=(16, 32, 64, 128),
                   network_max_workers=(1, 2, 4, 8, 16, 32),
                   source_group=(0, 1, 2, 4, 8, 16))
    for key, allowed in choices.items():
        require(v[key] in allowed, f"V4 {key} must be one of {allowed}")
    require(v["decode_m64_min_rows"] > 0, "decode M64 row threshold must be positive")
    require(v["network_lanes"] > 0 and v["network_min_depth"] > 0,
            "network lanes and ring depth must be positive")
    require(0 < v["network_get_workers"] < v["network_workers"],
            "dispatch and return must both own network workers")
    require(v["network_progress_workers"] > 0, "network progress needs at least one worker")
    require(not v["cta_count"] or
            v["network_workers"] + v["network_progress_workers"] + 1 <= v["cta_count"],
            "network workers exceed the configured CTA count")
    require(1 <= v["network_credit_windows"] <= 8,
            "network credit windows exceed the compiled eight-window directory")
    require(not v["backward_return_spans"] or v["window_ordered_tape"],
            "precomputed Return spans require a window-ordered tape")
    require(not v["backward_striped_signals"] or
            v["network_lanes"] >= v["network_workers"] - v["network_get_workers"],
            "striped Return signals require one channel lane per Put worker")
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
