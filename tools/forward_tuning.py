"""Forward tuning parameters and compiled-kernel resource constraints."""

from typing import Any, Mapping

from autotune_common import require as _require, require_choice as _require_choice
import unified_tuning
import pf_tuning as v4_tuning


PARAMETER_MACROS = {
    "main_hidden": "WEFTMOE_FWD_MAIN_HIDDEN",
    "main_intermediate": "WEFTMOE_FWD_MAIN_INTERMEDIATE",
    "tile_m": "WEFTMOE_FWD_TILE_M",
    "fc1_tile_n": "WEFTMOE_FWD_FC1_TILE_N",
    "fc2_tile_n": "WEFTMOE_FWD_FC2_TILE_N",
    "tile_k": "WEFTMOE_FWD_TILE_K",
    "fc1_stages": "WEFTMOE_FWD_FC1_STAGES",
    "fc2_stages": "WEFTMOE_FWD_FC2_STAGES",
    "static_v4_fc1_stages": "WEFTMOE_FWD_STATIC_V4_FC1_STAGES",
    "static_v4_fc2_stages": "WEFTMOE_FWD_STATIC_V4_FC2_STAGES",
    "static_v4_prefill_min_tokens": "WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS",
    "fc2_claim_tiles": "WEFTMOE_FWD_FC2_CLAIM_TILES",
    "opportunistic_reduce_warps": "WEFTMOE_FWD_OPPORTUNISTIC_REDUCE_WARPS",
    "gin_channel_pack_limit": "WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT",
    "gin_receive_rows": "WEFTMOE_FWD_GIN_RECEIVE_ROWS",
    "gin_fanout_lanes": "WEFTMOE_FWD_GIN_FANOUT_LANES",
    "compute_threads": "WEFTMOE_FWD_COMPUTE_THREADS",
    "math_warpgroups": "WEFTMOE_FWD_MATH_WARPGROUPS",
    "producer_warpgroups": "WEFTMOE_FWD_PRODUCER_WARPGROUPS",
    "math_registers": "WEFTMOE_FWD_MATH_REGISTERS",
    "producer_registers": "WEFTMOE_FWD_PRODUCER_REGISTERS",
    "wgmma_wait_group": "WEFTMOE_FWD_WGMMA_WAIT_GROUP",
    "fc2_store_atom_n": "WEFTMOE_FWD_FC2_STORE_ATOM_N",
    "cluster_size": "WEFTMOE_FWD_CLUSTER_SIZE",
    "fc2_tma_store": "WEFTMOE_FWD_FC2_TMA_STORE",
    "rdma_ctas": "WEFTMOE_FWD_RDMA_CTAS",
    "relay_ctas": "WEFTMOE_FWD_RELAY_CTAS",
    "relay_tma_warps": "WEFTMOE_FWD_RELAY_TMA_WARPS",
    "local_dispatch_tma_warps": "WEFTMOE_FWD_LOCAL_DISPATCH_TMA_WARPS",
    "relay_scatter_warps": "WEFTMOE_FWD_RELAY_SCATTER_WARPS",
    "local_return_batch_rows":
        "WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS",
    "dispatch_publish_batch_rows":
        "WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS",
    "relay_local_return": "WEFTMOE_FWD_RELAY_LOCAL_RETURN",
    "local_route_batch_rows": "WEFTMOE_FWD_LOCAL_ROUTE_BATCH_ROWS",
    "startup_mblocks": "WEFTMOE_FWD_STARTUP_MBLOCKS",
    "fc1_locality_group": "WEFTMOE_FWD_FC1_LOCALITY_GROUP",
    "fc1_locality_group4_begin":
        "WEFTMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN",
    "fc1_locality_group4_end": "WEFTMOE_FWD_FC1_LOCALITY_GROUP4_END",
    "fc2_priority_burst": "WEFTMOE_FWD_FC2_PRIORITY_BURST",
    "task_claim_chunk": "WEFTMOE_FWD_TASK_CLAIM_CHUNK",
    "task_idle_nanosleep": "WEFTMOE_FWD_TASK_IDLE_NANOSLEEP",
    "transport_idle_nanosleep": "WEFTMOE_FWD_TRANSPORT_IDLE_NANOSLEEP",
    "setup_nanosleep": "WEFTMOE_FWD_SETUP_NANOSLEEP",
    "completion_peer_probes":
        "WEFTMOE_FWD_COMPLETION_PEER_PROBES",
    "reduce_scheduler": "WEFTMOE_FWD_REDUCE_SCHEDULER",
    "reduce_warps": "WEFTMOE_FWD_REDUCE_WARPS",
    "reduce_tokens_per_claim": "WEFTMOE_FWD_REDUCE_TOKENS_PER_CLAIM",
    "reduce_registers": "WEFTMOE_FWD_REDUCE_REGISTERS",
    "return_idle_nanosleep": "WEFTMOE_FWD_RETURN_IDLE_NANOSLEEP",
    "reduce_queue_nanosleep": "WEFTMOE_FWD_REDUCE_QUEUE_NANOSLEEP",
    "finalize_nanosleep": "WEFTMOE_FWD_FINALIZE_NANOSLEEP",
    "unified_workers": "WEFTMOE_FWD_UNIFIED_WORKERS",
    **unified_tuning.parameter_macros("forward"),
}


BOOLEAN_PARAMETERS = {
    "fc2_tma_store",
    "relay_local_return",
}


ENUM_PARAMETERS = {
    "reduce_scheduler": {
        "queue": 0,
        "scan": 1,
        "static": 2,
    },
}


def validate_tuning(config: Mapping[str, Any]) -> None:
    if "v4" in config:
        v4_tuning.validate(config)
        config = {k: v for k, v in config.items() if k != "v4"}
    _require(set(config) == set(PARAMETER_MACROS),
             "expanded tuning config must define every parameter exactly once")
    for key, value in config.items():
        if key in BOOLEAN_PARAMETERS:
            _require(type(value) is bool, f"{key} must be true or false")
        elif key in ENUM_PARAMETERS:
            _require(value in ENUM_PARAMETERS[key],
                     f"{key}={value!r}, expected one of "
                     f"{tuple(ENUM_PARAMETERS[key])}")
        else:
            _require(type(value) is int, f"{key} must be an integer")
            _require(value >= 0, f"{key} must be non-negative")

    unified_tuning.validate(config)
    _require(1 <= config["opportunistic_reduce_warps"] <= 8,
             "opportunistic_reduce_warps must be in [1, 8]")
    _require_choice(config, "gin_fanout_lanes", (1, 2, 4, 8, 16, 32))
    _require(1 <= config["gin_receive_rows"] <= 128,
             "gin_receive_rows must be in [1, 128]")
    _require(1 <= config["gin_channel_pack_limit"] <= 8,
             "gin_channel_pack_limit must be in [1, 8]")
    _require(config["main_hidden"] > 0 and config["main_hidden"] % 256 == 0,
             "main_hidden must be a positive multiple of 256")
    _require(config["main_intermediate"] > 0 and config["main_intermediate"] % 256 == 0,
             "main_intermediate must be a positive multiple of 256")
    _require(config["tile_m"] == 128,
             "cluster multicast requires tile_m=128")
    _require_choice(config, "fc1_tile_n", (64, 128, 256))
    _require_choice(config, "fc2_tile_n", (64, 128, 256))
    _require_choice(config, "tile_k", (64, 128))
    _require_choice(config, "fc1_stages", (2, 3, 4))
    _require_choice(config, "fc2_stages", (2, 3, 4))
    _require_choice(config, "static_v4_fc1_stages", (2, 3, 4, 5, 6))
    _require_choice(config, "static_v4_fc2_stages", (2, 3, 4, 5, 6))
    _require_choice(config, "fc2_claim_tiles", (1, 2, 4, 8))
    _require(config["static_v4_prefill_min_tokens"] >= 0,
             "static_v4_prefill_min_tokens must be nonnegative")
    if config["static_v4_prefill_min_tokens"]:
        _require(tuple(config[k] for k in (
            "fc1_tile_n", "fc2_tile_n", "tile_k", "static_v4_fc1_stages",
            "static_v4_fc2_stages", "fc2_claim_tiles")) == (64, 128, 128, 3, 3, 1),
            "wide prefill requires N64/N128 K128 3/3-stage claim1 decode profile")
    _require(config["compute_threads"] == 384,
             "the current producer + two-math-WG kernel requires 384 threads")
    _require(config["math_warpgroups"] == 2,
             "the current WGMMA mapping requires two math warpgroups")
    _require(config["producer_warpgroups"] == 1,
             "the current TMA producer requires one producer warpgroup")
    for key in ("math_registers", "producer_registers", "reduce_registers"):
        _require(1 <= config[key] <= 255, f"{key} must be in [1, 255]")
    compute_registers = 128 * (
        config["math_warpgroups"] * config["math_registers"] +
        config["producer_warpgroups"] * config["producer_registers"])
    _require(compute_registers <= 65536,
             f"compute register allocation {compute_registers} exceeds 65536")
    reduce_registers = config["compute_threads"] * config["reduce_registers"]
    _require(reduce_registers <= 65536,
             f"reduce register allocation {reduce_registers} exceeds 65536")
    _require_choice(config, "wgmma_wait_group", (0, 1))
    _require(config["fc2_store_atom_n"] == 64,
             "the current STSM/TMA epilogue store atom is N64")
    _require(config["cluster_size"] == 2,
             "Forward requires cluster_size=2 for TMA multicast")
    _require(config["fc2_tma_store"],
             "Forward requires the FC2 TMA-store epilogue")
    if not config["static_v4_prefill_min_tokens"]:
        unified_tuning.require_resources(
            config["fc1_tile_n"] == config["fc2_tile_n"] and
            config["fc1_tile_n"] in (128, 256) and config["tile_k"] == 64 and
            config["static_v4_fc1_stages"] == config["static_v4_fc2_stages"],
            "PF requires equal N128/256, K64 and a shared FC1/FC2 stage ring")
    unified_tuning.require_resources(
        dynamic_shared_memory(config, 16)["compute_bytes"] + 1024 <= 227 * 1024,
        "PF/DT shared memory exceeds Hopper CTA capacity")

    # Two-host GIN owns eight physical contexts. One-node NVLink has no RDMA
    # owner and tunes Relay independently in small four-CTA increments.
    _require_choice(config, "rdma_ctas", (0, 8))
    if config["rdma_ctas"] == 0:
        _require(config["relay_ctas"] >= 4 and config["relay_ctas"] % 4 == 0,
                 "relay_ctas must be a positive multiple of 4")
        _require(config["local_dispatch_tma_warps"] < 512 // 32,
                 "local split Relay reserves one FC1-ready publisher warp")
    else:
        _require(config["relay_ctas"] >= config["rdma_ctas"],
                 "relay_ctas must cover every RDMA context")
        _require(config["relay_ctas"] % config["rdma_ctas"] == 0,
                 "relay_ctas must be a multiple of rdma_ctas")
    _require(config["relay_local_return"], "local Return requires relay_local_return=true")
    _require_choice(config, "relay_tma_warps", (0, 4, 8, 12))
    _require_choice(config, "local_dispatch_tma_warps", (4, 8, 12))
    _require_choice(config, "relay_scatter_warps", (4, 8, 12))
    _require_choice(config, "local_return_batch_rows", (32, 64, 128, 256, 512, 1024))
    _require_choice(config, "dispatch_publish_batch_rows",
                    (4, 8, 16, 32, 64, 128))

    _require_choice(config, "local_route_batch_rows", (32, 64, 128, 256))
    _require_choice(config, "startup_mblocks", (1, 2, 4))
    _require_choice(config, "fc1_locality_group", (1, 2, 4))
    begin = config["fc1_locality_group4_begin"]
    end = config["fc1_locality_group4_end"]
    _require(begin >= 2 and begin % 2 == 0 and end >= begin and end <= 255 and
             (end - begin) % 4 == 0,
             "FC1 locality window must be even, ordered, <=255, and span % 4 == 0")
    _require_choice(config, "fc2_priority_burst", (0, 2, 4, 8))
    _require(config["task_claim_chunk"] == 1,
             "only the proven one-task work-queue claim is implemented")

    _require_choice(config, "completion_peer_probes", (1, 2, 4))
    _require_choice(config, "reduce_warps", (4, 8, 12))
    _require(1 <= config["reduce_tokens_per_claim"] <= config["reduce_warps"],
             "reduce_tokens_per_claim must be in [1, reduce_warps]")
    for key in (
        "task_idle_nanosleep",
        "transport_idle_nanosleep",
        "setup_nanosleep",
        "return_idle_nanosleep",
        "reduce_queue_nanosleep",
        "finalize_nanosleep",
    ):
        _require_choice(config, key, (0, 16, 32, 64, 128, 256, 512))


def dynamic_shared_memory(config: Mapping[str, Any], world: int) -> dict[str, int]:
    if config["static_v4_prefill_min_tokens"]:
        # DT uses a fixed M64 tile.
        stages = config.get("v4", {}).get("decode_stages", 3)
        payload = 3 * 1024 + stages * (64 + 128) * 64 * 2 + 64 * 128 * 2 + 3 * 4096
        size = (payload + (2 * stages + 3) * 8 + 1023) // 1024 * 1024
    else:
        size = v4_tuning.local_streaming_shared_bytes(config)
    return {"compute_bytes": size, "transport_bytes": 0}
