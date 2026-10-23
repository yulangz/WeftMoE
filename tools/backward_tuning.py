"""Backward tuning parameters and compiled-kernel resource constraints."""

from typing import Any, Mapping

from autotune_common import require, require_choice
import unified_tuning


PARAMETER_MACROS = {
    "main_hidden": "WEFTMOE_BWD_MAIN_HIDDEN",
    "main_intermediate": "WEFTMOE_BWD_MAIN_INTERMEDIATE",
    "tile_m": "WEFTMOE_BWD_TILE_M",
    "dab_tile_n": "WEFTMOE_BWD_DAB_TILE_N",
    "dx_tile_n": "WEFTMOE_BWD_DX_TILE_N",
    "dx_claim_tiles": "WEFTMOE_BWD_DX_CLAIM_TILES",
    "return_window_scale": "WEFTMOE_BWD_RETURN_WINDOW_SCALE",
    "dw_tile_m": "WEFTMOE_BWD_DW_TILE_M",
    "dw_tile_n": "WEFTMOE_BWD_DW_TILE_N",
    "tile_k": "WEFTMOE_BWD_TILE_K",
    "activation_tile_k": "WEFTMOE_BWD_ACTIVATION_TILE_K",
    "dab_stages": "WEFTMOE_BWD_DAB_STAGES",
    "dx_stages": "WEFTMOE_BWD_DX_STAGES",
    "dw1_stages": "WEFTMOE_BWD_DW1_STAGES",
    "dw2_stages": "WEFTMOE_BWD_DW2_STAGES",
    "compute_threads": "WEFTMOE_BWD_COMPUTE_THREADS",
    "math_warpgroups": "WEFTMOE_BWD_MATH_WARPGROUPS",
    "producer_warpgroups": "WEFTMOE_BWD_PRODUCER_WARPGROUPS",
    "math_registers": "WEFTMOE_BWD_MATH_REGISTERS",
    "producer_registers": "WEFTMOE_BWD_PRODUCER_REGISTERS",
    "rdma_ctas": "WEFTMOE_BWD_RDMA_CTAS",
    "relay_ctas": "WEFTMOE_BWD_RELAY_CTAS",
    "dispatch_batch_rows": "WEFTMOE_BWD_DISPATCH_BATCH_ROWS",
    "setup_threads": "WEFTMOE_BWD_SETUP_THREADS",
    "setup_idle_nanosleep": "WEFTMOE_BWD_SETUP_IDLE_NANOSLEEP",
    "queue_idle_nanosleep": "WEFTMOE_BWD_QUEUE_IDLE_NANOSLEEP",
    "ready_word_stagger": "WEFTMOE_BWD_READY_WORD_STAGGER",
    "activation_owner_group": "WEFTMOE_BWD_ACTIVATION_OWNER_GROUP",
    "epilogue_rows_per_task": "WEFTMOE_BWD_EPILOGUE_ROWS_PER_TASK",
    "dw2_claim_tiles": "WEFTMOE_BWD_DW2_CLAIM_TILES",
    "dw1_claim_tiles": "WEFTMOE_BWD_DW1_CLAIM_TILES",
    "reduce_threads": "WEFTMOE_BWD_REDUCE_THREADS",
    "reduce_idle_nanosleep": "WEFTMOE_BWD_REDUCE_IDLE_NANOSLEEP",
    "unified_workers": "WEFTMOE_BWD_UNIFIED_WORKERS",
    **unified_tuning.parameter_macros("backward"),
}


def validate_tuning(config: Mapping[str, Any]) -> None:
    require(set(config) == set(PARAMETER_MACROS),
            "expanded tuning config must define every parameter exactly once")
    for key, value in config.items():
        require(type(value) is int, f"{key} must be an integer")
        require(value >= 0, f"{key} must be non-negative")

    unified_tuning.validate(config)
    if config["unified_workers"]:
        require(config["unified_producer_registers"] in
                (0, config["unified_control_registers"]),
                "Backward currently shares the producer and communication register quota")
    require(config["main_hidden"] > 0 and config["main_hidden"] % 256 == 0,
             "main_hidden must be a positive multiple of 256")
    require(config["main_intermediate"] > 0 and config["main_intermediate"] % 256 == 0,
             "main_intermediate must be a positive multiple of 256")
    require(config["tile_m"] == 128,
            "the activation dependency block is fixed at M128")
    require_choice(config, "dab_tile_n", (128, 256))
    require_choice(config, "dx_tile_n", (128, 256))
    unified_tuning.require_resources(
        config["main_intermediate"] % (2 * config["dab_tile_n"]) == 0 and
        config["main_hidden"] % (2 * config["dx_tile_n"]) == 0,
        "PF stream requires cluster-N2 tiles; select N128 for odd N256 extents")
    require(config["unified_workers"] == 1 and
            config["unified_threads"] == 384 and
            config["unified_comm_base"] == 288 and
            config["unified_comm_threads"] == 96,
            "PF stream uses 384 threads with a 288/96 compute/communication split")
    require(config["dx_claim_tiles"] > 0, "dx_claim_tiles must be positive")
    require(config["return_window_scale"] > 0, "return_window_scale must be positive")
    require_choice(config, "dw_tile_m", (128, 256))
    require_choice(config, "dw_tile_n", (128, 256))
    require(config["main_hidden"] % config["dw_tile_m"] == 0,
            "dW2 M tile must divide hidden")
    require(config["main_intermediate"] % config["dw_tile_n"] == 0,
            "dW2 N tile must divide intermediate")
    require(2 * config["main_intermediate"] % config["dw_tile_m"] == 0,
            "dW1 M tile must divide twice intermediate")
    require(config["main_hidden"] % config["dw_tile_n"] == 0,
            "dW1 N tile must divide hidden")
    require(config["tile_k"] == 64,
            "the weight-gradient pipeline uses tile_k=64")
    require_choice(config, "activation_tile_k", (64, 128))
    unified_tuning.require_resources(
        config["activation_tile_k"] == 64 or config["wgmma_n256_atom_n"] == 256 or
        max(config["dab_tile_n"], config["dx_tile_n"]) == 128,
        "split N256 WGMMA requires activation_tile_k=64")
    for key in ("dab_stages", "dx_stages", "dw1_stages", "dw2_stages"):
        require_choice(config, key, (2, 3, 4))
    require(config["dab_stages"] == config["dx_stages"],
            "dAB and dX share one activation pipeline stage count")
    require(config["dw1_stages"] == config["dw2_stages"],
            "dW1 and dW2 share one weight-gradient pipeline stage count")
    require(config["compute_threads"] == 384,
            "the current producer plus two math-WG kernel requires 384 threads")
    require(config["math_warpgroups"] == 2 and
            config["producer_warpgroups"] == 1,
            "the current WGMMA mapping requires two math and one producer WG")
    for key in ("math_registers", "producer_registers"):
        require(1 <= config[key] <= 255, f"{key} must be in [1, 255]")
    registers = 128 * (
        config["math_warpgroups"] * config["math_registers"] +
        config["producer_warpgroups"] * config["producer_registers"]
    )
    require(registers <= 65536,
            f"compute register allocation {registers} exceeds 65536")

    if config["unified_workers"]:
        unified_tuning.require_resources(unified_tuning.shared_memory_upper_bound(
            config, dynamic_shared_memory(config), phase="backward") <= 227 * 1024,
            "unified compute and communication staging exceed Hopper shared memory")
    require(config["rdma_ctas"] in (0, 8),
            "RDMA CTAs must be zero for local or eight for GIN")
    require(config["relay_ctas"] >= 8 and config["relay_ctas"] % 4 == 0,
            "relay_ctas must be >=8 and a multiple of 4")
    require_choice(config, "dispatch_batch_rows", (12, 16, 32, 64, 128))
    for key in ("setup_threads", "reduce_threads"):
        require_choice(config, key, (128, 256, 384, 512))
    require_choice(config, "ready_word_stagger", (0, 1))
    require_choice(config, "activation_owner_group", (1, 2, 4, 8, 16, 32))
    require_choice(config, "epilogue_rows_per_task",
                   (12, 16, 24, 32, 48, 64, 96, 128))
    require_choice(config, "dw2_claim_tiles", (1, 2, 4, 8, 16))
    require_choice(config, "dw1_claim_tiles", (1, 2, 4, 8, 16))
    for key in (
        "setup_idle_nanosleep",
        "queue_idle_nanosleep",
        "reduce_idle_nanosleep",
    ):
        require_choice(config, key, (0, 16, 32, 64, 128, 256, 512))


def dynamic_shared_memory(config: Mapping[str, Any]) -> int:
    storage_tile_n = max(config["dab_tile_n"], config["dx_tile_n"],
                         config["dw_tile_n"])
    pair_n = (config["main_intermediate"] % (2 * config["dab_tile_n"]) == 0
              and config["main_hidden"] % (2 * config["dx_tile_n"]) == 0)
    activation_m = config["tile_m"] if pair_n else config["tile_m"] // 2
    activation_elements = (activation_m + storage_tile_n) * config["activation_tile_k"]
    dw_elements = (
        config["dw_tile_m"] + config["dw_tile_n"]
    ) * config["tile_k"]
    activation_arena = config["dab_stages"] * activation_elements
    dw_arena = (config["dw1_stages"] * dw_elements +
                config["dw_tile_m"] * config["dw_tile_n"])
    barriers = 2 * (config["dab_stages"] + config["dw1_stages"])
    return 2 * max(activation_arena, dw_arena) + 8 * barriers
