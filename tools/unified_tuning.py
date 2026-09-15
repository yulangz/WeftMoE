"""Common, shape-independent resource controls for the all-CTA workers."""

from autotune_common import PlanError, require, require_choice


PARAMETER_MACROS = {
    "gin_ring_depth": "WEFTMOE_GIN_RING_DEPTH",
    "unified_threads": "WEFTMOE_UNIFIED_THREADS",
    "unified_comm_threads": "WEFTMOE_UNIFIED_COMM_THREADS",
    "unified_comm_base": "WEFTMOE_UNIFIED_COMM_BASE",
    "unified_math_registers": "WEFTMOE_UNIFIED_MATH_REGISTERS",
    "unified_control_registers": "WEFTMOE_UNIFIED_CONTROL_REGISTERS",
    "unified_producer_registers": "WEFTMOE_UNIFIED_PRODUCER_REGISTERS",
    "unified_segment_bytes": "WEFTMOE_UNIFIED_SEGMENT_BYTES",
    "unified_copy_buffers": "WEFTMOE_UNIFIED_COPY_BUFFERS",
    "unified_row_prefetch": "WEFTMOE_UNIFIED_ROW_PREFETCH",
    "unified_optimize_gin_local_transport": "WEFTMOE_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT",
    "unified_return_drain_rows": "WEFTMOE_UNIFIED_RETURN_DRAIN_ROWS",
    "unified_split_nvlink": "WEFTMOE_UNIFIED_SPLIT_NVLINK",
    "unified_gin_context_probes": "WEFTMOE_UNIFIED_GIN_CONTEXT_PROBES",
    "unified_gin_pack_slots": "WEFTMOE_UNIFIED_GIN_PACK_SLOTS",
    "wgmma_n256_atom_n": "WEFTMOE_WGMMA_N256_ATOM_N",
}

# These affect shared protocol storage or common instruction adapters. A
# combined binary must choose one value explicitly for both phases.
SHARED_PARAMETERS = frozenset({"gin_ring_depth", "unified_gin_context_probes",
                               "unified_gin_pack_slots", "wgmma_n256_atom_n"})
# Retained in Backward plans for compatibility. Backward has its own fixed
# local scheduling policy and does not consume these Forward-only controls.
FORWARD_ONLY_PARAMETERS = frozenset({
    "unified_row_prefetch", "unified_optimize_gin_local_transport",
    "unified_return_drain_rows", "unified_split_nvlink"})


def parameter_macros(phase):
    """Keep plan keys stable while making generated resources phase-owned."""
    if phase not in ("forward", "backward"):
        raise ValueError(f"unknown phase: {phase}")
    prefix = "FWD" if phase == "forward" else "BWD"
    return {key: macro if key in SHARED_PARAMETERS else
            macro.replace("WEFTMOE_UNIFIED_", f"WEFTMOE_{prefix}_UNIFIED_")
            for key, macro in PARAMETER_MACROS.items()}


def validate_shared_parameters(forward, backward):
    for key in sorted(SHARED_PARAMETERS):
        if forward[key] != backward[key]:
            raise ValueError(
                f"shared parameter {key} differs between Forward ({forward[key]}) "
                f"and Backward ({backward[key]}); select one common value")

# These are compatibility defaults for old plans. Current public search spaces
# explicitly choose their incumbent, including eight physical GIN ring slots.
DEFAULTS = dict(gin_ring_depth=4, unified_workers=0, unified_threads=384,
                unified_comm_threads=96, unified_comm_base=288,
                unified_math_registers=232, unified_control_registers=40,
                unified_producer_registers=0,
                unified_segment_bytes=2048, unified_copy_buffers=1,
                unified_row_prefetch=1, unified_optimize_gin_local_transport=0,
                unified_return_drain_rows=128,
                unified_split_nvlink=0,
                unified_gin_context_probes=2,
                unified_gin_pack_slots=1,
                wgmma_n256_atom_n=256)


def upgrade_legacy_groups(groups, *, search=False, extra_defaults=None):
    """Make old frozen plans explicit without changing their worker layout."""
    group = groups.setdefault("1_gemm", {})
    for key, value in dict(DEFAULTS, **(extra_defaults or {})).items():
        group.setdefault(key, [value] if search else value)



class ResourceCombinationError(PlanError):
    """Individually valid resource choices exceed a combined hardware budget."""


def require_resources(condition, message):
    if not condition:
        raise ResourceCombinationError(message)


def filter_generated_candidates(document, flatten_groups, validate_tuning):
    """Prune resource conflicts only in an automatically expanded unified grid.

    Explicit frozen candidates never call this helper. Malformed values remain
    errors, and legacy grids keep their prior strict validation behavior.
    """
    fixed = flatten_groups(document["fixed"], require_all=True)
    configs = [dict(fixed, **flatten_groups(candidate.get("override", {}),
                                          require_all=False))
               for candidate in document["candidates"]]
    if not any(config["unified_workers"] == 1 for config in configs):
        return
    accepted, rejected = [], []
    for candidate, config in zip(document["candidates"], configs):
        try:
            validate_tuning(config)
        except ResourceCombinationError as error:
            if config["unified_workers"] != 1:
                raise
            rejected.append(dict(name=candidate["name"],
                                 override=candidate.get("override", {}),
                                 reason=str(error)))
        else:
            accepted.append(candidate)
    require(accepted, "generated unified grid has no legal resource combination: "
            + (rejected[0]["reason"] if rejected else "empty grid"))
    document["candidate_filter"] = dict(generated=len(configs),
                                        accepted=len(accepted), rejected=rejected)
    document["candidates"] = accepted
    # Generated grid indices remain stable, including when the first legal
    # candidate becomes the incumbent. The tuning driver uses candidates[0].
    selection = document.get("selection", {})
    count = selection.get("confirmation_candidates")
    if type(count) is int and 1 <= count <= len(configs):
        selection["confirmation_candidates"] = min(count, len(accepted))


def validate(config):
    require_choice(config, "gin_ring_depth", (4, 8, 16))
    require_choice(config, "unified_workers", (0, 1))
    require_choice(config, "unified_threads", (384, 512, 640))
    require_choice(config, "unified_split_nvlink", (0, 1))
    require_choice(config, "unified_row_prefetch", (0, 1))
    require_choice(config, "unified_optimize_gin_local_transport", (0, 1))
    require(32 <= config["unified_return_drain_rows"] <= 1024 and
            config["unified_return_drain_rows"] % 32 == 0,
            "unified Return drain rows must be a multiple of 32 in [32, 1024]")
    require(config["unified_gin_context_probes"] > 0,
            "unified GIN context probe count must be positive")
    require(1 <= config["unified_gin_pack_slots"] <= 8,
            "unified GIN packing slot count must be in [1, 8]")
    require_choice(config, "unified_copy_buffers", (1, 2))
    require_choice(config, "wgmma_n256_atom_n", (128, 256))
    threads = config["unified_threads"]
    base, communication = config["unified_comm_base"], config["unified_comm_threads"]
    require(base >= 288 and base % 32 == 0 and communication >= 64
            and communication % 32 == 0 and base + communication <= threads,
            "unified communication must occupy complete warps after the producer")
    require(not config["unified_split_nvlink"] or communication >= 96,
            "independent Dispatch/Return requires two copy warps and a publisher")
    for key in ("unified_math_registers", "unified_control_registers"):
        require(24 <= config[key] <= 256 and config[key] % 8 == 0,
                f"{key} must be a multiple of eight in [24, 256]")
    producer = config["unified_producer_registers"] or config["unified_control_registers"]
    require(24 <= producer <= 256 and producer % 8 == 0,
            "unified producer registers must be zero or a multiple of eight in [24, 256]")
    require(base >= 384 or producer == config["unified_control_registers"],
            "producer and communication in one warpgroup must share a register quota")
    # setmaxnreg reallocates the CTA's initial register pool, which is rounded
    # down to eight registers per thread. Checking only the SM's 65536 total
    # would admit configurations that can wait forever for unavailable registers.
    initial_registers = (65536 // threads // 8) * 8
    pool = initial_registers * threads
    # PTX requires .inc to raise (or retain) the previous maximum and .dec
    # to lower (or retain) it, even when the total requested pool would fit.
    require(config["unified_math_registers"] >= initial_registers,
            f"unified math setmaxnreg.inc must be >= initial {initial_registers}")
    require(max(producer, config["unified_control_registers"]) <= initial_registers,
            f"unified control setmaxnreg.dec must be <= initial {initial_registers}")
    registers = (256 * config["unified_math_registers"] + 128 * producer +
                 (threads - 384) * config["unified_control_registers"])
    require_resources(registers <= pool,
            f"unified register allocation {registers} exceeds initial CTA pool {pool}")
    segment = config["unified_segment_bytes"]
    require(segment >= 16 and segment % 16 == 0,
            "unified TMA segment must be a positive multiple of 16 bytes")
    require(not config["unified_workers"] or threads == 384 or
            config["wgmma_n256_atom_n"] == 128,
            "CUDA 12.9 needs N128 WGMMA atoms for unified CTAs larger than 384 threads")


def shared_memory_upper_bound(config, compute_bytes, *, phase="forward"):
    """Include bounded communication staging and both roles' control storage.

    Kernel launch checks the compiled sizeof as well. The 512-byte metadata
    allowance covers either phase's control metadata; the 2048-byte allowance
    covers static SMEM.
    """
    if phase not in ("forward", "backward"):
        raise ValueError(f"unknown phase: {phase}")
    warps = config["unified_comm_threads"] // 32 - 1
    staging = warps * config["unified_segment_bytes"] * config["unified_copy_buffers"]
    # Both the base communication storage and the Backward extension have
    # 128-byte alignment. Segments need only 16-byte alignment, so their
    # combined payload is not necessarily already a multiple of 128 bytes.
    return ((compute_bytes + 127) // 128 * 128 + 2048 + 512 +
            (staging + 127) // 128 * 128)
