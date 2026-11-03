"""Graph staging for SGLang's replicated-input expert layout."""

import triton
import triton.language as tl


@triton.jit
def _fill_missing_experts(expert, slot, top_k: tl.constexpr):
    # There are at least as many unused IDs in [0, top_k) as missing slots.
    # Pair missing slots with unused IDs in order; keep every valid ID intact.
    used = tl.sum((expert[None, :] == slot[:, None]).to(tl.int32), 1) > 0
    free = (slot < top_k) & ~used
    free_rank = tl.cumsum(free.to(tl.int32), 0) - 1
    missing_rank = tl.cumsum(((expert < 0) & (slot < top_k)).to(tl.int32), 0) - 1
    replacement = tl.sum(tl.where(
        free[None, :] & (missing_rank[:, None] == free_rank[None, :]),
        slot[None, :], 0), 1)
    return tl.where(expert >= 0, expert, replacement)


@triton.jit
def sanitize_routes(ids, weights, routes, gates, top_k: tl.constexpr,
                    block: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, block)
    offset = row * top_k + slot
    expert = tl.load(ids + offset, slot < top_k, other=-1)
    weight = tl.load(weights + offset, slot < top_k, other=0)
    tl.store(routes + offset, _fill_missing_experts(expert, slot, top_k), slot < top_k)
    tl.store(gates + offset, tl.where(expert >= 0, weight, 0), slot < top_k)


@triton.jit
def stage_inputs(
    x,
    ids,
    weights,
    staged_x,
    routes,
    gates,
    x_row_stride: tl.constexpr,
    x_col_stride: tl.constexpr,
    id_row_stride: tl.constexpr,
    id_col_stride: tl.constexpr,
    weight_row_stride: tl.constexpr,
    weight_col_stride: tl.constexpr,
    start: tl.constexpr,
    valid_rows: tl.constexpr,
    hidden: tl.constexpr,
    top_k: tl.constexpr,
    hidden_block: tl.constexpr,
    route_block: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, hidden_block)
    values = tl.load(
        x + (start + row) * x_row_stride + col * x_col_stride,
        (row < valid_rows) & (col < hidden),
        other=0,
    )
    tl.store(staged_x + row * hidden + col, values, col < hidden)
    slot = tl.arange(0, route_block)
    # Empty source rows carry zero X/gates. Reuse a real row's expert IDs so
    # padding does not introduce unrelated experts and extra GEMM work.
    source_row = tl.where(row < valid_rows, start + row, 0)
    expert = tl.load(
        ids + source_row * id_row_stride + slot * id_col_stride,
        slot < top_k,
        other=-1,
    )
    weight = tl.load(
        weights + (start + row) * weight_row_stride + slot * weight_col_stride,
        (row < valid_rows) & (slot < top_k),
        other=0,
    ).to(tl.float32)
    tl.store(
        routes + row * top_k + slot, _fill_missing_experts(expert, slot, top_k), slot < top_k
    )
    tl.store(gates + row * top_k + slot, tl.where(expert >= 0, weight, 0), slot < top_k)


@triton.jit
def scatter_output(
    output,
    result,
    start: tl.constexpr,
    valid_rows: tl.constexpr,
    hidden: tl.constexpr,
    elements: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0) * block + tl.arange(0, block)
    local = index - start * hidden
    value = tl.load(
        output + local,
        (local >= 0) & (local < valid_rows * hidden) & (index < elements),
        other=0,
    )
    tl.store(result + index, value, index < elements)
