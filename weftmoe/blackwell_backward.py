"""Blackwell BF16 grouped dgrad/wgrad with deterministic NVLink gather and return."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@dataclass(eq=False)
class Tape:
    output: torch.Tensor
    state: tuple | None = None


@triton.jit
def _ordinals(Sorted, Permutation, Bounds, Dest, n, BLOCK: tl.constexpr):
    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(Sorted + j, j < n, 0)
    slot = tl.load(Permutation + j, j < n, 0)
    start = tl.load(Bounds + tl.maximum(0, expert))
    tl.store(Dest + slot, j - start, j < n)


def checkpoint(buffer, tokens, device, alignment, *, pad_empty=False):
    # Stable source slots make both the rank-interleaved owner rows and the
    # K reduction for each expert weight gradient independent of CTA arrival.
    ids, permutation = buffer.topk_idx[:tokens].reshape(-1).int().sort(stable=True)
    experts = torch.arange(buffer.num_experts + 1, device=device, dtype=torch.int32)
    bounds = torch.searchsorted(ids, experts, out_int32=True)
    ordinals = torch.empty_like(ids)
    if ids.numel():
        _ordinals[(triton.cdiv(ids.numel(), 256),)](ids, permutation, bounds, ordinals, ids.numel(), 256)
    counts = bounds.diff()
    dist.all_reduce(counts, group=buffer.group)
    local = counts.chunk(buffer.group.size())[buffer.group.rank()]
    padded = ((local + alignment - 1) // alignment) * alignment
    if pad_empty:
        # CUTLASS grouped NVFP4 requires positive K even for an idle expert.
        padded = padded.clamp_min(alignment)
    ends = padded.cumsum(0, dtype=torch.int32)
    starts = ends - padded
    rows = int(ends[-1])
    logical_ends = starts + local
    # K-grouped GEMM may read the tail of its last K tile. Keep padding zero.
    x = torch.zeros(rows, buffer.hidden, device=device, dtype=torch.bfloat16)
    ab = torch.zeros(rows, 2 * buffer.intermediate_hidden, device=device, dtype=torch.bfloat16)
    gates = torch.zeros(rows, device=device)
    origins = torch.full((rows,), -1, device=device, dtype=torch.int32)
    return (ordinals, starts, x, ab, gates, origins), ends if pad_empty else logical_ends


@triton.jit
def _gather_dy(
    Peers,
    Origins,
    Out,
    rows,
    X_OFFSET: tl.constexpr,
    CAP: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    orig = tl.load(Origins + row, row < rows, -1)
    addr = tl.maximum(orig, 0)
    base = tl.load(Peers + addr // (CAP * K)) + X_OFFSET
    dy = base.to(tl.pointer_type(tl.bfloat16))
    token = addr % (CAP * K) // K
    col = tl.arange(0, BH)
    val = tl.load(
        dy[:, None] + token[:, None] * H + col[None, :], (orig[:, None] >= 0) & (col[None, :] < H), 0.0
    )
    tl.store(Out + row[:, None] * H + col[None, :], val, (row[:, None] < rows) & (col[None, :] < H))


@triton.jit
def _swiglu(
    AB,
    DU,
    Gates,
    U,
    DAB,
    DG,
    rows,
    I: tl.constexpr,
    BI: tl.constexpr,
    R: tl.constexpr,
    EXACT: tl.constexpr = False,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    col = tl.arange(0, BI)
    mask = (row[:, None] < rows) & (col[None, :] < I)
    offset = row[:, None] * I + col[None, :]
    ab = row[:, None] * (2 * I) + col[None, :]
    a = tl.load(AB + ab, mask, 0.0).to(tl.float32)
    b = tl.load(AB + ab + I, mask, 0.0).to(tl.float32)
    du = tl.load(DU + offset, mask, 0.0).to(tl.float32)
    g = tl.load(Gates + row, row < rows, 0.0)
    if EXACT:
        sig = tl.div_rn(1.0, 1.0 + libdevice.exp(-a))
    else:
        sig = tl.sigmoid(a)
    silu = a * sig
    act = silu * b
    tl.store(U + offset, act * g[:, None], mask)
    tl.store(DAB + ab, du * g[:, None] * b * (sig * (1.0 + a * (1.0 - sig))), mask)
    tl.store(DAB + ab + I, du * g[:, None] * silu, mask)
    tl.store(DG + row, tl.sum(du * act, 1), row < rows)


@triton.jit
def _return(
    Peers,
    Origins,
    DX,
    DG,
    rows,
    DX_OFFSET: tl.constexpr,
    DG_OFFSET: tl.constexpr,
    CAP: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0) * R + tl.arange(0, R)
    orig = tl.load(Origins + row, row < rows, -1)
    addr = tl.maximum(orig, 0)
    base = tl.load(Peers + addr // (CAP * K))
    slot = addr % (CAP * K)
    dx = (base + DX_OFFSET).to(tl.pointer_type(tl.bfloat16))
    dg = (base + DG_OFFSET).to(tl.pointer_type(tl.float32))
    # Return buffer has layout [slot, token, H] as in the forward epilogue.
    dst = (slot % K * CAP + slot // K).to(tl.int64) * H
    col = tl.arange(0, BH)
    mask = (orig[:, None] >= 0) & (col[None, :] < H)
    val = tl.load(DX + row[:, None] * H + col[None, :], mask, 0.0)
    tl.store(dx[:, None] + dst[:, None] + col[None, :], val, mask)
    tl.store(dg + slot, tl.load(DG + row, orig >= 0, 0.0), orig >= 0)


@triton.jit
def _combine(
    DX,
    DG,
    Routes,
    Out,
    OutG,
    tokens,
    E: tl.constexpr,
    K: tl.constexpr,
    CAP: tl.constexpr,
    H: tl.constexpr,
    BK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    slots = tl.arange(0, BK)
    ids = tl.load(Routes + row * K + slots, slots < K, -1)
    keys = tl.sort(tl.where((ids >= 0) & (ids < E), ids, E) * BK + slots)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for j in tl.static_range(K):
        key = tl.sum(tl.where(slots == j, keys, 0), 0)
        live = key // BK < E
        offset = (j * CAP + row).to(tl.int64) * H
        acc += tl.load(DX + offset + col, live & (col < H), 0.0).to(tl.float32)
        if tl.program_id(1) == 0:
            dg = tl.load(DG + row * K + j, live, 0.0)
            tl.store(OutG + row * K + key % BK, dg)
    tl.store(Out + row * H + col, acc, col < H)


def expert_gradients(runtime, w1, w2, x, ab, gates, origins, ends):
    """Read staged dY, return dW and leave each dX/dGate in its source slot."""
    import deep_gemm as dg

    b = runtime.buffer
    rows, h = x.shape
    k = b.num_topk
    i, cap = b.intermediate_hidden, b.num_max_tokens_per_rank
    base = b.buffer.data_ptr()
    x_offset = b.x.data_ptr() - base
    dg_offset = b.topk_weights.data_ptr() - base
    dx_bytes = cap * k * h * 2
    dx_offset = b.buffer.nbytes - dx_bytes
    if b.group.size() > 1:
        b.handle.barrier()
    recv = torch.empty_like(x)
    u = torch.empty((rows, i), dtype=x.dtype, device=x.device)
    du = torch.zeros_like(u)
    dab = torch.empty_like(ab)
    eg = torch.empty_like(gates)
    dw1, dw2 = torch.empty_like(w1), torch.empty_like(w2)
    if rows:
        grid = (triton.cdiv(rows, 4),)
        _gather_dy[grid](
            runtime.peers, origins, recv, rows, x_offset, cap, k, h, triton.next_power_of_2(h), 4
        )
        dg.m_grouped_bf16_gemm_nn_contiguous(recv, w2, du, ends, use_psum_layout=True)
        _swiglu[grid](ab, du, gates, u, dab, eg, rows, i, triton.next_power_of_2(i), 4)
        local_dx = torch.empty_like(x)
        dg.m_grouped_bf16_gemm_nn_contiguous(dab, w1, local_dx, ends, use_psum_layout=True)
        _return[grid](
            runtime.peers,
            origins,
            local_dx,
            eg,
            rows,
            dx_offset,
            dg_offset,
            cap,
            k,
            h,
            triton.next_power_of_2(h),
            4,
        )
        # On multiple nodes, the rail can return dX/dGate while the tensor
        # cores finish the two independent weight gradients.
        compute = getattr(runtime, "wgrad_stream", None)
        if compute is not None:
            compute.wait_stream(torch.cuda.current_stream())
            for tensor in (dab, x, dw1, recv, u, dw2, ends):
                tensor.record_stream(compute)
        with torch.cuda.stream(compute):
            dg.k_grouped_bf16_gemm_tn_contiguous(dab, x, dw1, None, ends, None, use_psum_layout=True)
            dg.k_grouped_bf16_gemm_tn_contiguous(recv, u, dw2, None, ends, None, use_psum_layout=True)
    else:
        dw1.zero_()
        dw2.zero_()
    if b.group.size() > 1:
        b.handle.barrier()
    return b.buffer[dx_offset:].view(torch.bfloat16), dw1, dw2


def gradients(runtime, tape, dy):
    if tape not in runtime.tapes or tape.state is None:
        raise RuntimeError("Backward requires a live training tape from this runtime")
    routes, w1, w2, x, ab, gates, origins, ends = tape.state
    b = runtime.buffer
    if dy.dtype != torch.bfloat16 or dy.shape != tape.output.shape or dy.device != x.device:
        raise ValueError("Backward requires BF16 dY matching the forward output")
    n, h = dy.shape
    k, cap = b.num_topk, b.num_max_tokens_per_rank
    getattr(runtime, "dy_buffer", b.x)[:n].copy_(dy)
    mailbox, dw1, dw2 = getattr(runtime, "expert_gradients", expert_gradients)(
        runtime, w1, w2, x, ab, gates, origins, ends
    )
    dx, dgate = torch.empty_like(dy), torch.empty_like(routes, dtype=torch.float32)
    if n:
        _combine[(n, triton.cdiv(h, 512))](
            mailbox,
            b.topk_weights,
            routes,
            dx,
            dgate,
            n,
            b.num_experts,
            k,
            cap,
            h,
            triton.next_power_of_2(k),
            512,
        )
    tape.state = None
    runtime.tapes.discard(tape)
    return dx, dgate, dw1, dw2
