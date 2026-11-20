"""B300 NVFP4 block-16 E2M1 operands with E4M3 block scales."""

from dataclasses import dataclass
import math
import weakref
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from .blackwell import BlackwellRuntime


@triton.jit
def _encode_row(x, tensor_scale, COLS: tl.constexpr):
    values = x.to(tl.float32).reshape(COLS // 16, 16)
    # The instruction consumes pairs of adjacent elements and returns one byte per pair.
    maxval = tl.max(tl.abs(values), 1)
    sf = tl.minimum(tl.div_rn(maxval, 6.0 * tensor_scale), 448.0).to(tl.float8e4nv)
    divisor = sf.to(tl.float32) * tensor_scale
    normalized = tl.where(divisor[:, None] > 0, tl.div_rn(values, divisor[:, None]), 0.0).reshape(COLS)
    lo, hi = tl.split(normalized.reshape(COLS // 2, 2))
    packed = tl.inline_asm_elementwise(
        asm="{ .reg .b8 pair; cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; cvt.u16.u8 $0, pair; }",
        constraints="=h,f,f",
        args=[lo, hi],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8), sf.to(tl.uint8, bitcast=True)


@triton.jit
def _pack_rows(
    X,
    Data,
    Scale,
    tensor_scale,
    N: tl.constexpr,
    K: tl.constexpr,
    COLS: tl.constexpr,
    WEIGHT: tl.constexpr,
    INTERLEAVE: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.program_id(1) * COLS
    group = tl.arange(0, COLS // 16)
    x = tl.load(X + row.to(tl.int64) * K + start + tl.arange(0, COLS), start + tl.arange(0, COLS) < K, 0)
    packed, sf = _encode_row(x, tensor_scale, COLS)
    dest = row
    if WEIGHT:
        e, local = row // N, row % N
        if INTERLEAVE:
            half = N // 2
            local = ((local % half) // 8 * 2 + local // half) * 8 + local % 8
        dest = e * N + local
        sf_m = local // 128 * 128 + local % 32 * 4 + (local % 128) // 32
        sg = start // 16 + group
        sf_dest = ((e.to(tl.int64) * (K // 64) + sg // 4) * N + sf_m) * 4 + sg % 4
    else:
        sf_dest = row.to(tl.int64) * (K // 16) + start // 16 + group
    tl.store(Scale + sf_dest, sf, start // 16 + group < K // 16)
    col = start // 2 + tl.arange(0, COLS // 2)
    tl.store(Data + dest.to(tl.int64) * (K // 2) + col, packed, col < K // 2)


@triton.jit
def _route(R, G, SR, SG, E: tl.constexpr, K: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.arange(0, BK)
    ids = tl.load(R + row * K + slot, slot < K, -1)
    gates = tl.load(G + row * K + slot, slot < K, 0.0)
    keys = tl.sort(tl.where((ids >= 0) & (ids < E), ids, E) * BK + slot, descending=False)
    tl.store(SR + row * K + slot, tl.where(keys // BK < E, keys // BK, -1), slot < K)
    tl.store(SG + row * K + slot, tl.gather(gates, keys % BK, 0), slot < K)


class NVFP4Scales(NamedTuple):
    """Layer-wide dequantization factors, independent of the token batch."""

    x: float = 1 / 32
    w1: float = 1 / 64
    intermediate: float = 1 / 512
    w2: float = 1 / 64
    dy: float = 1 / 32
    dgate_up: float = 1 / 512


@dataclass(frozen=True)
class NVFP4Weights:
    fc1: tuple
    fc2: tuple
    tensor_scales: tuple


def _validate_scales(scales):
    if any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("NVFP4 tensor scales must be finite and positive")


def prepare_nvfp4_weights(w1, w2, *, weight_scales=(1 / 64, 1 / 64)):
    """Quantize canonical BF16 expert weights. Scales are layer-wide dequantization factors."""
    if w1.ndim != 3 or w2.ndim != 3 or w1.shape != (w2.shape[0], 2 * w2.shape[2], w2.shape[1]):
        raise ValueError("Expected canonical expert weights [E,2I,H] and [E,H,I]")
    if w1.device != w2.device or any(
        not w.is_cuda or w.dtype != torch.bfloat16 or not w.is_contiguous() for w in (w1, w2)
    ):
        raise ValueError("Expected contiguous CUDA BF16 weights on the same device")
    if min(w1.shape) <= 0 or len(weight_scales) != 2 or w1.shape[1] % 256 or w1.shape[2] % 128:
        raise ValueError("NVFP4 requires H and I divisible by 128 and two weight scales")
    _validate_scales(weight_scales)
    matrices = []
    for idx, (w, scale) in enumerate(zip((w1, w2), weight_scales)):
        e, n, k = w.shape
        data = torch.empty((e, n, k // 2), dtype=torch.int8, device=w.device)
        sf = torch.empty((e, k // 64, n), dtype=torch.int, device=w.device)
        _pack_rows[(e * n, triton.cdiv(k, 512))](
            w, data, sf.view(torch.uint8), scale, n, k, 512, True, idx == 0
        )
        matrices.append((data, sf.transpose(1, 2)))
    return NVFP4Weights(*matrices, tuple(weight_scales))


def stage_nvfp4(buffer, x, routes, gates, scale):
    n, k = x.shape
    if n:
        _pack_rows[(n, triton.cdiv(k, 512))](
            x, buffer.x, buffer.x_sf.view(torch.uint8), scale, 1, k, 512, False, False
        )
        _route[(n,)](
            routes,
            gates,
            buffer.topk_idx,
            buffer.topk_weights,
            buffer.num_experts,
            buffer.num_topk,
            triton.next_power_of_2(buffer.num_topk),
        )


def validate_prepared(b, device, x, routes, gates, weights, x_scale, intermediate_scale):
    _validate_scales((x_scale, intermediate_scale))
    if not isinstance(weights, NVFP4Weights):
        raise TypeError("Expected prepared NVFP4 expert weights")
    e = b.num_experts // b.group.size()
    w1, w2 = (weights.fc1[0], weights.fc2[0])
    if w1.shape != (e, 2 * b.intermediate_hidden, b.hidden // 2) or w2.shape != (
        e,
        b.hidden,
        b.intermediate_hidden // 2,
    ):
        raise ValueError("NVFP4 weights do not match the prepared experts")
    if any(t.device != device for t in (x, routes, gates, *weights.fc1, *weights.fc2)):
        raise ValueError("NVFP4 inputs and weights must share the workspace device")
    if x.ndim != 2 or x.shape[1] != b.hidden or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("Expected contiguous [tokens, H] BF16 input")
    for t, dtype in ((routes, torch.int32), (gates, torch.float32)):
        if t.shape != (x.shape[0], b.num_topk) or t.dtype != dtype or not t.is_contiguous():
            raise ValueError("Invalid NVFP4 routing shape, dtype or layout")


def save_routed_x(runtime, saved_x, origins):
    from .blackwell_backward import _gather_dy

    b = runtime.buffer
    _gather_dy[(triton.cdiv(len(origins), 4),)](
        runtime.dy_peers,
        origins,
        saved_x,
        len(origins),
        0,
        b.num_max_tokens_per_rank,
        b.num_topk,
        b.hidden,
        triton.next_power_of_2(b.hidden),
        4,
    )
    if b.group.size() > 1:
        b.handle.barrier()


class NVFP4Runtime(BlackwellRuntime):
    """Persistent NVLink expert kernel on one Blackwell node."""

    def __init__(
        self, group, experts, top_k, hidden, intermediate, capacity, stream, training=False, max_saved=1,
    ):
        import deep_gemm as dg

        self.buffer = dg.SymmBuffer(
            group, experts, max(1, capacity), top_k, hidden, intermediate, mma_type="nvfp4",
        )
        self.stream = (
            stream
            if training or stream != torch.cuda.default_stream(stream.device)
            else torch.cuda.Stream(device=stream.device)
        )

        self.launch = dg.nvfp4_mega_moe
        self.training, self.max_saved = training, max_saved
        self.tapes = weakref.WeakSet()
        self.scales = NVFP4Scales()
        if training:
            import torch.distributed._symmetric_memory as symm
            from .nvfp4_backward import expert_gradients

            self.expert_gradients = expert_gradients
            self.alignment = 128
            allocator = torch if group.size() == 1 else symm
            self.dy_buffer = allocator.empty(
                (self.buffer.num_max_tokens_per_rank, hidden), dtype=torch.bfloat16, device=stream.device
            )
            self.dy_handle = None if group.size() == 1 else symm.rendezvous(self.dy_buffer, group=group)
            pointers = self.dy_handle.buffer_ptrs if self.dy_handle else [self.dy_buffer.data_ptr()]
            self.dy_peers = torch.tensor(pointers, device=stream.device, dtype=torch.int64)
            self.peers = torch.tensor(self.buffer.handle.buffer_ptrs, device=stream.device, dtype=torch.int64)

    def _run(self, x, routes, gates, w1, w2, *, save=False):
        import deep_gemm as dg
        from .blackwell_backward import Tape, checkpoint

        sx, s1, sm, s2 = self.scales[:4]
        weights = prepare_nvfp4_weights(w1, w2, weight_scales=(s1, s2))
        if not save:
            return self.inference(x, routes, gates, weights, sx, sm)
        if len(self.tapes) >= self.max_saved:
            raise RuntimeError("Blackwell training tape capacity exhausted")
        b = self.buffer
        stage_nvfp4(b, x, routes, gates, sx)
        self.dy_buffer[: len(x)].copy_(x)
        buffers, ends = checkpoint(b, len(x), x.device, self.alignment, pad_empty=True)
        y = torch.empty_like(x)
        dg.nvfp4_mega_moe(
            y, weights.fc1, weights.fc2, b, tensor_scales=(sx * s1, sm, sm * s2), saved=buffers
        )
        _, _, saved_x, ab, expert_gates, origins = buffers
        save_routed_x(self, saved_x, origins)
        tape = Tape(y, (routes.clone(), w1, w2, saved_x, ab, expert_gates, origins, ends))
        self.tapes.add(tape)
        return tape

    def inference(self, x, routes, gates, weights, x_scale, intermediate_scale):
        import deep_gemm as dg

        b = self.buffer
        validate_prepared(b, self.stream.device, x, routes, gates, weights, x_scale, intermediate_scale)
        caller = torch.cuda.current_stream(x.device)
        if caller != self.stream:
            self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            stage_nvfp4(b, x, routes, gates, x_scale)
            out = torch.empty_like(x)
            dg.nvfp4_mega_moe(
                out,
                weights.fc1,
                weights.fc2,
                b,
                tensor_scales=(
                    x_scale * weights.tensor_scales[0],
                    intermediate_scale,
                    intermediate_scale * weights.tensor_scales[1],
                ),
            )
            if caller != self.stream:
                for t in (x, routes, gates, *weights.fc1, *weights.fc2):
                    t.record_stream(self.stream)
                out.record_stream(caller)
        if caller != self.stream:
            caller.wait_stream(self.stream)
        return out

    def usage(self):
        result = super().usage()
        if self.training:
            result["symmetric_bytes"] += self.dy_buffer.nbytes
        return result

    def retire_peer_mappings(self):
        self.buffer.destroy()
        if self.training:
            self.dy_handle = self.dy_buffer = None
