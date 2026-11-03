"""Hopper block-wise E4M3 compute and explicitly prepared expert weights.

Scales always multiply FP8 values to recover the represented value. Weight
preparation belongs outside forward/backward; activations use current amax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import weakref

import torch

from .runtime import _extension


@lru_cache(None)
def _check_arch(device: torch.device):
    if torch.cuda.get_device_capability(device) != (9, 0):
        raise ValueError("WeftMoE block-wise FP8 requires Hopper/SM90")


def _check_tensor(tensor, *, dtypes, ndim=(2, 3)):
    if (not isinstance(tensor, torch.Tensor) or not tensor.is_cuda
            or tensor.dtype not in dtypes or tensor.ndim not in ndim
            or not tensor.is_contiguous()):
        raise ValueError(f"Expected a contiguous CUDA tensor of rank {ndim}, dtype {dtypes}")
    _check_arch(tensor.device)


@dataclass(frozen=True)
class FP8Tensor:
    """K-contiguous E4M3 tensor with an explicit logical (unpadded) shape.

    The last dimension of data is padded to 128. scales is contiguous FP32,
    with one value per [block_rows, 128] block in each logical matrix.
    """
    data: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_rows: int = 1

    def __post_init__(self):
        _check_tensor(self.data, dtypes=(torch.float8_e4m3fn,))
        _check_tensor(self.scales, dtypes=(torch.float32,))
        if (self.block_rows not in (1, 128) or len(self.shape) not in (2, 3)
                or any(not isinstance(n, int) or n < 0 for n in self.shape)):
            raise ValueError("Invalid FP8 logical shape or block_rows")
        m, k = self.shape[-2:]
        prefix = self.shape[:-2]
        if (self.data.shape != (*prefix, m, (k + 127) // 128 * 128)
                or self.scales.shape != (*prefix, (m + self.block_rows - 1) // self.block_rows,
                                         (k + 127) // 128)
                or self.scales.device != self.data.device):
            raise ValueError("FP8 storage/scales do not match the logical matrix")

    def matrix(self, expert: int) -> FP8Tensor:
        if len(self.shape) != 3:
            raise ValueError("Only batched FP8 tensors have expert matrices")
        return FP8Tensor(self.data[expert], self.scales[expert], self.shape[1:], self.block_rows)

    def dequantize(self) -> torch.Tensor:
        """Diagnostic/reference operation; compute kernels never call this."""
        scale = self.scales.repeat_interleave(self.block_rows, -2).repeat_interleave(128, -1)
        return (self.data.float() * scale[..., :self.shape[-2], :])[..., :self.shape[-1]]


def _storage(shape, block_rows, device):
    m, k = shape[-2:]
    q = torch.empty((*shape[:-2], m, (k + 127) // 128 * 128),
                    dtype=torch.float8_e4m3fn, device=device)
    s = torch.empty((*shape[:-2], (m + block_rows - 1) // block_rows, (k + 127) // 128),
                    dtype=torch.float32, device=device)
    return FP8Tensor(q, s, tuple(shape), block_rows)


def quantize_fp8(tensor: torch.Tensor, *, block_rows: int = 1,
                 transpose: bool = False) -> FP8Tensor:
    """Quantize directly from BF16/FP32, optionally along original columns.

    transpose=True computes amax on 128 original rows per channel, then
    writes K-contiguous transposed data. It never requantizes a rowwise FP8
    tensor. Batch boundaries (experts) are independent quantization domains.
    """
    _check_tensor(tensor, dtypes=(torch.bfloat16, torch.float32))
    if block_rows not in (1, 128):
        raise ValueError("block_rows must be 1 or 128")
    shape = tuple(tensor.shape)
    if transpose:
        shape = (*shape[:-2], shape[-1], shape[-2])
    with torch.cuda.device(tensor.device):
        result = _storage(shape, block_rows, tensor.device)
        stream = torch.cuda.current_stream()
        _extension().fp8_quantize(tensor, result.data, result.scales,
                                  block_rows, transpose, stream.cuda_stream)
        tensor.record_stream(stream)
    return result


def _quantize_fp8_weight_pair(tensor: torch.Tensor) -> tuple[FP8Tensor, FP8Tensor]:
    """Read each 128x128 block once and emit both orientations in one kernel."""
    _check_tensor(tensor, dtypes=(torch.bfloat16, torch.float32))
    shape = tuple(tensor.shape)
    with torch.cuda.device(tensor.device):
        row = _storage(shape, 128, tensor.device)
        column = _storage((*shape[:-2], shape[-1], shape[-2]), 128, tensor.device)
        stream = torch.cuda.current_stream()
        _extension().fp8_quantize_weight(tensor, row.data, row.scales,
                                         column.data, column.scales, stream.cuda_stream)
        tensor.record_stream(stream)
    return row, column


def transpose_fp8_weight(weight: FP8Tensor) -> FP8Tensor:
    """Transpose 128x128 weight blocks and their scales, preserving bits."""
    if weight.block_rows != 128:
        raise ValueError("Rowwise activations require a new columnwise quantization")
    shape = (*weight.shape[:-2], weight.shape[-1], weight.shape[-2])
    with torch.cuda.device(weight.data.device):
        result = _storage(shape, 128, weight.data.device)
        stream = torch.cuda.current_stream()
        _extension().fp8_transpose_weight(weight.data, weight.scales,
            result.data, result.scales, weight.shape[-1], stream.cuda_stream)
        weight.data.record_stream(stream)
        weight.scales.record_stream(stream)
    return result


def fp8_gemm(a: FP8Tensor, b: FP8Tensor, *, dtype=torch.bfloat16,
             out: torch.Tensor | None = None, accumulate: bool = False) -> torch.Tensor:
    """Reference/diagnostic A @ B.T using FP8 WGMMA and K128 FP32 sums.

    This standalone kernel is not pipelined; use PF/DT for production MoE.
    """
    if (len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[1]
            or a.data.device != b.data.device):
        raise ValueError("FP8 GEMM requires [M,K], [N,K] on the same device")
    if dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("FP8 GEMM output dtype must be BF16 or FP32")
    if accumulate and (out is None or out.dtype != torch.float32):
        raise ValueError("FP8 accumulation requires an existing FP32 output")
    with torch.cuda.device(a.data.device):
        if out is None:
            out = torch.empty((a.shape[0], b.shape[0]), dtype=dtype, device=a.data.device)
        elif (out.dtype != dtype or out.device != a.data.device
              or out.shape != (a.shape[0], b.shape[0]) or not out.is_contiguous()):
            raise ValueError("FP8 output shape, dtype, device or layout mismatch")
        stream = torch.cuda.current_stream()
        _extension().fp8_gemm(a.data, a.scales, b.data, b.scales, out,
            a.shape[1], a.block_rows, b.block_rows, accumulate, stream.cuda_stream)
        for tensor in (a.data, a.scales, b.data, b.scales, out):
            tensor.record_stream(stream)
    return out


@dataclass(frozen=True)
class PreparedFP8Weights:
    """One immutable weight version, retaining both FWD and Dgrad layouts.

    Refresh by constructing another object after optimizer.step(). Existing
    tapes retain the old object. Mutating any data/scale tensor is an error.
    The FP32-master adapter explicitly invalidates old versions before reusing
    their storage; those objects and tapes can no longer be used for compute.
    """
    w1: FP8Tensor
    w2: FP8Tensor
    w1_t: FP8Tensor | None
    w2_t: FP8Tensor | None
    version: int
    _ready: torch.cuda.Event = field(repr=False, compare=False)
    _tensor_versions: tuple[int, ...] = field(repr=False)
    _sources: tuple = field(default=(), repr=False, compare=False)
    _cache_key: int = field(default=0, repr=False, compare=False)

    def _tensors(self):
        return tuple(t for w in (self.w1, self.w2, self.w1_t, self.w2_t)
                     if w is not None for t in (w.data, w.scales))

    def _validate_version(self, *, training: bool = False):
        if tuple(t._version for t in self._tensors()) != self._tensor_versions:
            raise RuntimeError("Prepared FP8 weights were modified; prepare a new weight version")
        if training and (self.w1_t is None or self.w2_t is None):
            raise ValueError("Backward requires weights prepared with training=True")

    def wait(self, *, training: bool = False):
        self._validate_version(training=training)
        with torch.cuda.device(self.w1.data.device):
            stream = torch.cuda.current_stream()
            stream.wait_event(self._ready)
            for tensor in self._tensors():
                tensor.record_stream(stream)

    def validate_parameters(self, w1, w2):
        """Reject a stale cache when prepared from these trainable parameters.

        Imported FP8 checkpoints have no master-parameter provenance; their
        caller explicitly chooses the associated high-precision parameters.
        """
        for parameter, source in zip((w1, w2), self._sources):
            if source is None:
                continue
            reference, version = source
            if reference() is not parameter:
                raise ValueError("FP8 cache was prepared from different parameters")
            if version is not None and parameter._version != version:
                raise RuntimeError("FP8 cache is stale after a parameter update; prepare new weights")


def _seal_weights(weights, version, sources=(), *, cache_key=0):
    """Publish an already-written weight version without copying its tensors."""
    with torch.cuda.device(weights[0].data.device):
        event = torch.cuda.Event()
        event.record()
    tensors = tuple(t for w in weights if w is not None for t in (w.data, w.scales))
    return PreparedFP8Weights(*weights, version, event,
                              tuple(t._version for t in tensors), tuple(sources), cache_key)


@torch.inference_mode(False)
@torch.no_grad()
def prepare_fp8_weights(w1: torch.Tensor | FP8Tensor,
                        w2: torch.Tensor | FP8Tensor, *, training: bool = True,
                        version: int = 0) -> PreparedFP8Weights:
    """Prepare outside the MoE call, from BF16/FP32 or prequantized weights.

    Training quantization emits W and W.T together, using the same FP8 bytes
    and block scales. Prequantized inputs are copied and transposed without
    changing quantization values.
    BF16/FP32 inputs stay asynchronous: nonfinite source blocks produce NaN
    scales. Imported FP8 bytes/scales are validated with one host synchronization.
    W1 uses canonical [gate, up] rows; internal packing must preserve their
    separate logical block scales. Scale values need not be powers of two.
    """
    if not isinstance(version, int) or not 0 <= version < 2**64:
        raise ValueError("weight version must be an unsigned 64-bit integer")
    sources = []
    for value in (w1, w2):
        if isinstance(value, torch.Tensor):
            sources.append((weakref.ref(value), None if value.is_inference() else value._version))
        else:
            sources.append(None)
    prepared = []
    transposed = []
    for w in (w1, w2):
        wt = None
        if isinstance(w, FP8Tensor):
            if w.block_rows != 128:
                raise ValueError("Expert weights require 128x128 quantization")
            with torch.cuda.device(w.data.device):
                prepared.append(FP8Tensor(w.data.clone(), w.scales.clone(), w.shape, 128))
        elif training:
            row, wt = _quantize_fp8_weight_pair(w)
            prepared.append(row)
        else:
            prepared.append(quantize_fp8(w, block_rows=128))
        transposed.append(wt)
    a, b = prepared
    if (len(a.shape) != 3 or len(b.shape) != 3 or a.shape[0] != b.shape[0]
            or a.shape[1] != 2 * b.shape[2] or a.shape[2] != b.shape[1]
            or a.data.device != b.data.device or min(*a.shape, *b.shape) == 0
            or a.shape[2] % 128 or b.shape[2] % 128):
        raise ValueError("Expected W1 [E,2I,H], W2 [E,H,I], H/I multiples of 128")
    with torch.cuda.device(a.data.device):
        # Only imported checkpoints require host validation. Our quantizer
        # propagates nonfinite source blocks through NaN scales without a
        # blocking scalar read on the ordinary BF16/FP32 forward path.
        valid = []
        for source, w in zip((w1, w2), prepared):
            if isinstance(source, FP8Tensor):
                valid.append((torch.isfinite(w.scales) & (w.scales > 0)).all())
                # E4M3FN has no infinity encoding; check NaN without FP32 copies.
                valid.append(((w.data.view(torch.uint8) & 0x7f) != 0x7f).all())
        if valid and not torch.stack(valid).all().item():
            raise ValueError("FP8 weights must be finite and scales finite and positive")
        at, bt = ((qt if qt is not None else transpose_fp8_weight(q)) if training else None
                  for q, qt in zip(prepared, transposed))
    return _seal_weights((a, b, at, bt), version, sources)
