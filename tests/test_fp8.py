"""SM90 FP8: BF16 global-diff acceptance and exact quantization layouts.

Run with WEFTMOE_TENSOR_EXTENSION pointing at a freshly built tensor adapter.
"""
import os

import pytest
import torch
from benchmarks.fp8_validation import assert_diff

from weftmoe.fp8 import (FP8Tensor, fp8_gemm, prepare_fp8_weights,
                         quantize_fp8, transpose_fp8_weight, _quantize_fp8_weight_pair)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not os.environ.get("WEFTMOE_TENSOR_EXTENSION")
    or torch.cuda.get_device_capability() != (9, 0),
    reason="requires SM90 and the native tensor adapter")


def quant_reference(x, block_rows, transpose):
    if transpose:
        x = x.transpose(-1, -2)
    m, k = x.shape[-2:]
    padded = torch.zeros((*x.shape[:-2], (m + block_rows - 1) // block_rows * block_rows,
                          (k + 127) // 128 * 128), device=x.device, dtype=torch.float64)
    padded[..., :m, :k] = x.double()
    blocks = padded.reshape(*x.shape[:-2], -1, block_rows, (k + 127) // 128, 128)
    amax = blocks.abs().amax((-1, -3))
    # CPU log2 avoids CUDA's approximate log2 crossing an exact power-of-two
    # boundary (e.g. amax=0.0546875). The cast kernel must not double that scale.
    scales = torch.where(amax.cpu() == 0, 1,
                         2.0 ** torch.ceil(torch.log2(amax.cpu() / 448))).to(x.device)
    scales = scales.clamp_min(torch.finfo(torch.float32).tiny).float()
    expanded = scales.repeat_interleave(block_rows, -2).repeat_interleave(128, -1)
    q = (padded / expanded).to(torch.float8_e4m3fn)[..., :m, :].contiguous()
    return q, scales


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1), (67, 259), (2, 129, 257)])
@pytest.mark.parametrize("block_rows,transpose", [(1, False), (1, True), (128, False)])
def test_quantization(dtype, shape, block_rows, transpose):
    torch.manual_seed(12)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    x[..., 0, 0] = 0
    x[..., -1, -1] = 87
    actual = quantize_fp8(x, block_rows=block_rows, transpose=transpose)
    q, scales = quant_reference(x, block_rows, transpose)
    torch.testing.assert_close(actual.data.view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(actual.scales, scales, rtol=0, atol=0)


def test_zero_small_and_nonfinite():
    x = torch.zeros(4, 256, dtype=torch.float32, device="cuda")
    x[1, :128] = 2.0 ** -135
    x[2, :128] = float("inf")
    x[3, 0] = float("nan")
    actual = quantize_fp8(x)
    assert torch.equal(actual.scales[0], torch.ones_like(actual.scales[0]))
    assert actual.scales[1, 0] == torch.finfo(torch.float32).tiny
    torch.testing.assert_close(actual.dequantize()[1], x[1], rtol=0, atol=0)
    assert actual.scales[2:, 0].isnan().all()
    assert torch.equal(actual.scales[2:, 1], torch.ones_like(actual.scales[2:, 1]))


@pytest.mark.parametrize("shape", [(1, 3), (129, 257), (2, 131, 270)])
def test_weight_transpose_preserves_quantization(shape):
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q = quantize_fp8(x, block_rows=128)
    qt = transpose_fp8_weight(q)
    torch.testing.assert_close(qt.dequantize(), q.dequantize().transpose(-1, -2), rtol=0, atol=0)
    roundtrip = transpose_fp8_weight(qt)
    torch.testing.assert_close(roundtrip.data.view(torch.uint8), q.data.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(roundtrip.scales, q.scales, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 3), (129, 257), (2, 131, 270),
                                  (2, 256, 384), (0, 128), (128, 0), (0, 128, 256)])
def test_weight_pair_matches_independent_quantization(dtype, shape):
    torch.manual_seed(42)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if x.numel():
        x[..., :128, :128] *= 16
        x[..., -1, -1] = -87
    row, column = _quantize_fp8_weight_pair(x)
    if not x.numel():
        assert row.data.numel() == column.data.numel() == 0
        assert row.scales.numel() == column.scales.numel() == 0
        return
    # Independent reference in each direction also checks all padded bytes.
    for actual, transpose in ((row, False), (column, True)):
        q, scales = quant_reference(x, 128, transpose)
        torch.testing.assert_close(actual.data.view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual.scales, scales, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(256, 384), (129, 257)])
def test_weight_pair_zero_tiny_nonfinite_and_graph(shape):
    x = torch.zeros(shape, device="cuda", dtype=torch.float32)
    x[:128, 128:256] = 2.0 ** -135
    x[128, 0] = float("inf")
    x[128, 128] = float("nan")
    old = quantize_fp8(x, block_rows=128)
    old_t = transpose_fp8_weight(old)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        row, column = _quantize_fp8_weight_pair(x)
    graph.replay()
    for actual, expected in ((row, old), (column, old_t)):
        torch.testing.assert_close(actual.data.view(torch.uint8), expected.data.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual.scales, expected.scales, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_weight_pair_unaligned_input(dtype):
    x = torch.randn(128 * 256 + 1, device="cuda", dtype=dtype)[1:].view(128, 256)
    assert x.is_contiguous() and x.data_ptr() % (4 * x.element_size())
    row, column = _quantize_fp8_weight_pair(x)
    for actual, transpose in ((row, False), (column, True)):
        q, scales = quant_reference(x, 128, transpose)
        torch.testing.assert_close(actual.data.view(torch.uint8), q.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(actual.scales, scales, rtol=0, atol=0)


def test_weight_pair_native_validation():
    from weftmoe.runtime import _extension
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    row, column = _quantize_fp8_weight_pair(x)
    quantize = _extension().fp8_quantize_weight
    stream = torch.cuda.current_stream().cuda_stream
    with pytest.raises(ValueError, match="alias"):
        quantize(x, row.data, row.scales, row.data, column.scales, stream)
    with pytest.raises(ValueError, match="alias"):
        quantize(x, row.data, row.scales, column.data, row.scales, stream)
    with pytest.raises(ValueError, match="matching shape"):
        quantize(x, row.data, row.scales, column.data[:, :-1], column.scales, stream)
    unaligned = torch.empty(128 * 128 + 1, device="cuda", dtype=torch.uint8)[1:]
    unaligned = unaligned.view(torch.float8_e4m3fn).view(128, 128)
    with pytest.raises(ValueError, match="aligned"):
        quantize(x, unaligned, row.scales, column.data, column.scales, stream)


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (65, 137, 259), (128, 256, 4096), (0, 64, 128), (64, 0, 128), (3, 9, 0)])
@pytest.mark.parametrize("b_block_rows", [1, 128])
def test_gemm(m, n, k, b_block_rows):
    torch.manual_seed(1234)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    # Different scales on each K block catch the invalid single-epilogue-scale
    # implementation. M/N tails also exercise accumulator lane mapping.
    if k:
        a[:, :min(128, k)] *= 4
        b[:, :min(128, k)] *= 0.0625
    aq, bq = quantize_fp8(a), quantize_fp8(b, block_rows=b_block_rows)
    expected = aq.dequantize().double() @ bq.dequantize().double().T
    actual = fp8_gemm(aq, bq, dtype=torch.float32)
    # Separate quantization error from arithmetic correctness: BF16 operands
    # use 5e-3, while the already-quantized FP64 reference requires 1e-6.
    original = a.double() @ b.double().T
    assert_diff(actual, original, 'gemm')
    assert_diff(actual, expected, 'quantized_gemm')
    out = torch.full((m, n), 0.5, device="cuda", dtype=torch.float32)
    fp8_gemm(aq, bq, dtype=torch.float32, out=out, accumulate=True)
    assert_diff(out, original + 0.5, 'gemm_accumulate')
    assert_diff(out, expected + 0.5, 'quantized_gemm_accumulate')
    bf16 = fp8_gemm(aq, bq)
    torch.testing.assert_close(bf16, actual.bfloat16(), rtol=0, atol=0)


def test_columnwise_quantization_is_not_a_transposed_rowwise_cast():
    x = torch.randn(129, 128, device="cuda")
    x[0] *= 1000
    row = quantize_fp8(x)
    col = quantize_fp8(x, transpose=True)
    assert not torch.equal(row.dequantize().T, col.dequantize())
    with pytest.raises(ValueError, match="columnwise"):
        transpose_fp8_weight(row)


def test_weight_preparation_and_versions():
    w1 = torch.randn(2, 256, 128, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(2, 128, 128, device="cuda", dtype=torch.bfloat16)
    for version in (-1, 2**64):
        with pytest.raises(ValueError, match="unsigned 64-bit"):
            prepare_fp8_weights(w1, w2, version=version)
    original = prepare_fp8_weights(w1, w2, version=7)
    original.wait(training=True)
    original.validate_parameters(w1, w2)
    with pytest.raises(ValueError, match="different parameters"):
        original.validate_parameters(w1.clone(), w2)
    torch.testing.assert_close(original.w1_t.dequantize(), original.w1.dequantize().mT, rtol=0, atol=0)
    # An imported checkpoint may use general FP32 dequantization scales.
    scales = original.w1.scales * 1.25
    imported_w1 = FP8Tensor(original.w1.data, scales, original.w1.shape, 128)
    imported = prepare_fp8_weights(imported_w1, original.w2, version=8)
    scales.zero_()
    imported.wait(training=True)  # importing owns a snapshot
    assert (imported.w1.scales > 0).all()
    original.w1.scales.mul_(2)
    with pytest.raises(RuntimeError, match="modified"):
        original.wait()
    inference = prepare_fp8_weights(w1, w2, training=False)
    with pytest.raises(ValueError, match="training=True"):
        inference.wait(training=True)
    with torch.inference_mode():
        inference = prepare_fp8_weights(w1, w2, training=False)
        inference.wait()
    w1.add_(0.1)
    with pytest.raises(RuntimeError, match="stale"):
        inference.validate_parameters(w1, w2)


def test_output_validation_and_cuda_graph():
    x = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    aq, bq = quantize_fp8(x), quantize_fp8(x, block_rows=128)
    with pytest.raises(ValueError, match="existing FP32"):
        fp8_gemm(aq, bq, accumulate=True)
    out = torch.empty((64, 64), device="cuda", dtype=torch.float32)
    fp8_gemm(aq, bq, out=out, dtype=torch.float32)
    expected = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fp8_gemm(aq, bq, out=out, dtype=torch.float32)
    out.zero_()
    graph.replay()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize('nan_byte', [0x7f, 0xff])
def test_imported_nan_weight_is_rejected(nan_byte):
    x = torch.ones(1, 256, 128, device='cuda', dtype=torch.bfloat16)
    w = quantize_fp8(x, block_rows=128)
    w.data.view(torch.uint8)[0, 0, 0] = nan_byte
    with pytest.raises(ValueError, match='finite'):
        prepare_fp8_weights(w, quantize_fp8(x[:, :128].contiguous(), block_rows=128))


@pytest.mark.parametrize('bad', [float('inf'), float('nan')])
def test_nonfinite_source_weight_is_rejected(bad):
    x = torch.ones(1, 256, 128, device='cuda', dtype=torch.bfloat16)
    x[0, 0, 0] = bad
    with pytest.raises(ValueError, match='finite'):
        prepare_fp8_weights(x, x[:, :128].contiguous())
