"""Recapture must not reuse request/KV indices from a completed rollout."""

from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch


def test_recapture_without_optional_version_specific_buffers():
    from weftmoe.sglang import _reset_decode_inputs
    buffers = SimpleNamespace(seq_lens=torch.full((4,), 19),
                              seq_lens_cpu=torch.full((4,), 19))
    _reset_decode_inputs(SimpleNamespace(buffers=buffers, seq_len_fill_value=2))
    assert torch.equal(buffers.seq_lens, torch.full((4,), 2))
    assert torch.equal(buffers.seq_lens_cpu, torch.full((4,), 2))


def test_recapture_restores_sglang_constructor_inputs():
    pytest.importorskip("sglang")
    from sglang.srt.model_executor.cuda_graph_runner import DecodeInputBuffers
    from weftmoe.sglang import _reset_decode_inputs

    kwargs = dict(
        device=torch.device("cpu"), max_bs=4, max_num_token=4, hidden_size=128,
        vocab_size=256, dtype=torch.bfloat16, dp_size=2, pp_size=2,
        is_encoder_decoder=True, require_mlp_tp_gather=True,
        seq_len_fill_value=2, encoder_len_fill_value=3, num_tokens_per_bs=1,
        cache_loc_dtype=torch.int64, enable_mamba_track=True,
    )
    buffers = DecodeInputBuffers.create(**kwargs)
    reference = DecodeInputBuffers.create(**kwargs)
    addresses = {}
    for field in fields(buffers):
        value = getattr(buffers, field.name)
        tensors = value.values() if isinstance(value, dict) else [value]
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor):
                addresses[id(tensor)] = tensor.data_ptr()
                tensor.fill_(19 if tensor.dtype != torch.bool else False)
    _reset_decode_inputs(SimpleNamespace(
        buffers=buffers, seq_len_fill_value=2, encoder_len_fill_value=3))
    for field in fields(buffers):
        actual, expected = getattr(buffers, field.name), getattr(reference, field.name)
        pairs = zip(actual.values(), expected.values()) if isinstance(actual, dict) else [(actual, expected)]
        for actual_tensor, expected_tensor in pairs:
            if actual_tensor is not None:
                assert torch.equal(actual_tensor, expected_tensor), field.name
                assert addresses[id(actual_tensor)] == actual_tensor.data_ptr()


def test_tms_private_api_guard(monkeypatch):
    from weftmoe.sglang import _check_fp8_tms_compatibility
    impl = SimpleNamespace(_primary_mem_pool=object())
    monkeypatch.setattr(torch, "__version__", "2.9.1+cu129")
    _check_fp8_tms_compatibility(impl)
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    with pytest.raises(RuntimeError, match="revalidate"):
        _check_fp8_tms_compatibility(impl)
    monkeypatch.setattr(torch, "__version__", "2.9.1")
    monkeypatch.setattr(torch.cuda.memory, "_cuda_releasePool", None)
    with pytest.raises(RuntimeError, match="allocation-pool"):
        _check_fp8_tms_compatibility(impl)
