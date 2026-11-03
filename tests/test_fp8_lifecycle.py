"""Weight publication, optimizer invalidation and residency regressions."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from weftmoe.megatron import MegatronAdapter
from weftmoe._master_weights import MasterWeights


@torch.no_grad()
def test_unbound_data_updates_do_not_reuse_quantized_weights(monkeypatch):
    import weftmoe.fp8 as fp8

    @dataclass
    class Snapshot:
        value: torch.Tensor
        _cache_key: int = 0

    monkeypatch.setattr(
        fp8, "prepare_fp8_weights", lambda a, b, **kw: Snapshot(a.clone())
    )
    manager = MegatronAdapter.__new__(MegatronAdapter)
    manager.fp8 = True
    manager.master_weights = None
    manager._optimizer_hook = None
    manager.fp8_weights = {}
    manager.packed = {}
    manager.weight_version = 0
    manager.get_runtime = lambda layer: SimpleNamespace(
        forward_fp8=lambda x, r, g, weights, **kw: weights
    )
    layer = object()
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameters = [parameter, parameter]
    forward = lambda: manager.forward_experts(layer, None, None, None, parameters)
    first = forward()
    version = parameter._version
    parameter.data.add_(1)
    assert parameter._version == version
    assert torch.equal(forward().value, first.value + 1)
    optimizer = torch.optim.SGD([parameter], lr=0.5)
    manager.bind_optimizer(optimizer)
    cached = forward()
    assert forward() is cached
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.equal(forward().value, cached.value - 0.5)
    manager._optimizer_hook.remove()


def test_close_does_not_discard_materialized_parameters(monkeypatch):
    parameter = torch.nn.Parameter(torch.ones(2))
    parameter._weftmoe_fp8_storage = parameter.untyped_storage()
    parameter._weftmoe_fp8_writer = None
    parameter.untyped_storage().resize_(0)
    manager = MegatronAdapter.__new__(MegatronAdapter)
    manager.runtime = None
    manager.session = object()
    manager.packed = {}
    manager.fp8_weights = {}
    manager._optimizer_hook = None

    def materialize():
        parameter.untyped_storage().resize_(8)
        parameter.data.fill_(3)

    manager.master_weights = SimpleNamespace(
        materialize=materialize, bindings={0: (None, None, None, parameter)}
    )
    # suspend after materialize would discard the restored BF16 storage again.
    manager.suspend = lambda: parameter.untyped_storage().resize_(0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    manager.close()
    assert torch.equal(parameter, torch.full((2,), 3.0))
    assert not hasattr(parameter, "_weftmoe_fp8_writer")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_master_publish_waits_for_all_writer_streams(monkeypatch):
    import weftmoe._master_weights as module
    from weftmoe.fp8 import _storage

    device = torch.device("cuda", torch.cuda.current_device())
    parameters = [torch.empty(128, 128, device=device) for _ in range(2)]
    layer = object()
    master = MasterWeights.__new__(MasterWeights)
    master.device = device
    master.offloaded = False
    master.writer_streams = {}
    master.bindings = {id(p): (layer, i, 0, p) for i, p in enumerate(parameters)}
    master.weights = {
        layer: tuple(_storage((1, 128, 128), 128, device) for _ in range(4))
    }
    master.adapter = SimpleNamespace(weight_version=1, _parameters=lambda _: parameters)

    def quantize(source, row, scale, col, col_scale, stream):
        row.copy_(source)
        col.copy_(source.T)
        scale.fill_(1)
        col_scale.fill_(1)

    monkeypatch.setattr(
        module, "_extension", lambda: SimpleNamespace(fp8_quantize_weight=quantize)
    )
    streams = [torch.cuda.Stream() for _ in parameters]
    for i, (parameter, writer) in enumerate(zip(parameters, streams)):
        writer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(writer):
            torch.cuda._sleep(20_000_000)
            source = torch.full((128, 128), i + 1.0, device=device)
            master.write(parameter, source)
    master.publish()
    prepared = master.adapter.fp8_weights[layer][1]
    with torch.cuda.stream(torch.cuda.Stream()):
        prepared.wait(training=True)
        snapshots = [
            t.float().cpu()
            for t in (
                prepared.w1.data,
                prepared.w2.data,
                prepared.w1_t.data,
                prepared.w2_t.data,
            )
        ]
    for i, snapshot in enumerate(snapshots):
        assert torch.equal(snapshot, torch.full_like(snapshot, i % 2 + 1))
    assert not master.writer_streams


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("discard", [False, True])
@pytest.mark.parametrize("already_offloaded", [False, True])
def test_sglang_offload_preserves_or_requires_full_update(
    monkeypatch, discard, already_offloaded
):
    import weftmoe.sglang as adapter
    from weftmoe.fp8 import _storage, _seal_weights

    device = torch.device("cuda", torch.cuda.current_device())
    row = [_storage((1, 128, 128), 128, device) for _ in range(2)]
    for w in row:
        w.data.fill_(2)
        w.scales.fill_(1)
    weights = _seal_weights((*row, None, None), 0, cache_key=123)
    for name, value in dict(
        _fp8_weights={0: weights},
        _fp8_devices={0: device},
        _fp8_offloaded=False,
        _fp8_needs_update=False,
        _graph_inputs={},
        _runtime=None,
        _decode_runtime=None,
        _eager_pending=None,
    ).items():
        monkeypatch.setattr(adapter, name, value)
    if already_offloaded:
        adapter.close(SimpleNamespace(), keep_fp8=True)
    adapter.close(SimpleNamespace(), keep_fp8=True, discard_fp8=discard)
    for tensor in weights._tensors():
        assert tensor.device.type == "cpu"
        assert tensor.untyped_storage().nbytes() == (
            0 if discard else tensor.numel() * tensor.element_size()
        )
    if discard:
        with pytest.raises(RuntimeError, match="Complete"):
            adapter.resume_fp8_weights()
        adapter.resume_fp8_weights(for_update=True)
        with pytest.raises(RuntimeError, match="Complete"):
            adapter._prepare_fp8_layer(SimpleNamespace(layer_id=0))
        for w in row:
            w.data.fill_(3)
            w.scales.fill_(1)
        adapter.commit_fp8_weights(1)
    else:
        adapter.resume_fp8_weights()
    prepared = adapter._prepare_fp8_layer(SimpleNamespace(layer_id=0))
    prepared.wait()
    assert prepared._cache_key == 123
    assert torch.equal(
        prepared.w1.data.float(),
        torch.full((1, 128, 128), 3.0 if discard else 2.0, device=device),
    )
