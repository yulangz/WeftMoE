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
    assert first._cache_key == 0
    version = parameter._version
    parameter.data.add_(1)
    assert parameter._version == version
    assert torch.equal(forward().value, first.value + 1)
    optimizer = torch.optim.AdamW([parameter], lr=0.5)
    manager.bind_optimizer(optimizer)
    cached = forward()
    assert cached._cache_key == id(layer)
    assert forward() is cached
    hook = manager._optimizer_hook
    # Wrappers must not be silently unwrapped or remove a working binding.
    for unsupported in (SimpleNamespace(optimizer=optimizer),
                        torch.optim.SGD([parameter], lr=0.5)):
        with pytest.raises(TypeError, match='torch.optim.AdamW.*MegatronOptimizer'):
            manager.bind_optimizer(unsupported)
        assert manager._optimizer_hook is hook
        assert forward() is cached
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    updated = forward()
    assert not torch.equal(updated.value, cached.value)
    assert torch.equal(updated.value, parameter.unsqueeze(0))
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

    master = MasterWeights.__new__(MasterWeights)
    master.materialize = materialize
    master.bindings = {0: (None, None, None, parameter)}
    master.adamw_writers = {}
    master.bucket_groups = []
    master.buffers = []
    manager.master_weights = master
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
    master.dirty = False
    master.bindings = {id(p): (layer, i, 0, p) for i, p in enumerate(parameters)}
    master.weights = {
        layer: tuple(_storage((1, 128, 128), 128, device) for _ in range(4))
    }
    master.adapter = SimpleNamespace(fp8=True, weight_version=1, _parameters=lambda _: parameters)
    master.publish()
    old = master.adapter.fp8_weights[layer][1]

    def quantize(source, row, scale, col, col_scale, stream):
        # Emulate a raw CUDA write: no automatic PyTorch version increment.
        row.data.copy_(source)
        col.data.copy_(source.T)
        scale.data.fill_(1)
        col_scale.data.fill_(1)

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
    assert master.dirty and master.adapter.weight_version == 2
    with pytest.raises(RuntimeError, match="modified"):
        old._validate_version()
    from weftmoe.megatron import _Routed
    with pytest.raises(RuntimeError, match="changed before FP8 backward"):
        _Routed.backward(SimpleNamespace(manager=master.adapter, weight_version=1), None)
    master.publish()
    assert not master.dirty
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
    weights = _seal_weights((*row, None, None), 7, cache_key=123)
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
    assert prepared.version == (1 if discard else 7)
    assert torch.equal(
        prepared.w1.data.float(),
        torch.full((1, 128, 128), 3.0 if discard else 2.0, device=device),
    )


def test_master_dirty_blocks_forward_until_publication():
    manager = MegatronAdapter.__new__(MegatronAdapter)
    manager.master_weights = SimpleNamespace(dirty=True)
    manager.weight_version = 4
    manager.get_runtime = lambda _: pytest.fail("must reject before touching runtime")
    with pytest.raises(RuntimeError, match="Publish"):
        manager.forward_experts(None, None, None, None, [])
    def publish():
        manager.master_weights.dirty = False
    manager.master_weights.publish = publish
    manager.weights_updated()
    assert manager.weight_version == 4  # write already invalidated this generation
    assert not manager.master_weights.dirty


@pytest.fixture
def staged_master_inputs():
    """Small CPU objects exercising validation before any native allocation."""
    backing = torch.zeros(32, dtype=torch.bfloat16)
    params = [torch.nn.Parameter(backing[i:i + 16].view(4, 4)) for i in (0, 16)]

    class Leaf:
        def drain(self):
            pass

        def set_master_writers(self, writers):
            pytest.fail("invalid binding must not install writers")

    leaf = Leaf()
    leaf.param_groups = [{"params": params}]
    leaf.state = {p: {"master_param": torch.ones_like(p, dtype=torch.float32)} for p in params}
    leaf.staged_config = SimpleNamespace(bucket_numel=16)
    wrapper = SimpleNamespace(
        optimizer=leaf,
        config=SimpleNamespace(use_distributed_optimizer=True, use_precision_aware_optimizer=True),
        model_param_group_index_map={p: (0, i) for i, p in enumerate(params)},
        _get_model_param_range_map=lambda p: {"param": SimpleNamespace(start=0, end=p.numel())},
    )
    buffer = SimpleNamespace(params=params.copy(), param_data=backing, data_parallel_world_size=1)
    group = SimpleNamespace(
        params=params.copy(), ddp_config=SimpleNamespace(use_distributed_optimizer=True),
        intra_distributed_optimizer_instance_size=1, param_gather_handle=None,
    )
    model = SimpleNamespace(expert_parallel_buffers=[buffer], expert_parallel_bucket_groups=[group])
    layer = SimpleNamespace(experts=SimpleNamespace(linear_fc1=object(), linear_fc2=object()))
    adapter = SimpleNamespace(layers=[layer], _parameters=lambda _: params, _models=[model])
    return SimpleNamespace(adapter=adapter, wrapper=wrapper, leaf=leaf, buffer=buffer,
                           group=group, params=params, backing=backing)


@pytest.mark.parametrize("size", [0, 32, 96])
def test_master_binding_rejects_incomplete_flat_storage(staged_master_inputs, size):
    inputs = staged_master_inputs
    storage = inputs.backing.untyped_storage()
    original = storage.nbytes()
    try:
        storage.resize_(size)
        with pytest.raises(ValueError, match="buffer must be fully resident"):
            MasterWeights(inputs.adapter, inputs.wrapper)
        assert storage.nbytes() == size
        assert not hasattr(inputs.params[0], "_weftmoe_fp8_writer")
    finally:
        storage.resize_(original)


def test_master_binding_rejects_parameter_outside_flat_storage(staged_master_inputs):
    inputs = staged_master_inputs
    storage = inputs.backing.untyped_storage()
    size = storage.nbytes()
    storage.resize_(80)
    with torch.no_grad():
        inputs.params[1].set_(storage, 24, (4, 4), (4, 1))
    storage.resize_(size)
    try:
        with pytest.raises(ValueError, match="beyond its allocated storage"):
            MasterWeights(inputs.adapter, inputs.wrapper)
    finally:
        with torch.no_grad():
            inputs.params[1].set_(storage, 16, (4, 4), (4, 1))


@pytest.mark.parametrize("case,match", [
    ("distributed", "use_distributed_optimizer=True"),
    ("precision", "use_precision_aware_optimizer=True"),
    ("state", "initialized staged optimizer master_param"),
    ("mixed_buffer", "buffer must contain only selected experts"),
    ("mixed_bucket", "bucket must contain only selected experts"),
    ("buffer_dp", "expert DP=1"),
    ("bucket_dp", "expert DP=1"),
    ("slot", "exceeds the AdamW staging slot"),
])
def test_master_binding_rejects_unsupported_adamw(staged_master_inputs, case, match):
    inputs = staged_master_inputs
    if case == "distributed":
        inputs.wrapper.config.use_distributed_optimizer = False
        del inputs.wrapper.model_param_group_index_map
    elif case == "precision":
        inputs.wrapper.config.use_precision_aware_optimizer = False
    elif case == "state":
        del inputs.leaf.state[inputs.params[0]]["master_param"]
    elif case.startswith("mixed_"):
        target = inputs.buffer if case == "mixed_buffer" else inputs.group
        target.params.append(torch.nn.Parameter(torch.ones(1)))
    elif case == "buffer_dp":
        inputs.buffer.data_parallel_world_size = 2
    elif case == "bucket_dp":
        inputs.group.intra_distributed_optimizer_instance_size = 2
    elif case == "slot":
        inputs.leaf.staged_config.bucket_numel = 15
    with pytest.raises(ValueError, match=match):
        MasterWeights(inputs.adapter, inputs.wrapper)
    assert inputs.backing.untyped_storage().nbytes() == 64
    assert not hasattr(inputs.params[0], "_weftmoe_fp8_writer")


def test_master_drain_ignores_ordinary_dense_leaf():
    drained = []
    master = MasterWeights.__new__(MasterWeights)
    staged = SimpleNamespace(drain=lambda: drained.append("staged"))
    ordinary = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))])
    master.leaves = (SimpleNamespace(optimizer=staged), SimpleNamespace(optimizer=ordinary))
    master.optimizer = SimpleNamespace(chained_optimizers=master.leaves)
    master._drain()
    assert drained == ["staged"]
    master.optimizer.drain = lambda: drained.append("chain")
    master._drain()
    assert drained == ["staged", "chain"]


@pytest.mark.parametrize("operation", ["refresh", "materialize", "resume"])
def test_master_restore_rejects_replaced_checkpoint_tensor(operation):
    parameter = torch.nn.Parameter(torch.ones(2))
    tensor = torch.ones(2)
    leaf = SimpleNamespace(state={parameter: {"master_param": tensor}})
    master = MasterWeights.__new__(MasterWeights)
    master.masters = {id(parameter): tensor}
    master.master_sources = {id(parameter): (leaf, parameter)}
    master.optimizer = SimpleNamespace(drain=lambda: None)
    master.offloaded = True
    # In-place checkpoint loads retain the supported slab identity.
    tensor.copy_(torch.full_like(tensor, 3))
    master._validate_masters()
    leaf.state[parameter]["master_param"] = tensor.clone()
    with pytest.raises(RuntimeError, match="replaced a bound FP32 master"):
        getattr(master, operation)()
    assert master.offloaded  # Fail before attempting allocation or suspension.


def test_master_detach_restores_writers_and_group_sync():
    from types import MethodType

    class Group:
        def start_param_sync(self, force_sync=False):
            return "class method"

    inherited, overridden = Group(), Group()
    original = lambda force_sync=False: "instance override"
    overridden.start_param_sync = original
    groups = [(g, g.__dict__.get("start_param_sync")) for g in (inherited, overridden)]
    for group, _ in groups:
        group.param_gather_handle = None
        group.param_gather_dispatched = False
        group.start_param_sync = MethodType(MasterWeights._skip_param_sync, group)
        group.start_param_sync(force_sync=True)
        assert group.param_gather_dispatched
    inherited.param_gather_handle = object()
    with pytest.raises(AssertionError):
        inherited.start_param_sync()
    inherited.param_gather_handle = None

    calls = []

    class Leaf:
        def set_master_writers(self, writers):
            calls.append(writers)

    parameter = torch.nn.Parameter(torch.ones(2))
    parameter._weftmoe_fp8_storage = parameter.untyped_storage()
    parameter._weftmoe_fp8_writer = object()
    buffer = SimpleNamespace(_weftmoe_fp8_master_managed=True)
    master = MasterWeights.__new__(MasterWeights)
    master.adamw_writers = {Leaf(): {parameter: object()}}
    master.bucket_groups = groups
    master.buffers = [buffer]
    master.bindings = {id(parameter): (None, None, None, parameter)}
    master.detach()
    assert calls == [{}]
    assert "start_param_sync" not in inherited.__dict__
    assert inherited.start_param_sync() == "class method"
    assert overridden.start_param_sync is original
    assert overridden.start_param_sync() == "instance override"
    assert not hasattr(buffer, "_weftmoe_fp8_master_managed")
    assert not hasattr(parameter, "_weftmoe_fp8_writer")
    assert not hasattr(parameter, "_weftmoe_fp8_storage")
