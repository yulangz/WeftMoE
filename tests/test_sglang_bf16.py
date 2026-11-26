from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from weftmoe import _sglang_bf16 as resident
from weftmoe.blackwell import prepare_bf16_weights
from weftmoe.blackwell_cutlass import max_decode_tile, tile_m


def test_decode_workspace_covers_non_monotonic_tiles():
    assert tile_m(8, 256, 800, 8) == 240
    assert tile_m(8, 256, 1024, 8) == 192
    assert max_decode_tile(8, 256, 1024, 8, {}) == 240
    assert max_decode_tile(8, 256, 16, 8, {8: (128, False), 17: (240, False)}) == 128


@pytest.fixture
def canonical_layer():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA BF16 storage")
    layer = torch.nn.Module()
    for key, shape in (("w13_weight", (8, 512, 512)), ("w2_weight", (8, 512, 256))):
        tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        setattr(layer, key, torch.nn.Parameter(tensor, requires_grad=False))
    yield layer
    resident.close()


@pytest.fixture
def layer(canonical_layer):
    resident.serving_weights(canonical_layer)
    return canonical_layer


def test_shard_batch_snapshot_failure_and_graph_replay(layer, monkeypatch):
    holder = resident._layers[layer]
    with resident.canonical_weights():
        snapshot = layer.state_dict()
        with pytest.raises(RuntimeError, match="weight update"):
            resident.prepare_for_replay()
    canonical = holder.canonical_copy()
    assert all(torch.equal(v, t) for v, t in zip(snapshot.values(), canonical))
    restores = []
    restore = holder.restore

    def count_restore():
        restores.append(1)
        restore()

    monkeypatch.setattr(holder, "restore", count_restore)

    @resident.checkpoint_loader
    def write_shard(layer, expert_id, weight_name="w13_weight", shard_id="w1"):
        layer.w13_weight[expert_id].fill_(expert_id + 1)

    with ThreadPoolExecutor(3) as readers:
        list(readers.map(lambda expert: write_shard(layer, expert), range(3)))
    assert not holder.active and len(restores) == 1
    updated = layer.state_dict()
    resident.prepare_for_replay()
    assert holder.active
    assert torch.equal(holder.canonical_copy()[0], updated["w13_weight"])
    for expert in range(3):
        assert torch.all(updated["w13_weight"][expert] == expert + 1)

    def fail_load(*args):
        raise RuntimeError("interrupted load")

    hook = layer.register_load_state_dict_pre_hook(fail_load)
    with pytest.raises(RuntimeError, match="interrupted"):
        layer.load_state_dict(snapshot)
    with pytest.raises(RuntimeError, match="weight update"):
        resident.prepare_for_replay()
    hook.remove()
    layer.load_state_dict(snapshot)
    resident.prepare_for_replay()
    assert torch.equal(holder.canonical_copy()[0], canonical[0])


def test_scratch_bound_and_replaced_parameter_storage(layer):
    holder = resident._layers[layer]
    addresses = [weight.data_ptr() for weight in holder.storage]
    limit = layer.w13_weight[0].nbytes
    for operation in (holder.restore, lambda: holder.pack(layer)):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        operation()
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() - before <= limit

    with resident.canonical_weights(layer):
        layer.w13_weight.data = torch.randn_like(layer.w13_weight)
        layer.w2_weight.data = torch.randn_like(layer.w2_weight)
        expected = prepare_bf16_weights(layer.w13_weight, layer.w2_weight)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    assert torch.cuda.max_memory_allocated() == before
    assert [v.data_ptr() for v in holder.tiles] == addresses
    assert all(torch.equal(v, t) for v, t in zip(holder.tiles, expected))


def test_first_pack_allocation_failure_can_retry(canonical_layer, monkeypatch):
    layer = canonical_layer
    expected = [weight.detach().clone() for weight in layer.parameters()]
    with monkeypatch.context() as patch:
        def oom(_):
            raise torch.OutOfMemoryError("first pack scratch")

        patch.setattr(resident._ExpertWeights, "scratch", oom)
        with pytest.raises(torch.OutOfMemoryError) as error:
            resident.serving_weights(layer)
    assert "first pack scratch" in str(error.value)
    assert layer not in resident._layers
    resident.prepare_for_replay()
    assert all(torch.equal(w, v) for w, v in zip(layer.parameters(), expected))
    actual = resident.serving_weights(layer)
    assert all(torch.equal(w, v) for w, v in zip(actual, prepare_bf16_weights(*expected)))
    assert all(torch.equal(w, v) for w, v in zip(layer.state_dict().values(), expected))


@pytest.mark.parametrize("replay", [False, True])
def test_retry_failed_shards_before_serving(layer, replay):
    original = layer.state_dict()
    addr = [p.data_ptr() for p in layer.parameters()]

    @resident.checkpoint_loader
    def load(layer, weight_name, shard_id, expert_id, fail=False):
        layer.w13_weight[expert_id].fill_(expert_id + 1)
        if fail:
            raise ValueError("short read")

    def assert_blocked():
        for op in (lambda: resident.serving_weights(layer), resident.prepare_for_replay, resident.prepare_all):
            with pytest.raises(RuntimeError):
                op()

    for expert in (1, 2):
        with pytest.raises(ValueError, match="short read"):
            load(layer, weight_name="w13_weight", shard_id="w1", expert_id=expert, fail=True)
    assert_blocked()
    with resident.canonical_weights(layer):
        layer.state_dict()
    load(layer, "w13_weight", "w1", 0)
    load(layer, "w13_weight", "w1", 1)
    assert_blocked()
    load(layer, "w13_weight", "w1", 2)
    if replay:
        resident.prepare_for_replay()
    else:
        resident.serving_weights(layer)
    resident.prepare_for_replay()
    assert resident._layers[layer].active
    assert [p.data_ptr() for p in layer.parameters()] == addr
    for expert in range(3):
        original["w13_weight"][expert].fill_(expert + 1)
    assert all(torch.equal(w, v) for w, v in zip(layer.state_dict().values(), original.values()))


@pytest.mark.parametrize("return_status", [False, True])
def test_failed_weight_update_blocks_all_layers_until_batch_retry(layer, return_status):
    other = torch.nn.Module()
    for name, value in layer.state_dict().items():
        setattr(other, name, torch.nn.Parameter(value, requires_grad=False))
    resident.serving_weights(other)
    layers = (layer, other)
    addresses = [owner.w13_weight.data_ptr() for owner in layers]

    @resident.checkpoint_loader
    def load(owner, weight_name, shard_id, expert_id, value):
        if value < 0:
            raise ValueError("checkpoint read failed")
        owner.w13_weight[expert_id].fill_(value)

    def update_weights_from_tensor(named_tensors):
        try:
            for name, value in named_tensors:
                load(layers[int(name)], "w13_weight", "w1", 0, value)
        except ValueError as exc:
            if return_status:
                return False, str(exc)
            raise
        return True, "OK"

    update = resident.weight_api(update_weights_from_tensor, writes=True)
    export = resident.weight_api(lambda: layer.state_dict(), writes=False)
    if return_status:
        assert update([("0", 5), ("1", -1)]) == (False, "checkpoint read failed")
    else:
        with pytest.raises(ValueError, match="checkpoint read failed"):
            update([("0", 5), ("1", -1)])
    assert torch.all(export()["w13_weight"][0] == 5)
    # Repairing the individual failed shard alone does not finish the original batch.
    assert update([("1", 6)])[0]
    for op in (resident.prepare_for_replay, resident.prepare_all):
        with pytest.raises(RuntimeError):
            op()
    for owner in layers:
        with pytest.raises(RuntimeError):
            resident.serving_weights(owner)
    assert update([("1", 8), ("0", 7)])[0]
    resident.prepare_for_replay()
    for owner, address, value in zip(layers, addresses, (7, 8)):
        assert resident.serving_weights(owner)[0].data_ptr() == address
        assert torch.all(owner.state_dict()["w13_weight"][0] == value)


def test_partial_raw_update_requires_committed_recovery(layer):
    original = layer.state_dict()

    def assert_blocked():
        for op in (resident.prepare_for_replay, resident.prepare_all, lambda: resident.serving_weights(layer)):
            with pytest.raises(RuntimeError):
                op()

    resident.restore_all()  # An update can also run while graph addresses are suspended.
    with pytest.raises(ValueError, match="incomplete"):
        with resident.canonical_weights():
            layer.w13_weight.fill_(0)
            raise ValueError("incomplete transfer")
    with resident.canonical_weights():
        layer.state_dict()
    assert_blocked()
    with pytest.raises(ValueError, match="rollback"):
        with resident.canonical_weights(recover=True):
            layer.w13_weight.copy_(original["w13_weight"])
            raise ValueError("rollback interrupted")
    assert_blocked()
    def update_weights_from_tensor(named_tensors):
        return False, "failed"

    update = resident.weight_api(update_weights_from_tensor, writes=True)
    with pytest.raises(RuntimeError, match="incomplete weight update"):
        with resident.canonical_weights(recover=True):
            assert not update([("w13_weight", None)])[0]
    assert_blocked()
    with resident.canonical_weights(recover=True):
        layer.load_state_dict(original)
        assert_blocked()
    resident.prepare_all()  # Resume the suspended worker.
    resident.prepare_for_replay()
    assert all(torch.equal(x, y) for x, y in zip(layer.state_dict().values(), original.values()))


def test_async_transfer_error_is_not_committed(layer, monkeypatch):
    pending_error = False
    sync = torch.cuda.synchronize

    def synchronize(*args):
        nonlocal pending_error
        if pending_error:
            pending_error = False
            raise RuntimeError("deferred transfer error")
        sync(*args)

    def update_weights_from_distributed(names, fail=False):
        nonlocal pending_error
        layer.w2_weight.fill_(2)
        pending_error = fail
        return True, "OK"

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    update = resident.weight_api(update_weights_from_distributed, writes=True)
    with pytest.raises(RuntimeError, match="deferred transfer error"):
        update(["w2_weight"], fail=True)
    with pytest.raises(RuntimeError, match="weight update"):
        resident.prepare_for_replay()
    assert update(["w2_weight"])[0]
    resident.prepare_for_replay()
    assert torch.all(layer.state_dict()["w2_weight"] == 2)


@pytest.mark.parametrize("failure", ["shape", "copy", "missing", "unexpected", "late_hook"])
def test_root_load_outcome_guards_every_expert(layer, failure):
    other = torch.nn.Module()
    for key, value in layer.state_dict().items():
        setattr(other, key, torch.nn.Parameter(value, requires_grad=False))
    resident.serving_weights(other)
    root = torch.nn.ModuleList([layer, other])
    inputs = root.state_dict()
    inputs["0.w13_weight"].fill_(5)
    bad = dict(inputs)
    hook = None
    if failure == "shape":
        bad["1.w2_weight"] = bad["1.w2_weight"][:1]
    elif failure == "copy":
        bad["1.w2_weight"] = torch.empty_like(bad["1.w2_weight"], device="meta")
    elif failure == "missing":
        del bad["1.w2_weight"]
    elif failure == "unexpected":
        bad["unused"] = torch.empty(0)
    else:
        def reject(*args):
            raise RuntimeError("reject at end of load")
        hook = root.register_load_state_dict_post_hook(reject)
    with pytest.raises(RuntimeError):
        root.load_state_dict(bad, strict=failure != "copy")
    if hook is not None:
        hook.remove()
    for owner in root:
        with pytest.raises(RuntimeError):
            resident.serving_weights(owner)
    with pytest.raises(RuntimeError):
        resident.prepare_for_replay()
    missing = root.load_state_dict({"0.w13_weight": inputs["0.w13_weight"]}, strict=False)
    assert "1.w2_weight" in missing.missing_keys
    with pytest.raises(RuntimeError):
        resident.prepare_for_replay()
    root.load_state_dict(inputs)
    resident.prepare_for_replay()
    assert all(torch.equal(v, inputs[k]) for k, v in root.state_dict().items())
    # Healthy non-strict partial loads retain PyTorch semantics.
    root.load_state_dict({"1.w2_weight": inputs["1.w2_weight"] + 1}, strict=False)
    resident.prepare_for_replay()
    assert torch.equal(root.state_dict()["1.w2_weight"], inputs["1.w2_weight"] + 1)


@pytest.mark.parametrize("method_name", ["save_sharded_model", "save_remote_model"])
def test_sync_export_borrows_only_until_save_returns(layer, method_name):
    expected = layer.state_dict()

    def save():
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        state = layer.state_dict()
        assert torch.cuda.max_memory_allocated() == before
        for key, value in state.items():
            assert value.data_ptr() == getattr(layer, key).data_ptr()
            assert torch.equal(value, expected[key])
        raise OSError("writer failed")

    save.__name__ = method_name
    with resident.canonical_weights():
        with pytest.raises(OSError):
            resident.weight_api(save, writes=False)()
        snapshot = layer.state_dict()
        for key, value in snapshot.items():
            assert value.data_ptr() != getattr(layer, key).data_ptr()
    resident.prepare_for_replay()
    assert all(torch.equal(v, expected[k]) for k, v in snapshot.items())


@pytest.mark.parametrize("replay", [False, True])
def test_consecutive_weight_buckets_convert_once(layer, monkeypatch, replay):
    weights = resident._layers[layer]
    calls = []
    for method in ["restore", "pack"]:
        original = getattr(weights, method)
        def count(*args, method=method, original=original):
            calls.append(method)
            return original(*args)
        monkeypatch.setattr(weights, method, count)

    def update_weights_from_tensor(named_tensors):
        for name, val in named_tensors:
            layer.w2_weight[int(name)].fill_(val)
        return True, "OK"

    update = resident.weight_api(update_weights_from_tensor, writes=True)
    for bucket in range(5):
        assert update([(str(bucket), bucket)])[0]
    assert calls == ["restore"]
    if replay:
        resident.prepare_for_replay()
    resident.serving_weights(layer)
    assert calls == ["restore", "pack"]
    for bucket in range(5):
        assert torch.all(layer.state_dict()["w2_weight"][bucket] == bucket)


def test_state_load_repack_allocation_can_retry(layer, monkeypatch):
    weights = resident._layers[layer]
    inputs = layer.state_dict()
    inputs["w2_weight"].fill_(3)
    with monkeypatch.context() as patch:
        def oom(_):
            raise torch.OutOfMemoryError("repack scratch")
        patch.setattr(weights, "pack", oom)
        with pytest.raises(torch.OutOfMemoryError, match="repack scratch"):
            layer.load_state_dict(inputs)
    resident.prepare_for_replay()
    assert torch.equal(layer.state_dict()["w2_weight"], inputs["w2_weight"])
