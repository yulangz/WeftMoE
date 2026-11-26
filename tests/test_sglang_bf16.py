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
