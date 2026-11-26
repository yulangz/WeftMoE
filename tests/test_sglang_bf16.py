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
def layer():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA BF16 storage")
    layer = torch.nn.Module()
    for key, shape in (("w13_weight", (8, 512, 512)), ("w2_weight", (8, 512, 256))):
        tensor = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        setattr(layer, key, torch.nn.Parameter(tensor, requires_grad=False))
    resident.serving_weights(layer)
    yield layer
    resident.close()


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

    def write_shard(expert):
        with resident.loading_shard(layer):
            layer.w13_weight[expert].fill_(expert + 1)

    with ThreadPoolExecutor(3) as readers:
        list(readers.map(write_shard, range(3)))
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
