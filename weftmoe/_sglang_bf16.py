"""Single-copy BF16 residency for SGLang's serialized serving and weight-update phases."""

from contextlib import contextmanager
from threading import RLock
from weakref import WeakKeyDictionary, WeakSet

import torch

from .blackwell import _bf16_weight_views, _validate_expert_weights, _weight_tile_shape

_layers = WeakKeyDictionary()
_inactive = WeakSet()


def _copy_experts(destination, source, scratch):
    # Reuse one expert-sized buffer on the current stream. Allocate it before
    # mutating either matrix so an allocation failure leaves the layer intact.
    temp = scratch[:source[0].numel()].view(source.shape[1:])
    for dst, src in zip(destination.unbind(), source.unbind(), strict=True):
        temp.copy_(src)
        dst.copy_(temp)


class _ExpertWeights:
    def __init__(self, layer):
        self.storage = (layer.w13_weight.detach(), layer.w2_weight.detach())
        _validate_expert_weights(*self.storage)
        self.tiles = tuple(p.view(_weight_tile_shape(*p.shape, 64)) for p in self.storage)
        self.active = False
        self.pending_shards = False
        self.loader_lock = RLock()
        _inactive.add(self)

    def scratch(self):
        return self.storage[0].new_empty(self.storage[0][0].numel())

    def canonical_copy(self):
        result = tuple(torch.empty_like(p) for p in self.storage)
        for view, tile in zip(_bf16_weight_views(*result), self.tiles, strict=True):
            view.copy_(tile.view(view.shape))
        return result

    @torch.inference_mode()
    def pack(self, layer):
        if self.active:
            return
        params = (layer.w13_weight, layer.w2_weight)
        _validate_expert_weights(*params)
        if any(p.shape != buf.shape or p.device != buf.device for p, buf in zip(params, self.storage)):
            raise ValueError("Reloaded BF16 experts must match the serving shape and device")
        inplace = [p.data_ptr() == buf.data_ptr() for p, buf in zip(params, self.storage)]
        # A replacement parameter may be another slice of a flattened allocation.
        # Only disjoint replacements can be copied directly back to the graph address.
        for p, same in zip(params, inplace):
            if not same and any(
                p.data_ptr() < buf.data_ptr() + buf.nbytes and buf.data_ptr() < p.data_ptr() + p.nbytes
                for buf in self.storage
            ):
                raise ValueError("Reloaded BF16 experts partially overlap serving storage")
        scratch = self.scratch() if any(inplace) else None
        for p, buf, tile, src, same in zip(params, self.storage, self.tiles, _bf16_weight_views(*params), inplace):
            dst = tile.view(src.shape)
            if same:
                _copy_experts(dst, src, scratch)
            else:
                dst.copy_(src)
                p.data = buf
        self.active = True
        self.pending_shards = False
        _inactive.discard(self)

    @torch.inference_mode()
    def restore(self):
        if self.active:
            scratch = self.scratch()
            for dst, tile in zip(_bf16_weight_views(*self.storage), self.tiles):
                _copy_experts(dst, tile.view(dst.shape), scratch)
            self.active = False
            _inactive.add(self)
        self.pending_shards = False


@torch.no_grad()
def serving_weights(layer):
    weights = _layers.get(layer)
    if weights is None or not weights.active:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare SGLang BF16 experts before CUDA Graph capture")
        if weights is None:
            weights = _layers[layer] = _ExpertWeights(layer)
            if not getattr(layer, "_weftmoe_state_hooks", False):
                layer.register_state_dict_post_hook(_save_state)
                layer.register_load_state_dict_pre_hook(_before_load)
                layer.register_load_state_dict_post_hook(_after_load)
                layer._weftmoe_state_hooks = True
        elif not weights.pending_shards:
            raise RuntimeError("BF16 experts are unavailable during a weight update or suspension")
        if weights.pending_shards:
            torch.cuda.synchronize()
        weights.pack(layer)
    return weights.tiles


def _save_state(layer, state, prefix, metadata):
    weights = _layers.get(layer)
    if weights is None:
        return
    keys = (prefix + "w13_weight", prefix + "w2_weight")
    if weights.active:
        state[keys[0]], state[keys[1]] = weights.canonical_copy()
    else:
        for key in keys:
            state[key] = state[key].clone()


@contextmanager
def loading_shard(layer):
    weights = _layers.get(layer)
    if weights is None:
        yield
        return
    # Parallel checkpoint readers can invoke multiple loaders on one layer.
    # A fused loader may also delegate to another loader on the same thread.
    with weights.loader_lock:
        standalone = weights.active or weights.pending_shards
        if weights.active:
            device = weights.storage[0].device
            torch.cuda.synchronize(device)
            weights.restore()
            torch.cuda.synchronize(device)
        weights.pending_shards = False
        yield
        weights.pending_shards = standalone


def prepare_for_replay():
    if _inactive:
        if not all(w.pending_shards for w in _inactive):
            raise RuntimeError("BF16 experts are unavailable during a weight update or suspension")
        prepare_all()


def _before_load(layer, *args):
    weights = _layers.get(layer)
    if weights is not None and (weights.active or weights.pending_shards):
        torch.cuda.synchronize()
        weights.restore()
        torch.cuda.synchronize()
        layer._weftmoe_repack_after_load = True


def _after_load(layer, incompatible_keys):
    if getattr(layer, "_weftmoe_repack_after_load", False):
        torch.cuda.synchronize()
        _layers[layer].pack(layer)
        layer._weftmoe_repack_after_load = False


@torch.no_grad()
def restore_all():
    if _layers:
        torch.cuda.synchronize()
        for weights in list(_layers.values()):
            weights.restore()
        torch.cuda.synchronize()


@torch.no_grad()
def prepare_all():
    if _layers:
        torch.cuda.synchronize()
    for layer, weights in list(_layers.items()):
        weights.pack(layer)
    if _layers:
        torch.cuda.synchronize()


@contextmanager
def canonical_weights(layer=None):
    """Expose canonical BF16 parameters for a serialized read or update, then repack."""
    candidates = _layers.items() if layer is None else ((layer, _layers.get(layer)),)
    active = [(owner, w) for owner, w in candidates if w is not None and (w.active or w.pending_shards)]
    try:
        if active:
            torch.cuda.synchronize()
            for _, weights in active:
                weights.restore()
            torch.cuda.synchronize()
        yield
    finally:
        if active:
            torch.cuda.synchronize()
            with torch.no_grad():
                for layer, weights in active:
                    weights.pack(layer)
                torch.cuda.synchronize()


def close():
    restore_all()
    _layers.clear()
    _inactive.clear()
