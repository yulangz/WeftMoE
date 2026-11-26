"""Single-copy BF16 residency for SGLang's serialized serving and weight-update phases."""

from contextlib import contextmanager
from functools import wraps
from inspect import signature
from threading import RLock
from weakref import WeakKeyDictionary, WeakSet

import torch

from .blackwell import _bf16_weight_views, _validate_expert_weights, _weight_tile_shape

_layers = WeakKeyDictionary()
_inactive = WeakSet()
_OPAQUE_UPDATE = object()


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
        self.failed_shards = set()
        self.failed_updates = set()
        self.canonical_depth = 0
        self.loader_lock = RLock()

    @property
    def shards_ready(self):
        return self.pending_shards and not (self.failed_shards or self.failed_updates or self.canonical_depth)

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
        if self.canonical_depth or self.failed_updates or self.failed_shards:
            raise RuntimeError("Complete or retry the BF16 weight update before inference")
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
            weights = _ExpertWeights(layer)
            weights.pack(layer)
            _layers[layer] = weights
            if not getattr(layer, "_weftmoe_state_hooks", False):
                layer.register_state_dict_post_hook(_save_state)
                layer.register_load_state_dict_pre_hook(_before_load)
                layer.register_load_state_dict_post_hook(_after_load)
                layer._weftmoe_state_hooks = True
        elif weights.pending_shards:
            torch.cuda.synchronize()
            weights.pack(layer)
        else:
            raise RuntimeError("BF16 experts are unavailable during a weight update or suspension")
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


def checkpoint_loader(method):
    loader_signature = signature(method)

    def shard_key(layer, args, kwargs):
        bound = loader_signature.bind(layer, *args, **kwargs)
        bound.apply_defaults()
        call = bound.arguments
        return call["weight_name"], call["shard_id"], call.get("expert_id")

    @wraps(method)
    def load(layer, *args, **kwargs):
        weights = _layers.get(layer)
        if weights is None:
            return method(layer, *args, **kwargs)
        # Parallel readers can load one layer, and fused loaders can delegate
        # to another loader on the same thread.
        with weights.loader_lock:
            standalone = weights.active or weights.pending_shards
            if weights.active:
                device = weights.storage[0].device
                torch.cuda.synchronize(device)
                weights.restore()
                torch.cuda.synchronize(device)
            weights.pending_shards = False
            try:
                result = method(layer, *args, **kwargs)
            except BaseException:
                weights.failed_shards.add(shard_key(layer, args, kwargs))
                raise
            else:
                if weights.failed_shards:
                    weights.failed_shards.discard(shard_key(layer, args, kwargs))
                return result
            finally:
                weights.pending_shards = standalone

    return load


def prepare_for_replay():
    if _inactive:
        if not all(w.shards_ready for w in _inactive):
            raise RuntimeError("BF16 experts are unavailable during a weight update or suspension")
        prepare_all()


def _before_load(layer, *args):
    weights = _layers.get(layer)
    if weights is not None and (weights.active or weights.shards_ready):
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


def _update_key(name, arguments):
    # Use only declared destinations. Never retain uploaded tensors in a failed
    # batch key or consume a one-shot iterator to identify a retry.
    names = arguments.get("names")
    tensors = arguments.get("named_tensors")
    if isinstance(tensors, (tuple, list)):
        names = [n for n, _ in tensors]
    elif isinstance(tensors, dict) and isinstance(tensors.get("metadata"), (tuple, list)):
        names = [entry["name"] if isinstance(entry, dict) else entry.name for entry in tensors["metadata"]]
    if isinstance(names, (tuple, list)):
        return name, arguments.get("load_format"), tuple(sorted(names))
    if name == "update_weights_from_disk" and arguments.get("weight_name_filter") is None:
        return name, arguments.get("model_path")
    return _OPAQUE_UPDATE


def weight_api(method, *, writes):
    method_signature = signature(method)

    @wraps(method)
    def call(*args, **kwargs):
        if not _layers:
            return method(*args, **kwargs)
        key = _update_key(method.__name__, method_signature.bind(*args, **kwargs).arguments) if writes else None
        with _canonical_access(None, key) as reject:
            result = method(*args, **kwargs)
            if writes and (result is False or (isinstance(result, tuple) and result and result[0] is False)):
                reject()
            return result

    return call


@contextmanager
def _canonical_access(layer, key, recover=False):
    candidates = _layers.items() if layer is None else ((layer, _layers.get(layer)),)
    owners = [(owner, w) for owner, w in candidates if w is not None]
    repack = [(owner, w) for owner, w in owners if w.active or w.pending_shards]
    # Track suspended layers too. A failed write during offload must still
    # prevent resume from packing a mixture of old and new parameters.
    previous = {}
    rejected = False

    def reject():
        nonlocal rejected
        rejected = True

    try:
        if owners:
            torch.cuda.synchronize()
            for _, w in repack:
                w.restore()
            torch.cuda.synchronize()
        for _, w in owners:
            w.canonical_depth += 1
            if recover:
                previous[w] = w.failed_updates, w.failed_shards
                w.failed_updates, w.failed_shards = set(), set()
            elif key is not None and key is not _OPAQUE_UPDATE:
                w.failed_updates.discard(key)
        try:
            yield reject
            if recover and any(w.failed_updates or w.failed_shards for _, w in owners):
                raise RuntimeError("BF16 recovery contains an incomplete weight update")
            if owners and key is not None and not rejected:
                # A transfer can report its CUDA error only at synchronization.
                # Wait before committing its version, including during offload.
                torch.cuda.synchronize()
        except BaseException:
            reject()
            raise
        finally:
            for _, w in owners:
                w.canonical_depth -= 1
                if rejected and key is not None:
                    w.failed_updates.add(key)
                    if recover:
                        old_updates, old_shards = previous[w]
                        w.failed_updates.update(old_updates)
                        w.failed_shards.update(old_shards)
    finally:
        if repack:
            # Restore serving eligibility only on a clean commit. Keep failed
            # layers canonical and preserve the updater's original error/status.
            for _, w in repack:
                if not w.active:
                    w.pending_shards = True
            if not rejected or key is None:
                if key is None:
                    torch.cuda.synchronize()
                for owner, w in repack:
                    if w.shards_ready:
                        w.pack(owner)
                torch.cuda.synchronize()


@contextmanager
def canonical_weights(layer=None, *, recover=False):
    """Expose canonical weights. recover=True certifies a complete reload or rollback on successful exit."""
    with _canonical_access(layer, _OPAQUE_UPDATE, recover):
        yield


def close():
    restore_all()
    _layers.clear()
    _inactive.clear()
