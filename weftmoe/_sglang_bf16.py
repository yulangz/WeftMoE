"""Single-copy BF16 residency for SGLang's serialized serving and weight-update phases."""

from contextlib import contextmanager
from weakref import WeakKeyDictionary

import torch

from .blackwell import _bf16_weight_views, _validate_expert_weights, _weight_tile_shape

_layers = WeakKeyDictionary()


class _ExpertWeights:
    def __init__(self, layer):
        self.storage = (layer.w13_weight.detach(), layer.w2_weight.detach())
        _validate_expert_weights(*self.storage)
        self.tiles = tuple(p.view(_weight_tile_shape(*p.shape, 64)) for p in self.storage)
        self.active = False

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
        # Materialize both sources before writing either destination. Graphs and
        # TMS/AWEX aliases retain the original allocation, even if a loader replaced .data.
        sources = [view.contiguous() for view in _bf16_weight_views(*params)]
        for p, buf, tile, src in zip(params, self.storage, self.tiles, sources, strict=True):
            tile.copy_(src.view_as(tile))
            if p.data_ptr() != buf.data_ptr():
                p.data = buf
        self.active = True

    @torch.inference_mode()
    def restore(self):
        if self.active:
            copies = [tile.clone() for tile in self.tiles]
            for view, copy in zip(_bf16_weight_views(*self.storage), copies, strict=True):
                view.copy_(copy.view(view.shape))
            self.active = False


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
        weights.pack(layer)
    return weights.tiles


def _save_state(layer, state, prefix, metadata):
    weights = _layers.get(layer)
    if weights is not None and weights.active:
        state[prefix + "w13_weight"], state[prefix + "w2_weight"] = weights.canonical_copy()


def _before_load(layer, *args):
    weights = _layers.get(layer)
    if weights is not None and weights.active:
        torch.cuda.synchronize()
        weights.restore()
        layer._weftmoe_repack_after_load = True


def _after_load(layer, incompatible_keys):
    if getattr(layer, "_weftmoe_repack_after_load", False):
        serving_weights(layer)
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
    active = [(owner, w) for owner, w in candidates if w is not None and w.active]
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
