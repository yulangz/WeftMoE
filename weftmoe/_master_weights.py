"""FP8 compute weights backed by the optimizer's authoritative FP32 masters."""

from __future__ import annotations

import torch

from .fp8 import _seal_weights, _storage
from .runtime import _extension


class MasterWeights:
    """Keep Parameter metadata/hooks, discard expert BF16 allocations.

    Only complete, locally owned FP32 matrices are accepted. The optimizer
    invokes write() on its compute stream before recycling the FP32 staging
    slot. No CPU BF16 shadow or second quantization is used for publication.
    """

    def __init__(self, adapter, optimizer):
        if torch.distributed.get_world_size(optimizer.pg_collection.expt_dp) != 1:
            raise ValueError("WeftMoE FP32-master weights currently require expert DP=1")
        self.adapter = adapter
        self.optimizer = optimizer
        self.bindings = {}
        self.masters = {}
        self.weights = {}
        self.writer_streams = {}
        self.device = next(iter(adapter._parameters(adapter.layers[0]))).device
        self.offloaded = False
        self.materialized = False
        leaves = [leaf.optimizer for leaf in optimizer.chained_optimizers]
        optimizer.drain()
        owners = {id(unit.param): leaf
                  for leaf in leaves if getattr(leaf, "optimizer_kind", None) == "muon"
                  for unit in getattr(leaf, "units", ())}
        selected = {id(p) for layer in adapter.layers for p in adapter._parameters(layer)}
        expert_linears = {id(linear) for layer in adapter.layers
                          for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2)}
        selected_storages = set()
        for layer in adapter.layers:
            for p in adapter._parameters(layer):
                if id(p) not in owners:
                    raise ValueError("Every FP8 expert must have a locally owned Muon master")
                storage = p.untyped_storage()
                if (p.storage_offset() or not p.is_contiguous() or not storage.resizable()
                        or storage.nbytes() != p.numel() * p.element_size()
                        or storage._cdata in selected_storages):
                    raise ValueError("FP8 masters require separate whole expert allocations")
                selected_storages.add(storage._cdata)
        for model in adapter._models:
            protected = list(torch.nn.Module.buffers(model))
            for name, module in model.named_modules(remove_duplicate=False):
                if (id(module) not in expert_linears or not name.endswith(
                        ("experts.linear_fc1", "experts.linear_fc2"))):
                    protected.extend(module.parameters(recurse=False))
            if any(t.untyped_storage()._cdata in selected_storages for t in protected):
                raise ValueError("FP8 expert storage aliases a registered buffer or dense parameter")
            for p in model.parameters():
                if id(p) not in selected and p.untyped_storage()._cdata in selected_storages:
                    raise ValueError("FP8 expert storage aliases another parameter")
                grad = getattr(p, "main_grad", None)
                if grad is not None and grad.untyped_storage()._cdata in selected_storages:
                    raise ValueError("FP8 expert storage aliases a gradient")
        with torch.no_grad():
            for layer in adapter.layers:
                params = adapter._parameters(layer)
                count = layer.num_local_experts
                shapes = [(count, *params[0].shape), (count, *params[count].shape)]
                row = [_storage(shape, 128, self.device) for shape in shapes]
                col = [_storage((shape[0], shape[2], shape[1]), 128, self.device)
                       for shape in shapes]
                self.weights[layer] = (*row, *col)
                for index, p in enumerate(params):
                    leaf = owners[id(p)]
                    master = leaf.state[p]["master_param"]
                    if master.dtype != torch.float32 or master.numel() != p.numel():
                        raise ValueError("Expected a complete FP32 master matrix")
                    self.masters[id(p)] = master
                    self.bindings[id(p)] = (layer, index // count, index % count, p)
                    p._weftmoe_fp8_storage = p.untyped_storage()
                    p._weftmoe_fp8_writer = self.write
                    self.write(p, master.to(device=self.device).view(p.shape))
                torch.cuda.synchronize(self.device)
                for p in params:
                    p.untyped_storage().resize_(0)
        self.publish()

    def primary(self, parameter):
        layer, kind, expert, _ = self.bindings[id(parameter)]
        return self.weights[layer][kind].matrix(expert)

    @torch.no_grad()
    def write(self, parameter, master):
        if self.offloaded:
            raise RuntimeError("Restore FP8 compute weights before optimizer.step")
        layer, kind, expert, _ = self.bindings[id(parameter)]
        row = self.weights[layer][kind].matrix(expert)
        col = self.weights[layer][kind + 2].matrix(expert)
        stream = torch.cuda.current_stream(self.device)
        _extension().fp8_quantize_weight(master, row.data, row.scales,
                                         col.data, col.scales, stream.cuda_stream)
        master.record_stream(stream)
        self.writer_streams[stream.cuda_stream] = stream

    def publish(self):
        if self.offloaded:
            return
        consumer = torch.cuda.current_stream(self.device)
        for writer in self.writer_streams.values():
            consumer.wait_stream(writer)
        self.writer_streams.clear()
        self.adapter.fp8_weights = {
            layer: (tuple(p._version for p in self.adapter._parameters(layer)),
                    _seal_weights(weights, self.adapter.weight_version, cache_key=id(layer)))
            for layer, weights in self.weights.items()
        }

    @torch.no_grad()
    def offload(self):
        if self.offloaded:
            return
        torch.cuda.synchronize(self.device)
        for weights in self.weights.values():
            for weight in weights:
                for tensor in (weight.data, weight.scales):
                    # The CPU FP32 master is authoritative. Retaining another
                    # full FP8 W/Wt snapshot can exhaust host memory when the
                    # colocated rollout offloads its own model as well.
                    metadata = torch.empty_like(tensor, device="cpu")
                    metadata.untyped_storage().resize_(0)
                    tensor.data = metadata
        self.offloaded = True
        self.adapter.fp8_weights.clear()

    @torch.no_grad()
    def resume(self):
        if not self.offloaded:
            return
        for weights in self.weights.values():
            for weight in weights:
                for tensor in (weight.data, weight.scales):
                    tensor.data = torch.empty_like(tensor, device=self.device)
        self.offloaded = False
        self._rebuild()
        self.publish()

    @torch.no_grad()
    def _rebuild(self):
        for key, (_, _, _, p) in self.bindings.items():
            self.write(p, self.masters[key].to(device=self.device).view(p.shape))

    @torch.no_grad()
    def materialize(self):
        """Checkpoint-only BF16 view derived from FP32; never a saved shadow."""
        self.adapter.suspend()
        self.optimizer.drain()
        for key, (_, _, _, p) in self.bindings.items():
            p.untyped_storage().resize_(p.numel() * p.element_size())
            p.data.copy_(self.masters[key].view(p.shape))
        self.materialized = True

    @torch.no_grad()
    def refresh(self):
        """Rebuild after a checkpoint updates the optimizer's FP32 master values."""
        self.optimizer.drain()
        self.discard_materialized()
        if self.offloaded:
            self.resume()
        else:
            self._rebuild()
        self.adapter.weight_version += 1
        self.publish()

    @torch.no_grad()
    def discard_materialized(self):
        if self.materialized:
            torch.cuda.synchronize(self.device)
            for _, _, _, p in self.bindings.values():
                p.untyped_storage().resize_(0)
            self.materialized = False
