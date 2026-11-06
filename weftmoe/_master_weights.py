"""FP8 compute weights backed by the optimizer's authoritative FP32 masters."""

from __future__ import annotations

from functools import partial
from types import MethodType

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
        self.adapter = adapter
        self.optimizer = optimizer
        self.bindings = {}
        self.masters = {}
        self.weights = {}
        self.writer_streams = {}
        self.dirty = False
        self.device = next(iter(adapter._parameters(adapter.layers[0]))).device
        self.offloaded = False
        self.materialized = False
        self.leaves = tuple(getattr(optimizer, "chained_optimizers", (optimizer,)))
        self._drain()
        owners = {}
        self.adamw_writers = {}
        for wrapper in self.leaves:
            leaf = wrapper.optimizer
            if getattr(leaf, "optimizer_kind", None) == "muon":
                if torch.distributed.get_world_size(optimizer.pg_collection.expt_dp) != 1:
                    raise ValueError("FP8 master weights require expert DP=1")
                owners.update((id(unit.param), (leaf, unit.param)) for unit in getattr(leaf, "units", ()))
            elif callable(getattr(leaf, "set_master_writers", None)):
                for parameter, (group, index) in getattr(wrapper, "model_param_group_index_map", {}).items():
                    region = wrapper._get_model_param_range_map(parameter)["param"]
                    if region.start == 0 and region.end == parameter.numel():
                        shard = leaf.param_groups[group]["params"][index]
                        owners[id(parameter)] = (leaf, shard)
        selected = {id(p) for layer in adapter.layers for p in adapter._parameters(layer)}
        expert_linears = {id(linear) for layer in adapter.layers
                          for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2)}
        self.buffers = []
        self.bucket_groups = []
        flat_storages = set()
        for model in adapter._models:
            for buf in getattr(model, "expert_parallel_buffers", ()):
                if not any(id(p) in selected for p in buf.params):
                    continue
                if buf.param_data is None:
                    continue
                if not all(id(p) in selected for p in buf.params):
                    raise ValueError("FP8 master buffer must contain only selected experts")
                if buf.data_parallel_world_size != 1:
                    raise ValueError("FP8 master weights require expert DP=1")
                self.buffers.append(buf)
                flat_storages.add(buf.param_data.untyped_storage()._cdata)
            for group in getattr(model, "expert_parallel_bucket_groups", ()):
                if not group.ddp_config.use_distributed_optimizer:
                    continue
                if any(id(p) in selected for p in group.params):
                    if not all(id(p) in selected for p in group.params):
                        raise ValueError("FP8 master bucket must contain only selected experts")
                    if group.intra_distributed_optimizer_instance_size != 1:
                        raise ValueError("FP8 master buckets require expert DP=1")
                    if group.param_gather_handle is not None:
                        raise ValueError("Finish parameter gathers before binding FP8 masters")
                    self.bucket_groups.append((group, group.__dict__.get("start_param_sync")))
        self.storages = {}
        for layer in adapter.layers:
            for p in adapter._parameters(layer):
                if id(p) not in owners:
                    raise ValueError("Every FP8 expert must have a complete staged AdamW or Muon master")
                leaf, shard = owners[id(p)]
                master = leaf.state[shard]["master_param"]
                if master.dtype != torch.float32 or master.numel() != p.numel():
                    raise ValueError("Expected a complete FP32 master matrix")
                self.masters[id(p)] = master
                if callable(getattr(leaf, "set_master_writers", None)):
                    if p.numel() > leaf.staged_config.bucket_numel:
                        raise ValueError("FP8 expert matrix exceeds the AdamW staging slot size")
                    self.adamw_writers.setdefault(leaf, {})[shard] = partial(self.write, p)
                storage = p.untyped_storage()
                if not storage.resizable() or not p.is_contiguous():
                    raise ValueError("FP8 masters require contiguous, resizable expert storage")
                if storage._cdata not in flat_storages and (
                        p.storage_offset() or storage.nbytes() != p.numel() * p.element_size()
                        or storage._cdata in self.storages):
                    raise ValueError("FP8 masters require whole expert allocations or expert-only DDP buffers")
                self.storages[storage._cdata] = (storage, storage.nbytes())
        selected_storages = set(self.storages)
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
                    master = self.masters[id(p)]
                    self.bindings[id(p)] = (layer, index // count, index % count, p)
                    p._weftmoe_fp8_storage = p.untyped_storage()
                    p._weftmoe_fp8_writer = self.write
                    self.write(p, master.to(device=self.device).view(p.shape))
                torch.cuda.synchronize(self.device)
                for p in params:
                    if p.untyped_storage()._cdata not in flat_storages:
                        p.untyped_storage().resize_(0)
        for leaf, writers in self.adamw_writers.items():
            leaf.set_master_writers(writers)
        for buf in self.buffers:
            buf._weftmoe_fp8_master_managed = True
        for group, _ in self.bucket_groups:
            group.start_param_sync = MethodType(self._skip_param_sync, group)
        for storage, _ in self.storages.values():
            storage.resize_(0)
        self.publish()

    def _drain(self):
        drain = getattr(self.optimizer, "drain", None)
        if callable(drain):
            drain()
        else:
            for leaf in self.leaves:
                leaf.optimizer.drain()

    @staticmethod
    def _skip_param_sync(group, force_sync=False):
        # These buckets have expert DP=1. FP8 publication waits on the writer
        # streams; gathering their absent BF16 buffers would be invalid.
        assert group.param_gather_handle is None
        group.param_gather_dispatched = True

    def detach(self):
        """Restore optimizer writeback/sync after materializing BF16 on close."""
        for leaf in self.adamw_writers:
            leaf.set_master_writers({})
        for group, original in self.bucket_groups:
            if original is None:
                del group.start_param_sync
            else:
                group.start_param_sync = original
        for buf in self.buffers:
            del buf._weftmoe_fp8_master_managed
        for _, _, _, parameter in self.bindings.values():
            del parameter._weftmoe_fp8_writer
            del parameter._weftmoe_fp8_storage

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
        if not self.dirty:
            self.adapter.weight_version += 1
            self.dirty = True
        # Native writes bypass PyTorch's version counter. Invalidate retained
        # PreparedFP8Weights too, even when a caller bypasses the adapter.
        torch.autograd.graph.increment_version((row.data, row.scales, col.data, col.scales))
        stream = torch.cuda.current_stream(self.device)
        _extension().fp8_quantize_weight(master.view(parameter.shape), row.data, row.scales,
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
        self.dirty = False

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
        self._drain()
        for storage, size in self.storages.values():
            storage.resize_(size)
            torch.empty(0, dtype=torch.uint8, device=self.device).set_(
                storage, 0, (size,), (1,)).zero_()
        for key, (_, _, _, p) in self.bindings.items():
            p.data.copy_(self.masters[key].view(p.shape))
        self.materialized = True

    @torch.no_grad()
    def refresh(self):
        """Rebuild after a checkpoint updates the optimizer's FP32 master values."""
        self._drain()
        self.discard_materialized()
        if self.offloaded:
            self.resume()
        else:
            self._rebuild()
            self.publish()

    @torch.no_grad()
    def discard_materialized(self):
        if self.materialized:
            torch.cuda.synchronize(self.device)
            for storage, _ in self.storages.values():
                storage.resize_(0)
            self.materialized = False
