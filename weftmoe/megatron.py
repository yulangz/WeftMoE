"""Megatron routed-MoE adapter; router, shared experts and DDP remain Megatron-owned."""

from __future__ import annotations

from types import MethodType
import os

import torch

from .runtime import TensorRuntime


class _Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, manager, layer, x, routes, gates, *parameters):
        runtime = manager.get_runtime(layer)
        ctx.manager = manager
        ctx.weight_version = manager.weight_version
        # Retain the original tensors and let autograd detect a premature
        # optimizer update before backward repacks their values.
        ctx.save_for_backward(*parameters)
        ctx.runtime = runtime
        ctx.tape = manager.forward_experts(layer, x, routes, gates, parameters)
        return ctx.tape.output

    @staticmethod
    def backward(ctx, dy):
        # A single fixed weight workspace serves all layers. Forward is
        # complete before reuse; restore this layer's weights before backward
        # consumes the retained lease. Optimizer steps happen after all tapes.
        if ctx.manager.fp8 and ctx.weight_version != ctx.manager.weight_version:
            raise RuntimeError("Expert weights changed before FP8 backward")
        parameters = ctx.saved_tensors
        if not ctx.manager.fp8:
            ctx.manager.pack(parameters)
        dx, dg, dw1, dw2 = ctx.runtime.backward_async(ctx.tape, dy)
        ctx.tape = None
        ctx.runtime = None
        ctx.manager = None
        # Returning each parameter gradient invokes Megatron DDP's normal hooks.
        return (None, None, dx, None, dg, *dw1.unbind(), *dw2.unbind())


class MegatronAdapter:
    def __init__(self, models, *, max_tokens, fp8=None):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        self.max_tokens = max_tokens
        self.fp8 = os.environ.get("WEFTMOE_FP8", "0") == "1" if fp8 is None else fp8
        self.fp8_weights = {}
        self.master_weights = None
        self._models = tuple(models)
        self.weight_version = 0
        self.runtime = None
        # Keep the NCCL session across residency switches; rebuild the larger
        # compute/communication workspaces when training next needs them.
        self.session = None
        self.packed = {}
        self.layers = []
        for model in self._models:
            if getattr(
                getattr(model, "ddp_config", None), "overlap_param_gather", False
            ):
                raise ValueError("WeftMoE requires overlap_param_gather=False")
            for layer in model.modules():
                if not isinstance(layer, MoELayer):
                    continue
                config = layer.config
                if (
                    config.moe_latent_size
                    or config.moe_shared_expert_overlap
                    or config.add_bias_linear
                    or config.expert_tensor_parallel_size != 1
                    or layer.moe_layer_recompute
                    or not config.gated_linear_unit
                    or config.activation_func is not torch.nn.functional.silu
                    or config.moe_apply_probs_on_input
                    or config.activation_func_clamp_value is not None
                    or config.glu_linear_offset != 0
                ):
                    raise ValueError(
                        "WeftMoE requires ETP1, bias-free SwiGLU, separate shared experts, "
                        "and no partial MoE checkpoint (full-layer checkpoint is supported)"
                    )
                self.layers.append(layer)
                layer.forward = MethodType(self._make_forward(), layer)
        if not self.layers:
            raise ValueError("No Megatron MoE layers found")

    def prepare(self):
        """Validate and allocate the communication context before rollout starts."""
        self.get_runtime(self.layers[0])

    def bind_fp32_masters(self, optimizer):
        if self.fp8:
            from ._master_weights import MasterWeights
            self.master_weights = MasterWeights(self, optimizer)

    def pack(self, parameters):
        count = len(parameters) // 2
        key = (count, parameters[0].shape, parameters[count].shape)
        packed = self.packed.get(key)
        if packed is None:
            packed = (
                parameters[0].new_empty((count, *parameters[0].shape)),
                parameters[count].new_empty((count, *parameters[count].shape)),
            )
            self.packed[key] = packed
        torch.stack(parameters[:count], out=packed[0])
        torch.stack(parameters[count:], out=packed[1])
        return packed

    def release_bf16_weights(self):
        """Restore FP8 compute storage; BF16 experts remain metadata only."""
        if self.master_weights is not None:
            self.master_weights.discard_materialized()
            self.master_weights.resume()

    def restore_bf16_weights(self):
        """Materialize checkpoint BF16 weights on demand from FP32 masters."""
        if self.master_weights is not None:
            self.master_weights.materialize()

    @staticmethod
    def _parameters(layer):
        return [getattr(linear, f"weight{i}")
                for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2)
                for i in range(layer.num_local_experts)]

    def weights_updated(self):
        """Invalidate after optimizer/checkpoint writes, including .data copies."""
        self.weight_version += 1
        if self.master_weights is not None:
            self.master_weights.publish()
        else:
            self.fp8_weights.clear()

    def forward_experts(self, layer, x, routes, gates, parameters, *, retain=True):
        runtime = self.get_runtime(layer)
        if not self.fp8:
            return runtime.forward_async(x, routes, gates, *self.pack(parameters),
                                         retain_backward_state=retain)
        from .fp8 import prepare_fp8_weights

        versions = tuple(p._version for p in parameters)
        entry = self.fp8_weights.get(layer)
        if entry is None or entry[0] != versions:
            if self.master_weights is not None:
                raise RuntimeError("FP32-master FP8 weights are not ready for compute")
            weights = prepare_fp8_weights(*self.pack(parameters), training=True,
                                          version=self.weight_version)
            entry = (versions, weights)
            self.fp8_weights[layer] = entry
        return runtime.forward_fp8(x, routes, gates, entry[1], synchronize=False,
                                   retain_backward_state=retain,
                                   weight_grad_dtype=torch.bfloat16)

    def get_runtime(self, layer):
        if self.runtime is None:
            cfg = layer.config
            self.runtime = TensorRuntime(
                group=layer.ep_group,
                experts=cfg.num_moe_experts,
                top_k=cfg.moe_router_topk,
                hidden=cfg.hidden_size,
                intermediate=cfg.moe_ffn_hidden_size,
                max_tokens=self.max_tokens,
                max_rows=self.max_tokens * cfg.moe_router_topk * 2
                + layer.num_local_experts * 128,
                training=True,
                max_saved=len(self.layers) + 8,
                session=self.session,
                fp8_workspace=self.fp8,
            )
            self.session = self.runtime.native.session()
        return self.runtime

    def _make_forward(self):
        manager = self

        def forward(layer, hidden_states, intermediate_tensors=None, padding_mask=None):
            if intermediate_tensors is not None:
                raise ValueError(
                    "WeftMoE replaces the whole routed path; partial MoE execution is unsupported"
                )
            if padding_mask is not None:
                padding_mask = padding_mask.transpose(0, 1).bool()
            shared = layer.shared_experts_compute(hidden_states)
            probabilities, _ = layer.route(hidden_states, padding_mask)
            gates, routes = probabilities.topk(layer.config.moe_router_topk, dim=-1)
            parameters = manager._parameters(layer)
            inputs = (
                hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous(),
                routes.int().contiguous(),
                gates.float().contiguous(),
            )
            # Megatron's full checkpoint runs the first pass under no_grad;
            # only its grad-enabled replay needs a tape. Check outside
            # autograd.Function.forward, which always runs with grad disabled.
            if torch.is_grad_enabled():
                output = _Routed.apply(manager, layer, *inputs, *parameters)
            else:
                output = manager.forward_experts(layer, *inputs, parameters,
                                                 retain=False).output
            output = output.view_as(hidden_states)
            if shared is not None:
                output = output + shared
            return output, None

        return forward

    def suspend(self):
        """Release idle training workspaces while rollout owns the GPUs."""
        if self.runtime is not None:
            torch.cuda.synchronize()
            self.runtime.drain()
            if self.runtime.native.usage()["notify_slots_in_use"]:
                raise RuntimeError(
                    "Cannot offload WeftMoE while autograd still owns saved X/AB"
                )
            self.runtime.close()
            self.runtime = None
        if self.master_weights is not None:
            self.master_weights.discard_materialized()
            self.master_weights.offload()
        else:
            self.weights_updated()
        self.packed.clear()

    def close(self):
        """Release buffers and the session when the training engine is destroyed."""
        self.suspend()
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.packed.clear()
        self.fp8_weights.clear()
        self.session = None
        if self.master_weights is not None:
            for _, _, _, parameter in self.master_weights.bindings.values():
                del parameter._weftmoe_fp8_writer
                del parameter._weftmoe_fp8_storage
            self.master_weights = None
        torch.cuda.empty_cache()
