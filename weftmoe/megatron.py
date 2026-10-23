"""Megatron routed-MoE adapter; router, shared experts and DDP remain Megatron-owned."""

from __future__ import annotations

from types import MethodType

import torch

from .runtime import TensorRuntime


class _Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, manager, layer, x, routes, gates, *parameters):
        packed = manager.pack(parameters)
        runtime = manager.get_runtime(layer)
        ctx.manager = manager
        # Retain the original tensors and let autograd detect a premature
        # optimizer update before backward repacks their values.
        ctx.save_for_backward(*parameters)
        ctx.runtime = runtime
        ctx.tape = runtime.forward_async(x, routes, gates, *packed)
        return ctx.tape.output

    @staticmethod
    def backward(ctx, dy):
        # A single fixed weight workspace serves all layers. Forward is
        # complete before reuse; restore this layer's weights before backward
        # consumes the retained lease. Optimizer steps happen after all tapes.
        ctx.manager.pack(ctx.saved_tensors)
        dx, dg, dw1, dw2 = ctx.runtime.backward_async(ctx.tape, dy)
        ctx.tape = None
        ctx.runtime = None
        ctx.manager = None
        # Returning each parameter gradient invokes Megatron DDP's normal hooks.
        return (None, None, dx, None, dg, *dw1.unbind(), *dw2.unbind())


class MegatronAdapter:
    def __init__(self, models, *, max_tokens):
        from megatron.core.transformer.moe.moe_layer import MoELayer

        self.max_tokens = max_tokens
        self.runtime = None
        # Communication workspaces survive train/rollout residency switches,
        # like Megatron's DeepEP buffer. close() releases them at engine teardown.
        self.session = None
        self.packed = {}
        self.layers = []
        for model in models:
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
            parameters = [
                getattr(layer.experts.linear_fc1, f"weight{i}")
                for i in range(layer.num_local_experts)
            ]
            parameters += [
                getattr(layer.experts.linear_fc2, f"weight{i}")
                for i in range(layer.num_local_experts)
            ]
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
                packed = manager.pack(parameters)
                output = manager.get_runtime(layer).forward_async(
                    *inputs, *packed, retain_backward_state=False
                ).output
            output = output.view_as(hidden_states)
            if shared is not None:
                output = output + shared
            return output, None

        return forward

    def suspend(self):
        """Finish training uses before offload, retaining communication buffers."""
        if self.runtime is None:
            return
        torch.cuda.synchronize()
        self.runtime.drain()
        if self.runtime.native.usage()["notify_slots_in_use"]:
            raise RuntimeError(
                "Cannot offload WeftMoE while autograd still owns saved X/AB"
            )

    def close(self):
        """Release buffers and the session when the training engine is destroyed."""
        self.suspend()
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        self.packed.clear()
        self.session = None
        torch.cuda.empty_cache()
