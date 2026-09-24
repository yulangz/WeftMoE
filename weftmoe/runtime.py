"""Tensor execution with one persistent native workspace per explicit EP group.

This module is separate from the benchmark API. Preparation is collective;
all ranks must call layers in the same order. Expert weights use BF16
[local_experts, 2 * intermediate, hidden] and [local_experts, hidden, intermediate].
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import socket

import torch
import torch.distributed as dist


def _extension():
    import sys

    if "weftmoe._runtime" in sys.modules:
        return sys.modules["weftmoe._runtime"]
    path = Path(os.environ["WEFTMOE_TENSOR_EXTENSION"]).resolve()
    spec = importlib.util.spec_from_file_location("weftmoe._runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[spec.name] = module
    return module


class TensorRuntime:
    def __init__(
        self,
        *,
        group,
        experts: int,
        top_k: int,
        hidden: int,
        intermediate: int,
        max_tokens: int,
        max_rows: int,
        training: bool,
        max_saved: int = 64,
        stream=None,
    ):
        if group is None:
            raise ValueError("Pass the explicit expert process group")
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        hosts = [None] * self.world
        dist.all_gather_object(hosts, socket.gethostname(), group=group)
        ordered_hosts = list(dict.fromkeys(hosts))
        self.nodes = len(ordered_hosts)
        expected = [
            host for host in ordered_hosts for _ in range(self.world // self.nodes)
        ]
        if hosts != expected:
            raise ValueError(
                "WeftMoE requires equal-size, contiguous node-local EP teams"
            )
        self.experts, self.top_k = experts, top_k
        self.hidden, self.intermediate = hidden, intermediate
        self.training = training
        self.max_tokens = max_tokens
        self.stream = stream
        if self.stream is None:
            self.stream = (
                torch.cuda.current_stream() if training else torch.cuda.Stream()
            )
        native = _extension()
        bootstrap = [native.bootstrap_id() if self.rank == 0 else None]
        dist.broadcast_object_list(
            bootstrap, src=dist.get_global_rank(group, 0), group=group
        )
        self.native = native.Runtime(
            bootstrap[0],
            self.rank,
            self.world,
            self.nodes,
            torch.cuda.current_device(),
            self.stream.cuda_stream,
            self.stream,
            experts,
            top_k,
            hidden,
            intermediate,
            max_tokens,
            max_rows,
            training,
            max_saved if training else 1,
        )

    def _check(self, x, routes, gates, w1, w2):
        if self.training and torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Runtime must be invoked on its preparation stream")
        if x.ndim != 2 or x.shape[1] != self.hidden or x.shape[0] > self.max_tokens:
            raise ValueError("Activation shape exceeds prepared capacity")
        shapes = (
            (x.shape[0], self.top_k),
            (x.shape[0], self.top_k),
            (self.experts // self.world, 2 * self.intermediate, self.hidden),
            (self.experts // self.world, self.hidden, self.intermediate),
        )
        for tensor, shape in zip((routes, gates, w1, w2), shapes):
            if tensor.shape != shape:
                raise ValueError(f"Expected {shape}, got {tuple(tensor.shape)}")
        for tensor, dtype in zip(
            (x, routes, gates, w1, w2),
            (
                torch.bfloat16,
                torch.int32,
                torch.float32,
                torch.bfloat16,
                torch.bfloat16,
            ),
        ):
            if (
                tensor.dtype != dtype
                or tensor.device != x.device
                or not tensor.is_cuda
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"Expected contiguous CUDA {dtype} on {x.device}")

    def forward(self, x, routes, gates, w1, w2):
        self._check(x, routes, gates, w1, w2)
        return self.native.forward(x, routes, gates, w1, w2)

    def backward(self, tape, dy):
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Backward must use the preparation stream")
        return self.native.backward(tape, dy.contiguous())

    def inference(self, layer, x, routes, gates, w1, w2):
        self._check(x, routes, gates, w1, w2)
        return self.native.graph_forward(
            layer, x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream
        )

    def inference_eager(self, x, routes, gates, w1, w2):
        self._check(x, routes, gates, w1, w2)
        return self.native.eager_inference(
            x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream
        )

    def invalidate_inference_layer(self, layer):
        """Release all token-bucket plans before replacing a layer's weights.

        Call on every EP rank outside capture, after discarding any outer CUDA
        graphs containing this layer. This waits for in-flight device work.
        Then update weights, warm the layer again, and recapture outer graphs.
        Graph weights are immutable between these boundaries, including in-place
        updates: pointer equality does not refresh the packed FC1 weight copy.
        """
        self.native.invalidate_inference_layer(
            layer, torch.cuda.current_stream().cuda_stream
        )

    def inference_eager_async(self, x, routes, gates, w1, w2):
        """Enqueue inference and defer host completion until the next call."""
        self._check(x, routes, gates, w1, w2)
        return self.native.eager_inference_async(
            x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream
        )

    def close(self):
        """Collectively release native CUDA/IPC/GIN storage at a phase boundary."""
        if self.native is None:
            return
        torch.cuda.synchronize()
        self.native.drain_eager_inference()
        if self.training and self.native.usage()["notify_slots_in_use"]:
            raise RuntimeError("Live backward tapes prevent closing the runtime")
        dist.barrier(group=self.group)
        self.native = None


class _MoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, x, routes, gates, w1, w2):
        ctx.runtime = runtime
        ctx.tape = runtime.forward(x, routes, gates, w1, w2)
        return ctx.tape.output

    @staticmethod
    def backward(ctx, dy):
        dx, dg, dw1, dw2 = ctx.runtime.backward(ctx.tape, dy)
        ctx.tape = None
        ctx.runtime = None
        return None, dx, None, dg, dw1, dw2


def moe(runtime, x, routes, gates, w1, w2):
    """Apply the fused operation, preserving router and expert autograd edges."""
    if torch.is_grad_enabled() and not runtime.training:
        raise ValueError("Autograd requires a runtime prepared with training=True")
    return _MoE.apply(runtime, x, routes, gates, w1, w2)
