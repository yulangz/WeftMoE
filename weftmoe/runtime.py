"""Tensor execution with one persistent native workspace per explicit EP group.

This module is separate from the benchmark API. Preparation is collective;
all ranks must call layers in the same order. Expert weights use BF16
[local_experts, 2 * intermediate, hidden] and [local_experts, hidden, intermediate].
"""

from __future__ import annotations

import logging
import os
import socket

import torch
import torch.distributed as dist


def _extension():
    from .jit import helpers
    return helpers()


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
        input_layout: str = "dp",
        interconnect: str | None = None,
        decode_only: bool = False,
        fp8_workspace: bool = False,
        session=None,
        batch_invariant: bool = True,
        allow_unvalidated_shape: bool = False,
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
        self.decode_only = decode_only
        self.fp8_workspace = fp8_workspace
        self.max_tokens = max_tokens
        self.stream = stream
        if self.stream is None:
            self.stream = (
                torch.cuda.current_stream() if training else torch.cuda.Stream()
            )
        from . import jit
        key = dict(gpu=torch.cuda.get_device_name(),
                   interconnect=interconnect or ("nvlink" if self.nodes == 1 else "nvlink+gin"),
                   dtype="fp8" if fp8_workspace else "bfloat16", hidden=hidden,
                   intermediate=intermediate, experts=experts, top_k=top_k,
                   nodes=self.nodes, gpus_per_node=self.world // self.nodes,
                   input_layout=input_layout,
                   mode="decode" if decode_only else "training" if training else "inference")
        # Agree before compilation or native collectives. A bad shape/build on
        # one node must report a collective error, not strand peers in bootstrap.
        requests = [None] * self.world
        request = dict(key=key, max_tokens=max_tokens, batch_invariant=batch_invariant,
                       fp8_workspace=fp8_workspace, decode_only=decode_only,
                       allow_unvalidated_shape=allow_unvalidated_shape)
        dist.all_gather_object(requests, request, group=group)
        if any(r != request for r in requests):
            raise ValueError("EP ranks requested incompatible WeftMoE configurations")
        native, tuning, entry = None, None, None
        decision = dict(error=None)
        try:
            explicit = os.environ.get("WEFTMOE_TENSOR_EXTENSION")
            if explicit:
                native = jit.load(explicit)
                if os.environ.get("WEFTMOE_AUTOTUNE_TABLE"):
                    from .tuning import select_profile
                    legacy_key = {k: v for k, v in key.items() if k != "mode"}
                    legacy_key.update(dtype="bfloat16", max_tokens=max_tokens, training=training)
                    tuning = select_profile(os.environ["WEFTMOE_AUTOTUNE_TABLE"], legacy_key,
                                            native.forward_tuning_id())
                identity = native.forward_tuning_id()
            else:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Prepare WeftMoE before CUDA Graph capture")
                entry = jit.prepare(key, max_tokens, batch_invariant=batch_invariant,
                                    allow_unvalidated_shape=allow_unvalidated_shape)
                tuning = entry["tuning"]
                native = jit.load(entry["extension"])
                identity = entry["header_digest"]
            if tuple(native.forward_shape()) != (hidden, intermediate):
                raise ValueError("Native library shape does not match the requested model")
            # A session is owned by its native module. Different bucket binaries
            # bootstrap separately; never pass a foreign C++ session across DSOs.
            if session is not None and not isinstance(session, native.Session):
                logging.getLogger(__name__).warning(
                    "Cannot reuse a WeftMoE session from another native library; "
                    "creating a separate NCCL/GIN session and workspace")
                session = None
            decision.update(identity=identity, tuning=tuning, reuse_session=session is not None)
        except Exception as error:
            decision["error"] = f"{type(error).__name__}: {error}"
        decisions = [None] * self.world
        dist.all_gather_object(decisions, decision, group=group)
        if any(d["error"] for d in decisions):
            raise RuntimeError(f"collective WeftMoE preparation failed: {decisions}")
        if any(d != decision for d in decisions):
            raise ValueError("EP ranks selected incompatible WeftMoE binaries")
        self.compilation = entry
        self._module = native  # keep its DSO alive for every tape/graph
        bootstrap = [b""]
        if session is None:
            bootstrap = [native.bootstrap_id() if self.rank == 0 else None]
            dist.broadcast_object_list(
                bootstrap, src=dist.get_global_rank(group, 0), group=group
            )
        arguments = (
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
        if fp8_workspace:
            tuning = dict(tuning or {}, fp8_workspace=True)
        if decode_only:
            tuning = dict(tuning or {}, decode_only=True, window_tokens=max_tokens)
        if session is not None:
            self.native = native.Runtime(*arguments, tuning or {}, session)
        else:
            self.native = (
                native.Runtime(*arguments, tuning) if tuning
                else native.Runtime(*arguments)
            )

    def _check(self, x, routes, gates, w1, w2):
        if self.fp8_workspace:
            raise ValueError("Compact FP8 workspace requires forward_fp8 or moe(..., fp8_weights=...)")
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

    def _source_input(self, value):
        from .fp8 import FP8Tensor

        if isinstance(value, FP8Tensor):
            if value.block_rows != 1 or len(value.shape) != 2:
                raise ValueError("FP8 source requires rowwise 1x128 quantization")
            data = value.data
            if data.data_ptr() % 16:
                raise ValueError("FP8 source data must be 16-byte aligned")
            if tuple(data.shape) != value.shape:
                raise ValueError("FP8 source must have an unpadded hidden dimension")
            buffers = (data, value.scales)
        else:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
                raise ValueError("Source must be BF16 or FP8Tensor with explicit row scales")
            data, buffers = value, (value,)
        if (data.ndim != 2 or data.shape[1] != self.hidden
                or data.shape[0] > self.max_tokens or not data.is_cuda
                or not data.is_contiguous() or data.device != self.stream.device):
            raise ValueError("Source shape/device/layout does not match runtime")
        return data, buffers

    def forward_fp8(self, x, routes, gates, weights, *,
                    synchronize=True, retain_backward_state=True,
                    weight_grad_dtype=torch.float32):
        """PF/DT forward with BF16 or prequantized rowwise FP8Tensor input.

        Prepare weights with weftmoe.fp8.prepare_fp8_weights before this call.
        FP8Tensor inputs require block_rows=1 and finite positive power-of-two
        FP32 scales and 16-byte-aligned data. Data/scales are copied unchanged
        into source packets. External scale values are a caller contract, not checked on the hot
        path (which would require device synchronization). Invalid scales can
        produce nonfinite outputs or additional rounding error in backward.
        Only BF16 inputs are quantized. Backward accepts either input format
        through the usual backward API and still returns BF16 dX.
        """
        from .fp8 import PreparedFP8Weights

        if weight_grad_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("weight_grad_dtype must be torch.bfloat16 or torch.float32")
        if not isinstance(weights, PreparedFP8Weights):
            raise TypeError("weights must be PreparedFP8Weights")
        if self.training and torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Runtime must be invoked on its preparation stream")
        source, source_buffers = self._source_input(x)
        for value, dtype, shape in (
            (routes, torch.int32, (source.shape[0], self.top_k)),
            (gates, torch.float32, (source.shape[0], self.top_k)),
        ):
            if (value.shape != shape or value.dtype != dtype or not value.is_cuda
                    or value.device != source.device or not value.is_contiguous()):
                raise ValueError(f"Expected contiguous CUDA {dtype} of shape {shape}")
        if (weights.w1.shape != (self.experts // self.world, 2 * self.intermediate, self.hidden)
                or weights.w2.shape != (self.experts // self.world, self.hidden, self.intermediate)
                or weights.w1.data.device != source.device):
            raise ValueError("Prepared FP8 weights do not match runtime shape/device")
        # Native calls run on the preparation stream. Include preparation of
        # weights and caller-produced inputs in its dependency chain.
        caller = torch.cuda.current_stream(source.device)
        if caller == self.stream:
            return self.native.forward_fp8(x, routes, gates, weights,
                                           synchronize, retain_backward_state,
                                           weight_grad_dtype == torch.float32)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            for value in (*source_buffers, routes, gates):
                value.record_stream(self.stream)
            tape = self.native.forward_fp8(x, routes, gates, weights,
                                          synchronize, retain_backward_state,
                                          weight_grad_dtype == torch.float32)
        caller.wait_stream(self.stream)
        return tape

    def backward(self, tape, dy):
        """Consume BF16 dY, or a rowwise FP8Tensor for an FP8 tape."""
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Backward must use the preparation stream")
        dy = dy.contiguous() if isinstance(dy, torch.Tensor) else dy
        self._source_input(dy)
        return self.native.backward(tape, dy)

    def forward_async(self, x, routes, gates, w1, w2, *, retain_backward_state=True):
        """Enqueue FWD, optionally omitting X/AB for checkpoint's no-grad pass."""
        self._check(x, routes, gates, w1, w2)
        return self.native.forward_async(
            x, routes, gates, w1, w2, retain_backward_state
        )

    def backward_async(self, tape, dy):
        """Return stream-ordered gradients; reclaim completed calls without waiting."""
        if torch.cuda.current_stream() != self.stream:
            raise RuntimeError("Backward must use the preparation stream")
        dy = dy.contiguous() if isinstance(dy, torch.Tensor) else dy
        self._source_input(dy)
        return self.native.backward_async(tape, dy)

    def drain(self):
        """Finish the pending training call before inspecting or releasing leases."""
        self.native.drain_training()

    def inference(self, layer, x, routes, gates, w1, w2):
        self._check(x, routes, gates, w1, w2)
        return self.native.graph_forward(
            layer, x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream
        )

    def inference_fp8(self, layer, x, routes, gates, weights):
        """Replay a prepared FP8 inference graph, with stable input/scale addresses.

        Warm each layer/token bucket before outer capture. Replacing prepared
        weights requires invalidate_inference_layer and recapture. X may be
        BF16 or rowwise FP8Tensor; source preparation stays inside the DT kernel.
        """
        from .fp8 import PreparedFP8Weights
        if self.training:
            raise ValueError("inference_fp8 requires training=False")
        source, _ = self._source_input(x)
        if not isinstance(weights, PreparedFP8Weights):
            raise TypeError("weights must be PreparedFP8Weights")
        if (weights.w1.shape != (self.experts // self.world, 2 * self.intermediate, self.hidden)
                or weights.w2.shape != (self.experts // self.world, self.hidden, self.intermediate)
                or weights.w1.data.device != source.device):
            raise ValueError("Prepared FP8 weights do not match runtime")
        for value, dtype in ((routes, torch.int32), (gates, torch.float32)):
            if (value.shape != (source.shape[0], self.top_k) or value.dtype != dtype
                    or value.device != source.device or not value.is_contiguous()):
                raise ValueError("Invalid inference routes/gates")
        # A warmed native plan already owns completed weight preparation.
        # Outer capture must not add a dependency on the preparation stream.
        if torch.cuda.is_current_stream_capturing():
            weights._validate_version()
        else:
            weights.wait(training=False)
        return self.native.graph_forward(layer, x, routes, gates,
            weights.w1.data, weights.w2.data, torch.cuda.current_stream().cuda_stream, weights)

    def inference_eager(self, x, routes, gates, w1, w2, *, layer=None):
        """Run changing inputs/routes; an optional layer caches weight descriptors.

        Invalidate before replacing weight addresses. Complete in-place updates
        before submitting inference; BF16 descriptors read the original tensors.
        """
        self._check(x, routes, gates, w1, w2)
        return self.native.eager_inference(
            x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream,
            *(() if layer is None else (layer,))
        )

    def invalidate_inference_layer(self, layer):
        """Release all token-bucket plans before replacing a layer's weights.

        Call on every EP rank outside capture, after discarding any outer CUDA
        graphs containing this layer. This waits for in-flight device work.
        Then update weights, warm the layer again, and recapture outer graphs.
        BF16 updates at unchanged addresses do not require invalidation; the
        caller must complete all writes before inference resumes.
        """
        self.native.invalidate_inference_layer(
            layer, torch.cuda.current_stream().cuda_stream
        )

    def inference_eager_async(self, x, routes, gates, w1, w2, *, layer=None):
        """Defer completion; optional layer caches weight descriptors and owners."""
        self._check(x, routes, gates, w1, w2)
        return self.native.eager_inference_async(
            x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream,
            *(() if layer is None else (layer,))
        )

    def close(self):
        """Collectively release native CUDA/IPC/GIN storage at a phase boundary."""
        if self.native is None:
            return
        torch.cuda.synchronize()
        self.drain()
        self.native.drain_eager_inference()
        if self.training and self.native.usage()["notify_slots_in_use"]:
            raise RuntimeError("Live backward tapes prevent closing the runtime")
        dist.barrier(group=self.group)
        self.native.retire_peer_mappings()
        # No exporter may free its storage while another rank still has an
        # IPC import. This boundary also covers inference graph workspaces.
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


class _MoEFP8(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, x, routes, gates, w1, w2, weights):
        if (w1.shape != weights.w1.shape or w2.shape != weights.w2.shape
                or w1.dtype not in (torch.bfloat16, torch.float32)
                or w2.dtype not in (torch.bfloat16, torch.float32)
                or w1.device != x.device or w2.device != x.device):
            raise ValueError("FP8 autograd requires matching BF16/FP32 parameter tensors")
        ctx.runtime = runtime
        ctx.save_for_backward(w1, w2)
        weights.validate_parameters(w1, w2)
        grad_dtype = torch.bfloat16 if w1.dtype == w2.dtype == torch.bfloat16 else torch.float32
        ctx.tape = runtime.forward_fp8(x, routes, gates, weights, weight_grad_dtype=grad_dtype)
        return ctx.tape.output

    @staticmethod
    def backward(ctx, dy):
        w1, w2 = ctx.saved_tensors  # Detect optimizer updates before backward.
        dx, dg, dw1, dw2 = ctx.runtime.backward(ctx.tape, dy)
        ctx.tape = None
        ctx.runtime = None
        return None, dx, None, dg, dw1.to(w1.dtype), dw2.to(w2.dtype), None


def moe(runtime, x, routes, gates, w1, w2, *, fp8_weights=None):
    """Apply the fused operation, preserving router and expert autograd edges."""
    if torch.is_grad_enabled() and not runtime.training:
        raise ValueError("Autograd requires a runtime prepared with training=True")
    if fp8_weights is not None:
        return _MoEFP8.apply(runtime, x, routes, gates, w1, w2, fp8_weights)
    return _MoE.apply(runtime, x, routes, gates, w1, w2)
