"""SGLang BF16 expert adapter with warmed CUDA child graphs and no X/AB retention."""

from __future__ import annotations

import os
import functools

import torch
import torch.distributed as dist
import triton

from ._sglang_kernels import scatter_output, stage_inputs
from .runtime import TensorRuntime

_runtime = None
_decode_runtime = None
_recapture = False
_stream = None
_dp_layout_patched = False
# True once patch_dp_attention_layout() converted the layout so the MoE block
# sees DP-sharded rows. With the stock layout the layer communicator gathers
# every rank's rows, so the transport must shard them itself.
_rows_are_dp_sharded = False
_layout_active = False
_eager_pending = False
_graph_inputs = {}


def _prepare_runtime(layer, group, top_k, max_tokens):
    global _stream
    if _stream is None:
        _stream = torch.cuda.Stream()
    return TensorRuntime(
        group=group,
        experts=layer.num_experts,
        top_k=top_k,
        hidden=layer.hidden_size,
        intermediate=layer.w2_weight.shape[-1],
        max_tokens=max_tokens,
        max_rows=max_tokens * top_k * 2 + layer.num_local_experts * 128,
        training=False,
        stream=_stream,
    )


def participate_idle_layers(model):
    """Join one transport handshake per MoE layer with a single dummy row.

    A DP rank that owns no tokens still has to take part in the transport's
    per-layer dispatch/return protocol, otherwise its peers block. Running the
    layer stack would feed empty tensors to kernels that assume non-empty
    batches, so the rank instead dispatches one dummy row per MoE layer; the
    result is discarded and the logits path afterwards is shared with every
    rank.
    """
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    for layer in model.layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None:
            continue
        top_k = experts.top_k
        device = experts.w13_weight.device
        hidden = experts.hidden_size
        x = torch.zeros((1, hidden), dtype=torch.bfloat16, device=device)
        ids = torch.arange(top_k, dtype=torch.int32, device=device).reshape(1, top_k)
        weights = torch.zeros((1, top_k), dtype=torch.float32, device=device)
        topk = StandardTopKOutput(weights, ids.clone(), None)
        # Call the installed adapter forward (bound to this layer).
        experts.forward(x, topk)

# Temporary diagnostics: synchronize (and report) after every transport call so
# a corrupted CUDA context is attributed to the layer that caused it.
_DEBUG_SYNC = os.environ.get("WEFTMOE_DEBUG_SYNC") == "1"
_DEBUG_CALLS = os.environ.get("WEFTMOE_DEBUG_CALLS") == "1"
_SYNC_EAGER = os.environ.get("WEFTMOE_SGLANG_SYNC_EAGER") == "1"
_call_counts = {}
_cudart = None


def _cuda_last_error():
    """Return cudaGetLastError() without going through torch's error checks."""
    global _cudart
    if _cudart is None:
        import ctypes

        _cudart = ctypes.CDLL("libcudart.so")
        _cudart.cudaGetLastError.restype = ctypes.c_int
    return _cudart.cudaGetLastError()


def _sync_result():
    """Synchronize and report the outcome without raising."""
    try:
        torch.cuda.synchronize()
        return "ok"
    except Exception as error:  # noqa: BLE001 - diagnostic path
        return "FAILED:%s" % error


def install():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    # This adapter destroys and recaptures graphs to release native IPC/GIN
    # addresses. TMS graph remapping assumes graphs stay alive and cannot own
    # the same graph allocations. KV and model memory still use TMS normally.
    os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "0"
    global _dp_layout_patched
    if not _dp_layout_patched:
        patch_dp_attention_layout()
        _dp_layout_patched = True
    if getattr(FusedMoE, "_weftmoe_installed", False):
        return

    # Graph-buffer IPC mappings live for the all-reduce communicator's entire
    # lifetime. Its existing TMS staging path avoids retaining destroyed graph
    # allocations, even though WeftMoE itself does not use TMS graph remapping.
    # Retained so a diagnostic build can bypass the transport.
    original_forward = FusedMoE.forward

    original_ar_init = CustomAllreduce.__init__

    @functools.wraps(original_ar_init)
    def ar_init(self, *args, **kwargs):
        original_ar_init(self, *args, **kwargs)
        self.tms_cudagraph = True

    CustomAllreduce.__init__ = ar_init
    original_capture = GroupCoordinator.graph_capture

    @functools.wraps(original_capture)
    def graph_capture(self, graph_capture_context=None, stream=None):
        if graph_capture_context is None and stream is None:
            if not hasattr(self, "_weftmoe_capture_stream"):
                self._weftmoe_capture_stream = self.device_module.Stream()
            stream = self._weftmoe_capture_stream
        return original_capture(self, graph_capture_context, stream)

    # Reuse the capture stream so cuBLAS does not allocate another per-stream
    # workspace on every residency transition.
    GroupCoordinator.graph_capture = graph_capture

    original_graph_replay = CudaGraphRunner.replay

    @functools.wraps(original_graph_replay)
    def graph_replay(self, *args, **kwargs):
        # A captured decode graph reuses the same serialized native workspace
        # without invoking FusedMoE.forward in Python.  Finalize the last eager
        # generation once at the eager->graph boundary, before graph replay can
        # overwrite its status/diagnostic records.  Subsequent decode replays
        # pay only this Python boolean branch.
        global _eager_pending
        if _eager_pending and _runtime is not None:
            _runtime.native.drain_eager_inference()
            _eager_pending = False
        return original_graph_replay(self, *args, **kwargs)

    CudaGraphRunner.replay = graph_replay

    def forward(layer, hidden_states, topk_output):
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        global _runtime, _decode_runtime, _eager_pending
        if os.environ.get("WEFTMOE_NULL_MOE") == "1":
            # Diagnostic: keep the layout conversion but remove the transport
            # entirely, so a crash can be attributed to one of the two.
            return torch.zeros_like(hidden_states)
        if os.environ.get("WEFTMOE_BYPASS_TRANSPORT") == "1":
            return original_forward(layer, hidden_states, topk_output)
        if _DEBUG_CALLS:
            rank_now = dist.get_rank() if dist.is_initialized() else -1
            count = _call_counts.get(rank_now, 0)
            if count < 6:
                _call_counts[rank_now] = count + 1
                print(
                    "[WEFTMOE-CALL] rank=%s n=%d layer=%s tokens=%d dp=%s"
                    % (
                        rank_now,
                        count + 1,
                        getattr(layer, "layer_id", "?"),
                        hidden_states.shape[0],
                        get_global_server_args().enable_dp_attention,
                    ),
                    flush=True,
                )
        group = get_tp_group().device_group
        world = dist.get_world_size(group)
        if layer.moe_ep_size != world or layer.moe_tp_size != 1:
            raise ValueError("WeftMoE SGLang requires EP=TP and expert TP=1")
        if (
            layer.w13_weight.dtype != torch.bfloat16
            or layer.w2_weight.dtype != torch.bfloat16
        ):
            raise ValueError("WeftMoE SGLang requires unquantized BF16 experts")
        tokens, hidden = hidden_states.shape
        if tokens == 0:
            return torch.empty_like(hidden_states)
        args = get_global_server_args()
        capturing = get_is_capture_mode()
        rows_are_dp_sharded = _rows_are_dp_sharded and args.enable_dp_attention
        if _runtime is None:
            max_tokens = max(
                args.chunked_prefill_size or 8192,
                args.max_prefill_tokens or 16384,
                args.cuda_graph_max_bs or 1,
            )
            if not rows_are_dp_sharded:
                max_tokens = (max_tokens + world - 1) // world
            _runtime = _prepare_runtime(
                layer, group, topk_output.topk_ids.shape[-1], max_tokens
            )
        rank = dist.get_rank(group)
        # Two layouts reach this point:
        #  * replicated tokens (TP attention, or DP attention with the layer
        #    communicator's gather): shard rows across ranks here, and let the
        #    model's tensor-parallel reduce/scatter reassemble every row;
        #  * DP-sharded tokens: these rows already belong to this rank alone,
        #    the transport returns their complete expert results, and
        #    patch_dp_attention_layout() removed the redundant collectives.
        sharded_rows = not rows_are_dp_sharded
        if sharded_rows:
            local_tokens = (tokens + world - 1) // world
            start = rank * local_tokens
            valid_tokens = max(0, min(local_tokens, tokens - start))
            if capturing or valid_tokens == local_tokens:
                x = hidden_states[start : start + local_tokens]
            else:
                x = hidden_states.new_zeros((local_tokens, hidden))
                x[:valid_tokens].copy_(hidden_states[start : start + valid_tokens])
        else:
            local_tokens = tokens
            start = 0
            valid_tokens = tokens
            x = hidden_states
        k = topk_output.topk_ids.shape[-1]
        ids = topk_output.topk_ids[start : start + valid_tokens]
        weights = topk_output.topk_weights[start : start + valid_tokens]
        if capturing:
            # The child graph binds stable addresses. Fuse padding, invalid-ID
            # sanitization and the three staging copies into one outer node.
            if local_tokens not in _graph_inputs:
                _graph_inputs[local_tokens] = (
                    hidden_states.new_empty((local_tokens, hidden)),
                    torch.empty((local_tokens, k), dtype=torch.int32, device=x.device),
                    torch.empty((local_tokens, k), dtype=torch.float32, device=x.device),
                )
            x, routes, gates = _graph_inputs[local_tokens]
            stage_inputs[(local_tokens,)](
                hidden_states, topk_output.topk_ids, topk_output.topk_weights,
                x, routes, gates, *hidden_states.stride(),
                *topk_output.topk_ids.stride(), *topk_output.topk_weights.stride(),
                start, valid_tokens, hidden, k,
                triton.next_power_of_2(hidden), triton.next_power_of_2(k),
            )
        # In the DP-sharded layout every row is a real SGLang row and the
        # standard top-k path already returns exactly the int32/float32,
        # contiguous tensors consumed by TensorRuntime.  Rebuilding them used
        # to allocate two tensors and launch arange/clone/compare/where/copy
        # kernels in every MoE layer.  Besides being pure overhead in prefill,
        # those kernels sat outside the warmed child graph.  Keep the padded
        # construction only for the replicated-input compatibility layout,
        # where this rank may own fewer than ``local_tokens`` real rows.
        elif (
            not sharded_rows
            and os.environ.get("WEFTMOE_SGLANG_FORCE_ROUTE_COPY") != "1"
            and valid_tokens == local_tokens
            and ids.dtype == torch.int32
            and weights.dtype == torch.float32
            and ids.is_contiguous()
            and weights.is_contiguous()
        ):
            routes = ids
            gates = weights
        else:
            routes = (
                torch.arange(k, dtype=torch.int32, device=x.device)
                .expand(local_tokens, k)
                .clone()
            )
            gates = torch.zeros(
                (local_tokens, k), dtype=torch.float32, device=x.device
            )
            valid = ids >= 0
            routes[:valid_tokens].copy_(
                torch.where(valid, ids, routes[:valid_tokens])
            )
            gates[:valid_tokens].copy_(weights.float() * valid)
        inputs = (
            x,
            routes,
            gates,
            layer.w13_weight,
            layer.w2_weight,
        )
        if capturing:
            if _eager_pending:
                _runtime.native.drain_eager_inference()
                _eager_pending = False
            runtime = _runtime
            # SGLang captures collective, equally padded decode buckets. Eager
            # DP sources may have unequal lengths and must all use the general
            # pool, never choose a transport independently from their row count.
            if local_tokens <= 64:
                if _decode_runtime is None:
                    _decode_runtime = _prepare_runtime(layer, group, k, 64)
                runtime = _decode_runtime
            output = runtime.inference(layer.layer_id, *inputs)
        elif _SYNC_EAGER:
            output = _runtime.inference_eager(*inputs)
        else:
            output = _runtime.inference_eager_async(*inputs)
            _eager_pending = True
        if _DEBUG_SYNC and not get_is_capture_mode():
            print(
                "[WEFTMOE] rank=%d layer=%s tokens=%d local=%d sync=%s cuda_err=%d"
                % (
                    rank,
                    getattr(layer, "layer_id", "?"),
                    tokens,
                    local_tokens,
                    _sync_result(),
                    _cuda_last_error(),
                ),
                flush=True,
            )
        if not sharded_rows:
            # Every row in this rank's batch is complete already.
            return output
        # Qwen2MoeSparseMoeBlock subsequently sums routed + TP-sharded shared
        # expert contributions. Contribute each routed row from its owner and
        # let that existing reduction reconstruct it, avoiding all-gather/divide.
        if not layer.reduce_results:
            if capturing:
                result = hidden_states.new_empty((tokens, hidden))
                scatter_output[(triton.cdiv(tokens * hidden, 4096),)](
                    output, result, start, valid_tokens, hidden, tokens * hidden, 4096,
                )
                return result
            result = hidden_states.new_zeros((tokens, hidden))
            result[start : start + valid_tokens].copy_(output[:valid_tokens])
            return result
        gathered = hidden_states.new_empty((local_tokens * world, hidden))
        dist.all_gather_into_tensor(gathered, output, group=group)
        return gathered[:tokens]

    FusedMoE.forward = forward
    FusedMoE._weftmoe_installed = True


def patch_dp_attention_layout():
    """Let the WeftMoE transport own dispatch/combine under DP attention.

    SGLang's DP-attention layout gathers every DP rank's tokens into each
    layer's MoE block and reduce-scatters the result back, because a rank only
    computes the experts it owns and its shared expert is tensor-parallel
    sharded. WeftMoE instead routes each rank's own rows to the experts that
    own them and returns the results to the row owner, so both collectives are
    redundant: the block can consume and produce the DP-sharded layout
    directly. This is the layout SGLang itself selects when a token dispatcher
    (``moe_a2a_backend``) is configured, so only the layout selection and the
    two reduce sites that assume the gathered layout need to change.
    """
    from sglang.srt.layers.communicator import LayerScatterModes, ScatterMode
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock
    from sglang.srt.server_args import get_global_server_args

    # Debug switch: "all" applies the full conversion, "layout" only moves
    # dispatch/combine out of the communicator, "dispatch" only removes the two
    # reduce sites, "off" disables the conversion (used to bisect failures).
    layout_mode = os.environ.get("WEFTMOE_DP_LAYOUT", "all")
    if layout_mode == "off":
        return
    apply_layout = layout_mode in ("all", "layout")
    global _rows_are_dp_sharded, _layout_active
    _rows_are_dp_sharded = apply_layout
    _layout_active = apply_layout
    apply_dispatch = layout_mode in ("all", "dispatch")

    original_mlp_mode = LayerScatterModes._compute_mlp_mode.__func__

    def mlp_mode(cls, context):
        original = original_mlp_mode(cls, context)
        if (
            apply_layout
            and context.is_layer_sparse
            and original == ScatterMode.FULL
            and get_global_server_args().enable_dp_attention
        ):
            return ScatterMode.SCATTERED
        return original

    LayerScatterModes._compute_mlp_mode = classmethod(mlp_mode)

    original_init = Qwen2MoeSparseMoeBlock.__init__

    def block_init(self, layer_id, config, quant_config=None, alt_stream=None,
                   prefix="", is_nextn=False):
        original_init(
            self,
            layer_id=layer_id,
            config=config,
            quant_config=quant_config,
            alt_stream=alt_stream,
            prefix=prefix,
            is_nextn=is_nextn,
        )
        if not get_global_server_args().enable_dp_attention:
            return
        if not apply_dispatch:
            return
        # The routed path already returns complete DP-local rows, so the
        # block's own tensor-parallel reduce would sum unrelated tokens.
        self.tp_size = 1
        if getattr(self, "shared_expert", None) is not None:
            # A tensor-parallel shared expert would need the same cross-rank
            # sum. Replicate it (the layout SGLang uses with a token
            # dispatcher) so the whole dense path stays DP-local.
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=prefix + ".shared_expert",
                tp_rank=0,
                tp_size=1,
            )

    Qwen2MoeSparseMoeBlock.__init__ = block_init

    if not apply_layout:
        return

    # A DP rank without local tokens still has to join one transport handshake
    # per MoE layer: the dispatch/return protocol involves every EP rank, and
    # the kernels downstream assume non-empty local batches (in the gathered
    # layout the per-layer DP gather fills such a rank with its peers' rows).
    #
    # Instead of fabricating a padded SGLang batch (which fights several batch
    # invariants), the idle rank skips the layer stack entirely and calls the
    # transport once per MoE layer with a single dummy row, then falls through
    # to the logits path (which every rank enters anyway).
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    original_lm_forward = Qwen3_5ForCausalLM.forward

    def lm_forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        pp_proxy_tensors=None,
        input_deepstack_embeds=None,
    ):
        if (
            _layout_active
            and get_global_server_args().enable_dp_attention
            and forward_batch is not None
            and forward_batch.forward_mode.is_idle()
            and (input_ids is None or input_ids.numel() == 0)
        ):
            participate_idle_layers(self)
            weight = self.embed_tokens.weight
            return torch.empty(
                (0, weight.shape[1]), dtype=weight.dtype, device=weight.device
            )
        return original_lm_forward(
            self,
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
            input_deepstack_embeds=input_deepstack_embeds,
        )

    Qwen3_5ForCausalLM.forward = lm_forward

    # is_allocation_symmetric() -> is_dp_max_padding() reads this class state,
    # which SGLang only assigns inside set_dp_buffer_len(). The first forward of
    # a process (the embedding) can run before any batch went through the DP
    # sync, so give the wrapper the same defaults it would have before the
    # first padded batch.
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper

    for name, value in (
        ("_global_dp_buffer_len", 0),
        ("_local_dp_buffer_len", 0),
        ("_dp_max_padding", False),
        ("_global_num_tokens", None),
    ):
        if not hasattr(_DpGatheredBufferWrapper, name):
            setattr(_DpGatheredBufferWrapper, name, value)



def suspend(model_runner):
    """Discard outer graphs before releasing any of their native addresses."""
    global _runtime, _decode_runtime, _recapture, _eager_pending
    if _runtime is None:
        return
    torch.cuda.synchronize()
    runner = getattr(model_runner, "graph_runner", None)
    if runner is not None:
        runner.graphs.clear()
        runner.output_buffers.clear()
        _recapture = True
    if _decode_runtime is not None:
        _decode_runtime.close()
        _decode_runtime = None
    _runtime.close()
    _runtime = None
    _eager_pending = False
    _graph_inputs.clear()
    torch.cuda.empty_cache()


def resume(model_runner):
    global _recapture
    if _recapture:
        # Rebuild against restored addresses once per residency phase. Native
        # communication contexts remain persistent across every layer/token call.
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

        with torch.no_grad(), model_capture_mode():
            model_runner.graph_runner.capture()
        _recapture = False
