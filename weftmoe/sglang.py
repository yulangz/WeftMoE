"""SGLang expert adapter with warmed CUDA child graphs and no X/AB retention."""

from __future__ import annotations

import os
import functools
import logging
from dataclasses import replace

import torch
import torch.distributed as dist
import triton

from ._sglang_kernels import sanitize_routes, scatter_output, stage_inputs
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
_eager_pending = None
_decode_batch = False
_graph_inputs = {}
_fp8_weights = {}
_fp8_devices = {}
_fp8_offloaded = False
_fp8_needs_update = False
_fp8_initial_pool = None


def _use_fp8():
    return os.environ.get("WEFTMOE_FP8", "0") == "1"


def _prepare_fp8_layer(layer):
    from .fp8 import prepare_fp8_weights
    global _fp8_initial_pool

    if _fp8_needs_update:
        raise RuntimeError("Complete the FP8 weight update before inference")

    if layer.layer_id not in _fp8_weights:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm FP8 expert weights before graph capture")
        # End the temporary allocation pool after the loader's outer TMS
        # region has exited. PyTorch cannot destroy nested pools inside another
        # active use_mem_pool scope. Live BF16 allocations remain valid until
        # their tensors are released below; freed blocks can now be reclaimed.
        _fp8_initial_pool = None
        parameters = (layer.w13_weight, layer.w2_weight)
        # SGLang FusedMoE owns its two complete allocations. Fail before
        # releasing an unexpectedly shared flat buffer belonging to other layers.
        if any(p.storage_offset() != 0 or p.numel() * p.element_size()
               != p.untyped_storage().nbytes() for p in parameters):
            raise ValueError("FP8 SGLang residency requires whole expert weight allocations")
        _fp8_weights[layer.layer_id] = replace(prepare_fp8_weights(
            layer.w13_weight, layer.w2_weight, training=False), _cache_key=id(layer))
        _fp8_devices[layer.layer_id] = layer.w13_weight.device
        torch.cuda.synchronize(layer.w13_weight.device)
        for p in parameters:
            p.untyped_storage().resize_(0)
        torch.cuda.empty_cache()
    return _fp8_weights[layer.layer_id]


@torch.no_grad()
def resume_fp8_weights(*, for_update=False):
    global _fp8_offloaded
    if _fp8_needs_update and not for_update:
        raise RuntimeError("Complete the FP8 weight update before inference")
    if not _fp8_offloaded:
        return
    for key, weights in _fp8_weights.items():
        for tensor in weights._tensors():
            tensor.data = (torch.empty_like(tensor, device=_fp8_devices[key])
                           if _fp8_needs_update else tensor.to(device=_fp8_devices[key]))
    _fp8_offloaded = False
    if not _fp8_needs_update:
        commit_fp8_weights(0)


def commit_fp8_weights(version):
    global _fp8_needs_update
    from .fp8 import _seal_weights
    for key, weights in list(_fp8_weights.items()):
        _fp8_weights[key] = _seal_weights((weights.w1, weights.w2, None, None), version,
                                         cache_key=weights._cache_key)
    _fp8_needs_update = False


def fp8_parameter_map(model):
    """AWEX binds these views directly; BF16 expert Parameters are metadata only."""
    if not _use_fp8():
        return {}
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    resume_fp8_weights(for_update=True)
    result = {}
    for layer in model.modules():
        if isinstance(layer, FusedMoE):
            weights = _fp8_weights.get(layer.layer_id)
            if weights is None:
                weights = _prepare_fp8_layer(layer)
            for parameter, weight in ((layer.w13_weight, weights.w1), (layer.w2_weight, weights.w2)):
                result[id(parameter)] = (weight.data.view(torch.uint8), weight.scales)
    return result


def _prepare_runtime(layer, group, top_k, max_tokens, *, decode_only=False):
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
        decode_only=decode_only,
        fp8_workspace=_use_fp8(),
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


@torch.no_grad()
def _prepare_bf16_memory_budget(model):
    """Make persistent BF16 allocations visible to SGLang's KV profiler."""
    global _decode_batch, _eager_pending
    if _use_fp8():
        return
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    layers = [layer for layer in model.modules() if isinstance(layer, FusedMoE)]
    if not layers:
        return
    previous_decode = _decode_batch
    try:
        # Allocate both transport workspaces before KV/Mamba capacity is
        # chosen. A one-token zero-gate warmup uses
        # the same preparation path as real prefill/decode, without attention
        # state or an extra estimate that can drift from native allocations.
        for decode in (False, True):
            _decode_batch = decode
            for layer in layers:
                device = layer.w13_weight.device
                x = torch.zeros((1, layer.hidden_size), dtype=torch.bfloat16, device=device)
                ids = torch.arange(layer.top_k, dtype=torch.int32, device=device).view(1, -1)
                gates = torch.zeros((1, layer.top_k), dtype=torch.float32, device=device)
                layer(x, StandardTopKOutput(gates, ids, None))
        if _eager_pending is not None:
            _eager_pending.native.drain_eager_inference()
            _eager_pending = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logging.getLogger(__name__).info(
            "Prepared BF16 WeftMoE before KV sizing: %d layers; "
            "PF/DT workspaces included in available-memory measurement",
            len(layers),
        )
    finally:
        _decode_batch = previous_decode


def install():
    # Keep graph allocations outside TMS: their native child-graph addresses
    # stay resident, or are explicitly destroyed by the low-memory policy.
    # Model weights and KV memory still use TMS normally.
    os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"] = "0"
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
    from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )

    global _dp_layout_patched
    if not _dp_layout_patched:
        patch_dp_attention_layout()
        _dp_layout_patched = True
    if getattr(FusedMoE, "_weftmoe_installed", False):
        return

    from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
    original_init_memory_pool = ModelRunnerKVCacheMixin.init_memory_pool

    @functools.wraps(original_init_memory_pool)
    def init_memory_pool(self, *args, **kwargs):
        _prepare_bf16_memory_budget(self.model)
        return original_init_memory_pool(self, *args, **kwargs)

    ModelRunnerKVCacheMixin.init_memory_pool = init_memory_pool

    original_moe_init = FusedMoE.__init__

    @functools.wraps(original_moe_init)
    def moe_init(self, *args, **kwargs):
        global _fp8_initial_pool
        if _use_fp8():
            from sglang.srt.utils.torch_memory_saver_adapter import _memory_saver
            impl = getattr(_memory_saver, "_impl", None)
            if impl is not None and impl._binary_wrapper.cdll.tms_get_interesting_region():
                # These initial BF16 tensors are discarded after quantization.
                # A persistent TMS pool would retain/remap their freed blocks.
                # Keep the ordinary pool until the outer loader region exits;
                # TMS.disable() destroys its nested pool too early on torch2.9.
                # torch2.9's allocation filters select the outer pool first;
                # suspend its filter while entering our ordinary pool. Balance
                # begin/end/release exactly as use_mem_pool does, then restore
                # the loader's filter before its context manager resumes.
                memory = torch.cuda.memory
                device = torch.cuda.current_device()
                outer_id = impl._primary_mem_pool.id
                memory._cuda_endAllocateToPool(device, outer_id)
                memory._cuda_releasePool(device, outer_id)
                impl._binary_wrapper.cdll.tms_set_interesting_region(False)
                try:
                    if _fp8_initial_pool is None:
                        _fp8_initial_pool = torch.cuda.MemPool()
                    with torch.cuda.use_mem_pool(_fp8_initial_pool):
                        original_moe_init(self, *args, **kwargs)
                finally:
                    impl._binary_wrapper.cdll.tms_set_interesting_region(True)
                    memory._cuda_beginAllocateCurrentThreadToPool(device, outer_id)
                return
        original_moe_init(self, *args, **kwargs)

    FusedMoE.__init__ = moe_init

    # Graph-buffer IPC mappings live for the all-reduce communicator's entire
    # lifetime. Use the existing TMS staging path so graph allocations are never
    # exported through long-lived IPC mappings, including on final teardown.
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
        # pay only this Python branch.
        global _eager_pending
        if _eager_pending is not None:
            _eager_pending.native.drain_eager_inference()
            _eager_pending = None
        return original_graph_replay(self, *args, **kwargs)

    CudaGraphRunner.replay = graph_replay

    def forward(layer, hidden_states, topk_output):
        from sglang.srt.distributed import get_tp_group
        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        global _runtime, _decode_runtime, _eager_pending
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
        # DP rows need no X copy. Masked top-k entries still need sanitization:
        # raw -1 expert IDs cannot enter the native dispatch planner. Match the
        # graph staging kernel in one launch without copying activations.
        elif (
            not sharded_rows
            and valid_tokens == local_tokens
            and ids.dtype == torch.int32
            and weights.dtype == torch.float32
            and ids.is_contiguous()
            and weights.is_contiguous()
        ):
            routes = torch.empty_like(ids)
            gates = torch.empty_like(weights)
            sanitize_routes[(local_tokens,)](
                ids, weights, routes, gates, k, triton.next_power_of_2(k))
        else:
            x = hidden_states.new_empty((local_tokens, hidden))
            routes = torch.empty((local_tokens, k), dtype=torch.int32, device=x.device)
            gates = torch.empty((local_tokens, k), dtype=torch.float32, device=x.device)
            stage_inputs[(local_tokens,)](
                hidden_states, topk_output.topk_ids, topk_output.topk_weights,
                x, routes, gates, *hidden_states.stride(),
                *topk_output.topk_ids.stride(), *topk_output.topk_weights.stride(),
                start, valid_tokens, hidden, k,
                triton.next_power_of_2(hidden), triton.next_power_of_2(k))
        inputs = (
            x,
            routes,
            gates,
            layer.w13_weight,
            layer.w2_weight,
        )
        runtime = _runtime
        # Select by collective phase, never by a rank's actual token count.
        # Decode uses the same DT implementation for every captured bucket;
        # dynamic prefill and training retain the tuned PF specialization.
        if capturing or _decode_batch:
            if _decode_runtime is None:
                capacity = max(
                    args.cuda_graph_max_bs or 1, args.max_running_requests or 1
                )
                if not rows_are_dp_sharded:
                    capacity = (capacity + world - 1) // world
                _decode_runtime = _prepare_runtime(
                    layer, group, k, capacity, decode_only=True
                )
            runtime = _decode_runtime
        if _use_fp8():
            prepared = _prepare_fp8_layer(layer)
            if _eager_pending is not None:
                _eager_pending.native.drain_eager_inference()
                _eager_pending = None
            if capturing:
                output = runtime.inference_fp8(layer.layer_id, x, routes, gates, prepared)
            else:
                output = runtime.forward_fp8(
                    x, routes, gates, prepared, synchronize=False,
                    retain_backward_state=False).output
                _eager_pending = runtime
        elif capturing:
            if _eager_pending is not None:
                _eager_pending.native.drain_eager_inference()
                _eager_pending = None
            output = runtime.inference(layer.layer_id, *inputs)
        else:
            if _eager_pending is not None and _eager_pending is not runtime:
                _eager_pending.native.drain_eager_inference()
            output = runtime.inference_eager_async(*inputs, layer=layer.layer_id)
            _eager_pending = runtime
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

    global _rows_are_dp_sharded
    _rows_are_dp_sharded = True

    original_mlp_mode = LayerScatterModes._compute_mlp_mode.__func__

    def mlp_mode(cls, context):
        original = original_mlp_mode(cls, context)
        if (
            context.is_layer_sparse
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
            if os.environ.get("AWEX_META_SERVER_ADDR"):
                from .awex import replicate_qwen35_shared_experts

                replicate_qwen35_shared_experts()

    Qwen2MoeSparseMoeBlock.__init__ = block_init

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
        global _decode_batch
        previous_decode = _decode_batch
        mode = forward_batch.global_forward_mode or forward_batch.forward_mode
        # global_forward_mode is populated only by some SGLang schedules.
        # is_extend_in_batch is the existing DP-wide phase agreement even
        # when local ranks mix decode, prefill and idle work.
        _decode_batch = not forward_batch.is_extend_in_batch and (
            mode.is_decode() or mode.is_idle()
        )
        try:
            return run_lm_forward(
                self, input_ids, positions, forward_batch, input_embeds,
                pp_proxy_tensors, input_deepstack_embeds,
            )
        finally:
            _decode_batch = previous_decode

    def run_lm_forward(
        self, input_ids, positions, forward_batch, input_embeds,
        pp_proxy_tensors, input_deepstack_embeds,
    ):
        if (
            get_global_server_args().enable_dp_attention
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



def _release_prefill():
    global _runtime, _eager_pending
    if _runtime is None and _eager_pending is None:
        return
    torch.cuda.synchronize()
    if _eager_pending is not None:
        _eager_pending.native.drain_eager_inference()
        _eager_pending = None
    if _runtime is not None:
        _runtime.close()
        _runtime = None
    torch.cuda.empty_cache()


def suspend(model_runner, *, discard_fp8=False):
    """Keep decode graphs, unless the low-memory policy is set."""
    global _recapture
    if _use_fp8() or os.environ.get("WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS", "1") == "0":
        # Prepared FP8 versions are immutable. Rebuild plans after AWEX rather
        # than leaving captured graphs pointing at the previous weight version.
        _recapture = getattr(model_runner, "graph_runner", None) is not None
        close(model_runner, keep_fp8=_use_fp8(), discard_fp8=discard_fp8)
    else:
        _release_prefill()


def resume(model_runner):
    global _recapture
    resume_fp8_weights()
    if _recapture:
        from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

        with torch.no_grad(), model_capture_mode():
            _reset_decode_inputs(model_runner.graph_runner)
            model_runner.graph_runner.capture()
        _recapture = False
    # AWEX has completed the update at the same addresses. BF16 graphs read
    # the original weights directly, so no weight refresh or recapture is needed.


def _reset_decode_inputs(runner):
    """Restore capture dummy inputs after live requests and KV-cache release.

    Keep shared buffer addresses, but never reuse request/sequence/Mamba indices
    written by the last decode batch after the request pool has been flushed.
    Matches SGLang DecodeInputBuffers.create().
    """
    buffers = runner.buffers
    for name in (
        "input_ids", "input_embeds", "req_pool_indices", "out_cache_loc",
        "positions", "mrope_positions", "num_token_non_padded",
        "next_token_logits_buffer", "mamba_track_indices", "mamba_track_mask",
        "global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu",
    ):
        tensor = getattr(buffers, name, None)
        if tensor is not None:
            tensor.zero_()
    buffers.seq_lens.fill_(runner.seq_len_fill_value)
    buffers.seq_lens_cpu.fill_(runner.seq_len_fill_value)
    if getattr(buffers, "custom_mask", None) is not None:
        buffers.custom_mask.fill_(True)
    if getattr(buffers, "encoder_lens", None) is not None:
        buffers.encoder_lens.fill_(runner.encoder_len_fill_value)
    if getattr(buffers, "pp_proxy_tensors", None) is not None:
        for tensor in buffers.pp_proxy_tensors.values():
            tensor.zero_()
    info = getattr(buffers, "ngram_embedding_info", None)
    if info is not None:
        info.column_starts.zero_()
        info.out_column_starts.zero_()
        info.req_lens.fill_(1)
        info.out_req_lens.fill_(1)


def close(model_runner, *, keep_fp8=False, discard_fp8=False):
    """Final teardown: discard outer graphs before freeing native addresses."""
    global _decode_runtime, _fp8_offloaded, _fp8_needs_update
    torch.cuda.synchronize()
    runner = getattr(model_runner, "graph_runner", None)
    if runner is not None:
        runner.graphs.clear()
        runner.output_buffers.clear()
    _release_prefill()
    if _decode_runtime is not None:
        _decode_runtime.close()
        _decode_runtime = None
    _graph_inputs.clear()
    if keep_fp8 and not _fp8_offloaded:
        for weights in _fp8_weights.values():
            for tensor in weights._tensors():
                host = torch.empty_like(tensor, device="cpu", pin_memory=not discard_fp8)
                if discard_fp8:
                    host.untyped_storage().resize_(0)
                else:
                    host.copy_(tensor)
                tensor.data = host
        _fp8_offloaded = True
        _fp8_needs_update = discard_fp8
    elif keep_fp8 and discard_fp8:
        # A preceding KV-only offload may already have saved a CPU snapshot.
        # Once a full weight update is scheduled, that snapshot is obsolete.
        for weights in _fp8_weights.values():
            for tensor in weights._tensors():
                tensor.untyped_storage().resize_(0)
        _fp8_needs_update = True
    elif not keep_fp8:
        _fp8_weights.clear()
        _fp8_devices.clear()
        _fp8_offloaded = False
        _fp8_needs_update = False
    torch.cuda.empty_cache()
