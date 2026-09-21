"""SGLang adapter for the BF16 MegaMoE kernel in DeepGEMM-bk.

The adapter deliberately owns the same DP-sharded MoE layout as the native
WeftMoE adapter.  Each rank publishes its local rows to one DeepGEMM symmetric
buffer and receives the complete routed result for those rows.

DeepGEMM chooses a compile-time wave schedule from ``y.size(0)``.  A mixed
Extend/Decode batch can have very different local row counts across ranks, but
all ranks must execute the same wave protocol.  We therefore pad the launch
shape to the maximum DP-local row count for the current forward and mask the
extra routes with expert id -1.  The returned tensor is sliced back to the
rank's real row count.
"""

from __future__ import annotations

import functools
import os

import torch
import torch.distributed as dist

_buffer = None
_deep_gemm = None
_max_tokens = 0
_prepared = False


def _expert_layers(model_runner):
    language_model = getattr(model_runner.model, "model", model_runner.model)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("DeepGEMM SGLang adapter could not find language-model layers")
    result = []
    for decoder_layer in layers:
        experts = getattr(getattr(decoder_layer, "mlp", None), "experts", None)
        if experts is not None:
            result.append(experts)
    if not result:
        raise ValueError("DeepGEMM SGLang adapter found no FusedMoE layers")
    return result


def _warm_kernel(layer, sizes):
    """JIT/load every decode configuration before CUDA graph capture begins."""
    rank = dist.get_rank(_buffer.group)
    top_k = layer.top_k
    for tokens in sizes:
        if tokens <= 0 or tokens > _max_tokens:
            continue
        _buffer.x[:tokens].zero_()
        routes = torch.arange(
            tokens * top_k, dtype=torch.int64, device=_buffer.x.device
        ).reshape(tokens, top_k)
        routes.add_(rank * 17).remainder_(layer.num_experts)
        _buffer.topk_idx[:tokens].copy_(routes)
        _buffer.topk_weights[:tokens].fill_(1.0 / top_k)
        output = torch.empty(
            (tokens, layer.hidden_size),
            dtype=torch.bfloat16,
            device=_buffer.x.device,
        )
        _deep_gemm.bf16_mega_moe(
            output,
            layer._deepgemm_l1_weight,
            layer._deepgemm_l2_weight,
            _buffer,
        )


def _prepare(model_runner):
    global _buffer, _deep_gemm, _max_tokens, _prepared
    if _prepared:
        # SGLang calls init_device_graphs() again after an in-place RL weight
        # update.  Preserve the transformed tensor addresses referenced by any
        # graphs while refreshing their contents from the newly loaded model
        # weights.  L2 aliases w2_weight and is already current.
        for layer in _expert_layers(model_runner):
            l1_weight, _ = _deep_gemm.transform_weights_for_mega_moe(
                layer.w13_weight, layer.w2_weight
            )
            layer._deepgemm_l1_weight.copy_(l1_weight)
        torch.cuda.synchronize()
        dist.barrier(group=_buffer.group)
        return

    import deep_gemm
    from sglang.srt.distributed import get_tp_group

    args = model_runner.server_args
    if not args.enable_dp_attention:
        raise ValueError("DeepGEMM SGLang adapter currently requires DP attention")
    if args.enable_eplb or args.init_expert_location != "trivial":
        raise ValueError(
            "DeepGEMM SGLang adapter requires contiguous trivial expert placement"
        )

    group = get_tp_group().device_group
    world = dist.get_world_size(group)
    layers = _expert_layers(model_runner)
    first = layers[0]
    if first.moe_ep_size != world or first.moe_tp_size != 1:
        raise ValueError("DeepGEMM SGLang adapter requires EP=TP and expert TP=1")
    if first.w13_weight.dtype != torch.bfloat16 or first.w2_weight.dtype != torch.bfloat16:
        raise ValueError("DeepGEMM SGLang adapter requires unquantized BF16 experts")
    if first.num_fused_shared_experts:
        raise ValueError("DeepGEMM SGLang adapter does not support fused shared experts")

    configured_max = int(os.environ.get("DEEPGEMM_SGLANG_MAX_TOKENS", "0"))
    _max_tokens = configured_max or max(
        int(args.chunked_prefill_size or 0),
        int(args.cuda_graph_max_bs or 0),
        1,
    )
    _deep_gemm = deep_gemm
    _buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=first.num_experts,
        num_max_tokens_per_rank=_max_tokens,
        num_topk=first.top_k,
        hidden=first.hidden_size,
        intermediate_hidden=first.w2_weight.shape[-1],
        mma_type="bf16xbf16",
    )

    for layer in layers:
        if (
            layer.num_experts != first.num_experts
            or layer.top_k != first.top_k
            or layer.w13_weight.shape != first.w13_weight.shape
            or layer.w2_weight.shape != first.w2_weight.shape
        ):
            raise ValueError("DeepGEMM requires one common MoE shape across layers")
        l1_weight, l2_weight = deep_gemm.transform_weights_for_mega_moe(
            layer.w13_weight, layer.w2_weight
        )
        # Keep the transformed storage alive for eager launches and graph replay.
        layer._deepgemm_l1_weight = l1_weight
        layer._deepgemm_l2_weight = l2_weight

    # The JIT wrapper performs module loading and creates TMA descriptors on its
    # first call.  Do that before capture; subsequent layer calls share the same
    # compiled shape/config and differ only in weight pointers.
    capture_sizes = sorted(set(int(value) for value in args.cuda_graph_bs))
    _warm_kernel(first, capture_sizes + [_max_tokens])
    torch.cuda.synchronize()
    dist.barrier(group=group)
    _prepared = True
    if dist.get_rank(group) == 0:
        gib = _buffer.buffer.numel() / (1024**3)
        print(
            "[DeepGEMM-SGLang] prepared %d layers, max_tokens=%d, symm_buffer=%.2f GiB"
            % (len(layers), _max_tokens, gib),
            flush=True,
        )


def _launch_tokens(local_tokens: int) -> int:
    """Return one common launch size for the current DP forward."""
    from sglang.srt.layers.dp_attention import _DpGatheredBufferWrapper
    from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

    global_tokens = _DpGatheredBufferWrapper.get_dp_global_num_tokens()
    if global_tokens is not None:
        return max(local_tokens, max(int(value) for value in global_tokens))
    # During graph capture every rank captures the same batch size.  Outside
    # capture, missing DP metadata would allow ranks to select different wave
    # configurations, so fail instead of risking a transport deadlock.
    if get_is_capture_mode():
        return local_tokens
    if dist.get_world_size(_buffer.group) != 1:
        raise RuntimeError("DeepGEMM launch is missing SGLang DP token metadata")
    return local_tokens


def install():
    """Install the adapter in one SGLang scheduler process."""
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.model_executor.model_runner import ModelRunner

    if getattr(FusedMoE, "_deepgemm_megamoe_installed", False):
        return

    # Reuse the already-tested SGLang layout conversion.  This function only
    # patches layout/shared-expert/idle-rank handling; it does not install the
    # WeftMoE FusedMoE.forward implementation.
    from .sglang import patch_dp_attention_layout

    patch_dp_attention_layout()

    original_init_device_graphs = ModelRunner.init_device_graphs

    @functools.wraps(original_init_device_graphs)
    def init_device_graphs(model_runner, *args, **kwargs):
        _prepare(model_runner)
        return original_init_device_graphs(model_runner, *args, **kwargs)

    ModelRunner.init_device_graphs = init_device_graphs

    def forward(layer, hidden_states, topk_output):
        if not _prepared:
            raise RuntimeError("DeepGEMM adapter was invoked before model preparation")
        if hidden_states.ndim != 2 or hidden_states.shape[1] != layer.hidden_size:
            raise ValueError("Unexpected hidden-state shape for DeepGEMM MegaMoE")
        tokens = hidden_states.shape[0]
        if tokens == 0:
            return torch.empty_like(hidden_states)
        launch_tokens = _launch_tokens(tokens)
        if launch_tokens > _max_tokens:
            raise ValueError(
                "DeepGEMM launch has %d rows but capacity is %d"
                % (launch_tokens, _max_tokens)
            )

        _buffer.x[:tokens].copy_(hidden_states)
        _buffer.topk_idx[:tokens].copy_(topk_output.topk_ids)
        _buffer.topk_weights[:tokens].copy_(topk_output.topk_weights)
        if launch_tokens > tokens:
            _buffer.topk_idx[tokens:launch_tokens].fill_(-1)
            _buffer.topk_weights[tokens:launch_tokens].zero_()

        output = hidden_states.new_empty((launch_tokens, layer.hidden_size))
        _deep_gemm.bf16_mega_moe(
            output,
            layer._deepgemm_l1_weight,
            layer._deepgemm_l2_weight,
            _buffer,
        )
        return output[:tokens]

    FusedMoE.forward = forward
    FusedMoE._deepgemm_megamoe_installed = True
