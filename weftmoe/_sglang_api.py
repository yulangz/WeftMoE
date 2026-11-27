"""SGLang worker hooks before and after the 0.5.20 executor split."""

from functools import cache
from types import SimpleNamespace


def _workspace_tokens(schedule, spec, graph_bs, *, decode):
    """Bound MoE input rows before allocating either worker workspace."""
    requests = schedule.max_running_requests or 1
    draft_topk = getattr(spec, "speculative_eagle_topk", None) or 1
    width = max(
        getattr(spec, "speculative_num_draft_tokens", None) or 1,
        getattr(spec, "max_speculative_num_draft_tokens", None) or 1,
        (getattr(spec, "speculative_num_steps", None) or 0) + 1,
        draft_topk,
    )
    if decode:
        # Capture can run any speculative phase here. Eager decode only
        # expands the batch for the draft model's top-k branches.
        return max(graph_bs * width, requests * draft_topk)
    return max(
        max(graph_bs, requests) * width,
        schedule.chunked_prefill_size or 8192,
        schedule.max_prefill_tokens or 16384,
    )


def _legacy_workspace_tokens(args, *, decode):
    graph_bs = 0 if args.disable_cuda_graph else max(
        args.cuda_graph_max_bs or 1, max(args.cuda_graph_bs or (), default=1)
    )
    return _workspace_tokens(args, args, graph_bs, decode=decode)


def _split_workspace_tokens(context, *, decode):
    graph = context.get_exec().graph.cuda_graph_config
    graph_bs = 0
    if graph is not None and getattr(graph.decode, "backend", None) != "disabled":
        graph_bs = max(graph.decode.max_bs or 1, max(getattr(graph.decode, "bs", None) or (), default=1))
    get_spec = getattr(context, "get_spec", None)
    spec = get_spec() if get_spec is not None else context.get_server_args()
    return _workspace_tokens(context.get_schedule(), spec, graph_bs, decode=decode)


@cache
def worker_api():
    try:
        from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner, get_is_capture_mode, model_capture_mode
    except ModuleNotFoundError as exc:
        if exc.name != "sglang.srt.model_executor.cuda_graph_runner":
            raise
    else:
        from sglang.srt.model_executor.model_runner import ModelRunner
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
        from sglang.srt.server_args import get_global_server_args

        return SimpleNamespace(
            prefill_capture=None, model_runner=ModelRunner,
            pool=(ModelRunnerKVCacheMixin, "init_memory_pool"), replay=(CudaGraphRunner, "replay"),
            capturing=get_is_capture_mode, capture_mode=model_capture_mode, weight_apis=(ModelRunner,),
            dp=lambda: get_global_server_args().enable_dp_attention,
            workspace_tokens=lambda *, decode: _legacy_workspace_tokens(get_global_server_args(), decode=decode),
            private_weights=lambda: True,
        )

    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.weight_exporter import WeightExporter
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode, model_capture_mode
    from sglang.srt import runtime_context

    return SimpleNamespace(
        prefill_capture=PrefillCudaGraphRunner, model_runner=ModelRunner,
        pool=(ModelRunner, "alloc_memory_pool"), replay=(DecodeCudaGraphRunner, "execute"),
        capturing=get_is_capture_mode, capture_mode=model_capture_mode, weight_apis=(WeightUpdater, WeightExporter),
        dp=lambda: runtime_context.get_parallel().enable_dp_attention,
        workspace_tokens=lambda *, decode: _split_workspace_tokens(runtime_context, decode=decode),
        private_weights=lambda: runtime_context.get_model().weight_cache_mode == "off",
    )
