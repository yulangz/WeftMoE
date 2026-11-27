"""SGLang worker hooks before and after the 0.5.20 executor split."""

from functools import cache
from types import SimpleNamespace


def _graph_tokens(max_bs, buckets, verify_width):
    """Bound activation rows in a decode or target-verify graph."""
    largest_bucket = max(buckets or (), default=max_bs or 1)
    return max(max_bs or 1, largest_bucket * (verify_width or 1))


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

        def graph_tokens():
            args = get_global_server_args()
            return _graph_tokens(args.cuda_graph_max_bs, args.cuda_graph_bs,
                                 args.speculative_num_draft_tokens)

        return SimpleNamespace(
            prefill_capture=None, model_runner=ModelRunner,
            pool=(ModelRunnerKVCacheMixin, "init_memory_pool"), replay=(CudaGraphRunner, "replay"),
            capturing=get_is_capture_mode, capture_mode=model_capture_mode, weight_apis=(ModelRunner,),
            dp=lambda: get_global_server_args().enable_dp_attention, schedule=get_global_server_args,
            graph_tokens=graph_tokens,
            private_weights=lambda: True,
        )

    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.weight_exporter import WeightExporter
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode, model_capture_mode
    from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_schedule, get_spec

    def graph_tokens():
        config = get_exec().graph.cuda_graph_config
        if config is None:
            return 1
        return _graph_tokens(config.decode.max_bs, config.decode.bs,
                             get_spec().speculative_num_draft_tokens)

    return SimpleNamespace(
        prefill_capture=PrefillCudaGraphRunner, model_runner=ModelRunner,
        pool=(ModelRunner, "alloc_memory_pool"), replay=(DecodeCudaGraphRunner, "execute"),
        capturing=get_is_capture_mode, capture_mode=model_capture_mode, weight_apis=(WeightUpdater, WeightExporter),
        dp=lambda: get_parallel().enable_dp_attention, schedule=get_schedule, graph_tokens=graph_tokens,
        private_weights=lambda: get_model().weight_cache_mode == "off",
    )
