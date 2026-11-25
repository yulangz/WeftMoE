"""SGLang worker hooks before and after the 0.5.20 executor split."""

from functools import cache
from types import SimpleNamespace


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
            dp=lambda: get_global_server_args().enable_dp_attention, schedule=get_global_server_args,
            graph_bs=lambda: get_global_server_args().cuda_graph_max_bs or 1,
            private_weights=lambda: True,
        )

    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.model_runner_components.weight_exporter import WeightExporter
    from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
    from sglang.srt.model_executor.runner.decode_cuda_graph_runner import DecodeCudaGraphRunner
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import PrefillCudaGraphRunner
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode, model_capture_mode
    from sglang.srt.runtime_context import get_exec, get_model, get_parallel, get_schedule
    def graph_bs():
        config = get_exec().graph.cuda_graph_config
        return (config.decode.max_bs if config is not None else None) or 1

    return SimpleNamespace(
        prefill_capture=PrefillCudaGraphRunner, model_runner=ModelRunner,
        pool=(ModelRunner, "alloc_memory_pool"), replay=(DecodeCudaGraphRunner, "execute"),
        capturing=get_is_capture_mode, capture_mode=model_capture_mode, weight_apis=(WeightUpdater, WeightExporter),
        dp=lambda: get_parallel().enable_dp_attention, schedule=get_schedule, graph_bs=graph_bs,
        private_weights=lambda: get_model().weight_cache_mode == "off",
    )
