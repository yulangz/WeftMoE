"""Real SGLang EP parity, outer CUDA graph replay, and residency restart."""

import os
from contextlib import nullcontext
from types import SimpleNamespace, MethodType

import torch
import torch.distributed as dist
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

import weftmoe.sglang as adapter


def check_output(actual, expected):
    if os.environ.get("WEFTMOE_FP8") == "1":
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        diff = ((actual.double() - expected.double()).square().sum()
                / (actual.double().square().sum() + expected.double().square().sum()).clamp_min(1e-30))
        assert diff < 5e-3, diff.item()
    else:
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.00002)


@torch.no_grad()
def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    experts = int(os.environ.get("TEST_EXPERTS", 16))
    top_k = int(os.environ.get("TEST_TOP_K", 4))
    tokens = int(os.environ.get("TEST_TOKENS", 17))
    hidden = int(os.environ.get("TEST_HIDDEN", 2048))
    intermediate = int(os.environ.get("TEST_INTERMEDIATE", 512))
    dp_attention = os.environ.get("TEST_DP_ATTENTION") == "1"
    torch.cuda.set_device(local_rank)
    args = ServerArgs(
        model_path="dummy",
        device="cuda",
        tp_size=world,
        ep_size=world,
        dp_size=world if dp_attention else 1,
        enable_dp_attention=dp_attention,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        chunked_prefill_size=max(512 * world, tokens),
        max_prefill_tokens=max(512 * world, tokens),
        cuda_graph_max_bs=max(512, tokens),
    )
    set_global_server_args_for_scheduler(args)
    initialize_moe_config(args)
    init_distributed_environment(world, rank, local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world, expert_model_parallel_size=world,
        attention_data_parallel_size=world if dp_attention else 1,
    )
    if dp_attention:
        from sglang.srt.layers.dp_attention import initialize_dp_attention, set_dp_buffer_len

        initialize_dp_attention(args, SimpleNamespace(hidden_size=hidden, dtype=torch.bfloat16))
        set_dp_buffer_len(0, 0, True)
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode
    original_forward = FusedMoE.forward
    adapter.install()

    memory_saver = None
    if os.environ.get("TEST_TMS") == "1":
        from torch_memory_saver.entrypoint import TorchMemorySaver

        memory_saver = TorchMemorySaver()
        memory_saver.hook_mode = "torch"
        from sglang.srt.utils import torch_memory_saver_adapter
        torch_memory_saver_adapter._memory_saver = memory_saver
    with (
        memory_saver.region(tag="weights", enable_cpu_backup=True)
        if memory_saver else nullcontext()
    ), torch.device("cuda"):
        layer = FusedMoE(
            experts, hidden, intermediate, layer_id=0, top_k=top_k,
            params_dtype=torch.bfloat16, inplace=False,
        ).cuda()
    if memory_saver is not None and adapter._use_fp8():
        segments = memory_saver._impl._primary_mem_pool.snapshot()
        for p in (layer.w13_weight, layer.w2_weight):
            assert not any(s["address"] <= p.data_ptr() < s["address"] + s["total_size"]
                           for s in segments), "Temporary BF16 weight retained by TMS pool"
    torch.manual_seed(37 + rank)
    layer.w13_weight.normal_(0, 0.03)
    layer.w2_weight.normal_(0, 0.03)
    reference_layer = FusedMoE(
        experts, hidden, intermediate, layer_id=0, top_k=top_k,
        params_dtype=torch.bfloat16, inplace=False,
    ).cuda()
    reference_layer.w13_weight.copy_(layer.w13_weight)
    reference_layer.w2_weight.copy_(layer.w2_weight)
    torch.manual_seed(100 + (rank if dp_attention else 0))
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.03
    ids = (
        torch.arange(tokens * top_k, device="cuda").reshape(tokens, top_k) % experts
    ).int()
    gates = torch.full((tokens, top_k), 1 / top_k, device="cuda")
    topk = StandardTopKOutput(gates, ids, torch.empty(tokens, experts, device="cuda"))
    baseline = MethodType(original_forward, reference_layer)

    def update_weights():
        reference_layer.w13_weight.mul_(0.75)
        reference_layer.w2_weight.mul_(0.875)
        if adapter._use_fp8():
            from weftmoe.fp8 import prepare_fp8_weights
            prepared = prepare_fp8_weights(reference_layer.w13_weight,
                                            reference_layer.w2_weight, training=False)
            targets = adapter.fp8_parameter_map(layer)
            for p, source in ((layer.w13_weight, prepared.w1), (layer.w2_weight, prepared.w2)):
                data, scales = targets[id(p)]
                data.copy_(source.data.view(torch.uint8))
                scales.copy_(source.scales)
                assert torch.equal(data, source.data.view(torch.uint8))
                assert torch.equal(scales, source.scales)
                assert p.untyped_storage().nbytes() == 0
            adapter.commit_fp8_weights(1)
        else:
            layer.w13_weight.copy_(reference_layer.w13_weight)
            layer.w2_weight.copy_(reference_layer.w2_weight)

    def reference():
        if not dp_attention:
            out = baseline(x, topk)
            dist.all_reduce(out)
            return out
        def gather(tensor):
            out = torch.empty((world * tensor.shape[0], *tensor.shape[1:]),
                              device=tensor.device, dtype=tensor.dtype)
            dist.all_gather_into_tensor(out, tensor.contiguous())
            return out
        out = baseline(gather(x), StandardTopKOutput(gather(gates), gather(ids), None))
        dist.all_reduce(out)
        return out.chunk(world)[rank]

    expected = reference()
    adapter.install()
    if not adapter._use_fp8():
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import ModelRunnerKVCacheMixin
        def resolve_pool(pre_model_load_memory):
            assert adapter._runtime is not None and adapter._decode_runtime is not None
            assert adapter._eager_pending is None
            assert all(rt.native.usage()["compute_bytes"] > 0
                       for rt in (adapter._runtime, adapter._decode_runtime))
            return "profiled_after_weftmoe"
        runner = SimpleNamespace(
            model=layer, spec_algorithm=SimpleNamespace(is_none=lambda: True),
            is_draft_worker=False, _resolve_memory_pool_config=resolve_pool,
            _apply_memory_pool_config=lambda config: None, device="cuda", gpu_id=local_rank,
        )
        ModelRunnerKVCacheMixin.init_memory_pool(runner, 0)
        assert runner.memory_pool_config == "profiled_after_weftmoe"
        print(rank, "SGLANG_BF16_MEMORY_BUDGET_PASS", flush=True)
        # Another layer must only add descriptors, not a full FC1 weight copy.
        # Workspaces were warmed above; the reference layer is already allocated.
        before = torch.cuda.memory_allocated()
        reference_layer.layer_id = 1
        adapter._prepare_bf16_memory_budget(reference_layer)
        growth = torch.cuda.memory_allocated() - before
        assert growth < layer.w13_weight.numel() * layer.w13_weight.element_size(), growth
        print(rank, "SGLANG_BF16_NO_WEIGHT_COPY_PASS", growth, flush=True)
    actual = layer(x, topk)
    if adapter._use_fp8():
        assert layer.w13_weight.untyped_storage().nbytes() == 0
        assert layer.w2_weight.untyped_storage().nbytes() == 0
    if not dp_attention:
        dist.all_reduce(actual)
    check_output(actual, expected)
    def capture():
        with model_capture_mode():
            layer(x, topk)
            assert adapter._runtime.max_tokens > 64
            assert adapter._decode_runtime.max_tokens >= (tokens + world - 1) // world
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer(x, topk)
        return graph, captured

    graph, captured = capture()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as probe:
        graph.replay()
        torch.cuda.synchronize()
    kernels = [event.name for event in probe.events() if "weftmoe_dt" in event.name]
    assert kernels, "decode capture did not use DT"
    for iteration in range(3):
        x.mul_(0.9)
        ids.add_(1).remainder_(experts)
        if iteration == 2:
            ids[-1, -1] = -1  # SGLang's masked route must stage as zero gate.
        gates.mul_(0.95)
        eager = layer(x, topk)
        graph.replay()
        torch.cuda.synchronize()
        try:
            check_output(captured, eager)
        except AssertionError as error:
            raise AssertionError(("graph_vs_eager", rank, iteration, str(error))) from error
    # Keep the existing BF16 graph while AWEX updates weights in place.
    # Both PF and DT must observe the new original weights without refreshing.
    model_runner = SimpleNamespace(graph_runner=None, memory_saver_adapter=memory_saver)
    discard_graphs = adapter._use_fp8() or os.environ.get("WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS", "1") == "0"
    if discard_graphs:
        del graph, captured
    adapter.suspend(model_runner)
    if memory_saver:
        memory_saver.pause("weights")
        scratch = torch.empty_like(layer.w13_weight)
        scratch.fill_(7)
        torch.cuda.synchronize()
        del scratch
        memory_saver.resume("weights")
    ids.add_(1).remainder_(experts)
    update_weights()
    expected = reference()
    adapter.resume(model_runner)
    if discard_graphs:
        graph, captured = capture()
    if not adapter._use_fp8():
        resumed_eager = layer(x, topk)
        if not dp_attention:
            dist.all_reduce(resumed_eager)
        check_output(resumed_eager, expected)
    graph.replay()
    actual = captured.clone()
    if not dp_attention:
        dist.all_reduce(actual)
    check_output(actual, expected)
    del graph, captured
    adapter.close(model_runner, keep_fp8=adapter._use_fp8())
    assert adapter._runtime is None
    assert adapter._decode_runtime is None
    assert not adapter._graph_inputs
    # Baseline parity below uses valid routes; the masked-ID check above uses
    # the adapter's existing eager sanitization as its reference.
    ids.add_(1).remainder_(experts)
    update_weights()
    expected = reference()
    restored = layer(x, topk)
    actual = restored.clone()
    if not dp_attention:
        dist.all_reduce(actual)
    check_output(actual, expected)
    graph, captured = capture()
    graph.replay()
    torch.cuda.synchronize()
    # Eager prefill uses PF; decode uses DT with a different GEMM reduction.
    check_output(captured, restored)
    previous = captured.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, previous, rtol=0, atol=0)
    del graph, captured
    adapter.close(model_runner)
    print(rank, "SGLANG_PARITY_GRAPH_RESTART_PASS", flush=True)
    dist.destroy_process_group()
    if memory_saver is not None:
        # This test owns a local TMS instance. Destroy its private MemPool
        # while HookUtil's pluggable allocator is still alive, after tensor
        # owners are gone; the default attribute teardown order does the reverse.
        del baseline, reference_layer, layer, model_runner
        import gc
        gc.collect()
        torch.cuda.synchronize()
        with memory_saver._impl._with_region_config(tag="weights", enable_cpu_backup=True):
            del memory_saver._impl._primary_mem_pool


if __name__ == "__main__":
    main()
