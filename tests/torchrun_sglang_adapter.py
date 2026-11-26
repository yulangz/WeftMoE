"""Real SGLang EP parity, outer CUDA graph replay, and residency restart."""

import os
from contextlib import nullcontext
from types import SimpleNamespace, MethodType

import pytest
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
    api = adapter.worker_api()
    modern = api.pool[1] == "alloc_memory_pool"
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
        max_running_requests=max(512, tokens),
        **({} if modern else {"cuda_graph_max_bs": max(512, tokens)}),
    )
    set_global_server_args_for_scheduler(args)
    if modern:
        initialize_moe_config()
    else:
        initialize_moe_config(args)
    from sglang.srt.distributed.parallel_state import set_custom_all_reduce
    set_custom_all_reduce(False)
    init_distributed_environment(world, rank, local_rank=local_rank)
    initialize_model_parallel(
        tensor_model_parallel_size=world, expert_model_parallel_size=world,
        attention_data_parallel_size=world if dp_attention else 1,
    )
    if dp_attention:
        from sglang.srt.layers.dp_attention import initialize_dp_attention, set_dp_buffer_len

        initialize_dp_attention(args, SimpleNamespace(hidden_size=hidden, dtype=torch.bfloat16, hf_config=None))
        set_dp_buffer_len(0, 0, True)
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    model_capture_mode = api.capture_mode
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
    # Stop at the real pool hook, just before SGLang allocates KV memory.
    class PoolReady(Exception):
        pass

    def resolve_pool(*args, **kwargs):
        assert adapter._runtime is not None and adapter._decode_runtime is not None
        assert adapter._eager_pending is None
        for rt in (adapter._runtime, adapter._decode_runtime):
            usage = rt.native.usage()
            assert usage.get("compute_bytes", usage.get("symmetric_bytes", 0)) > 0
        raise PoolReady()

    runner = SimpleNamespace(model=layer, spec_algorithm=SimpleNamespace(is_none=lambda: True),
                             is_draft_worker=False, _resolve_memory_pool_config=resolve_pool,
                             init_kv_cache_configurator=resolve_pool, device="cuda", gpu_id=local_rank)
    try:
        if modern:
            getattr(*api.pool)(runner)
        else:
            getattr(*api.pool)(runner, 0)
    except PoolReady:
        pass
    else:
        raise AssertionError("SGLang pool sizing hook did not run")
    print(rank, "SGLANG_MEMORY_BUDGET_PASS", flush=True)
    if adapter._use_fp8():
        assert all(p.untyped_storage().nbytes() == 0
                   for p in (layer.w13_weight, layer.w2_weight))
        print(rank, "SGLANG_FP8_MEMORY_BUDGET_PASS", flush=True)
    else:
        # Another layer must only add descriptors, not a full FC1 weight copy.
        # Workspaces were warmed above; the reference layer is already allocated.
        probe_layer = FusedMoE(experts, hidden, intermediate, layer_id=1, top_k=top_k,
                               params_dtype=torch.bfloat16, inplace=False).cuda()
        before = torch.cuda.memory_allocated()
        adapter._prepare_memory_budget(probe_layer)
        growth = torch.cuda.memory_allocated() - before
        assert growth < layer.w13_weight.numel() * layer.w13_weight.element_size(), growth
        print(rank, "SGLANG_BF16_NO_WEIGHT_COPY_PASS", growth, flush=True)
    tiled = adapter._runtime._sglang_tiled_bf16
    if adapter._runtime.supports_tiled_weights and not adapter._use_fp8():
        # This worker owns its weights. Test the serving default and its opt-out.
        assert tiled == (os.environ.get("WEFTMOE_SGLANG_TILED_BF16", "1") == "1")
    if tiled:
        storage = (layer.w13_weight, layer.w2_weight)
        views = adapter._sglang_bf16.serving_weights(layer)
        assert all(p.data_ptr() == v.data_ptr() and v.ndim == 5 for p, v in zip(storage, views))
        # The SGLang checkpoint loader can also update just one expert shard.
        gate = reference_layer.w13_weight[0, :intermediate].clone()
        # A bad tensor from the checkpoint reader leaves this expert unavailable
        # until a valid copy of the same shard arrives.
        with pytest.raises(RuntimeError):
            layer.w13_weight.weight_loader(layer.w13_weight, gate.unsqueeze(0).expand(2, -1, -1),
                                           "w13_weight", "w1", rank * layer.num_local_experts)
        with pytest.raises(RuntimeError):
            adapter._sglang_bf16.prepare_for_replay()
        layer.w13_weight.weight_loader(layer.w13_weight, gate, weight_name="w13_weight",
                                       shard_id="w1", expert_id=rank * layer.num_local_experts)
        weights = adapter._sglang_bf16._layers[layer]
        assert not weights.active and weights.pending_shards
        layer.w13_weight.weight_loader(layer.w13_weight, gate, weight_name="w13_weight",
                                       shard_id="w1", expert_id=rank * layer.num_local_experts)
        assert not weights.active and weights.pending_shards
        snapshot = layer.state_dict()
        assert torch.equal(snapshot["w13_weight"], reference_layer.w13_weight)
        assert torch.equal(snapshot["w2_weight"], reference_layer.w2_weight)
        # Partial loads leave the other matrix intact and preserve graph addresses.
        layer.load_state_dict({"w13_weight": snapshot["w13_weight"]}, strict=False)
        assert torch.equal(layer.state_dict()["w2_weight"], reference_layer.w2_weight)
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
    symbol = "routed_bf16" if adapter._decode_runtime.supports_tiled_weights else "weftmoe_dt"
    kernels = [event.name for event in probe.events() if symbol in event.name]
    if adapter._decode_runtime.supports_tiled_weights:
        assert adapter._decode_runtime.native.buffer.decode_capacity > 0
    assert kernels, "decode capture did not use DT"
    for iteration in range(3):
        x.mul_(0.9)
        ids.add_(1).remainder_(experts)
        if iteration == 2:
            ids[-1] = torch.arange(top_k, device='cuda')
            ids[-1, 0] = top_k - 1
            ids[-1, -1] = -1  # SGLang's masked route must stage as zero gate.
        gates.mul_(0.95)
        eager = layer(x, topk)
        graph.replay()
        torch.cuda.synchronize()
        try:
            check_output(captured, eager)
        except AssertionError as error:
            raise AssertionError(("graph_vs_eager", rank, iteration, str(error))) from error
    if tiled:
        # The stock Triton reference below expects valid expert IDs.
        ids.clamp_min_(0)
        previous = layer(x, topk).clone()
        if modern:
            # Exercise the installed SGLang updater with the real expert loader.
            # The first tensor is written before the second tensor fails.
            def load_checkpoint(named_tensors):
                for shard, tensor in named_tensors:
                    layer.w13_weight.weight_loader(layer.w13_weight, tensor, "w13_weight",
                                                   shard, rank * layer.num_local_experts)

            updater = SimpleNamespace(device="cuda", tp_rank=rank, custom_weight_loaders=(),
                                      _assert_weight_cache_inactive=lambda _: None,
                                      get_model=lambda: SimpleNamespace(load_weights=load_checkpoint))
            update = api.weight_apis[0].update_weights_from_tensor
            gate, up = reference_layer.w13_weight[0].chunk(2)
            gate = gate * 0.5
            with pytest.raises(RuntimeError):
                update(updater, [("w1", gate), ("w3", up.unsqueeze(0).expand(2, -1, -1))])
            with pytest.raises(RuntimeError, match="weight update"):
                getattr(*api.replay)(None)
            with pytest.raises(RuntimeError, match="weight update"):
                adapter._sglang_bf16.serving_weights(layer)
            assert update(updater, [("w1", gate), ("w3", up)])[0]
            reference_layer.w13_weight[0, :intermediate].copy_(gate)
            print(rank, "SGLANG_PARTIAL_UPDATE_RETRY_PASS", flush=True)
        with adapter.canonical_weights():
            reference_layer.w13_weight[:, :intermediate].mul_(0.5)
            layer.w13_weight[:, :intermediate].copy_(reference_layer.w13_weight[:, :intermediate])
        eager = layer(x, topk)
        graph.replay()
        assert not torch.equal(eager, previous)
        assert torch.equal(eager.view(torch.int16), captured.view(torch.int16))
        actual = eager.clone()
        if not dp_attention:
            dist.all_reduce(actual)
        check_output(actual, reference())
        print(rank, "SGLANG_SHARED_BF16_UPDATE_PASS", flush=True)
    # Keep the existing BF16 graph while AWEX updates weights in place.
    # Both PF and DT must observe the new weights at their captured addresses.
    # Exercise resume's recapture callback, not just a manual capture afterwards.
    runner = SimpleNamespace(
        graphs={0: graph}, output_buffers={0: captured}, seq_len_fill_value=1,
        buffers=SimpleNamespace(seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
                                seq_lens_cpu=torch.ones(1, dtype=torch.int32)))
    def recapture():
        nonlocal graph, captured
        graph, captured = capture()
        runner.graphs[0], runner.output_buffers[0] = graph, captured
    runner.capture = recapture
    model_runner = SimpleNamespace(graph_runner=runner, memory_saver_adapter=memory_saver)
    discard_graphs = adapter._use_fp8() or os.environ.get("WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS", "1") == "0"
    if discard_graphs:
        del graph, captured
    adapter.suspend(model_runner, discard_fp8=adapter._use_fp8())
    if tiled:
        suspended_state = layer.state_dict()
        before_update = (reference_layer.w13_weight.clone(), reference_layer.w2_weight.clone())
        # Call the installed runner methods. The guard must fire before any
        # backend access, even when the retained graph bypasses Python forward.
        replay_methods = [getattr(*api.replay)]
        if api.prefill_capture is not None:
            replay_methods.append(api.prefill_capture.execute)
        for replay in replay_methods:
            try:
                replay(None)
            except RuntimeError as exc:
                assert "weight update or suspension" in str(exc)
            else:
                raise AssertionError("Replayed with canonical BF16 experts")
    if adapter._use_fp8():
        assert all(t.untyped_storage().nbytes() == 0
                   for weights in adapter._fp8_weights.values() for t in weights._tensors())
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
    if not adapter._use_fp8():
        if tiled:
            assert torch.equal(suspended_state["w13_weight"], before_update[0])
            assert torch.equal(suspended_state["w2_weight"], before_update[1])
        resumed_eager = layer(x, topk)
        if not dp_attention:
            dist.all_reduce(resumed_eager)
        check_output(resumed_eager, expected)
    graph.replay()
    actual = captured.clone()
    if not dp_attention:
        dist.all_reduce(actual)
    check_output(actual, expected)
    if adapter._use_fp8():
        version = adapter._fp8_weights[layer.layer_id].version
        del graph, captured
        adapter.suspend(model_runner)  # KV-only: restore the same weights/version.
        adapter.resume(model_runner)
        assert adapter._fp8_weights[layer.layer_id].version == version
        graph.replay()
        actual = captured.clone()
        if not dp_attention:
            dist.all_reduce(actual)
        check_output(actual, expected)
        print(rank, "SGLANG_FP8_KV_ONLY_VERSION_PASS", flush=True)
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
        if tiled:
            del storage, views
        del baseline, reference_layer, layer, model_runner
        import gc
        gc.collect()
        torch.cuda.synchronize()
        if hasattr(memory_saver._impl, "_primary_mem_pool"):
            with memory_saver._impl._with_region_config(tag="weights", enable_cpu_backup=True):
                del memory_saver._impl._primary_mem_pool
        else:
            with memory_saver._impl._with_region_config(tag="weights", enable_cpu_backup=True,
                                                        cpu_backup_backend=None):
                memory_saver._impl._mem_pools.clear()


if __name__ == "__main__":
    main()
