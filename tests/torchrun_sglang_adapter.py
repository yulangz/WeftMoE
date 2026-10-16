"""Real SGLang EP parity, outer CUDA graph replay, and residency restart."""

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from sglang.srt.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.moe.utils import initialize_moe_config
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

import weftmoe.sglang as adapter


@torch.no_grad()
def main():
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    experts = int(os.environ.get("TEST_EXPERTS", 16))
    top_k = int(os.environ.get("TEST_TOP_K", 4))
    tokens = int(os.environ.get("TEST_TOKENS", 17))
    torch.cuda.set_device(local_rank)
    args = ServerArgs(
        model_path="/storage/openpsi/models/Qwen__Qwen3.5-35B-A3B-Base",
        tp_size=world,
        ep_size=world,
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
        tensor_model_parallel_size=world, expert_model_parallel_size=world
    )
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.cuda_graph_runner import model_capture_mode

    layer = FusedMoE(
        experts, 2048, 512, layer_id=0, top_k=top_k,
        params_dtype=torch.bfloat16, inplace=False,
    ).cuda()
    torch.manual_seed(37 + rank)
    layer.w13_weight.normal_(0, 0.03)
    layer.w2_weight.normal_(0, 0.03)
    torch.manual_seed(100)
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16) * 0.03
    ids = (
        torch.arange(tokens * top_k, device="cuda").reshape(tokens, top_k) % experts
    ).int()
    gates = torch.full((tokens, top_k), 1 / top_k, device="cuda")
    topk = StandardTopKOutput(gates, ids, torch.empty(tokens, experts, device="cuda"))
    baseline = layer.forward
    expected = baseline(x, topk)
    dist.all_reduce(expected)
    adapter.install()
    actual = layer(x, topk)
    dist.all_reduce(actual)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.00002)
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
        torch.testing.assert_close(captured, eager, rtol=0.02, atol=0.00002)
    del graph, captured
    adapter.suspend(SimpleNamespace(graph_runner=None))
    assert adapter._runtime is None
    assert adapter._decode_runtime is None
    assert not adapter._graph_inputs
    # Baseline parity below uses valid routes; the masked-ID check above uses
    # the adapter's existing eager sanitization as its reference.
    ids.add_(1).remainder_(experts)
    layer.w13_weight.mul_(0.75)
    layer.w2_weight.mul_(0.875)
    expected = baseline(x, topk)
    dist.all_reduce(expected)
    restored = layer(x, topk)
    actual = restored.clone()
    dist.all_reduce(actual)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.00002)
    graph, captured = capture()
    graph.replay()
    torch.cuda.synchronize()
    # Eager prefill uses PF; decode uses DT with a different GEMM reduction.
    torch.testing.assert_close(captured, restored, rtol=0.02, atol=0.00002)
    previous = captured.clone()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, previous, rtol=0, atol=0)
    del graph, captured
    adapter.suspend(SimpleNamespace(graph_runner=None))
    print(rank, "SGLANG_PARITY_GRAPH_RESTART_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
