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
        chunked_prefill_size=max(128, tokens),
        max_prefill_tokens=max(128, tokens),
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
    expected = layer(x, topk)
    dist.all_reduce(expected)
    adapter.install()
    actual = layer(x, topk)
    dist.all_reduce(actual)
    torch.testing.assert_close(actual, expected, rtol=0.025, atol=0.00002)
    with model_capture_mode():
        layer(x, topk)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(x, topk)
    for _ in range(3):
        x.mul_(0.9)
        ids.add_(1).remainder_(experts)
        gates.mul_(0.95)
        eager = layer(x, topk)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, eager, rtol=0.02, atol=0.00002)
    del graph
    adapter.suspend(SimpleNamespace(graph_runner=None))
    assert adapter._runtime is None
    assert not adapter._graph_inputs
    restored = layer(x, topk)
    torch.testing.assert_close(restored, eager, rtol=0, atol=0)
    adapter.suspend(SimpleNamespace(graph_runner=None))
    print(rank, "SGLANG_PARITY_GRAPH_RESTART_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
