#!/usr/bin/env python3
"""SGLang routed MoE decode on datacenter Blackwell, with DP=EP."""

import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.models import add_model_arguments, resolve_model
from benchmarks.nccl_library import loaded_nccl_libraries, nccl_runtime_version
from benchmarks.sm100.hardware import check_blackwell, check_finite


RUNNER = "flashinfer_trtllm_routed"


def make_layer(args, rank):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.utils import RoutingMethodType
    from sglang.srt.layers.quantization.nvfp4_online import (
        ModelOptNvFp4OnlineFusedMoEMethod,
        NvFp4OnlineConfig,
    )

    fp4 = args.precision == "fp4"
    with torch.device("cuda"):
        layer = FusedMoE(
            args.experts, args.hidden, args.intermediate, layer_id=0, top_k=args.top_k,
            params_dtype=torch.bfloat16, inplace=False, prefix="bench.moe",
            quant_config=NvFp4OnlineConfig() if fp4 else None,
            routing_method_type=RoutingMethodType.TopK,
        ).eval()
    if fp4 and not isinstance(layer.quant_method, ModelOptNvFp4OnlineFusedMoEMethod):
        raise RuntimeError("SGLang did not select the online NVFP4 MoE method")

    generator = torch.Generator(device="cuda").manual_seed(37 + rank)
    owned = layer.num_local_experts
    for local_expert in range(owned):
        expert = rank * owned + local_expert
        for param, name, shard, shape in (
            (layer.w13_weight, "w13_weight", "w1", (args.intermediate, args.hidden)),
            (layer.w13_weight, "w13_weight", "w3", (args.intermediate, args.hidden)),
            (layer.w2_weight, "w2_weight", "w2", (args.hidden, args.intermediate)),
        ):
            weight = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            weight.normal_(0, 0.03, generator=generator)
            param.weight_loader(param, weight, weight_name=name, shard_id=shard, expert_id=expert)
    layer.quant_method.process_weights_after_loading(layer)
    return layer


def measure(args, tokens, layer, group, rank, world):
    from sglang.srt.layers.dp_attention import (
        DpPaddingMode,
        dp_gather_partial,
        dp_reduce_scatter_tensor,
        set_dp_buffer_len,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    global_tokens = tokens * world
    set_dp_buffer_len(global_tokens, tokens, True, [tokens] * world)
    x = torch.empty((tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    gathered = x.new_empty((global_tokens, args.hidden))
    output = torch.empty_like(x)
    logits = torch.empty((global_tokens, args.experts), device=x.device)
    ids = torch.empty((global_tokens, args.top_k), device=x.device, dtype=torch.int32)
    gates = torch.empty_like(ids, dtype=torch.float32)
    topk = StandardTopKOutput(gates, ids, logits)
    batch = SimpleNamespace(dp_padding_mode=DpPaddingMode.MAX_LEN)
    routing_rng = torch.Generator(device=x.device).manual_seed(args.seed + global_tokens)
    input_rng = torch.Generator(device=x.device).manual_seed(args.seed + global_tokens + rank)
    bias = torch.randn(args.experts, device=x.device, generator=routing_rng) * args.route_bias_std

    def prepare():
        x.normal_(0, 0.03, generator=input_rng)
        logits.normal_(generator=routing_rng).add_(bias)
        scores, selected = logits.topk(args.top_k, dim=-1)
        ids.copy_(selected)
        gates.copy_(scores.softmax(-1))

    def forward():
        dp_gather_partial(gathered, x, batch)
        dp_reduce_scatter_tensor(output, layer(gathered, topk))

    prepare()
    with model_capture_mode(), group.graph_capture() as capture:
        for _ in range(3):
            forward()
        capture.stream.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture.stream):
            forward()

    # Check graph replay against the eager path on a fresh input.
    prepare()
    forward()
    expected = output.clone()
    graph.replay()
    matches = (torch.isclose(output, expected, atol=2e-5, rtol=0.025).all()
               & torch.isfinite(output).all() & (output != 0).any()).int()
    dist.all_reduce(matches, op=dist.ReduceOp.MIN)
    if not matches.item():
        raise RuntimeError("SGLang graph replay failed the eager output check")

    samples = []
    workload = hashlib.sha256()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for step in range(args.warmups + args.samples):
        prepare()
        workload.update(ids.cpu().numpy().tobytes())
        workload.update(gates.cpu().numpy().tobytes())
        dist.barrier()
        torch.cuda.synchronize()
        begin.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        if step >= args.warmups:
            samples.append(begin.elapsed_time(end) * 1000 / args.replays)
        check_finite(output)

    per_rank = [None] * world
    dist.all_gather_object(per_rank, samples)
    if rank == 0:
        slowest = list(map(max, zip(*per_rank)))
        print("SGLANG_DECODE_JSON " + json.dumps(dict(
            backend="sglang", model=args.model, precision=args.precision,
            compute_dtype="nvfp4" if args.precision == "fp4" else "bf16", input_dtype="bf16",
            world=world, local_world_size=int(os.environ["LOCAL_WORLD_SIZE"]),
            dp_size=world, tokens_scope="dp_local", tokens_per_rank=tokens,
            experts=args.experts, top_k=args.top_k, hidden=args.hidden, intermediate=args.intermediate,
            runner_backend=RUNNER, moe_a2a_backend="none",
            communication="all_gather+reduce_scatter",
            route_bias_std=args.route_bias_std, seed=args.seed, routing_sha256=workload.hexdigest(),
            warmups=args.warmups, samples=args.samples, replays=args.replays,
            rank_max_p50_us=statistics.median(slowest), rank_max_samples_us=slowest,
            per_rank_us=per_rank, gpu=torch.cuda.get_device_name(),
            compute_capability=torch.cuda.get_device_capability(),
            implementation=dict(torch=torch.__version__, sglang=version("sglang"),
                                flashinfer=version("flashinfer-python"),
                                nccl_runtime_version=nccl_runtime_version(),
                                loaded_nccl_libraries=loaded_nccl_libraries()),
        )), flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--precision", choices=("bf16", "fp4"), default="bf16")
    parser.add_argument("--tokens", type=int, nargs="+", default=[64, 512])
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--route-bias-std", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1300)
    args = parser.parse_args()
    resolve_model(args)
    if min(min(args.tokens), args.samples, args.replays) < 1 or args.warmups < 0:
        parser.error("tokens, samples and replays must be positive; warmups must be nonnegative")
    if not math.isfinite(args.route_bias_std) or args.route_bias_std < 0:
        parser.error("--route-bias-std must be finite and nonnegative")

    rank, world = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE"))
    local_rank = int(os.environ["LOCAL_RANK"])
    if args.experts % world or not 1 <= args.top_k <= args.experts:
        parser.error("experts must be divisible by EP and top-k must be between 1 and experts")
    if args.hidden % 128 or args.intermediate % 128:
        parser.error("FlashInfer TRT-LLM requires hidden and intermediate dimensions divisible by 128")
    torch.cuda.set_device(local_rank)
    check_blackwell()

    from sglang.srt.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.layers.moe.utils import get_moe_runner_backend, initialize_moe_config
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    server = ServerArgs(
        model_path="dummy", device="cuda", tp_size=world, ep_size=world, dp_size=world,
        enable_dp_attention=True, moe_runner_backend=RUNNER, moe_a2a_backend="none",
        quantization="nvfp4_online" if args.precision == "fp4" else None,
        disable_custom_all_reduce=True,
    )
    publish(server, role="scheduler")
    initialize_moe_config()
    if not get_moe_runner_backend().is_flashinfer_trtllm_routed():
        raise RuntimeError(f"SGLang did not select {RUNNER}")
    set_custom_all_reduce(False)
    init_distributed_environment(world, rank, local_rank=local_rank, timeout=180)
    try:
        initialize_model_parallel(
            tensor_model_parallel_size=world, expert_model_parallel_size=world,
            attention_data_parallel_size=world,
        )
        initialize_dp_attention(
            server, SimpleNamespace(hidden_size=args.hidden, dtype=torch.bfloat16, hf_config=None),
        )
        group = get_tp_group()
        layer = make_layer(args, rank)
        for tokens in args.tokens:
            measure(args, tokens, layer, group, rank, world)
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
