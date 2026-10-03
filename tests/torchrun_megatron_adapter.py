"""Compare the real Megatron router/experts/autograd path with the adapter."""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.distributed_data_parallel_config import (
    DistributedDataParallelConfig,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from weftmoe.megatron import MegatronAdapter


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    tp = int(os.environ.get("TEST_TP_SIZE", "1"))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
        create_gloo_process_groups=False,
    )
    # Router weights are initialized on CPU; CUDA's model-parallel RNG alone
    # neither reproduces them nor gives replicas the same initial parameters.
    torch.manual_seed(123)
    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=2,
        tensor_model_parallel_size=tp,
        sequence_parallel=tp > 1,
        hidden_size=2048,
        num_attention_heads=8,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        num_moe_experts=16,
        moe_router_topk=4,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_grouped_gemm=True,
        use_te_activation_func=True,
        recompute_granularity="selective",
        recompute_modules=["moe_act"],
    )
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=16, moe_grouped_gemm=True, use_te_activation_func=True
    )
    model = torch.nn.ModuleList(
        [
            build_module(
                spec.submodules.mlp,
                config=config,
                pg_collection=get_default_pg_collection(),
            )
            .cuda()
            .bfloat16()
            for _ in range(2)
        ]
    )
    for i, layer in enumerate(model):
        layer.set_layer_number(i + 1)
    ddp = None
    if os.environ.get("TEST_DDP") == "1":
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(grad_reduce_in_fp32=True),
            model,
        )

    def gradient(parameter):
        return parameter.main_grad if ddp is not None else parameter.grad

    def backward(output, derivative):
        output.backward(derivative)
        # Accumulate a second microbatch through the same parameter hooks.
        if ddp is not None:
            run(x).backward(derivative)
            ddp.finish_grad_sync()

    def run(x):
        for layer in model:
            residual, _ = layer(x)
            x = x + residual
        return x

    x = (
        torch.randn(16, 1, 2048, device="cuda", dtype=torch.bfloat16) * 0.03
    ).requires_grad_()
    dy = torch.randn_like(x) * 0.03
    originals = [layer.forward for layer in model]
    adapter = MegatronAdapter([ddp if ddp is not None else model], max_tokens=32)
    adapter.prepare()
    adapted = [layer.forward for layer in model]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    def clear_gradients():
        optimizer.zero_grad(set_to_none=True)
        if ddp is not None:
            ddp.zero_grad_buffer()
        x.grad = None

    for step in range(2):
        clear_gradients()
        for layer, forward in zip(model, originals):
            layer.forward = forward
        reference = run(x)
        backward(reference, dy)
        expected = {
            name: gradient(p).clone()
            for name, p in model.named_parameters()
            if gradient(p) is not None
        }
        expected_dx = x.grad.clone()
        clear_gradients()
        for layer, forward in zip(model, adapted):
            layer.forward = forward
        actual = run(x)
        backward(actual, dy)
        pairs = [("output", actual, reference), ("dx", x.grad, expected_dx)]
        pairs += [
            (name, gradient(p), expected[name])
            for name, p in model.named_parameters()
            if name in expected
        ]
        for name, a, b in pairs:
            error = (a.float() - b.float()).square().mean().sqrt()
            scale = b.float().square().mean().sqrt().clamp_min(1e-10)
            if error / scale > 0.025:
                raise AssertionError(
                    (
                        step,
                        name,
                        (error / scale).item(),
                        (a.float() - b.float()).abs().max().item(),
                    )
                )
        if ddp is not None:
            for p in model.parameters():
                p.grad = p.main_grad.to(p.dtype)
        optimizer.step()
        usage = adapter.runtime.native.usage()
        resident_bytes = sum(usage[key] for key in ("compute_bytes", "gin_bytes", "notify_bytes"))
        torch.cuda.empty_cache()
        before = torch.cuda.mem_get_info()[0]
        adapter.suspend()
        after = torch.cuda.mem_get_info()[0]
        assert adapter.runtime is None and not adapter.packed
        assert after - before >= resident_bytes, (after - before, resident_bytes)
        print(
            dist.get_rank(),
            "MEGATRON_PARITY_AND_OFFLOAD_PASS",
            "step",
            step,
            "freed_bytes",
            after - before,
            flush=True,
        )
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
