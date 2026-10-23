"""Compare the real Megatron router/experts/autograd path with the adapter."""

import os
from functools import partial

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
from megatron.core.tensor_parallel import checkpoint, model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

from weftmoe.megatron import MegatronAdapter


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    tp = int(os.environ.get("TEST_TP_SIZE", "1"))
    experts = int(os.environ.get("TEST_EXPERTS", "16"))
    top_k = int(os.environ.get("TEST_TOP_K", "4"))
    capacity = int(os.environ.get("TEST_TOKENS", "16"))
    tokens = capacity
    uneven = os.environ.get("TEST_UNEVEN") == "1"
    if uneven:
        tokens -= (dist.get_rank() // tp) * max(1, capacity // 16)
    hidden = int(os.environ.get("TEST_HIDDEN", "2048"))
    intermediate = int(os.environ.get("TEST_INTERMEDIATE", "512"))
    recompute = os.environ.get("TEST_RECOMPUTE") == "1"
    reference_adapter = os.environ.get("TEST_REFERENCE_ADAPTER") == "1"
    use_checkpoint = recompute
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
        hidden_size=hidden,
        num_attention_heads=8,
        ffn_hidden_size=intermediate,
        moe_ffn_hidden_size=intermediate,
        num_moe_experts=experts,
        moe_router_topk=top_k,
        expert_model_parallel_size=world,
        expert_tensor_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        moe_router_dtype="fp32",
        moe_aux_loss_coeff=0.01,
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall" if uneven else "allgather",
        variable_seq_lengths=uneven,
        use_te_activation_func=True,
        recompute_granularity="full" if recompute else "selective",
        recompute_method="uniform" if recompute else None,
        recompute_num_layers=2 if recompute else None,
        recompute_modules=None if recompute else ["moe_act"],
    )
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=experts, moe_grouped_gemm=True, use_te_activation_func=True
    )
    # MCore 0.19 exposes an MLP builder instead of a ModuleSpec.
    build_mlp = spec.submodules.mlp
    if not callable(build_mlp):
        build_mlp = partial(build_module, build_mlp)
    model = torch.nn.ModuleList(
        [
            build_mlp(
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

    def run_layers(x):
        for layer in model:
            residual, _ = layer(x)
            x = x + residual
        return x

    def run(x):
        return checkpoint(run_layers, False, x) if use_checkpoint else run_layers(x)

    x = (
        torch.randn(tokens, 1, hidden, device="cuda", dtype=torch.bfloat16) * 0.03
    ).requires_grad_()
    dy = torch.randn_like(x) * 0.03
    originals = [layer.forward for layer in model]
    adapter = MegatronAdapter([ddp if ddp is not None else model], max_tokens=max(32, capacity))
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
        # At long sequence lengths, tiny BF16 differences between GEMM
        # implementations can flip the next layer's top-k boundary. Compare
        # checkpoint against the same adapter without checkpoint to isolate
        # recomputation, keeping native Megatron as the default reference.
        use_checkpoint = recompute and not reference_adapter
        for layer, forward in zip(model, adapted if reference_adapter else originals):
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
        use_checkpoint = recompute
        for layer, forward in zip(model, adapted):
            layer.forward = forward
        actual = run(x)
        if recompute:
            adapter.runtime.drain()
            # The initial no-grad pass must not leave backward tapes alive.
            assert adapter.runtime.native.usage()["notify_slots_in_use"] == 0
        backward(actual, dy)
        pairs = [("output", actual, reference), ("dx", x.grad, expected_dx)]
        pairs += [
            (name, gradient(p), expected[name])
            for name, p in model.named_parameters()
            if name in expected
        ]
        for name, a, b in pairs:
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), (step, name)
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
        runtime = adapter.runtime
        adapter.suspend()
        assert adapter.runtime is runtime
        assert runtime.native.usage()["notify_slots_in_use"] == 0
        print(
            dist.get_rank(),
            "MEGATRON_PARITY_AND_RETAINED_BUFFERS_PASS",
            "step",
            step,
            "full_recompute",
            recompute,
            "reference",
            "adapter_no_recompute" if reference_adapter else "megatron",
            flush=True,
        )
    adapter.close()
    assert adapter.runtime is None and adapter.session is None and not adapter.packed
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
