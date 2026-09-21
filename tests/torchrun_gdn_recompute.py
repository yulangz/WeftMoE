"""Verify the AReaL GDN checkpoint with real TP and FLA kernels."""

import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_gated_delta_net_module_spec,
)
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.transformer_config import TransformerConfig

import areal.engine.megatron_utils.megatron_bridge_patches  # noqa: F401


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    tp = dist.get_world_size()
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=tp)
    model_parallel_cuda_manual_seed(1)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=2048,
        num_attention_heads=8,
        tensor_model_parallel_size=tp,
        sequence_parallel=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        normalization="RMSNorm",
        add_bias_linear=False,
        activation_func=torch.nn.functional.silu,
        experimental_attention_variant="gated_delta_net",
        linear_attention_freq=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_conv_kernel_dim=4,
    )
    layer = (
        build_module(
            get_gated_delta_net_module_spec(config),
            config=config,
            layer_number=1,
            pg_collection=get_default_pg_collection(),
        )
        .cuda()
        .bfloat16()
    )
    x = torch.randn(
        512, 2, 2048, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    dy = torch.randn_like(x) * 0.01
    ref, _ = layer(x, None)
    ref.backward(dy)
    grads = {n: p.grad.clone() for n, p in layer.named_parameters()}
    dx = x.grad.clone()
    for p in layer.parameters():
        p.grad = None
    x.grad = None
    config.recompute_granularity = "selective"
    config.recompute_modules = ["core_attn", "layernorm"]
    out, _ = layer(x, None)
    out.backward(dy)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, dx, rtol=0, atol=0)
    for name, p in layer.named_parameters():
        torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    print(dist.get_rank(), "GDN_RECOMPUTE_PARITY_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
