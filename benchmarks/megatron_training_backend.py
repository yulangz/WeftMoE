"""Production Megatron dispatcher and grouped experts for the training benchmark.

The operator boundary starts with selected routes/probabilities, like WeftMoE;
router projection/top-k and the optimizer are not part of either measurement.
"""

from importlib.metadata import version
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig

from benchmarks.nccl_library import loaded_nccl_libraries, nccl_runtime_version


class MegatronTrainingBackend:
    def __init__(self, args, w1, w2):
        world, rank = dist.get_world_size(), dist.get_rank()
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=world,
            expert_tensor_parallel_size=1, create_gloo_process_groups=False)
        model_parallel_cuda_manual_seed(args.seed)
        config = TransformerConfig(
            num_layers=1, hidden_size=args.hidden, num_attention_heads=8,
            ffn_hidden_size=args.intermediate, moe_ffn_hidden_size=args.intermediate,
            num_moe_experts=args.experts, moe_router_topk=args.top_k,
            expert_model_parallel_size=world, expert_tensor_parallel_size=1,
            moe_token_dispatcher_type=args.dispatcher, moe_flex_dispatcher_backend="deepep",
            moe_deepep_num_sms=args.deepep_num_sms, variable_seq_lengths=True,
            bf16=True, params_dtype=torch.bfloat16, moe_router_dtype="fp32",
            add_bias_linear=False, gated_linear_unit=True, activation_func=F.silu,
            moe_grouped_gemm=True,
            moe_permute_fusion=getattr(args, "moe_permute_fusion", True),
            bias_activation_fusion=getattr(args, "bias_activation_fusion", True),
            gradient_accumulation_fusion=getattr(args, "accumulate_main_grad", False),
            recompute_granularity="selective", recompute_modules=["moe_act"])
        owned = args.experts // world
        groups = get_default_pg_collection()
        dispatcher = (MoEFlexTokenDispatcher if args.dispatcher == "flex"
                      else MoEAlltoAllTokenDispatcher)
        self.dispatcher = dispatcher(owned, list(range(rank * owned, (rank + 1) * owned)),
                                     config, groups)
        spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.experts, moe_grouped_gemm=True)
        self.experts = spec.submodules.mlp.submodules.experts(owned, config, pg_collection=groups)
        self.w1 = [getattr(self.experts.linear_fc1, f"weight{i}") for i in range(owned)]
        self.w2 = [getattr(self.experts.linear_fc2, f"weight{i}") for i in range(owned)]
        self.accumulate_main_grad = config.gradient_accumulation_fusion
        if self.accumulate_main_grad:
            for weight in self.w1 + self.w2:
                weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        with torch.no_grad():
            for index in range(owned):
                self.w1[index].copy_(w1[index])
                self.w2[index].copy_(w2[index])
        self.num_experts = args.experts
        self.metadata = dict(
            dispatcher=args.dispatcher, deepep_num_sms=args.deepep_num_sms,
            moe_permute_fusion=config.moe_permute_fusion,
            bias_activation_fusion=config.bias_activation_fusion,
            gradient_accumulation_fusion=config.gradient_accumulation_fusion,
            recompute_granularity="selective", recompute_modules=["moe_act"],
            megatron_core=version("megatron-core"), transformer_engine=version("transformer-engine"),
            torch=torch.__version__, torch_nccl_compile_version=torch.cuda.nccl.version(),
            nccl_runtime_version=nccl_runtime_version(), loaded_nccl_libraries=loaded_nccl_libraries(),
            deep_ep=version("deep-ep") if args.dispatcher == "flex" else None)

    def forward(self, x, routes, gates, w1, w2):
        x.requires_grad_(True)
        gates.requires_grad_(True)
        indices = routes.long()
        routing = torch.zeros((x.shape[0], self.num_experts), dtype=torch.bool,
                              device=x.device).scatter_(1, indices, True)
        probabilities = torch.zeros_like(routing, dtype=torch.float32).scatter(1, indices, gates)
        d = self.dispatcher
        hidden, probs = d.dispatch_preprocess(x, routing, probabilities)
        hidden, probs = d.token_dispatch(hidden, probs)
        hidden, counts, probs = d.dispatch_postprocess(hidden, probs)
        hidden, _ = self.experts(hidden, counts, probs)
        output = d.combine_postprocess(d.token_combine(d.combine_preprocess(hidden)))
        return SimpleNamespace(output=output, x=x, gates=gates)

    def backward(self, tape, dy):
        torch.autograd.backward(tape.output, dy)
        gradients = (tape.x.grad, tape.gates.grad,
                     tuple(weight.main_grad if self.accumulate_main_grad else weight.grad
                           for weight in self.w1),
                     tuple(weight.main_grad if self.accumulate_main_grad else weight.grad
                           for weight in self.w2))
        tape.x.grad = tape.gates.grad = None
        self.experts.zero_grad(set_to_none=True)
        return gradients

    def close(self):
        torch.cuda.synchronize()
        del self.dispatcher, self.experts, self.w1, self.w2
        parallel_state.destroy_model_parallel()
