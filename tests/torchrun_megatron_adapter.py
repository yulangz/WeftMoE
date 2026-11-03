"""Compare the real Megatron router/experts/autograd path with the adapter."""

import os
import io
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
    use_muon = os.environ.get("TEST_MUON") == "1"
    if use_muon:
        from megatron.core.optimizer import OptimizerConfig
        from megatron.core.process_groups_config import ProcessGroupCollection
        from areal.engine.megatron_utils.gpu_staged_muon import (
            GPUStagedMuonConfig, get_megatron_optimizer_with_gpu_staged_muon,
        )

        optimizer = get_megatron_optimizer_with_gpu_staged_muon(
            OptimizerConfig(
                optimizer="muon", use_layer_wise_distributed_optimizer=True,
                lr=0.001, min_lr=0.0, bf16=True,
                use_distributed_optimizer=False, use_precision_aware_optimizer=False,
                muon_scalar_optimizer="adam", muon_split_qkv=False,
                muon_tp_mode="duplicated", clip_grad=0.0,
            ),
            [ddp if ddp is not None else model],
            GPUStagedMuonConfig(buffer_count=1, slot_size_mb=128),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(
                ["tp", "expt_tp", "pp", "cp", "ep", "mp", "tp_ep_pp", "dp", "dp_cp", "expt_dp"]),
        )
        if os.environ.get("TEST_FP32_MASTER_CHECKPOINT") == "1":
            optimizer.bind_managed_checkpoint_process_group(dist.new_group(backend="gloo"))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        adapter.bind_optimizer(optimizer)
    primary = os.environ.get("TEST_FP32_MASTER") == "1"
    if primary:
        assert adapter.fp8 and use_muon and reference_adapter
        adapter.bind_fp32_masters(optimizer)

    def clear_gradients():
        optimizer.zero_grad() if use_muon else optimizer.zero_grad(set_to_none=True)
        if ddp is not None:
            ddp.zero_grad_buffer()
        x.grad = None

    for step in range(2):
        if primary:
            adapter.release_bf16_weights()
        clear_gradients()
        # At long sequence lengths, tiny BF16 differences between GEMM
        # implementations can flip the next layer's top-k boundary. Compare
        # checkpoint against the same adapter without checkpoint to isolate
        # recomputation, keeping native Megatron as the default reference.
        use_checkpoint = recompute and (not reference_adapter or use_muon)
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
        expected_fp8 = None
        if adapter.fp8 and not primary:
            if reference_adapter:
                expected_fp8 = {layer: tuple(t.clone() for t in entry[1]._tensors())
                                for layer, entry in adapter.fp8_weights.items()}
            # Rebuild the standalone BF16-backed cache for the replay check.
            adapter.fp8_weights.clear()
        actual = run(x)
        if adapter.fp8:
            assert adapter.runtime.native.usage()['fp8_weight_cache_entries'] == len(model)
        if recompute:
            adapter.runtime.drain()
            # The initial no-grad pass must not leave backward tapes alive.
            assert adapter.runtime.native.usage()["notify_slots_in_use"] == 0
        backward(actual, dy)
        if expected_fp8 is not None:
            for layer, tensors in expected_fp8.items():
                assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                           for a, b in zip(adapter.fp8_weights[layer][1]._tensors(), tensors))
            del expected_fp8
        pairs = [("output", actual, reference), ("dx", x.grad, expected_dx)]
        pairs += [
            (name, gradient(p), expected[name])
            for name, p in model.named_parameters()
            if name in expected
        ]
        for name, a, b in pairs:
            assert torch.isfinite(a).all() and torch.isfinite(b).all(), (step, name)
            if adapter.fp8:
                diff = ((a.double() - b.double()).square().sum()
                        / (a.double().square().sum() + b.double().square().sum()).clamp_min(1e-30))
                if diff >= 5e-3:
                    raise AssertionError((step, name, "diff", diff.item()))
                continue
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
        if ddp is not None and not use_muon:
            for p in model.parameters():
                p.grad = p.main_grad.to(p.dtype)
        weight_version = adapter.weight_version
        update = optimizer.step()
        if use_muon:
            assert update[0] and torch.isfinite(torch.as_tensor(update[1])), update
            adapter.weights_updated()
        assert adapter.weight_version == weight_version + 1
        if primary:
            from weftmoe.fp8 import _quantize_fp8_weight_pair
            optimizer.drain()
            for layer in model:
                weights = adapter.fp8_weights[layer][1]
                for kind, index in ((0, 0), (1, layer.num_local_experts)):
                    p = adapter._parameters(layer)[index]
                    assert p.untyped_storage().nbytes() == 0
                    master = adapter.master_weights.masters[id(p)].cuda().view(p.shape)
                    row, col = _quantize_fp8_weight_pair(master)
                    actual_pair = ((weights.w1, weights.w1_t), (weights.w2, weights.w2_t))[kind]
                    for expected_q, actual_q in zip((row, col), actual_pair):
                        assert torch.equal(expected_q.data.view(torch.uint8), actual_q.data[0].view(torch.uint8))
                        assert torch.equal(expected_q.scales, actual_q.scales[0])
        runtime = adapter.runtime
        session = adapter.session
        adapter.suspend()
        if primary:
            assert all(t.untyped_storage().nbytes() == 0
                       for weights in adapter.master_weights.weights.values()
                       for w in weights for t in (w.data, w.scales))
        assert adapter.runtime is None and (runtime is None or runtime.native is None)
        assert adapter.session is session
        assert not adapter.packed and not adapter.fp8_weights
        if os.environ.get("TEST_WEIGHT_OFFLOAD") == "1":
            from types import SimpleNamespace
            from areal.engine.megatron_utils.weight_residency import MegatronWeightResidency
            assert ddp is not None
            residency = MegatronWeightResidency(SimpleNamespace(model=[ddp], optimizer=optimizer))
            resident = [p for p in model.parameters() if p.untyped_storage().nbytes()]
            saved = [p.detach().cpu().clone() for p in resident]
            identities = [p.untyped_storage()._cdata for p in model.parameters()]
            residency.release_memory(tags=["weights"])
            if use_muon and not primary:
                assert len(residency._master_backed_weights) > 0
            assert all(p.untyped_storage().nbytes() == 0 for p in model.parameters())
            residency.resume_memory(tags=["weights"])
            residency.ensure_grad_buffers()
            assert identities == [p.untyped_storage()._cdata for p in model.parameters()]
            assert all(torch.equal(a, p.cpu()) for a, p in zip(saved, resident))
        print(
            dist.get_rank(),
            "MEGATRON_PARITY_AND_WORKSPACE_RELEASE_PASS",
            "step",
            step,
            "full_recompute",
            recompute,
            "reference",
            ("adapter_full_recompute" if use_muon else "adapter_no_recompute")
            if reference_adapter else "megatron",
            flush=True,
        )
    if primary and os.environ.get("TEST_FP32_MASTER_CHECKPOINT") == "1":
        import copy
        adapter.release_bf16_weights()
        manager = adapter.master_weights
        saved_weights = {layer: tuple(t.cpu().clone() for w in weights for t in (w.data, w.scales))
                         for layer, weights in manager.weights.items()}
        adapter.restore_bf16_weights()
        for key, (_, _, _, parameter) in manager.bindings.items():
            assert torch.equal(parameter.cpu(), manager.masters[key].view(parameter.shape).bfloat16())
        checkpoint_state = copy.deepcopy(optimizer.state_dict())
        for master in manager.masters.values():
            master.mul_(0.5)
        manager.refresh()
        adapter.restore_bf16_weights()
        optimizer.load_state_dict(checkpoint_state)
        manager.refresh()
        for layer, expected_tensors in saved_weights.items():
            for actual_tensor, expected_tensor in zip(adapter.fp8_weights[layer][1]._tensors(), expected_tensors):
                assert torch.equal(actual_tensor.view(torch.uint8).cpu(), expected_tensor.view(torch.uint8))
        assert all(p.untyped_storage().nbytes() == 0 for layer in model for p in adapter._parameters(layer))
        print(dist.get_rank(), "FP32_MASTER_CHECKPOINT_ROUNDTRIP_PASS", flush=True)
    if primary:
        close_expected = {id(p): adapter.master_weights.masters[id(p)].view(p.shape).bfloat16().cpu()
                          for layer in model for p in adapter._parameters(layer)}
    adapter.close()
    if primary:
        for layer in model:
            for p in adapter._parameters(layer):
                assert torch.equal(p.cpu(), close_expected[id(p)])
        # A checkpoint remains readable after removing the adapter.
        torch.save(model.state_dict(), io.BytesIO())
    assert adapter.runtime is None and adapter.session is None and not adapter.packed
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
