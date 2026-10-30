"""PF FP8 forward, Dgrad and Wgrad against six standalone FP8 GEMMs.

Uses real EP dispatch/combine, including unequal token counts and empty
experts. Run using torchrun --standalone --nproc-per-node=2.
For GIN set TEST_FP8_CANONICAL=1 to model the window-ordered training tape.
"""
import os

import torch
import torch.distributed as dist

from weftmoe.fp8 import FP8Tensor, fp8_gemm, prepare_fp8_weights, quantize_fp8
from weftmoe.runtime import TensorRuntime, moe


@torch.no_grad()
def reference(x, routes, gates, weights, dy, lengths):
    canonical = os.environ.get("TEST_FP8_CANONICAL", "0") == "1"
    y = torch.zeros_like(x, dtype=torch.float32)
    dx = torch.zeros_like(y)
    dg = torch.zeros_like(gates)
    dw1 = torch.zeros(weights.w1.shape, device=x.device, dtype=torch.float32)
    dw2 = torch.zeros(weights.w2.shape, device=x.device, dtype=torch.float32)
    for expert in range(weights.w1.shape[0]):
        rows, slots = torch.where(routes == expert)
        if not rows.numel():
            continue
        # Independent model of the persistent ring: round-robin over source
        # ranks, preserving source-local route order and skipping exhausted ranks.
        sources, begin = [], 0
        for length in lengths:
            sources.append(torch.where((rows >= begin) & (rows < begin + length))[0].tolist())
            begin += length
        permutation = torch.tensor([source[i] for i in range(max(map(len, sources)))
                                    for source in sources if i < len(source)], device=x.device)
        if canonical:
            # GIN visits token windows, then source groups starting with the
            # destination's node. Within a group it concatenates source spans.
            local_world = int(os.environ["LOCAL_WORLD_SIZE"])
            group_size = int(os.environ.get("TEST_FP8_SOURCE_GROUP", local_world))
            groups = len(lengths) // group_size
            owner = expert // (weights.w1.shape[0] // len(lengths))
            first_group = (owner // local_world) * local_world // group_size
            window = int(os.environ.get("WEFTMOE_STATIC_RING_TOKENS", "4096"))
            order = []
            row_ids = rows.tolist()
            starts = [sum(lengths[:rank]) for rank in range(len(lengths))]
            for token_begin in range(0, max(lengths), window):
                for offset in range(groups):
                    group = (first_group + offset) % groups
                    for rank in range(group * group_size, (group + 1) * group_size):
                        order.extend(index for index in sources[rank]
                                     if token_begin <= row_ids[index] - starts[rank] < token_begin + window)
            assert len(order) == rows.numel()
            permutation = torch.tensor(order, device=x.device)
        rows, slots = rows[permutation], slots[permutation]
        xe, de = x[rows].contiguous(), dy[rows].contiguous()
        g = gates[rows, slots, None]
        qx, qd = quantize_fp8(xe), quantize_fp8(de)
        ab = fp8_gemm(qx, weights.w1.matrix(expert))
        if os.environ.get("TEST_FP8_AB", "1") == "1":
            ab = quantize_fp8(ab).dequantize()
        a, b = ab.float().chunk(2, -1)
        sig = torch.sigmoid(a)
        u = (a * sig) * b
        z = (u * g).bfloat16()
        ye = fp8_gemm(quantize_fp8(z), weights.w2.matrix(expert))
        y.index_add_(0, rows, ye.float())
        dz = fp8_gemm(qd, weights.w2_t.matrix(expert)).float()
        dg[rows, slots] = (u * dz).sum(-1)
        du = g * dz
        dab = torch.cat((du * b * (sig * (1 + a * (1 - sig))), du * (a * sig)), -1).bfloat16()
        dxe = fp8_gemm(quantize_fp8(dab), weights.w1_t.matrix(expert))
        dx.index_add_(0, rows, dxe.float())
        source_cache = os.environ.get("TEST_FP8_SOURCE_CACHE", "1") == "1"
        column_x = qx.dequantize().contiguous() if source_cache else xe
        column_dy = qd.dequantize().contiguous() if source_cache else de
        dw1[expert] = fp8_gemm(quantize_fp8(dab, transpose=True),
                              quantize_fp8(column_x, transpose=True), dtype=torch.float32)
        dw2[expert] = fp8_gemm(quantize_fp8(column_dy, transpose=True),
                              quantize_fp8(z, transpose=True), dtype=torch.float32)
    return y.bfloat16(), dx.bfloat16(), dg, dw1, dw2


def compare(name, actual, expected, rank):
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    if not actual.numel():
        return
    expected = expected.to(actual.dtype)
    actual, expected = actual.float(), expected.float()
    delta = actual - expected
    relative = delta.norm() / expected.norm().clamp_min(1e-20)
    print(rank, name, "relative_l2", relative.item(), "max_abs", delta.abs().max().item(), flush=True)
    # The standalone N64 and persistent N128/N256 WGMMA paths share K128
    # scaling. SwiGLU uses CUDA exp/rcp versus Torch sigmoid, so allow BF16
    # boundary changes, while catching wrong scales, permutations and routing.
    assert relative < 0.01, (name, relative.item())
    assert torch.isfinite(actual).all()
    if not expected.any():
        assert not actual.any(), name


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    h = int(os.environ.get("TEST_HIDDEN", 2048))
    i = int(os.environ.get("TEST_INTERMEDIATE", 512))
    experts, top_k = int(os.environ.get("TEST_EXPERTS", max(4, 2 * world))), 2
    assert experts % world == 0
    base_tokens = int(os.environ.get("TEST_TOKENS", 65))
    lengths = [max(1, base_tokens - 7 * r) for r in range(world)]
    if os.environ.get("TEST_ZERO_RANK") == "1":
        lengths[-1] = 0
    start = sum(lengths[:rank])
    torch.manual_seed(1729)
    all_x = torch.randn(sum(lengths), h, device="cuda", dtype=torch.bfloat16) * 0.03
    all_dy = torch.randn_like(all_x) * 0.03
    all_w1 = torch.randn(experts, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.03
    all_w2 = torch.randn(experts, h, i, device="cuda", dtype=torch.bfloat16) * 0.03
    # Independent scales across both axes, including gate/up halves. Uniform
    # random matrices otherwise tend to choose the same scale in every block.
    def block_factors(n):
        return 2.0 ** ((torch.arange(n, device="cuda") // 128) % 3 - 1)
    all_w1.mul_(block_factors(2 * i)[:, None] * block_factors(h)[None, :])
    all_w2.mul_(block_factors(h)[:, None] / block_factors(i)[None, :])
    all_x.mul_(block_factors(h))
    all_dy.mul_(block_factors(h).flip(0))
    if os.environ.get("TEST_ROW_SCALES") == "1":
        # Force different row scales, source-rank imbalance and underflow-prone
        # columns. Constant-scale Gaussian inputs can hide broken row->col casts.
        row_factors = 2.0 ** ((torch.arange(sum(lengths), device="cuda") % 9) - 4)
        all_x.mul_(row_factors[:, None])
        all_dy.mul_(row_factors.flip(0)[:, None])
        all_x[:, 1::128] *= 2.0 ** -12
        all_dy[:, 3::128] *= 2.0 ** -12
    all_gates = torch.rand(sum(lengths), top_k, device="cuda")
    all_gates[::11, 0] = 0
    weights = prepare_fp8_weights(all_w1, all_w2, version=0)
    w1 = all_w1.chunk(world)[rank].clone().requires_grad_()
    w2 = all_w2.chunk(world)[rank].clone().requires_grad_()
    local_weights = prepare_fp8_weights(w1, w2, version=0)
    runtime = TensorRuntime(group=dist.group.WORLD, experts=experts, top_k=top_k,
        hidden=h, intermediate=i, max_tokens=base_tokens,
        max_rows=max(2048, sum(lengths) * top_k + experts * 128), training=True, max_saved=4,
        fp8_workspace=os.environ.get("TEST_FP8_COMPACT", "1") == "1")
    for skew in (False, True):
        route = torch.arange(sum(lengths) * top_k, device="cuda", dtype=torch.int32).reshape(-1, top_k)
        route = (route % (top_k if skew else experts)).contiguous()
        x = all_x[start:start + lengths[rank]].clone().requires_grad_()
        gates = all_gates[start:start + lengths[rank]].clone().requires_grad_()
        local_routes = route[start:start + lengths[rank]].contiguous()
        dy = all_dy[start:start + lengths[rank]].contiguous()
        w1.grad = w2.grad = None
        if runtime.fp8_workspace and not skew:
            try:
                runtime.forward(x, local_routes, gates, w1, w2)
            except ValueError as error:
                assert "Compact FP8 workspace" in str(error)
            else:
                raise AssertionError("compact workspace accepted BF16 inputs")
        bf16_before = None
        if not runtime.fp8_workspace:
            probe = runtime.forward(x.detach(), local_routes, gates.detach(), w1.detach(), w2.detach())
            bf16_before = (probe.output, *runtime.backward(probe, dy))
        print(rank, "FP8_BEGIN", "skew", skew, flush=True)
        y = moe(runtime, x, local_routes, gates, w1, w2, fp8_weights=local_weights)
        print(rank, "FP8_FORWARD_DONE", flush=True)
        y.backward(dy)
        print(rank, "FP8_BACKWARD_DONE", flush=True)
        expected = reference(all_x, route, all_gates, weights, all_dy, lengths)
        actual = y, x.grad, gates.grad, w1.grad, w2.grad
        for name, value, target in zip(("y", "dx", "dg", "dw1", "dw2"), actual, expected):
            target = (target.chunk(world)[rank] if name.startswith("dw")
                      else target[start:start + lengths[rank]])
            compare(name, value, target, rank)
        del y
        if bf16_before is not None:
            probe = runtime.forward(x.detach(), local_routes, gates.detach(), w1.detach(), w2.detach())
            bf16_after = (probe.output, *runtime.backward(probe, dy))
            for before, after in zip(bf16_before, bf16_after):
                torch.testing.assert_close(before, after, rtol=0, atol=0)
            print(rank, "MIXED_BF16_FP8_POOL_PASS", flush=True)
            del probe, bf16_before, bf16_after
        print(rank, "FP8_PARITY_PASS", "skew", skew, runtime.native.usage(), flush=True)
    # Retain independent versions and output dtypes, then run reverse async
    # backward. BF16 and FP32 tapes must not share a mutable dtype decision.
    tapes = []
    for version, factor in enumerate((0.9, 1.1, 0.7), 1):
        global_version = prepare_fp8_weights(all_w1 * factor, all_w2, version=version)
        local_version = prepare_fp8_weights(w1.detach() * factor, w2, version=version)
        grad_dtype = torch.bfloat16 if version == 3 else torch.float32
        tape = runtime.forward_fp8(x.detach(), local_routes, gates.detach(), local_version,
                                   synchronize=False, weight_grad_dtype=grad_dtype)
        if os.environ.get("TEST_FP8_SOURCE_X") == "1":
            assert tape.saved_x_bytes == 0, "source FP8 must not allocate BF16 X"
        # An empty source owning no routed experts can have an empty BF16
        # checkpoint on the GIN path (compact FP8 pools still allocate scales).
        assert tape.tensor_bytes > 0 or lengths[rank] == 0
        assert runtime.native.usage()["last_forward_tensor_bytes"] == tape.tensor_bytes
        tapes.append((tape, global_version, grad_dtype))
    for tape, version_weights, grad_dtype in reversed(tapes):
        gradients = runtime.backward_async(tape, dy)
        assert gradients[2].dtype == gradients[3].dtype == grad_dtype
        expected = reference(all_x, route, all_gates, version_weights, all_dy, lengths)
        actual = (tape.output, *gradients)
        for name, value, target in zip(("y", "dx", "dg", "dw1", "dw2"), actual, expected):
            target = (target.chunk(world)[rank] if name.startswith("dw")
                      else target[start:start + lengths[rank]])
            compare("async_" + name, value, target, rank)
    runtime.drain()
    assert runtime.native.usage()["notify_slots_in_use"] == 0
    print(rank, "FP8_MULTIVERSION_ASYNC_PASS", flush=True)
    # Reusing a native prepared lease must still reject mutated tensor versions.
    local_version.w1.scales.mul_(1.0)
    try:
        runtime.forward_fp8(x.detach(), local_routes, gates.detach(), local_version)
    except RuntimeError as error:
        assert "modified" in str(error)
    else:
        raise AssertionError("cached FP8 lease accepted mutated scales")
    print(rank, "FP8_CACHED_VERSION_CHECK_PASS", flush=True)


    # Imported checkpoint scales need not be powers of two; never silently
    # quantize these weights again. Exercise independent forward/transpose use.
    def imported(q, factor, local=False):
        data, scales = q.data, q.scales * factor
        if local:
            data, scales = data.chunk(world)[rank], scales.chunk(world)[rank]
        return FP8Tensor(data, scales, tuple(data.shape), 128)
    checkpoint = prepare_fp8_weights(imported(weights.w1, 1.125), imported(weights.w2, 0.875))
    local_checkpoint = prepare_fp8_weights(imported(weights.w1, 1.125, True),
                                          imported(weights.w2, 0.875, True))
    tape = runtime.forward_fp8(x.detach(), local_routes, gates.detach(), local_checkpoint)
    actual = (tape.output, *runtime.backward(tape, dy))
    expected = reference(all_x, route, all_gates, checkpoint, all_dy, lengths)
    for name, value, target in zip(("y", "dx", "dg", "dw1", "dw2"), actual, expected):
        target = (target.chunk(world)[rank] if name.startswith("dw")
                  else target[start:start + lengths[rank]])
        compare("checkpoint_" + name, value, target, rank)
    print(rank, "FP8_CHECKPOINT_PASS", flush=True)

    # Real optimizer updates with FP32 master parameters must rebuild the cache
    # and preserve FP32 gradients through the custom autograd function.
    p1, p2 = w1.detach().float().requires_grad_(), w2.detach().float().requires_grad_()
    optimizer = torch.optim.SGD((p1, p2), lr=0.01)
    for step in range(2):
        prepared = prepare_fp8_weights(p1, p2, version=step)
        y = moe(runtime, x.detach(), local_routes, gates.detach(), p1, p2,
                fp8_weights=prepared)
        y.backward(dy)
        assert p1.grad.dtype == p2.grad.dtype == torch.float32
        assert torch.isfinite(p1.grad).all() and torch.isfinite(p2.grad).all()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        try:
            prepared.validate_parameters(p1, p2)
        except RuntimeError as error:
            assert "stale" in str(error)
        else:
            raise AssertionError("optimizer update must invalidate the old FP8 cache")
    print(rank, "FP8_OPTIMIZER_PASS", flush=True)
    runtime.close()

    # The same prepared forward layout is usable without training tape storage.
    inference = TensorRuntime(group=dist.group.WORLD, experts=experts, top_k=top_k,
        hidden=h, intermediate=i, max_tokens=base_tokens,
        max_rows=max(2048, sum(lengths) * top_k + experts * 128), training=False,
        fp8_workspace=os.environ.get("TEST_FP8_COMPACT", "1") == "1")
    inference_weights = prepare_fp8_weights(w1, w2, training=False)
    result = inference.forward_fp8(x.detach(), local_routes, gates.detach(), inference_weights,
                                    retain_backward_state=False)
    compare("inference", result.output,
            reference(all_x, route, all_gates, weights, all_dy, lengths)[0][start:start + lengths[rank]], rank)
    del result
    inference.close()
    print(rank, "FP8_INFERENCE_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
