"""PF FP8 FWD/BWD against original BF16 math with one global diff policy."""
import os

import torch
import torch.distributed as dist

from weftmoe.fp8 import FP8Tensor, prepare_fp8_weights, quantize_fp8
from weftmoe.runtime import TensorRuntime, moe
from benchmarks.fp8_validation import global_diff, passes, limit_for


@torch.no_grad()
def reference(x, routes, gates, w1, w2, dy, lengths):
    # Unquantized BF16 inputs/weights/dY, independent Torch expert math.
    # No row/column FP8 conversion or WGMMA result is used as the oracle.
    y = torch.zeros_like(x, dtype=torch.float32)
    dx = torch.zeros_like(y)
    dg = torch.zeros_like(gates)
    dw1, dw2 = torch.zeros_like(w1), torch.zeros_like(w2)
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(routes == expert)
        if not rows.numel():
            continue
        with torch.enable_grad():
            inputs = [v.detach().requires_grad_() for v in
                      (x[rows], gates[rows, slots, None], w1[expert], w2[expert])]
            ex, eg, first, second = inputs
            a, b = torch.nn.functional.linear(ex, first).float().chunk(2, -1)
            h = (torch.nn.functional.silu(a) * b * eg).bfloat16()
            ye = torch.nn.functional.linear(h, second)
            gx, gg, g1, g2 = torch.autograd.grad(ye, inputs, dy[rows])
        y.index_add_(0, rows, ye.detach().float())
        dx.index_add_(0, rows, gx.float())
        dg[rows, slots] = gg.flatten()
        dw1[expert], dw2[expert] = g1, g2
    return y.bfloat16(), dx.bfloat16(), dg, dw1, dw2


def compare(name, actual, expected, rank):
    # Check dW expert-by-expert, so empty/large experts cannot hide another's error.
    pairs = zip(actual.unbind(), expected.unbind()) if name.endswith(('dw1', 'dw2')) else [(actual, expected)]
    values = [global_diff(a, b) for a, b in pairs]
    valid = all(passes(value, name) for value in values)
    print(rank, name, 'diffs', values, 'limit', limit_for(name), flush=True)
    agreed = torch.tensor(int(valid), device=actual.device)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    assert agreed.item(), (name, values, 'at least one rank failed BF16 global diff')


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
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
        expected = reference(all_x, route, all_gates, all_w1, all_w2, all_dy, lengths)
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
        # Prequantized X/dY must reproduce the same source packets. Exercise
        # mixed input formats across ranks, empty sources and reverse BWD order.
        qx, qdy = quantize_fp8(x.detach()), quantize_fp8(dy)
        baseline = runtime.forward_fp8(x.detach(), local_routes, gates.detach(),
                                       local_weights, weight_grad_dtype=torch.bfloat16)
        baseline_values = (baseline.output, *runtime.backward(baseline, dy))
        pending = []
        for mode in ("x", "dy", "both"):
            source = qx if mode != "dy" else x.detach()
            tape = runtime.forward_fp8(source, local_routes, gates.detach(), local_weights,
                                       synchronize=False, weight_grad_dtype=torch.bfloat16)
            pending.append((tape, mode))
        for tape, mode in reversed(pending):
            grads = runtime.backward_async(tape, qdy if mode != "x" else dy)
            for value, expected_value in zip((tape.output, *grads), baseline_values):
                torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        runtime.drain()
        # Fresh temporary sources leave ownership entirely to native leases.
        tape = runtime.forward_fp8(quantize_fp8(x.detach()) if rank % 2 else x.detach(),
                                   local_routes, gates.detach(), local_weights,
                                   synchronize=False, weight_grad_dtype=torch.bfloat16)
        grads = runtime.backward_async(tape, quantize_fp8(dy) if rank % 2 == 0 else dy)
        for value, expected_value in zip((tape.output, *grads), baseline_values):
            torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        runtime.drain()
        # Reject ambiguous bare FP8 and the weight block layout before launch.
        for invalid in (qx.data, quantize_fp8(x.detach(), block_rows=128)):
            try:
                runtime.forward_fp8(invalid, local_routes, gates.detach(), local_weights)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid prequantized source was accepted")
        del pending, tape, baseline, baseline_values, grads, qx, qdy
        print(rank, "FP8_PREQUANT_INPUT_PASS", "skew", skew, flush=True)
        print(rank, "FP8_PARITY_PASS", "skew", skew, runtime.native.usage(), flush=True)
    # Retain independent versions and output dtypes, then run reverse async
    # backward. BF16 and FP32 tapes must not share a mutable dtype decision.
    tapes = []
    for version, factor in enumerate((0.9, 1.1, 0.7), 1):
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
        tapes.append((tape, all_w1 * factor, all_w2, grad_dtype))
    for tape, reference_w1, reference_w2, grad_dtype in reversed(tapes):
        gradients = runtime.backward_async(tape, dy)
        assert gradients[2].dtype == gradients[3].dtype == grad_dtype
        expected = reference(all_x, route, all_gates, reference_w1, reference_w2, all_dy, lengths)
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
    # Imported-only checkpoints have no pre-quantization BF16 weights. Use
    # their represented values as BF16 weights for this import-specific test.
    expected = reference(all_x, route, all_gates, checkpoint.w1.dequantize().bfloat16(),
                         checkpoint.w2.dequantize().bfloat16(), all_dy, lengths)
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
            reference(all_x, route, all_gates, all_w1, all_w2, all_dy, lengths)[0][start:start + lengths[rank]], rank)
    inference_expected = result.output.clone()
    del result
    prequantized_result = inference.forward_fp8(quantize_fp8(x.detach()), local_routes,
                                                gates.detach(), inference_weights,
                                                retain_backward_state=False)
    torch.testing.assert_close(prequantized_result.output, inference_expected, rtol=0, atol=0)
    del inference_expected, prequantized_result
    inference.close()
    print(rank, "FP8_INFERENCE_PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
