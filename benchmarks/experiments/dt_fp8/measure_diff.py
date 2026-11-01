"""Untimed EP8 FP8/BF16 comparison using DeepGEMM's normalized squared error.

DIFF_MODE=moe: full DT FWD with 256 experts and production top-k.
DIFF_MODE=expert: PF training, one isolated expert per rank, BF16 dW output.
All operands start in BF16; no elementwise allclose or relative-error filter.
Run with torchrun and the same native/profile environment as run.py.
"""
import json
import math
import os

import torch
import torch.distributed as dist

from weftmoe.fp8 import prepare_fp8_weights, quantize_fp8
from weftmoe.runtime import TensorRuntime
from benchmarks.fp8_validation import limit_for, passes, POLICY


def metrics(sums):
    error2, actual2, reference2 = sums
    denominator = actual2 + reference2
    diff = error2 / denominator if denominator else 0.0
    return dict(diff=diff,
                relative_l2=math.sqrt(error2 / reference2) if reference2 else (0.0 if error2 == 0 else None),
                norm_ratio=math.sqrt(actual2 / reference2) if reference2 else None,
                sqnr_db=10 * math.log10(reference2 / error2) if error2 and reference2 else None,
                sqnr_from_diff_db=-10 * math.log10(2 * diff) if diff else None)


def compare(actual, target, **labels):
    name = labels["tensor"]
    limit = limit_for(name)
    assert actual.shape == target.shape, (actual.shape, target.shape)
    valid = torch.tensor(int(torch.isfinite(actual).all() and torch.isfinite(target).all()),
                         device=actual.device)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    assert valid.item(), ('nonfinite tensor', labels)
    a, b = actual.detach().double(), target.detach().double()
    sums = torch.stack(((a - b).square().sum(), a.square().sum(), b.square().sum()))
    peers = [torch.empty_like(sums) for _ in range(dist.get_world_size())]
    dist.all_gather(peers, sums)
    per_rank = [metrics(p.tolist()) for p in peers]
    combined = metrics(torch.stack(peers).sum(0).tolist())
    result = dict(**labels, actual_dtype=str(actual.dtype), reference_dtype=str(target.dtype),
                  limit=limit, global_metrics=combined, per_rank=per_rank,
                  max_rank_diff=max(p['diff'] for p in per_rank),
                  pass_global=passes(combined['diff'], name),
                  pass_every_rank=all(passes(p['diff'], name) for p in per_rank))
    if dist.get_rank() == 0:
        print('DIFF ' + json.dumps(result, allow_nan=False), flush=True)
    assert result['pass_every_rank'], (labels, result['max_rank_diff'], limit)
    return result


def materialize(tape, gradients):
    return [v.detach().clone() if isinstance(v, torch.Tensor) else torch.stack(v).detach().clone()
            for v in (tape.output, *gradients)]


def expert_reference(x, gates, w1, w2, dy):
    with torch.enable_grad():
        values = [t.detach().requires_grad_() for t in (x, gates, w1[0], w2[0])]
        ex, eg, first, second = values
        a, b = torch.nn.functional.linear(ex, first).float().chunk(2, -1)
        h = (torch.nn.functional.silu(a) * b * eg).bfloat16()
        y = torch.nn.functional.linear(h, second)
        dx, dg, dw1, dw2 = torch.autograd.grad(y, values, dy)
    return [y.detach(), dx, dg, dw1.unsqueeze(0), dw2.unsqueeze(0)]


def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i = int(os.environ['TEST_HIDDEN']), int(os.environ['TEST_INTERMEDIATE'])
    mode = os.getenv('DIFF_MODE', 'moe')
    training = mode == 'expert'
    experts = world if training else 256
    topk = 1 if training else int(os.environ['BENCH_TOPK'])
    sizes = [16, 129, 512] if training else [1, 16, 32, 64, 128, 256, 512]
    seeds = [901, 1300, 1700]
    kwargs = dict(group=dist.group.WORLD, experts=experts, top_k=topk,
                  hidden=h, intermediate=i, max_tokens=max(sizes),
                  max_rows=max(sizes)*world*topk+experts*128,
                  training=training, decode_only=not training, max_saved=2)
    bf = TensorRuntime(**kwargs)
    fp = TensorRuntime(**kwargs, fp8_workspace=True)
    for seed in seeds:
        torch.manual_seed(seed * 31 + rank)
        w1 = torch.randn(experts//world, 2*i, h, device='cuda', dtype=torch.bfloat16)*.03
        w2 = torch.randn(experts//world, h, i, device='cuda', dtype=torch.bfloat16)*.03
        weights = prepare_fp8_weights(w1, w2, version=seed, training=training)
        for n in sizes:
            x = torch.randn(n, h, device='cuda', dtype=torch.bfloat16)*.03
            dy = torch.randn_like(x)*.03 if training else None
            gates = torch.rand(n, topk, device='cuda')
            if not training:
                gates /= gates.sum(1, keepdim=True)
            for routing in (['isolated-local-expert'] if training else ['uniform', 'hotspot']):
                if training:
                    routes = torch.full((n, 1), rank, dtype=torch.int32, device='cuda')
                elif routing == 'uniform':
                    routes = torch.rand(n, experts, device='cuda').topk(topk, dim=1).indices.int().contiguous()
                else:
                    # Distinct top-k experts, all on rank 0; no duplicate routes.
                    routes = torch.arange(topk, device='cuda', dtype=torch.int32).expand(n, -1).contiguous()
                labels = dict(mode=mode, hidden=h, intermediate=i, ep=world, experts=experts,
                              topk=topk, seed=seed, tokens=n, routing=routing,
                              input_format='bf16-online-quant', reference='production-bf16')
                tb = bf.forward(x, routes, gates, w1, w2)
                if training:
                    baseline = materialize(tb, bf.backward(tb, dy))
                    del tb
                    tf = fp.forward_fp8(x, routes, gates, weights, weight_grad_dtype=torch.bfloat16)
                    measured = materialize(tf, fp.backward(tf, dy))
                    del tf
                    names = ['output', 'dx', 'dprob', 'dw1', 'dw2']
                    for name, value, target in zip(names, measured, baseline):
                        compare(value, target, tensor=name, **labels)
                    # Independent BF16 math reference cross-check, without FP8.
                    independent = expert_reference(x, gates, w1, w2, dy)
                    for name, value, target in zip(names, baseline, independent):
                        compare(value, target, tensor=name,
                                **dict(labels, mode='bf16-reference-check', reference='torch-bf16-autograd'))
                    del measured, baseline, independent
                else:
                    baseline = tb.output.clone()
                    del tb
                    tf = fp.forward_fp8(x, routes, gates, weights, retain_backward_state=False)
                    measured = tf.output.clone()
                    del tf
                    compare(measured, baseline, tensor='output', **labels)
                    tq = fp.forward_fp8(quantize_fp8(x), routes, gates, weights, retain_backward_state=False)
                    same = torch.tensor(int(torch.equal(measured, tq.output)), device='cuda')
                    dist.all_reduce(same, op=dist.ReduceOp.MIN)
                    assert same.item(), 'online/prequantized mismatch'
                    del tq, measured, baseline
        del weights, w1, w2
    bf.close()
    fp.close()
    if rank == 0:
        print('COMPLETE ' + mode, flush=True)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
