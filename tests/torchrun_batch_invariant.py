"""Strict FWD batch/PF/DT parity and repeated BWD stability for tuning finalists.

Every PF invocation retains a real tape and consumes it with backward. The
canonical reference uses the same anchor tokens independently of surrounding rows.
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime
from weftmoe.fp8 import prepare_fp8_weights


def agree(ok, detail):
    valid = torch.tensor(int(ok), device='cuda')
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise AssertionError(detail)


def same(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def diff(a, b):
    a, b = a.double(), b.double()
    denominator = (a.square().sum() + b.square().sum()).clamp_min(1e-30)
    return float(((a-b).square().sum() / denominator).item())


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hidden', type=int, required=True)
    p.add_argument('--intermediate', type=int, required=True)
    p.add_argument('--experts', type=int, required=True)
    p.add_argument('--top-k', type=int, required=True)
    p.add_argument('--tokens', type=int, nargs='+', default=[17, 63, 64, 65, 129])
    p.add_argument('--candidate', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--skip-dt', action='store_true')
    args = p.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i, k = args.hidden, args.intermediate, args.top_k
    torch.manual_seed(20261107 + rank)
    w1 = torch.randn(args.experts//world, 2*i, h, device='cuda', dtype=torch.bfloat16) * .03
    w2 = torch.randn(args.experts//world, h, i, device='cuda', dtype=torch.bfloat16) * .03
    anchor_n = min(17, max(args.tokens))
    anchor = torch.randn(anchor_n, h, device='cuda', dtype=torch.bfloat16) * .03
    anchor_routes = ((torch.arange(anchor_n*k, device='cuda').reshape(anchor_n, k)+rank*3) % args.experts).int()
    anchor_gates = torch.rand(anchor_n, k, device='cuda')
    os.environ['WEFTMOE_TENSOR_EXTENSION'] = args.reference
    os.environ.pop('WEFTMOE_AUTOTUNE_TABLE', None)
    weights = prepare_fp8_weights(w1, w2, training=True)
    capacity = max(args.tokens)

    def runtime(extension, decode=False):
        os.environ['WEFTMOE_TENSOR_EXTENSION'] = extension
        return TensorRuntime(group=dist.group.WORLD, experts=args.experts, top_k=k,
                             hidden=h, intermediate=i, max_tokens=capacity,
                             max_rows=capacity*world*k+args.experts*128,
                             training=not decode, decode_only=decode)

    def run(rt, x, routes, gates, fp8, backward=True):
        tape = (rt.forward_fp8(x, routes, gates, weights, retain_backward_state=backward,
                               weight_grad_dtype=torch.bfloat16) if fp8
                else rt.forward(x, routes, gates, w1, w2))
        rt.drain()
        y = tape.output.clone()
        grads = rt.backward(tape, torch.full_like(x, .01)) if backward else None
        rt.drain()
        if grads is not None:
            agree(all(torch.isfinite(g).all().item() for g in grads), 'nonfinite BWD')
            grads = tuple(g.clone() for g in grads)
        agree(torch.isfinite(y).all().item(), 'nonfinite FWD')
        del tape
        return y, grads

    reference = runtime(args.reference)
    anchor_sizes = {min(n, anchor_n) for n in args.tokens}
    expected = {(fp8, size): run(reference, anchor[:size], anchor_routes[:size],
                                anchor_gates[:size], fp8)[0]
                for fp8 in (False, True) for size in sorted(anchor_sizes)}
    reference.close()
    torch.cuda.empty_cache()
    for decode in ([False] if args.skip_dt else [False, True]):
        rt = runtime(args.candidate, decode)
        for n in sorted(set(args.tokens)):
            size = min(n, anchor_n)
            torch.manual_seed(123 + n + rank)
            x = torch.randn(n, h, device='cuda', dtype=torch.bfloat16)*.03
            routes = ((torch.arange(n*k, device='cuda').reshape(n,k)+rank) % args.experts).int()
            gates = torch.rand(n,k,device='cuda')
            for at_end in (False, True):
                pos = slice(n-size, n) if at_end else slice(0,size)
                x[pos], routes[pos], gates[pos] = anchor[:size], anchor_routes[:size], anchor_gates[:size]
                for fp8 in (False, True):
                    y, grads = run(rt,x,routes,gates,fp8,not decode)
                    agree(same(y[pos],expected[fp8,size]), (n,decode,fp8,at_end,'batch/PF/DT mismatch'))
                    repeated, grads2 = run(rt,x,routes,gates,fp8,not decode)
                    agree(same(y,repeated), (n,decode,fp8,'FWD repeat mismatch'))
                    if grads is not None:
                        agree(all(diff(a,b) <= 1e-6 for a,b in zip(grads,grads2)), (n,fp8,'BWD repeat instability'))
                    del grads, grads2, y, repeated
                # Restore distracting rows before moving the anchors.
                torch.manual_seed(123 + n + rank)
                x = torch.randn(n,h,device='cuda',dtype=torch.bfloat16)*.03
                routes = ((torch.arange(n*k,device='cuda').reshape(n,k)+rank) % args.experts).int()
                gates = torch.rand(n,k,device='cuda')
            if rank == 0:
                print('BATCH_INVARIANT '+json.dumps(dict(tokens=n,decode=decode,passed=True)),flush=True)
        rt.close()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
