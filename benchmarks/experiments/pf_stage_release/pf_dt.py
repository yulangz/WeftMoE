"""Compare PF and DT on identical inference inputs/weights, without timing."""
import json
import os
import torch
import torch.distributed as dist
from weftmoe.runtime import TensorRuntime
from weftmoe.fp8 import prepare_fp8_weights, quantize_fp8
from benchmarks.fp8_validation import global_diff, passes

@torch.no_grad()
def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i = int(os.environ['TEST_HIDDEN']), int(os.environ['TEST_INTERMEDIATE'])
    experts, topk = 2 * world, 2
    torch.manual_seed(1729)
    w1 = torch.randn(experts, 2*i, h, device='cuda', dtype=torch.bfloat16) * .03
    w2 = torch.randn(experts, h, i, device='cuda', dtype=torch.bfloat16) * .03
    factors = lambda n: 2. ** ((torch.arange(n, device='cuda') // 128) % 3 - 1)
    w1 *= factors(2*i)[:, None] * factors(h)[None, :]
    w2 *= factors(h)[:, None] / factors(i)[None, :]
    w1, w2 = w1.chunk(world)[rank].contiguous(), w2.chunk(world)[rank].contiguous()
    weights = prepare_fp8_weights(w1, w2, version=0, training=False)
    cases = []
    torch.manual_seed(901 + rank)
    for size in (1, 16, 65, 129, 512, -1):
        n = (0 if rank == 0 else 17 + rank) if size == -1 else size
        x = (torch.randn(n, h, device='cuda', dtype=torch.bfloat16) * .03 * factors(h)).bfloat16()
        gates = torch.rand(n, topk, device='cuda')
        for skew in (False, True):
            routes = ((torch.arange(n*topk, device='cuda').reshape(n, topk) + rank*3) % experts).int()
            if skew:
                routes.fill_(experts - 1)
            cases.append((size, skew, x, routes, gates))
    saved = {}
    for decode in (False, True):
        runtime = TensorRuntime(group=dist.group.WORLD, experts=experts, top_k=topk,
            hidden=h, intermediate=i, max_tokens=512, max_rows=512*world*topk+experts*128,
            training=False, decode_only=decode, fp8_workspace=False)
        for idx, (size, skew, x, routes, gates) in enumerate(cases):
            for fmt in ('bf16', 'fp8', 'prequant'):
                if fmt == 'bf16':
                    tape = runtime.forward(x, routes, gates, w1, w2)
                else:
                    source = quantize_fp8(x) if fmt == 'prequant' else x
                    tape = runtime.forward_fp8(source, routes, gates, weights, retain_backward_state=False)
                runtime.drain()
                out = tape.output.clone()
                torch.cuda.synchronize()
                del tape
                if not decode:
                    saved[idx, fmt] = out
                    continue
                pf = saved[idx, fmt]
                diff = global_diff(out, pf)
                rec = dict(rank=rank, size=size, skew=skew, format=fmt, diff=diff,
                    bitwise_equal=torch.equal(out, pf),
                    mismatch_fraction=float((out != pf).float().mean()) if out.numel() else 0.,
                    max_abs=float((out.float()-pf.float()).abs().max()) if out.numel() else 0.)
                print('PF_DT ' + json.dumps(rec), flush=True)
                valid = torch.tensor(int(passes(diff)), device='cuda')
                dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                assert valid.item(), rec
        runtime.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
