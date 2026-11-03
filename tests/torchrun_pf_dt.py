"""Require bitwise equality between training PF (retained, consumed tape) and DT."""
import json
import os
import torch
import torch.distributed as dist
from weftmoe.runtime import TensorRuntime
from weftmoe.fp8 import prepare_fp8_weights, quantize_fp8
from benchmarks.fp8_validation import global_diff

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
    weights = prepare_fp8_weights(w1, w2, version=0, training=True)
    cases = []
    torch.manual_seed(901 + rank)
    for size in (1, 2, 4, 8, 16, 32, 63, 64, 65, 127, 128, 129, 256, 512, -1):
        n = (0 if rank == 0 else 17 + rank) if size == -1 else size
        x = (torch.randn(n, h, device='cuda', dtype=torch.bfloat16) * .03 * factors(h)).bfloat16()
        gates = torch.rand(n, topk, device='cuda')
        for skew in (False, True):
            routes = ((torch.arange(n*topk, device='cuda').reshape(n, topk) + rank*3) % experts).int()
            if skew:
                routes[:] = torch.arange(experts - topk, experts, device="cuda", dtype=torch.int32)
            cases.append((size, skew, x, routes, gates))
    compact = os.environ.get("TEST_PF_COMPACT", "0") == "1"
    formats = ('fp8', 'prequant') if compact else ('bf16', 'fp8', 'prequant')
    saved = {}
    for decode in (False, True):
        runtime = TensorRuntime(group=dist.group.WORLD, experts=experts, top_k=topk,
            hidden=h, intermediate=i, max_tokens=512, max_rows=512*world*topk+experts*128,
            training=not decode, decode_only=decode, fp8_workspace=compact)
        for idx, (size, skew, x, routes, gates) in enumerate(cases):
            for fmt in formats:
                if rank == 0 and not decode:
                    print('TRAIN', size, skew, fmt, flush=True)
                if fmt == 'bf16':
                    tape = runtime.forward(x, routes, gates, w1, w2)
                else:
                    source = quantize_fp8(x) if fmt == 'prequant' else x
                    tape = runtime.forward_fp8(source, routes, gates, weights, retain_backward_state=not decode, weight_grad_dtype=torch.bfloat16)
                runtime.drain()
                out = tape.output.clone()
                if not decode:
                    # A real training tape must own caches and remain consumable.
                    cache_bytes = torch.tensor(tape.tensor_bytes, device=x.device, dtype=torch.int64)
                    dist.all_reduce(cache_bytes)
                    assert cache_bytes.item() > 0
                    gradients = runtime.backward(tape, torch.full_like(x, .01))
                    runtime.drain()
                    valid = torch.tensor(int(all(torch.isfinite(g).all().item() for g in gradients)), device=x.device)
                    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                    assert valid.item(), (size, fmt, "nonfinite training gradients")
                    del gradients
                torch.cuda.synchronize()
                del tape
                if not decode:
                    saved[idx, fmt] = out
                    continue
                pf = saved[idx, fmt]
                diff = global_diff(out, pf)
                rec = dict(rank=rank, size=size, skew=skew, format=fmt, compact_workspace=compact, diff=diff,
                    bitwise_equal=torch.equal(out.view(torch.int16), pf.view(torch.int16)),
                    mismatch_fraction=float((out != pf).float().mean()) if out.numel() else 0.,
                    max_abs=float((out.float()-pf.float()).abs().max()) if out.numel() else 0.)
                print('PF_DT ' + json.dumps(rec), flush=True)
                valid = torch.tensor(int(torch.equal(out.view(torch.int16), pf.view(torch.int16))), device='cuda')
                dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                assert valid.item(), rec
        runtime.close()
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
