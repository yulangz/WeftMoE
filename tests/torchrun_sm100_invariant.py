"""Bitwise cross-EP and route-order parity with dynamic BF16 M tiles."""

import argparse
import os

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-tokens", type=int)
    parser.add_argument("--training-tokens", type=int)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world in (8, 16, 32)
    experts, k, h, i = (256 if world == 32 else 64), 8, 2048, 512
    total = args.total_tokens or (128 if world == 8 else 32)
    assert total >= world and total % world == 0
    single, _ = dist.new_subgroups(group_size=1)
    torch.manual_seed(27)
    first = torch.randn(experts, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.03
    second = torch.randn(experts, h, i, device="cuda", dtype=torch.bfloat16) * 0.03
    x = torch.randn(total, h, device="cuda", dtype=torch.bfloat16) * 0.03
    ids = torch.rand(total, experts, device="cuda").topk(k)[1].int()
    gates = torch.rand(total, k, device="cuda")
    # All eight slots collide for one token, and another token has no routes.
    ids[0] = 3
    ids[1] = -1
    counts = [total, total // world]
    outputs = []
    for group, n in zip((single, dist.group.WORLD), counts):
        rt = TensorRuntime(
            group=group,
            experts=experts,
            top_k=k,
            hidden=h,
            intermediate=i,
            max_tokens=n,
            max_rows=total * k,
            training=False,
        )
        rows = slice(0, total) if group.size() == 1 else slice(rank * n, (rank + 1) * n)
        w1, w2 = [w.chunk(group.size())[group.rank()] for w in (first, second)]
        local = (x[rows], ids[rows], gates[rows])
        y = rt.inference_eager(*local, w1, w2)
        torch.testing.assert_close(rt.inference_eager(*local, w1, w2), y, atol=0, rtol=0)
        permutation = torch.arange(n - 1, -1, -1, device="cuda")
        z = rt.inference_eager(*(v[permutation] for v in local), w1, w2)
        assert torch.equal(y.view(torch.int16)[permutation], z.view(torch.int16))
        short = n // 2
        prefix = rt.inference_eager(*(v[:short] for v in local), w1, w2)
        assert torch.equal(prefix.view(torch.int16), y[:short].view(torch.int16))
        outputs.append(y.clone())
        rt.close()
    same = torch.equal(outputs[0].chunk(world)[rank].view(torch.int16), outputs[1].view(torch.int16))
    agreed = torch.tensor(int(same), device="cuda")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    assert agreed.item(), f"EP1/EP{world} output bits differ"
    if rank == 0:
        print(f"BF16 EP1/EP{world} bitwise parity passed")
    if args.forward_only:
        dist.destroy_process_group()
        return
    # Train twice on identical routes interleaved with a different live tape.
    # A zero-token source and a completely idle expert both participate.
    n = 0 if rank == 0 else 3 + rank
    local_x, local_ids, local_gates = x[:n], ids[:n].clone(), gates[:n]
    capacity = max(32, world + 2)
    if args.training_tokens is not None:
        capacity = args.training_tokens
        assert capacity >= world + 2
        n = 0 if rank == 0 else capacity - rank
        torch.manual_seed(53 + rank)
        local_x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16) * 0.03
        local_ids = torch.rand(n, experts, device="cuda").topk(k)[1].int()
        local_gates = torch.rand(n, k, device="cuda")
        local_ids[:1] = 3
        local_ids[1:2] = -1
    local_ids.clamp_max_(experts - 2)
    runtime_options = dict(
        group=dist.group.WORLD,
        experts=experts,
        top_k=k,
        hidden=h,
        intermediate=i,
        max_tokens=capacity,
        max_rows=world * capacity * k,
        max_saved=2,
    )
    rt = TensorRuntime(training=True, **runtime_options)
    local_w = (first.chunk(world)[rank], second.chunk(world)[rank])
    dy = torch.randn_like(local_x)
    expected = None
    for _ in range(2):
        t = rt.forward_async(local_x, local_ids, local_gates, *local_w)
        other = rt.forward_async(local_x, local_ids.roll(1, dims=1), local_gates, *local_w)
        rt.backward_async(other, dy)
        result = (t.output, *rt.backward_async(t, dy))
        if expected is None:
            expected = tuple(v.clone() for v in result)
        else:
            for actual, ref in zip(result, expected):
                assert torch.equal(actual.view(torch.uint8), ref.view(torch.uint8))
    rt.close()
    inference = TensorRuntime(training=False, **runtime_options)
    y = inference.inference_eager(local_x, local_ids, local_gates, *local_w)
    assert torch.equal(y.view(torch.int16), expected[0].view(torch.int16))
    inference.close()
    if rank == 0:
        print(f"BF16 EP{world} FWD and all four BWD tensors repeated bitwise, capacity {capacity}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
