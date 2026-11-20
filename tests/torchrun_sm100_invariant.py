"""Bitwise cross-EP and route-order parity at identical BF16 MMA tile sizes."""

import argparse
import os

import deep_gemm as dg
import torch
import torch.distributed as dist

from weftmoe.blackwell_tuning import STRICT_BLOCK_M
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
    tiles = set()
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
        for window in getattr(rt.native, "windows", [rt.native]):
            buf = window.buffer
            assert buf.full_mma_tiles and buf.fixed_block_m == STRICT_BLOCK_M
            launched_tokens = window.dummy.shape[0] if hasattr(window, "dummy") else n
            tiles.add(
                dg.get_block_m_for_mega_moe(
                    buf.group.size(),
                    buf.num_experts,
                    buf.num_max_tokens_per_rank,
                    launched_tokens,
                    k,
                    "bf16xbf16",
                    buf.fixed_block_m,
                )
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
    assert len(tiles) == 1, tiles
    same = torch.equal(outputs[0].chunk(world)[rank].view(torch.int16), outputs[1].view(torch.int16))
    agreed = torch.tensor(int(same), device="cuda")
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    assert agreed.item(), f"EP1/EP{world} output bits differ"
    if rank == 0:
        print(f"BF16 EP1/EP{world} bitwise parity passed, MMA tile {next(iter(tiles))}x128x64")
    # Exercise the public policy switch independently of the timing harness.
    opts = dict(group=single, experts=experts, top_k=k, hidden=h, intermediate=i,
                max_tokens=total, max_rows=total * k, training=False)
    for requested in (None, 32):
        relaxed = TensorRuntime(**opts, batch_invariant=False, blackwell_block_m=requested)
        assert relaxed.native.buffer.fixed_block_m == (requested or 0)
        assert not relaxed.native.buffer.full_mma_tiles
        result = relaxed.inference_eager(x, ids, gates, first, second)
        assert torch.isfinite(result).all()
        relaxed.close()
    for group, override, reason in ((single, 32, "Noncanonical"),
                                    (dist.group.WORLD, 32 if rank else None, "incompatible")):
        try:
            TensorRuntime(**dict(opts, group=group), blackwell_block_m=override)
        except ValueError as exc:
            assert reason in str(exc)
        else:
            raise AssertionError("Strict mode accepted an inconsistent M policy")
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
