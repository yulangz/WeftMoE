"""Check each NVFP4 gradient against independently rounded FP32 GEMMs."""

import argparse
import json
import os
import torch
import torch.distributed as dist

from torchrun_sm100_nvfp4 import round_nvfp4, reference
from weftmoe.nvfp4 import NVFP4Scales, prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime, moe


def check(actual, expected, name, tolerance=0.0006):
    diff = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    dist.all_reduce(diff, op=dist.ReduceOp.MAX)
    assert diff.item() < tolerance, (name, diff.item())
    return diff.item()


def gather(t):
    result = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(result, t)
    return torch.cat(result)


def check_scale_midpoint():
    from weftmoe.nvfp4_backward import quantize

    # The E4M3 value 0.78125 lies halfway between 0.75 and 0.8125.
    # Both orientations have this block maximum divided by (6 * 0.025).
    x = torch.full((128, 128), 0.1171875, device="cuda", dtype=torch.bfloat16)
    x[::4, 1::2] = 0.016845703125
    ends = torch.tensor([128], device=x.device, dtype=torch.int32)
    row, col = quantize(x, ends, 0.025, row=True)
    expected_sf = torch.tensor(0.75, device=x.device, dtype=torch.float8_e4m3fn).view(
        torch.uint8
    )
    codes = torch.full_like(x, 7, dtype=torch.uint8)  # E2M1 6.0
    codes[::4, 1::2] = 2  # E2M1 1.0
    for (packed, sf), expected in ((row, codes), (col, codes.T)):
        assert (sf == expected_sf).all()
        assert torch.equal(packed, expected[:, ::2] | (expected[:, 1::2] << 4))


def checkpoint_view(runtime, tape, n, k):
    """Decode source pointers independently, then concatenate windows per expert."""
    windows = getattr(runtime, "windows", [runtime])
    states = (
        [t.state for t in tape.state] if hasattr(runtime, "windows") else [tape.state]
    )
    expert_parts = [[] for _ in range(len(states[0][7]))]
    for wi, (w, state) in enumerate(zip(windows, states)):
        saved_x, ab, gates, origins, ends = state[3:8]
        cap = w.buffer.num_max_tokens_per_rank
        addr = origins.long().clamp_min(0)
        src_rank, token = addr // (cap * k), addr.remainder(cap * k) // k
        if hasattr(w, "node"):
            src_rank += token // w.cap * w.buffer.group.size()
            token = token.remainder(w.cap) + wi * getattr(runtime, "split", 0)
        global_slot = torch.where(
            origins >= 0, (src_rank * n + token) * k + addr % k, -1
        )
        start = 0
        for parts, end in zip(expert_parts, ends.tolist()):
            parts.append(tuple(t[start:end] for t in (saved_x, ab, gates, global_slot)))
            start = end
    return [tuple(torch.cat(v) for v in zip(*parts)) for parts in expert_parts]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=36)
    parser.add_argument("--capacity", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--port-counters", action="store_true")
    parser.add_argument(
        "--scales",
        nargs=6,
        type=float,
        metavar=NVFP4Scales._fields,
        help="Layer dequantization factors in NVFP4Scales order",
    )
    args = parser.parse_args()
    index = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(index)
    dist.init_process_group("nccl", device_id=torch.device("cuda", index))
    check_scale_midpoint()
    rank, world = dist.get_rank(), dist.get_world_size()
    e, h, i, k, n = max(16, world), args.hidden, 256, 4, args.tokens
    torch.manual_seed(442)

    def rand(*shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    ws = rand(e, 2 * i, h) * 0.025, rand(e, h, i) * 0.025
    w1, w2 = (w.chunk(world)[rank] for w in ws)
    torch.manual_seed(1351 + rank)
    x, dy = rand(n, h) * 0.2, rand(n, h) * 0.2
    r = torch.rand(n, e - 1, device="cuda").topk(k)[1].int()
    g = torch.rand(n, k, device="cuda")
    r[0] = 3
    r[1] = -1
    g[2] = 0
    if rank == 1:
        r.fill_(-1)
    runtime_options = dict(
        group=dist.group.WORLD,
        experts=e,
        top_k=k,
        hidden=h,
        intermediate=i,
        max_tokens=max(n, args.capacity),
        max_rows=n * k * world,
        precision="nvfp4",
        nvfp4_scales=NVFP4Scales(*args.scales) if args.scales else NVFP4Scales(),
        max_saved=2,
    )
    rt = TensorRuntime(training=True, **runtime_options)
    gs = []
    tape1, tape2 = [rt.forward(x, r, g, w1, w2) for _ in range(2)]
    expert_parts = checkpoint_view(rt.native, tape1, n, k)
    counters = None
    if args.port_counters and index == 0:
        from benchmarks.sm100.gin_ports import PhysicalCounters

        counters = PhysicalCounters()
    dist.barrier()
    torch.cuda.synchronize()
    for tape in (tape2, tape1):
        if counters is not None and tape is tape2:
            before = counters.read()
        gs.append(rt.backward(tape, dy))
        if counters is not None and tape is tape2:
            after = counters.read()
            print(
                "GIN_BACKWARD_PORT_BYTES "
                + json.dumps(
                    {
                        "rank": rank,
                        "tokens": n,
                        "tx_rx": {
                            p: [a - b for a, b in zip(pair, before[p])]
                            for p, pair in after.items()
                        },
                    }
                ),
                flush=True,
            )
    assert torch.equal(tape1.output, tape2.output)
    for a, b in zip(*gs):
        assert torch.equal(a, b)
    sx, s1, su, s2, sdy, sdab = rt.native.scales
    y = reference(x, r, g, *ws, (sx, s1, su, s2))
    errors = {"y": check(tape1.output, y, "y", tolerance=0.005)}
    all_x, all_dy, all_r, all_g = map(gather, (x, dy, r, g))
    dx_slots = torch.zeros((n * world, k, h), device=x.device, dtype=x.dtype)
    dg_slots = torch.zeros((n * world, k), device=x.device)
    ew1, ew2 = torch.zeros_like(w1), torch.zeros_like(w2)
    for j, (saved_x, ab, eg, ids) in enumerate(expert_parts):
        expert = rank * (e // world) + j
        live = ids >= 0
        live_ids = ids[live]
        tokens = live_ids // k
        # Recover original slots using stable sort by expert.
        order = torch.where((all_r >= 0) & (all_r < e), all_r, e).argsort(stable=True)
        slots = order[tokens, live_ids % k]
        assert (all_r[tokens, slots] == expert).all() and len(tokens) == (
            all_r == expert
        ).sum()
        xx = torch.zeros_like(saved_x)
        xx[live] = all_x[tokens]
        yy = torch.zeros_like(xx)
        yy[live] = all_dy[tokens]
        gate = torch.zeros_like(eg)
        gate[live] = all_g[tokens, slots]
        assert torch.equal(xx, saved_x) and torch.equal(gate, eg)
        aa = (round_nvfp4(xx, sx) @ round_nvfp4(w1[j], s1).T).bfloat16()
        check(ab, aa, "ab")
        a, b = aa.float().chunk(2, 1)
        du = (
            (round_nvfp4(yy, sdy) @ round_nvfp4(w2[j].T.contiguous(), s2).T)
            .bfloat16()
            .float()
        )
        sig = torch.sigmoid(a)
        silu = a * sig
        u = (silu * b * gate[:, None]).bfloat16()
        dab = torch.cat(
            (
                du * gate[:, None] * b * (sig * (1 + a * (1 - sig))),
                du * gate[:, None] * silu,
            ),
            1,
        ).bfloat16()
        dxx = (
            round_nvfp4(dab, sdab) @ round_nvfp4(w1[j].T.contiguous(), s1).T
        ).bfloat16()
        dx_slots[tokens, slots] = dxx[live]
        dg_slots[tokens, slots] = (du * silu * b).sum(1)[live]
        ew1[j] = (
            round_nvfp4(dab.T.contiguous(), sdab) @ round_nvfp4(xx.T.contiguous(), sx).T
        )
        ew2[j] = (
            round_nvfp4(yy.T.contiguous(), sdy) @ round_nvfp4(u.T.contiguous(), su).T
        )
    dist.all_reduce(dx_slots)
    dist.all_reduce(dg_slots)
    dx = dx_slots.view(world, n, k, h)[rank].float().sum(1).bfloat16()
    dg = dg_slots.view(world, n, k)[rank]
    for name, actual, expected in zip(
        ("dx", "dg", "dw1", "dw2"), gs[0], (dx, dg, ew1, ew2)
    ):
        errors[name] = check(actual, expected, name)
    if n <= 64:
        with torch.enable_grad():
            xx, gg, ww1, ww2 = [t.detach().requires_grad_() for t in (x, g, w1, w2)]
            output = moe(rt, xx, r, gg, ww1, ww2)
            grads = torch.autograd.grad(output, (xx, gg, ww1, ww2), dy)
        assert torch.equal(output, tape1.output)
        assert all(
            torch.equal(actual, expected) for actual, expected in zip(grads, gs[0])
        )
    if rank == 0:
        print(
            f"EP{world} repeated tapes bitwise identical; independent relative L2 {errors}",
            flush=True,
        )
    rt.close()
    inference = TensorRuntime(training=False, **runtime_options)
    weights = prepare_nvfp4_weights(w1, w2, weight_scales=(s1, s2))
    y = inference.inference_nvfp4(0, x, r, g, weights)
    assert torch.equal(y.view(torch.int16), tape1.output.view(torch.int16))
    inference.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
