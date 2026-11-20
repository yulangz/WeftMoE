"""Validate native block-16 NVFP4 against explicit E2M1 rounding and FP32 GEMMs."""

import argparse
import os

import torch
import torch.distributed as dist

from weftmoe.blackwell_tuning import STRICT_BLOCK_M
from weftmoe.nvfp4 import _pack_rows, prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime


def round_nvfp4(x, tensor_scale):
    block = x.float().unflatten(-1, (-1, 16))
    sf = (block.abs().amax(-1, keepdim=True) / (6 * tensor_scale)).clamp(max=448)
    sf = sf.to(torch.float8_e4m3fn).float() * tensor_scale
    absolute = torch.where(sf > 0, block.abs() / sf, 0.0)
    midpoints = x.new_tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32
    )
    magnitudes = x.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    code = torch.bucketize(absolute.contiguous(), midpoints)
    # Midpoints whose lower code is odd round up to the even E2M1 code.
    code += ((absolute == 0.75) | (absolute == 1.75) | (absolute == 3.5)).long()
    return (magnitudes[code] * block.sign() * sf).flatten(-2)


def reference(x, routes, gates, w1, w2, scales):
    sx, s1, sm, s2 = scales
    x = round_nvfp4(x, sx)
    result = torch.zeros_like(x)
    for e in range(len(w1)):
        for slot in range(routes.shape[1]):
            rows = torch.where(routes[:, slot] == e)[0]
            if rows.numel():
                a, b = (
                    (x[rows] @ round_nvfp4(w1[e], s1).T).bfloat16().float().chunk(2, 1)
                )
                u = torch.nn.functional.silu(a) * b * gates[rows, slot, None]
                component = round_nvfp4(u, sm) @ round_nvfp4(w2[e], s2).T
                result[rows] += component.bfloat16().float()
    return result.bfloat16()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--total-tokens", type=int, default=64)
    parser.add_argument(
        "--capacity", type=int, help="Exercise a larger prepared window"
    )
    args = parser.parse_args()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    world = dist.get_world_size()
    rank = dist.get_rank()
    single, _ = dist.new_subgroups(group_size=1)
    e, h, i, k, total = 64, 1024, args.intermediate, 4, args.total_tokens
    assert total % world == 0
    torch.manual_seed(6841)
    weights = [
        torch.randn(shape, dtype=torch.bfloat16, device="cuda") * 0.023
        for shape in ((e, 2 * i, h), (e, h, i))
    ]
    x = torch.randn(total, h, device="cuda", dtype=torch.bfloat16) * 0.16
    ids = torch.rand(total, e, device="cuda").topk(k)[1].int()
    gates = torch.rand(total, k, device="cuda")
    ids[0] = 3
    ids[1] = -1
    x[2] = 0
    gates[3] = 0
    scales = (1 / 32, 1 / 64, 1 / 512, 1 / 64)
    expected = reference(x, ids, gates, *weights, scales)
    import deep_gemm as dg

    results, tiles = [], []
    for g in (single, dist.group.WORLD):
        n = total // g.size()
        rows = slice(g.rank() * n, (g.rank() + 1) * n)
        rt = TensorRuntime(
            group=g,
            experts=e,
            top_k=k,
            hidden=h,
            intermediate=i,
            max_tokens=max(n, args.capacity or 0),
            max_rows=total * k,
            training=False,
            precision="nvfp4",
        )
        window = rt.native.windows[0] if hasattr(rt.native, "windows") else rt.native
        buf = window.buffer
        assert buf.full_mma_tiles and buf.fixed_block_m == STRICT_BLOCK_M
        launched = len(window.dummy) if hasattr(window, "dummy") else n
        tiles.append(
            dg.get_block_m_for_mega_moe(
                buf.group.size(),
                buf.num_experts,
                buf.num_max_tokens_per_rank,
                launched,
                k,
                "nvfp4",
                buf.fixed_block_m,
            )
        )
        w = prepare_nvfp4_weights(*(v.chunk(g.size())[g.rank()] for v in weights))
        y = rt.inference_nvfp4(0, x[rows], ids[rows], gates[rows], w)
        short = n // 2
        prefix = rt.inference_nvfp4(0, x[rows][:short], ids[rows][:short], gates[rows][:short], w)
        assert torch.equal(prefix.view(torch.int16), y[:short].view(torch.int16))
        if g.size() == 1:
            # A truncated expert metadata pool overwrites the beginning of packed X.
            # Check the stored bytes even when dispatch consumed the rows before the overwrite.
            count = min(256, n)
            data = torch.empty_like(buf.x[:count])
            sf = torch.empty_like(buf.x_sf[:count])
            _pack_rows[(count, (h + 511) // 512)](
                x[rows], data, sf.view(torch.uint8), scales[0], 1, h, 512, False, False
            )
            intact = torch.tensor(
                torch.equal(buf.x[:count], data) and torch.equal(buf.x_sf[:count], sf),
                dtype=torch.int32,
                device=x.device,
            )
            dist.all_reduce(intact, op=dist.ReduceOp.MIN)
            assert intact.item(), "Expert metadata overwrote the input workspace"
        z = rt.inference_nvfp4(
            0, x[rows].flip([0]), ids[rows].flip([0]), gates[rows].flip([0]), w
        )
        difference = y.view(torch.int16) != z.flip([0]).view(torch.int16)
        mismatch = difference.any().int()
        if mismatch.item():
            first = difference.nonzero()[0].tolist()
            row, col = first
            print(
                f"rank={rank} EP={g.size()} tile={tiles[-1]} unequal={int(difference.sum())} "
                f"rows={int(difference.any(1).sum())} first={first} "
                f"values={float(y[row, col])},{float(z[n - 1 - row, col])} "
                f"reference={float(expected[rows][row, col])}",
                flush=True,
            )
        dist.all_reduce(mismatch, op=dist.ReduceOp.MAX)
        assert not mismatch.item(), f"EP{g.size()} permutation changed output bits"
        if g is dist.group.WORLD:
            fixed_x, fixed_ids, fixed_gates = (v[rows].clone() for v in (x, ids, gates))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = rt.inference_nvfp4(0, fixed_x, fixed_ids, fixed_gates, w)
            fixed_x.copy_(x[rows].flip([0]))
            fixed_ids.copy_(ids[rows].flip([0]))
            fixed_gates.copy_(gates[rows].flip([0]))
            graph.replay()
            assert torch.equal(captured.view(torch.int16), z.view(torch.int16))
            live = 0 if rank == 0 else n
            partial = rt.inference_nvfp4(
                0, x[rows][:live], ids[rows][:live], gates[rows][:live], w
            )
            assert torch.equal(partial, y[:live])
            del graph
        results.append(y)
        rt.close()
    assert torch.equal(results[1].view(torch.int16), results[0][rows].view(torch.int16))
    err = (results[0].float() - expected.float()).norm() / expected.float().norm()
    assert err.item() < 0.005, err
    if rank == 0:
        print(
            f"NVFP4 EP1/EP{world} bitwise parity, M={tiles} N=128 K={256 if i % 256 == 0 else 128}; independent reference relative L2 {err.item():.6g}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
