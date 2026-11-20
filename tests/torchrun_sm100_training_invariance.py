"""Cross-EP train/inference forward bits and same-EP backward replay on B300."""

import argparse
import os

import torch
import torch.distributed as dist

from weftmoe.blackwell_cutlass import tile_m as select_m
from weftmoe.nvfp4 import prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime


def batch(size, hidden, experts, top_k, device):
    x = torch.randn(size, hidden, device=device, dtype=torch.bfloat16) * 0.16
    dy = torch.randn_like(x) * 0.12
    scores = torch.randn(size, experts, device=device) + torch.linspace(
        -1, 1, experts, device=device
    )
    routes = scores.topk(top_k).indices.int()
    gates = torch.rand(size, top_k, device=device)
    return x, routes, gates, dy


def different_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    dtype = {torch.bfloat16: torch.int16, torch.float32: torch.int32}[actual.dtype]
    return (actual.contiguous().view(dtype) != expected.contiguous().view(dtype)).sum()


def forward(runtime, data, weights, packed):
    x, routes, gates, _ = data
    if runtime.training:
        return runtime.forward(x, routes, gates, *weights)
    if packed is not None:
        return runtime.inference_nvfp4(0, x, routes, gates, packed)
    return runtime.inference_eager(x, routes, gates, *weights)


def tile_m(runtime, rows):
    window = getattr(runtime.native, "windows", [runtime.native])[0]
    buf = window.buffer
    return select_m(
        buf.group.size(),
        buf.num_experts,
        len(window.dummy) if hasattr(window, "dummy") else rows,
        runtime.top_k,
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=["bf16", "nvfp4"], required=True)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    world = dist.group.WORLD
    ep, rank = world.size(), world.rank()
    assert ep in (8, 16, 32)
    single, _ = dist.new_subgroups(group_size=1)
    e, h, i, k = args.experts, args.hidden, args.intermediate, args.top_k
    torch.manual_seed(8479)
    weights = tuple(
        torch.randn(shape, device=device, dtype=torch.bfloat16) * 0.023
        for shape in ((e, 2 * i, h), (e, h, i))
    )
    local_weights = tuple(w.chunk(ep)[rank] for w in weights)

    def runtime(group, capacity, training):
        rt = TensorRuntime(
            group=group,
            experts=e,
            top_k=k,
            hidden=h,
            intermediate=i,
            max_tokens=capacity,
            max_rows=capacity * group.size() * k,
            training=training,
            precision=args.precision,
            batch_invariant=True,
        )
        windows = getattr(rt.native, "windows", ())
        device_sms = torch.cuda.get_device_properties(device).multi_processor_count
        for child in windows:
            assert child.buffer.sms == (device_sms - 4) // 4 * 4
            assert child.buffer.domains.numel() == device_sms
        for child in windows or (rt.native,):
            if child.buffer.fp4:
                assert child.buffer.layout.sf_ring == 8 * child.buffer.layout.ring
        return rt

    def pack(weights, training):
        if args.precision == "nvfp4" and not training:
            return prepare_nvfp4_weights(*weights)

    anchors = batch(128, h, e, k, device)
    anchors[1][0] = 3
    anchors[1][1] = -1
    anchors[2][2] = 0
    anchor_count = len(anchors[0])
    reference = {}
    for training in (False, True):
        rt = runtime(single, anchor_count, training)
        result = forward(rt, anchors, weights, pack(weights, training))
        reference[training] = (result.output if training else result).clone()
        assert (
            torch.isfinite(reference[training]).all()
            and reference[training][3:].count_nonzero()
        )
        if training:
            rt.backward(result, anchors[3])
        reference_m = tile_m(rt, anchor_count)
        rt.close()

    windows = [(8192 // ep, [128, 256, 640, 1536, 2304, 8192])]
    if ep == 16:
        windows.append(
            (4096, [2051 * ep])
        )  # Cross the pipelined runtime's window boundary.
    for capacity, lengths in windows:
        diffs = torch.zeros(len(lengths), 6, dtype=torch.long, device=device)
        matched = torch.zeros(len(lengths), dtype=torch.long, device=device)
        tiles = []
        for training in (False, True):
            rt = runtime(world, capacity, training)
            packed = pack(local_weights, training)
            for j, total in enumerate(lengths):
                torch.manual_seed(3029 + total)
                n = total // ep
                labels = torch.randperm(total, device=device).chunk(ep)[rank]
                data = batch(n, h, e, k, device)
                positions = (labels < anchor_count).nonzero().flatten()
                ids = labels[positions]
                for dest, src in zip(data, anchors, strict=True):
                    dest[positions] = src[ids]
                result = forward(rt, data, local_weights, packed)
                out = result.output if training else result
                diffs[j, int(training)] = different_bits(
                    out[positions], reference[not training][ids]
                )
                if training:
                    grads = tuple(v.clone() for v in rt.backward(result, data[3]))
                    other = tuple(
                        v.flip([0])[: max(1, n // 2)].contiguous() for v in data
                    )
                    rt.backward(forward(rt, other, local_weights, None), other[3])
                    again = rt.backward(forward(rt, data, local_weights, None), data[3])
                    for col, (a, b) in enumerate(zip(grads, again, strict=True), 2):
                        assert torch.isfinite(a).all() and torch.isfinite(b).all()
                        diffs[j, col] = different_bits(a, b)
                else:
                    matched[j] = len(positions)
                    tiles.append(tile_m(rt, n))
            rt.close()
        dist.all_reduce(matched)
        dist.all_reduce(diffs)
        assert (matched == anchor_count).all(), matched.tolist()
        assert not diffs.any(), (
            f"FWD train/infer and BWD dx/dgate/dw1/dw2 mismatches: {diffs.tolist()}"
        )
        if rank == 0:
            print(
                f"{args.precision} H={h} I={i} EP1/EP{ep} tokens={lengths} "
                f"M={reference_m}/{tiles}: both FWD directions and all four gradients bitwise equal",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
