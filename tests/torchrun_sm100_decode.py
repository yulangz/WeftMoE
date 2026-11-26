"""Compare single-node decode with training under changing rank sizes and adversarial routes."""

import os

import torch
import torch.distributed as dist

from weftmoe.blackwell import prepare_bf16_weights
from weftmoe.nvfp4 import NVFP4Scales, prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime


@torch.no_grad()
def check(group, owned, capacity, precision, intermediate=256):
    rank, world = group.rank(), group.size()
    h, i, k = 512, intermediate, 6
    experts = owned * world
    config = dict(group=group, experts=experts, hidden=h, intermediate=i, top_k=k,
                  max_tokens=capacity, max_rows=capacity * world * k, training=False, precision=precision)
    prefill = TensorRuntime(**config)
    decode = TensorRuntime(**config, decode_only=True)
    torch.manual_seed(1234 + rank)
    w1 = torch.randn(owned, 2 * i, h, device="cuda", dtype=torch.bfloat16) * 0.03
    w2 = torch.randn(owned, h, i, device="cuda", dtype=torch.bfloat16) * 0.03
    packed = prepare_nvfp4_weights(w1, w2) if precision == "nvfp4" else None

    tiled = prepare_bf16_weights(w1, w2) if packed is None else prepare_nvfp4_weights(w1, w2, tiled=True)

    def forward(rt, x, ids, gates, use_tiles=False):
        if packed is None:
            return rt.inference(0, x, ids, gates, *(tiled if use_tiles else (w1, w2)))
        return rt.inference_nvfp4(0, x, ids, gates, tiled if use_tiles else packed, intermediate_scale=0.003)

    try:
        sizes = (capacity, 0, capacity - rank, 0 if rank == 0 else 1, capacity, 0 if rank == 0 else capacity)
        samples = []
        for step, n in enumerate(sizes):
            x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16) * 0.03
            gates = torch.rand(n, k, device="cuda")
            ids = torch.randint(experts, (n, k), device="cuda", dtype=torch.int32)
            if step in (0, 4, 5):
                # Every slot, including duplicates, hits the same expert. Change
                # its owner on reuse to exercise empty ranks and stale signals.
                ids.fill_(experts - 1 if step == 4 else 0)
            elif step == 2:
                ids[::2] = -1
                ids[1::2, 0] = experts
            else:
                ids.fill_(-1)
            reference = forward(prefill, x, ids, gates)
            samples.append((x, ids, gates, reference))
            actual = forward(decode, x, ids, gates)
            same = torch.isfinite(actual).all() & (reference.view(torch.int16) == actual.view(torch.int16)).all()
            tiled_out = forward(decode, x, ids, gates, use_tiles=True)
            shared_training = forward(prefill, x, ids, gates, use_tiles=True)
            same &= (reference.view(torch.int16) == tiled_out.view(torch.int16)).all()
            same &= (reference.view(torch.int16) == shared_training.view(torch.int16)).all()
            dist.all_reduce(same, op=dist.ReduceOp.MIN, group=group)
            assert same.item(), (precision, owned, capacity, step)
        # Queue generations without host/collective synchronization. Delay a
        # rotating rank while other sources can already publish their next input.
        comparisons = []
        for step in range(4 * len(samples)):
            x, ids, gates, reference = samples[step % len(samples)]
            if step % world == rank:
                torch.cuda._sleep(300_000)
            actual = forward(decode, x, ids, gates, use_tiles=bool(step % 2))
            comparisons.append((reference.view(torch.int16) == actual.view(torch.int16)).all())
        equal = torch.stack(comparisons).all().int()
        dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
        assert equal.item(), (precision, owned, capacity, "skewed workspace reuse")

        if owned == 32:
            # %gridid repeats on replay. Change every input between consecutive
            # executions of the same graph and compare every generation with training.
            train = TensorRuntime(**(config | {"training": True}),
                                  nvfp4_scales=NVFP4Scales(intermediate=0.003) if packed else None)
            try:
                fixed = tuple(t.clone() for t in samples[0][:3])
                references = []
                for j in range(8):
                    x = torch.randn_like(fixed[0]) * 0.03
                    ids = torch.randint(experts, fixed[1].shape, device="cuda", dtype=torch.int32)
                    if j % 2:
                        ids.fill_((rank + j) % experts)
                        ids[::3, 0] = -1
                    gates = torch.rand_like(fixed[2])
                    references.append(((x, ids, gates), train.forward(x, ids, gates, w1, w2).output))
                for runtime in (prefill, decode):
                    forward(runtime, *fixed, use_tiles=True)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        out = forward(runtime, *fixed, use_tiles=True)
                    comparisons = []
                    for j in range(128):
                        inputs, reference = references[j % len(references)]
                        for buf, source in zip(fixed, inputs, strict=True):
                            buf.copy_(source)
                        if j % world == rank:
                            torch.cuda._sleep(300_000)
                        graph.replay()
                        comparisons.append((out.view(torch.int16) == reference.view(torch.int16)).all())
                    equal = torch.stack(comparisons).all().int()
                    dist.all_reduce(equal, op=dist.ReduceOp.MIN, group=group)
                    assert equal.item(), (precision, capacity, "decode" if runtime is decode else "training", "same graph, new inputs")
            finally:
                train.close()
    finally:
        decode.close()
        prefill.close()


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        group = dist.group.WORLD
        for precision in ("bf16", "nvfp4"):
            # A warp-sized expert directory and a directory spanning multiple warps.
            for owned, capacity in ((32, 16), (32, 512), (33, 16), (33, 512)):
                check(group, owned, capacity, precision)
        # Intermediate divisible by 128 only uses 64-byte NVFP4 K tiles.
        check(group, 8, 16, "nvfp4", intermediate=384)
        if group.rank() == 0:
            print(f"decode/training bitwise route and workspace reuse checks passed on EP{group.size()}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
