#!/usr/bin/env python3
"""Single-node decode comparison, with shared inputs and interleaved graphs.

Point --deepgemm-package at the directory containing the frozen deep_gemm
package. MegaMoE's required input staging is measured in the same graph as its
kernel, not estimated by adding independently measured medians.
"""

import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

from weftmoe.runtime import TensorRuntime


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, choices=(1, 64), required=True)
    parser.add_argument("--deepgemm-package", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.deepgemm_package.resolve()))
    import deep_gemm

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from torchrun_static_inference import reference

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300),
                            device_id=torch.device("cuda", local_rank))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8 or int(os.environ["LOCAL_WORLD_SIZE"]) != 8:
        raise ValueError("This MegaMoE reference uses single-node EP8 pointers")
    t, h, i, e, k = args.tokens, 2048, 512, 64, 4
    deep_gemm.set_num_sms(132)
    generator = torch.Generator(device="cuda").manual_seed(123)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                           generator=generator) * 0.03

    all_w1, all_w2 = rand(e, 2 * i, h), rand(e, h, i)
    w1 = all_w1.chunk(world)[rank].contiguous()
    w2 = all_w2.chunk(world)[rank].contiguous()
    generator.manual_seed(900 + rank)
    x = rand(t, h)
    # Same route geometry as the native Forward benchmark (route stride53).
    rows = torch.arange(t, device="cuda", dtype=torch.int32)[:, None]
    slots = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    routes = ((rows * 17 + rank * 31 + slots * 53) % e).contiguous()
    gates = torch.rand(t, k, device="cuda", generator=generator)
    gates.div_(gates.sum(dim=1, keepdim=True))
    expected = reference(x, routes, gates, all_w1, all_w2)
    runtimes, outputs, calls = {}, {}, {}
    for ring in (512, 0):
        os.environ["WEFTMOE_STATIC_RING_TOKENS"] = str(ring)
        os.environ["WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS"] = "8"
        name = "weft_ring512" if ring else "weft_arena"
        runtime = TensorRuntime(
            group=dist.group.WORLD, experts=e, top_k=k, hidden=h,
            intermediate=i, max_tokens=t,
            max_rows=world * t * k + (e // world) * 128, training=False,
        )
        runtimes[name] = runtime
        calls[name] = lambda runtime=runtime: runtime.inference(
            0, x, routes, gates, w1, w2)
        outputs[name] = calls[name]()

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        dist.group.WORLD, e, t, k, h, i, mma_type="bf16xbf16")
    packed_w1, packed_w2 = deep_gemm.transform_weights_for_mega_moe(w1, w2)
    routes64 = routes.long()
    mega_output = torch.empty_like(x)

    def copy_inputs():
        buffer.x[:t].copy_(x)
        buffer.topk_idx[:t].copy_(routes64)
        buffer.topk_weights[:t].copy_(gates)

    def mega_kernel():
        deep_gemm.bf16_mega_moe(mega_output, packed_w1, packed_w2, buffer,
                               activation_clamp=None, fast_math=True)
        return mega_output

    def mega_with_copies():
        copy_inputs()
        return mega_kernel()

    calls["mega_kernel"] = mega_kernel
    calls["mega_with_copies"] = mega_with_copies
    copy_inputs()
    outputs["mega_kernel"] = mega_kernel()
    outputs["mega_with_copies"] = mega_output
    graphs = {}
    for name, call in calls.items():
        call()  # JIT and Weft child-graph preparation are outside measurement.
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graphs[name] = graph
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(outputs[name].float(), expected.float(),
                                   rtol=0.02, atol=2e-5)
    if not torch.equal(outputs["weft_ring512"], outputs["weft_arena"]):
        raise AssertionError("Weft ring/arena output bits differ")

    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    card = ctypes.c_void_p()
    if nvml.nvmlInit_v2() or nvml.nvmlDeviceGetHandleByIndex_v2(
            local_rank, ctypes.byref(card)):
        raise RuntimeError("NVML initialization failed")

    def clock():
        mhz = ctypes.c_uint()
        if nvml.nvmlDeviceGetClockInfo(card, 1, ctypes.byref(mhz)):
            raise RuntimeError("NVML SM clock query failed")
        return mhz.value

    samples = {name: [] for name in calls}
    clocks = {name: [] for name in calls}
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    names = list(calls)
    for iteration in range(30):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            dist.barrier()
            torch.cuda.synchronize()
            before = clock()
            begin.record()
            for _ in range(20):
                graphs[name].replay()
            end.record()
            end.synchronize()
            after = clock()
            if iteration >= 10:
                samples[name].append(begin.elapsed_time(end) * 1000 / 20)
                clocks[name].append((before, after))
    local = {
        "samples": samples, "clocks": clocks,
        "output_sha256": {
            name: hashlib.sha256(
                value.view(torch.uint16).cpu().numpy().tobytes()).hexdigest()
            for name, value in outputs.items()
        },
        "max_abs": {name: (value.float() - expected.float()).abs().max().item()
                    for name, value in outputs.items()},
        "mega_symm_bytes": buffer.buffer.numel(),
    }
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    if rank == 0:
        maximum = {name: [max(r["samples"][name][j] for r in gathered)
                          for j in range(20)] for name in names}
        print("MEGAMOE_DECODE_COMPARE " + json.dumps({
            "world": world, "tokens": t, "hidden": h, "intermediate": i,
            "experts": e, "top_k": k, "warmup": 10, "measured": 20, "batch": 20,
            "deepgemm_package": deep_gemm.__file__,
            "rank_max_p50_us": {
                name: statistics.median(values) for name, values in maximum.items()
            },
            "rank_max_samples_us": maximum, "per_rank": gathered,
        }, sort_keys=True), flush=True)
    nvml.nvmlShutdown()
    graphs.clear()
    del graph
    for runtime in runtimes.values():
        runtime.close()
    buffer.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
