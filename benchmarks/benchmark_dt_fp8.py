"""Single-node graph replay: identical routes/weights, source work included."""

import json, os, statistics
import torch
import torch.distributed as dist
from weftmoe.runtime import TensorRuntime
from benchmarks.fp8_validation import global_diff, passes, DIFF_LIMIT
from weftmoe.fp8 import prepare_fp8_weights, quantize_fp8


@torch.no_grad()
def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i = (
        int(os.getenv("TEST_HIDDEN", "2048")),
        int(os.getenv("TEST_INTERMEDIATE", "512")),
    )
    experts, topk = (
        int(os.getenv("BENCH_EXPERTS", "256")),
        int(os.getenv("BENCH_TOPK", "8")),
    )
    sizes = list(map(int, os.getenv("BENCH_SIZES", "1 16 32 64 128 256 512").split()))
    seed = int(os.getenv("BENCH_SEED", "900"))
    torch.manual_seed(123)
    w1 = (
        torch.randn(experts // world, 2 * i, h, device="cuda", dtype=torch.bfloat16)
        * 0.03
    )
    w2 = torch.randn(experts // world, h, i, device="cuda", dtype=torch.bfloat16) * 0.03
    weights = prepare_fp8_weights(w1, w2, version=0, training=False)
    kwargs = dict(
        group=dist.group.WORLD,
        experts=experts,
        top_k=topk,
        hidden=h,
        intermediate=i,
        max_tokens=max(sizes),
        max_rows=max(sizes) * world * topk + experts * 128,
        training=False,
        decode_only=True,
    )
    bf = TensorRuntime(**kwargs)
    fp = TensorRuntime(**kwargs, fp8_workspace=True)
    torch.manual_seed(seed + rank)
    for n in sizes:
        x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16) * 0.03
        r = (
            torch.rand(n, experts, device="cuda")
            .topk(topk, dim=1)
            .indices.int()
            .contiguous()
        )
        g = torch.rand(n, topk, device="cuda")
        g /= g.sum(1, keepdim=True)
        q = quantize_fp8(x)
        calls = [
            lambda: bf.inference(0, x, r, g, w1, w2),
            lambda: fp.inference_fp8(1, x, r, g, weights),
            lambda: fp.inference_fp8(2, q, r, g, weights),
        ]
        graphs = []
        outputs = []
        repeats = int(os.getenv("BENCH_REPEATS", "50"))
        for call in calls:
            outputs.append(call())
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(repeats):
                    call()
            graphs.append(graph)
        measurements = [[], [], []]
        for sweep in range(int(os.getenv("BENCH_SWEEPS", "7"))):
            for idx in [0, 1, 2] if sweep % 2 == 0 else [2, 1, 0]:
                dist.barrier()
                graphs[idx].replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[idx].replay()
                end.record()
                end.synchronize()
                ms = torch.tensor(
                    start.elapsed_time(end) * 1000 / repeats, device="cuda"
                )
                dist.all_reduce(ms, op=dist.ReduceOp.MAX)
                if sweep:
                    measurements[idx].append(ms.item())
        row = dict(
            seed=seed,
            tokens=n,
            hidden=h,
            intermediate=i,
            experts=experts,
            topk=topk,
            ep=world,
            routing="uniform random top-k, static timed replays",
            bf16_us=statistics.median(measurements[0]),
            fp8_online_us=statistics.median(measurements[1]),
            fp8_prequant_us=statistics.median(measurements[2]),
            samples_us=measurements,
        )
        valid = torch.tensor(
            int(
                torch.equal(outputs[1], outputs[2])
                and all(torch.isfinite(o).all().item() for o in outputs)
            ),
            device="cuda",
        )
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        assert valid.item(), "nonfinite output or online/prequantized output mismatch"
        diff = torch.tensor(
            global_diff(outputs[1], outputs[0]), dtype=torch.float64, device="cuda"
        )
        dist.all_reduce(diff, op=dist.ReduceOp.MAX)
        assert passes(diff.item()), ("FP8/BF16 global diff", diff.item(), DIFF_LIMIT)
        row.update(fp8_vs_bf16_max_rank_diff=diff.item(), fp8_diff_limit=DIFF_LIMIT)
        if rank == 0:
            print("RESULT " + json.dumps(row), flush=True)
        if os.getenv("BENCH_PROFILE") == "1":
            for name, rt, inp, ww1, ww2, fw in [
                ("bf16", bf, x, w1, w2, None),
                ("fp8", fp, x, weights.w1.data, weights.w2.data, weights),
            ]:
                stats = rt.native.profile_inference(
                    inp,
                    r,
                    g,
                    ww1,
                    ww2,
                    torch.cuda.current_stream().cuda_stream,
                    None,
                    fw,
                )
                del stats["output"]
                if rank == 0:
                    print(
                        "PHASES " + json.dumps(dict(tokens=n, variant=name, **stats)),
                        flush=True,
                    )
            for fmt in (1, 2):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    graphs[fmt].replay()
                    torch.cuda.synchronize()
                kernels = [
                    e.name
                    for e in prof.events()
                    if e.device_type == torch.autograd.DeviceType.CUDA
                ]
                assert len(kernels) == repeats and all(
                    "weftmoe_dt::kernel" in k and "true" in k for k in kernels
                ), kernels
                if rank == 0:
                    print(
                        "KERNELS "
                        + json.dumps(
                            dict(
                                input_format="bf16" if fmt == 1 else "fp8",
                                kernels=kernels,
                            )
                        ),
                        flush=True,
                    )
        del graph, graphs
        bf.invalidate_inference_layer(0)
        fp.invalidate_inference_layer(1)
        fp.invalidate_inference_layer(2)
    bf.close()
    fp.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
