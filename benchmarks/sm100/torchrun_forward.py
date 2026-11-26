"""Compare the public Blackwell forward path with the matching pre-staged kernel."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.models import add_model_arguments, resolve_model
from benchmarks.sm100.hardware import check_blackwell
from benchmarks.workload import decode_routes, make_workload, workload_sha256
from weftmoe.runtime import TensorRuntime


def decode_trace(args, n, rank, world):
    logits = torch.empty(n * world, args.experts, device="cuda")
    routes = decode_routes(args, logits)
    next(routes)  # CUDA Graph capture input
    next(routes)  # eager/replay parity input
    trace = []
    digest = hashlib.sha256()
    for _ in range(args.warmups + args.samples):
        ids, gates = next(routes)
        digest.update(ids.cpu().numpy().tobytes())
        digest.update(gates.cpu().numpy().tobytes())
        rows = slice(n * rank, n * (rank + 1))
        trace.append((ids[rows].cpu(), gates[rows].cpu()))
    return trace, digest.hexdigest()


def pack_mega_mixed(w1, w2):
    import deep_gemm as dg
    from deep_gemm.utils import per_token_cast_to_fp4

    matrices = []
    for w in (w1, w2):
        e, n, k = w.shape
        data = torch.empty((e, n, k // 2), dtype=torch.int8, device=w.device)
        scales = torch.empty((e, n, k // 32), dtype=torch.float32, device=w.device)
        for expert in range(e):
            data[expert], scales[expert] = per_token_cast_to_fp4(w[expert], use_ue8m0=True, gran_k=32)
        matrices.append((data, dg.transform_sf_into_required_layout(scales, n, k, (1, 32), e)))
    return dg.transform_weights_for_mega_moe(*matrices)


def prepare_tiled_weights(w1, w2, nvfp4):
    if nvfp4:
        from weftmoe.nvfp4 import prepare_nvfp4_weights
        return prepare_nvfp4_weights(w1, w2, tiled=True)
    from weftmoe.blackwell import prepare_bf16_weights
    return prepare_bf16_weights(w1, w2)


def measure(args, n, w1, w2, packed, group, mixed_weights=None, tiled_weights=None):
    if args.compare_mega:
        import deep_gemm as dg

    rank, world = group.rank(), group.size()
    decode = args.mode == "decode"
    nvfp4 = args.precision == "fp4"
    if decode:
        trace, digest = decode_trace(args, n, rank, world)
    else:
        trace = make_workload(args, n, rank, "cuda")
        digest = workload_sha256(trace)
    capacity = torch.tensor(max(len(ids) for ids, _ in trace), device="cuda")
    dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
    cap = args.decode_capacity if args.decode_capacity is not None else int(capacity)
    config = dict(
        group=group,
        experts=args.experts,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        max_tokens=cap,
        max_rows=cap * world * args.top_k,
        training=False,
        precision="nvfp4" if nvfp4 else "bf16",
        batch_invariant=args.batch_invariant,
    )
    use_decode = decode and group.size() == int(os.environ["LOCAL_WORLD_SIZE"]) and torch.cuda.get_device_capability() == (10, 3)
    if args.compare_training and not use_decode:
        raise ValueError("decode/training comparison requires single-node SM103 decode")
    rt = TensorRuntime(**config, decode_only=use_decode)
    if tiled_weights is not None and not rt.supports_tiled_weights:
        rt.close()
        raise ValueError("Weight-layout comparison requires single-node SM103 inference")
    training = TensorRuntime(**config) if args.compare_training else None
    ref = (
        dg.SymmBuffer(
            group,
            args.experts,
            cap,
            args.top_k,
            args.hidden,
            args.intermediate,
            mma_type="fp8xfp4" if mixed_weights else "nvfp4" if nvfp4 else "bf16xbf16",
        )
        if args.compare_mega and packed is not None and group.size() == int(os.environ["LOCAL_WORLD_SIZE"])
        else None
    )
    x = torch.empty(cap, args.hidden, device="cuda", dtype=torch.bfloat16)
    routes = torch.empty(cap, args.top_k, device="cuda", dtype=torch.int32)
    weights = torch.empty_like(routes, dtype=torch.float32)
    expected = torch.empty_like(x)
    input_rng = torch.Generator(device="cuda").manual_seed(args.seed + n * world + rank)

    def prepare(ids, gates):
        count = len(ids)
        x[:count].normal_(0, 0.03, generator=input_rng)
        routes[:count].copy_(ids)
        weights[:count].copy_(gates)
        if ref is not None:
            ordered, perm = routes[:count].sort(dim=1, stable=True)
            if mixed_weights:
                from deep_gemm.utils import per_token_cast_to_fp8

                ref.x[:count], ref.x_sf[:count] = per_token_cast_to_fp8(
                    x[:count], use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
                )
            elif nvfp4:
                from weftmoe.nvfp4 import stage_nvfp4

                stage_nvfp4(ref, x[:count], routes[:count], weights[:count], 1 / 32)
            else:
                ref.x[:count].copy_(x[:count])
            ref.topk_idx[:count].copy_(ordered)
            ref.topk_weights[:count].copy_(weights[:count].gather(1, perm))
        return count

    def run_weft(count, runtime=rt):
        if nvfp4:
            return runtime.inference_nvfp4(0, x[:count], routes[:count], weights[:count], packed)
        return runtime.inference(0, x[:count], routes[:count], weights[:count], w1, w2)

    def run_mega(count):
        if mixed_weights:
            dg.fp8_fp4_mega_moe(expected[:count], *mixed_weights, ref)
        elif nvfp4:
            dg.nvfp4_mega_moe(
                expected[:count],
                packed.fc1,
                packed.fc2,
                ref,
                tensor_scales=(packed.tensor_scales[0] / 32, 1 / 512, packed.tensor_scales[1] / 512),
            )
        else:
            dg.bf16_mega_moe(expected[:count], *packed, ref)
        return expected[:count]

    runners = [run_weft, run_mega] if ref is not None else [run_weft]
    names = ["weft"]
    if ref is not None:
        names.append("mega_mxfp8_mxfp4" if mixed_weights else "nvfp4_prestaged" if nvfp4 else "mega_prestaged")
    if training is not None:
        runners.append(lambda count: run_weft(count, training))
        names.append("training")
    if tiled_weights is not None:
        def run_tiled(count):
            inputs = (0, x[:count], routes[:count], weights[:count])
            if nvfp4:
                return rt.inference_nvfp4(*inputs, tiled_weights)
            return rt.inference(*inputs, *tiled_weights)

        runners.append(run_tiled)
        names.append("tiled_decode" if use_decode else "tiled_training")
    count = prepare(*trace[0])
    for runner in runners:
        runner(count)
    if decode:
        graphs = []
        for runner in runners:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = runner(count)
            graphs.append((graph, output))
    times = [[] for _ in runners]
    start_event, stop_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    # Eager baseline primes the full trace before warmup and sampling.
    iterations = trace if decode else trace * 2
    counters = None
    if args.port_counters and int(os.environ["LOCAL_RANK"]) == 0:
        from benchmarks.sm100.gin_ports import PhysicalCounters

        counters = PhysicalCounters()
        before = counters.read()
    for step, sample in enumerate(iterations):
        count = prepare(*sample)
        outputs = [None] * len(runners)
        for position in range(len(runners)):
            backend = (step + position) % len(runners)
            dist.barrier()
            torch.cuda.synchronize()
            if decode:
                graph, output = graphs[backend]
                start_event.record()
                for _ in range(args.replays):
                    graph.replay()
                stop_event.record()
                stop_event.synchronize()
                us = start_event.elapsed_time(stop_event) * 1000 / args.replays
            else:
                start = time.perf_counter_ns()
                output = runners[backend](count)
                torch.cuda.synchronize()
                us = (time.perf_counter_ns() - start) / 1000
            outputs[backend] = output
            if step >= len(iterations) - args.samples:
                times[backend].append(us)
        valid = torch.isfinite(outputs[0]).all()
        if ref is not None:
            valid &= torch.isfinite(outputs[1]).all() if mixed_weights else (outputs[0] == outputs[1]).all()
        for backend in ("training", "tiled_decode", "tiled_training"):
            if backend in names:
                valid &= (outputs[0].view(torch.int16) == outputs[names.index(backend)].view(torch.int16)).all()
        valid = valid.int()
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise RuntimeError(f"Non-finite or mismatched MoE output at step {step}")
    if counters is not None:
        after = counters.read()
        print(
            "GIN_PORT_BYTES "
            + json.dumps(
                dict(
                    rank=rank,
                    model=args.model,
                    tokens=n,
                    tx_rx={port: [a - b for a, b in zip(pair, before[port])] for port, pair in after.items()},
                )
            ),
            flush=True,
        )
    gathered = [None] * world
    dist.all_gather_object(gathered, (digest, times))
    if rank == 0:
        result = dict(
            model=args.model,
            mode=args.mode,
            batch_invariant=args.batch_invariant,
            weft_kernel="decode" if use_decode else "training",
            tokens_per_rank=n,
            capacity_per_rank=cap,
            ep=world,
            dtype="nvfp4" if nvfp4 else "bf16",
            samples=args.samples,
            warmups=args.warmups,
            replays=args.replays if decode else 1,
            routing_sha256=[d for d, _ in gathered],
        )
        for backend, name in enumerate(names):
            samples = list(map(max, zip(*(pair[backend] for _, pair in gathered))))
            result[name + "_us"] = samples
            result[name + "_p50_us"] = statistics.median(samples)
        print("BLACKWELL_FORWARD_JSON " + json.dumps(result), flush=True)
    rt.close()
    if training is not None:
        training.close()
    if ref is not None:
        ref.destroy()


@torch.no_grad()
def main():

    p = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(p)
    p.add_argument("--mode", choices=("decode", "prefill"), required=True)
    p.add_argument("--precision", choices=("bf16", "fp4"), default="bf16")
    p.add_argument("--batch-invariant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokens", type=int, nargs="+")
    p.add_argument("--decode-capacity", type=int, help="Prepare a fixed capacity shared by all decode buckets")
    p.add_argument("--samples", type=int)
    p.add_argument("--warmups", type=int)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument("--compare-weight-layouts", action="store_true", help="Compare standard and tiled inference weights on the same route trace")
    p.add_argument("--compare-training", action="store_true", help="Compare decode with the training kernel on the same inputs and require bitwise parity")
    p.add_argument("--compare-mega", action="store_true", help="Also time the optional single-node DeepGEMM baseline")
    p.add_argument(
        "--mega-mixed", action="store_true", help="Compare NVFP4 latency to pre-staged upstream MXFP8 x MXFP4"
    )
    p.add_argument("--port-counters", action="store_true", help="Record per-host physical bond TX/RX bytes")
    p.add_argument("--seed", type=int)
    p.add_argument("--route-bias-std", type=float, default=0.5)
    p.add_argument("--token-std-fraction", type=float, default=0.1)
    args = p.parse_args()
    if args.compare_training and (
        args.mode != "decode" or os.environ["WORLD_SIZE"] != os.environ["LOCAL_WORLD_SIZE"]
    ):
        p.error("--compare-training requires single-node decode")
    args.compare_mega |= args.mega_mixed
    if args.compare_mega:
        import deep_gemm as dg
    resolve_model(args)
    decode = args.mode == "decode"
    for key, default in dict(
        tokens=[64, 512] if decode else [8192, 20480],
        samples=20 if decode else 30,
        warmups=5 if decode else 10,
        seed=1300 if decode else 123,
    ).items():
        if getattr(args, key) is None:
            setattr(args, key, default)
    if args.decode_capacity is not None and (not decode or args.decode_capacity < max(args.tokens)):
        p.error("--decode-capacity requires decode mode and must cover all token counts")
    if min(*args.tokens, args.samples, args.replays) < 1 or args.warmups < 0:
        p.error("Invalid token or iteration count")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    check_blackwell()
    dist.init_process_group("nccl", device_id=torch.device("cuda", torch.cuda.current_device()))
    try:
        group = dist.group.WORLD
        torch.manual_seed(args.seed + group.rank())
        owned = args.experts // group.size()
        w1 = (
            torch.randn(owned, 2 * args.intermediate, args.hidden, device="cuda", dtype=torch.bfloat16) * 0.03
        )
        w2 = torch.randn(owned, args.hidden, args.intermediate, device="cuda", dtype=torch.bfloat16) * 0.03
        packed = (
            dg.transform_weights_for_mega_moe(w1, w2)
            if args.compare_mega and args.precision == "bf16" and group.size() == int(os.environ["LOCAL_WORLD_SIZE"])
            else None
        )
        if args.precision == "fp4":
            from weftmoe.nvfp4 import prepare_nvfp4_weights

            packed = prepare_nvfp4_weights(w1, w2)
        if args.mega_mixed and (
            args.precision != "fp4" or group.size() != int(os.environ["LOCAL_WORLD_SIZE"])
        ):
            p.error("--mega-mixed requires single-node NVFP4")
        mixed = pack_mega_mixed(w1, w2) if args.mega_mixed else None
        tiled = prepare_tiled_weights(w1, w2, args.precision == "fp4") if args.compare_weight_layouts else None
        for n in args.tokens:
            measure(args, n, w1, w2, packed, group, mixed, tiled)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
