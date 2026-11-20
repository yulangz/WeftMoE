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
from benchmarks.workload import make_workload
from weftmoe.runtime import TensorRuntime


def decode_trace(args, n, rank, world):
    # Mirror SGLang's global routing RNG, including the capture and parity inputs.
    rng = torch.Generator(device="cuda").manual_seed(args.seed + n * world)
    bias = torch.randn(args.experts, device="cuda", generator=rng) * args.route_bias_std
    logits = torch.empty(n * world, args.experts, device="cuda")
    trace = []
    digest = hashlib.sha256()
    for step in range(2 + args.warmups + args.samples):
        scores, ids = logits.normal_(generator=rng).add_(bias).topk(args.top_k)
        if step < 2:
            continue
        ids, gates = ids.int(), scores.softmax(-1)
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


def measure(args, n, w1, w2, packed, group, mixed_weights=None):
    import deep_gemm as dg

    rank, world = group.rank(), group.size()
    decode = args.mode == "decode"
    nvfp4 = args.precision == "fp4"
    if decode:
        trace, digest = decode_trace(args, n, rank, world)
    else:
        trace = make_workload(args, n, rank, "cuda")
        digest = hashlib.sha256()
        for ids, gates in trace:
            digest.update(len(ids).to_bytes(8, "little"))
            digest.update(ids.numpy().tobytes())
            digest.update(gates.numpy().tobytes())
        digest = digest.hexdigest()
    capacity = torch.tensor(max(len(ids) for ids, _ in trace), device="cuda")
    dist.all_reduce(capacity, op=dist.ReduceOp.MAX)
    cap = int(capacity)
    rt = TensorRuntime(
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
        if packed is not None and group.size() == int(os.environ["LOCAL_WORLD_SIZE"])
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

    def run_weft(count):
        if nvfp4:
            return rt.inference_nvfp4(0, x[:count], routes[:count], weights[:count], packed)
        return rt.inference(0, x[:count], routes[:count], weights[:count], w1, w2)

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
    times = [[], []]
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
        outputs = [None, None]
        for backend in reversed(range(len(runners))) if step % 2 else range(len(runners)):
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
        valid = valid.int()
        dist.all_reduce(valid, op=dist.ReduceOp.MIN)
        if not valid.item():
            raise RuntimeError(f"WeftMoE/pre-staged parity failed at step {step}")
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
            tokens_per_rank=n,
            ep=world,
            dtype="nvfp4" if nvfp4 else "bf16",
            samples=args.samples,
            warmups=args.warmups,
            replays=args.replays if decode else 1,
            routing_sha256=[d for d, _ in gathered],
        )
        for backend, name in enumerate(
            (
                "weft",
                "mega_mxfp8_mxfp4" if mixed_weights else "nvfp4_prestaged" if nvfp4 else "mega_prestaged",
            )[: len(runners)]
        ):
            samples = list(map(max, zip(*(pair[backend] for _, pair in gathered))))
            result[name + "_us"] = samples
            result[name + "_p50_us"] = statistics.median(samples)
        print("BLACKWELL_FORWARD_JSON " + json.dumps(result), flush=True)
    rt.close()
    if ref is not None:
        ref.destroy()


@torch.no_grad()
def main():
    import deep_gemm as dg

    p = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(p)
    p.add_argument("--mode", choices=("decode", "prefill"), required=True)
    p.add_argument("--precision", choices=("bf16", "fp4"), default="bf16")
    p.add_argument("--batch-invariant", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tokens", type=int, nargs="+")
    p.add_argument("--samples", type=int)
    p.add_argument("--warmups", type=int)
    p.add_argument("--replays", type=int, default=50)
    p.add_argument(
        "--mega-mixed", action="store_true", help="Compare NVFP4 latency to pre-staged upstream MXFP8 x MXFP4"
    )
    p.add_argument("--port-counters", action="store_true", help="Record per-host physical bond TX/RX bytes")
    p.add_argument("--seed", type=int)
    p.add_argument("--route-bias-std", type=float, default=0.5)
    p.add_argument("--token-std-fraction", type=float, default=0.1)
    args = p.parse_args()
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
            if args.precision == "bf16" and group.size() == int(os.environ["LOCAL_WORLD_SIZE"])
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
        for n in args.tokens:
            measure(args, n, w1, w2, packed, group, mixed)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
