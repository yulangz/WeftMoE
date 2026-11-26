"""Offline SM103 decode tile/raster search with bitwise checks against training and prefill."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.models import add_model_arguments, resolve_model
from benchmarks.workload import decode_routes
from tools.autotune_common import choose_confirmed_candidate, confirmation_finalists
from weftmoe.blackwell import prepare_bf16_weights
from weftmoe.blackwell_cutlass import (
    DECODE_TILES,
    decode_config,
    decode_key,
    load_decode_plan,
)
from weftmoe.nvfp4 import NVFP4Scales, prepare_nvfp4_weights
from weftmoe.runtime import TensorRuntime


def publish(path, key, winners):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        profiles = json.loads(path.read_text()) if path.exists() else {}
        profiles.setdefault(key, {}).update(
            {str(n): config for n, config in winners.items()}
        )
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path.parent, delete=False
        ) as tmp:
            json.dump(profiles, tmp, indent=2, sort_keys=True)
            tmp.write("\n")
        os.replace(tmp.name, path)


def check_bits(actual, reference, label):
    valid = (actual.view(torch.int16) == reference.view(torch.int16)).all().int()
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if not valid.item():
        raise RuntimeError(f"Bitwise training mismatch: {label}")


def benchmark_candidates(configs, traces, input_buffers, graphs, replays, seed):
    """Randomize order identically on every rank and take rank-max per trace."""
    rng = random.Random(seed)
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    samples = {config: [] for config in configs}
    for index, (inputs, reference) in enumerate(traces):
        for tensor, value in zip(input_buffers, inputs, strict=True):
            tensor.copy_(value)
        order = list(configs)
        rng.shuffle(order)
        for config in order:
            graph, out = graphs[config]
            dist.barrier()
            torch.cuda.synchronize()
            start.record()
            for _ in range(replays):
                graph.replay()
            stop.record()
            stop.synchronize()
            samples[config].append(start.elapsed_time(stop) * 1000 / replays)
            check_bits(out, reference, f"{config}, trace {index}")
    times = torch.tensor(list(samples.values()), device="cuda", dtype=torch.float64)
    dist.all_reduce(times, op=dist.ReduceOp.MAX)
    return {
        cfg: statistics.median(row)
        for cfg, row in zip(samples, times.tolist(), strict=True)
    }


def tune_bucket(args, n, configs, incumbent, runtimes, weights, packed, group, *, tiled_weights=None):
    decode, prefill, train = runtimes
    rank, world = group.rank(), group.size()
    h, e = args.hidden, args.experts
    rng = torch.Generator(device="cuda").manual_seed(args.seed + n * world + rank)
    routes = decode_routes(args, torch.empty(n * world, e, device="cuda"))

    def inference(rt, inputs):
        params = packed if packed is not None else weights
        if rt is decode and tiled_weights is not None:
            params = tiled_weights
        if packed is not None:
            return rt.inference_nvfp4(0, *inputs, params)
        return rt.inference(0, *inputs, *params)

    traces = []
    for idx in range(1 + args.samples + args.confirm_samples):
        x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16, generator=rng) * 0.03
        ids, gates = (v.chunk(world)[rank].clone() for v in next(routes))
        if idx == 0:
            # Stress duplicate experts, invalid routes and a single hot owner.
            ids.fill_(0)
            ids[::2, 0] = -1
            gates[::3, -1] = 0
        inputs = (x, ids, gates)
        ref = train.forward(*inputs, *weights).output
        finite = ref.isfinite().all().int()
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise RuntimeError("Non-finite training reference")
        check_bits(inference(prefill, inputs), ref, "prefill (training kernel)")
        traces.append((inputs, ref))

    buffers = tuple(value.clone() for value in traces[0][0])
    graphs = {}
    for config in configs:
        decode.native.buffer.decode_tuning[n] = config
        inference(decode, buffers)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = inference(decode, buffers)
        # Consecutive replays of the same node must see the new generation,
        # including the first output after switching owners.
        for inputs, reference in (traces[0], traces[1], traces[0]):
            for buf, value in zip(buffers, inputs, strict=True):
                buf.copy_(value)
            graph.replay()
            check_bits(out, reference, f"{config}, changing routes on graph replay")
        graphs[config] = (graph, out)
    split = args.samples + 1
    scores = benchmark_candidates(
        configs, traces[1:split], buffers, graphs, args.replays, args.seed
    )
    candidates = [
        dict(name=str(c), config=c, selection_metric_us=t) for c, t in scores.items()
    ]
    finalists = confirmation_finalists(candidates, str(incumbent), 3)
    confirm = benchmark_candidates(
        [f["config"] for f in finalists],
        traces[split:],
        buffers,
        graphs,
        args.replays,
        args.seed + 1,
    )
    for item in finalists:
        item["aggregate_metric_us"] = confirm[item["config"]]
    winner, decision = choose_confirmed_candidate(
        finalists, str(incumbent), args.min_gain
    )
    if rank == 0:
        print(
            "SM103_TUNE "
            + json.dumps(
                dict(
                    tokens=n,
                    precision=args.precision,
                    weight_layout="tiled" if tiled_weights is not None else "standard",
                    model=args.model,
                    search=candidates,
                    confirmed=finalists,
                    decision=decision,
                )
            ),
            flush=True,
        )
    return winner["config"]


@torch.no_grad()
def tune(args, group):
    ranks, rank = group.size(), group.rank()
    if torch.cuda.get_device_capability() != (10, 3) or ranks != int(
        os.environ["LOCAL_WORLD_SIZE"]
    ):
        raise ValueError("SM103 decode tuning requires a single-node SM103 group")
    e, h, i, k = args.experts, args.hidden, args.intermediate, args.top_k
    key = decode_key(
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count,
        ranks,
        e,
        h,
        i,
        k,
        args.precision,
    )
    plan = load_decode_plan(key)
    cap = args.capacity
    base = dict(
        group=group,
        experts=e,
        hidden=h,
        intermediate=i,
        top_k=k,
        max_tokens=cap,
        max_rows=cap * ranks * k,
        precision=args.precision,
        batch_invariant=True,
    )
    incumbents = {n: plan.get(n, decode_config(ranks, e, n, k)) for n in args.tokens}
    # Size one symmetric window for all candidates before capturing any graphs.
    max_m = max(*args.tiles, *(m for m, _ in incumbents.values()))
    decode = TensorRuntime(
        **base, training=False, decode_only=True, decode_plan={cap: (max_m, False)}
    )
    prefill = TensorRuntime(**base, training=False)
    train = TensorRuntime(
        **base,
        training=True,
        nvfp4_scales=NVFP4Scales() if args.precision == "nvfp4" else None,
    )
    try:
        torch.manual_seed(args.seed + rank)
        weights = tuple(
            torch.randn(s, device="cuda", dtype=torch.bfloat16) * 0.03
            for s in ((e // ranks, 2 * i, h), (e // ranks, h, i))
        )
        packed = prepare_nvfp4_weights(*weights) if args.precision == "nvfp4" else None
        tiled_weights = None
        if args.tiled_weights:
            tiled_weights = (prepare_nvfp4_weights(*weights, tiled=True)
                             if packed is not None else prepare_bf16_weights(*weights))
        winners = {}
        for n, incumbent in incumbents.items():
            configs = list(
                dict.fromkeys(
                    [incumbent]
                    + [(m, raster) for m in args.tiles for raster in (False, True)]
                )
            )
            winners[n] = tune_bucket(
                args, n, configs, incumbent, (decode, prefill, train), weights, packed, group, tiled_weights=tiled_weights
            )
        if rank == 0:
            publish(args.output, key, winners)
    finally:
        for rt in (train, prefill, decode):
            rt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--precision", choices=["bf16", "nvfp4"], required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 16, 256, 512])
    parser.add_argument("--capacity", type=int, default=512)
    parser.add_argument("--tiled-weights", action="store_true", help="Tune decode with prepared tiled weights")
    parser.add_argument(
        "--tiles", type=int, nargs="+", choices=DECODE_TILES, default=DECODE_TILES
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--confirm-samples", type=int, default=9)
    parser.add_argument("--replays", type=int, default=40)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=1.5,
        help="Minimum confirmed latency reduction in percent",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--route-bias-std", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolve_model(args)
    if (
        min(*args.tokens, args.replays, args.samples, args.confirm_samples) < 1
        or max(args.tokens) > args.capacity
    ):
        parser.error(
            "Positive token/iteration counts and capacity covering all buckets are required"
        )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    try:
        tune(args, dist.group.WORLD)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
