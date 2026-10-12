#!/usr/bin/env python3
"""PF/DT/V3/MegaMoE inference on a shared route trace (MegaMoE/DT: one node).

Measures eager forward + device synchronize, including MegaMoE input staging.
MegaMoE packs immutable weights once. PF eager packs on each call by default;
--prepared-weights reuses its layer weight cache for a matched comparison.
No saved training activations, backward, or optimizer are included.
Fixed-token PF/DT comparisons can use CUDA graphs; runtime capacity and kernel
tracing are explicit so small batches cannot silently benchmark a fallback.
"""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.training_reference import gather_rows
from tools.benchmark_training import make_workload
from weftmoe.runtime import TensorRuntime


def megamoe_buffer(deep_gemm, args, capacity):
    group = dist.group.WORLD
    if args.megamoe_buffer_mib is None and not args.megamoe_minimum_ring:
        return deep_gemm.get_symm_buffer_for_mega_moe(
            group, args.experts, capacity, args.top_k, args.hidden,
            args.intermediate, mma_type="bf16xbf16")
    from deep_gemm.mega import SymmBuffer

    api = deep_gemm._C
    alignment = api.get_token_alignment_for_mega_moe()
    capacity = (capacity + alignment - 1) // alignment * alignment
    minimum, maximum = api.get_ring_limit_for_mega_moe(
        capacity, args.experts // group.size(), args.top_k, group.size())
    if args.megamoe_minimum_ring:
        return SymmBuffer(group, args.experts, capacity, args.top_k, args.hidden,
                          args.intermediate, minimum, mma_type="bf16xbf16")

    def size(rows):
        return api.get_symm_buffer_size_for_mega_moe(
            group.size(), args.experts, capacity, args.top_k, args.hidden,
            args.intermediate, "bf16xbf16", "swiglu", rows)[0]

    budget = int(args.megamoe_buffer_mib * 2**20)
    minimum_bytes = size(minimum)
    if minimum_bytes > budget:
        raise ValueError(f"MegaMoE minimum legal buffer is {minimum_bytes / 2**20:.3f} MiB "
                         f"({minimum} ring rows); requested {args.megamoe_buffer_mib} MiB")
    low, high = minimum // alignment, maximum // alignment
    while low < high:
        middle = (low + high + 1) // 2
        if size(middle * alignment) <= budget:
            low = middle
        else:
            high = middle - 1
    return SymmBuffer(group, args.experts, capacity, args.top_k, args.hidden,
                      args.intermediate, low * alignment, mma_type="bf16xbf16")


def reference(x, routes, gates, w1, w2):
    lengths = [None] * dist.get_world_size()
    dist.all_gather_object(lengths, x.shape[0])
    all_x, all_routes, all_gates = (gather_rows(v, lengths) for v in (x, routes, gates))
    output = torch.zeros_like(all_x, dtype=torch.float32)
    rank = dist.get_rank()
    for expert in range(w1.shape[0]):
        rows, slots = torch.where(all_routes == rank * w1.shape[0] + expert)
        if not rows.numel():
            continue
        a, b = (all_x[rows] @ w1[expert].T).chunk(2, dim=-1)
        u = (torch.nn.functional.silu(a.float()) * b.float()
             * all_gates[rows, slots, None]).to(x.dtype)
        output.index_add_(0, rows, (u @ w2[expert].T).float())
    dist.all_reduce(output)
    begin = sum(lengths[:rank])
    return output[begin:begin + lengths[rank]].to(x.dtype)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pf", "dt", "v3", "megamoe"), required=True)
    parser.add_argument("--deepgemm-package", type=Path)
    capacity = parser.add_mutually_exclusive_group()
    capacity.add_argument("--megamoe-buffer-mib", type=float,
                        help="Cap total symmetric allocation, retaining MegaMoE's worst-route safety bounds")
    capacity.add_argument("--megamoe-minimum-ring", action="store_true",
                         help="Use the smallest ring legal for arbitrary routes at this token capacity")
    parser.add_argument("--ring-tokens", type=int, default=2048)
    parser.add_argument("--ring-depth", type=int, default=2)
    for name in ("hidden", "intermediate", "experts", "top-k"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 64, 8192, 20480])
    parser.add_argument("--pool-tokens", type=int, default=0,
                        help="Minimum runtime capacity, independent of the measured token count")
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--token-std-fraction", type=float, default=0.1)
    parser.add_argument("--route-bias-std", type=float, default=0.5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-kernel", action="store_true",
                        help="During verification, trace the first call and require the selected PF/DT entry")
    parser.add_argument("--trace-stage-kernels", action="store_true",
                        help="Separately profile single-node PF inference notify/main CUDA durations")
    parser.add_argument("--profile-device", action="store_true",
                        help="Also report PF's native kernel clock / MegaMoE's staged CUDA interval; excludes weight repack")
    parser.add_argument("--graph-replays", type=int, default=0,
                        help="Measure a PF/DT CUDA graph with this many calls; requires fixed token counts")
    parser.add_argument("--prepared-weights", action="store_true",
                        help="Reuse PF's immutable layer weights across changing token counts/routes")
    parser.add_argument("--forward-cache", choices=("off", "on"),
                        help="PF forward-only cache experiment; both modes use the same native tape API")
    args = parser.parse_args()
    if args.trace_stage_kernels and (args.backend != "pf" or args.verify or
                                    args.forward_cache or args.graph_replays):
        parser.error("--trace-stage-kernels requires eager PF inference without verification")
    if args.forward_cache and (args.backend != "pf" or args.profile_device):
        parser.error("--forward-cache requires PF without --profile-device")
    if args.verify_kernel and (not args.verify or args.backend not in ("pf", "dt")):
        parser.error("--verify-kernel requires --verify and --backend pf/dt")
    if args.graph_replays and (args.backend not in ("pf", "dt") or args.token_std_fraction
                              or args.forward_cache or args.profile_device):
        parser.error("--graph-replays requires PF/DT, --token-std-fraction 0, no cache/profile flags")
    if args.backend in ("pf", "dt"):
        os.environ.pop("WEFTMOE_DISABLE_STATIC_FORWARD_V4", None)
        os.environ["WEFTMOE_STATIC_RING_TOKENS"] = str(args.ring_tokens)
        os.environ["WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS"] = str(args.ring_depth)
    else:
        os.environ["WEFTMOE_DISABLE_STATIC_FORWARD_V4"] = "1"
        os.environ.pop("WEFTMOE_STATIC_RING_TOKENS", None)
        os.environ.pop("WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS", None)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.experts % world:
        parser.error("requires an equal expert split")
    if args.backend in ("dt", "megamoe") and world != int(os.environ["LOCAL_WORLD_SIZE"]):
        parser.error("DT/MegaMoE require single-node EP")
    if args.backend == "megamoe":
        if args.deepgemm_package is None:
            parser.error("MegaMoE requires --deepgemm-package")
        sys.path.insert(0, str(args.deepgemm_package.resolve()))
        import deep_gemm
        deep_gemm.set_num_sms(torch.cuda.get_device_properties(0).multi_processor_count)
    generator = torch.Generator(device="cuda").manual_seed(args.seed + rank)

    def rand(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16,
                           generator=generator) * 0.03

    owned = args.experts // world
    w1 = rand(owned, 2 * args.intermediate, args.hidden)
    w2 = rand(owned, args.hidden, args.intermediate)
    if args.backend == "megamoe":
        packed_w1, packed_w2 = deep_gemm.transform_weights_for_mega_moe(w1, w2)
    for nominal in args.tokens:
        batches = make_workload(args, nominal, rank, "cuda")
        digest = hashlib.sha256()
        for routes, gates in batches:
            digest.update(routes.shape[0].to_bytes(8, "little"))
            digest.update(routes.numpy().tobytes())
            digest.update(gates.numpy().tobytes())
        workload = dict(tokens=[r.shape[0] for r, _ in batches], sha256=digest.hexdigest())
        workloads = [None] * world
        dist.all_gather_object(workloads, workload)
        capacity = max(args.pool_tokens, max(max(w["tokens"]) for w in workloads))
        incoming = torch.stack([torch.bincount(r.flatten().long(), minlength=args.experts)
                                for r, _ in batches]).cuda()
        dist.all_reduce(incoming)
        max_rows = int(incoming.reshape(len(batches), world, owned).sum(-1).max()) + owned * 128
        if args.backend != "megamoe":
            runtime = TensorRuntime(group=dist.group.WORLD, experts=args.experts,
                                    top_k=args.top_k, hidden=args.hidden,
                                    intermediate=args.intermediate, max_tokens=capacity,
                                    max_rows=max_rows, training=args.forward_cache == "on",
                                    **({"stream": torch.cuda.current_stream()}
                                       if args.forward_cache else {}))
        else:
            buffer = megamoe_buffer(deep_gemm, args, capacity)
        output_storage = torch.empty(capacity, args.hidden, device="cuda", dtype=torch.bfloat16)
        samples, device_samples, device_phases, errors = [], [], [], []
        measurement_intervals_ns = []
        kernel_names = []
        output_sha256 = []
        stage_probe = None
        graph = None
        if args.graph_replays:
            graph_x = torch.empty(nominal, args.hidden, device="cuda", dtype=torch.bfloat16)
            graph_routes = torch.empty(nominal, args.top_k, device="cuda", dtype=torch.int32)
            graph_gates = torch.empty(nominal, args.top_k, device="cuda", dtype=torch.float32)
            device_begin, device_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        if args.profile_device and args.backend == "megamoe":
            device_begin, device_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        rng_state = generator.get_state()
        priming = 0 if args.verify else len(batches)
        for step, (cpu_routes, cpu_gates) in enumerate(batches * (1 if args.verify else 2)):
            if args.trace_stage_kernels and step == priming + args.warmups:
                stage_probe = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
                stage_probe.start()
            if step == priming:
                generator.set_state(rng_state)
            routes, gates = cpu_routes.cuda(), cpu_gates.cuda()
            x = rand(routes.shape[0], args.hidden)
            output = output_storage[:x.shape[0]]
            if args.graph_replays:
                graph_x.copy_(x)
                graph_routes.copy_(routes)
                graph_gates.copy_(gates)
                if graph is None:
                    graph_output = runtime.inference(0, graph_x, graph_routes, graph_gates, w1, w2)
                    torch.cuda.synchronize()
                    dist.barrier()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(args.graph_replays):
                            runtime.inference(0, graph_x, graph_routes, graph_gates, w1, w2)
            dist.barrier()
            torch.cuda.synchronize()
            probe = None
            if args.verify_kernel and step == 0:
                probe = torch.profiler.profile(activities=[
                    torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                probe.start()
            interval_start = time.time_ns()
            started = time.perf_counter_ns()
            if args.graph_replays:
                device_begin.record()
                graph.replay()
                device_end.record()
                output = graph_output
            elif args.backend != "megamoe":
                if args.forward_cache:
                    tape = runtime.native.forward(x, routes, gates, w1, w2,
                        **({"layer": 0} if args.prepared_weights else {}))
                    output = tape.output
                    del tape
                elif args.profile_device:
                    profile = runtime.native.profile_inference(
                        x, routes, gates, w1, w2, torch.cuda.current_stream().cuda_stream,
                        *((0,) if args.prepared_weights else ()))
                    output, device_us = profile["output"], profile["total_us"]
                else:
                    output = runtime.inference_eager(
                        x, routes, gates, w1, w2,
                        **({"layer": 0} if args.prepared_weights else {}))
            else:
                if args.profile_device:
                    device_begin.record()
                buffer.x[:x.shape[0]].copy_(x)
                buffer.topk_idx[:x.shape[0]].copy_(routes)
                buffer.topk_weights[:x.shape[0]].copy_(gates)
                deep_gemm.bf16_mega_moe(output, packed_w1, packed_w2, buffer,
                                       activation_clamp=None, fast_math=True)
                if args.profile_device:
                    device_end.record()
            torch.cuda.synchronize()
            elapsed = (time.perf_counter_ns() - started) / 1000
            interval_end = time.time_ns()
            if probe is not None:
                probe.stop()
                kernel_names = sorted({event.name for event in probe.events()
                                       if event.device_type == torch.autograd.DeviceType.CUDA})
                expected_entry = (("weftmoe_pf::local::kernel" if runtime.nodes == 1
                                   else "weftmoe_pf::network::kernel") if args.backend == "pf"
                                  else "weftmoe_dt::kernel")
                assert any(expected_entry in name for name in kernel_names), kernel_names
                if os.environ.get("WEFTMOE_PF_SPLIT_INFERENCE_NOTIFY"):
                    assert any("local::inference_notify_kernel" in name for name in kernel_names), kernel_names
            if args.verify:
                expected = reference(x, routes, gates, w1, w2).float()
                delta = output.float() - expected
                error = dict(relative_l2=(delta.norm() / expected.norm().clamp_min(1e-30)).item(),
                             max_abs=delta.abs().max().item())
                # Fail collectively: otherwise successful peers enter the next
                # reference collective while a failed rank tears down NCCL.
                failed = torch.tensor(int(error["relative_l2"] >= 0.01 or not
                    torch.allclose(output.float(), expected, rtol=0.02, atol=1e-4)),
                    device="cuda")
                dist.all_reduce(failed, op=dist.ReduceOp.MAX)
                assert not failed.item(), {"rank": rank, **error}
                errors.append(error)
                output_sha256.append(hashlib.sha256(
                    output.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest())
            elif step >= priming + args.warmups:
                samples.append(elapsed)
                measurement_intervals_ns.append((interval_start, interval_end))
                if args.profile_device or args.graph_replays:
                    if args.backend == "megamoe" or args.graph_replays:
                        device_us = device_begin.elapsed_time(device_end) * 1000 / (args.graph_replays or 1)
                    device_samples.append(device_us)
                    if args.profile_device and args.backend != "megamoe":
                        device_phases.append({key: value for key, value in profile.items()
                                              if key != "output"})
        stage_kernels = {}
        if stage_probe is not None:
            stage_probe.stop()
            split = "WEFTMOE_PF_SPLIT_INFERENCE_NOTIFY" in os.environ
            for label, name, count in (
                ("notify", "local::inference_notify_kernel", args.samples if split else 0),
                ("main", "weftmoe_pf::local::kernel", args.samples),
                ("mini_notify", "training_count_notify_kernel", 0),
            ):
                events = sorted((event for event in stage_probe.events()
                                 if event.device_type == torch.autograd.DeviceType.CUDA
                                 and name in event.name), key=lambda event: event.time_range.start)
                assert len(events) == count, (label, len(events), count)
                stage_kernels[label] = [event.time_range.elapsed_us() for event in events]
        records = [None] * world
        dist.all_gather_object(records, dict(samples_us=samples,
                                            device_samples_us=device_samples,
                                            measurement_intervals_ns=measurement_intervals_ns,
                                            device_phases=device_phases, errors=errors,
                                            kernel_names=kernel_names,
                                            stage_kernels_us=stage_kernels,
                                            output_sha256=output_sha256))
        workspace = (runtime.native.usage() if args.backend != "megamoe"
                     else dict(symm_bytes=buffer.buffer.nbytes, ring_tokens=buffer.num_ring_tokens))
        if rank == 0:
            print("MEGAMOE_PREFILL_JSON " + json.dumps(dict(
                backend=args.backend, world=world, nominal_tokens_per_rank=nominal,
                pool_tokens=capacity,
                hidden=args.hidden, intermediate=args.intermediate, experts=args.experts,
                top_k=args.top_k, warmups=args.warmups, samples=args.samples,
                priming_steps=priming, token_std_fraction=args.token_std_fraction,
                graph_replays=args.graph_replays,
                route_bias_std=args.route_bias_std, workload=workloads,
                wall_slowest_rank_p50_us=None if args.verify else statistics.median(
                    map(max, zip(*(r["samples_us"] for r in records)))),
                scope=("CUDA graph batch wall + synchronize; see device_scope for per-call kernel timing"
                       if args.graph_replays else
                       "eager forward + tape release + synchronize; includes input staging and per-call cache preparation"
                       if args.forward_cache else
                       "eager inference + synchronize; includes input staging; no training cache"),
                forward_cache=args.forward_cache,
                weights_prepared_once=args.backend == "megamoe" or args.prepared_weights,
                device_slowest_rank_p50_us=(statistics.median(
                    map(max, zip(*(r["device_samples_us"] for r in records))))
                    if device_samples else None),
                device_scope=("CUDA graph events divided by graph_replays; includes fused source staging; excludes external input updates and preparation"
                              if args.graph_replays else
                              "PF/DT: native device envelope (split PF includes notify and launch gap); MegaMoE: CUDA interval including input staging and enqueue gaps; excludes weight repack"
                              if args.profile_device else None),
                megamoe_buffer_mib=args.megamoe_buffer_mib,
                megamoe_minimum_ring=args.megamoe_minimum_ring,
                workspace_rank0=workspace, per_rank=records,
                package=deep_gemm.__file__ if args.backend == "megamoe" else
                    os.environ.get("WEFTMOE_TENSOR_EXTENSION"))), flush=True)
        if args.backend != "megamoe":
            del graph
            runtime.close()
        else:
            buffer.destroy()
            del buffer
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
