#!/usr/bin/env python3
"""Single-node PF/V3/MegaMoE inference on the training benchmark's dynamic trace.

Measures eager forward + device synchronize, including MegaMoE input staging.
MegaMoE packs immutable weights once. PF eager packs on each call by default;
--prepared-weights reuses its layer weight cache for a matched comparison.
No saved training activations, backward, or optimizer are included.
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
    parser.add_argument("--backend", choices=("pf", "v3", "megamoe"), required=True)
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
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--token-std-fraction", type=float, default=0.1)
    parser.add_argument("--route-bias-std", type=float, default=0.5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--profile-device", action="store_true",
                        help="Also report PF's native kernel clock / MegaMoE's staged CUDA interval; excludes weight repack")
    parser.add_argument("--prepared-weights", action="store_true",
                        help="Reuse PF's immutable layer weights across changing token counts/routes")
    args = parser.parse_args()
    if args.backend == "pf":
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
    if world != int(os.environ["LOCAL_WORLD_SIZE"]) or args.experts % world:
        parser.error("requires single-node EP with an equal expert split")
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
        capacity = max(max(w["tokens"]) for w in workloads)
        incoming = torch.stack([torch.bincount(r.flatten().long(), minlength=args.experts)
                                for r, _ in batches]).cuda()
        dist.all_reduce(incoming)
        max_rows = int(incoming.reshape(len(batches), world, owned).sum(-1).max()) + owned * 128
        if args.backend != "megamoe":
            runtime = TensorRuntime(group=dist.group.WORLD, experts=args.experts,
                                    top_k=args.top_k, hidden=args.hidden,
                                    intermediate=args.intermediate, max_tokens=capacity,
                                    max_rows=max_rows, training=False)
        else:
            buffer = megamoe_buffer(deep_gemm, args, capacity)
        output_storage = torch.empty(capacity, args.hidden, device="cuda", dtype=torch.bfloat16)
        samples, device_samples, device_phases, errors = [], [], [], []
        if args.profile_device and args.backend == "megamoe":
            device_begin, device_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        rng_state = generator.get_state()
        priming = 0 if args.verify else len(batches)
        for step, (cpu_routes, cpu_gates) in enumerate(batches * (1 if args.verify else 2)):
            if step == priming:
                generator.set_state(rng_state)
            routes, gates = cpu_routes.cuda(), cpu_gates.cuda()
            x = rand(routes.shape[0], args.hidden)
            output = output_storage[:x.shape[0]]
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            if args.backend != "megamoe":
                if args.profile_device:
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
            if args.verify:
                expected = reference(x, routes, gates, w1, w2).float()
                delta = output.float() - expected
                error = dict(relative_l2=(delta.norm() / expected.norm().clamp_min(1e-30)).item(),
                             max_abs=delta.abs().max().item())
                assert error["relative_l2"] < 0.01, error
                torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=1e-4)
                errors.append(error)
            elif step >= priming + args.warmups:
                samples.append(elapsed)
                if args.profile_device:
                    if args.backend == "megamoe":
                        device_us = device_begin.elapsed_time(device_end) * 1000
                    device_samples.append(device_us)
                    if args.backend != "megamoe":
                        device_phases.append({key: value for key, value in profile.items()
                                              if key != "output"})
        records = [None] * world
        dist.all_gather_object(records, dict(samples_us=samples,
                                            device_samples_us=device_samples,
                                            device_phases=device_phases, errors=errors))
        workspace = (runtime.native.usage() if args.backend != "megamoe"
                     else dict(symm_bytes=buffer.buffer.nbytes, ring_tokens=buffer.num_ring_tokens))
        if rank == 0:
            print("MEGAMOE_PREFILL_JSON " + json.dumps(dict(
                backend=args.backend, world=world, nominal_tokens_per_rank=nominal,
                hidden=args.hidden, intermediate=args.intermediate, experts=args.experts,
                top_k=args.top_k, warmups=args.warmups, samples=args.samples,
                priming_steps=priming, token_std_fraction=args.token_std_fraction,
                route_bias_std=args.route_bias_std, workload=workloads,
                wall_slowest_rank_p50_us=None if args.verify else statistics.median(
                    map(max, zip(*(r["samples_us"] for r in records)))),
                scope="eager inference + synchronize; includes input staging; no training cache",
                weights_prepared_once=args.backend == "megamoe" or args.prepared_weights,
                device_slowest_rank_p50_us=(statistics.median(
                    map(max, zip(*(r["device_samples_us"] for r in records))))
                    if device_samples else None),
                device_scope=("PF: native fused-kernel clock; MegaMoE: CUDA interval including input staging and enqueue gaps; excludes weight repack"
                              if args.profile_device else None),
                megamoe_buffer_mib=args.megamoe_buffer_mib,
                megamoe_minimum_ring=args.megamoe_minimum_ring,
                workspace_rank0=workspace, per_rank=records,
                package=deep_gemm.__file__ if args.backend == "megamoe" else
                    os.environ.get("WEFTMOE_TENSOR_EXTENSION"))), flush=True)
        if args.backend != "megamoe":
            runtime.close()
        else:
            buffer.destroy()
            del buffer
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
