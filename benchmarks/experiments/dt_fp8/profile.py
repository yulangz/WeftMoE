"""Bounded EP8 single-pass NCU captures; counter timings are not benchmarks.

Example: python .../profile.py --output build/dt-profile --models qwen35b
Only rank 2 is profiled. Never use multi-pass kernel replay with rank barriers.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import sys

from autotune import ROOT, MODELS, atomic_json, idle_gpus, run_command

GROUPS = {
    "shared": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,"
              "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,"
              "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    "basic": "gpu__time_duration.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,"
             "dram__throughput.avg.pct_of_peak_sustained_elapsed,"
             "smsp__issue_active.avg.pct_of_peak_sustained_active,smsp__warps_eligible.avg.per_cycle_active",
    "stalls": "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,"
              "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,"
              "smsp__warp_issue_stalled_barrier_per_warp_active.pct,"
              "smsp__warp_issue_stalled_wait_per_warp_active.pct",
}


def rank_entry(args):
    command = [sys.executable, str(Path(__file__).resolve()), "--device", "--tokens",
               str(args.tokens[0]), "--variant", args.variant]
    if int(os.environ["LOCAL_RANK"]) == 2:
        command = [shutil.which("ncu"), "--target-processes", "application-only", "--replay-mode", "kernel",
            "--profile-from-start", "off", "--clock-control", "none", "--cache-control", "none",
            "--kernel-name-base", "demangled", "--kernel-name", "regex:.*weftmoe_dt::kernel.*",
            "--launch-count", "1", "--metrics", os.environ["DT_NCU_METRICS"], "--export",
            os.environ["DT_NCU_OUTPUT"], "--force-overwrite", *command]
    os.execv(command[0], command)


def device_entry(args):
    import torch
    import torch.distributed as dist
    from weftmoe.runtime import TensorRuntime
    from weftmoe.fp8 import prepare_fp8_weights, quantize_fp8

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    h, i, k = MODELS[os.environ["DT_PROFILE_MODEL"]]
    n, experts = args.tokens[0], 256
    torch.manual_seed(123)
    w1 = torch.randn(experts // world, 2 * i, h, device="cuda", dtype=torch.bfloat16) * .03
    w2 = torch.randn(experts // world, h, i, device="cuda", dtype=torch.bfloat16) * .03
    weights = prepare_fp8_weights(w1, w2, version=0, training=False)
    rt = TensorRuntime(group=dist.group.WORLD, experts=experts, top_k=k, hidden=h, intermediate=i,
        max_tokens=n, max_rows=n * world * k + experts * 128, training=False, decode_only=True,
        fp8_workspace=args.variant != "bf16")
    torch.manual_seed(900 + rank)
    x = torch.randn(n, h, device="cuda", dtype=torch.bfloat16) * .03
    routes = torch.rand(n, experts, device="cuda").topk(k, dim=1).indices.int().contiguous()
    gates = torch.rand(n, k, device="cuda")
    gates /= gates.sum(1, keepdim=True)
    inp = quantize_fp8(x) if args.variant == "fp8_prequant" else x
    def call():
        return (rt.inference(0, x, routes, gates, w1, w2) if args.variant == "bf16"
                else rt.inference_fp8(0, inp, routes, gates, weights))
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    # Graph replay avoids host submission differences between ranked processes.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    graph.replay()
    torch.cuda.synchronize()
    expected = output.clone()
    dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()
    graph.replay()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    valid = torch.tensor(int(torch.equal(expected, output) and torch.isfinite(output).all()), device="cuda")
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    assert valid.item(), "profiled replay changed output"
    del graph
    rt.invalidate_inference_layer(0)
    rt.close()
    dist.destroy_process_group()


def collect(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "summary.json"
    records = json.loads(summary_path.read_text())["records"] if summary_path.exists() else []
    for model in args.models:
        for candidate in args.candidates:
            native = args.native_root.resolve() / model / candidate / "native"
            for n in args.tokens:
                for variant in args.variants:
                    if variant == "bf16" and candidate != "default":
                        continue  # FP8-only tuning knobs cannot change BF16.
                    for group in args.groups:
                        name = f"{model}-{candidate}-n{n}-{variant}-{group}"
                        stem = output / name
                        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", NCCL_DEBUG="WARN",
                            CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                            LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64",
                            WEFTMOE_TENSOR_EXTENSION=str(next(native.glob("_runtime*.so"))),
                            DT_PROFILE_MODEL=model, DT_NCU_OUTPUT=str(stem), DT_NCU_METRICS=GROUPS[group])
                        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8",
                            str(Path(__file__).resolve()), "--rank", "--tokens", str(n), "--variant", variant]
                        rec = dict(model=model, candidate=candidate, tokens=n, variant=variant, group=group,
                            name=name, gpu="NVIDIA L20X", ep=8, profiled_rank=2, native=str(native),
                            metrics_requested=GROUPS[group], command=command, status="running")
                        print("PROFILE " + name, flush=True)
                        try:
                            idle_gpus()
                            if not stem.with_suffix(".ncu-rep").exists():
                                run_command(command, cwd=ROOT, env=env, log=stem.with_suffix(".log"), timeout=90)
                            passes = re.findall(r"(\d+) pass(?:es)?", stem.with_suffix(".log").read_text())
                            if passes != ["1"]:
                                raise RuntimeError(f"unsafe/invalid replay count: {passes}")
                            run_command([shutil.which("ncu"), "--import", str(stem.with_suffix(".ncu-rep")),
                                "--csv", "--page", "raw", "--print-units", "base"], cwd=ROOT, env=env,
                                log=stem.with_suffix(".csv"), timeout=30)
                            lines = stem.with_suffix(".csv").read_text().splitlines()
                            header = next(j for j, line in enumerate(lines) if line.startswith('"ID",'))
                            rows = list(csv.DictReader(lines[header:]))
                            if len(rows) != 2:
                                raise RuntimeError(f"expected units and one kernel row, got {len(rows)}")
                            rec.update(status="passed", replay_passes=1, metrics={key:
                                dict(value=float(rows[1][key].replace(",", "")), unit=rows[0][key])
                                for key in GROUPS[group].split(",")})
                        except (RuntimeError, StopIteration) as error:
                            rec.update(status="failed", error=str(error))
                        records = [r for r in records if r["name"] != name] + [rec]
                        atomic_json(output / "summary.json", dict(
                            method="EP8 rank2; one-pass NCU only; no clock/cache control; separate from latency benchmarks",
                            records=records))
                        idle_gpus()
                        if rec["status"] != "passed":
                            raise RuntimeError(f"profile failed; stopped safely: {name}: {rec['error']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native-root", type=Path, default=ROOT / "build/dt-fp8-autotune-m64/jobs")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--candidates", nargs="+", default=["default", "m64-s3-r1-c132"])
    parser.add_argument("--tokens", nargs="+", type=int, default=[32, 512])
    # Stall counter collection can require replay; opt in explicitly. A timeout
    # is recorded as a failure, never interpreted as a production kernel hang.
    parser.add_argument("--groups", nargs="+", choices=GROUPS, default=["basic"])
    parser.add_argument("--variants", nargs="+", default=["bf16", "fp8_online"],
                        choices=["bf16", "fp8_online", "fp8_prequant"])
    parser.add_argument("--variant", default="fp8_online")
    parser.add_argument("--rank", action="store_true")
    parser.add_argument("--device", action="store_true")
    args = parser.parse_args()
    if args.rank:
        rank_entry(args)
    elif args.device:
        device_entry(args)
    else:
        if args.output is None:
            parser.error("--output is required")
        collect(args)
