"""Matched EP8/EP16 validation and timing of the standalone PF planner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT.parent / "WeftMoE-experiments"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("phase", choices=("verify", "skew", "verify-inference", "verify-graph",
                                     "performance", "trace", "inference", "graph", "sweep", "cycle"))
parser.add_argument("--nodes", type=int, choices=(1, 2), required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--tag", default="")
args = parser.parse_args()
out = ROOT / "build/pf_notify_production/results"
old = (EXPERIMENTS / "warp-budget-20261011/build/local_warp_budget/pull2" if args.nodes == 1
       else EXPERIMENTS / "borrow-training-20261011/build/cached")
new = ROOT / "build/pf_notify_production" / ("local" if args.nodes == 1 else "network")
variants = ("old", "new", "new", "old") if args.phase == "performance" else ("new",)
if args.phase == "cycle":
    variants = ("old", "new", "new", "old")
if args.phase == "sweep":
    variants = ("b2048", "b8192", "b4096", "b8192", "b2048", "b4096")
for i, variant in enumerate(variants):
    label = f"{args.phase}_n{args.nodes}_{i}_{variant}"
    if args.tag:
        label += "_" + args.tag
    inference = args.phase in ("inference", "graph", "verify-inference", "verify-graph")
    benchmark = ROOT / ("benchmarks/torchrun_megamoe_prefill.py" if inference
                        else "tools/benchmark_training.py")
    cli = [sys.executable, str(ROOT / "benchmarks/run_pf_network.py"),
           "--artifact", str(old if variant == "old" else new), "--label", label,
           "--nodes", str(args.nodes), "--output", str(out), "--master-port", str(args.port + i),
           "--benchmark", str(benchmark)]
    for env in ("WEFTMOE_PF_MIN_RING_DEPTH=2", "WEFTMOE_PF_CREDIT_WINDOWS=2",

                f"NCCL_DEBUG_FILE={out / label}.nccl.%h.%p.log"):
        cli += ["--env", env]
    cli += ["--", "--backend", "pf", "--hidden", "4096", "--intermediate", "2048",
            "--experts", "256", "--top-k", "6", "--tokens", "8192", "20480",
            "--ring-tokens", str(512 if args.phase == "skew" else
                                int(variant[1:]) if args.phase == "sweep" else
                                2048 if args.nodes == 1 else 4096), "--ring-depth", "2"]
    if args.phase in ("verify", "skew"):
        cli += ["--verify", "--trace-kernels", "--warmups", "0", "--samples", "2"]
        if args.phase == "skew":
            cli += ["--route-bias-std", "8"]
    elif inference:
        cli += ["--prepared-weights"]
        if args.phase.startswith("verify"):
            cli += ["--verify", "--verify-kernel", "--warmups", "0", "--samples", "2"]
        else:
            cli += ["--warmups", "8", "--samples", "24"]
        cli += (["--graph-replays", "2", "--token-std-fraction", "0"]
                if "graph" in args.phase else ["--profile-device"])
    else:
        cli += ["--prime-trace", "--warmups", "8", "--samples", "24"]
        if args.phase != "cycle":
            cli += ["--forward-only"]
        cli += ["--report-window-padding"]
        if args.phase == "trace":
            cli += ["--trace-stage-kernels"]
    print("START", label, flush=True)
    subprocess.run(cli, cwd=ROOT, check=True)
    for line in (out / f"{label}_ep{args.nodes * 8}_node0.log").read_text().splitlines():
        if line.startswith(("TRAINING_CYCLE_JSON ", "MEGAMOE_PREFILL_JSON ")):
            row = json.loads(line.partition(" ")[2])
            print("RESULT", label, row["nominal_tokens_per_rank"],
                  row.get("forward_ready_slowest_rank_p50_us") or
                  row.get("device_slowest_rank_p50_us") or row["wall_slowest_rank_p50_us"],
                  flush=True)
