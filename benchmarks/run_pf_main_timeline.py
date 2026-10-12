"""Serial EP16 main-pipeline diagnostics, numerical checks, or uninstrumented timing."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("artifact", type=Path)
parser.add_argument("label")
parser.add_argument("mode", choices=("trace", "performance", "verify", "skew", "inference"))
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--env", action="append", default=[])
args = parser.parse_args()
output = ROOT / "build/pf_main_timeline/results"
command = [sys.executable, str(ROOT / "benchmarks/run_pf_network.py"),
           "--artifact", str(args.artifact.resolve()), "--label", args.label,
           "--output", str(output), "--master-port", str(args.port),
           "--benchmark", str(ROOT / ("benchmarks/torchrun_megamoe_prefill.py"
                                      if args.mode == "inference" else "tools/benchmark_training.py"))]
for value in ("WEFTMOE_PF_MIN_RING_DEPTH=2", "WEFTMOE_PF_CREDIT_WINDOWS=2",
              "WEFTMOE_PF_WG3_PUBLICATION_PLAN=1", "WEFTMOE_PF_DEDUPLICATE_RETURN_READY=1",
              f"NCCL_DEBUG_FILE={output / args.label}.nccl.%h.%p.log", *args.env):
    command += ["--env", value]
command += ["--", "--backend", "pf", "--hidden", "4096", "--intermediate", "2048",
            "--experts", "256", "--top-k", "6", "--tokens", "8192", "20480",
            "--ring-tokens", "512" if args.mode == "skew" else "4096", "--ring-depth", "2"]
if args.mode in ("verify", "skew"):
    command += ["--verify", "--warmups", "0", "--samples", "2"]
    if args.mode == "skew":
        command += ["--route-bias-std", "8"]
elif args.mode == "trace":
    command += ["--forward-only", "--warmups", "2", "--samples", "2",
                "--timeline-dir", str(ROOT / "build/pf_main_timeline" / args.label)]
elif args.mode == "inference":
    command += ["--prepared-weights", "--profile-device", "--warmups", "8", "--samples", "24"]
else:
    command += ["--forward-only", "--prime-trace", "--warmups", "8", "--samples", "24"]
subprocess.run(command, cwd=ROOT, check=True)
