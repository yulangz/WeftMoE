"""Compare retained PF Return lifetime and sender-placement choices.

Builds and GPU measurements are separate commands: do not overlap them.
All GPU jobs run serially using the existing two-node launcher.
The retired expert-partitioned direct experiment is at commit 7bd4d75.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from autotune_common import candidate_digest
from autotune_training import generated_header

VARIANTS = {
    "bound": (1, 0),
    "split": (1, 1),
    "rail": (0, 1),
}
SHAPES = {
    "qwen": (2048, 512, 8, 1024, "pf_qwen35_compact_20261013"),
    "flash": (4096, 2048, 6, 2048, "pf_flash_strategy_20261015"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("build", "measure", "verify", "trace"))
    p.add_argument("--output", type=Path, default=ROOT / "build/pf_return_mechanism_20261015")
    p.add_argument("--shapes", nargs="+", choices=SHAPES, default=list(SHAPES))
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    p.add_argument("--routes", nargs="+", default=["random", "balanced"])
    p.add_argument("--tokens", nargs="+", type=int, default=[8192, 20480])
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--warmups", type=int, default=4)
    p.add_argument("--timeline", action="store_true", help="Build an instrumented artifact")
    p.add_argument("--label", default="r1")
    p.add_argument("--artifact-tag", default="20261015",
                   help="Use a new tag when rebuilding; keep prior binaries immutable")
    p.add_argument("--port", type=int, default=28700)
    p.add_argument("--window", type=int)
    p.add_argument("--bias", type=float, default=0.5)
    args = p.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    port = args.port
    for shape in args.shapes:
        h, i, topk, window, profile = SHAPES[shape]
        for variant in args.variants:
            timeline = args.timeline or args.action == "trace"
            suffix = "_trace" if timeline else ""
            name = f"return_mechanism_{shape}_{variant}_{args.artifact_tag}{suffix}"
            artifact = ROOT / "build/pf_main_timeline" / name
            if args.action == "build":
                config = copy.deepcopy(json.loads(
                    (ROOT / "configs" / profile / "candidate.json").read_text())["config"])
                rails, independent = VARIANTS[variant]
                config["v4"]["return_rails"] = rails
                overrides = dict(WEFTMOE_PF_RETURN_INDEPENDENT_TX=independent)
                experiment = dict(config=config, overrides=overrides)
                digest = candidate_digest(experiment)
                header = generated_header(config)
                header = header[:header.rfind("#define WEFTMOE_FORWARD_TUNING_ID")]
                header += "".join(f"#define {k} {v}\n" for k, v in overrides.items())
                header += f'#define WEFTMOE_FORWARD_TUNING_ID "{digest}"\n'
                directory = args.output / "profiles" / f"{shape}_{variant}{suffix}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "weftmoe_forward_tuning_generated.hpp").write_text(header)
                (directory / "candidate.json").write_text(json.dumps(experiment, indent=2))
                command = [sys.executable, "benchmarks/build_pf_main_timeline.py", name,
                           "--profile", str(directory)]
                if timeline:
                    command += ["--timeline"]
                print("BUILD", shape, variant, suffix, flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                continue
            for route in args.routes:
                label = f"{args.label}_{shape}_{variant}_{route}_{args.action}"
                cli = ["--backend", "pf", "--hidden", str(h), "--intermediate", str(i),
                       "--experts", "256", "--top-k", str(topk), "--ring-tokens",
                       str(args.window or window), "--ring-depth", "2",
                       "--route-distribution", route, "--route-bias-std", str(args.bias),
                       "--tokens", *map(str, args.tokens)]
                if args.action == "verify":
                    cli += ["--verify", "--warmups", "0", "--samples", "2"]
                else:
                    cli += ["--forward-only", "--prime-trace", "--warmups",
                            str(args.warmups), "--samples", str(args.samples),
                            "--report-window-padding"]
                    if timeline:
                        cli += ["--timeline-dir", str(args.output / label)]
                command = [sys.executable, "benchmarks/run_pf_network.py", "--artifact",
                           str(artifact), "--label", label, "--output", str(args.output),
                           "--nodes", "2", "--master-port", str(port),
                           "--env", "WEFTMOE_PF_MIN_RING_DEPTH=2",
                           "--env", "WEFTMOE_PF_CREDIT_WINDOWS=4", "--benchmark",
                           "tools/benchmark_training.py", "--", *cli]
                print("RUN", label, flush=True)
                subprocess.run(command, cwd=ROOT, check=True)
                for line in (args.output / f"{label}_ep16_node0.log").read_text().splitlines():
                    if line.startswith("TRAINING_CYCLE_JSON "):
                        result = json.loads(line.split(" ", 1)[1])
                        print("RESULT", label, result["nominal_tokens_per_rank"],
                              result["forward_ready_slowest_rank_p50_us"] / 1000,
                              flush=True)
                    elif line.startswith("TRAINING_PARITY_JSON "):
                        result = json.loads(line.split(" ", 1)[1])
                        print("PARITY", label, result["nominal_tokens_per_rank"],
                              result["step"], flush=True)
                port += 1


if __name__ == "__main__":
    main()
