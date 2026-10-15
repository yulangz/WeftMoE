"""Build matched diagnostic/production artifacts for the borrowed-warp PF path."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("label")
parser.add_argument("--timeline", action="store_true")
parser.add_argument("--local", action="store_true")
parser.add_argument("--profile", type=Path,
                    help="Directory containing the combined tuning header")
parser.add_argument("--define", action="append", default=[])
args = parser.parse_args()
output = ROOT / "build/pf_main_timeline" / args.label
output.mkdir(parents=True, exist_ok=True)
flags = ["-lineinfo", "-Xptxas=-v,--warn-on-spills",
         "-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1",
         "-I" + str(args.profile.resolve() if args.profile else
                    ROOT / ("configs/pf_inference_flash_20261008" if args.local else
                            "configs/pf_network_l132_20261009")),
         *("-D" + d for d in args.define)]
env = dict(os.environ, NVCCFLAGS=" ".join(flags),
           WEFTMOE_DEVICE_TIMELINE=str(int(args.timeline)))
library = output / "libweftmoe.so"
manifest = dict(
    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True).strip(),
    source_diff_sha256=hashlib.sha256(subprocess.check_output(
        ["git", "diff", "HEAD"], cwd=ROOT)).hexdigest(),
    flags=flags, timeline=args.timeline)
with (output / "build.log").open("w") as log:
    subprocess.run([sys.executable, str(ROOT / "build_native.py"), "lib",
                    "--output", str(library), "--jobs", "4"],
                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/build_tensor_runtime.py"),
                    "--library", str(library)], cwd=ROOT, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
manifest["library_sha256"] = hashlib.file_digest(library.open("rb"), "sha256").hexdigest()
(output / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(output, flush=True)
