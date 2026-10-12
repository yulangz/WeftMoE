"""Build matched diagnostic/production artifacts for the borrowed-warp PF path."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("label")
parser.add_argument("--timeline", action="store_true")
parser.add_argument("--local", action="store_true")
parser.add_argument("--define", action="append", default=[])
args = parser.parse_args()
output = ROOT / "build/pf_main_timeline" / args.label
output.mkdir(parents=True, exist_ok=True)
flags = ["-lineinfo", "-Xptxas=-v,--warn-on-spills",
         "-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1",
         "-I" + str(ROOT / ("configs/pf_inference_flash_20261008" if args.local else
                            "configs/pf_network_l132_20261009")),
         "-DWEFTMOE_COMMUNICATION_PHASES=0", "-DWEFTMOE_PF_COMMUNICATION_COPY=0",
         "-DWEFTMOE_PF_NETWORK_WORKERS=1", "-DWEFTMOE_PF_BORROW_ALL_CTA=1",
         "-DWEFTMOE_PF_BORROW_CONTROL_REGISTERS=88", *("-D" + d for d in args.define)]
env = dict(os.environ, NVCCFLAGS=" ".join(flags),
           WEFTMOE_DEVICE_TIMELINE=str(int(args.timeline)))
library = output / "libweftmoe.so"
with (output / "build.log").open("w") as log:
    subprocess.run([sys.executable, str(ROOT / "build_native.py"), "lib",
                    "--output", str(library), "--jobs", "4"],
                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/build_tensor_runtime.py"),
                    "--library", str(library)], cwd=ROOT, env=env,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
print(output, flush=True)
