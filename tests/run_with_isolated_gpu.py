"""Run a torchrun test with the one-visible-GPU layout used by AReaL actors.

Example: torchrun --nproc_per_node=8 tests/run_with_isolated_gpu.py \
    tests/torchrun_tensor_runtime.py
"""

import os
import runpy
import sys

rank = int(os.environ["LOCAL_RANK"])
devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
os.environ["LOCAL_RANK"] = "0"
script = sys.argv.pop(1)
runpy.run_path(script, run_name="__main__")
