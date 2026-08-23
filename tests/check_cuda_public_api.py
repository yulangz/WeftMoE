#!/usr/bin/env python3
"""Compile-negative proof that the pre-control protocol shortcut is private."""

import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = r'''
#include "megamoe/cuda/device_notify.hpp"
int main() {
  megamoe::cuda::CudaNotifyOperation operation;
  (void)operation.wait_preflight();
  (void)megamoe::cuda::start_cuda_rank_local_notify;
}
'''
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "forbidden.cpp"
    path.write_text(source)
    command = [os.environ.get("CXX", "c++"), "-std=c++17", "-Werror",
               "-I", str(root / "include"), "-c", str(path),
               "-o", str(Path(directory) / "forbidden.o")]
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode == 0:
        raise SystemExit("dangerous CUDA preflight/start API is still public")
    required = ("incomplete type", "start_cuda_rank_local_notify")
    if not any(marker in result.stderr for marker in required):
        raise SystemExit("compile failed for an unexpected reason:\n" + result.stderr)
print("[PASS] public API cannot start/wait device preflight outside coordinator")
