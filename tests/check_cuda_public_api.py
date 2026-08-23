#!/usr/bin/env python3
"""Compile-negative proofs for every retired pre-coordinator protocol entry."""

import os
import subprocess
import tempfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
cases = {
    "old_wait": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { megamoe::cuda::CudaNotifyOperation op; (void)op.wait(); }
''',
    "wait_preflight": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { megamoe::cuda::CudaNotifyOperation op; (void)op.wait_preflight(); }
''',
    "raw_notify_start": r'''
#include "megamoe/cuda/device_notify.hpp"
int main() { (void)megamoe::cuda::start_cuda_rank_local_notify; }
''',
    "control_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_control_collective; }
''',
    "config_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_config_collective; }
''',
    "count_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_count_collective; }
''',
    "summary_starter": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::start_summary_collective; }
''',
    "internal_operation": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclControlOperation operation; }
''',
    "internal_backend": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclAllGatherCountExchange backend; }
''',
    "bootstrap_hex": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { megamoe::cuda::NcclBootstrapId id; (void)id.hex(); }
''',
    "bootstrap_parse": r'''
#include "megamoe/cuda/nccl_count_exchange.hpp"
int main() { (void)megamoe::cuda::parse_nccl_bootstrap_id("00"); }
''',
}
with tempfile.TemporaryDirectory() as directory:
    temporary = Path(directory)
    for name, source in cases.items():
        path = temporary / f"{name}.cpp"
        path.write_text(source)
        command = [os.environ.get("CXX", "c++"), "-std=c++17", "-Werror",
                   "-I", str(root / "include"), "-c", str(path),
                   "-o", str(temporary / f"{name}.o")]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, check=False)
        if result.returncode == 0:
            raise SystemExit(f"retired API unexpectedly compiled: {name}")
        if not result.stderr.strip():
            raise SystemExit(f"{name} failed without a compiler diagnostic")
print("[PASS] 11 retired raw/preflight/collective/bootstrap text APIs are compile-negative")
