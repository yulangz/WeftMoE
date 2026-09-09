#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build NCCL 2.29.7 with GIN RoCE LAG connection-order fixes.

Backports upstream PR #2150 and offsets context connection order by sender
rank to balance fixed-context return traffic. The patch changes QP connection order without changing device APIs or kernels.
It applies to round-robin queue affinity; it does not change NIC configuration.

The result is a private dependency, not a system installation. Pass the printed
library directory to --nccl-lib-dir on both nodes, or prepend it to
LD_LIBRARY_PATH in the application. Existing NCCL installations are untouched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


NCCL_COMMIT = "b91894bd5b190c874d98a017f93f5daa515b65d0"
NCCL_TAG = "v2.29.7-1"
NCCL_URL = "https://github.com/NVIDIA/nccl.git"
PATCHED_SOURCE = "src/transport/net_ib/gdaki/gin_host_gdaki.cc"
TOOL_DIR = Path(__file__).resolve().parent


def run(argv, *, cwd=None, check=True, capture_output=False):
    return subprocess.run(argv, cwd=cwd, check=check, text=True,
                          capture_output=capture_output)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_source(work_dir: Path) -> Path:
    source = work_dir / "source"
    patch = TOOL_DIR / "nccl" / "nccl-2.29.7-gin-lag.patch"
    if not source.exists():
        run(["git", "clone", "--depth", "1", "--branch", NCCL_TAG,
             NCCL_URL, str(source)])
    actual = run(["git", "rev-parse", "HEAD"], cwd=source,
                 capture_output=True).stdout.strip()
    if actual != NCCL_COMMIT:
        raise RuntimeError(f"Expected NCCL commit {NCCL_COMMIT}, found {actual}")
    changed = run(["git", "diff", "HEAD", "--name-only"], cwd=source,
                  capture_output=True).stdout.splitlines()
    expected_file = PATCHED_SOURCE
    if any(name != expected_file for name in changed):
        raise RuntimeError("NCCL checkout has unrelated tracked modifications")
    # Check the exact preimage, or recognize this patch on an incremental run.
    check = run(["git", "apply", "--check", str(patch)], cwd=source,
                check=False, capture_output=True)
    if check.returncode == 0:
        if changed:
            raise RuntimeError("NCCL QP source has pre-existing modifications")
        run(["git", "apply", str(patch)], cwd=source)
    else:
        run(["git", "apply", "--reverse", "--check", str(patch)], cwd=source)
        # Reverse-apply alone can also succeed when unrelated edits exist in
        # the same file. Compare the full resulting diff before reusing it.
        actual_patch = run(["git", "diff", "--no-ext-diff", "--no-color",
                            "--", expected_file], cwd=source,
                           capture_output=True).stdout
        if actual_patch != "diff --git " + patch.read_text().split("diff --git ", 1)[1]:
            raise RuntimeError("NCCL QP source differs from the supported patch")
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path,
                        default=TOOL_DIR.parent / "artifacts" / "nccl-gin-lag")
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--cuda-home", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    source = prepare_source(work_dir)
    if args.prepare_only:
        print(source)
        return
    build_dir = work_dir / "build"
    # This dependency build targets the SM90 GPUs used by WeftMoE.
    command = ["make", "-j", str(args.jobs), "src.build",
               f"BUILDDIR={build_dir}", f"CUDA_HOME={args.cuda_home.resolve()}",
               "NVCC_GENCODE=-gencode=arch=compute_90,code=sm_90"]
    run(command, cwd=source)
    library = build_dir / "lib" / "libnccl.so.2.29.7"
    manifest = {
        "nccl_commit": NCCL_COMMIT,
        "patch_sha256": sha256(TOOL_DIR / "nccl" / "nccl-2.29.7-gin-lag.patch"),
        "upstream_fix": "https://github.com/NVIDIA/nccl/pull/2150",
        "library_sha256": sha256(library),
        "library": str(library),
        "build_command": command,
    }
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Patched NCCL library directory: {library.parent}")


if __name__ == "__main__":
    main()
