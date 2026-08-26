#!/usr/bin/env python3
"""Validate the installed v2 boundary without any repository include path."""

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "megamoe/version.hpp",
    "megamoe/api/errors.hpp",
    "megamoe/api/notify_dispatch.hpp",
    "megamoe/config/runtime_config.hpp",
    "megamoe/cuda/activation_transport.hpp",
    "megamoe/cuda/control_protocol.hpp",
    "megamoe/cuda/device_notify.hpp",
    "megamoe/cuda/nccl_count_exchange.hpp",
    "megamoe/metadata/dispatch_handle.hpp",
    "megamoe/scoreboard/workspace_plan.hpp",
    "megamoe/transport/activation_protocol.hpp",
    "megamoe/transport/count_exchange.hpp",
}


with tempfile.TemporaryDirectory(prefix="megamoe-install-") as temporary:
    root = Path(temporary)
    subprocess.run(
        ["make", "install", f"DESTDIR={root}", "PREFIX=/usr/local"],
        cwd=ROOT,
        check=True,
    )
    prefix = root / "usr/local"
    installed = {
        str(path.relative_to(prefix / "include"))
        for path in (prefix / "include").rglob("*.hpp")
    }
    if installed != EXPECTED:
        missing = sorted(EXPECTED - installed)
        extra = sorted(installed - EXPECTED)
        raise SystemExit(f"installed header mismatch: missing={missing}, extra={extra}")

    output = root / "minimal_training.o"
    command = [
        os.environ.get("CXX", "c++"),
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(prefix / "include"),
        "-c",
        str(ROOT / "examples/minimal_training.cpp"),
        "-o",
        str(output),
    ]
    subprocess.run(command, cwd=root, check=True)
    if not output.is_file():
        raise SystemExit("minimal installed-prefix program produced no object")

print("[PASS] installed v2 headers and minimal training program")
