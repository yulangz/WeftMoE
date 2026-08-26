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

    output = root / "minimal_training"
    # The installed .pc file is link metadata for package-aware clients.  The
    # acceptance test deliberately has no pkg-config runtime dependency.
    flags = ["-I", str(prefix / "include"), "-L", str(prefix / "lib"),
             "-lmegamoe"]
    command = [
        os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
        "-std=c++17",
        "-I",
        str(prefix / "include"),
        str(ROOT / "examples/minimal_training.cpp"),
        "-o",
        str(output),
    ] + flags
    subprocess.run(command, cwd=root, check=True)
    if not output.is_file():
        raise SystemExit("minimal installed-prefix program produced no executable")
    subprocess.run(
        [str(output)], cwd=root, check=True,
        env={**os.environ,
             "LD_LIBRARY_PATH": str(prefix / "lib") + os.pathsep +
             os.environ.get("NCCL_RUNTIME_LIB_DIR", "") + os.pathsep +
             os.environ.get("LD_LIBRARY_PATH", "")},
    )

    library = prefix / "lib/libmegamoe.so.2.0.0"
    dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    if "SONAME" not in dynamic or "libmegamoe.so.2" not in dynamic:
        raise SystemExit("installed library has the wrong SONAME")
    if "RPATH" in dynamic or "RUNPATH" in dynamic:
        raise SystemExit("installed library unexpectedly contains an rpath")

print("[PASS] installed v2 headers, libmegamoe SONAME, link and run")
