#!/usr/bin/env python3
"""Validate the installed v2 boundary without any repository include path."""

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "weftmoe/version.hpp",
    "weftmoe/api/errors.hpp",
    "weftmoe/api/notify_dispatch.hpp",
    "weftmoe/config/runtime_config.hpp",
    "weftmoe/cuda/activation_transport.hpp",
    "weftmoe/cuda/backward.hpp",
    "weftmoe/cuda/control_protocol.hpp",
    "weftmoe/cuda/device_notify.hpp",
    "weftmoe/cuda/nccl_count_exchange.hpp",
    "weftmoe/metadata/dispatch_handle.hpp",
    "weftmoe/scoreboard/workspace_plan.hpp",
    "weftmoe/transport/activation_protocol.hpp",
    "weftmoe/transport/count_exchange.hpp",
}


def absolute_library_path(value):
    entries = []
    for entry in value.split(os.pathsep):
        if not entry:
            continue
        path = Path(entry)
        entries.append(str(path if path.is_absolute() else (ROOT / path).resolve()))
    return os.pathsep.join(entries)


with tempfile.TemporaryDirectory(prefix="weftmoe-install-") as temporary:
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

    output = root / "minimal_forward"
    # The installed .pc file is link metadata for package-aware clients.  The
    # acceptance test deliberately has no pkg-config runtime dependency.
    flags = ["-I", str(prefix / "include"), "-L", str(prefix / "lib"),
             "-lweftmoe"]
    command = [
        os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"),
        "-Wno-deprecated-gpu-targets",
        "-std=c++17",
        "-I",
        str(prefix / "include"),
        str(ROOT / "examples/minimal_forward.cpp"),
        "-o",
        str(output),
    ] + flags
    subprocess.run(command, cwd=root, check=True)
    if not output.is_file():
        raise SystemExit("minimal installed-prefix program produced no executable")
    subprocess.run(
        [str(output)], cwd=root, check=True, timeout=180,
        env={**os.environ,
             "LD_LIBRARY_PATH": str(prefix / "lib") + os.pathsep +
             absolute_library_path(
                 os.environ.get("NCCL_RUNTIME_LIB_DIR", "")) + os.pathsep +
             absolute_library_path(os.environ.get("LD_LIBRARY_PATH", ""))},
    )

    library = prefix / "lib/libweftmoe.so.2.0.0"
    dynamic = subprocess.check_output(["readelf", "-d", str(library)], text=True)
    if "SONAME" not in dynamic or "libweftmoe.so.2" not in dynamic:
        raise SystemExit("installed library has the wrong SONAME")
    if "RPATH" in dynamic or "RUNPATH" in dynamic:
        raise SystemExit("installed library unexpectedly contains an rpath")

print("[PASS] installed v2 headers, libweftmoe SONAME, link and run")
