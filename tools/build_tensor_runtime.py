"""Build the application tensor adapter against an already-built native library."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sysconfig

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    library = args.library.resolve()
    output = library.parent / ("_runtime" + sysconfig.get_config_var("EXT_SUFFIX"))
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            "-DWEFTMOE_ENABLE_GIN_RDMA="
            + os.environ.get("WEFTMOE_ENABLE_GIN_RDMA", "1"),
            "-DWEFTMOE_DEVICE_TIMELINE=" + os.environ.get("WEFTMOE_DEVICE_TIMELINE", "0"),
            "-I" + str(root / "src"),
            "-I" + str(root / "include"),
            "-I" + str(Path(torch.__file__).parent / "include"),
            "-I" + sysconfig.get_paths()["include"],
            "-I"
            + str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"),
            str(root / "src/python/runtime.cpp"),
            str(library),
            "-L" + str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "lib64"),
            "-lcudart",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(output),
        ],
        check=True,
    )
    # The native library advertises this SONAME.
    soname = library.parent / "libweftmoe.so"
    if soname != library and not soname.exists():
        soname.symlink_to(library.name)
    print(output)


if __name__ == "__main__":
    main()
