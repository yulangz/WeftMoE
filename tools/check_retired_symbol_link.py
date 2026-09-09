#!/usr/bin/env python3
"""Prove an explicitly declared retired diagnostic symbol cannot link."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import tempfile


RETIRED = (
    "_ZN7weftmoe4cuda39download_device_metadata_for_validation"
    "ERKNS0_20DeviceDispatchHandleERKNS0_13CudaExecutionE"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-dir", required=True, type=pathlib.Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="weftmoe-retired-link-") as directory:
        root = pathlib.Path(directory)
        source = root / "retired.cpp"
        source.write_text(
            f'extern "C" void retired() asm("{RETIRED}");\n'
            "int main() { retired(); return 0; }\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            ["c++", str(source), f"-L{args.library_dir}", "-lweftmoe",
             f"-Wl,-rpath,{args.library_dir}", "-o", str(root / "retired")],
            text=True, capture_output=True,
        )
        if result.returncode == 0:
            print("retired diagnostic symbol unexpectedly linked")
            return 1
    print("retired diagnostic symbol: expected link failure")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
