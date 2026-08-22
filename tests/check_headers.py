#!/usr/bin/env python3
"""Compile every public header alone and guard removed unsafe surface names."""

import os
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
HEADERS = sorted((ROOT / "include").rglob("*.hpp"))
CXX = os.environ.get("CXX", "c++")


def main() -> None:
    transport = (ROOT / "include/megamoe/transport/count_exchange.hpp").read_text()
    for forbidden in ("IntegerBuffer", "uintptr_t", ".address"):
        if forbidden in transport:
            raise RuntimeError(f"unsafe public transport surface remains: {forbidden}")

    with tempfile.TemporaryDirectory(prefix="megamoe-headers-") as temp:
        source = pathlib.Path(temp) / "header.cpp"
        for header in HEADERS:
            relative = header.relative_to(ROOT / "include")
            source.write_text(f'#include "{relative.as_posix()}"\n')
            subprocess.run(
                [
                    CXX,
                    "-std=c++17",
                    "-Wall",
                    "-Wextra",
                    "-Wpedantic",
                    "-Wconversion",
                    "-Wshadow",
                    "-Werror",
                    "-I",
                    str(ROOT / "include"),
                    "-fsyntax-only",
                    str(source),
                ],
                check=True,
            )
    print(f"[PASS] {len(HEADERS)} public headers compile standalone with -Werror")


if __name__ == "__main__":
    main()
