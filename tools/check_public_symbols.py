#!/usr/bin/env python3
"""Verify or explicitly refresh the libmegamoe v2 dynamic-symbol contract."""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


BANNED = (
    "::testing",
    "for_testing",
    "synthetic",
    "fault",
    "validation",
    "mutation",
    "download",
    "::internal",
    "::Impl",
    "Bridge",
    "DeviceActivationTransport",
    "LocalCudaP2pActivationTransport",
    "NcclSocketFallbackActivationTransport",
    "GinRdmaActivationTransport",
)


def symbols(binary: pathlib.Path) -> list[tuple[str, str]]:
    raw = subprocess.check_output(
        ["nm", "-D", "--defined-only", str(binary)], text=True
    )
    result: list[tuple[str, str]] = []
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        name = fields[-1].split("@", 1)[0]
        if name == "MEGAMOE_2":
            continue
        demangled = subprocess.check_output(["c++filt", name], text=True).strip()
        result.append((name, demangled))
    return sorted(set(result))


def approved(candidates: list[tuple[str, str]]) -> list[str]:
    result = []
    for raw, demangled in candidates:
        if not demangled.startswith("megamoe::"):
            continue
        lowered = demangled.lower()
        if any(term.lower() in lowered for term in BANNED):
            continue
        result.append(raw)
    return sorted(set(result))


def write_contract(path: pathlib.Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def write_version_script(path: pathlib.Path, names: list[str]) -> None:
    lines = ["MEGAMOE_2 {", "  global:"]
    lines.extend(f"    {name};" for name in names)
    lines.extend(["  local: *;", "};", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=pathlib.Path)
    parser.add_argument("--whitelist", required=True, type=pathlib.Path)
    parser.add_argument("--version-script", type=pathlib.Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()

    actual_pairs = symbols(args.binary)
    if args.update:
        names = approved(actual_pairs)
        write_contract(args.whitelist, names)
        if args.version_script:
            write_version_script(args.version_script, names)
        print(f"updated ABI v2 contract: {len(names)} symbols")
        return 0

    expected = {
        line.strip()
        for line in args.whitelist.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    actual = {raw for raw, _ in actual_pairs}
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    banned = sorted(
        demangled
        for _, demangled in actual_pairs
        if any(term.lower() in demangled.lower() for term in BANNED)
    )
    if extra or missing or banned:
        if extra:
            print("unexpected dynamic symbols:\n  " + "\n  ".join(extra))
        if missing:
            print("missing dynamic symbols:\n  " + "\n  ".join(missing))
        if banned:
            print("forbidden dynamic symbols:\n  " + "\n  ".join(banned))
        return 1
    print(f"public ABI v2: {len(actual)} symbols, exact whitelist match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
