#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Create or verify normalized CUDA main-kernel artifact metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


KERNELS = {
    "forward_compute_main": {
        "name": "local_p2p_forward_tma_store_kernel",
        "template": "ILb0ELb1ELb0ELb1ELj4ELj4EE",
        "configured_dynamic_shared_bytes": 196832,
    },
}
ARCHES = ("sm_90a",)


def run_cuobjdump(binary: Path, option: str) -> str:
    command = ["cuobjdump", option, str(binary)]
    completed = subprocess.run(command, check=True, text=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    return completed.stdout


def split_arches(output: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current = ""
    for line in output.splitlines():
        match = re.match(r"arch = (sm_[0-9a-z]+)$", line.strip())
        if match:
            current = match.group(1)
            sections.setdefault(current, [])
        if current:
            sections[current].append(line)
    return {arch: "\n".join(lines) for arch, lines in sections.items()}


def match_symbol(section: str, spec: dict[str, object]) -> str:
    candidates = []
    for line in section.splitlines():
        match = re.search(r"(?:Function\s*:|Function)\s+([^:]+):?$", line)
        if not match:
            continue
        symbol = match.group(1).strip()
        if str(spec["name"]) in symbol and str(spec["template"]) in symbol:
            candidates.append(symbol)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            f"expected one {spec['name']} {spec['template']} symbol, "
            f"found {len(unique)}")
    return unique[0]


def parse_resources(section: str, symbol: str) -> dict[str, int]:
    pattern = re.compile(r"Function\s+" + re.escape(symbol) +
                         r":\n\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) "
                         r"LOCAL:(\d+)")
    match = pattern.search(section)
    if not match:
        raise RuntimeError(f"resource record missing for {symbol}")
    return {
        "registers": int(match.group(1)),
        "stack_bytes": int(match.group(2)),
        "static_shared_bytes": int(match.group(3)),
        "local_bytes": int(match.group(4)),
    }


def normalized_sass(section: str, symbol: str) -> str:
    lines = section.splitlines()
    start = next((index for index, line in enumerate(lines)
                  if line.strip() == f"Function : {symbol}"), None)
    if start is None:
        raise RuntimeError(f"SASS record missing for {symbol}")
    instructions: list[str] = []
    instruction = re.compile(r"/\*[0-9a-fA-F]+\*/\s+(.*?)\s*;")
    for line in lines[start + 1:]:
        if line.strip().startswith("Function :"):
            break
        match = instruction.search(line)
        if match:
            # Instruction addresses and encoded control words are deliberately
            # excluded. Whitespace is canonicalized; operands and branch
            # targets remain because they are semantic.
            instructions.append(" ".join(match.group(1).split()) + ";")
    if not instructions:
        raise RuntimeError(f"no SASS instructions extracted for {symbol}")
    return "\n".join(instructions) + "\n"


def collect(binary: Path) -> dict[str, object]:
    resources = split_arches(run_cuobjdump(binary, "--dump-resource-usage"))
    sass = split_arches(run_cuobjdump(binary, "--dump-sass"))
    result: dict[str, object] = {
        "schema": 1,
        "runtime_arch": "sm_90a",
        "compile_only_arches": [],
        "artifact": "src/cuda/activation_transport.o",
        "dynamic_shared_shape": "main-shape world=16",
        "kernels": {},
    }
    kernels = result["kernels"]
    assert isinstance(kernels, dict)
    for logical_name, spec in KERNELS.items():
        records: dict[str, object] = {}
        for arch in ARCHES:
            if arch not in resources or arch not in sass:
                raise RuntimeError(f"artifact does not contain {arch}")
            symbol = match_symbol(resources[arch], spec)
            sass_symbol = match_symbol(sass[arch], spec)
            if symbol != sass_symbol:
                raise RuntimeError(f"resource/SASS symbol mismatch for {arch}")
            normalized = normalized_sass(sass[arch], symbol)
            record = parse_resources(resources[arch], symbol)
            record.update({
                "configured_dynamic_shared_bytes":
                    spec["configured_dynamic_shared_bytes"],
                "instruction_count": normalized.count(";\n"),
                "normalized_sass_sha256":
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            })
            records[arch] = record
        kernels[logical_name] = records
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    actual = collect(args.binary)
    rendered = json.dumps(actual, indent=2, sort_keys=True) + "\n"
    if args.update:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered, encoding="utf-8")
        print(f"updated {args.manifest}")
        return 0
    expected = json.loads(args.manifest.read_text(encoding="utf-8"))
    if actual != expected:
        print("kernel artifact manifest mismatch")
        print(rendered, end="")
        return 1
    print("kernel artifact manifest: PASS")
    for kernel, arches in actual["kernels"].items():
        for arch, record in arches.items():
            print(f"  {kernel} {arch}: reg={record['registers']} "
                  f"stack={record['stack_bytes']} "
                  f"static_smem={record['static_shared_bytes']} "
                  f"dynamic_smem={record['configured_dynamic_shared_bytes']} "
                  f"sha256={record['normalized_sass_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
