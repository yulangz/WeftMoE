#!/usr/bin/env python3
"""Attribute static local-memory instructions to PF device functions and sources.

LDL/STL counts include stack/call traffic; they are not dynamic spill bytes.
Pair this report with ptxas -v --warn-on-spills and inspect hot loops separately.
"""

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


def run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          **kwargs).stdout


def function_name(symbol):
    tail = symbol.rsplit("$", 1)[-1]
    match = re.search(r"weftmoe_pf(?:7network|4ring|5local|14backward_local)(\d+)(\w+)", tail)
    if not match:
        return tail
    name = match[2][:int(match[1])]
    if name == "transfer":
        name += "_get" if "ILb1EE" in tail else "_put"
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--kernel", choices=("local", "network", "backward_local"), default="network")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    library = args.library.resolve()
    cuda = Path("/usr/local/cuda/bin")
    run([str(cuda / "cuobjdump"), "--extract-elf", "activation_transport",
         str(library)], cwd=args.output)
    cubin = args.output / "activation_transport.sm_90a.cubin"
    symbols = run(["readelf", "-sW", str(cubin)])
    pattern = re.compile(
        r"^\s*(\d+):\s+([0-9a-f]+)\s+(\S+)\s+FUNC\s+\S+\s+\S+\s+"
        r"(?:\[.*?\]\s+)?(\d+)\s+(\S+)", re.M)
    functions = [dict(index=int(m[1]), begin=int(m[2], 16),
                      size=int(m[3], 0), section=int(m[4]), symbol=m[5])
                 for m in pattern.finditer(symbols)]
    key = f"weftmoe_pf{len(args.kernel)}{args.kernel}6kernel"
    kernel, = (f for f in functions if key in f["symbol"] and "$" not in f["symbol"])
    functions = sorted((f for f in functions if f["section"] == kernel["section"]),
                       key=lambda f: f["begin"])
    starts = [f["begin"] for f in functions]
    sass = run([str(cuda / "nvdisasm"), "--print-code", "--print-line-info-inline",
                "--cuda-function-index", str(kernel["index"]), str(cubin)])
    (args.output / "source.sass").write_text(sass)
    counts, sources, sites = {}, {}, {}
    source = "(no line information)"
    source_line, pending_location = 0, None
    for line in sass.splitlines():
        location = re.search(r'//## File "([^"]+)", line (\d+)', line)
        if location and pending_location is None:
            pending_location = (location[1], int(location[2]))
        instruction = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)\s*;", line)
        if not instruction:
            continue
        if pending_location is not None:
            source, source_line = pending_location
            pending_location = None
        address = int(instruction[1], 16)
        function = functions[bisect_right(starts, address) - 1]
        name = function_name(function["symbol"])
        record = counts.setdefault(name, Counter())
        record["instructions"] += 1
        op = re.search(r"\b(LDL|STL|HGMMA|HMMA)\b", instruction[2])
        if op:
            record[op[1]] += 1
            if op[1] in ("LDL", "STL"):
                sources.setdefault(name, Counter())[source] += 1
                sites.setdefault(name, Counter())[f"{source}:{source_line}"] += 1
    result = dict(
        library=str(library),
        sha256=hashlib.file_digest(library.open("rb"), "sha256").hexdigest(),
        kernel=kernel["symbol"], functions=counts, local_memory_sources=sources,
        local_memory_sites=sites,
        scope="Static instructions, including callees; not runtime traffic or proof of spills")
    (args.output / "local_memory.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
