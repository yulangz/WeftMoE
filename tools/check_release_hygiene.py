#!/usr/bin/env python3
"""Raw installed-header and tracked release-document hygiene gates."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys


HEADER_PATTERNS = {
    "for_testing": re.compile(r"for_testing"),
    "testing-scope": re.compile(r"testing::"),
    "simulation-entry": re.compile(r"simulate_"),
    "injection-entry": re.compile(r"inject_"),
    "validation-entry": re.compile(r"for_validation"),
    "mutation": re.compile(r"\bmutation\b", re.IGNORECASE),
    # `default` is ordinary C++; only the standalone term is repository-only.
    "fault": re.compile(r"\bfault\b", re.IGNORECASE),
}
PRIVATE_IPV4 = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
DOC_PATTERNS = {
    "private IPv4": PRIVATE_IPV4,
    "machine storage path": re.compile(r"/storage/"),
    "known developer identity": re.compile(r"wanghaitao|gpulingjun", re.I),
}


def tracked_files(root: pathlib.Path) -> list[pathlib.Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "*.md", "*.json"], cwd=root, text=True
    )
    return [root / line for line in output.splitlines() if line]


def scan(paths: list[pathlib.Path], patterns: dict[str, re.Pattern[str]]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    failures.append(f"{path}:{line_number}: {label}: {line.strip()}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--headers", nargs="*", type=pathlib.Path, default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    headers = args.headers or sorted((root / "include" / "megamoe").rglob("*.hpp"))
    failures = scan(headers, HEADER_PATTERNS)
    failures += scan(tracked_files(root), DOC_PATTERNS)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"release hygiene: {len(headers)} headers and tracked docs are clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
