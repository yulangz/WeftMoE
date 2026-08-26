#!/usr/bin/env python3
"""Deterministic release-text and installed-header hygiene gates.

The same checks run in a Git worktree and in a ``git archive`` extraction.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
import tempfile


HEADER_PATTERNS = {
    "implementation namespace": re.compile(r"\binternal\b"),
    "for_testing": re.compile(r"for_testing"),
    "testing-scope": re.compile(r"\btesting\b|testing::"),
    "synthetic": re.compile(r"\bsynthetic\b", re.IGNORECASE),
    "fault": re.compile(r"\bfault\b", re.IGNORECASE),
    "validation-entry": re.compile(r"for_validation"),
    "mutation": re.compile(r"\bmutation\b", re.IGNORECASE),
}
PRIVATE_IPV4 = re.compile(
    r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
    r"192\.168\.\d{1,3}\.\d{1,3}|"
    r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})\b"
)
RELEASE_PATTERNS = {
    "private IPv4": PRIVATE_IPV4,
    # Assemble sensitive literals so the checker does not exempt itself.
    "machine storage path": re.compile(
        r"(?<![A-Za-z0-9_.-])/" + "storage" + "/"
    ),
    "known developer identity": re.compile(
        "wang" + "haitao|gpu" + "lingjun", re.IGNORECASE
    ),
    "merge conflict marker": re.compile(r"^(?:<<<<<<<|=======|>>>>>>>)"),
}
RELEASE_ROOTS = (
    "include",
    "src",
    "tests",
    "benchmarks",
    "examples",
    "docs",
    "tools",
    "abi",
)
TOP_LEVEL_RELEASE_FILES = {
    "Makefile",
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY_NOTICES.md",
}
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "install",
    "install-stage",
}
EXCLUDED_SUFFIXES = {
    ".a",
    ".bin",
    ".cubin",
    ".fatbin",
    ".o",
    ".pyc",
    ".so",
    ".tar",
    ".tgz",
    ".zip",
}


def has_git_metadata(root: pathlib.Path) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def git_release_files(root: pathlib.Path) -> list[pathlib.Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    relative = sorted(
        pathlib.Path(item.decode("utf-8"))
        for item in output.split(b"\0")
        if item
    )
    return [root / path for path in relative if (root / path).is_file()]


def excluded_path(relative: pathlib.Path) -> bool:
    if any(part in EXCLUDED_DIRECTORY_NAMES or part.startswith("build-")
           for part in relative.parts[:-1]):
        return True
    return relative.suffix.lower() in EXCLUDED_SUFFIXES


def archive_release_files(root: pathlib.Path) -> list[pathlib.Path]:
    paths: set[pathlib.Path] = set()
    for root_name in RELEASE_ROOTS:
        base = root / root_name
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not excluded_path(path.relative_to(root)):
                paths.add(path)
    for name in TOP_LEVEL_RELEASE_FILES:
        path = root / name
        if path.is_file():
            paths.add(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def release_files(root: pathlib.Path) -> tuple[list[pathlib.Path], str]:
    if has_git_metadata(root):
        return git_release_files(root), "git"
    return archive_release_files(root), "archive"


def text_content(path: pathlib.Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_release_text(paths: list[pathlib.Path]) -> tuple[list[str], int]:
    failures: list[str] = []
    scanned = 0
    for path in paths:
        text = text_content(path)
        if text is None:
            continue
        scanned += 1
        for line_number, line_with_ending in enumerate(
                text.splitlines(keepends=True), 1):
            line = line_with_ending.rstrip("\r\n")
            if line.endswith((" ", "\t")):
                failures.append(f"{path}:{line_number}: trailing whitespace")
            for label, pattern in RELEASE_PATTERNS.items():
                if pattern.search(line):
                    failures.append(
                        f"{path}:{line_number}: {label}: {line.strip()}")
        if text and not text.endswith("\n"):
            failures.append(f"{path}: missing final newline")
    return failures, scanned


def scan_headers(paths: list[pathlib.Path]) -> list[str]:
    failures: list[str] = []
    for path in paths:
        text = text_content(path)
        if text is None:
            failures.append(f"{path}: installed header is not UTF-8 text")
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            for label, pattern in HEADER_PATTERNS.items():
                if pattern.search(line):
                    failures.append(
                        f"{path}:{line_number}: {label}: {line.strip()}")
    return failures


def self_test() -> list[str]:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="megamoe-hygiene-") as directory:
        root = pathlib.Path(directory)
        trailing = root / "trailing.txt"
        trailing.write_text("bad trailing space \n", encoding="utf-8")
        private = root / "private.txt"
        private.write_text(
            "endpoint=" + "192" + ".168.23.41\n", encoding="utf-8"
        )
        trailing_failures, _ = scan_release_text([trailing])
        private_failures, _ = scan_release_text([private])
        if not any("trailing whitespace" in item for item in trailing_failures):
            failures.append("negative fixture: trailing whitespace was accepted")
        if not any("private IPv4" in item for item in private_failures):
            failures.append("negative fixture: private IPv4 was accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--headers", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    files, mode = release_files(root)
    headers = args.headers or sorted((root / "include" / "megamoe").rglob("*.hpp"))
    failures, text_count = scan_release_text(files)
    failures += scan_headers(headers)
    if args.self_test:
        failures += self_test()
    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"release hygiene: mode={mode}, text_files={text_count}, "
        f"installed_headers={len(headers)}, negative_fixtures="
        f"{'pass' if args.self_test else 'not-requested'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
