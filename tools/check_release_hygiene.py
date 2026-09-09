#!/usr/bin/env python3
"""Deterministic release-text and installed-header hygiene gates.

The same checks run in a Git worktree and in a ``git archive`` extraction.
"""

from __future__ import annotations

import argparse
import os
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
EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    ".cache",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".release-build",
    ".release-install",
    ".ruff_cache",
    ".vscode",
    "__pycache__",
    "build",
    "build-release",
    "build-release-werror",
}
EXCLUDED_SUFFIXES = {
    ".a",
    ".bin",
    ".cubin",
    ".fatbin",
    ".o",
    ".pyc",
    ".swo",
    ".swp",
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


def git_release_files(
        root: pathlib.Path) -> tuple[list[pathlib.Path], list[str]]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    relative = sorted(
        pathlib.Path(item.decode("utf-8"))
        for item in output.split(b"\0")
        if item
    )
    files: list[pathlib.Path] = []
    excluded: list[str] = []
    for path in relative:
        absolute = root / path
        if absolute.is_symlink():
            excluded.append(f"{path.as_posix()} (symlink not followed)")
        elif absolute.is_file():
            files.append(absolute)
    return files, excluded


def checked_explicit_excludes(
        root: pathlib.Path, values: list[str]) -> tuple[set[pathlib.Path], list[str]]:
    excludes: set[pathlib.Path] = set()
    failures: list[str] = []
    for value in values:
        relative = pathlib.Path(value)
        if (not value or relative.is_absolute() or value in (".", "..") or
                ".." in relative.parts or re.search(r"[*?\[\]{}]", value)):
            failures.append(
                f"explicit exclude must be an exact relative path: {value!r}")
            continue
        candidate = (root / relative).resolve(strict=False)
        if not candidate.is_relative_to(root):
            failures.append(
                f"explicit exclude escapes archive root: {value!r}")
            continue
        excludes.add(relative)
    return excludes, failures


def excluded_file(name: str) -> str | None:
    if name.endswith("~"):
        return "editor backup"
    if pathlib.Path(name).suffix.lower() in EXCLUDED_SUFFIXES:
        return "generated/binary artifact"
    if re.search(r"\.so(?:\.[0-9]+)*$", name):
        return "compiled shared library"
    return None


def archive_release_files(
        root: pathlib.Path,
        explicit_excludes: set[pathlib.Path] | None = None,
        ) -> tuple[list[pathlib.Path], list[str]]:
    """Walk every archive entry without following directory symlinks."""
    files: list[pathlib.Path] = []
    excluded: list[str] = []
    exact_excludes = explicit_excludes or set()

    def walk(directory: pathlib.Path) -> None:
        with os.scandir(directory) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
        for entry in entries:
            path = pathlib.Path(entry.path)
            relative = path.relative_to(root).as_posix()
            if entry.is_symlink():
                excluded.append(f"{relative} (symlink not followed)")
                continue
            if entry.is_dir(follow_symlinks=False):
                relative_path = path.relative_to(root)
                if relative_path in exact_excludes:
                    excluded.append(
                        f"{relative}/ (source=explicit, exact path)")
                elif entry.name in EXCLUDED_DIRECTORY_NAMES:
                    excluded.append(
                        f"{relative}/ (source=builtin, exact basename)")
                else:
                    walk(path)
                continue
            if not entry.is_file(follow_symlinks=False):
                excluded.append(f"{relative} (non-regular entry)")
                continue
            reason = excluded_file(entry.name)
            if reason is not None:
                excluded.append(f"{relative} ({reason})")
            else:
                files.append(path)

    walk(root)
    return files, excluded


def release_files(
        root: pathlib.Path,
        explicit_excludes: set[pathlib.Path] | None = None,
        ) -> tuple[list[pathlib.Path], list[str], str]:
    if has_git_metadata(root):
        files, excluded = git_release_files(root)
        return files, excluded, "git"
    files, excluded = archive_release_files(root, explicit_excludes)
    return files, excluded, "archive"


def text_content(path: pathlib.Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def scan_release_text(
        paths: list[pathlib.Path]) -> tuple[list[str], int, list[pathlib.Path]]:
    failures: list[str] = []
    scanned = 0
    binary: list[pathlib.Path] = []
    for path in paths:
        text = text_content(path)
        if text is None:
            binary.append(path)
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
    return failures, scanned, binary


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


def self_test() -> tuple[list[str], list[str], list[str]]:
    failures: list[str] = []
    fixtures = (
        (
            "config-private-ip",
            pathlib.Path("configs/example.json"),
            "endpoint=" + "192" + ".168.23.41\n",
            "private IPv4",
        ),
        (
            "tool-absolute-path",
            pathlib.Path("tools/example.py"),
            "path=/" + "storage" + "/machine/input\n",
            "machine storage path",
        ),
        (
            "dotfile-trailing-whitespace",
            pathlib.Path(".gitignore"),
            "generated/ \n",
            "trailing whitespace",
        ),
        (
            "unknown-directory-identity",
            pathlib.Path("future-module/new-file.txt"),
            "owner=" + "wang" + "haitao\n",
            "known developer identity",
        ),
        (
            "buildspec-source-private-ip",
            pathlib.Path("buildspec-source/new-source.txt"),
            "endpoint=" + "10" + ".23.45.67\n",
            "private IPv4",
        ),
        (
            "install-tools-absolute-path",
            pathlib.Path("install-tools/new-source.txt"),
            "path=/" + "storage" + "/release/input\n",
            "machine storage path",
        ),
        (
            "build-release-notes-trailing-whitespace",
            pathlib.Path("build-release-notes/file.txt"),
            "release note \n",
            "trailing whitespace",
        ),
        (
            "cache-algorithm-identity",
            pathlib.Path("cache-algorithm/source.txt"),
            "owner=" + "gpu" + "lingjun\n",
            "known developer identity",
        ),
    )
    passed: list[str] = []
    for label, relative, content, expected in fixtures:
        with tempfile.TemporaryDirectory(
                prefix="weftmoe-hygiene-") as directory:
            root = pathlib.Path(directory)
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--root", str(root)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if result.returncode == 0 or expected not in result.stdout:
                failures.append(
                    f"negative fixture {relative}: expected nonzero "
                    f"{expected}; output={result.stdout.strip()!r}")
            else:
                passed.append(label)
    exclude_fixtures = (
        ("builtin-build-release", pathlib.Path("build-release"), []),
        ("builtin-release-install", pathlib.Path(".release-install"), []),
        ("builtin-pycache", pathlib.Path("__pycache__"), []),
        (
            "explicit-release-output",
            pathlib.Path("release-output"),
            ["--exclude-dir", "release-output"],
        ),
    )
    excluded_passed: list[str] = []
    for label, relative, extra_arguments in exclude_fixtures:
        with tempfile.TemporaryDirectory(
                prefix="weftmoe-hygiene-") as directory:
            root = pathlib.Path(directory)
            path = root / relative / "must-not-scan.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "endpoint=" + "192" + ".168.44.55\n", encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(pathlib.Path(__file__).resolve()),
                 "--root", str(root), *extra_arguments],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            expected_source = "source=explicit" if extra_arguments else \
                "source=builtin"
            if result.returncode != 0 or expected_source not in result.stdout:
                failures.append(
                    f"exclude fixture {relative}: expected exact exclusion; "
                    f"output={result.stdout.strip()!r}")
            else:
                excluded_passed.append(label)
    return failures, passed, excluded_passed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    parser.add_argument("--headers", nargs="*", type=pathlib.Path, default=[])
    parser.add_argument(
        "--exclude-dir", action="append", default=[],
        help="exact archive-root-relative generated directory to exclude",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    explicit_excludes, failures = checked_explicit_excludes(
        root, args.exclude_dir)
    files, excluded, mode = release_files(root, explicit_excludes)
    headers = args.headers or sorted((root / "include" / "weftmoe").rglob("*.hpp"))
    text_failures, text_count, binary = scan_release_text(files)
    failures += text_failures
    excluded.extend(
        f"{path.relative_to(root).as_posix()} (binary/non-UTF8, not text-scanned)"
        for path in binary
    )
    failures += scan_headers(headers)
    fixture_labels: list[str] = []
    exclude_fixture_labels: list[str] = []
    if args.self_test:
        fixture_failures, fixture_labels, exclude_fixture_labels = self_test()
        failures += fixture_failures
    if failures:
        print("\n".join(failures))
        return 1
    excluded_text = "[]" if not excluded else "[" + ", ".join(excluded) + "]"
    print(
        f"release hygiene: mode={mode}, regular_files={len(files)}, "
        f"text_files={text_count}, excluded={excluded_text}, "
        f"installed_headers={len(headers)}, negative_fixtures="
        f"{','.join(fixture_labels) if args.self_test else 'not-requested'}, "
        f"exclude_fixtures="
        f"{','.join(exclude_fixture_labels) if args.self_test else 'not-requested'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
