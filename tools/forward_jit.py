#!/usr/bin/env python3
"""Content-addressed runtime compiler for Forward benchmark specializations.

The production library still has a conventional release build.  Autotuning,
however, needs a different compile-time-specialized Forward kernel for every
candidate.  This module builds those executables directly from Python, keeps a
small shared host-object cache, and atomically publishes complete artifacts.

The cache discipline follows DeepGEMM's JIT implementation: a key includes the
generated code, compiler identity, flags, repository inputs, and external
headers; compilation happens in a same-filesystem temporary directory; and a
directory rename makes a finished artifact visible to concurrent processes.
"""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


JIT_SCHEMA = 1
ARTIFACT_KIND = "megamoe-forward-test-binary"

HOST_SOURCES = (
    "src/cuda/backward_stub.cpp",
    "src/cuda/control_protocol.cpp",
    "src/metadata/notify_dispatch.cpp",
    "src/metadata/serialize.cpp",
    "src/transport/bootstrap_channel.cpp",
    "src/transport/activation_protocol.cpp",
    "src/transport/host_fake_count_exchange.cpp",
    "src/scoreboard/workspace_plan.cpp",
)

CUDA_SOURCES = (
    "src/cuda/device_notify.cu",
    "src/cuda/activation_transport.cu",
    "src/cuda/persistent_v3/kernel_entry.cu",
    "src/transport/nccl_count_exchange.cu",
)

TEST_SOURCE = "tests/test_cuda_notify.cpp"
SOURCE_SUFFIXES = frozenset({".h", ".hpp", ".cuh", ".inc", ".cpp", ".cu"})


class JitError(RuntimeError):
    """The runtime compiler configuration or cache is invalid."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def content_cache_key(signature: Mapping[str, Any],
                      generated_header: str) -> str:
    """Return the stable artifact key used by the on-disk JIT cache."""

    return hashlib.sha256(_canonical_json({
        "kind": ARTIFACT_KIND,
        "signature": signature,
        "generated_header": generated_header,
    })).hexdigest()


def _hash_parts(parts: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise JitError(f"include directory does not exist: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )


def digest_files(paths: Iterable[Path], *, relative_to: Path | None = None) -> str:
    """Hash paths and bytes in a deterministic order."""

    normalized = sorted({path.resolve() for path in paths})
    parts: list[bytes] = []
    for path in normalized:
        if not path.is_file():
            raise JitError(f"JIT input does not exist: {path}")
        if relative_to is not None:
            try:
                label = path.relative_to(relative_to.resolve()).as_posix()
            except ValueError:
                label = str(path)
        else:
            label = str(path)
        parts.extend((label.encode(), path.read_bytes()))
    return _hash_parts(parts)


def _run_version(tool: Path, arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        [str(tool), *arguments], stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    if completed.returncode != 0:
        raise JitError(f"cannot query compiler {tool}: {completed.stdout.strip()}")
    return completed.stdout.strip()


def _resolve_tool(configured: str | None, fallback: str,
                  preferred: Path | None = None) -> Path:
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        resolved = shutil.which(configured)
        if resolved:
            return Path(resolved).resolve()
        raise JitError(f"configured tool is unavailable: {configured}")
    if preferred is not None and preferred.is_file():
        return preferred.resolve()
    resolved = shutil.which(fallback)
    if not resolved:
        raise JitError(f"required tool is unavailable: {fallback}")
    return Path(resolved).resolve()


def _deep_gemm_include() -> Path:
    spec = importlib.util.find_spec("deep_gemm")
    if spec is not None and spec.origin is not None:
        installed = Path(spec.origin).resolve().parent / "include"
        if installed.is_dir():
            return installed
    raise JitError(
        "cannot locate the installed deep_gemm include tree containing CUTLASS/CUTE"
    )


def _architecture_flags(architecture: str) -> list[str]:
    match = re.fullmatch(r"sm_([0-9]{2,3}[af]?)", architecture)
    if match is None:
        raise JitError(
            f"invalid JIT architecture {architecture!r}; expected e.g. sm_90a"
        )
    suffix = match.group(1)
    return ["-gencode", f"arch=compute_{suffix},code=sm_{suffix}"]


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _nccl_identity(library_dir: Path | None) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    header_candidates = (
        Path("/usr/include/nccl.h"),
        Path("/usr/local/cuda/include/nccl.h"),
    )
    header = next((path for path in header_candidates if path.is_file()), None)
    if header is not None:
        identity["header"] = {
            **_file_identity(header),
            "sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        }
    if library_dir is not None:
        library_candidates = (
            library_dir / "libnccl.so",
            library_dir / "libnccl.so.2",
        )
        library = next(
            (path for path in library_candidates if path.exists()), None
        )
        if library is not None:
            identity["runtime_library"] = _file_identity(library)
    return identity


def validate_jit_config(config: Mapping[str, Any]) -> None:
    allowed = {
        "cache_root", "jobs", "architecture", "timeout_seconds",
        "nvcc", "cxx", "archiver",
    }
    unknown = set(config) - allowed
    if unknown:
        raise JitError(f"unknown JIT settings: {sorted(unknown)}")
    required = {"cache_root", "jobs", "architecture", "timeout_seconds"}
    missing = required - set(config)
    if missing:
        raise JitError(f"missing JIT settings: {sorted(missing)}")
    if not isinstance(config["cache_root"], str) or not config["cache_root"]:
        raise JitError("jit.cache_root must be a non-empty path")
    if type(config["jobs"]) is not int or not 1 <= config["jobs"] <= 32:
        raise JitError("jit.jobs must be in [1, 32]")
    if (type(config["timeout_seconds"]) is not int or
            config["timeout_seconds"] <= 0):
        raise JitError("jit.timeout_seconds must be a positive integer")
    _architecture_flags(str(config["architecture"]))
    for key in ("nvcc", "cxx", "archiver"):
        if key in config and (not isinstance(config[key], str) or not config[key]):
            raise JitError(f"jit.{key} must be a non-empty string")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
            (item for item in root.rglob("*") if item.is_dir()), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory(staging: Path, destination: Path) -> bool:
    """Atomically publish staging; return False when another process won."""

    _fsync_tree(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(destination)
        return True
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        shutil.rmtree(staging)
        return False


def _cache_manifest(directory: Path, expected_key: str,
                    expected_kind: str) -> dict[str, Any] | None:
    if not directory.exists():
        return None
    manifest_path = directory / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JitError(f"corrupt JIT cache {directory}: {error}") from error
    if (manifest.get("schema") != JIT_SCHEMA or
            manifest.get("kind") != expected_kind or
            manifest.get("cache_key") != expected_key):
        raise JitError(f"JIT cache manifest mismatch: {directory}")
    return manifest


def _object_name(source: str, variant: str = "") -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source)
    return f"{stem}{('.' + variant) if variant else ''}.o"


def _run_command(command: Sequence[str], cwd: Path,
                 environment: Mapping[str, str], timeout: int) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, env=dict(environment),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=timeout, check=False,
        )
        return completed.returncode, completed.stdout
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output + f"\nTIMEOUT after {timeout}s\n"


def _run_parallel(commands: Sequence[Sequence[str]], *, cwd: Path,
                  environment: Mapping[str, str], jobs: int, timeout: int,
                  log_path: Path) -> None:
    results: list[tuple[int, str] | None] = [None] * len(commands)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                _run_command, command, cwd, environment, timeout
            ): index
            for index, command in enumerate(commands)
        }
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()

    with log_path.open("a", encoding="utf-8") as log:
        for command, result in zip(commands, results):
            assert result is not None
            returncode, output = result
            log.write("COMMAND " + json.dumps(list(command)) + "\n")
            log.write(output)
            if output and not output.endswith("\n"):
                log.write("\n")
            log.write(f"RETURN_CODE {returncode}\n")
    failures = [
        (command, result[0]) for command, result in zip(commands, results)
        if result is not None and result[0] != 0
    ]
    if failures:
        raise JitError(
            "JIT compilation failed: " + ", ".join(
                f"{Path(command[0]).name} rc={returncode}"
                for command, returncode in failures
            )
        )


class ForwardJitCompiler:
    """Build and cache one compile-time-specialized Forward test executable."""

    def __init__(self, repo_root: Path, config: Mapping[str, Any],
                 environment: Mapping[str, str], *,
                 nccl_lib_dir: str | None = None):
        validate_jit_config(config)
        self.repo_root = repo_root.resolve()
        cache_root = Path(str(config["cache_root"]))
        self.cache_root = (
            cache_root if cache_root.is_absolute() else self.repo_root / cache_root
        ).resolve()
        self.jobs = int(config["jobs"])
        self.timeout = int(config["timeout_seconds"])
        self.architecture = str(config["architecture"])
        self.arch_flags = _architecture_flags(self.architecture)
        self.environment = dict(environment)
        self._host_cache_lock = threading.Lock()
        self.nvcc = _resolve_tool(
            config.get("nvcc"), "nvcc", Path("/usr/local/cuda/bin/nvcc")
        )
        self.cxx = _resolve_tool(config.get("cxx"), "g++")
        self.archiver = _resolve_tool(config.get("archiver"), "ar")
        self.deep_gemm_include = _deep_gemm_include().resolve()
        self.vendor_include = (self.repo_root / "src/cuda/vendor").resolve()
        configured_nccl = Path(nccl_lib_dir).resolve() if nccl_lib_dir else None
        self.nccl_lib_dir = (
            configured_nccl if configured_nccl is not None and
            configured_nccl.is_dir() else None
        )
        self.nccl_identity = _nccl_identity(self.nccl_lib_dir)

        self.host_flags = [
            "-Iinclude", "-Isrc", "-Itests", "-pthread",
            "-DMEGAMOE_DEBUG_VALIDATE=0", "-DMEGAMOE_BUILD_SHARED=1",
            "-DMEGAMOE_DEVICE_TIMELINE=0",
            "-DMEGAMOE_FORWARD_LIGHT_PROGRESS=0",
            "-DMEGAMOE_BUILD_TESTING=1", "-O2", "-g", "-std=c++17",
            "-Wall", "-Wextra", "-Wpedantic", "-Wconversion", "-Wshadow",
            "-fPIC", "-fvisibility=hidden", "-ffunction-sections",
            "-fdata-sections",
        ]
        self.cuda_flags = [
            "-Iinclude", "-Isrc", "-Itests", "-Xcompiler=-pthread",
            "-isystem", str(self.vendor_include),
            "-isystem", str(self.deep_gemm_include),
            "-DMEGAMOE_DEBUG_VALIDATE=0", "-DMEGAMOE_BUILD_SHARED=1",
            "-DMEGAMOE_DEVICE_TIMELINE=0",
            "-DMEGAMOE_FORWARD_LIGHT_PROGRESS=0",
            "-DMEGAMOE_BUILD_TESTING=1", "-O2", "-g", "-std=c++17",
            "--expt-relaxed-constexpr", "-Wno-deprecated-gpu-targets",
            "-Xcompiler=-fPIC,-fvisibility=hidden,-ffunction-sections,-fdata-sections",
            *self.arch_flags,
        ]
        self.cuda_library_flags = [
            *self.cuda_flags, "-Xcompiler=-Wall,-Wextra,-Werror",
        ]
        self.link_flags = [
            *self.arch_flags, "-Wno-deprecated-gpu-targets",
            "-Xcompiler=-pthread",
        ]
        self.toolchain = {
            "nvcc": str(self.nvcc),
            "nvcc_version": _run_version(self.nvcc, ("--version",)),
            "cxx": str(self.cxx),
            "cxx_version": _run_version(self.cxx, ("--version",)).splitlines()[0],
            "archiver": str(self.archiver),
            "architecture": self.architecture,
        }
        repository_inputs = []
        for root_name in ("include", "src"):
            repository_inputs.extend(_regular_files(self.repo_root / root_name))
        repository_inputs.extend(
            path for path in _regular_files(self.repo_root / "tests")
            if path.suffix in SOURCE_SUFFIXES
        )
        self.repository_digest = digest_files(
            repository_inputs, relative_to=self.repo_root
        )
        self.external_include_digest = digest_files(
            _regular_files(self.deep_gemm_include),
            relative_to=self.deep_gemm_include,
        )
        self.signature = {
            "schema": JIT_SCHEMA,
            "toolchain": self.toolchain,
            "host_flags": self.host_flags,
            "cuda_flags": self.cuda_flags,
            "cuda_library_flags": self.cuda_library_flags,
            "link_flags": self.link_flags,
            "repository_digest": self.repository_digest,
            "external_include_digest": self.external_include_digest,
            "nccl_lib_dir": str(self.nccl_lib_dir) if self.nccl_lib_dir else None,
            "nccl_identity": self.nccl_identity,
        }

    def cache_key(self, generated_header: str) -> str:
        return content_cache_key(self.signature, generated_header)

    def artifact_directory(self, generated_header: str) -> Path:
        key = self.cache_key(generated_header)
        return self.cache_root / "cache" / f"forward.{key}"

    def _host_cache_key(self) -> str:
        return hashlib.sha256(_canonical_json({
            "kind": "megamoe-host-testing-archive",
            "toolchain": self.toolchain["cxx_version"],
            "compiler": str(self.cxx),
            "archiver": str(self.archiver),
            "flags": self.host_flags,
            "repository_digest": self.repository_digest,
        })).hexdigest()

    def _build_host_archive(self, log_path: Path) -> tuple[Path, bool]:
        with self._host_cache_lock:
            return self._build_host_archive_locked(log_path)

    def _build_host_archive_locked(self, log_path: Path) -> tuple[Path, bool]:
        key = self._host_cache_key()
        destination = self.cache_root / "cache" / f"host.{key}"
        manifest = _cache_manifest(
            destination, key, "megamoe-host-testing-archive"
        )
        if manifest is not None:
            archive = destination / "libmegamoe_host_testing.a"
            if not archive.is_file():
                raise JitError(f"corrupt JIT host cache: {destination}")
            return archive, True

        temporary_root = self.cache_root / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        staging = temporary_root / f"host.{uuid.uuid4().hex}"
        objects = staging / "objects"
        objects.mkdir(parents=True)
        commands = []
        object_paths = []
        for source in HOST_SOURCES:
            output = objects / _object_name(source)
            object_paths.append(output)
            commands.append([
                str(self.cxx), *self.host_flags, "-c", source,
                "-o", str(output),
            ])
        try:
            _run_parallel(
                commands, cwd=self.repo_root, environment=self.environment,
                jobs=self.jobs, timeout=self.timeout, log_path=log_path,
            )
            archive = staging / "libmegamoe_host_testing.a"
            command = [str(self.archiver), "rcs", str(archive), *map(str, object_paths)]
            returncode, output = _run_command(
                command, self.repo_root, self.environment, self.timeout
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write("COMMAND " + json.dumps(command) + "\n" + output)
                log.write(f"RETURN_CODE {returncode}\n")
            if returncode != 0:
                raise JitError(f"archiver failed with rc={returncode}")
            shutil.rmtree(objects)
            _write_json(staging / "manifest.json", {
                "schema": JIT_SCHEMA,
                "kind": "megamoe-host-testing-archive",
                "cache_key": key,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "toolchain": self.toolchain,
                "repository_digest": self.repository_digest,
            })
            published = _publish_directory(staging, destination)
            if not published:
                _cache_manifest(
                    destination, key, "megamoe-host-testing-archive"
                )
            return destination / "libmegamoe_host_testing.a", False
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    def build(self, *, candidate_name: str, config_digest: str,
              generated_header: str, log_path: Path,
              command_jobs: int | None = None) -> dict[str, Any]:
        start = time.monotonic()
        parallel_jobs = self.jobs if command_jobs is None else command_jobs
        if not 1 <= parallel_jobs <= self.jobs:
            raise JitError(
                f"command_jobs must be in [1, {self.jobs}], got {parallel_jobs}"
            )
        key = self.cache_key(generated_header)
        destination = self.artifact_directory(generated_header)
        manifest = _cache_manifest(destination, key, ARTIFACT_KIND)
        if manifest is not None:
            binary = destination / "test_cuda_notify"
            header = destination / "generated/megamoe_forward_tuning_generated.hpp"
            if not binary.is_file() or not os.access(binary, os.X_OK):
                raise JitError(f"corrupt JIT cache, executable missing: {destination}")
            if (not header.is_file() or
                    header.read_text(encoding="utf-8") != generated_header):
                raise JitError(f"corrupt JIT cache, generated header mismatch: {destination}")
            log_path.write_text(
                f"CACHE_HIT {destination}\n", encoding="utf-8"
            )
            return {
                "backend": "python-content-addressed-jit",
                "cache_hit": True,
                "binary_preexisted": True,
                "cache_key": key,
                "cache_dir": str(destination),
                "binary": str(binary),
                "generated_header": str(header),
                "elapsed_seconds": time.monotonic() - start,
                "returncode": 0,
                "log": str(log_path),
                "toolchain": self.toolchain,
            }

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        try:
            host_archive, host_cache_hit = self._build_host_archive(log_path)
        except BaseException as error:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"JIT_ERROR {type(error).__name__}: {error}\n")
            return {
                "backend": "python-content-addressed-jit",
                "cache_hit": False,
                "binary_preexisted": False,
                "cache_key": key,
                "cache_dir": str(destination),
                "binary": str(destination / "test_cuda_notify"),
                "elapsed_seconds": time.monotonic() - start,
                "returncode": 1,
                "error": str(error),
                "log": str(log_path),
                "toolchain": self.toolchain,
            }
        temporary_root = self.cache_root / "tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        staging = temporary_root / f"forward.{uuid.uuid4().hex}"
        generated_dir = staging / "generated"
        objects = staging / "objects"
        generated_dir.mkdir(parents=True)
        objects.mkdir(parents=True)
        header = generated_dir / "megamoe_forward_tuning_generated.hpp"
        header.write_text(generated_header, encoding="utf-8")
        specialization_flags = [
            f"-I{generated_dir}", "-DMEGAMOE_FORWARD_USE_GENERATED_TUNING=1"
        ]

        object_paths: list[Path] = []
        commands: list[list[str]] = []
        for source in CUDA_SOURCES:
            output = objects / _object_name(source)
            object_paths.append(output)
            commands.append([
                str(self.nvcc), *self.cuda_library_flags,
                *specialization_flags,
                "-c", source, "-o", str(output),
            ])
        transport_object = objects / _object_name(
            "src/cuda/activation_transport.cu", "transport"
        )
        object_paths.append(transport_object)
        commands.append([
            str(self.nvcc), *self.cuda_library_flags,
            *specialization_flags,
            "-DMEGAMOE_FORWARD_TRANSPORT_TU=1", "-diag-suppress=177",
            "-c", "src/cuda/activation_transport.cu",
            "-o", str(transport_object),
        ])
        test_object = objects / _object_name(TEST_SOURCE)
        commands.append([
            str(self.nvcc), *self.cuda_flags, *specialization_flags,
            "-c", TEST_SOURCE, "-o", str(test_object),
        ])
        try:
            _run_parallel(
                commands, cwd=self.repo_root, environment=self.environment,
                jobs=parallel_jobs, timeout=self.timeout, log_path=log_path,
            )
            cuda_archive = staging / "libmegamoe_cuda_testing.a"
            archive_command = [
                str(self.archiver), "rcs", str(cuda_archive),
                *map(str, object_paths),
            ]
            returncode, output = _run_command(
                archive_command, self.repo_root, self.environment, self.timeout
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write("COMMAND " + json.dumps(archive_command) + "\n")
                log.write(output)
                log.write(f"RETURN_CODE {returncode}\n")
            if returncode != 0:
                raise JitError(f"archiver failed with rc={returncode}")

            binary = staging / "test_cuda_notify"
            link_command = [
                str(self.nvcc), *self.link_flags, str(test_object),
                "-Xlinker=--whole-archive", str(host_archive), str(cuda_archive),
                "-Xlinker=--no-whole-archive",
            ]
            if self.nccl_lib_dir is not None:
                link_command.extend(("-L", str(self.nccl_lib_dir)))
            link_command.extend(
                ("-lnccl", "-lcublas", "-lcuda", "-ldl", "-o", str(binary))
            )
            returncode, output = _run_command(
                link_command, self.repo_root, self.environment, self.timeout
            )
            with log_path.open("a", encoding="utf-8") as log:
                log.write("COMMAND " + json.dumps(link_command) + "\n")
                log.write(output)
                log.write(f"RETURN_CODE {returncode}\n")
            if returncode != 0:
                raise JitError(f"linker failed with rc={returncode}")
            binary.chmod(0o755)
            shutil.rmtree(objects)
            cuda_archive.unlink()
            manifest = {
                "schema": JIT_SCHEMA,
                "kind": ARTIFACT_KIND,
                "cache_key": key,
                "candidate": candidate_name,
                "config_digest": config_digest,
                "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "signature": self.signature,
                "host_cache_key": self._host_cache_key(),
            }
            _write_json(staging / "manifest.json", manifest)
            published = _publish_directory(staging, destination)
            if not published:
                _cache_manifest(destination, key, ARTIFACT_KIND)
            final_binary = destination / "test_cuda_notify"
            final_header = (
                destination / "generated/megamoe_forward_tuning_generated.hpp"
            )
            return {
                "backend": "python-content-addressed-jit",
                "cache_hit": False,
                "binary_preexisted": False,
                "host_cache_hit": host_cache_hit,
                "cache_key": key,
                "cache_dir": str(destination),
                "binary": str(final_binary),
                "generated_header": str(final_header),
                "elapsed_seconds": time.monotonic() - start,
                "returncode": 0,
                "log": str(log_path),
                "toolchain": self.toolchain,
            }
        except BaseException as error:
            if staging.exists():
                shutil.rmtree(staging)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"JIT_ERROR {type(error).__name__}: {error}\n")
            return {
                "backend": "python-content-addressed-jit",
                "cache_hit": False,
                "binary_preexisted": False,
                "cache_key": key,
                "cache_dir": str(destination),
                "binary": str(destination / "test_cuda_notify"),
                "elapsed_seconds": time.monotonic() - start,
                "returncode": 1,
                "error": str(error),
                "log": str(log_path),
                "toolchain": self.toolchain,
            }
