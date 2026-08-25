#!/usr/bin/env python3
"""Launch a two-host NCCL protocol test with explicit remote/network options."""

import argparse
import base64
import json
import os
import re
import secrets
import select
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceTopology,
    build_reference_all,
)


MASK64 = (1 << 64) - 1
BOOTSTRAP_BYTES = 128
_active_bootstrap = b""


@dataclass(frozen=True)
class RankOutput:
    rank: int
    returncode: int
    stdout: str
    stderr: str


def scrub_output(value, bootstrap=None):
    """Remove bootstrap binary/text encodings from any aggregated diagnostic."""
    secret = _active_bootstrap if bootstrap is None else bytes(bootstrap)
    if isinstance(value, bytes):
        raw = value
        if secret:
            raw = raw.replace(secret, b"[REDACTED_SECRET]")
        text = raw.decode("utf-8", errors="replace")
    else:
        text = str(value)
    # Defense in depth for the retired stdout protocol, even before enough
    # bytes are known to derive its exact representation.
    text = re.sub(
        r"(?im)^BOOTSTRAP[ \t]+(?:[0-9a-f]{2})+[ \t]*$",
        "[REDACTED_SECRET]",
        text,
    )
    if secret:
        # Match the actual hexadecimal value case-insensitively per character.
        # Escaping the value keeps secret-derived text out of regex syntax.
        text = re.sub(
            re.escape(secret.hex()),
            "[REDACTED_SECRET]",
            text,
            flags=re.IGNORECASE,
        )
        representations = {
            base64.b64encode(secret).decode("ascii"),
            base64.urlsafe_b64encode(secret).decode("ascii"),
        }
        representations.update(item.rstrip("=") for item in tuple(representations))
        for representation in sorted(representations, key=len, reverse=True):
            if representation:
                text = text.replace(representation, "[REDACTED_SECRET]")
    return text


def seeded_inputs(world, experts, top_k, seed, max_tokens=10):
    def next_value():
        nonlocal seed
        seed = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
        return seed

    routes = []
    for _rank in range(world):
        rank_routes = []
        for _token in range(next_value() % (max_tokens + 1)):
            candidates = list(range(experts))
            selected = []
            for slot in range(top_k):
                index = slot + next_value() % (experts - slot)
                candidates[slot], candidates[index] = candidates[index], candidates[slot]
                selected.append(candidates[slot])
            rank_routes.append(selected)
        routes.append(rank_routes)
    return routes


def config_for(gpus_per_node):
    nodes = 2
    world = nodes * gpus_per_node
    experts = max(16, world * 2)
    top_k = min(6, experts)
    return ReferenceConfig(
        experts,
        top_k,
        7,
        ReferenceTopology(
            world,
            nodes,
            [node for node in range(nodes) for _ in range(gpus_per_node)],
            [local for _node in range(nodes) for local in range(gpus_per_node)],
            [expert % world for expert in range(experts)],
        ),
    )


def runner_args(binary, rank, gpus_per_node, seed, scenario, launch_id,
                bootstrap_fd, produces_bootstrap=False):
    result = [
        binary,
        "--rank-runner",
        "--rank", str(rank),
        "--nodes", "2",
        "--gpus-per-node", str(gpus_per_node),
        "--local-rank", str(rank % gpus_per_node),
        "--seed", str(seed),
        "--scenario", scenario,
        "--launch-id", launch_id,
    ]
    result.extend([
        "--bootstrap-output-fd" if produces_bootstrap
        else "--bootstrap-input-fd",
        str(bootstrap_fd),
    ])
    return result


def environment_overrides(args):
    overrides = {"NCCL_DEBUG": args.nccl_debug}
    if args.socket_ifname:
        overrides["NCCL_SOCKET_IFNAME"] = args.socket_ifname
    if args.disable_ib:
        overrides["NCCL_IB_DISABLE"] = "1"
    return overrides


def local_environment(args):
    environment = os.environ.copy()
    environment.update(environment_overrides(args))
    if args.nccl_lib_dir:
        prior = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = args.nccl_lib_dir + (
            (":" + prior) if prior else ""
        )
    return environment


def remote_command(args, runner):
    overrides = environment_overrides(args)
    if args.nccl_lib_dir:
        overrides["LD_LIBRARY_PATH"] = args.nccl_lib_dir
    command = ["env", *(f"{key}={value}" for key, value in overrides.items()), *runner]
    return [
        "ssh", "-o", "BatchMode=yes", "-p", str(args.port), args.remote,
        "exec " + shlex.join(command),
    ]


def read_bootstrap(fd, timeout):
    global _active_bootstrap
    deadline = time.monotonic() + timeout
    received = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("bootstrap secret pipe", timeout)
        ready, _, _ = select.select([fd], [], [], remaining)
        if not ready:
            raise subprocess.TimeoutExpired("bootstrap secret pipe", timeout)
        chunk = os.read(fd, BOOTSTRAP_BYTES + 1 - len(received))
        if not chunk:
            break
        received.extend(chunk)
        _active_bootstrap = bytes(received[:BOOTSTRAP_BYTES])
        if len(received) > BOOTSTRAP_BYTES:
            raise RuntimeError(
                "bootstrap secret channel malformed: expected=128, actual=>128"
            )
    if len(received) != BOOTSTRAP_BYTES:
        raise RuntimeError(
            "bootstrap secret channel short read: expected=128, "
            f"actual={len(received)}"
        )
    return bytes(received)


def send_bootstrap(process, bootstrap):
    stream = process.stdin
    primary_error = None
    try:
        if stream is None:
            raise RuntimeError(
                "bootstrap secret channel has no writable stdin"
            )
        try:
            frame = bytes(bootstrap)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "bootstrap secret channel requires exactly 128 bytes"
            ) from error
        if len(frame) != BOOTSTRAP_BYTES:
            raise RuntimeError(
                "bootstrap secret channel requires exactly 128 bytes; "
                f"actual={len(frame)}"
            )
        try:
            descriptor = stream.fileno()
        except (AttributeError, OSError, ValueError) as error:
            raise RuntimeError(
                "bootstrap secret channel has no valid descriptor"
            ) from error
        offset = 0
        while offset < BOOTSTRAP_BYTES:
            try:
                written = os.write(descriptor, frame[offset:])
            except InterruptedError:
                continue
            except BrokenPipeError as error:
                raise RuntimeError(
                    "bootstrap secret channel closed before 128 bytes"
                ) from error
            except OSError as error:
                raise RuntimeError(
                    "bootstrap secret channel write failed before 128 bytes"
                ) from error
            if written is None or not isinstance(written, int):
                raise RuntimeError(
                    "bootstrap secret channel write returned no byte count"
                )
            if written <= 0:
                raise RuntimeError(
                    "bootstrap secret channel write made zero progress"
                )
            if written > BOOTSTRAP_BYTES - offset:
                raise RuntimeError(
                    "bootstrap secret channel write exceeded remaining frame"
                )
            offset += written
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if stream is not None:
            try:
                stream.close()
            except (OSError, ValueError) as error:
                if primary_error is None:
                    raise RuntimeError(
                        "bootstrap secret channel close failed"
                    ) from error
            finally:
                # communicate() otherwise attempts to flush a closed stream.
                process.stdin = None


def remote_cleanup(args, launch_id, signal):
    # The unique argv marker lets cleanup target only this launcher even when
    # terminating the local ssh client did not propagate a signal remotely.
    pattern = f"--launch-id {launch_id}"
    command = (
        f"pkill -{signal} -f -- " + shlex.quote(pattern)
    )
    try:
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-p", str(args.port),
             args.remote, command],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        # Cleanup is best-effort here; the primary test error is preserved.
        pass


def stop_processes(processes, args, launch_id):
    for _rank, process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    drained = []
    for rank, process in processes:
        try:
            stdout, stderr = process.communicate(
                timeout=max(0.0, deadline - time.monotonic())
            )
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        drained.append(RankOutput(
            rank,
            process.returncode if process.returncode is not None else -1,
            scrub_output(stdout),
            scrub_output(stderr),
        ))
    remote_cleanup(args, launch_id, "TERM")
    time.sleep(0.25)
    remote_cleanup(args, launch_id, "KILL")
    return drained


def cleanup_diagnostic(outputs):
    details = []
    for result in outputs:
        tail = (result.stdout + "\n" + result.stderr)[-4000:]
        if tail.strip():
            details.append(
                f"rank={result.rank}, exit={result.returncode}, tail={tail}"
            )
    return scrub_output("; ".join(details))


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--remote", required=True,
                        help="explicit SSH destination, for example user@host")
    parser.add_argument("--remote-binary", required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4D554C54)
    parser.add_argument("--nccl-debug", default="WARN")
    parser.add_argument("--nccl-lib-dir",
                        help="optional libnccl directory on both hosts")
    parser.add_argument("--socket-ifname",
                        help="optional NCCL_SOCKET_IFNAME override")
    parser.add_argument("--disable-ib", action="store_true",
                        help="set NCCL_IB_DISABLE=1 explicitly")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--scenario",
        choices=("valid", "out_of_range", "duplicate", "max_mismatch",
                 "over_capacity", "generation_mismatch", "session_mismatch",
                 "mtile_mismatch", "topology_mismatch",
                 "expert_capacity_mismatch", "workspace_capacity_mismatch",
                 "gin_probe"),
        default="valid",
    )
    args = parser.parse_args(argv)
    if args.gpus_per_node not in (1, 8):
        raise SystemExit("gpus-per-node must be 1 or 8 for the acceptance topology")
    return args


def launch_rank_processes(args):
    global _active_bootstrap
    _active_bootstrap = b""

    environment = local_environment(args)
    world = 2 * args.gpus_per_node
    launch_id = secrets.token_hex(16)
    processes = []
    outputs = []
    bootstrap_read_fd = -1
    bootstrap_write_fd = -1
    pending_error = None
    try:
        bootstrap_read_fd, bootstrap_write_fd = os.pipe()
        rank0 = subprocess.Popen(
            runner_args(args.binary, 0, args.gpus_per_node, args.seed,
                        args.scenario, launch_id, bootstrap_write_fd, True),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            env=environment,
            pass_fds=(bootstrap_write_fd,),
        )
        processes.append((0, rank0))
        os.close(bootstrap_write_fd)
        bootstrap_write_fd = -1
        bootstrap = read_bootstrap(
            bootstrap_read_fd, min(args.timeout, 60.0)
        )
        os.close(bootstrap_read_fd)
        bootstrap_read_fd = -1

        for rank in range(1, world):
            runner = runner_args(
                args.binary if rank < args.gpus_per_node else args.remote_binary,
                rank, args.gpus_per_node, args.seed, args.scenario, launch_id,
                0,
            )
            command = runner if rank < args.gpus_per_node else remote_command(args, runner)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
            )
            processes.append((rank, process))
            send_bootstrap(process, bootstrap)

        deadline = time.monotonic() + args.timeout
        for rank, process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, args.timeout)
            stdout, stderr = process.communicate(timeout=remaining)
            outputs.append(RankOutput(
                rank,
                process.returncode,
                scrub_output(stdout, bootstrap),
                scrub_output(stderr, bootstrap),
            ))
    except BaseException as error:
        pending_error = error
    finally:
        if bootstrap_read_fd >= 0:
            os.close(bootstrap_read_fd)
        if bootstrap_write_fd >= 0:
            os.close(bootstrap_write_fd)
        drained = stop_processes(processes, args, launch_id)
    if pending_error is not None:
        diagnostic = cleanup_diagnostic(drained)
        if isinstance(pending_error, subprocess.TimeoutExpired):
            pending_error.output = diagnostic
            pending_error.stderr = ""
            raise pending_error
        message = scrub_output(str(pending_error))
        if diagnostic:
            message += "; cleanup=" + diagnostic
        # Do not retain an unsanitized exception as __cause__/debug repr.
        raise RuntimeError(message) from None
    return outputs


def main(argv=None):
    args = parse_args(argv)
    world = 2 * args.gpus_per_node
    outputs = launch_rank_processes(args)

    runtime_lines = []
    for result in outputs:
        evidence = [line for line in result.stdout.splitlines()
                    if line.startswith("NCCL_RUNTIME ")]
        if len(evidence) != 1:
            raise RuntimeError(
                f"rank={result.rank} missing unique NCCL runtime evidence; "
                f"exit={result.returncode}, stdout={result.stdout[-2000:]}, "
                f"stderr={result.stderr[-2000:]}"
            )
        runtime_lines.append(evidence[0])
    print("\n".join(runtime_lines))

    if args.scenario == "gin_probe":
        failures = []
        evidence = []
        for result in outputs:
            lines = [line for line in result.stdout.splitlines()
                     if line.startswith("GIN_DISTRIBUTED_PROBE ")]
            if result.returncode != 0 or len(lines) != 1:
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"evidence={len(lines)}; stdout={result.stdout[-4000:]}; "
                    f"stderr={result.stderr[-4000:]}"
                )
            else:
                evidence.append(lines[0])
        if failures:
            raise RuntimeError("distributed GIN probe failed:\n" +
                               "\n".join(failures))
        print("\n".join(evidence))
        initializable = all("runtime_initializable=1" in line
                            for line in evidence)
        print(f"[PASS] distributed GIN lifecycle probe world={world} "
              f"runtime_initializable={int(initializable)}")
        return

    if args.scenario != "valid":
        errors = []
        failures = []
        for result in outputs:
            markers = [line.removeprefix("CONTROL_ERROR ")
                       for line in result.stdout.splitlines()
                       if line.startswith("CONTROL_ERROR ")]
            recoveries = [line for line in result.stdout.splitlines()
                          if line.startswith(
                              f"RECOVERY_OK rank={result.rank} round=")]
            if (result.returncode == 0 or len(markers) != 20 or
                    len(recoveries) != 20):
                failures.append(
                    f"rank={result.rank} exit={result.returncode}, "
                    f"control_errors={len(markers)}, recoveries={len(recoveries)}; "
                    f"stdout={result.stdout[-4000:]}; stderr={result.stderr[-4000:]}"
                )
            elif any("origin_rank=1" not in marker for marker in markers):
                failures.append(
                    f"rank={result.rank} error lacks origin_rank=1: {markers[-1]}"
                )
            errors.append(markers)
        if failures:
            raise RuntimeError("invalid protocol launch failed:\n" + "\n".join(failures))
        for round_index in range(20):
            round_errors = [rank_errors[round_index] for rank_errors in errors]
            if len(set(round_errors)) != 1:
                raise AssertionError(
                    f"ranks selected different errors in round={round_index}: {round_errors}"
                )
        print(
            f"[PASS] real cross-host 20-round control rejection/recovery: "
            f"scenario={args.scenario}, world={world}, error={errors[0][0]}"
        )
        return

    actual = [None] * world
    failures = []
    for result in outputs:
        if result.returncode:
            failures.append(
                f"rank={result.rank} exit={result.returncode}; "
                f"stdout={result.stdout[-10000:]}; "
                f"stderr={result.stderr[-10000:]}"
            )
            continue
        lines = [line for line in result.stdout.splitlines()
                 if line.startswith("{")]
        if len(lines) != 1:
            failures.append(
                f"rank={result.rank} expected one metadata JSON line, "
                f"got {len(lines)}: {result.stdout[-4000:]}"
            )
            continue
        actual[result.rank] = json.loads(lines[0])
    if failures:
        raise RuntimeError("multinode NCCL launch failed:\n" + "\n".join(failures))

    config = config_for(args.gpus_per_node)
    routes = seeded_inputs(world, config.num_experts, config.top_k, args.seed)
    expected = build_reference_all(config, routes, "cuda-multinode", 701)
    if actual != expected:
        mismatch = next(rank for rank, pair in enumerate(zip(actual, expected))
                        if pair[0] != pair[1])
        raise AssertionError(f"multinode full-field mismatch at rank={mismatch}")
    if not any(routes[rank] for rank in range(args.gpus_per_node, world)):
        raise AssertionError("fixture did not contain any remote-node routes")
    print(
        f"[PASS] real cross-host NCCL control/config/count/summary: nodes=2, "
        f"gpus_per_node={args.gpus_per_node}, world={world}, full-field match"
    )


if __name__ == "__main__":
    try:
        main()
    except subprocess.TimeoutExpired as error:
        diagnostic = scrub_output(error.output or "")
        print(scrub_output(
            f"[FAIL] rank launch timed out after {error.timeout}s; "
            f"cleanup={diagnostic}"
        ), file=sys.stderr)
        raise SystemExit(124)
    except Exception as error:
        print(scrub_output(f"[FAIL] {error}"), file=sys.stderr)
        raise SystemExit(1)
    finally:
        _active_bootstrap = b""
