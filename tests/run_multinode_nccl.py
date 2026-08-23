#!/usr/bin/env python3
"""Launch a two-host NCCL protocol test with explicit remote/network options."""

import argparse
import json
import os
import queue
import secrets
import shlex
import subprocess
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceTopology,
    build_reference_all,
)


MASK64 = (1 << 64) - 1


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


def runner_args(binary, rank, gpus_per_node, seed, bootstrap, scenario,
                launch_id):
    return [
        binary,
        "--rank-runner",
        "--rank", str(rank),
        "--nodes", "2",
        "--gpus-per-node", str(gpus_per_node),
        "--local-rank", str(rank % gpus_per_node),
        "--seed", str(seed),
        "--id", bootstrap,
        "--scenario", scenario,
        "--launch-id", launch_id,
    ]


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


def read_bootstrap(process, timeout):
    result = queue.Queue(maxsize=1)

    def reader():
        prefix = []
        for line in iter(process.stdout.readline, ""):
            prefix.append(line)
            if line.startswith("BOOTSTRAP "):
                result.put((line.split(" ", 1)[1].strip(), "".join(prefix)))
                return
        result.put((None, "".join(prefix)))

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        bootstrap, prefix = result.get(timeout=timeout)
    except queue.Empty as error:
        raise RuntimeError(
            "rank 0 did not publish a bootstrap ID before timeout"
        ) from error
    if not bootstrap:
        raise RuntimeError(
            f"rank 0 exited before publishing bootstrap ID; stdout={prefix[-2000:]}"
        )
    return bootstrap, prefix


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
    for _rank, process, _prefix in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for _rank, process, _prefix in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
    for _rank, process, _prefix in processes:
        if process.poll() is None:
            process.wait()
    remote_cleanup(args, launch_id, "TERM")
    time.sleep(0.25)
    remote_cleanup(args, launch_id, "KILL")


def main():
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
                 "expert_capacity_mismatch", "workspace_capacity_mismatch"),
        default="valid",
    )
    args = parser.parse_args()
    if args.gpus_per_node not in (1, 8):
        raise SystemExit("gpus-per-node must be 1 or 8 for the acceptance topology")

    environment = local_environment(args)
    world = 2 * args.gpus_per_node
    launch_id = secrets.token_hex(16)
    processes = []
    outputs = []
    try:
        rank0 = subprocess.Popen(
            runner_args(args.binary, 0, args.gpus_per_node, args.seed,
                        "GENERATE", args.scenario, launch_id),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        processes.append((0, rank0, ""))
        bootstrap, prefix = read_bootstrap(rank0, min(args.timeout, 60.0))
        processes[0] = (0, rank0, prefix)

        for rank in range(1, world):
            runner = runner_args(
                args.binary if rank < args.gpus_per_node else args.remote_binary,
                rank, args.gpus_per_node, args.seed, bootstrap, args.scenario,
                launch_id,
            )
            command = runner if rank < args.gpus_per_node else remote_command(args, runner)
            processes.append((rank, subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            ), ""))

        deadline = time.monotonic() + args.timeout
        for rank, process, prefix in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, args.timeout)
            stdout, stderr = process.communicate(timeout=remaining)
            outputs.append((rank, process.returncode, prefix + stdout, stderr))
    finally:
        stop_processes(processes, args, launch_id)

    runtime_lines = []
    for rank, _returncode, stdout, _stderr in outputs:
        evidence = [line for line in stdout.splitlines()
                    if line.startswith("NCCL_RUNTIME ")]
        if len(evidence) != 1:
            raise RuntimeError(f"rank={rank} missing unique NCCL runtime evidence")
        runtime_lines.append(evidence[0])
    print("\n".join(runtime_lines))

    if args.scenario != "valid":
        errors = []
        failures = []
        for rank, returncode, stdout, stderr in outputs:
            markers = [line.removeprefix("CONTROL_ERROR ")
                       for line in stdout.splitlines()
                       if line.startswith("CONTROL_ERROR ")]
            recoveries = [line for line in stdout.splitlines()
                          if line.startswith(f"RECOVERY_OK rank={rank} round=")]
            if returncode == 0 or len(markers) != 20 or len(recoveries) != 20:
                failures.append(
                    f"rank={rank} exit={returncode}, control_errors={len(markers)}, "
                    f"recoveries={len(recoveries)}; stdout={stdout[-4000:]}; stderr={stderr[-4000:]}"
                )
            elif any("origin_rank=1" not in marker for marker in markers):
                failures.append(f"rank={rank} error lacks origin_rank=1: {markers[-1]}")
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
    for rank, returncode, stdout, stderr in outputs:
        if returncode:
            failures.append(
                f"rank={rank} exit={returncode}; stdout={stdout[-10000:]}; "
                f"stderr={stderr[-10000:]}"
            )
            continue
        lines = [line for line in stdout.splitlines() if line.startswith("{")]
        if len(lines) != 1:
            failures.append(
                f"rank={rank} expected one metadata JSON line, got {len(lines)}: "
                f"{stdout[-4000:]}"
            )
            continue
        actual[rank] = json.loads(lines[0])
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
        print(f"[FAIL] rank launch timed out after {error.timeout}s", file=sys.stderr)
        raise SystemExit(124)
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise
