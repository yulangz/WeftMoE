#!/usr/bin/env python3
"""Launch real NCCL ranks on this host and 10.10.19.233 and cross-check all fields."""

import argparse
import json
import os
import shlex
import subprocess
import sys

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceTopology,
    build_reference_all,
)


MASK64 = (1 << 64) - 1
NCCL_CUDA12_DIR = "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib"


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


def runner_args(binary, rank, gpus_per_node, seed, bootstrap):
    return [
        binary,
        "--rank-runner",
        "--rank", str(rank),
        "--nodes", "2",
        "--gpus-per-node", str(gpus_per_node),
        "--local-rank", str(rank % gpus_per_node),
        "--seed", str(seed),
        "--id", bootstrap,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--remote-binary", required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--remote", default="root@10.10.19.233")
    parser.add_argument("--port", default="2222")
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0x4D554C54)
    parser.add_argument("--nccl-debug", default="WARN")
    args = parser.parse_args()
    if args.gpus_per_node not in (1, 8):
        raise SystemExit("gpus-per-node must be 1 or 8 for the acceptance topology")

    environment = os.environ.copy()
    prior = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = NCCL_CUDA12_DIR + ((":" + prior) if prior else "")
    environment["NCCL_DEBUG"] = args.nccl_debug
    environment["NCCL_SOCKET_IFNAME"] = "=eth0"
    # The two hosts expose different external IB plugin availability. Force the
    # common real socket transport so both ranks select the same backend.
    environment["NCCL_IB_DISABLE"] = "1"
    world = 2 * args.gpus_per_node
    processes = []
    rank0_command = runner_args(args.binary, 0, args.gpus_per_node,
                                args.seed, "GENERATE")
    rank0_process = subprocess.Popen(
        rank0_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment,
    )
    bootstrap_line = rank0_process.stdout.readline().strip()
    if not bootstrap_line.startswith("BOOTSTRAP "):
        stderr = rank0_process.stderr.read()
        rank0_process.kill()
        raise RuntimeError(
            f"rank 0 failed to publish bootstrap ID: {bootstrap_line}; {stderr}"
        )
    bootstrap = bootstrap_line.split(" ", 1)[1]
    processes.append((0, rank0_process))
    for rank in range(1, world):
        if rank < args.gpus_per_node:
            command = runner_args(args.binary, rank, args.gpus_per_node,
                                  args.seed, bootstrap)
        else:
            remote_runner = runner_args(args.remote_binary, rank,
                                        args.gpus_per_node, args.seed, bootstrap)
            remote_command = [
                "env",
                f"LD_LIBRARY_PATH={NCCL_CUDA12_DIR}",
                f"NCCL_DEBUG={args.nccl_debug}",
                "NCCL_SOCKET_IFNAME==eth0",
                "NCCL_IB_DISABLE=1",
                *remote_runner,
            ]
            command = [
                "ssh", "-p", args.port, args.remote,
                shlex.join(remote_command),
            ]
        processes.append((rank, subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, env=environment,
        )))

    actual = [None] * world
    failures = []
    for rank, process in processes:
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            failures.append(f"rank={rank} timed out; stderr={stderr[-2000:]}")
            continue
        if process.returncode:
            failures.append(
                f"rank={rank} exit={process.returncode}; "
                f"stdout={stdout[-20000:]}; stderr={stderr[-20000:]}"
            )
            continue
        lines = [line for line in stdout.splitlines() if line.startswith("{")]
        if len(lines) != 1:
            failures.append(
                f"rank={rank} expected one JSON line, got {len(lines)}: {stdout[-2000:]}"
            )
            continue
        actual[rank] = json.loads(lines[0])
    if failures:
        raise RuntimeError("multinode NCCL launch failed:\n" + "\n".join(failures))

    config = config_for(args.gpus_per_node)
    routes = seeded_inputs(world, config.num_experts, config.top_k, args.seed)
    expected = build_reference_all(config, routes, "cuda-multinode", 701)
    if actual != expected:
        for rank, (lhs, rhs) in enumerate(zip(actual, expected)):
            if lhs != rhs:
                raise AssertionError(f"multinode full-field mismatch at rank={rank}")
        raise AssertionError("multinode full-field mismatch")
    remote_nonempty = any(routes[rank] for rank in range(args.gpus_per_node, world))
    if not remote_nonempty:
        raise AssertionError("fixture did not contain any remote-node routes")
    print(
        f"[PASS] real cross-host NCCL AllGather: nodes=2, "
        f"gpus_per_node={args.gpus_per_node}, world={world}, "
        f"full-field C++/Python match"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise
