#!/usr/bin/env python3
"""Run the Megatron EP baseline on two hosts over an explicit NCCL Socket path.

The torch elastic rendezvous uses an SSH reverse tunnel because the acceptance
hosts expose only the SSH endpoint to one another.  NCCL payload traffic is not
tunnelled: both nodes use the explicitly selected network interface.
"""

import argparse
import json
import os
import secrets
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nccl_library import configure_nccl_library


def extract_record(text):
    lines = [line for line in text.splitlines()
             if line.startswith("MEGATRON_BASELINE_JSON ")]
    if len(lines) != 1:
        raise RuntimeError(
            "expected one MEGATRON_BASELINE_JSON record, "
            f"actual={len(lines)}")
    return json.loads(lines[0].split(" ", 1)[1])


def terminate(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def remote_cleanup(args, launch_id, requested_signal):
    pattern = f"MEGAMOE_MULTINODE_LAUNCH_ID={launch_id}"
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-p", str(args.port), args.remote,
         f"pkill -{requested_signal} -f -- {shlex.quote(pattern)} || true"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=10.0, check=False)


def benchmark_arguments(args):
    result = [
        "benchmarks/megatron_dispatch_baseline.py",
        "--tokens", str(args.tokens),
        "--hidden", str(args.hidden),
        "--intermediate", str(args.intermediate),
        "--experts", str(args.experts),
        "--top-k", str(args.top_k),
        "--warmups", str(args.warmups),
        "--iterations", str(args.iterations),
        "--runs", "1",
        "--gate-mode", args.gate_mode,
        "--expert-body", args.expert_body,
    ]
    return result


def torchrun_arguments(args, node_rank):
    return [
        args.torchrun if node_rank == 0 else args.remote_torchrun,
        "--nnodes=2",
        f"--nproc-per-node={args.gpus_per_node}",
        f"--node-rank={node_rank}",
        "--master-addr=127.0.0.1",
        f"--master-port={args.master_port}",
        *benchmark_arguments(args),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--master-port", type=int, default=29621)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--torchrun", default="/opt/.venv/bin/torchrun")
    parser.add_argument("--remote-torchrun",
                        default="/opt/.venv/bin/torchrun")
    parser.add_argument("--nccl-lib-dir", required=True)
    parser.add_argument("--socket-ifname", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gate-mode", choices=("fixed-random", "unit"),
                        default="fixed-random")
    parser.add_argument("--expert-body", choices=("identity", "te-grouped-mlp"),
                        default="te-grouped-mlp")
    args = parser.parse_args()
    if args.gpus_per_node <= 0:
        parser.error("--gpus-per-node must be positive")

    launch_id = secrets.token_hex(16)
    environment = os.environ.copy()
    library = configure_nccl_library(environment, args.nccl_lib_dir)
    environment.update({
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": args.socket_ifname,
        "MEGAMOE_MULTINODE_LAUNCH_ID": launch_id,
    })
    remote_overrides = {
        key: environment[key] for key in
        ("LD_LIBRARY_PATH", "MEGAMOE_NCCL_LIBRARY", "NCCL_DEBUG",
         "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME",
         "MEGAMOE_MULTINODE_LAUNCH_ID")
    }
    local_process = None
    remote_process = None
    local_text = ""
    remote_text = ""
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as local_log, \
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as remote_log:
        try:
            local_process = subprocess.Popen(
                torchrun_arguments(args, 0), cwd=Path(__file__).resolve().parents[1],
                env=environment, stdout=local_log, stderr=subprocess.STDOUT,
                text=True)
            time.sleep(0.5)
            remote_command = [
                "env", *(f"{key}={value}"
                         for key, value in remote_overrides.items()),
                *torchrun_arguments(args, 1),
            ]
            remote_process = subprocess.Popen(
                ["ssh", "-o", "BatchMode=yes",
                 "-o", "ExitOnForwardFailure=yes",
                 "-R", f"{args.master_port}:127.0.0.1:{args.master_port}",
                 "-p", str(args.port), args.remote,
                 "cd " + shlex.quote(str(Path.cwd())) + " && exec " +
                 shlex.join(remote_command)],
                stdout=remote_log, stderr=subprocess.STDOUT, text=True)
            deadline = time.monotonic() + args.timeout
            while (local_process.poll() is None or
                   remote_process.poll() is None):
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(
                        "two-host Megatron Socket benchmark", args.timeout)
                time.sleep(0.1)
            if local_process.returncode != 0 or remote_process.returncode != 0:
                raise RuntimeError(
                    f"Megatron launch failed local={local_process.returncode} "
                    f"remote={remote_process.returncode}")
        except BaseException as error:
            terminate(local_process)
            terminate(remote_process)
            remote_cleanup(args, launch_id, "TERM")
            time.sleep(0.25)
            remote_cleanup(args, launch_id, "KILL")
            local_log.seek(0)
            remote_log.seek(0)
            local_text = local_log.read()
            remote_text = remote_log.read()
            raise RuntimeError(
                f"{error}; local_tail={local_text[-6000:]}; "
                f"remote_tail={remote_text[-6000:]}") from None
        finally:
            terminate(local_process)
            terminate(remote_process)
        local_log.seek(0)
        remote_log.seek(0)
        local_text = local_log.read()
        remote_text = remote_log.read()

    record = extract_record(local_text)
    if record.get("world") != 2 * args.gpus_per_node:
        raise RuntimeError(
            f"Megatron world mismatch expected={2 * args.gpus_per_node} "
            f"actual={record.get('world')}")
    if record.get("nccl_runtime_version") != "2.29.7":
        raise RuntimeError(
            f"unexpected NCCL runtime {record.get('nccl_runtime_version')}")
    record["network_backend"] = "nccl-socket"
    record["network_environment"] = {
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": args.socket_ifname,
    }
    record["explicit_nccl_library_path"] = library
    print("MULTINODE_MEGATRON_JSON " + json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        raise SystemExit(1)
