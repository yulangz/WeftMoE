#!/usr/bin/env python3
"""Run native cached-state validation and its independent autograd reference.

The two-node mode requires a shared repository/artifact path and SSH access.
An SSH reverse tunnel carries only torchrun rendezvous; NCCL uses the selected
interface. This program is a correctness check, never a performance sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import shlex
import subprocess
import sys
import time

from run_multinode_megatron_socket import remote_cleanup, terminate
from run_performance_matrix import ROOT, LIBPATH, environment


def run(command, env, log, timeout=600):
    with log.open("w") as output:
        output.write("COMMAND " + json.dumps(command) + "\n")
        output.flush()
        result = subprocess.run(command, env=env, cwd=ROOT, stdout=output,
                                stderr=subprocess.STDOUT, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"validation failed: {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world", type=int, choices=(1, 2, 8, 16), required=True)
    parser.add_argument("--tokens", type=int, default=129)
    parser.add_argument("--hidden", type=int, required=True)
    parser.add_argument("--intermediate", type=int, required=True)
    parser.add_argument("--experts", type=int,
                        help="global expert count; default is max(16, 2 * EP)")
    parser.add_argument("--top-k", type=int, choices=(1, 6, 8), required=True)
    parser.add_argument("--remote", default="root@10.10.19.233")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--socket-ifname", default="bond0")
    parser.add_argument("--master-port", type=int, default=29671)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    args = parser.parse_args()
    if args.reference_only and args.native_only:
        parser.error("choose at most one partial validation mode")
    experts = args.experts if args.experts is not None else max(16, args.world * 2)
    if experts < args.top_k or experts % args.world or args.tokens <= 0:
        parser.error("invalid expert count, top-k, EP, or token count")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    artifact = args.output / "tensors"
    artifact.mkdir(exist_ok=True)
    binary = str(args.binary.resolve())
    env = environment()
    env.update(NCCL_SOCKET_IFNAME=args.socket_ifname,
               WEFTMOE_BENCH_EXPERTS=str(experts),
               WEFTMOE_P2P_TOP_K=str(args.top_k),
               WEFTMOE_BACKWARD_TOKENS=str(args.tokens),
               WEFTMOE_BACKWARD_OUTPUT_DIR=str(artifact),
               WEFTMOE_BENCH_CHECK_REPEAT="1")
    if args.world < 16:
        env["NCCL_IB_DISABLE"] = "1"
        native = [sys.executable, str(ROOT / "tests/run_local_activation_p2p.py"),
                  "--binary", binary, "--world", str(args.world),
                  "--backward-validation", "--backward-tokens", str(args.tokens),
                  "--top-k", str(args.top_k), "--rounds", "3",
                  "--backward-output-dir", str(artifact), "--timeout", "600"]
    else:
        env.update(NCCL_IB_DISABLE="0", NCCL_IB_GID_INDEX="3",
                   NCCL_IB_HCA=",".join(f"mlx5_bond_{i}:1" for i in range(8)))
        native = [sys.executable, str(ROOT / "tests/run_multinode_nccl.py"),
                  "--binary", binary, "--remote-binary", binary,
                  "--remote", args.remote, "--port", str(args.port),
                  "--gpus-per-node", "8", "--scenario", "gin_backward_smoke",
                  "--nccl-lib-dir", LIBPATH, "--socket-ifname", args.socket_ifname,
                  "--enable-ib", "--timeout", "600"]
    if not args.reference_only:
        run(native, env, args.output / "native.log")
    if args.native_only:
        print(json.dumps(dict(native_passed=True, output=str(args.output))), flush=True)
        return
    oracle = [str(ROOT / "benchmarks/compare_backward_outputs.py"),
              "--artifact-dir", str(artifact), "--tokens", str(args.tokens),
              "--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
              "--experts", str(experts), "--top-k", str(args.top_k)]
    if args.world < 16:
        run(["/opt/.venv/bin/torchrun", "--standalone",
             f"--nproc-per-node={args.world}", *oracle], env,
            args.output / "reference.log")
    else:
        launch_id = secrets.token_hex(16)
        env["WEFTMOE_MULTINODE_LAUNCH_ID"] = launch_id
        def command(node):
            return ["/opt/.venv/bin/torchrun", "--nnodes=2", "--nproc-per-node=8",
                    f"--node-rank={node}", "--master-addr=127.0.0.1",
                    f"--master-port={args.master_port}", *oracle]
        remote_env = [f"{key}={value}" for key, value in env.items()
                      if key.startswith(("NCCL_", "WEFTMOE_")) or
                      key in ("LD_LIBRARY_PATH", "OMP_NUM_THREADS")]
        remote_command = ["ssh", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes",
                          "-R", f"{args.master_port}:127.0.0.1:{args.master_port}",
                          "-p", str(args.port), args.remote,
                          "cd " + shlex.quote(str(ROOT)) + " && exec " +
                          shlex.join(["env", *remote_env, *command(1)])]
        local = remote = None
        with (args.output / "reference.log").open("w") as left, \
                (args.output / "reference.remote.log").open("w") as right:
            try:
                local = subprocess.Popen(command(0), cwd=ROOT, env=env,
                                         stdout=left, stderr=subprocess.STDOUT)
                remote = subprocess.Popen(remote_command, stdout=right,
                                          stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 600
                while local.poll() is None or remote.poll() is None:
                    if time.monotonic() > deadline or any(
                            p.poll() not in (None, 0) for p in (local, remote)):
                        raise RuntimeError(f"distributed reference failed: {args.output}")
                    time.sleep(0.2)
            finally:
                failed = any(p and p.poll() != 0 for p in (local, remote))
                terminate(local)
                terminate(remote)
                if failed:
                    remote_cleanup(args, launch_id, "TERM")
                    remote_cleanup(args, launch_id, "KILL")
    records = [json.loads(line.split(" ", 1)[1])
               for line in (args.output / "reference.log").read_text().splitlines()
               if line.startswith("BACKWARD_REFERENCE_JSON ")]
    if len(records) != 1 or not records[0]["passed"]:
        raise RuntimeError(f"missing successful reference: {args.output}")
    (args.output / "result.json").write_text(json.dumps(records[0], indent=2) + "\n")
    print(json.dumps(dict(passed=True, output=str(args.output),
                          accuracy=records[0]["accuracy"])), flush=True)


if __name__ == "__main__":
    main()
