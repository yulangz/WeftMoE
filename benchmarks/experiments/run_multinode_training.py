"""Run one torchrun command on two hosts sharing the repository and native build."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--master-addr", required=True)
    parser.add_argument("--master-port", type=int, default=29641)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--script", default="benchmarks/torchrun_training.py")
    parser.add_argument("--env", action="append", default=[])
    args, cli = parser.parse_known_args()
    if cli and cli[0] == "--":
        cli = cli[1:]
    native = args.native.resolve()
    extension = next(native.glob("_runtime*.so"))
    environment = dict(
        CUDA_VISIBLE_DEVICES=",".join(map(str, range(args.gpus_per_node))),
        OMP_NUM_THREADS="1", NCCL_SOCKET_IFNAME="bond0", NCCL_IB_DISABLE="0",
        NCCL_IB_GID_INDEX="3",
        NCCL_IB_HCA=",".join(f"mlx5_bond_{i}:1" for i in range(8)),
        NCCL_DEBUG="WARN", WEFTMOE_NCCL_LIBRARY="/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64",
        WEFTMOE_TENSOR_EXTENSION=str(extension),
        # GIN and local PF both use source FP8 and compact checkpoints.
        TEST_FP8_COMPACT="1", TEST_FP8_AB="1", TEST_FP8_SOURCE_CACHE="1",
        TEST_FP8_CANONICAL="1",
    )
    for value in args.env:
        key, val = value.split("=", 1)
        environment[key] = val
    args.output.mkdir(parents=True, exist_ok=True)
    jobs, logs, commands = [], [], []
    started = time.time()
    for node in range(2):
        command = ["timeout", "--kill-after=10s", f"{args.timeout}s", "env",
                   "-u", "WEFTMOE_AUTOTUNE_TABLE",
                   *(f"{k}={v}" for k, v in environment.items()), sys.executable,
                   "-m", "torch.distributed.run", "--nnodes=2",
                   f"--nproc-per-node={args.gpus_per_node}", f"--node-rank={node}",
                   f"--master-addr={args.master_addr}", f"--master-port={args.master_port}",
                   args.script, *cli]
        if node:
            command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                       "-p", str(args.ssh_port), args.remote,
                       "cd " + shlex.quote(str(ROOT)) + " && exec " + shlex.join(command)]
        commands.append(command)
        log = (args.output / f"node{node}.log").open("w")
        logs.append(log)
        jobs.append(subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True))
    codes = [job.wait() for job in jobs]
    for log in logs:
        log.close()
    result = dict(commands=commands,environment=environment,returncodes=codes,
                  elapsed_seconds=time.time()-started,native=str(native))
    (args.output / "launch.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(returncodes=codes,output=str(args.output)),ensure_ascii=False),flush=True)
    return int(any(codes))


if __name__ == "__main__":
    raise SystemExit(main())
