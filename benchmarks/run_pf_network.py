#!/usr/bin/env python3
"""One- or two-node PF comparison, using a prepared native artifact."""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import threading

from run_gin_pack import monitor_clocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, default=Path("build/pf_network_20261009"))
    parser.add_argument("--remote", default="root@10.10.19.233")
    parser.add_argument("--port", type=int, default=2222, help="SSH port")
    parser.add_argument("--master-addr", default="10.10.16.219")
    parser.add_argument("--master-port", type=int, default=31621)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--nodes", type=int, choices=(1, 2), default=2)
    parser.add_argument("--benchmark", type=Path,
                        default=Path("benchmarks/torchrun_megamoe_prefill.py"))
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    benchmark = (root / args.benchmark).resolve()
    extension, = args.artifact.resolve().glob("_runtime*.so")
    args.output.mkdir(parents=True, exist_ok=True)
    env = dict(PYTHONPATH=str(root), OMP_NUM_THREADS="1", NCCL_DEBUG="WARN",
               NCCL_SOCKET_IFNAME="eth0", NCCL_IB_GID_INDEX="3", NCCL_IB_DISABLE="0",
               LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64",
               WEFTMOE_TENSOR_EXTENSION=str(extension), WEFTMOE_AUTOTUNE_TABLE="")
    env.update(value.split("=", 1) for value in args.env)
    cli = args.benchmark_args
    if cli[:1] == ["--"]:
        cli = cli[1:]
    library = args.artifact.resolve() / "libweftmoe.so"
    manifest = dict(library=str(library),
        library_sha256=hashlib.file_digest(library.open("rb"), "sha256").hexdigest(),
        extension=str(extension),
        extension_sha256=hashlib.file_digest(extension.open("rb"), "sha256").hexdigest(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                               text=True).strip(),
        source_diff_sha256=hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD"], cwd=root)).hexdigest(),
        environment=env, benchmark=str(benchmark), benchmark_args=cli,
        master_addr=args.master_addr, master_port=args.master_port,
        nodes=args.nodes, gpus_per_node=args.gpus_per_node)
    with (args.output / f"{args.label}.manifest.json").open("x") as file:
        json.dump(manifest, file, indent=2)
    processes, logs, clocks = [], [], []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_clocks, args=(args, stop, clocks))
    monitor.start()
    try:
        for node in range(args.nodes):
            command = ["env", *(f"{key}={value}" for key, value in env.items()),
                       "timeout", "--kill-after=20s", "420s", "/opt/.venv/bin/python",
                       "-m", "torch.distributed.run", f"--nnodes={args.nodes}",
                       f"--nproc-per-node={args.gpus_per_node}", f"--node-rank={node}",
                       f"--master-addr={args.master_addr}", f"--master-port={args.master_port}",
                       str(benchmark), *cli]
            if node:
                command = ["ssh", "-o", "BatchMode=yes", "-p", str(args.port), args.remote,
                           "cd " + shlex.quote(str(root)) + " && " + shlex.join(command)]
            log = (args.output / f"{args.label}_ep{args.nodes * args.gpus_per_node}_node{node}.log").open("w")
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=root, stdout=log,
                                               stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
        stop.set()
        monitor.join()
        (args.output / f"{args.label}.clocks.json").write_text(json.dumps(clocks))
    print(args.label, codes, flush=True)
    if any(codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
