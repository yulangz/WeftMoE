"""Continuous 100 ms clock samples for both PF benchmark nodes; SIGTERM stops it."""
from pathlib import Path
import signal
import subprocess
import sys

directory = Path(sys.argv[1])
directory.mkdir(parents=True, exist_ok=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
processes, files = [], []
try:
    for node in range(2):
        output = (directory / f"node{node}.csv").open("x")
        files.append(output)
        command = ["timeout", "7200", "nvidia-smi",
                   "--query-gpu=timestamp,index,clocks.sm,power.draw,utilization.gpu",
                   "--format=csv,noheader", "-lms", "100"]
        if node:
            command = ["ssh", "-o", "BatchMode=yes", "-p", "2222", "root@10.10.19.233", *command]
        processes.append(subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT))
    for process in processes:
        process.wait()
finally:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        process.wait()
    for output in files:
        output.close()
