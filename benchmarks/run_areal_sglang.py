"""Run the existing Qwen3.5 RL recipe with only the rollout MoE replaced.

Provide the model/dataset paths, AREAL_FILEROOT and AWEX_SOURCE_ROOT required by
AReaL's run_qwen35_comparison.sh. WeftMoE additionally needs the built tensor
extension. Both arms keep the actor's original Megatron implementation.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areal-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("baseline", "weftmoe"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["BACKEND"] = "baseline"
    env.setdefault("TRIAL", f"qwen35-sglang-{args.backend}-{time.strftime('%Y%m%d-%H%M%S')}")
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    enabled = args.backend == "weftmoe"
    rollout_env = {
        "AREAL_USE_WEFTMOE": "1" if enabled else "0",
        "NCCL_CUMEM_ENABLE": "1" if enabled else "0",
        "NCCL_NVLS_ENABLE": "0",
        "PYTHONPATH": "${actor.scheduling_spec.0.env_vars.PYTHONPATH}",
    }
    if enabled:
        rollout_env.update(
            WEFTMOE_TENSOR_EXTENSION=env["WEFTMOE_TENSOR_EXTENSION"],
            WEFTMOE_STATIC_RING_TOKENS=env.get("WEFTMOE_STATIC_RING_TOKENS", "512"),
            WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS="8",
            SGLANG_MEMORY_SAVER_CUDA_GRAPH="0",
        )
    if env.get("NODES", "1") == "2":
        for name in (
            "NCCL_IB_DISABLE", "NCCL_IB_HCA", "NCCL_IB_GID_INDEX", "NCCL_SOCKET_IFNAME"
        ):
            rollout_env[name] = "${actor.scheduling_spec.0.env_vars." + name + "}"
    # Replace the whole spec: the recipe's original spec aliases the actor's.
    fields = ",".join(f"{key}:{json.dumps(value)}" for key, value in rollout_env.items())
    spec = (
        "rollout.scheduling_spec=[{task_type:worker,gpu:1,port_count:2,"
        'mem:"${actor.scheduling_spec.0.mem}",'
        'cmd:"${actor.scheduling_spec.0.cmd}",env_vars:{' + fields + "}}]"
    )
    command = [
        "bash", "examples/cpu_staged_offload/run_qwen35_comparison.sh", spec,
        "stats_logger.wandb.mode=disabled", *overrides,
    ]
    if args.dry_run:
        print(shlex.join(command))
        return
    os.chdir(args.areal_root)
    os.execvpe(command[0], command, env)


if __name__ == "__main__":
    main()
