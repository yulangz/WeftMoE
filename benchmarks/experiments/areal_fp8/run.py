"""AReaL BF16/FP8 routed-expert RL, 16K, 32 prompts x 16 samples."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import secrets
import socket
import subprocess
import sys
import time

WEFT = Path(__file__).resolve().parents[3]
AREAL = WEFT.parent / "AReaL"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("baseline", "weftmoe", "fp8"))
    parser.add_argument("--netrc", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--attempt", default="")
    parser.add_argument("--dp-attention", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--areal-path", type=Path, default=AREAL)
    parser.add_argument("--deps-path", type=Path)
    parser.add_argument("--extension", type=Path)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-base-url", default=os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai"))
    parser.add_argument("--comparison-baseline", type=Path)
    parser.add_argument('--release-decode-graphs', action='store_true')
    args = parser.parse_args()
    _, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile_hard, nofile_hard))
    run = args.output_root.resolve() / args.variant
    if (run / "status.json").exists():
        raise FileExistsError(f"Preserve the existing trial; choose a new output root: {run}")
    run.mkdir(parents=True, exist_ok=True)
    weft = args.variant != "baseline"
    model = args.model_path.resolve()
    data = args.data_path.resolve()
    areal = args.areal_path.resolve()
    extension = args.extension.resolve() if args.extension else None
    assert args.netrc.is_file() and model.is_dir() and data.is_dir() and areal.is_dir()
    if weft and (extension is None or not extension.is_file()):
        parser.error("WeftMoE variants require --extension pointing to a built tensor adapter")
    python_paths = [str(areal), str(WEFT)]
    if args.deps_path:
        python_paths.insert(0, str(args.deps_path.resolve()))

    env = os.environ.copy()
    for key in list(env):
        if (key.startswith(("WEFTMOE_", "AREAL_", "WANDB_", "SGLANG_", "HAPI_", "CODEX_"))
                or any(word in key for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "ACCESS_KEY"))
                or key in (
            "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "CUDA_LAUNCH_BLOCKING",
            "TORCH_DISTRIBUTED_DEBUG", "TORCH_LOGS", "NCCL_ALGO",
        )):
            env.pop(key, None)
    env.update(
        PATH=f"{Path(sys.executable).parent}:{env['PATH']}",
        PYTHONPATH=os.pathsep.join(python_paths),
        AREAL_SPMD_MODE="0",
        AREAL_PROXY_ADMIN_API_KEY=secrets.token_urlsafe(32),
        AREAL_EXPERIMENT_ROOT=str(run / "run"),
        QWEN3_5_122B_A10B_MODEL_PATH=str(model),
        DAPO_MATH_17K_PATH=str(data),
        WANDB_BASE_URL=args.wandb_base_url, WANDB_MODE="online",
        WANDB_CONSOLE="off", WANDB_DISABLE_CODE="true",
        NETRC=str(args.netrc.resolve()),
        OMP_NUM_THREADS="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
    )
    worker_env = dict(
        PYTHONPATH=env["PYTHONPATH"],
        AREAL_USE_WEFTMOE="1" if weft else "0",
        CUDA_DEVICE_MAX_CONNECTIONS="1", NCCL_CUMEM_ENABLE="1",
        NCCL_DEBUG="WARN", NCCL_NVLS_ENABLE="0",
        NCCL_SOCKET_IFNAME="eth0", NCCL_IB_GID_INDEX="3",
        NCCL_IB_HCA=",".join(f"mlx5_bond_{n}:1" for n in range(8)),
        LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/local/cuda/lib64",
        OMP_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
        WEFTMOE_AUTOTUNE_TABLE="",
    )
    if weft:
        worker_env.update(
            WEFTMOE_TENSOR_EXTENSION=str(extension),
            WEFTMOE_FP8="1" if args.variant == "fp8" else "0",
            WEFTMOE_STATIC_RING_TOKENS="2048",
            WEFTMOE_STATIC_RING_CONTRIBUTION_WINDOWS="2",
            WEFTMOE_PF_MIN_RING_DEPTH="2", WEFTMOE_PF_CREDIT_WINDOWS="2",
            WEFTMOE_SGLANG_KEEP_DECODE_GRAPHS="0" if args.release_decode_graphs else "1",
        )
    env.update(worker_env)
    overrides = [
        f"experiment_name={args.experiment}",
        f"trial_name={args.variant}-b32n16-{args.steps}step{args.attempt}",
        f"total_train_steps={args.steps}", "seed=1",
        "train_dataset.batch_size=32", "gconfig.n_samples=16",
        "actor.reward_norm.group_size=16", "rollout.consumer_batch_size=32",
        "rollout.max_concurrent_rollouts=32", "rollout.queue_size=64",
        "rollout.max_head_offpolicyness=1",
        "++rollout.serialize_group_samples=false", "++rollout.deterministic_sampling=false",
        "sglang.max_running_requests=512", "sglang.cuda_graph_max_bs=512",
        "sglang.mem_fraction_static=0.7",
        "sglang.context_length=16384", "gconfig.max_tokens=16383",
        "gconfig.max_new_tokens=16384", "actor.mb_spec.max_tokens_per_mb=16384",
        "++sglang.max_prefill_tokens=32768", "++sglang.chunked_prefill_size=-1",
        "actor.megatron.moe_token_dispatcher_type=" + ("alltoall" if weft else "flex"),
        "++actor.megatron.moe_enable_deepep=" + ("false" if weft else "true"),
        "stats_logger.wandb.mode=online",
        f"+stats_logger.wandb.entity={args.wandb_entity}",
        f"+stats_logger.wandb.project={args.wandb_project}",
        f"+stats_logger.wandb.group={args.experiment}",
        "saver.freq_steps=null", "saver.freq_epochs=null", "saver.freq_secs=null",
        "recover.mode=disabled", "recover.freq_steps=null",
        "recover.freq_epochs=null", "recover.freq_secs=null",
        "memory_profiler.profile_steps=[]", "perf_tracer.enabled=false",
    ]
    if args.dp_attention:
        overrides += [
            "sglang.dp_size=8", "sglang.enable_dp_attention=true",
            "sglang.cuda_graph_max_bs=64",
        ]
    for role in ("actor", "rollout"):
        for key, value in worker_env.items():
            overrides.append(f"++{role}.scheduling_spec.0.env_vars.{key}={json.dumps(value)}")
    config_args = ["--config", "examples/cpu_staged_offload/dapo-math_grpo_qwen3_5_122b_muon.yaml", *overrides]
    command = [sys.executable, "examples/cpu_staged_offload/dapo-math_rl_cpu_staged.py", *config_args]
    os.environ.update(env)
    sys.path[:0] = python_paths
    os.chdir(areal)
    from areal.api.cli_args import load_expr_config
    from examples.multi_turn_math.config import MultiTurnGRPOConfig

    config, _ = load_expr_config(config_args, MultiTurnGRPOConfig)
    summary = dict(
        variant=args.variant, routed_expert_dtype="fp8_e4m3" if args.variant == "fp8" else "bfloat16",
        fp8_weight_grad_dtype="bfloat16", fp8_rollout_graph_policy="recapture_after_weight_update",
        fp8_bf16_residency="fp32_master_to_fp8_no_bf16_expert_shadow",
        bf16_rollout_graph_policy="recapture_after_weight_update" if args.release_decode_graphs else "retain",
        host=socket.gethostname(), steps=config.total_train_steps,
        nofile_limit=resource.getrlimit(resource.RLIMIT_NOFILE),
        model=config.actor.path, dataset=config.train_dataset.path,
        model_config_sha256=hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        experiment=args.experiment, trial=config.trial_name,
        allocation=config.allocation_mode, context=config.sglang.context_length,
        total_token_cap=config.gconfig.max_tokens, completion_cap=config.gconfig.max_new_tokens,
        microbatch_token_cap=config.actor.mb_spec.max_tokens_per_mb,
        prompts=config.train_dataset.batch_size, samples_per_prompt=config.gconfig.n_samples,
        concurrent_groups=config.rollout.max_concurrent_rollouts,
        consumer_groups=config.rollout.consumer_batch_size,
        queue_groups=config.rollout.queue_size,
        max_staleness=config.rollout.max_head_offpolicyness,
        max_running_requests=config.sglang.max_running_requests,
        dp_size=config.sglang.dp_size,
        enable_dp_attention=config.sglang.enable_dp_attention,
        cuda_graph_max_bs=config.sglang.cuda_graph_max_bs,
        mem_fraction_static=config.sglang.mem_fraction_static,
        max_prefill_tokens=config.sglang.max_prefill_tokens,
        reward_verifier="math_verify.verify(gold, prediction)",
        optimizer=config.actor.optimizer.type, recompute=config.actor.megatron.recompute_granularity,
        dispatcher=config.actor.megatron.moe_token_dispatcher_type,
        deepep=config.actor.megatron.moe_enable_deepep, wandb=config.stats_logger.wandb.mode,
        weft_commit=subprocess.check_output(["git", "-C", str(WEFT), "rev-parse", "HEAD"], text=True).strip(),
        areal_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        areal_diff_sha256=hashlib.sha256(subprocess.check_output(["git", "diff"])).hexdigest(),
        versions={p: importlib.metadata.version(p) for p in ("torch", "sglang", "awex", "deep-ep", "megatron-core", "flashinfer-python")},
        actor_effective_nccl_cumem="1" if weft else "0",
        command=command,
    )
    if weft:
        from weftmoe.runtime import _extension
        summary["native_tuning_id"] = _extension().forward_tuning_id()
        summary["native_sha256"] = hashlib.sha256(extension.with_name("libweftmoe.so").read_bytes()).hexdigest()
        summary["extension_sha256"] = hashlib.sha256(extension.read_bytes()).hexdigest()
    assert summary["context"] == 16384 and summary["microbatch_token_cap"] == 16384
    assert (config.train_dataset.batch_size, config.gconfig.n_samples) == (32, 16)
    assert config.actor.reward_norm.group_size == 16
    assert config.rollout.max_head_offpolicyness == 1
    assert config.sglang.max_running_requests == 512
    assert config.sglang.cuda_graph_max_bs == (64 if args.dp_attention else 512)
    assert config.sglang.dp_size == (8 if args.dp_attention else 1)
    assert config.recover.mode == "disabled"
    assert all(getattr(c, attr) is None for c in (config.saver, config.recover)
               for attr in ("freq_steps", "freq_epochs", "freq_secs"))
    (run / "weftmoe.patch").write_bytes(subprocess.check_output(["git", "-C", str(WEFT), "diff"]))
    (run / "areal.patch").write_bytes(subprocess.check_output(["git", "diff"]))
    (run / "manifest.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "command"}, indent=2), flush=True)
    if args.check:
        return
    with (run / "rl.log").open("w", buffering=1) as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        status = dict(pid=process.pid, host=socket.gethostname(), started_at=time.time())
        (run / "status.json").write_text(json.dumps(status, indent=2))
        print(f"Started {args.variant}: pid={process.pid}, log={run / 'rl.log'}", flush=True)
        status["exit_code"] = process.wait()
        status["finished_at"] = time.time()
        (run / "status.json").write_text(json.dumps(status, indent=2))
        report = [sys.executable, str(Path(__file__).with_name("report.py")), str(run),
                  "--label", args.variant]
        if args.comparison_baseline:
            report += ["--baseline", str(args.comparison_baseline.resolve())]
        subprocess.run(report, check=True)
        print(f"Finished {args.variant}: {status['exit_code']}", flush=True)
        sys.exit(status["exit_code"])


if __name__ == "__main__":
    main()
