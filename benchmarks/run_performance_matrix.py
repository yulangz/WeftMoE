#!/usr/bin/env python3
"""Reproducible BF16 routed-expert benchmarks with per-call tensor allocation.

Builds and measurements are separate. GPU cases run serially; compilation uses
at most 16 concurrent commands. Pass a frozen source tree to --source to retain
an independently runnable before binary. Tokens always mean tokens per rank.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune_forward as fw
import autotune_backward as bw
import unified_tuning

MODELS = {
    "qwen": dict(hidden=2048, intermediate=512, experts=256, top_k=8),
    "flash": dict(hidden=4096, intermediate=2048, experts=256, top_k=6),
}
LIBPATH = "/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def environment():
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("WEFTMOE_", "NCCL_", "NVSHMEM_"))}
    env.update(LD_LIBRARY_PATH=LIBPATH, NCCL_DEBUG="WARN", OMP_NUM_THREADS="1",
               WEFTMOE_NCCL_LIBRARY="/usr/lib/x86_64-linux-gnu/libnccl.so.2",
               WEFTMOE_FORWARD_BENCH_RETAIN_STATE="1",
               WEFTMOE_BENCH_SYNC_RANKS="1",
               WEFTMOE_BENCH_ALLOCATE_TENSORS="1")
    return env


def plan(args, model, ep, phase):
    module = fw if phase == "forward" else bw
    system = dict(nodes=ep // 8, gpus_per_node=8,
                  runtime_library_path=LIBPATH, socket_ifname=args.socket_ifname)
    if ep > 8:
        system.update(remote=args.remote, port=args.port, enable_ib=True)
    topology = "single_node" if ep == 8 else "multi_node"
    path = args.config or args.source / "configs" / f"{phase}_h200_{topology}_autotune.json"
    result = module.load_plan(path, model=MODELS[model], system=system)
    result["jit"]["cache_root"] = str(args.output / "compile")
    return module, result


def build(args):
    if args.tuning_headers:
        return build_profiles(args)
    tasks = [(m, e, p) for m in args.models for e in args.eps for p in args.phases]
    workers = min(8, len(tasks))

    def one(task):
        model, ep, phase = task
        key = f"{model}_ep{ep}_{phase}"
        module, config = plan(args, model, ep, phase)
        candidate = module.expand_candidates(config)[0]
        for override in args.override:
            parameter, value = override.split("=", 1)
            candidate["config"][parameter] = json.loads(value)
        module.validate_tuning(candidate["config"])
        candidate["name"] = key
        compiler_type = fw.ForwardJitCompiler if phase == "forward" else bw.BackwardJitCompiler
        compiler = compiler_type(args.source, config["jit"], environment(),
                                 nccl_lib_dir="/usr/lib/x86_64-linux-gnu")
        result = module.compile_candidate(candidate, compiler, args.output,
                                           command_jobs=max(1, 16 // workers))
        record = dict(model=model, ep=ep, phase=phase, candidate=candidate,
                      source=str(args.source), build=result)
        save(args.output / f"{key}.build.json", record)
        if result["returncode"]:
            raise RuntimeError(f"build failed: {key}; {result}")
        print(f"built {key}: {result['binary']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(one, tasks))


def build_profiles(args):
    """Build each checked-in combined profile once and reuse it across EP sizes."""
    if args.override:
        raise ValueError("--override cannot be combined with --tuning-headers")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    for model in args.models:
        profile = (args.tuning_headers / f"{model}.hpp").resolve()
        header = profile.read_text()
        digest = hashlib.sha256(header.encode()).hexdigest()
        _, config = plan(args, model, args.eps[0], "forward")
        compiler = fw.ForwardJitCompiler(
            args.source, config["jit"], environment(),
            nccl_lib_dir="/usr/lib/x86_64-linux-gnu")
        built = compiler.build(
            candidate_name=model, config_digest=digest, generated_header=header,
            log_path=args.output / f"{model}.build.log", command_jobs=3)
        for ep in args.eps:
            for phase in args.phases:
                key = f"{model}_ep{ep}_{phase}"
                save(args.output / f"{key}.build.json", dict(
                    model=model, ep=ep, phase=phase,
                    candidate=dict(name=key, tuning_header=str(profile)),
                    tuning_header_sha256=digest, source=str(args.source),
                    source_revision=revision, build=built))
        if built["returncode"]:
            raise RuntimeError(f"profile build failed: {model}; {built}")
        print(f"built profile {model}: {built['binary']}", flush=True)


def combined_generated_header(name, forward, backward):
    """Join independent phase resources with one checked shared protocol."""
    # Historical build records used four physical ring slots. Preserve that
    # policy without mutating the records or overriding an explicit value.
    compatibility_defaults = {
        key: unified_tuning.DEFAULTS[key]
        for key in ("gin_ring_depth",)}
    forward = {**compatibility_defaults, **forward}
    backward = {**compatibility_defaults, **backward}
    for dimension in ("main_hidden", "main_intermediate"):
        if forward[dimension] != backward[dimension]:
            raise ValueError(f"{name}: Forward/Backward dimensions differ")
    unified_tuning.validate_shared_parameters(forward, backward)
    header = fw.generated_header(name, forward) + "\n" + bw.generated_header(name, backward)
    for dimension in ("HIDDEN", "INTERMEDIATE"):
        header = header.replace(
            f"#define WEFTMOE_BWD_MAIN_{dimension} {backward['main_' + dimension.lower()]}",
            f"#define WEFTMOE_BWD_MAIN_{dimension} WEFTMOE_FWD_MAIN_{dimension}")
    return header


def build_pair(args):
    """Compile both selected phases into one deployable specialization."""
    if not args.forward_builds or not args.backward_builds:
        raise ValueError("build-pair requires --forward-builds and --backward-builds")
    tasks = [(m, e) for m in args.models for e in args.eps]
    workers = min(8, len(tasks))

    def one(task):
        model, ep = task
        key = f"{model}_ep{ep}"
        candidates = {}
        for phase, directory in (("forward", args.forward_builds),
                                 ("backward", args.backward_builds)):
            record = json.loads((directory / f"{key}_{phase}.build.json").read_text())
            candidates[phase] = record["candidate"]
        fc, bc = [candidates[p]["config"] for p in ("forward", "backward")]
        header = combined_generated_header(key, fc, bc)
        (args.output / f"{key}.tuning.hpp").write_text(header)
        _, config = plan(args, model, ep, "forward")
        compiler = fw.ForwardJitCompiler(args.source, config["jit"], environment(),
                                         nccl_lib_dir="/usr/lib/x86_64-linux-gnu")
        built = compiler.build(candidate_name=key,
                               config_digest=hashlib.sha256(header.encode()).hexdigest(),
                               generated_header=header,
                               log_path=args.output / f"{key}.build.log",
                               command_jobs=max(1, 16 // workers))
        if built["returncode"]:
            raise RuntimeError(f"combined build failed: {key}; {built}")
        for phase, candidate in candidates.items():
            save(args.output / f"{key}_{phase}.build.json",
                 dict(model=model, ep=ep, phase=phase, candidate=candidate,
                      combined_candidates=candidates, source=str(args.source), build=built))
        print(f"built combined {key}: {built['binary']}", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(one, tasks))


def weft(args):
    env = environment()
    if getattr(args, "repartition", False):
        env["WEFTMOE_FORWARD_REPARTITION_EP8"] = "1"
    if args.check_repeats:
        env["WEFTMOE_BENCH_CHECK_REPEAT"] = "1"
    for model in args.models:
        for ep in args.eps:
            for tokens in args.tokens:
                for phase in args.phases:
                    key = f"{model}_ep{ep}_{phase}"
                    built = json.loads((args.output / f"{key}.build.json").read_text())
                    module, config = plan(args, model, ep, phase)
                    benchmark = dict(config["benchmark"], tokens_per_rank=tokens,
                                     timeout_seconds=600)
                    name = f"{key}_t{tokens}_{args.label}"
                    path = args.output / f"{name}.json"
                    if path.exists():
                        old = json.loads(path.read_text())
                        if old.get("returncode") == 0 and old.get("record"):
                            if not (old["record"].get("allocate_tensors") and
                                    old["record"].get("synchronize_ranks")):
                                raise RuntimeError(f"existing record has a different timing scope; use a new label/output: {path}")
                            continue
                    run_env = dict(env)
                    if args.dump_output and phase == "forward":
                        directory = args.dump_output.resolve() / name
                        directory.mkdir(parents=True, exist_ok=True)
                        run_env.update(WEFTMOE_FORWARD_OUTPUT_DIR=str(directory),
                                       WEFTMOE_FORWARD_DUMP_TOKENS=str(args.dump_tokens),
                                       WEFTMOE_FORWARD_DUMP_INPUTS="prefix")
                    result = module.benchmark_candidate(
                        dict(built["candidate"], name=name), Path(built["build"]["binary"]),
                        benchmark, args.output, run_env, rounds=args.iterations + 10,
                        suffix="measure")
                    save(path, result)
                    if result["returncode"] or result["record"] is None:
                        raise RuntimeError(f"benchmark failed: {path}")
                    record = result["record"]
                    if not record.get("allocate_tensors"):
                        raise RuntimeError(f"tensor allocation not included: {path}")
                    if not record.get("synchronize_ranks"):
                        raise RuntimeError(f"untimed rank barrier missing: {path}")
                    gpu = "forward" if phase == "forward" else "total"
                    print(f"{name}: wall={record['wall_slowest_rank_us']['mean']/1000:.3f} ms "
                          f"GPU={record[gpu+'_slowest_rank_us']['mean']/1000:.3f} ms", flush=True)


def flex(args):
    for model in args.models:
        for ep in args.eps:
            for tokens in args.tokens:
                sms = args.flex_sms or (80 if ep == 8 else 48)
                name = f"{model}_ep{ep}_flex_t{tokens}_sm{sms}_{args.label}"
                path = args.output / f"{name}.json"
                if path.exists():
                    existing = json.loads(path.read_text())
                    if not (existing.get("allocate_tensors") and existing.get("synchronize_ranks")):
                        raise RuntimeError(f"existing Flex record has a different timing scope; use a new label/output: {path}")
                    continue
                shape = MODELS[model]
                options = ["--tokens", str(tokens), "--warmups", "10",
                           "--iterations", str(args.iterations), "--dispatcher", "flex",
                           "--deepep-num-sms", str(sms), "--gate-mode", "fixed-random",
                           "--validation-repeats", str(args.validation_repeats), "--allocate-tensors", "--measure-forward"]
                if args.dump_output:
                    options.extend(["--save-output-dir", str(args.dump_output.resolve() / name)])
                for key, value in shape.items():
                    options.extend(["--" + key.replace("_", "-"), str(value)])
                env = environment()
                env["NCCL_SOCKET_IFNAME"] = args.socket_ifname
                if ep == 8:
                    env["NCCL_IB_DISABLE"] = "1"
                    command = ["/opt/.venv/bin/torchrun", "--standalone", "--nproc-per-node=8",
                               str(ROOT / "benchmarks/megatron_backward_baseline.py"), *options]
                    prefix = "MEGATRON_BASELINE_JSON "
                else:
                    command = [sys.executable, str(ROOT / "benchmarks/run_multinode_megatron_rdma.py"),
                               *options, "--remote", args.remote, "--port", str(args.port),
                               "--nccl-lib-dir", LIBPATH, "--socket-ifname", args.socket_ifname,
                               "--ib-hca", ",".join(f"mlx5_bond_{i}:1" for i in range(8)),
                               "--ib-gid-index", "3", "--workload", "backward", "--timeout", "900",
                               "--evidence-dir", str(args.output / (name + ".evidence"))]
                    prefix = "MULTINODE_MEGATRON_RDMA_JSON "
                print(f"start {name}", flush=True)
                log = args.output / (name + ".log")
                with log.open("w") as output:
                    output.write("COMMAND " + json.dumps(command) + "\n")
                    output.flush()
                    process = subprocess.run(command, cwd=ROOT, env=env, stdout=output,
                                             stderr=subprocess.STDOUT, timeout=960)
                if process.returncode:
                    raise RuntimeError(f"Flex failed: {log}")
                records = [json.loads(line[len(prefix):]) for line in log.read_text().splitlines()
                           if line.startswith(prefix)]
                if len(records) != 1 or not records[0].get("allocate_tensors"):
                    raise RuntimeError(f"Flex record/scope mismatch: {log}")
                save(path, records[0])
                run = records[0]["runs"][0]
                print(f"{name}: FWD={run['forward_e2e']['host_wall']['mean_ms']:.3f} ms "
                      f"BWD={run['backward_only']['host_wall']['mean_ms']:.3f} ms", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "build-pair", "weft", "flex"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--tuning-headers", type=Path,
                        help="build checked-in combined MODEL.hpp profiles from this directory")
    parser.add_argument("--forward-builds", type=Path)
    parser.add_argument("--backward-builds", type=Path)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=JSON",
                        help="override one build parameter; repeat for several parameters")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--eps", nargs="+", type=int, choices=(8, 16), default=[8, 16])
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 65536])
    parser.add_argument("--phases", nargs="+", choices=("forward", "backward"), default=["forward", "backward"])
    parser.add_argument("--remote", default="root@10.10.19.233")
    parser.add_argument("--port", type=int, default=2222)
    parser.add_argument("--socket-ifname", default="bond0")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--flex-sms", type=int)
    parser.add_argument("--label", default="screen")
    parser.add_argument("--check-repeats", action="store_true")
    parser.add_argument("--repartition", action="store_true",
                        help="split each EP8 source shard into two EP16 shards (Forward validation)")
    parser.add_argument("--dump-output", type=Path)
    parser.add_argument("--dump-tokens", type=int, default=8192)
    parser.add_argument("--validation-repeats", type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    if args.iterations <= 0 or any(t <= 0 for t in args.tokens):
        parser.error("iterations and tokens must be positive")
    args.output = args.output.resolve()
    args.source = args.source.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    {"build": build, "build-pair": build_pair, "weft": weft, "flex": flex}[args.action](args)


if __name__ == "__main__":
    main()
