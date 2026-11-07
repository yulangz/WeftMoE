#!/usr/bin/env python3
"""Offline PF training/inference and DT tuning on reproducible route traces.

One config describes workload, topology, search and measurement. CLI options
override it. Multi-node runs use SSH and a shared repository/cache path.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import sysconfig
import time

import backward_tuning as backward
import forward_tuning as forward
from autotune_common import (PlanError, candidate_digest, choose_confirmed_candidate,
                             compile_in_parallel, confirmation_finalists, git_revision,
                             require, write_result)
import pf_tuning as ring
import unified_tuning


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.models import add_model_arguments, resolve_model  # noqa: E402
from weftmoe.jit import default_header, source_digest
from weftmoe.tuning import arithmetic, default_table, digest, publish_entry

# Conservative whitelist: new search axes are excluded until reviewed. Keep
# arithmetic tiles, WGMMA atoms, quantization and reduction policy canonical.
INVARIANT_FORWARD = frozenset({
    "fc1_stages", "fc2_stages", "static_v4_fc1_stages", "static_v4_fc2_stages",
    "fc2_claim_tiles", "task_idle_nanosleep", "transport_idle_nanosleep",
    "setup_nanosleep", "return_idle_nanosleep", "reduce_queue_nanosleep", "finalize_nanosleep",
})
INVARIANT_BACKWARD = frozenset({
    "dab_stages", "dx_stages", "dw1_stages", "dw2_stages", "dw1_claim_tiles", "dw2_claim_tiles",
})
INVARIANT_V4 = frozenset({"window_tokens", "contribution_windows", "decode_stages", "decode_cta_multiplier", "dt_fp8_stages", "dt_fp8_ctas"})


def invariant_changes(config, reference):
    allowed = dict(forward=INVARIANT_FORWARD, backward=INVARIANT_BACKWARD, v4=INVARIANT_V4)
    return [f"{phase}.{name}" for phase in allowed for name in config[phase]
            if name not in allowed[phase] and config[phase][name] != reference[phase][name]]


def length_bucket(tokens, boundaries):
    require(boundaries and all(type(n) is int and n > 0 for n in boundaries)
            and sorted(set(boundaries)) == boundaries, "token_buckets must be increasing positive integers")
    lower = 0
    for upper in boundaries:
        if tokens <= upper:
            return [lower, upper]
        lower = upper + 1
    upper = boundaries[-1]
    while tokens > upper:
        lower, upper = upper + 1, upper * 2
    return [lower, upper]


PHASES = {"forward": forward, "backward": backward}


def load_incumbent(spec, args):
    """Expand the frozen header using the same C++ defaults as native builds."""
    require(not (spec.get("batch_invariant", True) and spec.get("base_header")),
            "custom base_header requires --no-batch-invariant; invariant mode uses the canonical shape default")
    header = (args.config.parent / spec["base_header"]).resolve() if spec.get("base_header") else (
        ROOT / "configs/profiles" / args.model / "weftmoe_forward_tuning_generated.hpp")
    source = default_header(args.hidden, args.intermediate) if not spec.get("base_header") else header.read_text()
    source += '\n#include "cuda/forward/tuning_config.hpp"\n#include "cuda/backward/tuning_config.hpp"\n'
    for phase, module in PHASES.items():
        source += "".join(f"PF_VALUE {phase} {key} {macro}\n"
                          for key, macro in module.PARAMETER_MACROS.items())
    for key, macro in ring.MACROS.items():
        source += (f"#ifndef {macro}\n#define {macro} {ring.DEFAULTS[key]}\n#endif\n"
                   f"PF_VALUE v4 {key} {macro}\n")
    output = subprocess.check_output([
        spec.get("environment", {}).get("CXX", os.environ.get("CXX", "c++")),
        "-E", "-P", "-x", "c++", "-",
        "-I" + str(ROOT / "src"), "-I" + str(header.parent),
        ], input=source, text=True)
    config = {phase: {} for phase in (*PHASES, "v4")}
    for line in output.splitlines():
        if line.startswith("PF_VALUE "):
            _, phase, key, value = line.split()
            config[phase][key] = int(value)
    for key in forward.BOOLEAN_PARAMETERS:
        config["forward"][key] = bool(config["forward"][key])
    for key, values in forward.ENUM_PARAMETERS.items():
        config["forward"][key] = next(k for k, v in values.items() if v == config["forward"][key])
    for phase in PHASES:
        config[phase].update(main_hidden=args.hidden, main_intermediate=args.intermediate)
    # A new shape may have an odd N256 extent. Use the legal N128 seed;
    # the configured search still decides which tiles/stages to keep.
    for key, extent in (("dab_tile_n", args.intermediate), ("dx_tile_n", args.hidden)):
        if extent % (2 * config["backward"][key]):
            config["backward"][key] = 128
    topology = "local" if not args.remotes else "network"
    config["v4"] = ring.DEFAULTS | config["v4"] | spec["ring"][topology][0]
    reference = copy.deepcopy(config)
    require(set(spec.get("v4", {})) <= set(config["v4"]), "unknown v4 settings")
    config["v4"].update(spec.get("v4", {}))
    for phase, overrides in spec.get("initial", {}).items():
        require(phase in config and set(overrides) <= set(config[phase]),
                f"unknown initial tuning parameter: {phase}: {overrides}")
        config[phase].update(overrides)
    if spec.get("batch_invariant", True):
        changed = invariant_changes(config, reference)
        require(not changed, f"batch-invariant initial overrides change arithmetic/policy: {changed}")
        require(arithmetic(generated_header(config)) == arithmetic(default_header(args.hidden, args.intermediate)),
                "batch-invariant base header differs from canonical M/N/K arithmetic")
    spec["_invariant_reference"] = reference
    return config


def validate(config):
    for phase, module in PHASES.items():
        module.validate_tuning(config[phase])
    unified_tuning.validate_shared_parameters(config["forward"], config["backward"])
    ring.validate(dict(config["forward"], v4=config["v4"]))


def compiled_config(config):
    return dict(config, v4={key: config["v4"][key] for key in ring.MACROS})


def generated_header(config):
    validate(config)
    # Both phases must participate in the ID checked by TensorRuntime.
    digest = candidate_digest(compiled_config(config))
    macros = {}
    for phase, module in PHASES.items():
        for key, macro in module.PARAMETER_MACROS.items():
            value = config[phase][key]
            if phase == "forward":
                value = forward.ENUM_PARAMETERS.get(key, {}).get(value, value)
            macros[macro] = int(value)
    macros.update({macro: config["v4"][key] for key, macro in ring.MACROS.items()})
    return ("#pragma once\n\n// Generated by tools/autotune.py.\n"
            + "".join(f"#define {macro} {value}\n" for macro, value in macros.items())
            + f'#define WEFTMOE_FORWARD_TUNING_ID "{digest}"\n')


def candidates_for(stage, incumbent, spec, args):
    if stage == "ring":
        variants = [{"v4": item} for item in spec["ring"]["network" if args.remotes else "local"]]
    else:
        search = spec[stage]
        axes = search.get("axes", {})
        variants = []
        for variant, values in itertools.product(search.get("variants", [{}]),
                                                 itertools.product(*axes.values())):
            update = copy.deepcopy(search.get("fixed", {}))
            for dotted, value in list(variant.items()) + list(zip(axes, values)):
                phase, key = dotted.split(".")
                update.setdefault(phase, {})[key] = value
            variants.append(update)
    candidates, rejected, seen = [], [], set()
    for update in [{}] + variants:
        config = copy.deepcopy(incumbent["config"])
        for phase, overrides in update.items():
            require(phase in config and set(overrides) <= set(config[phase]),
                    f"unknown tuning parameter: {phase}: {overrides}")
            config[phase].update(overrides)
        name = candidate_digest(config)
        if name in seen:
            continue
        seen.add(name)
        try:
            if spec.get("batch_invariant", True):
                changed = invariant_changes(config, spec.get("_invariant_reference", incumbent["config"]))
                require(not changed, f"batch-invariant excludes {changed}")
            validate(config)
        except PlanError as error:
            rejected.append(dict(config=config, reason=str(error)))
            continue
        candidates.append(dict(name=name, config=config))
    return candidates, rejected


class NativeCompiler:
    def __init__(self, args, environment=None):
        self.jobs, self.cache = args.jobs, args.cache.resolve()
        self.environment = os.environ | (environment or {})

    def directory(self, candidate):
        return self.cache / candidate_digest(compiled_config(candidate["config"]))

    def compile(self, candidate, _compiler, log_dir, *, command_jobs):
        started = time.monotonic()
        directory = self.directory(candidate)
        directory.mkdir(parents=True, exist_ok=True)
        header = directory / "weftmoe_forward_tuning_generated.hpp"
        content = generated_header(candidate["config"])
        if not header.exists() or header.read_text() != content:
            header.write_text(content)
        library = directory / ("libweftmoe_" + candidate_digest(compiled_config(candidate["config"])) + ".so")
        log_path = log_dir / (candidate["name"] + ".build.log")
        env = self.environment | {"WEFTMOE_NATIVE_SONAME": library.name, "NVCCFLAGS": self.environment.get("NVCCFLAGS", "")
                           + " -lineinfo -Xptxas=-v,--warn-on-spills "
                           "-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 "
                           + shlex.quote("-I" + str(directory))}
        commands = [[sys.executable, "build_native.py", "lib", "--output", str(library),
                     "--jobs", str(command_jobs)],
                    [sys.executable, "tools/build_tensor_runtime.py", "--library", str(library)]]
        with log_path.open("w") as log:
            for command in commands:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    break
        return dict(returncode=result.returncode, elapsed_seconds=time.monotonic() - started,
                    log=str(log_path), library=str(library), header=str(header))


class Benchmark:
    def __init__(self, args, spec, output, compiler):
        self.args, self.spec, self.output, self.compiler = args, spec, output, compiler
        self.fingerprints = {}

    def invariance(self, candidate, label):
        args, spec = self.args, self.spec
        # Include the actual bucket endpoints and low-M boundaries, not just
        # the timing representatives. Anchor values/routes remain identical.
        lower, upper = args.token_interval
        sizes = sorted(set([1, 17, 63, 64, 65, 127, 128, 129, max(1, lower), upper]))
        extension = lambda c: self.compiler.directory(c) / ("_runtime" + sysconfig.get_config_var("EXT_SUFFIX"))
        cli = ["--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
               "--experts", str(args.experts), "--top-k", str(args.top_k),
               "--tokens", *map(str, sizes), "--candidate", str(extension(candidate)),
               "--reference", str(extension(self.reference))]
        for role, item in (("candidate", candidate), ("reference", self.reference)):
            cli += [f"--{role}-ring-tokens", str(item["config"]["v4"]["window_tokens"]),
                    f"--{role}-ring-depth", str(item["config"]["v4"]["contribution_windows"])]
        if args.remotes:
            cli.append("--skip-dt")
        paths, processes, streams = [], [], []
        env = {k:v for k,v in os.environ.items() if k.startswith(("NCCL_", "UCX_", "CUDA_")) or k == "LD_LIBRARY_PATH"}
        env.update(spec.get("environment", {}))
        env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
        try:
            for node, remote in enumerate([None] + args.remotes):
                command = ["env", "-u", "WEFTMOE_AUTOTUNE_TABLE", *(f"{k}={v}" for k,v in env.items()),
                           "timeout", "--kill-after=20s", str(spec["timeout_seconds"]),
                           sys.executable, "-m", "torch.distributed.run",
                           f"--nnodes={1+len(args.remotes)}", f"--nproc-per-node={args.gpus_per_node}",
                           f"--node-rank={node}", f"--master-addr={args.master_addr}",
                           f"--master-port={args.master_port}",
                           str(ROOT / "tests/torchrun_batch_invariant.py"), *cli]
                if remote:
                    command = ["ssh", "-p", str(args.ssh_port), "-o", "BatchMode=yes", remote,
                               "cd " + shlex.quote(str(ROOT)) + " && " + shlex.join(command)]
                path = self.output / f"{label}.node{node}.log"
                paths.append(path)
                streams.append(path.open("w"))
                processes.append(subprocess.Popen(command, cwd=ROOT, stdout=streams[-1], stderr=subprocess.STDOUT))
            codes = [process.wait() for process in processes]
        finally:
            for stream in streams:
                stream.close()
        require(not any(codes), f"batch-invariant verification failed ({codes}); see {paths}")
        return dict(passed=True, logs=list(map(str, paths)), tokens=sizes)

    def run(self, candidate, label, *, verify=False, final=False):
        args, spec = self.args, self.spec
        training = args.mode == "training"
        settings = spec["final" if final else "screen"]
        tokens = args.tokens if final or not verify else spec["verify_tokens"]
        cli = ["--backend", "dt" if args.mode == "decode" else "pf", "--tokens", *map(str, tokens),
               "--hidden", str(args.hidden), "--intermediate", str(args.intermediate),
               "--experts", str(args.experts), "--top-k", str(args.top_k),
               "--warmups", "0" if verify else str(settings["warmups"]),
               "--samples", str(spec["verify_samples"] if verify else settings["samples"]),
               "--seed", str(spec["seed"]),
               "--token-std-fraction", str(spec["token_std_fraction"]),
               "--route-bias-std", str(spec["route_bias_std"]),
               "--ring-tokens", str(candidate["config"]["v4"]["window_tokens"]),
               "--ring-depth", str(candidate["config"]["v4"]["contribution_windows"])]
        if args.fp8:
            cli += ["--fp8"] + (["--weight-grad-dtype", "bf16"] if training else [])
        if verify:
            cli.append("--verify")
            if not training:
                cli.append("--verify-kernel")
        elif training:
            cli.append("--prime-trace")
        if not training:
            cli.append("--prepared-weights")
            if not verify and spec.get("graph_replays", 0):
                cli += ["--graph-replays", str(spec["graph_replays"])]
        if training and spec.get("stream_ordered", False):
            cli.append("--stream-ordered")
        if training and not verify and spec.get("sample_pause_ms", 0):
            cli += ["--sample-pause-ms", str(spec["sample_pause_ms"])]
        extension = self.compiler.directory(candidate) / ("_runtime" + sysconfig.get_config_var("EXT_SUFFIX"))
        env = {k: v for k, v in os.environ.items()
               if k.startswith(("NCCL_", "UCX_", "CUDA_")) or k == "LD_LIBRARY_PATH"}
        env.update(PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", WEFTMOE_TENSOR_EXTENSION=str(extension),
                   NCCL_DEBUG_FILE=str(self.output / f"{label}.nccl.%h.%p.log"))
        env.update(spec.get("environment", {}))
        processes, logs, paths = [], [], []
        start = time.monotonic()
        print(f"START {label}", flush=True)
        for node, remote in enumerate([None] + args.remotes):
            command = ["env", "-u", "WEFTMOE_AUTOTUNE_TABLE", *(f"{k}={v}" for k, v in env.items()),
                       "timeout", "--kill-after=20s", str(spec["timeout_seconds"]),
                       sys.executable, "-m", "torch.distributed.run",
                       f"--nnodes={1 + len(args.remotes)}", f"--nproc-per-node={args.gpus_per_node}",
                       f"--node-rank={node}", f"--master-addr={args.master_addr}",
                       f"--master-port={args.master_port}",
                       str(ROOT / "benchmarks" / (
                           "torchrun_training.py" if training else "torchrun_inference.py")), *cli]
            if remote:
                command = ["ssh", "-p", str(args.ssh_port), "-o", "BatchMode=yes", remote,
                           "cd " + shlex.quote(str(ROOT)) + " && " + shlex.join(command)]
            path = self.output / f"{label}.node{node}.log"
            paths.append(path)
            logs.append(path.open("w"))
            processes.append(subprocess.Popen(command, cwd=ROOT, stdout=logs[-1], stderr=subprocess.STDOUT))
        codes = [process.wait() for process in processes]
        for log in logs:
            log.close()
        prefix = (("TRAINING_PARITY_JSON " if verify else "TRAINING_CYCLE_JSON ")
                  if training else "INFERENCE_BENCHMARK_JSON ")
        records = [json.loads(line[len(prefix):]) for line in paths[0].read_text().splitlines()
                   if line.startswith(prefix)]
        require(not any(codes), f"{label} failed ({codes}); see {paths}")
        require(len(records) == len(tokens) * (spec["verify_samples"] if verify and training else 1),
                f"incomplete result: {paths[0]}")
        result = dict(logs=list(map(str, paths)), elapsed_seconds=time.monotonic() - start,
                      records=records)
        if not verify:
            for record in records:
                key = (final, record["nominal_tokens_per_rank"])
                fingerprint = record["workload_sha256_per_rank"]
                require(self.fingerprints.setdefault(key, fingerprint) == fingerprint,
                        "candidates used different workloads")
                workspace = max(sum(p[k] for k in ("compute_bytes", "gin_bytes", "notify_bytes"))
                                for p in record["per_rank_workspace"])
                require(workspace <= spec["workspace_limit_bytes"], "candidate exceeds workspace limit")
            metric = ("device_slowest_rank_p50_us" if not training and spec.get("graph_replays", 0)
                      else "wall_slowest_rank_p50_us")
            result["metric_us"] = statistics.geometric_mean(r[metric] for r in records)
            print(f"PASS {label}: " + ", ".join(
                f"T{r['nominal_tokens_per_rank']}={r[metric] / 1000:.3f}ms"
                for r in records), flush=True)
        else:
            print(f"PASS {label}: numerical parity", flush=True)
        return result


def export_table(path, selected, args, compiler):
    record = selected["final"]["records"][0]
    header = generated_header(selected["config"])
    entry = dict(key=dict(gpu=record["gpu"], interconnect="nvlink+gin" if args.remotes else "nvlink",
                         dtype="fp8" if args.fp8 else "bfloat16", hidden=args.hidden,
                         intermediate=args.intermediate, experts=args.experts, top_k=args.top_k,
                         nodes=1 + len(args.remotes), gpus_per_node=args.gpus_per_node,
                         input_layout="dp", mode=args.mode),
                 tokens=args.token_interval, source=source_digest(), header=header,
                 header_digest=digest(header), batch_invariant=args.batch_invariant, validated=True,
                 window_tokens=selected["config"]["v4"]["window_tokens"],
                 contribution_windows=selected["config"]["v4"]["contribution_windows"])
    write_result(path, dict(schema=2, entries=[entry]))
    publish_entry(args.table, entry)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/autotune.json")
    parser.add_argument("--mode", choices=("training", "inference", "decode"), default="training")
    config_args, _ = parser.parse_known_args(argv)
    spec = json.loads(config_args.config.read_text())
    parser.set_defaults(mode=spec.get("workload", {}).get("mode", "training"))
    mode = parser.parse_known_args(argv)[0].mode
    overrides = spec.get("modes", {}).get(mode, {})
    spec = spec | overrides | {"workload": spec.get("workload", {}) | overrides.get("workload", {})}
    add_model_arguments(parser, default=spec.get("workload", {}).get("model", "qwen35b"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 20480])
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--remotes", nargs="*", default=[])
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29531)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=ROOT / "build/autotune/cache")
    parser.add_argument("--output", type=Path, default=ROOT / "build/autotune")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-invariant", action=argparse.BooleanOptionalAction,
                        default=spec.get("batch_invariant", True))
    parser.add_argument("--fp8", action="store_true", help="FP8 training (dW BF16)")
    parser.add_argument("--table", type=Path, default=default_table())
    parser.add_argument("--token-buckets", type=int, nargs="+", default=spec.get("token_buckets", [8192, 20480, 65536]))
    for section in ("workload", "system", "run"):
        values = dict(spec.get(section, {}))
        known = {action.dest for action in parser._actions}
        require(set(values) <= known, f"unknown {section} settings: {set(values) - known}")
        for name in ("cache", "output"):
            if name in values:
                path = Path(values[name]).expanduser()
                values[name] = path if path.is_absolute() else config_args.config.resolve().parent / path
        parser.set_defaults(**values)
    args = parser.parse_args(argv)
    spec["batch_invariant"] = args.batch_invariant
    if args.fp8 and args.mode == "decode":
        spec["decode"] = {"axes": {"v4.dt_fp8_stages": [1, 2, 3], "v4.dt_fp8_ctas": [0, 32, 64, 96, 132]}}
    args.config = args.config.resolve()
    resolve_model(args)
    fields = vars(args) | dict(nodes=1 + len(args.remotes),
                               world=(1 + len(args.remotes)) * args.gpus_per_node)
    for name in ("cache", "output"):
        path = Path(str(getattr(args, name)).format_map(fields)).expanduser()
        setattr(args, name, path.resolve())
    return args, spec


def run_search(args, spec):
    require(not args.remotes or args.master_addr != "127.0.0.1", "multi-node requires --master-addr")
    require(args.mode != "decode" or not args.remotes, "DT uses node-local communication")
    require(args.jobs > 0 and args.gpus_per_node > 0 and min(args.tokens) > 0,
            "jobs, GPUs and tokens must be positive")
    require(args.experts % ((1 + len(args.remotes)) * args.gpus_per_node) == 0,
            "experts must divide EP world size")
    require(spec["confirmation_repeats"] > 0 and spec["confirmation_candidates"] > 0,
            "confirmation counts must be positive")
    require(spec["verify_samples"] > 0 and min(spec["verify_tokens"]) > 0,
            "verification needs positive sample and token counts")
    config = load_incumbent(spec, args)
    validate(config)
    incumbent = dict(name=candidate_digest(config), config=config)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = NativeCompiler(args, spec.get("environment"))
    benchmark = Benchmark(args, spec, output, compiler)
    result = dict(schema=1, spec=spec, arguments={k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items()},
                  source=git_revision(ROOT), stages=[])
    if args.dry_run:
        for stage in spec["stages"]:
            candidates, rejected = candidates_for(stage, incumbent, spec, args)
            result["stages"].append(dict(stage=stage, candidates=candidates, rejected=rejected))
        write_result(output / "plan.json", result)
        print(output / "plan.json")
        return
    builds = {}
    reference = copy.deepcopy(incumbent)
    reference["build"] = compiler.compile(reference, compiler, output, command_jobs=args.jobs)
    require(reference["build"]["returncode"] == 0, "canonical reference compilation failed")
    benchmark.reference = reference
    builds[candidate_digest(compiled_config(reference["config"]))] = reference["build"]
    for stage_index, stage in enumerate(spec["stages"]):
        candidates, rejected = candidates_for(stage, incumbent, spec, args)
        stage_result = dict(stage=stage, candidates=candidates, rejected=rejected)
        result["stages"].append(stage_result)
        unique = {candidate_digest(compiled_config(c["config"])): c for c in candidates}
        pending = [c for digest, c in unique.items() if digest not in builds]
        if pending:
            compiled = compile_in_parallel(pending, compiler, output, compiler.compile)
            for candidate in pending:
                builds[candidate_digest(compiled_config(candidate["config"]))] = compiled[candidate["name"]]
        valid = []
        for candidate in candidates:
            candidate["build"] = builds[candidate_digest(compiled_config(candidate["config"]))]
            label = f"s{stage_index}_{stage}_{candidate['name']}"
            if candidate["build"]["returncode"] == 0:
                try:
                    candidate["screen"] = benchmark.run(candidate, label)
                    candidate["selection_metric_us"] = candidate["screen"]["metric_us"]
                    valid.append(candidate)
                except PlanError as error:
                    candidate["error"] = str(error)
            write_result(output / "result.json", result)
        require(valid, f"no candidate passed {stage}; see {output / 'result.json'}")
        current = next((c for c in valid if c["name"] == incumbent["name"]), None)
        if current is not None and spec.get("confirm_screen_improvements_only", False):
            best = min(valid, key=lambda c: c["selection_metric_us"])
            improvement = 100 * (1 - best["selection_metric_us"] / current["selection_metric_us"])
            if best is current or improvement < spec["minimum_improvement_percent"]:
                incumbent = current
                stage_result["decision"] = dict(
                    incumbent=current["name"], selected=current["name"], best_screened=best["name"],
                    reason="no_screened_improvement", screen_improvement_percent=improvement,
                    minimum_improvement_percent=spec["minimum_improvement_percent"])
                print(f"SELECT {stage}: {stage_result['decision']}", flush=True)
                write_result(output / "result.json", result)
                continue
        finalists = confirmation_finalists(valid, incumbent["name"], spec["confirmation_candidates"])
        for repeat in range(spec["confirmation_repeats"]):
            for candidate in (finalists if repeat % 2 == 0 else reversed(finalists)):
                if "error" in candidate:
                    continue
                label = f"s{stage_index}_confirm{repeat}_{candidate['name']}"
                try:
                    candidate.setdefault("confirmations", []).append(benchmark.run(candidate, label))
                except PlanError as error:
                    candidate["error"] = str(error)
                write_result(output / "result.json", result)
        for candidate in finalists:
            if "error" in candidate:
                continue
            candidate["aggregate_metric_us"] = statistics.mean(r["metric_us"] for r in candidate["confirmations"])
            try:
                candidate["parity"] = benchmark.run(candidate, f"s{stage_index}_parity_{candidate['name']}", verify=True)
                if args.batch_invariant:
                    candidate["invariance"] = benchmark.invariance(candidate, f"s{stage_index}_invariant_{candidate['name']}")
            except PlanError as error:
                candidate["error"] = str(error)
                del candidate["aggregate_metric_us"]
        incumbent, stage_result["decision"] = choose_confirmed_candidate(
            finalists, incumbent["name"], spec["minimum_improvement_percent"])
        print(f"SELECT {stage}: {stage_result['decision']}", flush=True)
        write_result(output / "result.json", result)
    incumbent["final_parity"] = benchmark.run(incumbent, "final_parity", verify=True, final=True)
    if args.batch_invariant:
        incumbent["batch_invariance"] = benchmark.invariance(incumbent, "final_invariance")
    incumbent["final"] = benchmark.run(incumbent, "final", final=True)
    result["selected"] = incumbent
    write_result(output / "result.json", result)
    export_table(output / "table.json", incumbent, args, compiler)
    write_result(output / "runtime.json", dict(environment={"WEFTMOE_AUTOTUNE_TABLE": str(args.table.resolve())},
                                              tokens=args.token_interval, batch_invariant=args.batch_invariant))
    print(f"Selected library: {incumbent['build']['library']}", flush=True)
    print(f"Validated {args.mode} table: {output / 'table.json'}", flush=True)


def main(argv=None):
    args, spec = parse_args(argv)
    groups = {}
    for tokens in args.tokens:
        interval = tuple(length_bucket(tokens, args.token_buckets))
        groups.setdefault(interval, []).append(tokens)
    for interval, tokens in groups.items():
        bucket_args = copy.deepcopy(args)
        bucket_args.tokens, bucket_args.token_interval = tokens, list(interval)
        bucket_args.output = args.output / f"tokens_{interval[0]}_{interval[1]}"
        run_search(bucket_args, copy.deepcopy(spec))


if __name__ == "__main__":
    main()
