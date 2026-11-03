#!/usr/bin/env python3
"""Unattended, resumable single-node EP8 DT FP8 Cartesian search.

Launch: python tools/autotune_dt_fp8.py --output build/dt-autotune --detach
Resume: repeat the command (source snapshot and successful stages are reused).
Stop: kill -TERM $(cat build/dt-autotune/pid)
Only DT FP8 compile-time overrides are changed; BF16 and 224/56 registers stay fixed.
No selected configuration is automatically installed into production headers.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.fp8_validation import POLICY, passes
MODELS = {"qwen35b": (2048, 512, 8), "qwen122b": (3072, 1024, 8), "flash": (4096, 2048, 6)}
SIZES = [1, 16, 32, 64, 128, 256, 512]
METRICS = ["fp8_online_us", "fp8_prequant_us"]
KERNEL_POLICY = "fixed-m64-v1"


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def candidates():
    result = [dict(id="default", stages=3, slices=1, ctas=0)]
    # M64 and the 224/56 register partition are fixed in every configuration.
    for stages, slices, ctas in itertools.product([1, 2, 3], [0, 1], [32, 64, 96, 132]):
        result.append(dict(id=f"m64-s{stages}-r{slices}-c{ctas}",
                           stages=stages, slices=slices, ctas=ctas))
    tail = result[1:]
    random.Random(20261031).shuffle(tail)
    return result[:1] + tail


def snapshot(output):
    dest = output / "source"
    if dest.exists():
        raise RuntimeError("incomplete setup: source exists without manifest; use a new output directory")
    dest.mkdir()
    for name in ("src", "include", "configs", "tools", "tests", "benchmarks", "weftmoe"):
        shutil.copytree(ROOT / name, dest / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.so", "*.pyc", ".pytest_cache"))
    for name in ("build_native.py", "setup.py", "pyproject.toml", "MANIFEST.in"):
        shutil.copy2(ROOT / name, dest / name)
    hashes = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(dest.rglob("*")) if p.is_file()}
    return hashes


def reuse_builds(output, previous, hashes):
    """Reuse immutable binaries only; never reuse old correctness verdicts."""
    previous = previous.resolve()
    old = json.loads((previous / "manifest.json").read_text())
    def native_inputs(values):
        return {k: v for k, v in values.items() if k.startswith(("src/", "include/", "configs/"))
                or k in ("build_native.py", "tools/build_tensor_runtime.py")}
    if native_inputs(old["source_hashes"]) != native_inputs(hashes):
        raise RuntimeError("cannot reuse binaries: native source/configuration hashes differ")
    count = 0
    for candidate in candidates():
        for model in MODELS:
            relative = Path("jobs") / model / candidate["id"] / "native"
            source = previous / relative
            marker = source / "complete.json"
            if not marker.exists():
                continue
            if json.loads(marker.read_text())["candidate"] != candidate:
                raise RuntimeError(f"candidate metadata mismatch: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source, target_is_directory=True)
            count += 1
    return dict(directory=str(previous), reused_builds=count)


def stop_group(process):
    # Every command starts a private session, including its torchrun workers.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_command(command, *, cwd, env, log, timeout):
    start = time.time()
    with log.open("w") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"exit {code}: {log}")
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(f"timeout after {timeout}s: {log}") from error
        finally:
            stop_group(process)
    return time.time() - start


def idle_gpus():
    # Never kill another job or benchmark alongside an unrelated GPU process.
    for attempt in range(31):
        result = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
        if not result.strip():
            return
        if attempt == 30:
            raise RuntimeError("GPU processes remain active; sweep stopped to avoid contaminated timings")
        time.sleep(2)


def parse_measurements(log):
    rows = [json.loads(line[7:]) for line in log.read_text().splitlines() if line.startswith("RESULT ")]
    if sorted(r["tokens"] for r in rows) != SIZES:
        raise RuntimeError(f"missing or duplicate benchmark rows: {log}")
    for row in rows:
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in ["bf16_us", *METRICS]):
            raise RuntimeError(f"invalid timing: {log}")
        if not passes(row.get("fp8_vs_bf16_max_rank_diff", float("nan"))):
            raise RuntimeError(f"benchmark numerical sanity failure: {log}")
    return rows


def model_records(output, model, phase):
    records = []
    for path in sorted((output / "jobs").glob(f"{model}/*/{phase}/result.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "passed":
            records.append(record)
    return records


def geo(values):
    return math.exp(statistics.mean(math.log(v) for v in values))


def summarize(output):
    summary = {"updated": time.time(), "production_defaults_changed": False, "models": {}}
    manifest_path = output / "manifest.json"
    summary["validation"] = POLICY
    summary["allow_known_one_token"] = (json.loads(manifest_path.read_text()).get("allow_known_one_token", False)
                                        if manifest_path.exists() else False)
    all_records = [json.loads(p.read_text()) for p in (output / "jobs").glob("*/*/*/result.json")]
    summary["job_counts"] = {status: sum(r.get("status") == status for r in all_records)
                             for status in ("passed", "failed", "running")}
    summary["failures"] = [{k: r[k] for k in ("model", "candidate", "phase", "error")}
                           for r in all_records if r.get("status") == "failed"]
    csv_rows = []
    for model in MODELS:
        phases = {}
        for phase in ("search", "confirm-1300", "confirm-1700"):
            records = model_records(output, model, phase)
            if not records:
                continue
            baseline = next((r for r in records if r["candidate"]["id"] == "default"), None)
            base = {r["tokens"]: r for r in baseline["rows"]} if baseline else None
            best = []
            overall = []
            for rec in records:
                ratios, bf16_ratios = [], []
                for row in rec["rows"]:
                    for metric in METRICS:
                        ratio = base[row["tokens"]][metric] / row[metric] if base else None
                        if ratio is not None:
                            ratios.append(ratio)
                        bf16_ratios.append(row["bf16_us"] / row[metric])
                        csv_rows.append(dict(model=model, phase=phase, candidate=rec["candidate"]["id"],
                            tokens=row["tokens"], input=metric, us=row[metric], bf16_us=row["bf16_us"],
                            speedup_vs_bf16=row["bf16_us"] / row[metric], speedup_vs_default_fp8=ratio))
                overall.append(dict(candidate=rec["candidate"],
                    geometric_speedup_vs_default_fp8=geo(ratios) if ratios else None,
                    geometric_speedup_vs_bf16=geo(bf16_ratios)))
            for n, metric in itertools.product(SIZES, METRICS):
                ranked = sorted(((next(row for row in rec["rows"] if row["tokens"] == n)[metric], rec)
                                 for rec in records), key=lambda pair: pair[0])
                us, rec = ranked[0]
                row = next(row for row in rec["rows"] if row["tokens"] == n)
                best.append(dict(tokens=n, input=metric, candidate=rec["candidate"], us=us,
                                 speedup_vs_bf16=row["bf16_us"] / us,
                                 speedup_vs_default_fp8=base[n][metric] / us if base else None))
            phases[phase] = dict(passed_candidates=len(records), best_per_case=best,
                                default_eligible=baseline is not None,
                                fixed_config_ranking=sorted(overall, key=lambda v: -v[
                                    "geometric_speedup_vs_default_fp8" if base else "geometric_speedup_vs_bf16"]))
        # Confirmation winners must pass both fresh-seed rounds, avoiding
        # selecting a candidate from only its favorable round.
        rounds = [model_records(output, model, f"confirm-{seed}") for seed in (1300, 1700)]
        by_round = [{r["candidate"]["id"]: r for r in group} for group in rounds]
        common = set(by_round[0]) & set(by_round[1])
        if common:
            confirmed = []
            aggregate = {cid: [] for cid in common}
            bf16_aggregate = {cid: [] for cid in common}
            has_default = "default" in common
            for n, metric in itertools.product(SIZES, METRICS):
                options = []
                for cid in sorted(common):
                    rows = [next(r for r in rnd[cid]["rows"] if r["tokens"] == n) for rnd in by_round]
                    relative = None
                    if has_default:
                        bases = [next(r for r in rnd["default"]["rows"] if r["tokens"] == n) for rnd in by_round]
                        relative = geo([b[metric] / r[metric] for b, r in zip(bases, rows)])
                        aggregate[cid].append(relative)
                    bf16_ratio = geo([r["bf16_us"] / r[metric] for r in rows])
                    bf16_aggregate[cid].append(bf16_ratio)
                    options.append(dict(candidate=by_round[0][cid]["candidate"],
                        geometric_us=geo([r[metric] for r in rows]),
                        speedup_vs_default_fp8=relative,
                        speedup_vs_bf16=bf16_ratio))
                confirmed.append(dict(tokens=n, input=metric,
                                      **max(options, key=lambda r: r[
                                          "speedup_vs_default_fp8" if has_default else "speedup_vs_bf16"])))
            phases["confirmed"] = dict(best_per_case=confirmed, default_eligible=has_default,
                fixed_config_ranking=sorted([dict(candidate=by_round[0][cid]["candidate"],
                     geometric_speedup_vs_default_fp8=geo(values) if values else None,
                     geometric_speedup_vs_bf16=geo(bf16_aggregate[cid])) for cid, values in aggregate.items()],
                     key=lambda r: -r["geometric_speedup_vs_default_fp8" if has_default else "geometric_speedup_vs_bf16"]))
        summary["models"][model] = phases
    atomic_json(output / "summary.json", summary)
    if csv_rows:
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        temporary.replace(output / "results.csv")
    return summary


def execute_job(output, manifest, model, candidate, phase, retry_failed):
    source = output / "source"
    base = output / "jobs" / model / candidate["id"]
    job = base / phase
    job.mkdir(parents=True, exist_ok=True)
    result_path = job / "result.json"
    rec = json.loads(result_path.read_text()) if result_path.exists() else {}
    if rec.get("status") == "passed" or (rec.get("status") == "failed" and not retry_failed):
        return rec
    rec = dict(model=model, candidate=candidate, phase=phase, status="running", started=time.time())
    atomic_json(result_path, rec)
    atomic_json(output / "status.json", dict(rec, pid=os.getpid()))
    print(f"START {model} {candidate['id']} {phase}", flush=True)
    env = dict(manifest["environment"])
    env.update(PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1")
    native = base / "native"
    native.mkdir(exist_ok=True)
    flags = {"STAGES": candidate["stages"],
             "SLICE_SCHEDULE": candidate["slices"], "CTAS": candidate["ctas"]}
    env["NVCCFLAGS"] = shlex.join(["-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1",
        "-I" + str(source / "configs/profiles" / model), "-Xptxas=-v",
        *[f"-DWEFTMOE_DT_FP8_{key}={value}" for key, value in flags.items()]])
    python = manifest["python"]
    try:
        if not (native / "complete.json").exists():
            run_command([python, "build_native.py", "lib", "--output", str(native / "libweftmoe.so"), "--jobs", "2"],
                        cwd=source, env=env, log=base / "build.log", timeout=1800)
            run_command([python, "tools/build_tensor_runtime.py", "--library", str(native / "libweftmoe.so")],
                        cwd=source, env=env, log=base / "adapter.log", timeout=600)
            atomic_json(native / "complete.json", dict(flags=env["NVCCFLAGS"], candidate=candidate))
        h, i, topk = MODELS[model]
        env.update(CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7", NCCL_DEBUG="WARN", OMP_NUM_THREADS="1",
            LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64",
            TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i), TEST_SIZES="1 16 65 129",
            WEFTMOE_TENSOR_EXTENSION=str(next(native.glob("_runtime*.so"))),
            BENCH_SIZES=" ".join(map(str, SIZES)), BENCH_TOPK=str(topk), BENCH_EXPERTS="256",
            BENCH_REPEATS="50" if phase == "search" else "100", BENCH_SWEEPS="7" if phase == "search" else "11",
            BENCH_SEED="900" if phase == "search" else phase.split("-")[1])
        if manifest.get("allow_known_one_token", False):
            env["TEST_DT_ALLOW_KNOWN_ONE_TOKEN"] = "1"
        launcher = [python, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8"]
        idle_gpus()
        if not (base / "correctness.json").exists():
            seconds = run_command([*launcher, "tests/torchrun_dt_fp8.py"], cwd=source, env=env,
                                  log=base / "correctness.log", timeout=420)
            log = base / "correctness.log"
            waived = log.read_text().count('"known_failure_waived": true') if log.exists() else 0
            atomic_json(base / "correctness.json", dict(passed=True, seconds=seconds, known_case_waivers=waived))
        idle_gpus()
        rec["benchmark_seconds"] = run_command([*launcher, "benchmarks/benchmark_dt_fp8.py"],
            cwd=source, env=env, log=job / "benchmark.log", timeout=600)
        rec.update(rows=parse_measurements(job / "benchmark.log"), status="passed")
    except RuntimeError as error:
        rec.update(status="failed", error=str(error))
    rec["finished"] = time.time()
    atomic_json(result_path, rec)
    print(f"{rec['status'].upper()} {model} {candidate['id']} {phase} {rec.get('error', '')}", flush=True)
    # A timeout must not leave live workers contaminating later jobs.
    idle_gpus()
    return rec


def worker(args):
    output = args.output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("kernel_policy") != KERNEL_POLICY:
        raise RuntimeError("DT kernel policy changed; use a new output directory")
    if manifest.get("validation") != POLICY:
        raise RuntimeError("validation policy changed; use a new output directory with --reuse-builds")
    # A repository-wide advisory lock prevents two instances of this driver
    # from sharing GPUs, even if their output directories differ.
    lock_path = Path(manifest["repository"]) / "build/dt-fp8-autotune.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / "pid").write_text(str(os.getpid()) + "\n")
        try:
            for rel, expected in manifest["source_hashes"].items():
                if hashlib.sha256((output / "source" / rel).read_bytes()).hexdigest() != expected:
                    raise RuntimeError(f"source snapshot modified: {rel}")
            idle_gpus()
            for candidate in manifest["candidates"]:
                for model in MODELS:
                    rec = execute_job(output, manifest, model, candidate, "search", args.retry_failed)
                    summarize(output)
            # Top two for each shape/input plus the best fixed config. All
            # finalists replay every shape on two independent routing seeds.
            for model in MODELS:
                records = model_records(output, model, "search")
                if not records:
                    atomic_json(output / f"finalists-{model}.json", [])
                    print(f"NO ELIGIBLE CANDIDATES {model}; inspect correctness logs", flush=True)
                    continue
                ids = {"default"} if any(r["candidate"]["id"] == "default" for r in records) else set()
                for n, metric in itertools.product(SIZES, METRICS):
                    ranked = sorted(records, key=lambda rec: next(row[metric] for row in rec["rows"] if row["tokens"] == n))
                    ids.update(rec["candidate"]["id"] for rec in ranked[:2])
                report = summarize(output)
                ids.add(report["models"][model]["search"]["fixed_config_ranking"][0]["candidate"]["id"])
                atomic_json(output / f"finalists-{model}.json", sorted(ids))
                for seed in (1300, 1700):
                    ordered = sorted(ids - {"default"})
                    random.Random(seed).shuffle(ordered)
                    for cid in (["default"] if "default" in ids else []) + ordered:
                        candidate = next(c for c in manifest["candidates"] if c["id"] == cid)
                        execute_job(output, manifest, model, candidate, f"confirm-{seed}", args.retry_failed)
                        summarize(output)
            atomic_json(output / "status.json", dict(status="complete", finished=time.time(), pid=os.getpid()))
            print("COMPLETE " + str(output / "summary.json"), flush=True)
        except BaseException as error:
            summarize(output)
            atomic_json(output / "status.json", dict(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=str(error), finished=time.time(), pid=os.getpid()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--reuse-builds", type=Path,
                        help="reuse identical native builds from an earlier sweep; recheck and remeasure all candidates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-known-one-token", action="store_true",
                        help="explicitly waive only the bounded Qwen35B rank7 replay-1-0-0 failure")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.dry_run:
        print(json.dumps(dict(candidates=candidates(), models=MODELS, sizes=SIZES,
            builds=len(candidates()) * len(MODELS), search_measurements=len(candidates()) * len(MODELS) * len(SIZES) * 2), indent=2))
        return
    if args.worker:
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "setup.lock").open("w") as setup_lock:
        fcntl.flock(setup_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "manifest.json"
        if not path.exists():
            hashes = snapshot(args.output)
            reuse = reuse_builds(args.output, args.reuse_builds, hashes) if args.reuse_builds else None
            # Store only execution-relevant settings, not credentials or unrelated
            # shell variables. Clear inherited tuning/profiling experiment flags.
            keys = ["PATH", "HOME", "CUDA_HOME", "CUTLASS_INCLUDE_DIR", "NCCL_INCLUDE_DIR", "NCCL_LIB_DIR",
                    "NCCL_SOCKET_IFNAME", "NCCL_IB_DISABLE", "CXX", "NVCC", "TMPDIR"]
            env = {k: os.environ[k] for k in keys if k in os.environ}
            atomic_json(path, dict(repository=str(ROOT), python=sys.executable, created=time.time(),
                git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                source_hashes=hashes, environment=env, candidates=candidates(), models=MODELS, sizes=SIZES,
                hardware=subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"], text=True),
                registers=[224, 56], candidate_builds=len(candidates()) * len(MODELS), includes_default=True,
                validation=POLICY, kernel_policy=KERNEL_POLICY,
                allow_known_one_token=args.allow_known_one_token,
                reuse=reuse,
                timing="EP8 max-rank CUDA events; alternating BF16/FP8 source variants; weights excluded; source preparation included"))
        if json.loads(path.read_text()).get("kernel_policy") != KERNEL_POLICY:
            raise RuntimeError("DT kernel policy changed; use a new output directory")
        if json.loads(path.read_text()).get("validation") != POLICY:
            raise RuntimeError("validation policy changed; use a new output directory with --reuse-builds")
        if args.allow_known_one_token and not json.loads(path.read_text()).get("allow_known_one_token", False):
            raise RuntimeError("known-case waiver changed; use a new output directory")
        command = [sys.executable, str(args.output / "source/tools/autotune_dt_fp8.py"),
                   "--output", str(args.output), "--worker"]
        if args.retry_failed:
            command.append("--retry-failed")
        if args.detach:
            with (args.output / "driver.log").open("a") as stream:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps(dict(pid=process.pid, output=str(args.output), log=str(args.output / "driver.log"))))
        else:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
