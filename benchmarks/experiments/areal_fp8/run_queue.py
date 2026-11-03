"""Run selected matched trials, optionally adopting an already running trial."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

import psutil

from cleanup import alive, cleanup
from report import summarize


def write_json(path, value):
    def finite(item):
        if isinstance(item, dict):
            return {k: finite(v) for k, v in item.items()}
        if isinstance(item, list):
            return [finite(v) for v in item]
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, allow_nan=False))
    temporary.replace(path)


def gpu_pids():
    return subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
    ], text=True, timeout=30).strip()


def fatal_trial_error(chunk):
    if any(marker in chunk for marker in (
        'torch.OutOfMemoryError:', 'CUDA error: an illegal memory access',
        'RuntimeError: NCCL Error',
        'torch.distributed.DistBackendError:',
    )):
        return 'fatal_cuda_or_collective_error'
    if re.search(r'TrainController (?:ERROR|CRITICAL):.*initializ', chunk, re.IGNORECASE):
        return 'initialization_failed'
    return None


class AdoptedProcess:
    """Monitor an existing launcher without requiring it to be our child."""

    def __init__(self, pid, run, variant):
        self.process = psutil.Process(pid)
        argv = self.process.cmdline()
        assert any(a.endswith('/areal_fp8/run.py') for a in argv), argv[:2]
        assert variant in argv and str(run.parent) in argv
        self.pid = pid

    def poll(self):
        return None if alive(self.process) else 0

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while self.poll() is None:
            if time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired('adopted launcher', timeout)
            time.sleep(0.5)

    def terminate(self):
        if alive(self.process):
            self.process.terminate()


def publish_result(root, variant, result):
    # Both node-local queues share this directory. Serialize aggregate updates.
    with (root / 'comparison.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_json(root / variant / 'queue_result.json', result)
        results = {v: json.loads(p.read_text()) for v in ('baseline', 'weftmoe', 'fp8')
                   if (p := root / v / 'queue_result.json').exists()}
        write_json(root / 'comparison.json', results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--wandb-project", required=True)
    parser.add_argument("--netrc", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--areal-path", type=Path)
    parser.add_argument("--deps-path", type=Path)
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-base-url")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=120)
    parser.add_argument('--variants', nargs='+', choices=('baseline', 'weftmoe', 'fp8'),
                        default=['baseline', 'weftmoe', 'fp8'])
    parser.add_argument('--queue-name', default='queue')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--attempt', default='')
    parser.add_argument('--release-decode-graphs', action='store_true')
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    assert len(args.variants) == len(set(args.variants))
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.queue_name)
    state_path = root / f'{args.queue_name}_status.json'
    previous = json.loads(state_path.read_text()) if args.resume and state_path.exists() else {}
    script = Path(__file__).with_name("run.py")
    common = ["--dp-attention", "--model-path", str(args.model_path.resolve()),
              "--extension", str(args.extension.resolve()), "--output-root", str(root),
              "--experiment", args.experiment, "--wandb-project", args.wandb_project,
              "--netrc", str(args.netrc.resolve()), "--steps", str(args.steps),
              f'--attempt={args.attempt}']
    common += ["--data-path", str(args.data_path.resolve()),
               "--wandb-entity", args.wandb_entity]
    for name in ("areal_path", "deps_path", "wandb_base_url"):
        value = getattr(args, name)
        if value is not None:
            common += ["--" + name.replace("_", "-"), str(value)]
    if args.release_decode_graphs:
        common.append('--release-decode-graphs')
    # Resolve selected new trials; preserve the manifest of an adopted trial.
    for variant in args.variants:
        if args.resume and (root / variant / 'status.json').exists():
            continue
        with (root / f"{variant}-config.log").open("w") as log:
            subprocess.run([sys.executable, str(script), variant, *common, "--check"],
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    manifests = [json.loads((root / v / "manifest.json").read_text())
                 for v in ("baseline", "weftmoe", "fp8")
                 if (root / v / "manifest.json").exists()]
    controlled = ("model", "model_config_sha256", "dataset", "allocation", "context",
                  "total_token_cap", "completion_cap", "microbatch_token_cap", "prompts",
                  "samples_per_prompt", "concurrent_groups", "consumer_groups", "queue_groups",
                  "max_staleness", "max_running_requests", "dp_size", "enable_dp_attention",
                  "cuda_graph_max_bs", "mem_fraction_static", "max_prefill_tokens", "optimizer",
                  "recompute", "steps")
    for key in controlled:
        assert all(m[key] == manifests[0][key] for m in manifests), key
    state = dict(pid=os.getpid(), host=socket.gethostname(), variants=args.variants,
                 experiment=args.experiment, stage="validated",
                 common={k: manifests[0][k] for k in controlled}, results=previous.get('results', {}))
    write_json(state_path, state)
    if args.check:
        return
    for variant in args.variants:
        if variant in state['results']:
            continue
        run = root / variant
        adopted = None
        if args.resume and (run / 'status.json').exists():
            trial_status = json.loads((run / 'status.json').read_text())
            assert trial_status['host'] == socket.gethostname(), 'Cannot adopt a remote trial'
            assert 'exit_code' not in trial_status, 'Trial already exited; inspect before resuming'
            trainer = psutil.Process(trial_status['pid'])
            assert trainer.environ().get('AREAL_EXPERIMENT_ROOT') == str(run / 'run')
            adopted = AdoptedProcess(trainer.ppid(), run, variant)
        if adopted is None and gpu_pids():
            state.update(stage="blocked_gpu_busy", next_variant=variant)
            write_json(state_path, state)
            raise RuntimeError("GPUs are occupied; refusing to overlap trials")
        state.update(stage="running", active_variant=variant, completed_steps=0)
        write_json(state_path, state)
        command = [sys.executable, str(script), variant, *common]
        if variant != "baseline":
            command += ["--comparison-baseline", str(root / "baseline")]
        if adopted is not None:
            process = adopted
        else:
            with (root / f"{variant}-supervisor.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
        state['launcher_pid'] = process.pid
        write_json(state_path, state)
        last_progress = time.monotonic()
        offset, completed, abort_reason, complete_at = 0, 0, None, None
        if adopted is not None:
            # Replay everything after the last successful step. Seeking to EOF
            # can miss a fatal startup error that occurred during adoption.
            completed = previous.get('completed_steps', 0)
            log_path = run / 'rl.log'
            if log_path.exists():
                with log_path.open(errors='replace') as log:
                    while line := log.readline():
                        if re.search(r'Train step \d+/\d+ done', line):
                            offset = log.tell()
            state['completed_steps'] = completed
            write_json(state_path, state)
        while process.poll() is None:
            log_path = run / "rl.log"
            if log_path.exists():
                with log_path.open(errors="replace") as log:
                    log.seek(offset)
                    chunk = log.read()
                    offset = log.tell()
                markers = re.findall(r"Train step (\d+)/\d+ done", chunk)
                if markers:
                    completed = max(completed, *(int(m) for m in markers))
                    last_progress = time.monotonic()
                    state["completed_steps"] = completed
                    write_json(state_path, state)
                if "Training completes!" in chunk:
                    complete_at = time.monotonic()
                failure = fatal_trial_error(chunk)
                if failure:
                    abort_reason = failure
                    state.update(stage='cleaning_failed_trial', error=abort_reason)
                    write_json(state_path, state)
                    break
            if complete_at is not None and time.monotonic() - complete_at > 60:
                abort_reason = "training_completed_cleanup"
                break
            if time.monotonic() - last_progress > args.stall_minutes * 60:
                abort_reason = "no_completed_step_within_stall_limit"
                break
            time.sleep(15)
        # Match only this trial and its descendants; do not kill other GPU users.
        trial = f"{variant}-b32n16-{args.steps}step{args.attempt}"
        cleanup_report = cleanup(args.experiment, trial, run / "run", apply=True)
        write_json(run / "cleanup.json", cleanup_report)
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=30)
        remaining = []
        for pid in cleanup_report['remaining']:
            try:
                remaining.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                pass
        deadline = time.monotonic() + 180
        # CUDA teardown may outlive SIGKILL briefly. Wait for the same owned
        # processes to disappear before declaring cleanup unsuccessful.
        while time.monotonic() < deadline:
            try:
                gpu_busy = bool(gpu_pids())
            except subprocess.TimeoutExpired:
                # NVML may briefly block while CUDA tears down a failed context.
                gpu_busy = True
            if not any(alive(p) for p in remaining) and not gpu_busy:
                break
            time.sleep(5)
        cleanup_report['remaining'] = [p.pid for p in remaining if alive(p)]
        write_json(run / 'cleanup.json', cleanup_report)
        result = summarize(run)
        result["termination_reason"] = abort_reason
        result["passed"] = (result["completed_steps"] == args.steps
                            and result["successful_updates"] == args.steps
                            and not result["nonfinite_grad_steps"])
        state["results"][variant] = result
        publish_result(root, variant, result)
        write_json(state_path, state)
        if cleanup_report['remaining']:
            raise RuntimeError('Trial processes remain; refusing to start another arm')
    state.update(stage="complete" if all(r["passed"] for r in state["results"].values())
                 else "finished_with_failures", active_variant=None)
    write_json(state_path, state)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-root" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-root") + 1]).resolve()
            name = sys.argv[sys.argv.index('--queue-name') + 1] if '--queue-name' in sys.argv else 'queue'
            path = root / f"{name}_status.json"
            if root.exists():
                state = json.loads(path.read_text()) if path.exists() else {}
                state.update(stage="queue_error", error=f"{type(error).__name__}: {error}")
                write_json(path, state)
        raise
