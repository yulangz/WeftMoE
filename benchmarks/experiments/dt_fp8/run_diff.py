"""Run the untimed global-diff measurements using existing default binaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from autotune import MODELS, ROOT, atomic_json, idle_gpus, run_command
from benchmarks.fp8_validation import POLICY


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-root', type=Path, default=ROOT / 'build/dt-fp8-autotune-m64/jobs')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for mode in ['moe', 'expert']:
        for model, (h, i, topk) in MODELS.items():
            native = args.native_root.resolve() / model / 'default/native'
            env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                NCCL_DEBUG='WARN', OMP_NUM_THREADS='1', TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i),
                BENCH_TOPK=str(topk), DIFF_MODE=mode,
                WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),
                LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
            env.pop('WEFTMOE_AUTOTUNE_TABLE', None)
            env.pop('WEFTMOE_STATIC_RING_TOKENS', None)
            log = output / f'{model}-{mode}.log'
            print('START', model, mode, flush=True)
            idle_gpus()
            try:
                run_command([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                    'benchmarks/experiments/dt_fp8/measure_diff.py'], cwd=ROOT, env=env, log=log, timeout=600)
                text = log.read_text()
                assert 'COMPLETE ' + mode in text
                rows = [dict(model=model, **json.loads(line[5:])) for line in text.splitlines() if line.startswith('DIFF ')]
                assert len(rows) == (42 if mode == 'moe' else 90), len(rows)
                records.extend(rows)
                print('DONE', model, mode, len(rows), flush=True)
            except (RuntimeError, AssertionError) as error:
                failures.append(dict(model=model, mode=mode, error=str(error), log=str(log)))
                print('FAILED', model, mode, str(error), flush=True)
            atomic_json(output / 'results.json', dict(records=records, failures=failures))
    groups = {}
    for record in records:
        key = '/'.join(record[k] for k in ['model', 'mode', 'tensor'])
        groups.setdefault(key, []).append(record)
    summary = {}
    for key, rows in groups.items():
        summary[key] = dict(cases=len(rows), limit=rows[0]['limit'],
            global_diff_min=min(r['global_metrics']['diff'] for r in rows),
            global_diff_max=max(r['global_metrics']['diff'] for r in rows),
            max_rank_diff=max(r['max_rank_diff'] for r in rows),
            global_passes=sum(r['pass_global'] for r in rows),
            all_ranks_passes=sum(r['pass_every_rank'] for r in rows),
            sqnr_db_min=min((r['global_metrics']['sqnr_db'] for r in rows if r['global_metrics']['sqnr_db'] is not None), default=None))
    script = ROOT / 'benchmarks/experiments/dt_fp8/measure_diff.py'
    atomic_json(output / 'summary.json', dict(summary=summary, failures=failures, validation=POLICY,
        measurement_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
        reference='original BF16 operands; production BF16 MoE/expert; expert also checked against Torch autograd',
        diff='sum((actual-reference)^2)/(sum(actual^2)+sum(reference^2)); FP64 reductions',
        sqnr_db='10*log10(sum(reference^2)/sum(error^2)); sqnr_from_diff_db is the equal-norm approximation',
        zero_error_sqnr='null means infinite SQNR when error is zero',
        native_root=str(args.native_root.resolve())))
    print('SUMMARY', output / 'summary.json', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
