"""Compare immutable baseline and row-to-column GEMM candidates on single-node EP8.

Run from the repository root. Build directories are {label}-{model}; BF16 uses
baseline-{model}. Weight preparation is excluded and all dW outputs are BF16.
Example: python benchmarks/experiments/fp8_rowcol_gemm/benchmark.py \
    --native-root build/fp8-rowcol-gemm --labels baseline wide bf16 \
    --mode backward --output build/fp8-rowcol-gemm/reproduce-bwd
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path.cwd()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root', type=Path, required=True)
    parser.add_argument('--labels', nargs='+', required=True)
    parser.add_argument('--models', nargs='+', choices=('qwen35b', 'qwen122b', 'flash'),
                        default=['qwen35b', 'qwen122b', 'flash'])
    parser.add_argument('--tokens', nargs='+', type=int, default=[8192, 20480])
    parser.add_argument('--mode', choices=('cycle', 'backward'), default='cycle')
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error('Choose an unused output directory to preserve measurements')
    output.mkdir(parents=True)
    records, hashes = [], {}
    for model in args.models:
        for label in args.labels:
            native = args.native_root.resolve() / f'{"baseline" if label == "bf16" else label}-{model}'
            extensions = list(native.glob('_runtime*.so'))
            if len(extensions) != 1:
                raise RuntimeError(f'Expected exactly one runtime extension in {native}')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                       OMP_NUM_THREADS='1', NCCL_DEBUG='WARN', TEST_FP8_COMPACT='1',
                       LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',
                       WEFTMOE_TENSOR_EXTENSION=str(extensions[0]))
            env.pop('WEFTMOE_AUTOTUNE_TABLE', None)
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                       '--nproc-per-node=8', 'benchmarks/torchrun_training.py',
                       '--backend', 'pf', '--model', model, '--tokens', *map(str, args.tokens),
                       '--warmups', str(args.warmups), '--samples', str(args.samples),
                       '--stream-ordered', '--prime-trace']
            if label != 'bf16':
                command += ['--fp8', '--weight-grad-dtype', 'bf16']
            if args.mode == 'backward':
                command += ['--backward-only']
            print('START', model, label, args.mode, flush=True)
            start = time.time()
            path = output / f'{label}-{model}.log'
            with path.open('w') as log:
                subprocess.run(['timeout', '--kill-after=10s', '420s', *command], cwd=ROOT,
                               env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            found = 0
            for line in path.read_text().splitlines():
                if not line.startswith('TRAINING_CYCLE_JSON '):
                    continue
                result = json.loads(line.removeprefix('TRAINING_CYCLE_JSON '))
                key = (model, result['nominal_tokens_per_rank'])
                workload_hash = result['workload_sha256_per_rank']
                if key in hashes and hashes[key] != workload_hash:
                    raise RuntimeError(f'Workload mismatch for {key}')
                hashes[key] = workload_hash
                records.append(dict(label=label, model=model, tokens=key[1], mode=args.mode,
                                    elapsed_seconds=time.time()-start, command=command,
                                    native_sha256=hashlib.sha256((native / 'libweftmoe.so').read_bytes()).hexdigest(),
                                    result=result))
                found += 1
                print('RESULT', model, label, key[1],
                      round(result['wall_slowest_rank_p50_us']/1000, 4), 'ms', flush=True)
            if found != len(args.tokens):
                raise RuntimeError(f'Missing benchmark records in {path}')
            (output / 'results.json').write_text(json.dumps(dict(
                source_id='weftmoe-fp8-rowcol-gemm-20261030', world=8,
                method='Unprofiled per-step slowest-rank wall median. Full workload priming, '
                       'then warmups and samples. BF16 dW; FP8 weight preparation excluded; '
                       'identical routes and inputs across variants.',
                records=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
