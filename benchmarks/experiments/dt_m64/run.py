"""Build and validate the fixed-M64 DT implementation on single-node EP8."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks/experiments/dt_fp8'))
from autotune import MODELS, snapshot, atomic_json, run_command, idle_gpus
BASE = ROOT / 'build/dt-m64-20261101'

def build(model):
    tree, out = BASE / 'source', BASE / 'native' / model
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'complete.json').exists():
        return
    env = dict(os.environ, NVCCFLAGS=f'-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I{tree}/configs/profiles/{model} -Xptxas=-v')
    run_command([sys.executable, 'build_native.py', 'lib', '--output', str(out / 'libweftmoe.so'), '--jobs', '2'],
                cwd=tree, env=env, log=out / 'build.log', timeout=1800)
    run_command([sys.executable, 'tools/build_tensor_runtime.py', '--library', str(out / 'libweftmoe.so')],
                cwd=tree, env=env, log=out / 'adapter.log', timeout=600)
    atomic_json(out / 'complete.json', dict(model=model, flags=env['NVCCFLAGS']))
    print('BUILT', model, flush=True)

def evaluate(stage):
    for model, (h, i, topk) in MODELS.items():
        native = BASE / 'native' / model
        env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', NCCL_DEBUG='WARN',
            CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i),
            WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),
            LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',
            TEST_DT_ALLOW_KNOWN_ONE_TOKEN='1', BENCH_TOPK=str(topk), BENCH_EXPERTS='256',
            BENCH_SIZES='1 16 32 64 128 256 512', BENCH_REPEATS='100', BENCH_SWEEPS='9', BENCH_SEED='1700')
        for key in ('WEFTMOE_AUTOTUNE_TABLE', 'WEFTMOE_STATIC_RING_TOKENS', 'BENCH_PROFILE'):
            env.pop(key, None)
        launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
        script = 'tests/torchrun_dt_fp8.py' if stage == 'check' else 'benchmarks/experiments/dt_fp8/benchmark.py'
        variants = ('before', 'after') if stage == 'bench' else ('after',)
        for variant in variants:
            selected = (ROOT / 'build/dt-m-shapes-20261101/native' / model / 'm64'
                        if variant == 'before' else native)
            env['WEFTMOE_TENSOR_EXTENSION'] = str(next(selected.glob('_runtime*.so')))
            idle_gpus()
            log = BASE / f'{"before-" if variant == "before" else ""}{model}-{stage}.log'
            run_command([*launcher, script], cwd=ROOT, env=env, log=log, timeout=600)
            print(stage.upper(), variant, model, flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['build', 'check', 'bench'])
    args = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if args.stage == 'build':
        if not (BASE / 'manifest.json').exists():
            atomic_json(BASE / 'manifest.json', dict(source_hashes=snapshot(BASE), models=MODELS,
                policy='Fixed M64 for BF16 and FP8; boolean slice scheduling, no M16 code or selector.'))
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(build, MODELS))
    else:
        evaluate(args.stage)
