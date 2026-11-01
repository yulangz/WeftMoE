"""Run retained-training PF / DT parity on all three single-node EP8 models."""
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks/experiments/dt_fp8'))
from autotune import MODELS, run_command, idle_gpus
base = ROOT / 'build/dt-m64-20261101'
base.mkdir(parents=True, exist_ok=True)
phase = sys.argv[1] if len(sys.argv) > 1 else 'before'
for model, (h, i, topk) in MODELS.items():
    native = (ROOT / 'build/dt-m-shapes-20261101/native' / model / 'm64'
              if phase == 'before' else base / 'native' / model)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS='1', NCCL_DEBUG='WARN',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i),
        WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),
        LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64')
    for key in ('WEFTMOE_AUTOTUNE_TABLE', 'WEFTMOE_STATIC_RING_TOKENS', 'BENCH_PROFILE'):
        env.pop(key, None)
    for compact in (0, 1):
        log = base / f'{phase}-{model}-parity{"-compact" if compact else ""}.log'
        env['TEST_PF_COMPACT'] = str(compact)
        idle_gpus()
        print('PARITY', phase, model, 'compact', compact, flush=True)
        run_command([sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8',
                     'tests/torchrun_pf_dt.py'], cwd=ROOT, env=env, log=log, timeout=1200)
