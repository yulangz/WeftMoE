import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks/experiments/dt_fp8'))
from autotune import MODELS, run_command, idle_gpus
BASE = ROOT / 'build/pf-stage-release-20261101'
for model, (h, i, _) in MODELS.items():
    for label in ('baseline', 'early'):
        native = BASE / f'{label}-{model}'
        env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
            OMP_NUM_THREADS='1', NCCL_DEBUG='WARN',
            WEFTMOE_TENSOR_EXTENSION=str(next(native.glob('_runtime*.so'))),
            LD_LIBRARY_PATH='/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64',
            TEST_HIDDEN=str(h), TEST_INTERMEDIATE=str(i), TEST_TOKENS='257',
            TEST_ZERO_RANK='1', TEST_ROW_SCALES='1', TEST_FP8_SOURCE_X='1',
            TEST_FP8_COMPACT='1', WEFTMOE_STATIC_RING_TOKENS='128')
        env.pop('WEFTMOE_AUTOTUNE_TABLE', None)
        launcher = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=8']
        idle_gpus()
        run_command([*launcher, 'benchmarks/experiments/pf_stage_release/check.py'], cwd=ROOT, env=env,
                    log=BASE/f'{label}-{model}-check.log', timeout=240)
        print('PASS', label, model, flush=True)
        if label == 'early':
            env.pop('WEFTMOE_STATIC_RING_TOKENS')
            run_command([*launcher, 'benchmarks/experiments/pf_stage_release/pf_dt.py'], cwd=ROOT, env=env,
                        log=BASE/f'pf-dt-{model}.log', timeout=300)
            print('PF_DT_PASS', model, flush=True)
