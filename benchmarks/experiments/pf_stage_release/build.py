"""Build only the earlier operand-stage retirement candidate, from a snapshot."""
import concurrent.futures
import difflib
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'benchmarks/experiments/dt_fp8'))
from autotune import MODELS, snapshot, atomic_json, run_command

BASE = ROOT / 'build/pf-stage-release-20261101'
REL = 'src/cuda/weftmoe_pf/backward_kernel.cuh'
BASE.mkdir(parents=True, exist_ok=True)
if not (BASE / 'manifest.json').exists():
    hashes = snapshot(BASE)
    old_tree = ROOT / 'build/dt-fp8-pipeline-20261101/trees/selected_auto'
    for rel, digest in hashes.items():
        if rel.startswith(('src/', 'include/', 'configs/')):
            assert hashlib.sha256((old_tree / rel).read_bytes()).hexdigest() == digest, rel
    path = BASE / 'source' / REL
    before = path.read_text()
    old = '''        if (a.fp8_x_row_data) {
          asm volatile("cp.async.wait_group 0;" ::: "memory");'''
    new = '''      }
      // The dR conversion is the final reader of the operand stage. X uses
      // the separate xraw/xscales arena, so let TMA refill while X converts.
      backward_compute_detail::release_stage(s.empty[stage]);
      if (cache_owner) {
        if (a.fp8_x_row_data) {
          asm volatile("cp.async.wait_group 0;" ::: "memory");'''
    assert before.count(old) == 1
    after = before.replace(old, new)
    old_tail = '''      }
      backward_compute_detail::release_stage(s.empty[stage]);
      stage = (stage + 1U) % kStages;'''
    assert after.count(old_tail) == 1
    after = after.replace(old_tail, '''      }
      stage = (stage + 1U) % kStages;''')
    path.write_text(after)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                fromfile='a/' + REL, tofile='b/' + REL))
    (Path(__file__).parent / 'early.patch').write_text(patch)
    atomic_json(BASE / 'manifest.json', dict(baseline_hashes=hashes,
        candidate_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

def build(model):
    tree = BASE / 'source'
    native = BASE / f'early-{model}'
    native.mkdir(exist_ok=True)
    baseline = BASE / f'baseline-{model}'
    if not baseline.exists():
        baseline.symlink_to(ROOT / 'build/dt-fp8-pipeline-20261101/native' / model / 'selected_auto')
    repeat = BASE / f'baseline_after-{model}'
    if not repeat.exists():
        repeat.symlink_to(baseline)
    if (native / 'complete.json').exists():
        return
    env = dict(os.environ, NVCCFLAGS=f'-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 -I{tree}/configs/profiles/{model} -Xptxas=-v')
    run_command([sys.executable, 'build_native.py', 'lib', '--output', str(native / 'libweftmoe.so'), '--jobs', '2'],
                cwd=tree, env=env, log=native / 'build.log', timeout=1800)
    run_command([sys.executable, 'tools/build_tensor_runtime.py', '--library', str(native / 'libweftmoe.so')],
                cwd=tree, env=env, log=native / 'adapter.log', timeout=600)
    atomic_json(native / 'complete.json', dict(model=model, flags=env['NVCCFLAGS']))
    print('BUILT', model, flush=True)

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    list(pool.map(build, MODELS))
