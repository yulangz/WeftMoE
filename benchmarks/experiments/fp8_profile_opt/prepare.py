"""Initialize scratch snapshots from a fixed224 baseline checkout."""
from pathlib import Path
import shutil
root=Path.cwd();base=root/'build/fp8-profile-opt';exp=root/'benchmarks/experiments/fp8_profile_opt'
assert 'kMathRegisters = 224U' in (root/'src/cuda/fp8/tile.cuh').read_text()
files=['src/cuda/backward/swiglu.cuh','src/cuda/weftmoe_pf/local_compute.cuh','src/cuda/fp8/column.cuh','src/cuda/weftmoe_pf/fp8_swiglu.cuh','src/cuda/weftmoe_pf/fp8_wgrad.cuh','src/cuda/weftmoe_pf/backward_weights.cuh','src/cuda/weftmoe_pf/backward_kernel.cuh','src/cuda/weftmoe_pf/local_kernel.cuh','src/cuda/fp8/tile.cuh']
if (base/'source').exists():raise RuntimeError('Preserve existing snapshots; use a clean experiment directory')
for f in files:
 p=base/'source'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/f,p)
shutil.copyfile(exp/'build.py',base/'build.py')
print('Initialized',base,'Build baseline with build.py baseline -Xptxas=-v.')
