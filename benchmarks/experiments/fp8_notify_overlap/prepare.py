"""Initialize a fresh reproduction after applying starting-state.patch."""
from pathlib import Path
import shutil
root=Path.cwd();base=root/'build/fp8-notify-overlap';source=base/'source';source.mkdir(parents=True,exist_ok=False)
for name in ['src/cuda/activation_transport.cu','src/cuda/weftmoe_pf/split_notify.cuh','src/cuda/fp8/source.cuh','src/cuda/weftmoe_pf/local_compute.cuh','src/cuda/fp8/tile.cuh']:
 target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
for model in ['qwen35b','qwen122b','flash']:(base/f'baseline_after-{model}').symlink_to(base/f'baseline-{model}',target_is_directory=True)
print('Initialized',source)
