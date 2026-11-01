from pathlib import Path
import subprocess, os, sys
base = Path('build/fp8-profile-review')
files = ['src/cuda/profiling/device_timeline_types.hpp', 'src/cuda/weftmoe_pf/local_compute.cuh', 'src/cuda/weftmoe_pf/backward_kernel.cuh', 'src/cuda/activation_transport.cu']
saved = {p: Path(p).read_bytes() for p in files}
try:
    for patch in ['instrumentation.patch', 'backward-context.patch']:
        subprocess.run(['git', 'apply', str(base / patch)], check=True)
    subprocess.run([sys.executable, str(base / 'build.py'), 'full', '-DWEFTMOE_PF_BWD_TASK_TIMING=1', '-Xptxas=-v'], env=dict(os.environ, WEFTMOE_DEVICE_TIMELINE='1'), check=True)
    subprocess.run([sys.executable, str(base / 'adapter.py'), 'full'], check=True)
finally:
    for path, data in saved.items():
        Path(path).write_bytes(data)
