"""Build isolated FP8 weight-layout experiments without changing production sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('rs', 'rs_scalar', 'warp_transpose'), default='rs')
    parser.add_argument('--model', choices=('qwen35b', 'qwen122b', 'flash'), default='qwen35b')
    parser.add_argument('--tile-n', type=int, choices=(128, 256), default=256)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[2]
    for filename, expected in json.loads((here / 'baseline_sha256.json').read_text()).items():
        actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
        if actual != expected:
            parser.error(f'{filename} changed since the experiment; rebase the patch first')
    output = args.output.resolve()
    if (output / 'libweftmoe.so').exists():
        parser.error('Choose an unused output directory to preserve existing benchmark binaries')
    output.mkdir(parents=True, exist_ok=True)
    source = Path(tempfile.mkdtemp(prefix='source-', dir=output))
    for name in ('src', 'include', 'configs', 'tests'):
        shutil.copytree(root / name, source / name)
    shutil.copy2(root / 'build_native.py', source / 'build_native.py')
    patch_name = 'warp_transpose' if args.variant == 'warp_transpose' else 'rs'
    subprocess.run(['patch', '-p1', '--batch'], cwd=source,
                   input=(here / (patch_name + '.patch')).read_bytes(), check=True)
    if args.variant == 'rs_scalar':
        shutil.copy2(here / 'rs_scalar.cuh', source / 'src/cuda/weftmoe_pf/fp8_weight_rs.cuh')
    profile = source / 'configs/profiles' / args.model
    if args.tile_n == 128:
        filename = profile / 'weftmoe_forward_tuning_generated.hpp'
        code = filename.read_text()
        for define in ('WEFTMOE_BWD_DAB_TILE_N', 'WEFTMOE_BWD_DX_TILE_N'):
            assert f'#define {define} 256' in code
            code = code.replace(f'#define {define} 256', f'#define {define} 128')
        filename.write_text(code)
    flags = ['-DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1', '-I' + str(profile)]
    if args.variant == 'warp_transpose':
        flags.append('-DWEFTMOE_FP8_TRANSPOSE_ON_LOAD=1')
    env = dict(os.environ)
    env['NVCCFLAGS'] = shlex.join(shlex.split(env.get('NVCCFLAGS', '')) + flags)
    cmd = [sys.executable, str(source / 'build_native.py'), 'lib', '--output',
           str(output / 'libweftmoe.so'), '--jobs', str(args.jobs)]
    (output / 'experiment.json').write_text(json.dumps(
        dict(variant=args.variant, model=args.model, tile_n=args.tile_n,
             source=str(source), command=cmd, nvccflags=env['NVCCFLAGS']), indent=2) + '\n')
    subprocess.run(cmd, env=env, check=True)
    subprocess.run([sys.executable, str(root / 'tools/build_tensor_runtime.py'),
                    '--library', str(output / 'libweftmoe.so')], env=env, check=True)


if __name__ == '__main__':
    main()
