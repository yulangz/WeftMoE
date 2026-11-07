#!/usr/bin/env python3
"""Verify lazy source installation outside the checkout; optionally link a supplied native library."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--wheel', type=Path)
    mode.add_argument('--editable', action='store_true')
    parser.add_argument('--native-library', type=Path,
                        help='Optional explicit/JIT library for the CUDA link/run probe')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='weftmoe-install-') as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / 'venv')
        python = root / 'venv/bin/python'
        install = ['-e', str(ROOT)] if args.editable else [str(args.wheel.resolve())]
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps',
                        *install], cwd=root, check=True)
        result = subprocess.check_output([str(python), '-c',
            'import json, weftmoe; from weftmoe.jit import source_root; '
            'print(json.dumps([weftmoe.get_include(), str(source_root())])); '
            'assert weftmoe.token_bucket(65).upper == 8192'], cwd=root, text=True)
        include, native_source = map(Path, json.loads(result))
        expected_root = ROOT if args.editable else root
        assert native_source == expected_root or expected_root in native_source.parents, 'unexpected native source location'
        for name in ('build_native.py', 'tools/build_tensor_runtime.py', 'src/python/runtime.cpp',
                     'src/python/ops.cpp', 'src/cuda/activation_transport.cu',
                     'configs/profiles/qwen35b/weftmoe_forward_tuning_generated.hpp'):
            assert (native_source / name).is_file(), name
        # Installation and public-header discovery do not need CUDA or torch.
        public_probe = root / 'public_probe.cpp'
        public_probe.write_text('#include "weftmoe/version.hpp"\n'
                                'static_assert(weftmoe::kPublicApiVersionMajor == 2);\n')
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-fsyntax-only',
                        '-I', str(include), str(public_probe)], cwd=root, check=True)
        if args.native_library:
            check_native(root, include, args.native_library.resolve())
    print('[PASS] package installation, source payload and public-header discovery')


def check_native(root, include, library):
    source = root / 'native_probe.cpp'
    source.write_text("""#include <iostream>
#include "weftmoe/cuda/activation_transport.hpp"
#include "weftmoe/cuda/backward.hpp"
int main() {
  const auto capability = weftmoe::cuda::probe_activation_transport_capability(1);
  std::cout << "GIN available: " << capability.gin_usable << std::endl;
  return weftmoe::cuda::backward_pipeline_available() ? 0 : 1;
}
""")
    output = root / 'native_probe'
    subprocess.run([os.environ.get('NVCC', '/usr/local/cuda/bin/nvcc'),
                    '-Wno-deprecated-gpu-targets', '-std=c++17',
                    '-I', str(include), str(source),
                    str(library), '-o', str(output)],
                   cwd=root, check=True)
    env = dict(os.environ)
    inherited = [str(Path(p).resolve()) for p in env.get('LD_LIBRARY_PATH', '').split(':') if p]
    env['LD_LIBRARY_PATH'] = ':'.join([str(library.parent), *inherited])
    subprocess.run([str(output)], cwd=root, env=env, check=True, timeout=180)
    dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
    assert 'SONAME' in dynamic
    assert 'RPATH' not in dynamic and 'RUNPATH' not in dynamic
    print('[PASS] explicit native library: C++ link and run')

if __name__ == '__main__':
    main()
