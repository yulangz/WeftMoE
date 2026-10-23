#!/usr/bin/env python3
"""Install a package outside the source tree and verify its native interface."""
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
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='weftmoe-install-') as temporary:
        root = Path(temporary)
        venv.EnvBuilder(with_pip=True).create(root / 'venv')
        python = root / 'venv/bin/python'
        install = ['-e', str(ROOT)] if args.editable else [str(args.wheel.resolve())]
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps',
                        *install], cwd=root, check=True)
        result = subprocess.check_output([str(python), '-c',
            'import json, weftmoe; print(json.dumps([weftmoe.get_include(), '
            'weftmoe.get_library_path()])); assert weftmoe.token_bucket(65).upper == 8192'],
            cwd=root, text=True)
        include, library = map(Path, json.loads(result))
        expected_root = ROOT if args.editable else root
        assert expected_root in library.parents, 'unexpected native resource location'
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
                        '-L', str(library.parent), '-lweftmoe', '-o', str(output)],
                       cwd=root, check=True)
        env = dict(os.environ)
        inherited = [str(Path(p).resolve()) for p in env.get('LD_LIBRARY_PATH', '').split(':') if p]
        env['LD_LIBRARY_PATH'] = ':'.join([str(library.parent), *inherited])
        subprocess.run([str(output)], cwd=root, env=env, check=True, timeout=180)
        dynamic = subprocess.check_output(['readelf', '-d', str(library)], text=True)
        assert 'SONAME' in dynamic and '[libweftmoe.so]' in dynamic
        assert 'RPATH' not in dynamic and 'RUNPATH' not in dynamic
    print('[PASS] package installation, resource discovery, C++ link and run')


if __name__ == '__main__':
    main()
