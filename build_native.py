"""Native compiler shared by package builds and repository development commands."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
HOST_SOURCES = (
    'src/cuda/backward_stub.cpp', 'src/cuda/control_protocol.cpp',
    'src/metadata/notify_dispatch.cpp', 'src/metadata/serialize.cpp',
    'src/transport/bootstrap_channel.cpp', 'src/transport/activation_protocol.cpp',
    'src/transport/host_fake_count_exchange.cpp', 'src/scoreboard/workspace_plan.cpp',
)
CUDA_SOURCES = (
    'src/cuda/device_notify.cu', 'src/cuda/activation_transport.cu',
    'src/transport/nccl_count_exchange.cu', 'src/cuda/fp8.cu',
)
TEST_SOURCE = 'tests/test_cuda_notify.cpp'
PUBLIC_HEADERS = tuple(str(p.relative_to(ROOT)) for p in sorted((ROOT / 'include').rglob('*.hpp'))
                       if p.relative_to(ROOT).as_posix() not in {
                           'include/weftmoe/kernels/persistent_contract.hpp',
                           'include/weftmoe/metadata/serialize.hpp'})


def tool(name, default):
    value = os.environ.get(name, default)
    path = shutil.which(value)
    if path is None:
        raise RuntimeError(f'{name}: compiler not found: {value}')
    return str(Path(path).resolve())


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def fresh(output, signature, command):
    try:
        info = json.loads(signature.read_text())
        return (info['command'] == command and output.is_file()
                and all(Path(p).stat().st_mtime_ns == stamp
                        for p, stamp in info['inputs'].items()))
    except (OSError, ValueError, KeyError):
        return False


def compile_source(source, command, cache):
    # Lock each object independently; concurrent wheels and developer builds
    # can share the cache. Never expose a partially written object or stamp.
    key = hashlib.sha256(json.dumps([source, command]).encode()).hexdigest()
    output = cache / f'{key}.o'
    signature = output.with_suffix('.json')
    cache.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not fresh(output, signature, command):
            print(f'compile {source}', flush=True)
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                obj = Path(temporary) / 'object.o'
                dep = Path(temporary) / 'object.d'
                # Capture dependencies before compilation. Recording only their
                # final mtimes can label an old object as fresh when an editor
                # changes a header while nvcc is still compiling it.
                def inputs():
                    dependencies = shlex.split(dep.read_text().replace('\\\n', '').split(':', 1)[1])
                    return {str(Path(p).resolve()): Path(p).stat().st_mtime_ns
                            for p in dependencies + [command[0], __file__]}

                run(command + ['-M', '-MF', str(dep), str(ROOT / source)])
                before = inputs()
                # -MD includes external CUDA, NCCL and CUTLASS headers too.
                run(command + ['-MD', '-MF', str(dep), '-c', str(ROOT / source), '-o', str(obj)])
                after = inputs()
                if before != after:
                    raise RuntimeError(f'{source}: inputs changed during compilation; rerun the build')
                stamp = Path(temporary) / 'signature.json'
                stamp.write_text(json.dumps({'command': command, 'inputs': after}))
                obj.replace(output)
                stamp.replace(signature)
    return output


def build(target='lib', output=None, jobs=None):
    if target not in {'lib', 'cuda-test', 'host-test', 'metadata-dump', 'metadata-bench'}:
        raise ValueError(f'unknown native target: {target}')
    cuda = target in {'lib', 'cuda-test'}
    testing = target != 'lib'
    jobs = jobs or int(os.environ.get('MAX_JOBS', '4'))
    if jobs < 1:
        raise ValueError('MAX_JOBS / --jobs must be positive')
    cache_root = Path(os.environ.get('WEFTMOE_ROOT') or '~/.cache/weftmoe').expanduser().resolve()
    cache = cache_root / 'compile' / 'package'
    cxx = tool('CXX', 'c++')
    cuda_home = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    nvcc = tool('NVCC', str(cuda_home / 'bin/nvcc')) if cuda else None
    common = ['-I' + str(ROOT / p) for p in ('include', 'src', 'tests')]
    common += [f'-DWEFTMOE_BUILD_TESTING={int(testing)}', '-DWEFTMOE_BUILD_SHARED=1']
    common += [f'-D{name}={os.environ.get(name, "0")}' for name in (
        'WEFTMOE_DEBUG_VALIDATE', 'WEFTMOE_DEVICE_TIMELINE', 'WEFTMOE_FORWARD_LIGHT_PROGRESS')]
    common += ['-DWEFTMOE_ENABLE_GIN_RDMA=' + os.environ.get(
        'WEFTMOE_ENABLE_GIN_RDMA', '1')]
    host = [cxx, *common, '-O2', '-g', '-std=c++17', '-pthread', '-Wall', '-Wextra',
            '-Wpedantic', '-Wconversion', '-Wshadow', '-fPIC', '-fvisibility=hidden',
            '-ffunction-sections', '-fdata-sections', *shlex.split(os.environ.get('CXXFLAGS', ''))]
    device = []
    if cuda:
        cutlass = Path(os.environ.get('CUTLASS_INCLUDE_DIR', '/opt/cutlass/include')).expanduser().resolve()
        if not (cutlass / 'cute/tensor.hpp').is_file():
            raise RuntimeError('CUTLASS_INCLUDE_DIR must contain cute/tensor.hpp')
        device = [nvcc, *common, '-isystem', str(cutlass), '-O2', '-g', '-std=c++17',
                  '--expt-relaxed-constexpr', '-Wno-deprecated-gpu-targets',
                  '-gencode', 'arch=compute_90a,code=sm_90a',
                  '-Xcompiler=-pthread,-fPIC,-fvisibility=hidden,-ffunction-sections,-fdata-sections,-Wall,-Wextra',
                  *shlex.split(os.environ.get('NVCCFLAGS', ''))]
        if os.environ.get('NCCL_INCLUDE_DIR'):
            device += ['-I' + str(Path(os.environ['NCCL_INCLUDE_DIR']).expanduser().resolve())]
    if os.environ.get('WERROR') == '1':
        host += ['-Werror']
        device += ['-Xcompiler=-Werror']
    sources = list(HOST_SOURCES) + (list(CUDA_SOURCES) if cuda else [])
    mains = {'cuda-test': TEST_SOURCE, 'host-test': 'tests/test_notify_dispatch.cpp',
             'metadata-dump': 'tests/metadata_dump.cpp',
             'metadata-bench': 'benchmarks/notify_dispatch_bench.cpp'}
    if testing:
        sources.append(mains[target])
    def compile_one(source):
        command = list(device if source.endswith('.cu') or source == TEST_SOURCE else host)
        if source == TEST_SOURCE:
            command += ['-x', 'cu']
        return compile_source(source, command, cache)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        objects = list(pool.map(compile_one, sources))
    names = {'lib': 'libweftmoe.so', 'cuda-test': 'test_cuda_notify',
             'host-test': 'test_notify_dispatch', 'metadata-dump': 'metadata_dump',
             'metadata-bench': 'notify_dispatch_bench'}
    output = Path(output or ROOT / 'build' / names[target]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc if cuda else cxx]
    if target == 'lib':
        command += ['-shared', '-Xlinker=-soname', '-Xlinker=libweftmoe.so', '-Xlinker=--gc-sections']
    command += [str(p) for p in objects]
    command += shlex.split(os.environ.get('LDFLAGS', ''))
    if cuda:
        if os.environ.get('NCCL_LIB_DIR'):
            command += ['-L' + str(Path(os.environ['NCCL_LIB_DIR']).expanduser().resolve())]
        command += ['-lnccl', '-lcublas', '-lcudart', '-lcuda', '-ldl']
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        binary = Path(temporary) / output.name
        run(command + ['-o', str(binary)])
        binary.replace(output)
    print(f'built {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', choices=['lib', 'cuda-test', 'host-test', 'metadata-dump', 'metadata-bench'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--jobs', type=int)
    args = parser.parse_args()
    build(args.target, args.output, args.jobs)
