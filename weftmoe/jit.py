"""Lazy native compilation. Never autotune, initialize EP, or benchmark here."""
from __future__ import annotations

from functools import lru_cache
import fcntl
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading

from .tuning import cache_root, default_table, digest, find_entry

logger = logging.getLogger(__name__)
_MODULES = {}
_LOAD_LOCK = threading.RLock()


def source_root():
    package = Path(__file__).resolve().parent
    checkout = package.parent
    return checkout if (checkout / 'build_native.py').is_file() else package / '_native_source'


@lru_cache(maxsize=None)
def source_digest(root=None):
    root = Path(root) if root else source_root()
    paths = [root / 'build_native.py', root / 'tools/build_tensor_runtime.py']
    for directory in ('include', 'src', 'configs/profiles'):
        paths.extend(p for p in (root / directory).rglob('*') if p.is_file())
    h = hashlib.sha256()
    for path in sorted(paths):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def default_header(hidden, intermediate):
    if any(type(x) is not int or x < 512 or x % 256 for x in (hidden, intermediate)):
        raise ValueError('automatic profiles require hidden/intermediate >= 512 and multiples of 256; '
                         'smaller shapes require a separately validated explicit extension')
    profile = {(2048, 512): 'qwen35b', (3072, 1024): 'qwen122b',
               (4096, 2048): 'flash'}.get((hidden, intermediate), 'qwen35b')
    header = (source_root() / 'configs/profiles' / profile /
              'weftmoe_forward_tuning_generated.hpp').read_text()
    replacements = {f'WEFTMOE_{phase}_MAIN_{dim}': value
                    for phase in ('FWD', 'BWD')
                    for dim, value in (('HIDDEN', hidden), ('INTERMEDIATE', intermediate))}
    for name, value in replacements.items():
        header = re.sub(r'(#define\s+' + name + r')\s+\d+', rf'\g<1> {value}', header)
    for name, extent in [('DAB', intermediate), ('DX', hidden)]:
        if extent % 512:
            header = re.sub(r'(#define WEFTMOE_BWD_' + name + r'_TILE_N)\s+\d+', r'\g<1> 128', header)
    header = re.sub(r'^#define WEFTMOE_FORWARD_TUNING_ID.*\n?', '', header, flags=re.MULTILINE)
    return header + f'\n#define WEFTMOE_FORWARD_TUNING_ID "{digest(header)}"\n'


def _stamp(path):
    path = Path(path)
    try:
        s = path.stat()
        return (str(path.resolve()), s.st_size, s.st_mtime_ns)
    except FileNotFoundError:
        return (str(path), None, None)


@lru_cache(maxsize=None)
def _dependency_stamps(roots):
    # Header/library changes invalidate the cache, even when versions do not.
    result = []
    for name in roots:
        root = Path(name)
        result.append(_stamp(root))
        if root.is_dir():
            result.extend(_stamp(p) for p in sorted(root.rglob('*'))
                          if p.is_file() and (p.suffix in ('.h', '.hpp', '.cuh', '.inl') or '.so' in p.name))
    return result


def build_context():
    import torch

    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
    env_names = ('CUDA_HOME', 'CUTLASS_INCLUDE_DIR', 'NCCL_INCLUDE_DIR', 'NCCL_LIB_DIR',
                 'CXX', 'NVCC', 'CXXFLAGS', 'NVCCFLAGS', 'LDFLAGS',
                 'WEFTMOE_ENABLE_GIN_RDMA', 'WEFTMOE_DEVICE_TIMELINE', 'WEFTMOE_DEBUG_VALIDATE')
    if 'WEFTMOE_FORWARD_LIGHT_PROGRESS' in os.environ:
        env_names += ('WEFTMOE_FORWARD_LIGHT_PROGRESS',)
    roots = (str(cuda / 'include'), os.environ.get('CUTLASS_INCLUDE_DIR', '/opt/cutlass/include'),
             str(Path(os.environ.get('NCCL_INCLUDE_DIR', '/usr/include')) / 'nccl.h'),
             str(Path(os.environ.get('NCCL_INCLUDE_DIR', '/usr/include')) / 'nccl_device'),
             str(Path(os.environ.get('NCCL_LIB_DIR', '/usr/lib/x86_64-linux-gnu')) / 'libnccl.so'),
             str(Path(torch.__file__).parent / 'include/pybind11'))
    return dict(source=source_digest(), python=sysconfig.get_config_var('SOABI'),
                torch=torch.__version__, cuda=torch.version.cuda,
                arch=torch.cuda.get_device_capability(),
                environment={k: os.environ.get(k, '') for k in env_names},
                compilers=[_stamp(shutil.which(os.environ.get('CXX', 'c++')) or 'c++'),
                           _stamp(shutil.which(os.environ.get('NVCC', str(cuda / 'bin/nvcc'))) or cuda / 'bin/nvcc')],
                dependencies=_dependency_stamps(roots))


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
        json.dump(value, f, sort_keys=True)
        temp = Path(f.name)
    temp.replace(path)


def _ready(directory):
    try:
        record = json.loads((directory / 'ready.json').read_text())
        return all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == value
                   for name, value in record['files'].items()) and bool(record['files'])
    except (OSError, ValueError, KeyError):
        return False


def compile_artifact(header, context, *, ops=False):
    """Immutable artifact, one builder per cache key; crashes cannot publish it."""
    key = digest(dict(context=context, header=header, ops=ops))
    root = cache_root() / 'jit'
    root.mkdir(parents=True, exist_ok=True)
    directory = root / key
    with (root / (key + '.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _ready(directory):
            logger.warning('Compiling WeftMoE %s (first use); cache: %s', 'FP8 helpers' if ops else 'MoE', directory)
            # The library has its own SONAME: loading a second shape must not
            # resolve its dependencies against the first shape's libweftmoe.so.
            soname = f'libweftmoe_{key}.so'
            directory.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=root) as temporary:
                stage = Path(temporary)
                (stage / 'weftmoe_forward_tuning_generated.hpp').write_text(header)
                env = os.environ | {'WEFTMOE_NATIVE_SONAME': soname}
                flags = env.get('NVCCFLAGS', '')
                if 'WEFTMOE_' in flags:
                    raise ValueError('automatic builds own WEFTMOE macros; unset NVCCFLAGS or use an explicit extension')
                env['NVCCFLAGS'] = flags + ' -DWEFTMOE_FORWARD_USE_GENERATED_TUNING=1 ' + shlex.quote('-I' + str(stage))
                root_source = source_root()
                commands = [[sys.executable, str(root_source / 'build_native.py'),
                             'fp8' if ops else 'lib', '--output', str(stage / soname)],
                            [sys.executable, str(root_source / 'tools/build_tensor_runtime.py'),
                             '--library', str(stage / soname), *(['--ops'] if ops else [])]]
                log_path = directory / 'build.log'
                with log_path.open('w') as log:
                    for command in commands:
                        result = subprocess.run(command, cwd=root_source, env=env,
                                                stdout=log, stderr=subprocess.STDOUT)
                        if result.returncode:
                            raise RuntimeError(f'WeftMoE compilation failed; see {log_path}')
                names = [soname, '_runtime' + sysconfig.get_config_var('EXT_SUFFIX')]
                files = {}
                for name in names:
                    files[name] = hashlib.sha256((stage / name).read_bytes()).hexdigest()
                    (stage / name).replace(directory / name)
                _atomic_json(directory / 'ready.json', dict(files=files, context=context, header=header))
    return directory / ('_runtime' + sysconfig.get_config_var('EXT_SUFFIX'))


def prepare(key, tokens, *, batch_invariant=True):
    """1. compiled request cache; 2. exact shape/bucket tune; 3. shape defaults."""
    header = default_header(key['hidden'], key['intermediate'])
    context = build_context()
    if tuple(context['arch']) != (9, 0):
        raise ValueError('WeftMoE automatic compilation currently supports Hopper/SM90 only')
    request = digest(dict(context=context, key=key, tokens=tokens, batch_invariant=batch_invariant,
                          table=str(default_table().resolve())))
    index = cache_root() / 'jit/requests' / (request + '.json')
    if index.exists():
        entry = json.loads(index.read_text())
        path = Path(entry['extension'])
        if _ready(path.parent):
            return entry
    tune = find_entry(default_table(), key, tokens, source=context['source'],
                      canonical_header=header, batch_invariant=batch_invariant)
    selected = tune['header'] if tune else header
    path = compile_artifact(selected, context)
    tuning = {k: tune[k] for k in ('window_tokens', 'contribution_windows')} if tune else {}
    entry = dict(extension=str(path), header_digest=digest(selected), tuning=tuning,
                 origin='tuned' if tune else 'default', batch_invariant=batch_invariant)
    _atomic_json(index, entry)
    return entry


def load(path):
    """Load each native library into a separate Python/ELF namespace once."""
    path = Path(path).resolve()
    with _LOAD_LOCK:
        if path not in _MODULES:
            name = 'weftmoe._jit_' + digest(str(path)) + '._runtime'
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[name] = module
            _MODULES[path] = module
        return _MODULES[path]


@lru_cache(maxsize=1)
def helpers():
    """Shape-independent quantization/GEMM helpers can precede MoE creation."""
    import torch

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('Warm up WeftMoE FP8 helpers before CUDA Graph capture')
    explicit = os.environ.get('WEFTMOE_TENSOR_EXTENSION')
    if explicit:
        return load(explicit)
    context = build_context()
    return load(compile_artifact('#pragma once\n', context, ops=True))
