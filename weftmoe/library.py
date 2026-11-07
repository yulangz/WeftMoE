"""Locate installed native resources without initializing CUDA or NCCL."""
from pathlib import Path


def get_include() -> str:
    path = Path(__file__).resolve().parent / 'include'
    if not path.is_dir():
        from .jit import source_root
        path = source_root() / 'include'
    return str(path)


def get_library_path() -> str:
    path = Path(__file__).resolve().parent / 'lib/libweftmoe.so'
    if not path.is_file():
        raise RuntimeError('No model-independent native library. Use TensorRuntime for JIT or build_native.py for an explicit library.')
    return str(path)
