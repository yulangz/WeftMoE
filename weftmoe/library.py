"""Locate installed native resources without initializing CUDA or NCCL."""
from pathlib import Path


def get_include() -> str:
    path = Path(__file__).resolve().parent / 'include'
    if not path.is_dir():
        from .jit import source_root
        path = source_root() / 'include'
    return str(path)


def get_library_path() -> str:
    """Legacy prebuilt package library only; lazy wheels have no such library.

    For C++ callers, build an explicit library with build_native.py. TensorRuntime
    and `python -m weftmoe.jit prepare` discover/build shape-specific JIT bindings.
    """
    path = Path(__file__).resolve().parent / 'lib/libweftmoe.so'
    if not path.is_file():
        raise RuntimeError('No model-independent native library. Use TensorRuntime for JIT or build_native.py for an explicit library.')
    return str(path)
