"""Locate installed native resources without initializing CUDA or NCCL."""
from pathlib import Path


def get_include() -> str:
    path = Path(__file__).resolve().parent / 'include'
    if not path.is_dir():
        raise RuntimeError('WeftMoE headers are not installed; run pip install .')
    return str(path)


def get_library_path() -> str:
    path = Path(__file__).resolve().parent / 'lib/libweftmoe.so'
    if not path.is_file():
        raise RuntimeError('WeftMoE native library is not built; run pip install .')
    return str(path)
