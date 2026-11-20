"""Blackwell tile policies in the shared offline tuning table."""
from functools import lru_cache
import hashlib
from pathlib import Path

from .tuning import default_table, match_entry

STRICT_BLOCK_M = 128
SUPPORTED_BLOCK_M = (16, 32, 64, 128, 192, 240)


@lru_cache(maxsize=1)
def source_digest():
    """Invalidate a measured choice when its runtime, JIT kernel or native library changes."""
    from deep_gemm import _C
    from .jit import source_root

    here = Path(__file__).resolve().parent
    native = Path(_C.__file__).resolve()
    paths = [native, here / "runtime.py", *here.glob("blackwell*.py"), *here.glob("nvfp4*.py"),
             *(source_root() / "src/cuda/blackwell").glob("*"),
             *(native.parent / "include/deep_gemm").rglob("*.cuh")]
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: (p.parent.name, p.name)):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def tile_policy(key, capacity, override):
    strict = key["batch_invariant"]
    if override is not None:
        if type(override) is not int or override not in SUPPORTED_BLOCK_M:
            raise ValueError(f"blackwell_block_m must be one of {SUPPORTED_BLOCK_M}")
        if strict and override != STRICT_BLOCK_M:
            raise ValueError("Noncanonical blackwell_block_m requires batch_invariant=False")
        return override
    path = default_table()
    entry = match_entry(path, key, capacity, source_digest()) if path.exists() else None
    if entry is None:
        return STRICT_BLOCK_M if strict else 0
    m = entry["fixed_block_m"]
    if (entry.get("validated") is not True or entry.get("batch_invariant") is not strict or
            type(m) is not int or m not in (0, *SUPPORTED_BLOCK_M) or (strict and m != STRICT_BLOCK_M)):
        raise ValueError("Invalid Blackwell tile policy in autotune table")
    return m
