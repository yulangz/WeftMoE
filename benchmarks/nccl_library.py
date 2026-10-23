"""Report the NCCL runtime loaded by the benchmark process."""

import ctypes
import os


def loaded_nccl_libraries():
    with open("/proc/self/maps", encoding="utf-8") as mappings:
        return sorted({line.rsplit(" ", 1)[-1].strip()
                       for line in mappings if "libnccl.so" in line})


def nccl_runtime_version():
    library = ctypes.CDLL(os.environ.get("WEFTMOE_NCCL_LIBRARY", "libnccl.so.2"))
    version = ctypes.c_int()
    status = library.ncclGetVersion(ctypes.byref(version))
    if status:
        raise RuntimeError(f"ncclGetVersion failed: {status}")
    value = version.value
    return f"{value // 10000}.{(value % 10000) // 100}.{value % 100}"
