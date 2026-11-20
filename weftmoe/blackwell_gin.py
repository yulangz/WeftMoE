"""Device-initiated inter-node slices across NCCL's rail team."""

import ctypes
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist


@cache
def extension():
    from torch.utils.cpp_extension import load
    from .jit import source_root

    # Resolve the SONAME binding. PyTorch can additionally dlopen its wheel's
    # absolute filename even when a matching NCCL was preloaded by the caller.
    class DlInfo(ctypes.Structure):
        _fields_ = [
            ("path", ctypes.c_char_p),
            ("base", ctypes.c_void_p),
            ("symbol", ctypes.c_char_p),
            ("address", ctypes.c_void_p),
        ]

    info = DlInfo()
    symbol = ctypes.CDLL("libnccl.so.2").ncclGetVersion
    if not ctypes.CDLL(None).dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)):
        raise RuntimeError("Cannot resolve the loaded NCCL SONAME")
    lib = Path(info.path.decode()).resolve()
    include = lib.parent.parent / "include"
    if not (include / "nccl_device.h").exists():
        raise RuntimeError(f"Missing NCCL device headers in {include}")
    major, minor = torch.cuda.get_device_capability()
    return load(
        name=f"weftmoe_gin_{major}{minor}",
        sources=[str(source_root() / "src/cuda/blackwell/gin.cu")],
        extra_include_paths=[str(include)],
        extra_cuda_cflags=["-std=c++17", f"-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}"],
        extra_ldflags=[str(lib)],
        verbose=False,
    )


class RailWindow:
    def __init__(self, group, size, nodes, stripes=16):
        device = torch.cuda.current_device()
        backend = group._get_backend(torch.device("cuda", device))
        self.group = group
        self.nodes = nodes
        self.compact = False
        try:
            module = extension()
            comm = backend._comm_ptr()
            status = module.rail_backend(comm, nodes)
        except (ImportError, OSError, RuntimeError) as exc:
            status = str(exc)
        statuses = [None] * group.size()
        dist.all_gather_object(statuses, status, group=group)
        if status not in (2, 3) or any(peer != status for peer in statuses):
            raise RuntimeError(
                f"EP ranks require matching NCCL rail GIN support. Set cuMem, HCA selection "
                f"and NIC merging before creating the communicator. Ranks reported {statuses}"
            )
        self.backend = "GDAKI" if status == 3 else "IB_PROXY"
        self.handle = module.RailWindow(comm, size, nodes, stripes)
        # The handle owns the NCCL VMM allocation until collective close.
        self.__cuda_array_interface__ = dict(
            shape=(size,), strides=None, typestr="|u1", data=(self.handle.address, False), version=3
        )
        self.data = torch.as_tensor(self, device=torch.device("cuda", device))

    def exchange(self, source, destination, stride, sizes, *, broadcast):
        if (
            sizes.dtype != torch.int64
            or sizes.shape != (self.nodes,)
            or not sizes.is_contiguous()
            or sizes.device != self.data.device
        ):
            raise ValueError("GIN requires one contiguous device int64 byte count per node")
        self.handle.exchange(
            source,
            destination,
            stride,
            sizes.data_ptr(),
            broadcast,
            torch.cuda.current_stream().cuda_stream,
            self.compact,
        )

    def close(self):
        self.data = None
        self.handle.close()
