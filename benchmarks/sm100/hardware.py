"""Check the CUDA runtime, which can report a different device name than NVML."""

import torch
import torch.distributed as dist


def check_blackwell():
    major, minor = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError(f"Expected datacenter Blackwell (SM 10.x), got SM {major}.{minor}")


def check_finite(*tensors):
    finite = torch.stack([torch.isfinite(tensor).all() for tensor in tensors]).all().int()
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise RuntimeError("Non-finite MoE output or gradient")
