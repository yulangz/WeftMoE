"""WeftMoE's CUTLASS/CuTe routed forward, specialization cache and NVLink workspace."""

import fcntl
import hashlib
import json
import math
import os
import subprocess
import tempfile
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import torch

from .jit import source_root
from .tuning import cache_root


def cutlass_include():
    path = Path(os.environ.get("CUTLASS_INCLUDE_DIR", source_root() / "third_party/cutlass/include"))
    if not (path / "cute/tensor.hpp").is_file():
        raise RuntimeError(
            "Initialize third_party/cutlass or set CUTLASS_INCLUDE_DIR to its include directory"
        )
    return path.resolve()


@cache
def _toolchain():
    from torch.utils.cpp_extension import CUDA_HOME

    if not CUDA_HOME:
        raise RuntimeError("CUDA_HOME must point to a CUDA toolkit with nvcc")
    cuda = Path(CUDA_HOME)
    src = source_root() / "src/cuda/blackwell"
    include = cutlass_include()
    version = subprocess.check_output([cuda / "bin/nvcc", "--version"])
    digest = hashlib.sha256(version)
    for root in (src, include):
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(os.fsencode(path.relative_to(root)))
                digest.update(path.read_bytes())
    return src, cuda, include, digest.hexdigest()[:24]


@cache
def native():
    from torch.utils.cpp_extension import load

    src, cuda, include, version = _toolchain()
    return load(
        name=f"weft_cutlass_host_{version}",
        sources=[str(src / "fused_host.cpp")],
        extra_include_paths=[str(include), str(src), str(cuda / "include"), str(cuda / "include/cccl")],
        extra_ldflags=[f"-L{cuda}/lib64/stubs", "-lcuda"],
        extra_cflags=["-O2", "-std=c++20"],
    )


@cache
def grouped_extension(kind, major, minor):
    from torch.utils.cpp_extension import load

    src, _, include, version = _toolchain()
    arch = f"{major}{minor}a"
    return load(
        name=f"weftmoe_{kind}_grouped_sm{arch}_{version}",
        sources=[str(src / f"{kind}_grouped.cu")],
        extra_include_paths=[str(include)],
        extra_cuda_cflags=[
            "--expt-relaxed-constexpr",
            "-std=c++17",
            "-O3",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ],
    )


def tile_m(ranks, experts, rows, topk):
    expected = rows * ranks * topk / experts
    if expected <= 24:
        return 16 if expected <= 10 else 32
    covered = expected + math.sqrt(expected)
    blocks = math.ceil(covered / 240)
    if blocks == 1 and covered > 192 and experts // ranks < 14:
        blocks = 2
    return next(m for m in (64, 128, 192, 240) if blocks * m >= covered)


DECODE_TILES = (16, 32, 64, 128, 192, 240)


def decode_key(sms, ranks, experts, hidden, intermediate, topk, precision):
    return f"sm103-{sms}-ep{ranks}-e{experts}-h{hidden}-i{intermediate}-k{topk}-{precision}"


def load_decode_plan(key, override=None):
    if override is None:
        with (source_root() / "configs/profiles/sm103_decode.json").open() as file:
            override = json.load(file).get(key, {})
        extra = os.environ.get("WEFTMOE_SM103_DECODE_PROFILE")
        if extra:
            with open(extra) as file:
                override.update(json.load(file).get(key, {}))
    plan = {}
    for rows, (m, static) in override.items():
        if (str(int(rows)) != str(rows) or int(rows) < 1 or type(m) is not int
                or m not in DECODE_TILES or type(static) is not bool):
            raise ValueError(f"Invalid DT tuning for {rows} tokens: {(m, static)}")
        plan[int(rows)] = (m, static)
    return plan


def decode_config(ranks, experts, rows, topk):
    return tile_m(ranks, experts, rows, topk), rows * ranks * topk <= 24 * experts


@cache
def _kernel(device, fp4, cap, h, i, experts, topk, ranks, ring, sf_ring, sms, m, save, exact, decode_capacity, single_cta, static_raster, tiled_weights=False):
    src, cuda, include, version = _toolchain()
    arch = "".join(map(str, torch.cuda.get_device_capability(device))) + "a"
    k = (256 if i % 256 == 0 else 128) if fp4 else 64
    store = 8 if m == 16 else 16 if m <= 64 else 32 if m <= 192 else 24 if fp4 else 40
    store2 = 8 if fp4 and m == 240 else store
    sf = 128 * math.ceil(m / 128)
    pull = h // 2 if fp4 else 2 * h
    while pull > (8192 if fp4 else 4096):
        pull //= 2

    def align_smem(n):
        return (n + 1023) // 1024 * 1024

    # Match the persistent kernel's SharedStorage. Four dispatch warps and two epilogue
    # warpgroups share staging space across FC1 and FC2. Barriers and task slots are fixed.
    dispatch = align_smem(experts * 4) + align_smem(pull * 4)
    epilogue = align_smem(max(2 * store * 64 * 2 * (1 if fp4 else 2), 2 * store2 * 128 * 2))
    fixed = dispatch + epilogue + (store * 8 * 4 if fp4 else 0) + 28 * 8 + 2 * native().task_bytes + 4
    # One pipeline stage holds half the A tile, one B tile, and the full/empty barriers.
    stage = align_smem(m // 2 * k // 2 if fp4 else m * k) + (128 * k // 2 if fp4 else 256 * k) + 16
    if fp4:
        stage += (sf + 128) * (k // 16)
    stages = (232448 - fixed) // stage
    if stages < 2:
        raise ValueError("Insufficient shared memory for the requested MoE shape")
    smem = fixed + stages * stage
    name = "routed_nvfp4" if fp4 else "routed_bf16"
    args = [cap, h, i, experts, topk, m, 128, k, store]
    if fp4:
        args.extend([store2, sf, 128])
    args.extend(
        [
            ring,
            *([sf_ring] if fp4 else []),
            stages,
            pull,
            128,
            128,
            256,
            sms,
            ranks,
            "__builtin_inff()",
            "true",
        ]
    )
    args.append(int(save))
    if fp4:
        args.append(int(exact))
    args.extend([decode_capacity, int(single_cta), int(static_raster), int(tiled_weights)])
    code = (
        f'#include "fused/impls/{name}.cuh"\n'
        f"void instantiate() {{ [[maybe_unused]] auto kernel = &weftmoe::sm100::{name}<{','.join(map(str, args))}>; }}\n"
    )
    directory = cache_root() / "blackwell" / version / arch
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (hashlib.sha256(code.encode()).hexdigest() + ".cubin")
    if not target.exists():
        with target.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not target.exists():
                with tempfile.TemporaryDirectory(dir=directory) as tmp:
                    cu = Path(tmp) / "kernel.cu"
                    cu.write_text(code)
                    binary = cu.with_suffix(".cubin")
                    subprocess.run(
                        [
                            cuda / "bin/nvcc",
                            "--cubin",
                            f"-arch=sm_{arch}",
                            "-std=c++20",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "-O3",
                            "-I",
                            include,
                            "-I",
                            src,
                            cu,
                            "-o",
                            binary,
                        ],
                        check=True,
                    )
                    os.replace(binary, target)
    return native().Kernel(str(target), smem, sms, tiled_weights), k, store


class ExpertWorkspace:
    def __init__(self, group, experts, capacity, topk, hidden, intermediate, *, fp4=False, compute_sms=None, decode_only=False, decode_plan=None):
        import torch.distributed._symmetric_memory as symm

        self.group, self.num_experts, self.num_topk = group, experts, topk
        self.hidden, self.intermediate_hidden = hidden, intermediate
        self.fp4 = fp4
        self.device = torch.cuda.current_device()
        device_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.sms = device_sms if compute_sms is None else compute_sms
        self.decode_capacity = capacity if decode_only else 0
        self.decode_tuning = decode_plan or {}
        self.max_decode_m = 0
        if decode_only:
            # All ranks allocate the same single-window layout. Each owner can
            # select a smaller M on replay, even when the source lengths differ.
            self.max_decode_m = max(
                [tile_m(group.size(), experts, capacity, topk)]
                + [m for n, (m, _) in self.decode_tuning.items() if n <= capacity]
            )
        layout = self.layout = native().Workspace(
            group.size(), experts, capacity, topk, hidden, intermediate, self.sms, fp4, self.max_decode_m
        )
        cap = self.num_max_tokens_per_rank = layout.capacity
        allocator = torch if group.size() == 1 else symm
        self.buffer = allocator.empty(layout.bytes, dtype=torch.int8, device=self.device)
        self.handle = (
            SimpleNamespace(buffer_ptrs=[self.buffer.data_ptr()])
            if group.size() == 1
            else symm.rendezvous(self.buffer, group=group)
        )
        self.buffer.zero_()
        # Non-localized weights use a balanced two-domain assignment per pair of SMs.
        self.domains = ((torch.arange(device_sms, device=self.device) // 2) % 2).to(torch.uint8)
        group.barrier().wait()
        torch.cuda.synchronize(self.device)

        def view(idx, dtype, cols):
            offset = layout.offsets[idx]
            size = dtype.itemsize * cap * cols
            return self.buffer[offset : offset + size].view(dtype).view(cap, cols)

        self.x = view(0, torch.uint8, hidden // 2) if fp4 else view(0, torch.bfloat16, hidden)
        self.x_sf = view(1, torch.int32, hidden // 64) if fp4 else None
        self.topk_idx = view(2, torch.int64, topk)
        self.topk_weights = view(3, torch.float32, topk)
        self.peers = self.handle.buffer_ptrs

    def destroy(self):
        self.handle = self.buffer = self.x = self.x_sf = self.topk_idx = self.topk_weights = None


def launch(y, first, second, workspace, *, saved=(), tensor_scales=(1.0, 1.0, 1.0)):
    b = workspace
    rows = y.size(0)
    if b.decode_capacity:
        config = b.decode_tuning.get(rows) or decode_config(b.group.size(), b.num_experts, rows, b.num_topk)
        m, static_raster = config
        m = min(m, b.max_decode_m)
    else:
        m = tile_m(b.group.size(), b.num_experts, rows, b.num_topk)
        static_raster = False
    tiled_weights = (first[0] if b.fp4 else first).ndim == 5
    kernel, k, store = _kernel(
        b.device,
        b.fp4,
        b.num_max_tokens_per_rank,
        b.hidden,
        b.intermediate_hidden,
        b.num_experts,
        b.num_topk,
        b.group.size(),
        b.layout.ring,
        b.layout.sf_ring,
        b.sms,
        m,
        bool(saved),
        b.fp4 and math.frexp(tensor_scales[1])[0] != 0.5,
        b.decode_capacity,
        bool(b.decode_capacity and rows * b.num_topk <= 128),
        static_raster,
        tiled_weights,
    )
    weights = (
        [v.data_ptr() for v in (*first, *second)] if b.fp4 else [first.data_ptr(), 0, second.data_ptr(), 0]
    )
    kernel.run(
        b.layout,
        b.peers,
        b.group.rank(),
        rows,
        m,
        k,
        store,
        y.data_ptr(),
        weights,
        [first[1].stride(-1), second[1].stride(-1)] if b.fp4 else [0, 0],
        [v.data_ptr() for v in saved] if saved else [0] * 6,
        b.domains.data_ptr(),
        tensor_scales,
        torch.cuda.current_stream(b.device).cuda_stream,
    )
