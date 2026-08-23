# Build and verification

The host-only build requires GNU Make, a C++17 compiler, and Python 3. It stays
available without CUDA or NCCL. The optional device slice requires NVCC, CUDA
runtime/development files, NCCL headers, and a runtime NCCL library compatible
with the installed driver/toolkit.

## Observed stage-2 environment

Both the local host and `ssh -p 2222 root@10.10.19.233` were inspected without
changing system configuration:

- 8 NVIDIA L20X GPUs per host, 143771 MiB each; `nvidia-smi topo -m` reports
  `NV18` between every local GPU pair and eight `mlx5_bond_*` NICs;
- driver `570.148.08`, driver-advertised CUDA `12.8`;
- CUDA toolkit/NVCC `12.9`, build `V12.9.86`;
- `nvidia-smi --query-gpu=compute_cap` reports `8.9`, while the CUDA 12.9
  runtime's `cudaGetDeviceProperties` reports `9.0`. The normal binary therefore
  contains both ordinary `sm_89` and `sm_90` cubins;
- system NCCL headers/library report `2.29.7+cuda13.2`, whose
  `ncclCommInitRank` fails here because that runtime requires a newer driver;
- `/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2`
  reports `2.28.3+cuda12.9` and is the compatible runtime used by tests. The
  code is compiled against the system 2.29.7 headers and uses stable APIs only.

An isolated `sm_90a` compile check is supported, but architecture-accelerated
`sm_90a` paths are not selected or claimed runtime-validated on this ambiguous
L20X inventory.

## Normal suite

```bash
make -j test
```

This runs:

- C++ rank-local, resource-lease lifetime/destructor-count, operation state
  machine, device shape/capacity, malicious-backend, context-provenance,
  corruption, descriptor, topology, overflow, and workspace tests;
- independent Python reference tests;
- JSON full-field C++/Python comparison over targeted, fixed-seed random,
  25 fixed random seeds plus targeted 73-node, skewed, empty-rank,
  empty-expert, and partial-M cases; every fixture is rebuilt for byte-identical
  repeat checking.
- every public header as an isolated `-Werror` translation unit.
- nine compile-negative probes for the retired `wait`, `wait_preflight`, raw
  notify start, phase-specific collective starters, and internal NCCL
  operation/backend types.

## Warning-clean build

```bash
make BUILD_DIR=build-werror test \
  CXXFLAGS='-O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror'
```

## ASan/UBSan

```bash
make BUILD_DIR=build-sanitize all \
  CXXFLAGS='-O1 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-omit-frame-pointer'

ASAN_LIB=$(g++ -print-file-name=libasan.so)
UBSAN_LIB=$(g++ -print-file-name=libubsan.so)
LD_PRELOAD="$ASAN_LIB:$UBSAN_LIB" build-sanitize/test_notify_dispatch
LD_PRELOAD="$ASAN_LIB:$UBSAN_LIB" build-sanitize/metadata_dump >/dev/null
make BUILD_DIR=build-sanitize reference-test
make BUILD_DIR=build-sanitize cross-check
```

## Optional CUDA/NCCL build and local tests

```bash
NCCL12=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib

make BUILD_DIR=build-cuda -j cuda-lib
make BUILD_DIR=build-cuda -j \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-test-local

# Include the one-GPU CUDA slice in the default host suite.
make BUILD_DIR=build-cuda ENABLE_CUDA=1 \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" test

# Compile-only architecture check; do not infer runtime coverage.
make BUILD_DIR=build-cuda cuda-sm90-compile
```

`cuda-test-local` runs real NCCL worlds 1, 2, and 8. Each world runs 25 fixed
seeds, a 20-run byte-for-byte repeat fixture, staggered rank enqueue timing,
different per-rank CUDA streams, and controlled download of every device field
for comparison to the host handle. World 2 additionally runs five independent
20-round invalid-to-valid recovery categories on one communicator, config-wire
corruption from rank 0 and rank 1, and exact raw-session identity mismatches
with equal display names/header summaries. World 1 runs the nine-point enqueue
audit/lifecycle table and a self-corruption integrity check.

## Real two-host NCCL

Build or copy the same clean-export binary to both hosts, then run:

```bash
PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
  --socket-ifname '=eth0' --disable-ib

PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 8 \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
  --socket-ifname '=eth0' --disable-ib

# Timeout-bounded 20-round distributed rejection/recovery on the same comm.
PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 --scenario out_of_range \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
  --socket-ifname '=eth0' --disable-ib
```

The launcher starts rank 0 locally and keeps the NCCL bootstrap owner alive,
then starts the remaining local and remote ranks. `--remote` is mandatory.
Library directory, socket interface, and IB disabling are optional and inherit
the process/remote environment unless explicitly supplied. The values above are
an example for the inspected environment, not launcher defaults. Arguments are
shell-quoted as individual remote `env` entries; timeout/error cleanup
terminates every still-owned local SSH/process handle and the launcher creates
no remote temporary file. It compares each valid 2/16-rank result against the
Python oracle. Every rank also reports header version, `ncclGetVersion` runtime
version, `dladdr`-resolved library path, and active network overrides.

Explicit preload is required in the current container because another injected
runtime otherwise precedes ASan.

CUDA memory checking uses valid and invalid device mapping kernels without
instrumenting the third-party NCCL library (this NCCL binary reports
`cudaErrorNoKernelImageForDevice` under Compute Sanitizer although it runs
normally):

```bash
make BUILD_DIR=build-cuda \
  NCCL_RUNTIME_LIB_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
  cuda-sanitizer-smoke

# When available, the same smoke command can be run with:
compute-sanitizer --tool racecheck --error-exitcode=1 \
  build-cuda/test_cuda_notify --device-smoke
compute-sanitizer --tool initcheck --error-exitcode=1 \
  build-cuda/test_cuda_notify --device-smoke
```

## Notify performance baseline

The benchmark is intentionally a correctness baseline, not a GIN/RDMA result:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib \
  build-cuda/test_cuda_notify --bench 8
```

The benchmark creates one session per rank once, reuses it for 10 warmups and
50 measured iterations, and reports mean, P50, P95, min, and max. Names
distinguish one-time session setup, per-generation coordinator/backend and
control/config/count/summary slot setup, and steady-state execution.
`BENCH_JSON` carries the same categories. Workspace reporting includes the
route input, unique notify device arena and pinned staging, 256-byte control
wire/slots, canonical config wire/slots, count/summary slots, final-metadata
subview, and event count. Shared-arena subviews are not double-counted.
Setup-inclusive E2E is calculated per rank as `rank_setup + rank_steady` before
choosing the slowest rank. Mapping has no per-route copy, allocation, or
synchronization.

On the inspected local 8-GPU L20X host, the refactored coordinator baseline
produced these slowest-rank statistics (microseconds, 10 warmups/50 measured):

| stage | mean | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| one-time NCCL session setup | 1717800 | 1710350 | 1757600 | 1681640 | 1757600 |
| per-generation setup | 8664.04 | 5782.74 | 25916.2 | 5212.38 | 34520.4 |
| setup-inclusive generation end-to-end | 105863 | 96360.8 | 167391 | 93290.3 | 204228 |
| steady-state execution | 97963.7 | 90727.2 | 162027 | 88618.6 | 178385 |
| config slot allocation | 1186.47 | 1184.98 | 1337.73 | 947.467 | 1453.19 |
| count slot allocation | 605.592 | 602.807 | 632.688 | 574.326 | 653.741 |
| summary slot allocation | 667.029 | 655.960 | 707.693 | 627.258 | 928.650 |

The corresponding unique peak workspace was 21,617,215 device bytes and
1,504,336 pinned bytes: route input 196,608; notify arena 21,361,407; notify
pinned staging 1,445,136; control slot 4,608; config wire 3,584; config slot
64,512; expert slot 49,152; summary slot 128. Final metadata is a 21,361,407-byte
arena view and therefore is not added again. Seven CUDA events are reported;
their opaque runtime allocation size is recorded as zero.

The earlier approximately 197 ms number is not a comparable regression point:
that runner rebuilt communicators every iteration and computed steady time by
subtracting independently selected slowest-rank maxima. The corrected 98.0 ms
mean uses one reused session and same-rank accounting, which is consistent with
the earlier approximately 90 ms correctness baseline. No mapping or
materialization algorithm was changed for this correction.

## Clean export

After committing, verify without ignored build state:

```bash
EXPORT_DIR=$(mktemp -d)
git archive HEAD | tar -x -C "$EXPORT_DIR"
make -C "$EXPORT_DIR" -j test
```

Run the same archive on a remote machine when its SSH endpoint is available.
Remote execution is a separate result and must not be inferred from local
shared storage or a successful local build.
