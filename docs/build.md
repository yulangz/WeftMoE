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
for comparison to the host handle.

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

# Timeout-bounded distributed rejection followed by recovery on the same comm.
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

The benchmark performs 10 warmups and 50 measured iterations. It reports mean,
P50, P95, min, and max for the slowest rank of every iteration, separating CUDA
event mapping time, host mapping wall time, control/count/summary completion,
materialization/staging wall time, and end-to-end wall time. `BENCH_JSON` is the
same result in machine-readable form. Workspace categories include route input,
device arena, pinned staging, NCCL control/data device and pinned slots, and the
logical final metadata resident in the arena (not double-counted in unique
peak). Mapping has no per-route copy, allocation, or synchronization; NCCL slot
allocation occurs at generation setup or after control agreement when the
control-only constructor is selected.

On the inspected local 8-GPU host, the slowest-rank statistics in microseconds
were:

| stage | mean | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| mapping CUDA event | 182.900 | 154.176 | 307.744 | 135.328 | 371.072 |
| mapping host wall | 4702.140 | 4690.630 | 4868.230 | 4553.200 | 4989.240 |
| control completion | 10177.300 | 9419.910 | 15187.500 | 7010.460 | 20758.900 |
| expert-count completion | 380.409 | 384.360 | 483.669 | 230.354 | 666.370 |
| packet-summary completion | 38.005 | 35.905 | 54.462 | 29.307 | 66.489 |
| all collectives | 10382.400 | 9669.850 | 15354.300 | 7183.200 | 20944.100 |
| D2H/materialize/H2D wall | 68310.100 | 67653.700 | 71966.000 | 66581.300 | 72085.200 |
| end-to-end host wall | 90229.500 | 89104.800 | 98051.200 | 86110.200 | 111809.000 |

Peak unique workspace was 23,057,039 bytes: route input 196,608; device
arena 21,361,407; pinned staging 1,445,136; NCCL device slots 26,944; NCCL
pinned slots 26,944. Final device metadata resides in the arena and is not
added twice.

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
