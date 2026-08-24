# Build and verification

The host-only build requires GNU Make, a C++17 compiler, and Python 3. It stays
available without CUDA or NCCL. The optional device slice requires NVCC, CUDA
runtime/development files, NCCL headers, and a runtime NCCL library compatible
with the installed driver/toolkit.

## Observed stage-4 environment and transport capability

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
- `/usr/include/nccl_device/gin.h` (2.29.7) exposes GIN put, remote indexed/VA
  signals, local counters, flush/wait, and `ncclTeamRail`. The selected 2.28.3
  runtime does not export `ncclGinOutboxCreateRequirement` or
  `ncclGinInboxA2ACreateRequirement`; the project reports `gin_usable=0` and
  never launches that backend.
- all 64 entries of the local 8x8 CUDA peer-access matrix are enabled. The
  corrected 64 MiB probe uses multi-block completion, last-block signaling and
  independent destination-stream validation; its early-signal negative seam is
  required to fail. With 5 warmups/20 measurements it reported 299.852 GB/s
  locally and 299.834 GB/s remotely for GPU0-to-GPU1. `nvidia-smi topo -m`
  reports NV18 between every pair.
- eight active `mlx5_bond_0..7` RoCE devices exist on each host. Observed ports
  use Ethernet/RoCE, active MTU 4096, and expose link-local plus IPv4-mapped
  RoCE v1/v2 GIDs. `ibv_devinfo` reports HCA atomics and ODP read/write/atomic.
  An HCA alone does not make the NCCL 2.28 runtime GIN-capable.

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
- twenty-six compile-negative probes for the retired `wait`, `wait_preflight`, raw
  notify start, phase-specific collective starters, internal NCCL
  operation/backend types, bootstrap hex encode/parse APIs, and the unpooled
  production coordinator constructor, generic route rebinding/wrong route
  dtype, raw dispatch views, removed generation handoff/reissue helpers, and
  the internal no-wait producer-race seam.

The activation-specific additions reject wrong activation dtype, generic
buffer rebinding, raw peer-pointer extraction, access to the private IPC
factory, and the retired caller-selected backend-enum entry.
- thirteen fake-runner/bootstrap-channel security groups covering success,
  local/remote failure, timeout and TERM-to-KILL escalation, SSH start failure,
  short/oversize bootstrap, runner exception, every 1/2/7/8/17/127/128-byte
  prefix representation, Python poison writers, top-level CLI output, and 20
  failure/recovery cycles. They inspect argv/environment, exact frame content,
  FD/process cleanup, scrubbed diagnostics, and absence of bootstrap files.

## Warning-clean build

```bash
make BUILD_DIR=build-werror test \
  CXXFLAGS='-O2 -g -std=c++17 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror'
```

The equivalent maintained shorthand is `make BUILD_DIR=build-werror
WERROR=1 test`; `WERROR=1` appends `-Werror` to host compilation and the NVCC
host-compiler flags. The emitted compiler commands make the effective warning
policy directly auditable.

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
NCCL12=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat

make BUILD_DIR=build-cuda -j cuda-lib
make BUILD_DIR=build-cuda -j \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-test-local

# Include the one-GPU CUDA slice in the default host suite.
make BUILD_DIR=build-cuda ENABLE_CUDA=1 \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" test

# Compile-only architecture check; do not infer runtime coverage.
make BUILD_DIR=build-cuda cuda-sm90-compile

# Auditable transport capability, correctness/sanitizer path and benchmark.
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --transport-capability
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --device-smoke
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --transport-bench
```

`cuda-test-local` runs real NCCL worlds 1, 2, and 8. Each world runs 25 fixed
seeds, a 20-run byte-for-byte repeat fixture, staggered rank enqueue timing,
different per-rank CUDA streams, and controlled download of every device field
for comparison to the host handle. World 2 additionally runs five independent
20-round invalid-to-valid recovery categories on one communicator, config-wire
corruption from rank 0 and rank 1, and exact raw-session identity mismatches
with equal display names/header summaries. World 1 runs the nine-point enqueue
audit/lifecycle table, three pooled count/summary/materialization completion
faults, real cancel/deadline recovery, and a self-corruption integrity check.
The default CUDA suite also runs the closed-loop fake-collective materializer:
explicitly initialized device count/summary matrices enter the exact production
prefix/materialization kernels and arena layout, then every field is downloaded
and compared to the oracle. Its 73-node/one-rank-per-node fixture exercises a
two-word mask, token-node dedup, same-rail proxying, fan-out, both signal
directions, 20 byte-identical repeats, an invalid expert, and a workspace plan
shortened by exactly one byte.

## Real two-host NCCL

Build or copy the same clean-export binary to both hosts, then run:

```bash
PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat \
  --socket-ifname '=eth0' --disable-ib

PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 8 \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat \
  --socket-ifname '=eth0' --disable-ib

# Timeout-bounded 20-round distributed rejection/recovery on the same comm.
PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote root@10.10.19.233 --port 2222 \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 --scenario out_of_range \
  --nccl-lib-dir /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat \
  --socket-ifname '=eth0' --disable-ib
```

The launcher starts rank 0 locally and receives its fixed 128-byte binary NCCL
bootstrap through an inherited private pipe. Remaining local ranks receive the
same bytes through dedicated stdin pipes; remote ranks receive them only over
SSH stdin. No bootstrap text is emitted and the secret never appears in argv,
environment variables, the remote shell command, process listings, or a
temporary file. `--remote` is mandatory.
Library directory, socket interface, and IB disabling are optional and inherit
the process/remote environment unless explicitly supplied. The values above are
an example for the inspected environment, not launcher defaults. Arguments are
shell-quoted as individual remote `env` entries. Captured stdout/stderr,
exceptions, and diagnostic tails are scrubbed for raw, hex, base64, and the
retired bootstrap marker. Each invocation adds a random `--launch-id`;
timeout/error cleanup terminates owned local handles and sends targeted
TERM/KILL cleanup for only that ID on the remote host. The launcher creates no
remote temporary file. It compares each valid 2/16-rank result against the
Python oracle. Every rank also reports header version, `ncclGetVersion` runtime
version, `dladdr`-resolved library path, and active network overrides.

The inspected hosts also require the CUDA 12.9 compatibility-driver directory
in `LD_LIBRARY_PATH`; the colon-separated `--nccl-lib-dir` example passes both
directories explicitly to local and remote ranks without changing the system.

CUDA memory checking uses valid and invalid device mapping kernels without
instrumenting the third-party NCCL library (this NCCL binary reports
`cudaErrorNoKernelImageForDevice` under Compute Sanitizer although it runs
normally):

```bash
make BUILD_DIR=build-cuda \
  NCCL_RUNTIME_LIB_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat \
  cuda-sanitizer-smoke
```

`cuda-sanitizer-smoke` runs memcheck, racecheck, and initcheck over that same
closed-loop 73-node path. All three tools were available locally and reported
0 errors/0 hazards. The remote image provides the executable at
`/usr/local/cuda/bin/compute-sanitizer` but does not include that directory in
the default `PATH`; explicit invocation from a clean export also reported
0 errors/0 hazards for all three tools. Host ASan/UBSan and world 1/2/8 ran
from the same clean export.

For the real two-process IPC test, `compute-sanitizer --target-processes all`
preserves the one-shot bootstrap FD and the full Python oracle still passes,
but NCCL 2.28.3 itself emits eight `cudaErrorNoKernelImageForDevice` reports
from `ncclInitKernelsForDevice`; therefore that invocation exits nonzero and is
not represented as a clean MegaMOE sanitizer result. Wrapping each rank
separately is also invalid because the wrapper retains the inherited bootstrap
write FD and prevents exact-frame EOF. The checked single-process two-GPU
ordering path plus all MegaMOE device simulation/materialization kernels are
covered by memcheck/racecheck/initcheck with zero reports; no NCCL suppression
is installed.

The transport portion of `--device-smoke` covers the device simulation oracle
and ordering probe. Real independent-process local P2P is separate and cannot
be silently replaced by simulation:

```bash
make BUILD_DIR=build-cuda NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-p2p-test
make BUILD_DIR=build-cuda NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-p2p-bench
```

`cuda-p2p-test` runs real world 2 and 8 processes, 20 byte-identical rounds,
and compares combined output, every route contribution, and every canonical
expert row to `tests/run_local_activation_p2p.py`, which independently
reimplements mapping and BF16 reduction.

## Activation transport performance baseline

### Real local CUDA P2P (current)

`cuda-p2p-bench` launches eight independent rank processes with real CUDA IPC
mappings. For `tokens/rank=8192`, `hidden=7168`, `top-k=6`, it uses 10 warmups
and 100 measured iterations in the release build. Each iteration is reduced to
the slowest rank before statistics are computed. Relay and synthetic return are
one fused destination phase.

| span | mean (us) | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| direct dispatch put | 1842.53 | 1842.46 | 1844.51 | 1840.13 | 1846.14 |
| fused relay + synthetic return | 2126.12 | 2125.70 | 2153.18 | 2094.18 | 2162.98 |
| expert-order FP32 reduce | 488.59 | 488.54 | 489.47 | 487.68 | 489.98 |
| transport steady total | 4473.00 | 4472.29 | 4499.26 | 4439.52 | 4512.38 |

Slowest-rank one-time pool/backend setup was 1391350 us. Each rank's single-slot
pool owns 3,092,974,032 device bytes and 208 pinned bytes. Logical six-route
traffic is 704,643,072 bytes/rank and logical node-dedup traffic is 117,440,512
bytes/rank, but actual remote-GPU fan-out is 616,562,688 bytes in dispatch plus
616,562,688 bytes in return per rank. Aggregate bidirectional effective peer
bandwidth is 2205.46 GB/s (275.68 GB/s per rank). Credit stalls are zero because
the trusted release path uses phase-batched completion instead of the debug
ring. `P2P_BENCH_JSON` reports all values separately.

The prior one-CTA serialized baseline was 426695.44 us mean. Bulk all-SM grids,
direct canonical writes and per-peer phase signals reduce it by 98.95% (about
95.4x). This remains CUDA IPC/NVLink—not GIN/RDMA or Socket transport.

### Megatron-Core comparable baseline

`benchmarks/megatron_dispatch_baseline.py` runs Megatron-Core 0.17.0's actual
`MoEAlltoAllTokenDispatcher` with the same world, BF16 shape, routing pattern
and synchronization boundary. Its expert body is identity so the comparison is
transport-only; `intermediate=3072` is recorded for the future FC benchmark.
The installed PyTorch backend reports NCCL 2.27.5 while MegaMOE loads NCCL
2.28.3; both versions and commands are emitted rather than hidden.

For 10 warmups and 100 samples, routing/local permutation is 0.859 ms mean,
dispatch AlltoAll 1.940 ms, combine 4.002 ms, and the identity-expert transport
roundtrip 6.896 ms (P50 6.890, P95 6.937). MegaMOE's 4.473 ms synthetic
transport is 35.1% faster at this boundary, exceeding the initial 15% target.
This is not yet a full FC1/SwiGLU/FC2 forward/backward comparison.

The same script also runs as a two-node `torchrun` job; it intentionally uses
the distributed world rather than `LOCAL_WORLD_SIZE` when constructing the EP
group. With 8 ranks/node, `NCCL_SOCKET_IFNAME=eth0`, `NCCL_IB_DISABLE=1`, 10
warmups and 100 samples, the measured world-16 means were 0.902 ms routing,
463.267 ms dispatch, 464.664 ms combine, and 919.151 ms identity transport
roundtrip (P50 926.951, P95 1073.007 ms). The corresponding estimated payload
rate was about 1.426 GB/s/direction/rank. This is a real Megatron cross-host
Socket-NCCL baseline; MegaMOE does not yet implement cross-host activation
transport, so no cross-host MegaMOE speedup is claimed.

Example two-node invocation (run once per node with node rank 0/1):

```bash
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  benchmarks/megatron_dispatch_baseline.py
```

### Device simulation (previous comparison point)

`--transport-bench` uses `tokens/rank=8192`, `hidden=7168`, `top-k=6`, a
logical eight-node descriptor topology, 10 warmups and 50 samples. It measures
the transport-only device-simulation worker, not GIN/RDMA/network bandwidth.

| stage | mean (us) | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| dispatch + relay + synthetic return | 75719.0 | 75717.8 | 75779.6 | 75642.5 | 75789.8 |
| deterministic combine reduce | 4387.79 | 4207.33 | 4841.02 | 4032.13 | 4841.18 |
| steady transport total | 80121.6 | 79941.7 | 80547.2 | 79838.2 | 80555.8 |

One-time transport-pool setup was 234.361 us. The single-slot pool owns
2,349,930,104 device bytes and 120 pinned bytes. The fixture deliberately puts
six selected experts on one destination node: raw traffic is 616,562,688 bytes
and node-deduplicated payload traffic is 102,760,448 bytes (6x reduction).
`TRANSPORT_BENCH_JSON` emits the same fields. These numbers contain no network
transfer and must not be compared to a future GIN/Socket result without keeping
the backend label.

## Notify performance baseline

The benchmark is intentionally a correctness baseline, not a GIN/RDMA result:

```bash
LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda-12.9/compat \
  build-cuda/test_cuda_notify --bench 8
```

The benchmark creates one session per rank once, reuses it for 10 warmups and
100 measured iterations, and reports mean, P50, P95, min, and max. Names
distinguish one-time session setup, per-generation coordinator/backend and
control/config/count/summary slot setup, and steady-state execution.
`BENCH_JSON` carries the same categories. Workspace reporting includes the
route input, unique notify device arena and pinned staging, 256-byte control
wire/slots, canonical config wire/slots, count/summary slots, final-metadata
subview, and event count. Shared-arena subviews are not double-counted.
Setup-inclusive E2E is calculated per rank as `rank_setup + rank_steady` before
choosing the slowest rank. Mapping has no per-route copy, allocation, or
synchronization.

Schema 7 additionally reports `prefix_planning_us` (the parallel release
prefix/packet plan or checked debug planner) separately from
`remaining_parallel_materialization_us`. Both are measured with precreated
CUDA events and remain components of `device_materialization_us`; observing
them adds no steady-state event creation.

On the inspected local 8-GPU L20X host, the release coordinator produced these
slowest-rank statistics (microseconds, 10 warmups/100 measured):

| stage | mean | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| one-time NCCL session setup | 1597360 | 1630280 | 1703040 | 1349080 | 1703040 |
| one-time two-slot pool setup | 13017.1 | 13415.2 | 14112.8 | 10011.8 | 14112.8 |
| per-generation setup | 2.979 | 2.778 | 4.197 | 1.909 | 7.729 |
| setup-inclusive generation E2E | 1227.07 | 1224.86 | 1310.28 | 1121.77 | 1436.93 |
| steady-state execution | 1224.28 | 1221.53 | 1307.60 | 1118.57 | 1434.25 |
| local stable mapping | 145.458 | 144.320 | 155.040 | 138.848 | 168.480 |
| control + cached config | 604.122 | 607.244 | 661.279 | 499.345 | 730.954 |
| expert-count AllGather | 171.921 | 173.537 | 234.506 | 106.330 | 276.773 |
| packet-summary AllGather | 24.889 | 23.259 | 29.699 | 21.309 | 65.463 |
| device materialization | 234.756 | 229.908 | 243.745 | 218.326 | 465.049 |
| prefix planning component | 104.506 | 102.176 | 108.192 | 99.904 | 206.336 |
| remaining parallel stages | 37.308 | 37.248 | 37.728 | 36.864 | 40.736 |

For `max_inflight_generations=2`, the corresponding complete unique workspace
was 42,624,126 device bytes and 78,992 pinned bytes: route input 589,824
(two pool convenience slots plus one router/test-owned production input), two
Notify arenas totaling 41,906,686, Notify pinned status staging 656, control
slots 9,216, config wire storage 8,192, config slots 147,456, expert slots
49,152, and summary slots 128. Final metadata is a 20,953,343-byte subview of
one arena and is not added again. Twenty-eight pooled CUDA events are reported;
opaque runtime allocation size is recorded as zero.

The phase-2 same-accounting steady baseline was approximately 97.6 ms, of
which roughly 69 ms was bulk D2H, host materialization, and bulk H2D. The first
all-device slice reduced the same-accounting steady mean to 38.0 ms, with
33.2 ms in its serial materializer. The pooled parallel baseline now measures
6.856 ms steady and 5.698 ms materialization while preserving production audit
`bulk route D2H / collective D2H / host materialize / metadata H2D = 0/0/0/0`.
Across ten pooled steady generations, CUDA API audit measured
`cudaMalloc/cudaFree/cudaHostAlloc/cudaFreeHost/cudaEventCreate*/`
`cudaEventDestroy/cudaDeviceSynchronize = 0` for every category. The local
ten-generation audit recorded bounded H2D/D2H/D2D traffic as
8,480/8,640/0 bytes; validation download was outside the interval. Parallel
one-block-per-expert reductions and deterministic node block scans reduce
planning from 5.575 ms to 0.105 ms and complete materialization from 5.698 ms
to 0.235 ms. The 1.224 ms Notify steady mean is 82.1% below the prior 6.856 ms
baseline.

The earlier approximately 197 ms number is not a comparable regression point:
that runner rebuilt communicators every iteration and computed steady time by
subtracting independently selected slowest-rank maxima. The corrected 98.0 ms
mean used one reused session and same-rank accounting. Both the 38.0 ms serial
device result, the 6.82 ms pooled-parallel result, and the 1.224 ms release
result use that corrected same-rank definition.

### Profiler evidence

Nsight Compute 2025.2.1 is available on the inspected host; Nsight Systems is
not installed. The exact-shape, single-rank NCU run used the release binary and
the same materialization kernels as world 8:

```bash
ncu --target-processes all --set basic \
  --kernel-name 'regex:materialize_.*_kernel' --launch-count 11 \
  --csv --log-file build/ncu_notify_mainshape.csv \
  build/test_cuda_notify --bench 1
```

NCU reported 6.176 us for the 384-block expert-prefix kernel (30.9% achieved
occupancy), 5.952 us for parallel finalization, and 118.080 us for the
deterministic token/node packet scan. Route, incoming-row, M-tile and fan-out
kernels were 5.984, 4.160, 13.920 and 12.704 us respectively. The packet scan,
not the former serial expert prefix, is now the largest materialization kernel.

The real two-process IPC trace profiled the release bulk kernels. For its small
oracle fixture, bulk dispatch and return were 114--132 us and 109--115 us;
phase publication (system fence plus peer signal) was about 109 us, waits about
5 us, and reduce 4.8 us under instrumentation. On the uninstrumented main
world-8 workload the CUDA-event spans above are authoritative. There are no
per-route system atomics or credit spins in the trusted phase-batched path;
the measured credit-stall counter is zero. NCU's local DRAM percentage does
not account for peer-fabric stores, so peer bytes divided by event time is used
for explicitly labeled effective NVLink/P2P bandwidth rather than treating
that metric as HBM bandwidth.

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
