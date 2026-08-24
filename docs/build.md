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
and 50 measured iterations. Each iteration is reduced to the slowest rank
before statistics are computed. Worker spans overlap by design (relay and
synthetic return progress concurrently), so stage spans do not sum to total.

| span | mean (us) | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| dispatch put | 175406.52 | 175405 | 175605 | 175156 | 175636 |
| relay | 224831.42 | 224787 | 224964 | 224689 | 225007 |
| synthetic return | 417737.06 | 417773 | 418050 | 417336 | 418107 |
| expert-order combine/reduce | 8794.23 | 8793.76 | 8796.83 | 8790.85 | 8800.13 |
| transport steady total | 426695.44 | 426735 | 427007 | 426294 | 427066 |

Slowest-rank one-time pool/backend setup was 935156 us. Each rank's single-slot
pool owns 3,089,435,048 device bytes and 168 pinned bytes. Raw six-route traffic
is 704,643,072 bytes/rank; node-deduplicated proxy ingress is 117,440,512
bytes/rank (6x reduction). The measured no-wrap capacity path had zero credit
stalls and peak observed occupancy 1 because consumers kept pace with the
published sequence watermark. `P2P_BENCH_JSON` contains the per-rank traffic,
workspace, stall/occupancy, and slowest-rank distributions.

The remote clean archive reproduced the same 10+50 benchmark with steady total
mean/P50/P95/min/max of 432377.22/432356/432570/432127/432676 us and the same
traffic/workspace cardinalities.

This correctness-first result is substantially slower than the earlier device
simulation and is not presented as final transport performance. Profiling
points to serialized source-CTA packet relay and small-route peer stores as the
next optimization target; it does not justify a GIN/RDMA claim.

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

Schema 6 additionally reports `prefix_planning_us` (the existing checked
prefix-plan kernel) separately from
`remaining_parallel_materialization_us`. Both are measured with precreated
CUDA events and remain components of `device_materialization_us`; observing
them adds no steady-state event creation.

On the inspected local 8-GPU L20X host, the pooled parallel coordinator
produced these slowest-rank statistics (microseconds, 10 warmups/50 measured):

| stage | mean | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| one-time NCCL session setup | 1325100 | 1312250 | 1351720 | 1303210 | 1351720 |
| one-time two-slot pool setup | 13352.6 | 13341.5 | 14204.3 | 11762.3 | 14204.3 |
| per-generation setup (pool acquire) | 2.243 | 2.051 | 2.903 | 1.496 | 4.567 |
| setup-inclusive generation end-to-end | 6857.58 | 6850.13 | 6920.50 | 6781.25 | 7248.20 |
| steady-state execution | 6855.91 | 6848.03 | 6918.36 | 6779.68 | 7246.86 |
| local stable mapping | 143.139 | 142.976 | 148.832 | 138.016 | 149.856 |
| control + exact config | 858.642 | 864.025 | 920.505 | 794.969 | 1058.21 |
| expert-count AllGather | 35.030 | 32.099 | 52.659 | 27.869 | 92.696 |
| packet-summary AllGather | 22.983 | 21.874 | 27.716 | 20.680 | 58.164 |
| parallel device materialization | 5697.89 | 5693.38 | 5722.69 | 5684.20 | 5848.47 |
| prefix planning (component) | 5574.66 | 5573.63 | 5580.70 | 5566.88 | 5582.91 |
| remaining parallel stages (component) | 37.296 | 37.248 | 38.016 | 36.608 | 38.112 |
| pool acquire + final release/drain | 20.311 | 18.863 | 28.502 | 15.861 | 43.904 |

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
8,480/8,640/0 bytes; validation download was outside the interval. The new
events expose that prefix planning accounts for about 5.575 ms, while the
remaining parallel stages account for about 0.037 ms. Compared with 6.824 ms,
the 6.856 ms steady mean is a 0.5% shift, not an unexplained material
regression. Parallelizing the checked planning scan remains the next
optimization.

The earlier approximately 197 ms number is not a comparable regression point:
that runner rebuilt communicators every iteration and computed steady time by
subtracting independently selected slowest-rank maxima. The corrected 98.0 ms
mean used one reused session and same-rank accounting. Both the 38.0 ms serial
device result and the new 6.82 ms pooled-parallel result use that corrected
same-rank definition.

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
