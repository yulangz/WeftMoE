# Build and verification

The host-only build requires GNU Make, a C++17 compiler, and Python 3. It stays
available without CUDA or NCCL. The optional device slice requires NVCC, CUDA
runtime/development files, NCCL headers, and a runtime NCCL library compatible
with the installed driver/toolkit.

## Observed stage-4 environment and transport capability

Both test hosts were inspected without
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
- the explicitly selected `$NCCL_RUNTIME_DIR/libnccl.so.2`
  reports `2.28.3+cuda12.9` and remains the default runtime used by frozen
  single-node tests;
- the official `libnccl2_2.29.7-1+cuda12.9_amd64.deb` works without a system
  install when extracted into a repository-external cache. Its SHA-256 is
  `be1e9f1246394cb5a33df2b9f1d687b558b83fe2eec18caf9f82a11aeab3fc5f`.
  The code is compiled against the system 2.29.7 headers.
- `/usr/include/nccl_device/gin.h` (2.29.7) exposes GIN put, remote indexed/VA
  signals, local counters, flush/wait, and `ncclTeamRail`. The selected 2.28.3
  runtime does not export `ncclGinOutboxCreateRequirement` or
  `ncclGinInboxA2ACreateRequirement`. The explicitly loaded CUDA 12.9 NCCL
  2.29.7 exports the complete required set and passes real two-host
  device-communicator/window initialization. The project data-plane backend is
  implemented; the explicit compatible runtime reports
  `gin_backend_implemented=1, gin_usable=1`, while the default 2.28.3 runtime
  remains unusable because the symbol gate fails.
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
  repeat checking;
- every public header as an isolated `-Werror` translation unit;
- thirty-four compile-negative probes for the retired `wait`, `wait_preflight`, raw
  notify start, phase-specific collective starters, internal NCCL
  operation/backend types, bootstrap hex encode/parse APIs, and the unpooled
  production coordinator constructor, generic route rebinding/wrong route
  dtype, raw dispatch views, removed generation handoff/reissue helpers, and
  the internal no-wait producer-race seam, and the non-installed runtime-shape
  harness.

The activation-specific additions reject wrong activation dtype, generic
buffer rebinding, raw peer-pointer extraction, access to the private IPC
factory, and the retired caller-selected backend-enum entry.
- thirteen fake-runner/bootstrap-channel security groups covering success,
  local/remote failure, timeout and TERM-to-KILL escalation, SSH start failure,
  short/oversize bootstrap, runner exception, every 1/2/7/8/17/127/128-byte
  prefix representation, Python poison writers, top-level CLI output, and 20
  failure/recovery cycles. They inspect argv/environment, exact frame content,
  FD/process cleanup, scrubbed diagnostics, and absence of bootstrap files.

## Installed API v2

```bash
make install DESTDIR=/tmp/megamoe-package PREFIX=/usr/local
make install-test
```

`install-test` performs a clean temporary install, checks the exact 12-header
manifest, then compiles, links with `-lmegamoe`, and runs the world-1 minimal
Forward+Backward example using only the installed prefix and `-Werror`; it
never adds the repository `src/` include path. Validation
downloads, synthetic transports, fixture/hash mutation, fault injection and
runtime-shape harnesses are exposed only by repository-internal testing headers.
See `docs/api_v2_migration.md`.

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
make BUILD_DIR=build-sanitize \
  SANITIZER_PRELOAD="$ASAN_LIB:$UBSAN_LIB" cross-check
```

## Optional CUDA/NCCL build and local tests

```bash
NCCL12="$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR"

make BUILD_DIR=build-cuda -j cuda-lib
make BUILD_DIR=build-cuda -j \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-test-local

# Correctness-only runtime H/I matrix (world 1 and real local world 2).
make BUILD_DIR=build-cuda \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" cuda-runtime-shape-test

# Generic-shape GIN uses tests/run_multinode_nccl.py with the compatible
# repository-external NCCL 2.29.7 CUDA-12.9 runtime and explicit
# MEGAMOE_RUNTIME_HIDDEN / MEGAMOE_RUNTIME_INTERMEDIATE values.

# Include the one-GPU CUDA slice in the default host suite.
make BUILD_DIR=build-cuda ENABLE_CUDA=1 \
  NCCL_RUNTIME_LIB_DIR="$NCCL12" test

# Compile-only architecture check; do not infer runtime coverage.
make BUILD_DIR=build-cuda cuda-sm90-compile

# Clean release gate: installed-header raw scan, exact dynamic ABI, retired
# symbol link failure, installed-prefix compile/link/run, kernel artifact
# manifest, public headers, compile-negative API suite, WERROR and hygiene.
make BUILD_DIR=build-release ENABLE_CUDA=1 \
  NCCL_RUNTIME_LIB_DIR="$NCCL_RUNTIME_DIR" release-check

# Auditable transport capability, correctness/sanitizer path and benchmark.
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --transport-capability
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --device-smoke
LD_LIBRARY_PATH="$NCCL12" build-cuda/test_cuda_notify --transport-bench

# Single-launch Forward correctness (cache/no-cache, 20 deterministic repeats).
NCCL_DEBUG=WARN LD_LIBRARY_PATH="$NCCL12" \
  python3 tests/run_local_activation_p2p.py \
    --binary build-cuda/test_cuda_notify --world 8 --forward-smoke \
    --top-k 6 --gate-mode fixed-random --rounds 20

# Main-shape Forward: 10 warmup + 100 measured generations.
NCCL_DEBUG=WARN LD_LIBRARY_PATH="$NCCL12" \
  python3 tests/run_local_activation_p2p.py \
    --binary build-cuda/test_cuda_notify --world 8 --forward-benchmark \
    --top-k 6 --gate-mode fixed-random --rounds 110

# Same routes/gates/BF16 weights under Megatron-Core 0.17.0.
MEGAMOE_NCCL_LIBRARY="$NCCL12/libnccl.so.2" \
NCCL_DEBUG=WARN LD_LIBRARY_PATH="$NCCL12" \
  torchrun --standalone --nproc-per-node=8 \
    benchmarks/megatron_dispatch_baseline.py \
    --expert-body te-grouped-mlp --gate-mode fixed-random \
    --warmups 10 --iterations 100 --runs 3
```

### Repository-external NCCL 2.29.7 CUDA 12.9 GIN runtime

The compatible runtime is downloaded and extracted, never installed globally
and never committed:

```bash
NCCL_GIN_CACHE=/absolute/path/outside/MegaMOE/nccl-2.29.7-cuda12.9
mkdir -p "$NCCL_GIN_CACHE/packages" "$NCCL_GIN_CACHE/root"
(cd "$NCCL_GIN_CACHE/packages" && \
  apt-get download 'libnccl2=2.29.7-1+cuda12.9')
sha256sum "$NCCL_GIN_CACHE/packages/libnccl2_2.29.7-1+cuda12.9_amd64.deb"
dpkg-deb -x \
  "$NCCL_GIN_CACHE/packages/libnccl2_2.29.7-1+cuda12.9_amd64.deb" \
  "$NCCL_GIN_CACHE/root"

NCCL_GIN_LIB="$NCCL_GIN_CACHE/root/usr/lib/x86_64-linux-gnu"
env -u NCCL_IB_DISABLE -u NCCL_SOCKET_IFNAME \
  python3 tests/run_multinode_nccl.py \
    --binary "$PWD/build/test_cuda_notify" \
    --remote REMOTE_HOST --port REMOTE_PORT \
    --remote-binary "$PWD/build/test_cuda_notify" \
    --gpus-per-node 1 --scenario gin_probe \
    --nccl-lib-dir "$NCCL_GIN_LIB:/usr/local/cuda-12.9/compat"

# Repeat with --gpus-per-node 8 for the complete 16-rank rail team.
```

The observed world-2 and world-16 result on both hosts is
`compiled=22907, runtime=22907, headers=1, symbols=1,
runtime_initializable=1`. The probe collectively creates a GIN-requiring
`ncclDevComm`, allocates symmetric storage with `ncclMemAlloc`, registers and
deregisters a strict-ordering NCCL window, and destroys the device comm. It
does not by itself claim data-plane usability. The separately typed full
Forward/Backward benchmark below proves project put/signal/local-completion.

The same compatible runtime also executes a real 1 KiB payload on every rail:

```bash
env -u NCCL_IB_DISABLE -u NCCL_SOCKET_IFNAME \
  python3 tests/run_multinode_nccl.py \
    --binary "$PWD/build-gin-put/test_cuda_notify" \
    --remote REMOTE_HOST --port REMOTE_PORT \
    --remote-binary "$PWD/build-gin-put/test_cuda_notify" \
    --gpus-per-node 8 --scenario gin_put_probe \
    --nccl-lib-dir "$NCCL_GIN_LIB:/usr/local/cuda-12.9/compat"
```

This probe uses `ncclTeamRail`, a registered NCCL allocation, a device-issued
`ncclGin::put`, an indexed remote signal ordered after payload settlement, and
a local completion counter before source reuse. The observed world-16 rail
shape is two nodes by eight local ranks, and all 16 received hashes match the
expected same-local-rank peer. It proves the runtime/plugin/NIC data-movement
path, but is deliberately not counted as the complete MegaMOE GIN backend.

The real activation transport slice is exercised at world 2 with:

```bash
env -u NCCL_IB_DISABLE -u NCCL_SOCKET_IFNAME \
  python3 tests/run_multinode_nccl.py \
    --binary "$PWD/build-cuda/test_cuda_notify" \
    --remote REMOTE_HOST --port REMOTE_PORT \
    --remote-binary "$PWD/build-cuda/test_cuda_notify" \
    --gpus-per-node 1 --scenario activation_gin \
    --nccl-lib-dir "$NCCL_GIN_LIB:/usr/local/cuda-12.9/compat"
```

Both ranks use ten tokens, top-k six, dynamic source-resident gates, and the
independent host oracle. Twenty generations complete with
`backend=nccl-gin-device` and `oracle=1`; no activation payload crosses the
host. The same test runs the complete two-node, eight-GPU-per-node same-rail
and local CUDA-IPC fan-out path by changing `--gpus-per-node 1` to
`--gpus-per-node 8`. All 16 ranks pass 20 deterministic generations with
`backend=nccl-gin-device` and `oracle=1`. Return contributions are aggregated
per deduplicated token packet (one put and signal), rather than per route.
This remains the transport-only oracle slice. Full FC execution uses the
Forward/Backward benchmark scenarios below and is reported separately.

The runtime team/pool setup is separately checked on all 16 ranks with
`--gpus-per-node 8 --scenario gin_team_pool`. The observed two-slot small-shape
pool reports `55,440` ordinary compute bytes and `6,672` GIN-registered bytes
per rank. All ranks validate rail size two, local team size eight, their
runtime local rank, registered windows, and seven node-local IPC mappings.

Backward smoke controls are forwarded identically to every local and remote
rank. For example, the concentrated-route reproduction uses
`MEGAMOE_BACKWARD_ROUTE_CASE=concentrated`,
`MEGAMOE_BACKWARD_SMOKE_REPEATS=20`, and optional
`MEGAMOE_BACKWARD_POOL2_VALIDATE=1`. `MEGAMOE_BACKWARD_CACHE_FIRST=1`
reverses the generation-900/901 cache order in a fresh launcher session. The
launcher prints the effective route case, repeat count, pool2 flag, cache
order, and `NCCL_IB_DISABLE` value before starting ranks. Use `--enable-ib` or
`--disable-ib` to set the latter explicitly; otherwise it is inherited.

The explicit Socket correctness backend uses the default CUDA-12.9 NCCL
runtime and requires the network choice at communicator creation. A two-rank
run is:

```bash
python3 tests/run_multinode_nccl.py \
  --binary "$PWD/build-cuda/test_cuda_notify" \
  --remote REMOTE_HOST --port REMOTE_PORT \
  --remote-binary "$PWD/build-cuda/test_cuda_notify" \
  --gpus-per-node 1 --scenario activation_socket \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --disable-ib

# Use --gpus-per-node 8 for the real two-host world-16 run.
```

The launcher prints the actual loaded `libnccl.so.2`, runtime version, network
overrides, concrete `nccl-socket-host-enqueued-fallback` audit label, and
per-rank output/contribution hashes. Each rank compares the downloaded test
result with the independent host transport oracle before printing
`oracle=1`. The complete host-enqueued correctness fallback uses the same
launcher with `--scenario socket_forward_bench` or
`--scenario socket_backward_bench`; set the corresponding
`MEGAMOE_FORWARD_BENCH_ROUNDS`/`MEGAMOE_BACKWARD_BENCH_ROUNDS` and cache
environment variables. It sends aggregates directly between ranks and runs
the frozen WGMMA/TMA FC core, but does not claim node-deduplicated same-rail
GIN execution or a single cross-node persistent payload kernel.

### World-16 full GIN Forward/Backward and Megatron network baselines

The H200 Forward specialization has an explicit, narrow compile-time tuning
plan in `configs/forward_h200_autotune.json`. Its four numbered sections cover
GEMM geometry/register allocation, RDMA/Relay role structure, mixed-scheduler
granularity, and completion/Reduce behavior. Every value is present in the
JSON, including parameters intentionally held to a single proven value. The
candidate list is explicit rather than a Cartesian product.

Validate the plan and generator without CUDA compilation with:

```bash
make forward-autotune-test
python3 tools/autotune_forward.py --generate-only
```

Build one content-addressed specialization without running the benchmark:

```bash
python3 tools/autotune_forward.py \
  --candidate current_m128_n128_n256 --build-only
```

Run the complete narrow world-16 search and the 100-sample final validation
with `make forward-autotune`. The Python launcher writes one generated header
and one build directory per candidate, verifies that the resulting cubin has
exactly one split Forward kernel, compares all input/weight/output hashes, and
selects `forward_slowest_rank_us.mean`. Result JSON and build/launcher logs are
written below `artifacts/forward-autotune/`. Kernel choices are not carried in
environment variables. Seed, token count, top-k, FC1-cache mode, generation
base, and sample counts are also explicit in the JSON; the launch layer only
translates the subset required by the existing benchmark executable ABI.

The production main-shape GIN benchmark runs without a host payload phase:

```bash
NCCL_GIN_LIB="$NCCL_RUNTIME_DIR"

env -u NCCL_IB_DISABLE -u NCCL_SOCKET_IFNAME \
  MEGAMOE_P2P_BENCH_TOKENS=8192 \
  MEGAMOE_FORWARD_BENCH_ROUNDS=110 MEGAMOE_FORWARD_CACHE=1 \
  python3 tests/run_multinode_nccl.py \
    --binary "$PWD/build/test_cuda_notify" \
    --remote REMOTE_HOST --port REMOTE_PORT \
    --remote-binary "$PWD/build/test_cuda_notify" \
    --gpus-per-node 8 --scenario gin_forward_bench \
    --nccl-lib-dir "$NCCL_GIN_LIB:/usr/local/cuda-12.9/compat"

# Use MEGAMOE_BACKWARD_BENCH_ROUNDS=110, MEGAMOE_BACKWARD_CACHE=1 and
# --scenario gin_backward_bench for the integrated Forward+Backward path.
```

The matching Megatron-Core 0.17.0 native Socket baseline is launched without
`LD_PRELOAD`; the wrapper resolves and loads `libnccl.so.2` explicitly:

```bash
python3 benchmarks/run_multinode_megatron_socket.py \
  --remote REMOTE_HOST --port REMOTE_PORT --gpus-per-node 8 \
  --nccl-lib-dir "$NCCL_GIN_LIB:/usr/local/cuda-12.9/compat" \
  --socket-ifname SOCKET_IFNAME --tokens 8192 --warmups 10 --iterations 100 \
  --expert-body te-grouped-mlp --benchmark-backward --training-only
```

Run each command in three independent sessions, then enforce fixture agreement
and every per-run 15% gate with `benchmarks/compare_multinode_training.py`.
The checked result is
`benchmarks/results/gin_world16_training_20260826.json`. Cross-run means are
41.069 ms GIN Forward, 52.738 ms GIN Backward, and 94.008 ms
Notify-inclusive GIN F+B, versus 816.308/720.933/1465.224 ms for Megatron
NCCL Socket. Every individual run clears all three gates.

The formal RDMA baseline uses the same Megatron benchmark body but fails unless
NCCL selects real IB/RoCE with GPUDirect RDMA. `NCCL_HCAS` is the explicit
comma-separated HCA:port allowlist for the target hosts; `SOCKET_IFNAME` is
only the bootstrap/OOB interface:

```bash
python3 benchmarks/run_multinode_megatron_rdma.py \
  --remote REMOTE_HOST --port REMOTE_PORT --gpus-per-node 8 \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --ib-hca "$NCCL_HCAS" \
  --ib-gid-index ROCE_V2_GID_INDEX \
  --tokens 8192 --hidden 7168 --intermediate 3072 \
  --experts 384 --top-k 6 --warmups 10 --iterations 100 \
  --expert-body te-grouped-mlp --benchmark-backward --training-only \
  --collective-breakdown --evidence-dir EVIDENCE_DIR
```

The launcher explicitly sets `NCCL_IB_DISABLE=0`, `NCCL_NET=IB`, the HCA/GID,
RoCE version, address family, GDR level and OOB interface. It rejects missing
`NET/IB`, missing `GDRDMA`, any `NET/Socket` payload selection, any configured
rail whose counters do not advance on both hosts, or a runtime other than
2.29.7. Full redacted INFO logs and a compact evidence JSON are written to the
requested evidence directory. Run three independent GIN Forward, GIN Backward
and Megatron RDMA sessions, then generate the checked result with:

```bash
python3 benchmarks/compare_multinode_rdma_training.py \
  --gin-forward GIN_FORWARD_1 GIN_FORWARD_2 GIN_FORWARD_3 \
  --gin-backward GIN_BACKWARD_1 GIN_BACKWARD_2 GIN_BACKWARD_3 \
  --megatron-rdma MEGATRON_RDMA_1 MEGATRON_RDMA_2 MEGATRON_RDMA_3 \
  --cpu-affinity CPUSET --gpu-set GPUSET \
  --nccl-runtime-sha256 NCCL_RUNTIME_SHA256 \
  --output benchmarks/results/gin_vs_megatron_rdma_world16_20260827.json
```

That result is the formal world-16 comparison. MegaMOE is slower by
34.36–35.47% Forward, 19.37–20.01% Backward and 23.56–23.96% F+B; consequently
the 15% cross-node target remains open. The Socket result above remains a
correctness/lower-bound record only.

For a small exact correctness cross-check, run both GIN and Socket scenarios
with identical `MEGAMOE_P2P_BENCH_TOKENS` and then compare the actual per-rank
device-storage hashes:

```bash
python3 benchmarks/compare_socket_gin_correctness.py \
  --gin-forward /tmp/gin-forward.log \
  --socket-forward /tmp/socket-forward.log \
  --gin-backward /tmp/gin-backward.log \
  --socket-backward /tmp/socket-backward.log
```

Real world-16 top-k 1/6 checks cover 224 Forward and 352 Backward rank-fields;
the top-k=1 Backward run additionally uses 20 validation generations. World-2
full Backward uses the same 384-expert, 7168x3072 fixture and matches all actual
dX/dW1/dW2/dGate hashes.

The full Socket fallback was also measured in three independent world-16
sessions with the same 10+100 boundary. Its cross-run slowest-rank means are
428.075 ms Forward, 424.088 ms Backward, and 1874.219 ms for the
Notify-inclusive F+B path. The large step wall includes every explicit Socket
payload phase and terminal boundary. The fallback's current correctness
preload completes an identity return before the FC body; its real bytes are
included in the audit rather than hidden. The machine-readable result is
`benchmarks/results/socket_world16_fallback_20260826.json`; every `gin_*`
counter is zero.

Runtime device counters report, per rail/rank/generation, 352.25--352.39 MB
raw remote Forward route payload, 117.44 MB after node dedup, and
352.25--352.39 MB return payload. Forward uses 24 dispatch puts, eight return
puts, and 16 signals; Backward uses 32/24 puts and 24 signals after the
distinct terminal FP32 dGate publication. Derived from
those actual byte counters and the slowest-rank device spans, payload rates are
13.26 GB/s dispatch and 14.89 GB/s return. The Forward pool separates
4,382,039,368 ordinary compute bytes from 1,766,522,888 GIN-registered bytes
per rank; weights add 3,170,893,824 ordinary bytes per rank.

The CUDA build uses the exact MIT-licensed DeepGEMM helper headers retained
under `src/cuda/vendor` for the frozen SM90 kernels. This keeps a `git archive`
self-contained without depending on an adjacent reference checkout. CUTLASS/
CuTe remains external and is discovered from the installed DeepGEMM package
without embedding its wheel path. Environments without that package must pass
`CUTLASS_INCLUDE_DIR=/path/to/cutlass/include`.

Keep `NCCL_DEBUG=WARN` for long launcher runs: rank logs are captured for
structured failure diagnostics, and per-collective `INFO` output is not part of
the timed workload.

On the inspected local 8-GPU host, three independent release sessions (each
10 warmups + 100 measurements) produced slowest-rank Forward E2E means of
15.2538, 15.2652, and 15.2611 ms. The exact same activation, route, gate, and
actual device weight bytes on Megatron-Core 0.17.0 produced native means of
18.4617, 18.6098, and 18.4789 ms. The paired speedups are 17.38%, 17.97%, and
17.41%; every independent session passes the 15% gate. Megatron's explicit
source-gate-after common-semantics mean across the three sessions is
22.3574 ms.
These two Megatron results remain
separate because its native path applies routing probability at a different
point.

The three MegaMOE stage means and complete distributions are stored in
`benchmarks/results/forward_actual_weight_hash_world8.json`. These device
intervals overlap and therefore must not be
summed. Dispatch remains an
initial bulk phase and is not claimed as hidden. The measured pool is
2,847,625,512 bytes/rank and immutable weights are 6,341,787,648 bytes/rank.
Each rank moves about 616.6 MB peer dispatch and 616.5 MB peer return payload;
FP32 gates remain local. The explicit main-shape cache lifecycle test uses two
simultaneously live slots and measures 7,104,537,680 device bytes/rank. It
checks disjoint A/B device intervals, stable gen1 bytes after gen2, exact
pool exhaustion, and selective reuse after one handle is released. Each rank
also rebuilds 24 cached FC1 values per generation in FP64, spanning three
experts, both A/B planes, and four non-adjacent intermediate columns. Rank-0
output hashes are identical under EP1/EP2/EP8,
cache/no-cache, release/debug, and 20 repeats. The independent NumPy FP64
oracle checks selected main-shape output columns plus every cached A/B element;
output error is zero and the cache max absolute/RMSE are 0.0019531/0.0003182
against fixed 0.02/0.008 limits. Byte identity with TransformerEngine's
internal instruction order is intentionally not required.

Fixture v3 computes every BF16 weight from a non-periodic 32-bit mix of
`(global_expert_id, tensor_kind, output_row, input_col)` and maps the final bit
to a stable random `{-1,+1}/256` value. The complete W1 and W2 device storage is
hashed outside timing with fixed 4-Mi-element chunks and two independent
64-bit modular reductions. MegaMOE and Megatron apply the same definition to
the bytes their Forward kernels actually consume. The fair launcher rejects
any per-rank W1 or W2 mismatch before timing; the formula-derived sample hash
is retained only as generator/layout diagnostics.

The production launch requests cluster dimension `2x1x1`. The checked
standalone Forward JSON observes `actual=2` and rank mask `3`; the current
deferred Forward-to-Backward integrated audit observes `actual=1`. JSON reports
requested and actual sizes per measurement. TMA multicast is off, so no
operand-multicast performance claim is made. The generic fallback uses one CTA.
Pool construction clears each slot once. A generation writes only
`2 * world * channels` completion words plus `2 * world` ready/ack words in a
small setup kernel; it performs no arena-wide memset. The Forward benchmark's
runtime API snapshot reports zero device/pinned allocation or free, event
create/destroy, and `cudaDeviceSynchronize` calls across every steady sample.

A clean tracked-file export on the second test host reproduced world-8 top-k=6,
cache/no-cache, and 20-repeat correctness with rank-0 hash
`17050206622106222467`; Forward initcheck reported zero errors. The same export
also passed real cross-host Notify world 2 and 16. Megatron-Core is not
installed on that host, so the formal paired speedup remains the local
same-host comparison above rather than a cross-machine estimate.

Nsight Compute on the main-shape world-1 compute slice reports 19.432 ms kernel
duration, 55.97% SM throughput, 66.55% DRAM throughput, 56.29% Tensor-pipe
active, and 18.75% occupancy. The prior capture was 25.349 ms, 42.37% SM and
48.87% DRAM. Current barrier and long-scoreboard stalls are 9.78 and 3.27
warps/issue-active; WGMMA/TMA wait reduction is therefore the next largest
optimization point. Nsight Systems is not installed on this host.

The frozen main-shape specialization is M=128, FC1 N=128/K=64 with three TMA
stages, FC2 N=256/K=64 with four TMA stages, 384 threads/CTA, one resident CTA
per SM, actual cluster size one, release task chunks of four, and one warp per
Reduce token. TMA multicast remains disabled: the tested multicast cluster
schedule was slower and did not satisfy long-run progress. Four-stage FC1,
BN=64/128 alternatives, direct-peer FC2 epilogues, dedicated Return CTAs,
two-warps/token Reduce, and per-route signalling were likewise rejected.
The selected worker policy uses all 132 CTAs for initial Dispatch, then a
shared FC1/FC2 counter; CTAs return the M-block whose final FC2 tile they
complete and enter Reduce as they leave compute. A fixed dedicated-CTA role
split lost to this dynamic phase conversion on the measured main shape.
The versioned artifact manifest reports the actually selected `sm_90` Forward
kernel at 61 registers/thread, zero stack/static shared and 196,704 bytes
configured dynamic shared at world 8; Backward is 64 registers, zero stack and
196,672 bytes configured dynamic shared. Compile-only `sm_90a` is recorded
separately at 168 registers and 16/120 bytes Forward/Backward stack.
`ptxas` still reports WGMMA accumulator dependency serialization and
an ignored `setmaxnreg`; these match the remaining barrier/scoreboard profile
and are not hidden from the performance evidence.

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
  --remote REMOTE_HOST --port REMOTE_PORT \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --disable-ib

PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote REMOTE_HOST --port REMOTE_PORT \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 8 \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --disable-ib

# Timeout-bounded 20-round distributed rejection/recovery on the same comm.
PYTHONPATH=. python3 tests/run_multinode_nccl.py \
  --binary ./build-cuda/test_cuda_notify \
  --remote REMOTE_HOST --port REMOTE_PORT \
  --remote-binary /absolute/remote/test_cuda_notify \
  --gpus-per-node 1 --scenario out_of_range \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --disable-ib
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
in `LD_LIBRARY_PATH`. `--nccl-lib-dir` parses a platform path-list: every entry
is added to the loader search path, while the first entry that actually owns
`libnccl.so.2` becomes `MEGAMOE_NCCL_LIBRARY`. Thus a CUDA compatibility-only
directory is never incorrectly suffixed with `/libnccl.so.2`. Megatron loads
that resolved file before importing PyTorch and reports both its path and
`ncclGetVersion`; no `LD_PRELOAD` is needed.

CUDA memory checking uses valid and invalid device mapping kernels without
instrumenting the third-party NCCL library (this NCCL binary reports
`cudaErrorNoKernelImageForDevice` under Compute Sanitizer although it runs
normally):

```bash
make BUILD_DIR=build-cuda \
  NCCL_RUNTIME_LIB_DIR="$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  cuda-sanitizer-smoke
```

`cuda-sanitizer-smoke` runs memcheck, racecheck, and initcheck over that same
closed-loop 73-node path and over an NCCL-free harness that launches the exact
production world-1 Forward WGMMA/TMA persistent kernel. All six invocations
reported 0 errors/0 hazards. The harness replaces only bootstrap/Notify with
an explicitly initialized world-1 dispatch view; it is not the device-smoke
kernel. The remote image provides the executable at
`/usr/local/cuda/bin/compute-sanitizer` but does not include that directory in
the default `PATH`; explicit Forward initcheck from a clean export also
reported 0 errors. Host ASan/UBSan and world 1/2/8 ran from the same clean
export.

For the real two-process IPC test, the launcher now reads the fixed 128-byte
bootstrap frame and uses a bounded nonblocking trailing-byte check. This avoids
waiting forever for EOF when Compute Sanitizer's subreaper retains an inherited
write descriptor; every wait remains under the launcher timeout. With the
current tool/runtime pair, real world-2 top-k=6 racecheck reaches the production
Forward+Backward kernels and reports zero errors/hazards. The prior shared-
memory alias was replaced by a thread-local scalar; no suppression is used.
Normal world-2 execution, twenty-repeat bitwise checks, and the independent
Python oracle also pass. The NCCL-independent closed-loop device path,
including transport worker and 73-node materializer, is covered by
memcheck/racecheck/initcheck with zero errors/hazards for all three.

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

### Phase-4 identity-transport Megatron-Core comparison

The measurements in this subsection are the frozen identity-expert transport
baseline from phase 4. They are retained for regression analysis and are not
the phase-5 full-Forward performance claim. The current
`benchmarks/run_integrated_comparison.py` instead runs the full WGMMA/TMA
Forward comparison described above, verifies per-rank hashes for activation,
routes, gates, and sampled weights before accepting timing, and fails unless
every independent run is at least 15% faster than Megatron native.

The phase-4 comparison launcher used three independent process worlds for each
implementation, with 10 warmups and
100 measured generations per run. MegaMOE's timed sample starts with a ready
device route/activation lease, creates a new Notify generation, passes that
exact `DeviceDispatchHandle` directly to the real `LocalCudaP2p` backend, runs
identity expert return and deterministic combine, and stops only after output
completion and the consumer drain required for safe pool reuse. Session, pool,
and IPC-team construction are the only excluded one-time costs. Host-wall
samples are first reduced with all-rank MAX; CUDA events provide diagnostic
phase spans. The former `Notify mean + transport device-slice mean` estimate is
retained only as a component baseline and is never used for a speedup claim.
Both launchers inherit the same recorded CPU-affinity mask, expose the same
eight GPUs, and use the same NCCL environment overrides. Each records both its
reported NCCL version and loaded library path; a framework-bundled version
difference is reported explicitly rather than silently hidden.

The common configuration uses contiguous ownership (rank 0 owns experts 0--47,
rank 1 owns 48--95, and so on), identical selected expert IDs and token/slot
order, deterministic BF16 activation, non-trivial per-route FP32 gates,
identity experts, fixed expert-ID FP32 accumulation, and one final BF16 cast.
The fixed-random generator defines each selected softmax score as an integer in
`[1,65521]`, derived from seed `0x4d4547414d4f4547`, rank, token and slot, then
normalizes the selected scores in FP32. Its per-rank tensor hash is recorded.
MegaMOE retains this tensor on the source (`8192*6*4 = 196,608` bytes/rank),
transports zero gate bytes, and applies it after identity return. Megatron's
native dispatcher transports FP32 probabilities and applies them at the
expert-side identity boundary. The common-semantics Megatron path keeps its
real dispatcher/collectives but performs an explicit source-side expert-order
FP32 reduction; it does not claim `index_add_` has a fixed route order.

The inspected single-node world-8 release result was:

| comparable host-wall E2E | run means (ms) | 3-run mean (ms) | variance (ms^2) |
|---|---|---:|---:|
| MegaMOE integrated dynamic gate | 5.0592 / 5.1517 / 5.0836 | 5.0982 | 0.00153414 |
| Megatron common semantics dynamic gate | 11.0240 / 11.0247 / 11.0049 | 11.0179 | 0.00008397 |
| Megatron native dynamic gate | 9.4805 / 9.4712 / 9.4905 | 9.4807 | 0.00006181 |

The formal native improvement is
`(9.4807 - 5.0982) / 9.4807 = 46.23%`; the reference-oriented common-semantics
improvement is 53.73%. Megatron native remains separately labelled because it
applies/casts gate-weighted expert output before return, while the common path
matches MegaMOE's frozen source-side gate-after order exactly.

One representative 100-sample run supplies full distribution and CUDA phase
statistics:

| MegaMOE slowest-rank span | mean (ms) | P50 | P95 | min | max |
|---|---:|---:|---:|---:|---:|
| integrated host-wall E2E | 5.0592 | 5.0546 | 5.1087 | 4.9987 | 5.2190 |
| Notify | 0.5096 | 0.5056 | 0.5265 | 0.4957 | 0.6668 |
| transport total | 4.5357 | 4.5343 | 4.5633 | 4.4835 | 4.5914 |
| transport generation setup | 0.0304 | 0.0301 | 0.0384 | 0.0230 | 0.0429 |
| dispatch put | 1.8298 | 1.8299 | 1.8314 | 1.8276 | 1.8332 |
| identity relay/return | 2.1072 | 2.1062 | 2.1303 | 2.0609 | 2.1593 |
| dynamic gated reduce | 0.5566 | 0.5567 | 0.5574 | 0.5553 | 0.5580 |
| terminal completion/drain | 0.0157 | 0.0158 | 0.0167 | 0.0128 | 0.0170 |

The corresponding first Megatron run measured common-semantics E2E mean
11.0240 ms, P50 10.9912, P95 11.1962, min 10.9144, max 12.1013. Native E2E was
9.4805/9.4569/9.6342/9.3867/10.1822 ms for mean/P50/P95/min/max. MegaMOE and
Megatron common semantics are byte-identical to the same dynamic-gate
reference. Megatron native reports max absolute error 0.03125 and max relative
error 0.0136986 because its gate/cast boundary is intentionally native.

The final clean archive on the second test host repeated the MegaMOE world-8 run in
three independent sessions. Integrated E2E means were 5.0932, 5.0835 and
5.0795 ms; the corresponding P95 values were 5.1153, 5.1030 and 5.1083 ms.
All eight per-rank gate hashes matched the local fixture in every session and
the downloaded output remained byte-identical. Megatron-Core is not installed
on that host, so no remote Megatron comparison is claimed.

Actual peer payload per representative rank is 616,562,688--616,591,360 bytes
for MegaMOE dispatch and 616,491,008--616,663,040 bytes for return. Megatron
reports 616,562,688 BF16 bytes in each direction plus 172,032 dispatch FP32
probability bytes, 1,233,297,408 bytes total. MegaMOE's aggregate effective peer
rate was 2168--2175 GB/s across eight GPUs. MegaMOE additionally retains
196,608 route bytes and 196,608 local gate bytes/rank; peer gate bytes are
exactly zero. These are physical remote-GPU bytes,
not logical node-dedup traffic.

Release generation setup performs no large arena clear. It increments a slot
epoch, copies the 8-byte cancel epoch, clears one status record, creates a
16-byte setup record, and executes a 16-byte/rank setup AllGather. The latter is
retained because it proves common transport slot/capacity readiness; removing
it caused asymmetric-capacity peers to enter P2P at different boundaries.
Static peer mappings, layouts, and capacities are one-time setup. After all
return signals arrive, output reduction plus its completion event is sufficient
on the normal release path, so no redundant terminal team AllGather is issued.
The runtime audit observes one 200-byte generation memset, nine kernel launches,
49,152 dispatch blocks, 49,146--49,158 return blocks, and 8,192 reduce blocks
per rank/generation. Across 100 measured generations it records zero `cudaMalloc/free`,
`cudaHostAlloc/free`, event create/destroy, or device synchronize calls and zero
large-generation memsets. Bounded control/status traffic was 208,626 H2D and
285,648 D2H bytes/rank over all 110 warmup+measurement generations; activation
bulk D2H/H2D remained zero.

The standard round-robin benchmark pool owns the measured 3,092,974,032 device
bytes/rank and 208 pinned bytes/rank. Exact continuous ownership needs six more
worst-case destination rows, so the integrated comparison reports
3,093,060,560 device bytes/rank; this is a checked capacity difference of
86,528 bytes rather than an unreported allocation. The machine-readable record
is emitted as `FAIR_BASELINE_JSON` and can be reproduced with:

```bash
NCCL_DEBUG=WARN python3 benchmarks/run_integrated_comparison.py \
  --binary build/test_cuda_notify \
  --nccl-lib-dir /path/to/nccl/lib \
  --output build/fair_baseline.json
```

`benchmarks/megatron_dispatch_baseline.py` uses Megatron-Core 0.17.0's actual
`MoEAlltoAllTokenDispatcher`. For this fairness run both processes load
`$NCCL_RUNTIME_DIR/libnccl.so.2`, and an
actual `ncclGetVersion` call reports 2.28.3 on both sides. PyTorch's build-time
NCCL version remains 2.27.5 and is recorded separately; it is not treated as
the loaded runtime. The comparison runner verifies the actual runtime version
and records each loaded library path. Project compile headers remain 2.29.7.
This is an integrated identity-expert transport comparison, not an
FC1/SwiGLU/FC2 or backward result; `intermediate=3072` is recorded for that
future work.

The same script also runs as a two-node `torchrun` job; it intentionally uses
the distributed world rather than `LOCAL_WORLD_SIZE` when constructing the EP
group. With 8 ranks/node, `NCCL_SOCKET_IFNAME=SOCKET_IFNAME`, `NCCL_IB_DISABLE=1`, 10
warmups and 100 samples, the measured world-16 means were 0.902 ms routing,
463.267 ms dispatch, 464.664 ms combine, and 919.151 ms identity transport
roundtrip (P50 926.951, P95 1073.007 ms). The corresponding estimated payload
rate was about 1.426 GB/s/direction/rank. This remains a real Megatron
cross-host Socket-NCCL identity-transport baseline. The newer full-FC
world-16 fair RDMA comparison (which records a GIN slowdown, not a speedup) is
reported in the Phase-7 section above;
the two benchmark semantics must not be mixed.

Example two-node invocation (run once per node with node rank 0/1):

```bash
NCCL_SOCKET_IFNAME=SOCKET_IFNAME NCCL_IB_DISABLE=1 torchrun \
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
LD_LIBRARY_PATH="$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
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

The real two-process IPC trace profiled the release bulk kernels with:

```bash
ncu --target-processes all --set basic \
  --kernel-name 'regex:local_p2p_(bulk|publish|wait|finish).*' \
  python3 tests/run_local_activation_p2p.py \
    --binary build/test_cuda_notify --world 2 --rounds 1 --top-k 6
```

For this small oracle fixture, bulk dispatch and return were 114--132 us and
109--115 us; phase publication (system fence plus peer signal) was about 109 us,
waits about 5 us, reduce 4.8 us, and finish 2.1 us under instrumentation. The
return fixture has only eight row blocks, so NCU reports 10.8% achieved
occupancy there; the main shape launches one block per route/row and its
uninstrumented world-8 CUDA-event spans above are authoritative. There are no
per-route system atomics or credit spins in the trusted phase-batched path; the
measured credit-stall counter is zero. NCU's local DRAM percentage does not
account for peer-fabric stores, so peer bytes divided by event time is used for
explicitly labeled effective NVLink/P2P bandwidth rather than treating that
metric as HBM bandwidth. Nsight Systems is not installed on this host. The
integrated JSON therefore also carries the audited ordered list of every
memcpy, memset, kernel, collective, signal wait, and terminal completion in the
timed interval; `large_generation_memsets=0` and the CUDA allocation counters
make the setup exclusion machine-checkable.

## Backward MegaKernel validation

Build the release CUDA binary and run the single-node cache/no-cache smoke:

```bash
make ENABLE_CUDA=1 WERROR=1 \
  NCCL_RUNTIME_LIB_DIR=/path/to/nccl/lib build/test_cuda_notify
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  python3 tests/run_local_activation_p2p.py \
    --binary "$PWD/build/test_cuda_notify" --world 8 --backward-smoke

# Empty-expert termination: 100 consecutive world=8/top-k=1 generations.
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  python3 tests/run_local_activation_p2p.py \
    --binary "$PWD/build/test_cuda_notify" --world 8 --backward-smoke \
    --backward-cache --top-k 1 --backward-smoke-repeats 100 --timeout 90

# Two simultaneous main-shape Backward handles and selective slot reuse.
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  python3 tests/run_local_activation_p2p.py \
    --binary "$PWD/build/test_cuda_notify" --world 8 \
    --backward-benchmark --backward-cache \
    --backward-cache-pool2-validation --rounds 11 --timeout 90
```

The Backward sanitizer path bypasses NCCL bootstrap but launches the exact
production world-1 Forward and Backward WGMMA/TMA kernels:

```bash
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  compute-sanitizer --tool memcheck --error-exitcode=1 \
    build/test_cuda_notify --backward-sanitizer
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  compute-sanitizer --tool racecheck --error-exitcode=1 \
    build/test_cuda_notify --backward-sanitizer
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  compute-sanitizer --tool initcheck --error-exitcode=1 \
    build/test_cuda_notify --backward-sanitizer
```

`make ENABLE_CUDA=1 cuda-sanitizer-smoke` includes all three commands. The
main-shape benchmark uses `--backward-benchmark --backward-cache`; its JSON
reports the single persistent launch, BF16 dW dtype, per-stage device clocks,
actual dY/dX peer bytes, complete per-rank pool bytes, and fixed-order hashes
of the complete device dX/dW1/dW2/dGate storage. `gradient_hash_repeats=20`
means all four hashes were byte-stable for twenty independent generations.

The formal training-step field is
`notify_inclusive_training_step_wall_slowest_rank_us`, measured from
route/gate readiness through Notify, Forward, Backward, and the final drain.
`post_notify_training_step_wall_slowest_rank_us` is diagnostic only and must
not be used for a speedup claim. The same JSON separately reports
`notify_wall`, `forward_wall`, `backward_wall`, `handoff`, and `final_drain`;
release runs also require fused/control/config/count/summary audit
`1/0/0/0/0` and dispatch-consumer acquire/release counts `1/1` on every rank.

To enforce the per-run comparison against a Megatron JSON log:

```bash
python3 benchmarks/compare_backward_training.py \
  --megamoe run1.log run2.log run3.log \
  --megatron megatron.log --minimum-percent 15
```

The launcher also emits `critical_path_*` statistics. For each iteration it
first selects the rank with the largest formal Notify-inclusive wall sample,
then takes every Notify component from that same rank and clock domain. This
prevents independently selected per-stage maxima from being misread as one
additive critical path.

The full world-2 production Forward+Backward racecheck command is:

```bash
LD_LIBRARY_PATH=/path/to/nccl/lib:$LD_LIBRARY_PATH \
  python3 tests/run_local_activation_p2p.py \
    --binary build/test_cuda_notify --world 2 --backward-smoke \
    --backward-cache --top-k 6 --backward-smoke-repeats 1 \
    --compute-sanitizer-tool racecheck --timeout 600
```

The former hazard mapped to Forward SASS offsets `+0x1c6b0/+0x1c730`: a
thread-0-only return-row scalar had been placed in cluster shared memory. It is
now a thread-local scalar; the command must report zero errors and hazards.

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
