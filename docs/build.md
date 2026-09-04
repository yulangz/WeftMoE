# Build and validation

## Requirements

- C++17 compiler
- CUDA toolkit with SM90a support
- CUTLASS/CUTE headers (discovered through the installed `deep_gemm` package)
- NCCL headers and library
- Python 3.10+

The current machines need the CUDA compatibility library ahead of the default
driver libraries. A known-good local runtime setup is:

```bash
export MEGAMOE_NCCL_DIR=/path/to/nccl/lib
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$MEGAMOE_NCCL_DIR:/opt/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
```

The driver may label the eight H200 devices as `NVIDIA L20X`; use the SM90a
build target supplied below rather than deriving the architecture from that
display string.

## Clean builds

Use a build directory outside the repository root. CPU/control-plane build:

```bash
CPU_BUILD=/tmp/megamoe-cpu
make BUILD_DIR="$CPU_BUILD" -j8 all
make BUILD_DIR="$CPU_BUILD" -j8 test
```

CUDA build for the target machine:

```bash
CUDA_BUILD=/tmp/megamoe-sm90
make BUILD_DIR="$CUDA_BUILD" ENABLE_CUDA=1 \
  CUDA_ARCH_FLAGS='-gencode arch=compute_90a,code=sm_90a' \
  -j8 cuda-lib "$CUDA_BUILD/test_cuda_notify"
```

## Test layers

Fast source/control checks:

```bash
make BUILD_DIR="$CPU_BUILD" test
python3 tests/test_forward_autotune.py -v
python3 tests/test_backward_autotune.py -v
python3 -m py_compile tests/*.py benchmarks/*.py tools/*.py
```

Core CUDA checks:

```bash
"$CUDA_BUILD/test_cuda_notify" --local 1
"$CUDA_BUILD/test_cuda_notify" --device-smoke
"$CUDA_BUILD/test_cuda_notify" --v3-protocol-smoke
python3 tests/run_control_protocol_regressions.py \
  --binary "$CUDA_BUILD/test_cuda_notify"
```

Single-node multi-process Forward checks:

```bash
python3 tests/run_local_activation_p2p.py \
  --binary "$CUDA_BUILD/test_cuda_notify" \
  --world 8 --forward-smoke --timeout 900

MEGAMOE_FORWARD_BENCH_ROUNDS=110 \
python3 tests/run_local_activation_p2p.py \
  --binary "$CUDA_BUILD/test_cuda_notify" \
  --world 8 --forward-benchmark --top-k 6 --timeout 900
```

For two hosts, build into a path visible at the same absolute location on both
machines. The repository's ignored artifacts tree is on shared storage:

```bash
SHARED_BUILD="$PWD/artifacts/validation/sm90"
make BUILD_DIR="$SHARED_BUILD" ENABLE_CUDA=1 \
  CUDA_ARCH_FLAGS='-gencode arch=compute_90a,code=sm_90a' \
  -j8 "$SHARED_BUILD/test_cuda_notify"

python3 tests/run_multinode_nccl.py \
  --binary "$SHARED_BUILD/test_cuda_notify" \
  --remote user@remote-host --port 22 \
  --remote-binary "$SHARED_BUILD/test_cuda_notify" \
  --gpus-per-node 8 --scenario gin_forward_smoke \
  --nccl-lib-dir "$MEGAMOE_NCCL_DIR" \
  --socket-ifname eth0 --enable-ib --timeout 300
```

Use `gin_forward_bench` for the measured run. The launcher fails if any rank
fails, times out, emits inconsistent correctness hashes, or uses a different
runtime shape.

Cached-state Backward uses the same launchers:

```bash
python3 tests/run_local_activation_p2p.py \
  --binary "$CUDA_BUILD/test_cuda_notify" --world 8 \
  --backward-validation --backward-tokens 8 --rounds 2

python3 tests/run_multinode_nccl.py \
  --binary "$SHARED_BUILD/test_cuda_notify" \
  --remote user@remote-host --port 22 \
  --remote-binary "$SHARED_BUILD/test_cuda_notify" \
  --gpus-per-node 8 --scenario gin_backward_bench \
  --nccl-lib-dir "$MEGAMOE_NCCL_DIR" \
  --socket-ifname eth0 --enable-ib --timeout 600
```

## Autotuning

The maintained plans are:

- `configs/forward_h200_autotune.json`: two-node world 16, GIN/RDMA;
- `configs/forward_h200_world8_autotune.json`: one-node world 8, CUDA IPC/NVLink.
- `configs/backward_h200_autotune.json`: two-node cached-state Backward;
- `configs/backward_h200_world8_autotune.json`: one-node cached-state Backward.

Run them with:

```bash
python3 tools/autotune_forward.py \
  --config configs/forward_h200_autotune.json

python3 tools/autotune_forward.py \
  --config configs/forward_h200_world8_autotune.json

python3 tools/autotune_backward.py \
  --config configs/backward_h200_autotune.json

python3 tools/autotune_backward.py \
  --config configs/backward_h200_world8_autotune.json
```

Each plan also has an explicit selection policy. The maintained plans confirm
the incumbent and the fastest initial challenger twice for 110 rounds each,
reversing their order in the second pass, and only replace the incumbent for
an aggregate improvement of at least 0.5%. The initial and confirmation means
are weighted by their reported measured-iteration counts.
`--confirmation-repeats`, `--confirmation-rounds`, and
`--minimum-improvement-percent` are available for deliberate experiments; the
checked-in JSON remains the source of truth.

Compilation and tuning are both driven at Python runtime. The autotuner does
not call Make: it generates the tuning header, invokes `g++`, `nvcc`, and `ar`
with argument arrays, links the validation executable, inspects its kernel
resources, and then launches the benchmark. To compile one specialization
without measuring it:

```bash
python3 tools/autotune_forward.py \
  --config configs/forward_h200_world8_autotune.json \
  --candidate world8_relay16_tma4 --build-only

python3 tools/autotune_backward.py \
  --config configs/backward_h200_autotune.json \
  --candidate incumbent --build-only
```

`jit.jobs` is a global compilation budget. The compiler gives each active
specialization at least two command lanes for its two heavy Forward translation
units and compiles several candidates concurrently when the budget permits.
The invariant host archive is protected by a process-local lock and published
once into the same content cache.

The JIT design follows DeepGEMM's default NVCC path. A cache key covers the
fully generated header, repository source bytes, the installed
DeepGEMM/CUTLASS include tree, architecture, flags, compiler versions, and
NCCL library location. Compilation occurs in a same-filesystem UUID temporary
directory. Files are flushed and the complete directory is atomically renamed
into the cache, so another process can never observe a partial binary.

All kernel choices remain compile-time constants. Environment variables are
used only by the benchmark harness for workload controls and runtime library
selection; they never select a tuned kernel. Compiler paths and the cache root
can be overridden explicitly with `--nvcc`, `--cxx`, and `--cache-root`.

The JSON result contains:

- the fully expanded candidate configuration and digest;
- build time and SASS resource inspection;
- slowest-rank mean/P50/P95/min/max;
- a correctness fingerprint over inputs, weights, routes, gates, and output;
- initial and confirmation measurements, the threshold decision, and final
  winner-validation timing;
- total autotune duration.

Backward candidates additionally run an untimed world-8 preflight. It executes
Backward twice and fingerprints dX, dGate, dW1, and dW2; only candidates whose
four fingerprints match the incumbent enter performance selection.

JIT artifacts are placed under `artifacts/forward-jit/cache` and
`artifacts/backward-jit/cache`, not in root-level `build-*` directories.
Intermediate objects are deleted before publication;
each candidate retains only its executable, generated header, and provenance
manifest. A cold compile populates the cache, while an identical later request
performs only dependency hashing and resource inspection. The cache may be
removed after the result JSON and selected configuration have been retained.

This path deliberately uses NVCC rather than NVRTC. MegaMOE's specializations
span CUDA kernels, NCCL GIN device code, host transport state, and the
distributed validation entry point. Extracting a standalone cubin ABI would
couple communication-resource lifetime changes to the build-system change.

Compilation is a distinct first phase: all candidates are present in the
content cache before any benchmark starts. This avoids interleaving a cold,
CPU-heavy compile with distributed measurements and gives every candidate the
same benchmark phase conditions.

## ABI and release checks

```bash
RELEASE_BUILD=/tmp/megamoe-release
make BUILD_DIR="$RELEASE_BUILD" ENABLE_CUDA=1 \
  CUDA_ARCH_FLAGS='-gencode arch=compute_90a,code=sm_90a' \
  shared-lib public-symbol-check header-test api-negative-test \
  kernel-artifact-check license-check release-text-check
```

`kernel-artifact-check` verifies the Forward kernel's normalized SASS and
resource record. `public-symbol-check` enforces the installed API allowlist;
the retired Backward implementation symbols must remain absent.

## Profiling

Build with `MEGAMOE_DEVICE_TIMELINE=1` for low-overhead `%globaltimer` events,
then summarize rank-local durations with:

```bash
python3 tools/analyze_device_timeline.py trace.rank*.forward.json \
  --json timeline-summary.json
```

Nsight Compute should profile a world-1 or otherwise self-contained launch.
Freezing one rank of a live distributed persistent pipeline can invalidate its
progress assumptions, so multi-process replay results are not accepted as
end-to-end timing evidence.
