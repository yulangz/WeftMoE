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

## Autotuning

The two maintained plans are:

- `configs/forward_h200_autotune.json`: two-node world 16, GIN/RDMA;
- `configs/forward_h200_world8_autotune.json`: one-node world 8, CUDA IPC/NVLink.

Run them with:

```bash
python3 tools/autotune_forward.py \
  --config configs/forward_h200_autotune.json

python3 tools/autotune_forward.py \
  --config configs/forward_h200_world8_autotune.json
```

Each candidate receives a generated tuning header and its own content-addressed
build directory. All kernel choices are compiled constants. Environment
variables remain only for benchmark workload controls and runtime library
selection; they do not select a tuned kernel.

The JSON result contains:

- the fully expanded candidate configuration and digest;
- build time and SASS resource inspection;
- slowest-rank mean/P50/P95/min/max;
- a correctness fingerprint over inputs, weights, routes, gates, and output;
- winner and final-validation timing;
- total autotune duration.

Candidate builds are placed under `artifacts/forward-autotune/build`, not in
root-level `build-*` directories. They may be removed after the result JSON and
selected configuration have been retained.

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
