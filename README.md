# MegaMOE

MegaMOE is a Hopper-focused Mixture-of-Experts Forward pipeline. It overlaps
Notify, activation dispatch, FC1/SwiGLU, FC2, return, and deterministic
gate-after reduction without a whole-grid barrier between the split transport
and compute kernels.

The current optimized contract is intentionally narrow:

- BF16 activations, weights, and outputs with FP32 gate/reduction arithmetic;
- hidden size 7168, intermediate size 3072, 384 experts, top-k 1 or 6;
- expert M blocks padded to 128 rows;
- one CTA per SM, with compile-time-specialized Hopper TMA/WGMMA kernels;
- homogeneous nodes with a dense `(node, local_rank)` topology;
- one-node CUDA IPC/NVLink and two-node NCCL GIN/RDMA execution.

There is no generic-shape or legacy single-node compute fallback. Unsupported
shapes fail closed. Backward is deliberately unavailable while it is redesigned
around this Forward architecture; `cuda::backward_pipeline_available()` is the
only retained Backward API and currently returns `false`.

## Pipeline

Notify creates immutable canonical route metadata and fixes every dispatch and
return address before the data-plane kernels start. The main path then uses
three disjoint CTA roles:

- RDMA CTAs own cross-node GIN put, signal, and local-completion progress. The
  world-16 production point reserves eight CTAs, one per rail.
- Relay CTAs own node-local CUDA IPC/NVLink movement and readiness publication.
  Row movement uses asynchronous TMA where the topology and alignment permit.
- Compute CTAs exclusively own FC1/SwiGLU, FC2, and final reduction. A shared
  work-conserving queue gives ready FC2 work priority so return starts early.

Producers publish bounded byte ranges with release ordering; consumers acquire
generation-tagged counters or queue entries. FC2 stores directly into the
precomputed return layout. Transport workers observe M-block/chunk readiness,
issue remote puts or local relay copies, and publish source-reduce readiness.
The split kernels do not use an inter-grid `grid.sync()`; their lifecycle is
closed by monotonic generation counters and terminal status records.

Single-node world 8 reuses the same split architecture with zero RDMA CTAs and
a tunable Relay/compute partition. The older combined local kernel selection
has been removed. The host-enqueued NCCL Socket backend remains only as an
explicit correctness fallback and is not a performance path.

See [architecture](docs/architecture.md) for data ownership and synchronization
details.

## Build and test

CPU/control-plane tests:

```bash
make BUILD_DIR=/tmp/megamoe-cpu -j8 test
```

Hopper CUDA build and core smoke tests:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/compat:/opt/.venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
make BUILD_DIR=/tmp/megamoe-sm90 ENABLE_CUDA=1 \
  CUDA_ARCH_FLAGS='-gencode arch=compute_90a,code=sm_90a' -j8 cuda-test
```

Single-node Forward validation:

```bash
python3 tests/run_local_activation_p2p.py \
  --binary /tmp/megamoe-sm90/test_cuda_notify \
  --world 8 --forward-smoke
```

Two-node GIN validation uses a binary at the same shared-storage path on both
hosts and an explicit remote host:

```bash
python3 tests/run_multinode_nccl.py \
  --binary "$PWD/artifacts/validation/sm90/test_cuda_notify" \
  --remote user@remote-host --port 22 \
  --remote-binary "$PWD/artifacts/validation/sm90/test_cuda_notify" \
  --gpus-per-node 8 --scenario gin_forward_smoke --enable-ib
```

The complete command matrix and runtime-library notes are in
[build and validation](docs/build.md).

## Autotuning

Tuning values are explicit JSON and become compile-time constants in a
generated header. The default two-node and single-node searches are deliberately
small:

```bash
python3 tools/autotune_forward.py \
  --config configs/forward_h200_autotune.json

python3 tools/autotune_forward.py \
  --config configs/forward_h200_world8_autotune.json
```

Candidate builds live under the ignored `artifacts/forward-autotune/build`
tree so tuning no longer creates hundreds of root-level `build-*` directories.
The result JSON records expanded parameters, compile/resource evidence,
correctness fingerprints, elapsed time, and slowest-rank latency.

## Repository layout

```text
include/megamoe/                 installed API
src/metadata/                    deterministic Notify materialization
src/cuda/forward/                Forward runtime, launch, and kernel pieces
src/cuda/persistent_v3/forward/  split role pipelines and queues
src/cuda/testing/                non-installed CUDA probes
tests/                           focused control, CUDA, and launcher tests
benchmarks/                      Forward-only MegaMOE/Megatron comparisons
configs/                         explicit autotune plans
tools/                           tuning, SASS, and timeline utilities
```

Benchmark claims and known limits are documented in
[performance summary](docs/performance_summary.md) and
[known limitations](docs/known_limitations.md).
