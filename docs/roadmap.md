# Staged implementation roadmap

## Phase 1 — complete

- Runtime topology and shape contracts.
- Deterministic Notify Dispatch metadata in C++.
- Pluggable count exchange with delivery-order tests.
- Immutable forward/backward dispatch handle.
- Runtime workspace cardinality plan.
- Independent Python oracle, unit tests, and host benchmark.

## Phase 2 — SM90 Notify backend and real count transport

- Implement device local histogram plus deterministic segmented scan. Parallel
  work may compute counts, but final offsets must be derived from fixed route
  partitions/prefixes rather than atomic return order.
- Add NCCL GIN/LSA count exchange behind `CountExchange`, including same-rail
  inter-node exchange and node-local propagation.
- Compare all CUDA metadata buffers byte-for-byte with the Python reference on
  SM90, including irregular topologies and capacity failures.
- Add epoch/generation workspace state and stream-safe handle lifetime.

## Phase 3 — forward single-node persistent MegaKernel

- Canonical activation dispatch, tile scoreboard, FC1 + SwiGLU, FC2, fixed-order
  combine, and partial-M tiles.
- Separate Comm/Relay/Comp roles with release/acquire publication.
- H200 profiling and tile/worker autotune without changing deterministic
  accumulation or row semantics.

## Phase 4 — multi-node forward

- Same-rail RDMA proxy, node-level payload dedup, bounded per-source ingress
  rings, credits, sequence numbers, and local-completion-safe reuse.
- Arrival-order stress and topology-crossing forward bitwise tests.

## Phase 5 — backward persistent MegaKernel

- Reuse the forward handle for dY dispatch and dX/dGate combine.
- Cache and no-FC1-output-cache modes, fused dU/SwiGLU backward, dX, dW1/dW2.
- `split_k=1`, canonical-M dW traversal, fixed-order dGate/dX reductions.
