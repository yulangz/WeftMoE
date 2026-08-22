# Staged implementation roadmap

## Phase 1 — rank-local foundation complete

- Strict homogeneous topology and same-local-rank rail mapping.
- Rank-local histogram, asynchronous integer collective contract, and pure
  materialization.
- Canonical row, reverse-route, packet/fan-out, and signal descriptors.
- Identity/generation/provenance-bound workspace planning.
- Independent Python oracle and full-field machine-readable comparison.

## Phase 2 — SM90 Notify backend and real integer collectives

- Device histogram plus deterministic segmented scan using fixed partition
  prefixes rather than arrival-order atomic allocation.
- Real collective backend behind the generic opaque contract. Backend selection
  and stream integration must not leak NCCL/GIN types into metadata headers.
- Device output ownership and asynchronous error propagation tests.
- Byte-for-byte comparison against the existing Python oracle on H200.

## Phase 3 — single-node forward persistent MegaKernel

- Canonical activation dispatch, tile scoreboard, FC1 + SwiGLU, FC2, fixed-order
  combine, and partial-M tiles.
- Release/acquire scoreboard publication and generation-safe workspace reuse.
- H200 tile/worker autotune without changing canonical mapping.

## Phase 4 — multi-node forward

- Same-rail RDMA proxy using the enforced homogeneous topology.
- Bounded ingress rings, credits, sequence numbers, node-level payload dedup,
  and descriptor delivery/retention.
- Packet aggregation tuning with explicit summary counts and unchanged rows.

## Phase 5 — backward persistent MegaKernel

- Reuse the exact forward handle for dY dispatch and dX/dGate combine.
- Cache/no-FC1-output-cache paths, fused SwiGLU backward, dX, dW1, and dW2.
- `split_k=1`, canonical-M dW traversal, and fixed-order reductions.
