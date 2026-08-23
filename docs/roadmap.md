# Staged implementation roadmap

## Phase 1 — rank-local foundation complete

- Strict homogeneous topology and same-local-rank rail mapping.
- Rank-local histogram, asynchronous integer collective contract, and pure
  materialization.
- Immutable provenance-bound phase handles, owned session/execution tokens,
  typed host storage, opaque device leases, and a move-only operation adaptor.
- Validated completion leases retain the full operation resource chain through
  the final validated-handle copy; device leases enforce checked shape/capacity.
- Host control-plane corruption and hostile-backend failures are contained as
  checked exceptions under ASan/UBSan.
- Canonical row, reverse-route, packet/fan-out, and signal descriptors.
- Identity/generation/provenance-bound workspace planning.
- Independent Python oracle and full-field machine-readable comparison.

## Phase 2 — device Notify and real integer collectives (correctness baseline)

- Runtime CUDA route validation, explicit total-order radix sort, segmented
  scan, deterministic local offsets, device histograms, topology coordinates,
  multiword node masks, and token-node summary dedup.
- Real NCCL AllGather correctness backend behind the opaque session/execution/
  lease contract, with event completion, typed D2H result, async-error checks,
  and contextual CUDA/NCCL failures.
- Fixed 256-byte little-endian control AllGather plus an exact canonical-config
  byte AllGather before every data collective; rank-0-baseline error selection,
  four-stage audit, and healthy-communicator recovery.
- Explicit capacity-bound workspace and complete device metadata staging for
  later forward/backward consumers; no hot-path device allocation.
- Single-GPU, local 2/8-GPU, and real two-host 2/16-rank full-field comparisons
  against host production and the Python oracle, plus deterministic repeats.

The correctness slice is implemented; acceptance remains focused on exhaustive
multi-host protocol replay and performance evidence. Still open inside the
Notify area: optimized device-only global prefix and
descriptor materialization, hierarchical/GIN count exchange, graph capture,
and SM90-specialized performance tuning. The current controlled D2H plus host
materializer is the correctness boundary, not a claimed final data plane.

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
