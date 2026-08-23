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
  lease contract, with event completion, device result leases, async-error checks,
  and contextual CUDA/NCCL failures.
- Fixed 256-byte little-endian control AllGather plus an exact canonical-config
  byte AllGather before every data collective; self-checking config envelope,
  exact original session identity, rank-0-baseline error selection,
  enqueue-point four-stage audit, and healthy-communicator recovery.
- Explicit capacity-bound workspace for later forward/backward consumers; no
  hot-path device allocation.
- Single-GPU, local 2/8-GPU, and real two-host 2/16-rank full-field comparisons
  against host production and the Python oracle, plus deterministic repeats.

The correctness and distributed agreement slice is complete.

## Phase 3 — all-device metadata materialization complete

- NCCL count and summary matrices remain in backend-owned device completion
  leases; the production path performs no bulk matrix/route download.
- Checked device-side source-rank/expert prefix, canonical rows, owned-expert
  prefixes, M tiles, reverse routes, multiword masks, packet/fan-out descriptors,
  and expected signals.
- Versioned immutable `DeviceDispatchHandle`, shared arena/completion lifetime,
  checked CUDA typed view, and explicit test-only metadata download.
- Full-field host/Python oracle comparison, 25 fixed seeds, 20 deterministic
  repeats, world 1/2/8 and two-host validation.

The first parallel baseline and session-level preallocation boundary are now
present: bulk routes, incoming rows, tiles, fan-out, and signals are emitted in
parallel from deterministic prefixes, and pooled steady generations perform no
CUDA/pinned/event allocation. Further parallelization of the small checked
planning scan, hierarchical/GIN count exchange, graph capture, and SM90-specific
tuning remain open. These may replace the baseline without changing layout
semantics.

## Phase 4 — single-node forward persistent MegaKernel

- Canonical activation dispatch, tile scoreboard, FC1 + SwiGLU, FC2, fixed-order
  combine, and partial-M tiles.
- Release/acquire scoreboard publication and generation-safe workspace reuse.
- H200 tile/worker autotune without changing canonical mapping.

## Phase 5 — multi-node forward

- Same-rail RDMA proxy using the enforced homogeneous topology.
- Bounded ingress rings, credits, sequence numbers, node-level payload dedup,
  and descriptor delivery/retention.
- Packet aggregation tuning with explicit summary counts and unchanged rows.

## Phase 6 — backward persistent MegaKernel

- Reuse the exact forward handle for dY dispatch and dX/dGate combine.
- Cache/no-FC1-output-cache paths, fused SwiGLU backward, dX, dW1, and dW2.
- `split_k=1`, canonical-M dW traversal, and fixed-order reductions.
