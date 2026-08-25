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
  byte AllGather before the first data collective (and every debug generation);
  release sessions reuse the immutable verified config; self-checking envelope,
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

The release planner now parallelizes expert/source totals, owner prefixes,
destination capacities and deterministic node packet scans. Session slots cache
the first exact static-config agreement; debug repeats full agreement every
generation. Pooled steady generations perform no CUDA/pinned/event allocation.
Hierarchical/GIN count exchange, graph capture, and SM90-specific tuning remain
open without changing layout semantics.

## Phase 4 — activation transport data plane (in progress)

Completed correctness components:

- typed BF16 activation provenance and a session-level transport pool;
- bounded per-link ring slots, monotonic sequences, credits, node-level packet
  descriptors, canonical relay, independent contribution slots, and
  expert-order FP32/BF16 combine;
- transport-only persistent CUDA worker, 73-node device simulation, a real
  multi-process local CUDA IPC dispatch/relay/return/combine path, deterministic
  independent Python oracle, and a last-block P2P ordering probe;
- trusted release all-SM bulk route/row/token grids with direct canonical writes
  and per-peer phase completion; debug retains the ring/credit worker;
- a checked-in Megatron-Core `MoEAlltoAllTokenDispatcher` comparison harness
  and same-generation Notify-to-P2P integrated release benchmark;
- immutable per-route FP32 gate provenance and the frozen source-side
  gate-after identity semantics. Dynamic gates remain local to their source
  and unit-gate compatibility runs execute the identical reduction path;
- an explicit GIN capability probe and fail-closed backend boundary.

Still required before phase 4 closes:

- implement and validate the host-enqueued NCCL Send/Recv correctness fallback
  across two hosts, including combine return;
- enable and validate true NCCL GIN put/signal/local-completion after a
  compatible runtime/plugin/device-comm setup is available;
- extend local P2P cancellation/quarantine fault injection and two-live-
  generation stress beyond the current all-rank cancellation boundary.

## Phase 5 — single-node forward compute integration (baseline implemented)

- One cooperative persistent data-plane launch now contains DispatchPut,
  three-stage dual-projection TMA/WGMMA FC1, SwiGLU, four-stage N=256
  TMA/WGMMA FC2, vectorized CombinePut and fixed-order TopKReduce.
- Runtime expert rows and partial M blocks use device task counters; FC1/FC2
  tasks retain the phase scoreboards and CAS claim validation in debug builds.
- `CACHE_FC1_INPUT_OUTPUT` and `NO_FC1_OUTPUT_CACHE` share identical Forward
  arithmetic. The former checkpoints BF16 A/B; neither mode retains transient
  S or FC2 outputs for backward.
- Main-shape single-node world 1/2/8 correctness and the real LocalCudaP2p
  benchmark are available. Other H/I shapes, cross-node Forward, backward,
  TMA multicast, and final worker/tile autotuning remain open.

The current main-shape release specialization passes the phase performance
gate on the inspected eight-GPU host: three independent 10-warmup/100-sample
runs with the shared actual-device-byte fixture measured 15.2538, 15.2652,
and 15.2611 ms versus Megatron native 18.4617, 18.6098, and 18.4789 ms, or
17.38%, 17.97%, and 17.41% improvement. The implementation is bitwise stable across
EP1/EP2/EP8, cache modes, build modes, and repeats, and the independent NumPy
FP64 oracle meets the fixed BF16/WGMMA tolerance.
Phase 5 is ready for verification; clustered weight multicast/weight reuse and
further WGMMA dependency reduction remain performance follow-ups rather than
correctness blockers.

## Phase 6 — backward persistent MegaKernel (correctness baseline implemented)

- The exact owning Forward handle now drives typed dY dispatch and dX/dGate
  return/combine in one cooperative persistent launch.
- Cache and no-cache paths share fused dV/gate/SwiGLU backward, dX, dW1 and
  dW2. No-cache recomputes A/B from canonical X without redispatching X.
- Main-shape GEMMs use TMA/WGMMA. dW is `split_k=1`, one CTA owns each output
  tile, canonical padded-M order is fixed, and dGate/dX reductions avoid
  floating atomics.
- A NCCL-free production-kernel harness covers memcheck, racecheck and
  initcheck. Single-node world 1/8 cache/no-cache correctness and steady
  allocation auditing are available.
- Performance closure remains open. Independent FP64 checks found that the
  earlier direct MN-major dW path produced incorrect nonzero-column gradients,
  so its 31.9 ms measurements are invalid. The corrected K-major path is exact
  on the selected BF16 dW samples and reports max absolute oracle error
  `3.51e-8` (from FP32 dGate). After fusing U/dAB transpose and using a
  coalesced 256x256 dY/X transpose, one 10-warmup/100-sample local 8-GPU run
  measured 41.538 ms Backward and 58.089 ms integrated Forward+Backward. This
  does not beat the matching 33.700--33.851 ms Megatron Backward baseline;
  dW (13.870 ms), dV (7.917 ms), dX (7.642 ms), and the remaining transpose
  (7.754 ms) are the next profiling targets. No Phase-6 completion claim or
  release commit is made until every independent run clears the 15% gates.
