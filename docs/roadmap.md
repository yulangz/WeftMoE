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
- Zero-row experts are ready at setup and receive real persistent-kernel dW
  zero stores. World=8/top-k=1, concentrated routing, an entirely empty owner
  rank, and a zero-token source rank therefore share the ordinary terminal
  predicate. Timeout-only phase dumps replace multi-minute hang diagnosis.
- Complete device dX/dW1/dW2/dGate hashes are checked across 20 generations.
  The main-shape pool=2 test retains both handles, rejects a third generation,
  and proves hash-preserving selective reuse after one handle is released.
- Performance closure remains open, but the current cache-mode candidate has
  crossed the local performance gates. The global dY/X transpose is gone;
  dW1 now uses an outer producer/two-consumer WGMMA pipeline, an STSM BF16
  epilogue, and four 128x64 TMA stores. Deterministic grid-stride ownership for
  the uniform main-shape dW1 and dX tiles removes their global claim atomics.
  The formal integrated boundary is now route/gate readiness through Notify,
  Forward, Backward, and final drain; the former post-Notify number is retained
  only as a diagnostic. Single-node trusted release packs status, expert
  counts, and packet summary into one `NotifyFastRecord` AllGather after the
  session config is verified. Forward moves its existing dispatch consumer to
  Backward, giving exactly one acquire/release across the training step. The
  fair comparator rejects each run unless both Backward-only and this
  Notify-inclusive training step beat the matching Megatron run by at least
  15%. The final three 10-warmup/100-measurement local runs were
  `28.650/28.633/28.691 ms` Backward and
  `46.008/45.973/46.068 ms` Notify-inclusive training-step wall. Against the
  matching Megatron runs this is `15.17/15.47/15.45%` and
  `15.07/15.28/15.10%`, respectively. The fused Notify itself measured
  `0.524/0.528/0.521 ms`; the device terminal rendezvous prevents prior-rank
  skew from being charged to the following generation. These pass the formal
  per-run 15% gate, while the aspirational absolute 28.5/45.6 ms margin remains
  narrowly unmet and is recorded rather than hidden.

  Reserving 1, 2, or 4 CTAs for early dW produced `29.263`, `29.023`, and
  `29.021` ms Backward in the directed short search, all slower than the
  unreserved candidate; those branches and their runtime knob were removed.
  The retained release path only eliminates redundant dW2 CTA barriers and
  keeps the stable all-worker dV/dX then dW schedule.
  No-cache remains slower because FC1 recomputation raises Backward to
  `37.168` ms despite avoiding the Forward checkpoint write.

  The isolated DeepGEMM grouped-GEMM comparison reaches `5.519` ms outside
  profiler replay and uses M128xN256xK64, three stages, 128 TMA plus 256 math
  threads, and a 2-CTA multicast cluster. The production path deliberately
  does not adopt the previously unstable multicast candidate. NCU reports the
  isolated DeepGEMM kernel at 85.61% SM throughput and 14.06% achieved
  occupancy; production world=8 NCU replay cannot currently progress through
  the peer-synchronized persistent kernel, while world=1 profiling exhausts
  memory because that rank owns all 384 experts. Correctness, sanitizer, and
  ordinary timing evidence therefore remain separate from this profiler
  limitation. No Phase-6 completion claim is made until external test handoff.

## Phase 7 — cross-node activation and complete training step (in progress)

- A repository-external NCCL 2.29.7 CUDA-12.9 runtime passes real two-host
  world-2/world-16 device-communicator and registered-window lifecycle probes.
  A dedicated world-2/world-16 probe also passes real same-rail device put,
  remote indexed signal, local completion, and full payload hash validation.
  The primitive is now integrated into a separately typed homogeneous runtime
  backend. Real world-2 and two-node world-16 device dispatch, remote signal,
  same-rail proxy ingress, local CUDA-IPC fan-out, packet-batched synthetic
  contribution return, and gate-after reduce pass 20 deterministic oracle
  checks. Cache Forward and Backward now execute their complete main-shape
  persistent kernels on the same world-16 GIN team, including FC1/FC2,
  dV/dX/dW, dGate and return/reduce.
- The concrete `NcclSocketFallback` passes real world-2 and world-16 NCCL
  Socket Send/Recv for complete cache Forward and Backward, including the
  frozen WGMMA/TMA expert body, aggregate contribution/dX/dGate return, and
  deterministic source reduce. It has no activation/gradient payload D2H/H2D,
  retains explicit host-enqueued phase boundaries, and its backend label cannot
  be selected by enum.
- GIN uses node-level payload dedup, same-rail destination proxies, local P2P
  fan-out and batched reverse traffic. Three independent world-16 10+100 runs
  pass the 15% Forward, Backward and Notify-inclusive F+B gates against the
  real Megatron NCCL Socket baseline; the checked machine-readable result is
  `benchmarks/results/gin_world16_training_20260826.json`.
- The Socket fallback never claims GIN/RDMA or a single persistent cross-node
  data plane. Runtime node counts other than the tested
  homogeneous two-node world-2/world-16 configurations also remain to validate.
