# MegaMOE design compliance

This document audits the implementation against `../MegaMOE-Design.md`. A
status of **Complete** is backed by a production path and a default or recorded
hardware test. **Environment-limited** means the implementation exists but the
required physical topology is unavailable. **Missing** is an implementation
gap; a simulation, capability probe, or compile-only target never counts as a
completed production feature.

| Design requirement | Implementation | Test / evidence | Performance evidence | Status |
|---|---|---|---|---|
| Runtime tokens, experts, top-k, rank/world and homogeneous topology | `RuntimeConfig`, `Topology`, device Notify configuration | host topology rejection tests; CUDA world 1/2/8; real world 2/16 | Notify benchmark JSON and fused/full protocol audits | Complete |
| Runtime hidden/intermediate, including tails | `runtime_shape_forward_megakernel` and `runtime_shape_backward_megakernel`; the frozen `7168/3072` branch remains the WGMMA/TMA specialization | world-1 oracle at 64/32, 96/48, 127/65 and 256/192; real world-2 LocalCudaP2p at 127/65 | correctness-only; not a performance claim | Complete |
| Non-main hidden/intermediate over GIN | The generic kernels consume the same registered GIN windows, same-rail proxy/fan-out, put/signal/completion and deterministic return layouts as the main specialization | physical world 2 at 64/32, 96/48, 127/65, 256/192; physical world 16 at 96/48 and 127/65; top-k 1/6, cache/no-cache, zero rank, empty owner, 20 hashes and pool=2 | correctness-only; 127/65 world-2 diagnostic is recorded separately from main-shape claims | Complete |
| Deterministic Notify local offsets and canonical rows | `device_notify.cu`, host metadata materializer and Python reference | 28 full-field fixtures, 25 fixed CUDA seeds, 20 byte-identical repeats | device Notify stage timings | Complete |
| Source-rank prefix, owned-expert prefix/M-tile metadata, reverse routes and combine order | device materializer and immutable `DeviceDispatchHandle` | C++/Python full-field comparison; row uniqueness/continuity properties | materialization benchmark split | Complete |
| Forward is one persistent data-plane kernel after Notify | `local_p2p_forward_megakernel`; Local, GIN and Socket full-FC adapters | world 1/2/8 and GIN world 2/16; launch audit equals one | single-node and world-16 Forward results | Complete |
| Backward is one persistent data-plane kernel after Forward | `local_p2p_backward_megakernel` | cache/no-cache, world 1/2/8 and GIN world 2/16; launch audit equals one | single-node and world-16 Backward results | Complete |
| Node dedup, same-rail proxy and local fan-out | Notify packet descriptors plus GIN rail/LSA teams and CUDA IPC peer table | GIN world 2/16 counters and Socket/G​IN output/hash comparison | per-rail put/signal/byte counters | Complete |
| Runtime arbitrary node count | topology, descriptor, rail and window sizing use runtime `num_nodes`; 3-node/4-node descriptor tests exist | physical execution is limited to the available two nodes | no physical 3+ node result | **Environment-limited** |
| Bounded ring, channel, credit, sequence and epoch | transport pool layouts and typed generation leases | wrap/credit/epoch tests and long-running local transport tests | ring high-water and credit counters | Complete |
| Dedicated Comm/Relay/Compute/Combine roles and role conversion | Forward/Backward scheduler state and completion counters | persistent progress, top-k 1/6, empty-expert and timeout diagnostics | stage overlap/device timestamp reports | Complete |
| Rule-based scoreboard, ready/claim state and no generic ready queue | scheduler/claim arenas and expected/completed counters | task counts, duplicate-claim guards, empty expert completion | worker/stage audit fields | Complete |
| Cache and no-cache DispatchContext lifetime | owning Forward/Backward handles and pool slots | byte-identical cache/no-cache, checkpoint oracle, pool=2 isolation | cache cost/net training-step measurements | Complete |
| dW split-k=1, canonical-M traversal and no float atomics | Backward WGMMA dW1/dW2 task ownership | stride regression, FP64 samples, repeated full gradient hashes | dW1/dW2 stage timings and roofline harness | Complete |
| Forward cross-topology bitwise determinism | canonical route slots and expert-ID ordered FP32 combine | EP1/2/8 fixture with identical actual input/weight/route/gate hashes | timing excluded from determinism check | Complete |
| Backward same-topology bitwise determinism | canonical-M dW and expert-ID dX combine | 20 full-device gradient hashes; cache/no-cache comparison | timing excluded from hash checks | Complete |
| Real GIN/RDMA data plane | backend-private NCCL device comm, registered windows, put/signal/completion | compatible NCCL 2.29.7 world 2/16 initialization and full F/B/F+B | checked world-16 GIN benchmark | Complete |
| Host-enqueued Socket correctness fallback | aggregate NCCL Send/Recv plus device relay/compute | real world 2/16 Forward/Backward and GIN byte comparison | separately labelled Socket lower bound | Complete |
| Workspace query, epoch, completion and two live generations | session pool, slot lease, consumer lease and completion events | exact/short capacity, pool exhaustion/reuse/quarantine, pool=2 | allocation audit: no steady allocation/event creation | Complete |
| Autotune cannot change canonical/reduction invariants | frozen main-shape specialization and recorded tuning parameters | output/gradient hashes are checked for every accepted candidate | recorded top candidates and final main-shape configuration | Complete |
| Installed public API excludes test/oracle/synthetic hooks | API v2 installs a fixed production-header manifest; internal declarations require `src/cuda/testing/api.hpp` or `src/metadata/testing_api.hpp`, neither of which is installed | temporary-DESTDIR compile/link/run, 12-header manifest, 35 compile-negative retired/internal entries | not applicable | Complete |
| H200 / SM90a runtime validation | SM90/SM90a targets compile; actual machines report L20X / CC 8.9 virtualization | SM90a compile and L20X runtime tests | no H200 measurement | **Environment-limited** |

## Audit findings and disposition

1. The runtime H/I gap is closed by a correctness-only, one-kernel Forward and
   Backward fallback. The frozen `7168/3072` WGMMA/TMA branch is selected by an
   explicit shape predicate and its normalized SASS/resource report is
   unchanged by the refactor. Across sm89/sm90/sm90a, normalized main Forward
   SASS hashes and resources are generated by
   `tools/kernel_artifact_manifest.py`. The runtime `sm_90` Forward/Backward
   hashes are `05d9f47c...ffb5236` and `fdbb45ca...64ee8a9`; compile-only
   `sm_90a` is tracked separately. `make kernel-artifact-check` rejects any
   unreviewed change.
2. Pointer-bearing Forward and Backward launch contracts and the generic
   fallback are internal `.cuh` modules. The large aggregation translation
   unit remains intentional: splitting the device definitions across separate
   translation units changed compiler visibility and would risk the frozen
   kernel. Public API, runtime pool, and backend ownership stay separate.
3. Runtime-shape harnesses, simulation, validation downloads, hash/mutation
   helpers and fault seams require a repository-internal testing header. API
   v2 installs only production declarations; an installed-prefix-only build
   cannot name the hidden symbols or add `src/` to its include path.
4. Native CUDA, IPC and GIN handles remain confined to internal headers and
   backend state. The public boundary consists of typed leases, owning handles,
   session/pool configuration, backend capability/selection and production
   status/error records.
5. Physical GIN coverage is two homogeneous eight-GPU nodes. Runtime 3+ node
   construction is tested, but physical 3+ node behavior and H200/SM90a
   execution remain environment-limited rather than complete claims.

The generic CUDA-core fallback is exercised on world 1, LocalCudaP2p world 2,
and real GIN world 2/16. It remains correctness-only; the optimized main-shape
GIN specialization and its performance evidence are unchanged.

## Frozen invariants

The review and fallback work must not change canonical row assignment,
expert-ID combine order, gate-after placement, BF16 cast points, main-shape K
accumulation order, dW split-k=1/canonical-M traversal, or release/acquire
ordering. The checked main-shape hashes and the recorded single-node/world-16
performance results are the regression gates.
