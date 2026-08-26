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
| Non-main hidden/intermediate over GIN | The generic kernels currently reject `use_gin` with a structured device error; the frozen `7168/3072` GIN path is unaffected | no physical non-main-shape GIN execution | none | **Missing** |
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
| Installed public API excludes test/oracle/synthetic hooks | New runtime-shape harnesses are internal, but legacy `testing`, synthetic transport and `*_for_validation` declarations remain in installed CUDA headers | compile-negative coverage protects the newly internalized harness only | not applicable | **Missing** |
| H200 / SM90a runtime validation | SM90/SM90a targets compile; actual machines report L20X / CC 8.9 virtualization | SM90a compile and L20X runtime tests | no H200 measurement | **Environment-limited** |

## Audit findings and disposition

1. The runtime H/I gap is closed by a correctness-only, one-kernel Forward and
   Backward fallback. The frozen `7168/3072` WGMMA/TMA branch is selected by an
   explicit shape predicate and its normalized SASS/resource report is
   unchanged by the refactor.
2. Pointer-bearing Forward and Backward launch contracts and the generic
   fallback are internal `.cuh` modules. The large aggregation translation
   unit remains intentional: splitting the device definitions across separate
   translation units changed compiler visibility and would risk the frozen
   kernel. Public API, runtime pool, and backend ownership stay separate.
3. Runtime-shape test entry points and NCCL-free harness declarations moved to
   `tests/cuda` and `src/cuda/testing`; they are absent from the installed
   production header. The legacy all-mode driver remains for launcher
   compatibility, with new cases added in focused modules.
4. Native CUDA, IPC and GIN handles are confined to internal headers and
   backend state. However, older simulation, fault/oracle and validation
   download declarations still occupy installed CUDA headers. Their names are
   explicit, but naming is not API hiding; completing this cleanup requires a
   compatibility-breaking public-header split and remains a recorded gap.
5. Physical GIN coverage is two homogeneous eight-GPU nodes. Runtime 3+ node
   construction is tested, but physical 3+ node behavior and H200/SM90a
   execution remain environment-limited rather than complete claims.

The generic CUDA-core fallback is fully exercised on world 1 and local P2P
world 2. Its arithmetic is transport-neutral, but the current implementation
explicitly rejects the GIN branch for non-main H/I. Consequently generic
cross-host GIN is an implementation gap, not merely an unavailable-hardware or
test-coverage limitation. The optimized main-shape GIN path remains complete.

## Frozen invariants

The review and fallback work must not change canonical row assignment,
expert-ID combine order, gate-after placement, BF16 cast points, main-shape K
accumulation order, dW split-k=1/canonical-M traversal, or release/acquire
ordering. The checked main-shape hashes and the recorded single-node/world-16
performance results are the regression gates.
