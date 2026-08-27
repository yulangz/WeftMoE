# MegaMOE

MegaMOE targets deterministic training MoE execution on H200/SM90 and later
devices. The current vertical slice implements rank-local deterministic Notify
Dispatch in production C++/CUDA, a fixed-width distributed control agreement,
a real NCCL AllGather count backend, opaque device metadata leases, and an
independent Python oracle. Phase 4 adds a BF16 activation data plane
baseline: typed router activation leases, a debug sequence/credit ring,
canonical direct writes, batched synthetic contribution return, and fixed
expert-order combine. `LocalCudaP2pActivationTransport` now uses real per-rank
CUDA IPC mappings and independent rank processes; the explicit
`DeviceActivationTransport` remains a simulation-only oracle. Phase 5 now has
a single-launch, single-node Forward baseline for the main 7168x3072 shape:
TMA/WGMMA FC1, an in-kernel SwiGLU epilogue, TMA/WGMMA FC2, peer return, and
source-side dynamic-gate deterministic reduce all execute in one cooperative
persistent kernel. Phase 6 adds a single-launch, single-node Backward baseline
for the same main shape, including cache/recompute paths, dY dispatch,
WGMMA/TMA dV/dX/dW1/dW2, deterministic dGate, and fixed-order dX combine.
Phase 7 adds a concrete, opt-in
`NcclSocketFallbackActivationTransport`: it performs host-enqueued NCCL
Send/Recv of GPU-resident BF16 payloads across two hosts, device relay,
synthetic return, and source-side dynamic-gate reduction without activation
D2H/H2D. The separately typed `NcclSocketFallbackForwardMegaKernel` and
`NcclSocketFallbackBackwardMegaKernel` extend that backend through the frozen
WGMMA/TMA FC1/FC2 and dV/dX/dW cores, aggregate dX/dGate return, and fixed-order
source reduce. They retain explicit host-enqueued payload phase boundaries and
remain a correctness/lower-bound backend. A separately typed
`GinRdmaActivationTransport` now runs full cache Forward and Backward with
device-issued NCCL GIN put/signal/local-completion inside the existing
persistent data-plane kernels. Real two-node world-16 execution uses eight
same-rail proxies, node-deduplicated ingress, CUDA-IPC local fan-out to
canonical rows, WGMMA/TMA FC/dW, packet-batched return, dGate, and deterministic
source reduction. Socket and GIN full Forward/Backward actual device hashes
match on real world-2 and world-16 runs for top-k 1/6. Arbitrary node counts
beyond the tested two-node homogeneous topology remain open validation work.

Hidden and intermediate sizes are runtime values. `7168/3072` selects the
frozen SM90 WGMMA/TMA performance specialization; all other positive sizes use
a correctness-only CUDA-core fallback that still executes Forward in one
persistent kernel and Backward in one persistent kernel. World-1 oracle tests
cover `64/32`, `96/48`, `127/65`, and `256/192`, including M/N/K tails, and
`127/65` is also exercised through the real two-process LocalCudaP2p path. The
same kernels run through real GIN at world 2 for 64/32, 96/48, 127/65 and
256/192 and at world 16 for 96/48 and 127/65, including cache/no-cache,
top-k 1/6 and deterministic gradients. The fallback carries no performance
claim; `7168/3072` still selects the frozen WGMMA/TMA path.

## Supported topology contract

The current transport model supports homogeneous nodes only:

- node IDs are dense `0..num_nodes-1`;
- every node has exactly the same runtime GPU count `G`;
- each node contains every `local_rank` exactly once in `0..G-1`;
- same-rail proxying maps a source GPU to the destination-node GPU with the
  same `local_rank`.

Heterogeneous nodes, empty nodes, missing rails, duplicate coordinates, sparse
local ranks, and out-of-range node/expert ownership are rejected explicitly.

## Rank-local production flow

The host control-plane path never asks one rank to provide every rank's routes:

```cpp
auto session = make_collective_session("comm-A", world_size, generation,
                                       backend_owned_session_state);
auto execution = make_execution_token("stream-7",
                                      backend_owned_execution_state);
auto context = make_rank_collective_context(session, calling_rank, execution);
auto local = build_rank_local_histogram(config, context, local_routes);

auto count_op = start_count_exchange(local, backend);
auto summary_op = start_notify_summary_exchange(local, backend);
// In a real collective all ranks start each corresponding round.
auto counts = count_op.wait();
auto summary = summary_op.wait();

auto handle = materialize_rank_metadata(local, counts, summary);
```

The three stages are:

1. Local stable histogram/scan over only `(local_token_id, topk_slot)`.
2. Asynchronous integer all-gathers for expert counts and packet summaries.
3. Pure rank-local metadata materialization with no backend calls.

`LocalHistogramHandle` and `ValidatedCollectiveHandle` are immutable opaque
handles. `CollectiveOperation` is move-only and retains local typed storage,
session/execution tokens, and the backend operation through a terminal state.
Successful `wait()` transfers shared ownership through an immutable completion
lease, so every validated-handle copy keeps those resources alive. Its
destructor performs best-effort cancellation and never propagates cleanup
exceptions.

`build_notify_dispatch_for_testing` is a host reference/test wrapper that
composes this path for all ranks. It is not a device entry point.

The CUDA production path keeps `selected_experts[num_tokens, top_k]` on the
calling GPU. It derives an explicit total-order key
`(expert_id, flat_route_index)`, performs radix sort plus segmented inclusive
scan, and scatters the stable local offsets. Atomic return values are never used
as route offsets. A CUDA/NCCL session then AllGathers the device expert-count and
token-node summary rows into backend-owned device completion leases. A checked
device kernel consumes those matrices directly and writes canonical rows,
owned-expert/tile geometry, reverse routes, packet fan-out, and signals into one
preplanned device arena. No route/count matrix is bulk-downloaded, no host
materializer runs, and no final metadata is bulk-uploaded on this path.

```cpp
auto execution = cuda::make_cuda_execution("notify-stream", device);
auto base_session = cuda::make_nccl_session("ep-comm", world, rank,
                                            first_generation, device,
                                            bootstrap_id);
auto pool = cuda::make_device_notify_resource_pool(
    base_session, execution,
    {max_shape_config, max_tokens_per_rank, max_config_wire_bytes,
     max_inflight_generations, max_inflight_consumers});
cuda::SelectedExpertRouteLease<std::int32_t> routes =
    router_output.selected_experts();
auto session = cuda::make_nccl_generation(base_session, "ep-comm",
                                          generation);
cuda::CudaNotifyGenerationCoordinator coordinator(session, execution, pool);
auto generation_result = coordinator.run_device(
    {config, routes, max_tokens_per_rank});
const auto& device_handle = generation_result.dispatch();
auto consumer = device_handle.acquire_consumer(execution,
    {generation, rank, world, local_routes.num_tokens, config.top_k,
     device_handle.identity().config_digest_word0,
     device_handle.identity().config_digest_word1});
forward_launch_adapter.submit(consumer); // adapter alone sees the kernel POD
consumer.finish();
```

`DeviceDispatchHandle` is immutable and copyable through a shared arena and
completion lease. Public code acquires a move-only owning consumer lease; only
friend launch adapters can form the short-lived pointer POD. The lease is bound
to generation, rank/world, runtime shape, config digest, execution identity,
slot epoch and a precreated completion event. Default/moved-from and
cross-generation consumers fail closed. Repository tests may explicitly use
the non-installed validation download declared by `src/cuda/testing/api.hpp`;
the coordinator and installed API cannot call it.

`DeviceNotifyResourcePool` preallocates every route slot, Notify arena, pinned
status region, NCCL control/config/count/summary slot, CUB temporary region,
and CUDA event at session setup. A generation acquires one exclusive slot and
monotonic epoch. The slot lease is shared by the workspace and every published
handle copy; only destruction of the last copy, followed by successful stream
completion drain, returns the slot. Pool exhaustion reports
`ResourceExhausted` and never falls back to an allocation. The lease strongly
owns the pool implementation, so a public pool object may be destroyed before
its handles. A failed drain leaves the slot unavailable (fail closed).

`run_device` accepts only `SelectedExpertRouteLease<std::int32_t>`, whose
dtype/layout, contiguous `[tokens,top_k]` shape, device, owner, pipeline
session/generation, producer execution and recorded ready event are immutable.
The generation comparison is unconditional: a lease issued for generation N
is rejected by N+1 even when its allocation is unchanged. Reusing immutable
test storage is implemented only by a private adapter that issues a fresh,
exact-generation lease; no test handoff exists in installed headers.
Cross-stream use requires `import_selected_expert_routes`, which enqueues a
stream wait while retaining producer provenance. Generic `DeviceBufferLease`
cannot be rebound as routes. `run` is an explicit host convenience path: with
a pool it uploads to the already allocated route slot. Test upload/reissue
utilities live only in private test headers. The public
production coordinator constructor requires a pool; the allocating constructor
is private and reachable only through an explicitly named `testing` factory.

This coordinator is the only public production protocol entry; device upload,
workspace allocation, preflight wait, control start, and data-collective start
are internal. With `MEGAMOE_DEBUG_VALIDATE=1`, every generation executes
exactly `dataized local preflight →
256-byte ControlHeader AllGather → header agreement → self-checking canonical
config wire AllGather → exact byte/field agreement → expert-count AllGather →
packet-summary AllGather`. The header is explicitly little-endian encoded as
32 words; no native struct or padding is copied to the wire. It is fully
initialized even for invalid local input.
The trusted-participant release build caches the first successful exact config
agreement in each session/backend slot. On a homogeneous single node, later
generations pack status/control, generation, the expert-count row, and the
packet-summary row into one preallocated fixed-layout `NotifyFastRecord` and
execute exactly one NCCL AllGather. The gathered count/summary payload stays on
device; only bounded control headers are staged for agreement. The collective,
unpack, and device materialization are ordered on one stream and share the
materialization completion wait. Cross-node release and every debug generation
retain the full control/config/count/summary protocol. Config mismatches and
route errors select the lowest origin rank and first field in schema order; all
ranks fail after the same fixed collective. A healthy communicator can be
reused through a new immutable generation view after a rejected generation.

The two header session words are diagnostic summaries, never the agreement
boundary. `CollectiveSession` owns bounded original stable-identity bytes. The
NCCL backend derives them from the shared `ncclUniqueId` bootstrap bytes plus
an explicit protocol namespace; neither object addresses nor `ncclComm_t`
values participate. Those original bytes are length-prefixed into canonical
config and compared exactly. Logs expose only display names, lengths, and safe
summaries—not bootstrap bytes.

The multinode launcher transports the raw `ncclUniqueId` only as a fixed-width
binary secret: rank 0 writes to an inherited private pipe, local peers read
dedicated stdin pipes, and remote peers read SSH stdin. Bootstrap material is
never placed in argv, environment variables, shell commands, stdout/stderr, or
temporary files. Aggregated diagnostics are additionally scrubbed for raw,
arbitrary-case hex, standard/URL-safe padded or unpadded base64, and the retired
text marker form. The same rules apply to every nonempty prefix received before
a malformed or truncated frame fails. On the C++ side, ownership of each pipe
descriptor is transferred to a move-only one-shot channel. The channel closes
on every terminal path and cannot be read or written again; in particular, byte
129 makes the input terminal immediately rather than leaving residual bytes for
a second frame.

Canonical config is an explicit little-endian envelope: `magic`, wire version,
64-byte header length, payload length, two zero reserved words, payload
checksum, normalized wire digest, then the canonical payload. Decoding checks
the 64 MiB bound, every length/add/multiply, reserved words, checksum, digest,
truncation, and trailing bytes before reading payload fields. Each gathered row
must also match its own `ControlHeader.config_digest`, including world=1.
Different declared row lengths use the maximum gathered length as the safe NCCL
slot width and are reconstructed by their declarations before exact comparison.
Digest equality never skips exact decoded-field and byte agreement.

`max_tokens_per_rank` is an explicit capacity promise for every rank in the
collective generation. It sizes worst-case incoming rows; device-side checked
prefix/materialization rejects a short capacity before publishing a handle.

## Deterministic mapping and descriptors

```text
local_offset(t,k) = stable occurrence index in calling rank
rank_base(r,e)    = sum(C[q,e] for q < r)
dst_row(t,k)      = expert_base(e) + rank_base(r,e) + local_offset(t,k)
```

The handle retains only:

- this rank's outgoing routes and reverse-combine identity;
- this rank's owned-expert incoming count/prefix/tile/row geometry;
- the global integer count matrix needed for prefixes;
- this rank's outgoing packets and proxy signal cardinalities.

Baseline packetization is one payload per `(local_token_id, destination_node)`.
Packets are ordered by `(destination_node, local_token_id)`; packet identity is
`(source_rank, destination_node, sequence_on_link)`. Fan-out is sorted by expert
ID and references the precomputed canonical row. Future aggregation may change
packet boundaries but cannot change route destination or canonical row.

## Activation transport boundary

`SelectedActivationLease<BFloat16>` is the router-to-transport boundary. It
binds contiguous `[tokens,hidden]` BF16 storage to the producer session,
generation, execution, device, owner, and ready event without exposing a
native address. `ActivationTransportResourcePool` preallocates every input,
ring, canonical-row, contribution, combine, status, and CUDA-event slot. A
generation has one epoch lease; exhaustion is `ResourceExhausted`, and a
failed completion quarantines rather than reuses a slot.

The local P2P backend exchanges checked CUDA IPC handles through a device NCCL
setup collective, opens only the one-node peer team, and retains every mapping
until the final output lease dies. Before each generation, ranks exchange and
validate the chosen pool-slot index and epoch. The release path launches
all-SM bulk grids for direct route-to-canonical-row writes, destination-side
synthetic contribution generation, peer return, and token reduction. It emits
one system-scope completion per peer and phase, not one per route. The debug
path retains the bounded ring/credit/sequence worker and full descriptor
validation. Stores precede a system fence and release signal; consumers
acquire-observe the signal before use. Reduction follows device
`combine_order` (expert-ID order), accumulates in FP32, and casts once to BF16.
The synthetic transform is only a transport oracle and is not FC1/FC2.

## Single-node Forward baseline

`LocalCudaP2pForwardMegaKernel` accepts the immutable dispatch handle, typed
BF16 activation/weight leases, and the dispatch-owned FP32 gate lease. For the
main `hidden=7168, intermediate=3072` specialization it launches one
384-thread CTA per SM. A 2-CTA cluster is requested where supported, but the
runtime audit on the current performance path reports an actual cluster size
of one. The single cooperative launch performs
DispatchPut, three-stage TMA/WGMMA FC1 for both projections, BF16 SwiGLU
materialization, four-stage N=256 TMA/WGMMA FC2, vectorized peer return, and
fixed expert-ID source reduction. FC1 and FC2 have no intervening kernel or
host scheduling boundary. The output lease owns the transport slot until the
last copy dies. Slot reuse is safe because every source acquire-observes every
expected peer return signal before reduction, and local grid completion proves
that all outgoing peer writes and local reads have finished; release mode does
not pay for a second terminal team barrier.

The two forward modes share the same arithmetic path. `NO_FC1_OUTPUT_CACHE`
retains canonical dispatch X and transient S only for the live generation;
`CACHE_FC1_INPUT_OUTPUT` additionally checkpoints the BF16 A/B projections for
the persistent Backward kernel. Both cast SwiGLU output to BF16 before FC2,
apply the per-route FP32 gate at the source after return, accumulate top-k in
device `combine_order`, and cast the final sum once to BF16. No S, ungated FC2
output, or peer contribution is retained beyond the Forward lease.

The correctness fixture is shared byte-for-byte with the Megatron baseline.
Activation, selected-route, FP32 gate, and sampled weight hashes are exchanged
and checked per rank before timing. Weights are rebuilt from an expert-aware
hash of `(global_expert, tensor_kind, row_group, column)`; W1 and W2 differ,
expert strides no longer collide, and the small BF16 range avoids SwiGLU
saturation. The EP1/EP2/EP8 fixture keeps all 384 global experts and changes
only contiguous ownership.

Forward completion signals have explicit lifetime semantics: pool setup clears
the backing allocation once, while each generation initializes only its
owner-local signal words to the new epoch and completes a device peer
ready/ack handshake before DispatchPut. No generation depends on `cudaMalloc`
contents or a prior generation's signal value.

The release benchmark uses the `sm_90` WGMMA/TMA cubin selected by the CUDA
runtime on the inspected L20X host; `sm_90a` is compile-only on this hardware,
and the fat binary also contains an ordinary `sm_89` target. The
launch audit records requested and actual cluster sizes separately. The
standalone frozen Forward result records `cluster_size=2`/rank mask `3`; the
current deferred Forward-to-Backward integrated path records actual size one.
TMA multicast remains disabled (`audit=0`) in both. Other positive H/I shapes
select the single-CTA correctness fallback described above.

## Single-node Backward baseline

`LocalCudaP2pBackwardMegaKernel` consumes the owning Forward handle and a typed
BF16 `[tokens, hidden]` output-gradient lease. One cooperative persistent
launch performs dY peer dispatch, dV, source-side gate multiplication,
SwiGLU backward, dX, split-K=1 dW1/dW2, peer dX/dGate return, and deterministic
top-k dX reduction. Main-shape GEMMs use TMA/WGMMA with FP32 accumulators;
dX/dW are stored as BF16 and dGate as FP32. dW tiles have one owner and traverse
each expert's canonical padded-M segment in increasing order, so no floating
atomic participates in a result.

`CACHE_FC1_INPUT_OUTPUT` reuses Forward's BF16 A/B checkpoint.
`NO_FC1_OUTPUT_CACHE` recomputes A/B from canonical X inside the same Backward
kernel; it does not redispatch X. U, dAB, dGate partials, returned dX, and dW
are generation-slot storage owned through the Backward handle. The resource
pool must be created with `enable_backward=true`; the steady path performs no
CUDA/pinned/event allocation or device-wide synchronization.

Zero-row owned experts are first-class Backward inputs. Generation setup marks
their dY/U/dAB readiness without waiting for a producer, and the persistent
kernel still owns deterministic vectorized dW1/dW2 zero-store tasks. Thus an
all-empty owner rank participates in terminal completion and exposes real zero
gradient storage instead of an implicit epoch convention. A timeout-only
mapped progress page can report per-rank/per-expert terminal state when
`MEGAMOE_BACKWARD_PROGRESS_DUMP=1`; it is absent from the normal release path.
After the final dX reduction, each rank release-publishes the generation epoch
to a dedicated peer terminal vector and acquire-waits for the complete local
vector. This device-only final drain neither launches a host collective nor
reuses the setup ready/ack words; the next generation therefore starts from a
closed all-rank boundary.

Validation can hash the complete device dX/dW1/dW2/dGate allocations in a
fixed order without bulk downloads. The benchmark repeats those hashes 20
times, and the pool=2 lifecycle test retains both Backward handles, proves
their arenas disjoint, observes `ResourceExhausted` for a third generation,
then verifies selective slot reuse without changing the still-live hash.

The NCCL-free `--backward-sanitizer` harness executes the exact production
Forward and Backward kernels with an initialized world-1 fixture. It is part of
`make cuda-sanitizer-smoke` for memcheck, racecheck, and initcheck.

The GIN probe reports headers, the full required symbol set, distributed
runtime initialization, backend implementation, and usability independently.
The default NCCL 2.28.3 runtime lacks the required lifecycle symbols. An
official NCCL 2.29.7 CUDA 12.9 package, explicitly loaded from a repository-
external cache, has passed real two-host world-2 and world-16 device-comm plus
collective registered-window lifecycle probes. A separate validation probe
also passes real same-rail device-issued put, remote indexed signal, and local
completion counter on world 2 and 16. The activation backend registers only
its network windows with `ncclMemAlloc` and passes 20 deterministic world-2
and world-16 generations with every rank sending. World-16 uses eight rail
peers plus seven node-local IPC mappings per rank. The project backend flag is
now true; with the explicit compatible 2.29.7 runtime the distributed probe
also reports `gin_runtime_initializable=1, gin_usable=1`. The default 2.28.3
runtime remains fail-closed because it lacks the required symbols.
`NcclSocketFallback` has a distinct production constructor and refuses to run
unless the communicator was explicitly configured for NCCL Socket
(`NCCL_IB_DISABLE=1`). Simulation, local P2P, Socket, and GIN audit labels are
emitted by their concrete backend
objects rather than selected by a caller enum.

## Repository layout

```text
include/megamoe/api/          rank-local Notify API and errors
include/megamoe/config/       runtime shape/topology/capacity
include/megamoe/metadata/     rank-local handle, descriptors, serialization
include/megamoe/transport/    async integer collective contract and host fake
include/megamoe/cuda/         opaque CUDA Notify/NCCL APIs and device schemas
include/megamoe/scoreboard/   identity-bound workspace plan
include/megamoe/kernels/      checked future launch contracts
src/cuda/device_notify.cu     Notify implementation boundary
src/cuda/forward/             private Forward contracts and fallback kernels
src/cuda/backward/            private Backward launch contract
src/cuda/testing/             non-installed CUDA harness declarations
src/cuda/*.cu                 aggregation TUs preserving device inlining
src/transport/                host fake plus optional real NCCL AllGather
src/metadata/*corruption*     internal sanitizer test hook, not public API
reference/                    independent Python oracle
tests/                        C++, Python, and full-field cross-language tests
benchmarks/                   host metadata baseline
docs/                         architecture, compliance, performance, limits
```

The installed API is version 2. `make install DESTDIR=... PREFIX=...` installs
`libmegamoe.so.2`, 12 production headers, pkg-config metadata, the minimal
Forward+Backward example, and license notices. Test/oracle/synthetic/fault
declarations remain under `src/`, are compiled only into the non-installed
`libmegamoe_testing.a`, and are unavailable to installed clients. Production
objects use hidden-by-default visibility; `abi/public_symbols_v2.txt` and the
`MEGAMOE_2` linker version script define the exact ABI. See
`docs/api_v2_migration.md` and `examples/minimal_training.cpp`.

MegaMOE-owned source is Apache-2.0 licensed. Build dependencies and inspected
reference repositories are listed in `THIRD_PARTY_NOTICES.md`; CUDA/NCCL and
third-party repository binaries are not redistributed.

## Build and test

```bash
make -j
make test
make bench
make install-test
make BUILD_DIR=build-release ENABLE_CUDA=1 \
  NCCL_RUNTIME_LIB_DIR=NCCL_RUNTIME_DIR release-check

# Optional CUDA/NCCL slice; see docs/build.md for the tested runtime library.
make BUILD_DIR=build-cuda cuda-lib
make BUILD_DIR=build-cuda NCCL_RUNTIME_LIB_DIR=/path/to/nccl/lib cuda-test-local

# Trusted release fast path (default) and full validation build.
make BUILD_DIR=build-release ENABLE_CUDA=1 MEGAMOE_DEBUG_VALIDATE=0 cuda-lib
make BUILD_DIR=build-debug ENABLE_CUDA=1 MEGAMOE_DEBUG_VALIDATE=1 cuda-lib

# Megatron-Core dispatcher baseline, 10 warmups + 100 samples.
torchrun --standalone --nproc-per-node=8 \
  benchmarks/megatron_dispatch_baseline.py --gate-mode fixed-random

# Fair integrated comparison: three independent 8-GPU runs on each side.
# Each measured MegaMOE sample creates a new Notify generation and runs through
# output-ready LocalCudaP2p completion; component means are not added together.
make BUILD_DIR=build \
  NCCL_RUNTIME_LIB_DIR=/path/to/common/nccl/lib fair-integrated-bench

# Historical two-node Socket lower bound: run on both nodes with NODE_RANK=0/1.
NCCL_SOCKET_IFNAME=SOCKET_IFNAME NCCL_IB_DISABLE=1 torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  benchmarks/megatron_dispatch_baseline.py

# Formal two-node RDMA baseline; the launcher rejects Socket fallback.
python3 benchmarks/run_multinode_megatron_rdma.py \
  --remote REMOTE_HOST --port REMOTE_PORT --gpus-per-node 8 \
  --nccl-lib-dir "$NCCL_RUNTIME_DIR:$CUDA_COMPAT_DIR" \
  --socket-ifname SOCKET_IFNAME --ib-hca "$NCCL_HCAS" \
  --ib-gid-index ROCE_V2_GID_INDEX --expert-body te-grouped-mlp \
  --benchmark-backward --training-only --collective-breakdown
```

The checked world-16 RDMA result does not meet the 15% target: MegaMOE GIN is
34.96% slower in Forward, 19.69% slower in Backward and 23.72% slower in the
Notify-inclusive F+B cross-session means. See
`docs/performance_summary.md`; Socket numbers are not used for this claim.

`make test` runs C++ contract tests, Python reference tests, and a machine-readable
full-field C++/Python comparison. See [docs/build.md](docs/build.md) for
`-Werror`, sanitizers, and clean-export validation; see
[design compliance](docs/design_compliance.md),
[performance](docs/performance_summary.md), and
[known limitations](docs/known_limitations.md) for the final audit.
