# MegaMOE

MegaMOE targets deterministic training MoE execution on H200/SM90 and later
devices. The current vertical slice implements rank-local deterministic Notify
Dispatch in production C++/CUDA, a fixed-width distributed control agreement,
a real NCCL AllGather count backend, opaque device metadata leases, and an
independent Python oracle. Phase 4 adds a transport-only BF16 data-plane
baseline: typed router activation leases, a debug sequence/credit ring,
canonical direct writes, batched synthetic contribution return, and fixed
expert-order combine. `LocalCudaP2pActivationTransport` now uses real per-rank
CUDA IPC mappings and independent rank processes; the explicit
`DeviceActivationTransport` remains a simulation-only oracle. It does **not**
implement FC1/FC2, backward, Socket transport, or NCCL GIN/RDMA.

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
cross-generation consumers fail closed. Tests and
debuggers may explicitly call `download_device_metadata_for_validation()`;
the coordinator never calls it.

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
agreement in each session/backend slot. Later generations still execute fixed
control/status, expert-count, and summary collectives, while reusing the exact
verified config bytes rather than repeating the variable config AllGather.
Config mismatches and route errors select the lowest origin rank and first
field in schema order; all ranks fail after the same fixed collective and never
enter a variable-width collective. A healthy NCCL communicator can be reused
through a new immutable generation view after a rejected generation.

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

The GIN probe reports headers, the full required symbol set, runtime minimal
initialization, backend implementation, and usability independently. The
2.29.7 headers contain `ncclGin` APIs, but the compatible 2.28.3 runtime lacks
the first required symbol (`ncclCommQueryProperties` in the observed load), and
the backend is deliberately not implemented; consequently `gin_usable=0`.
`NcclSocketFallback` and `Gin` have no production constructor and cannot be
selected by passing an enum. Simulation, local P2P, Socket, and GIN audit labels
therefore cannot be forged by the caller.

## Repository layout

```text
include/megamoe/api/          rank-local Notify API and errors
include/megamoe/config/       runtime shape/topology/capacity
include/megamoe/metadata/     rank-local handle, descriptors, serialization
include/megamoe/transport/    async integer collective contract and host fake
include/megamoe/cuda/         opaque CUDA Notify/NCCL APIs and device schemas
include/megamoe/scoreboard/   identity-bound workspace plan
include/megamoe/kernels/      checked future launch contracts
src/cuda/                     deterministic CUDA mapping and private leases
src/transport/                host fake plus optional real NCCL AllGather
src/metadata/*corruption*     internal sanitizer test hook, not public API
reference/                    independent Python oracle
tests/                        C++, Python, and full-field cross-language tests
benchmarks/                   host metadata baseline
docs/                         architecture, build, references, roadmap
```

## Build and test

```bash
make -j
make test
make bench

# Optional CUDA/NCCL slice; see docs/build.md for the tested runtime library.
make BUILD_DIR=build-cuda cuda-lib
make BUILD_DIR=build-cuda NCCL_RUNTIME_LIB_DIR=/path/to/nccl/lib cuda-test-local

# Trusted release fast path (default) and full validation build.
make BUILD_DIR=build-release ENABLE_CUDA=1 MEGAMOE_DEBUG_VALIDATE=0 cuda-lib
make BUILD_DIR=build-debug ENABLE_CUDA=1 MEGAMOE_DEBUG_VALIDATE=1 cuda-lib

# Megatron-Core dispatcher baseline, 10 warmups + 100 samples.
torchrun --standalone --nproc-per-node=8 \
  benchmarks/megatron_dispatch_baseline.py

# Two-node baseline: run on both nodes with NODE_RANK=0/1.
NCCL_SOCKET_IFNAME=eth0 NCCL_IB_DISABLE=1 torchrun \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
  benchmarks/megatron_dispatch_baseline.py
```

`make test` runs C++ contract tests, Python reference tests, and a machine-readable
full-field C++/Python comparison. See [docs/build.md](docs/build.md) for
`-Werror`, sanitizers, and clean-export validation.
