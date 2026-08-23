# MegaMOE

MegaMOE targets deterministic training MoE execution on H200/SM90 and later
devices. The current vertical slice implements rank-local deterministic Notify
Dispatch in production C++/CUDA, a fixed-width distributed control agreement,
a real NCCL AllGather count backend, opaque device metadata leases, and an
independent Python oracle. It does **not**
implement NCCL GIN/LSA, RDMA, activation movement, or persistent
forward/backward kernels.

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

The optional CUDA path keeps `selected_experts[num_tokens, top_k]` on the
calling GPU. It derives an explicit total-order key
`(expert_id, flat_route_index)`, performs radix sort plus segmented inclusive
scan, and scatters the stable local offsets. Atomic return values are never used
as route offsets. A CUDA/NCCL session then AllGathers the device expert-count and
token-node summary rows, stages a checked typed host result at completion, and
uses the same materializer as the host production path. The final stage writes
canonical rows, count geometry, reverse routes, packet fan-out, and signal
records into one preplanned device workspace.

```cpp
auto execution = cuda::make_cuda_execution("notify-stream", device);
auto session = cuda::make_nccl_session("ep-comm", world, rank, generation,
                                       device, bootstrap_id);
cuda::CudaNotifyGenerationCoordinator coordinator(session, execution);
auto generation_result = coordinator.run(
    {config, local_routes, max_tokens_per_rank});
const auto& device_handle = generation_result.dispatch();
```

This coordinator is the only public production protocol entry; device upload,
workspace allocation, preflight wait, control start, and data-collective start
are internal. Every generation executes exactly `dataized local preflight →
256-byte ControlHeader AllGather → header agreement → self-checking canonical
config wire AllGather → exact byte/field agreement → expert-count AllGather →
packet-summary AllGather`. The header is explicitly little-endian encoded as
32 words; no native struct or padding is copied to the wire. It is fully
initialized even for invalid local input.
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
collective generation. It sizes worst-case incoming rows; actual materialized
rows exceeding it fail before an H2D copy.

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
```

`make test` runs C++ contract tests, Python reference tests, and a machine-readable
full-field C++/Python comparison. See [docs/build.md](docs/build.md) for
`-Werror`, sanitizers, and clean-export validation.
