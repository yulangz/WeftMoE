# Rank-local Notify architecture

## Runtime topology

All shape and topology values remain runtime data. The baseline requires
homogeneous nodes with `G = world_size / num_nodes`. Rank coordinates form the
complete Cartesian set:

```text
{(node, local_rank) | node in [0,N), local_rank in [0,G)}
```

This makes the proxy function unique:

```text
proxy(destination_node, source_rank) =
    rank_at(destination_node, local_rank(source_rank))
```

Node IDs are dense. Heterogeneous GPU counts and missing rails are unsupported
and rejected before route validation, including zero-token calls.

## Distributed control protocol and rank-local API

Every rank unconditionally follows this generation sequence:

```text
host/device local preflight
  -> AllGather ControlHeader[256 bytes]
  -> deterministic header agreement
  -> AllGather canonical config wire[max declared byte length]
  -> exact decoded-field and byte agreement
  -> AllGather expert counts [world, num_experts]
  -> AllGather packet summaries [world, num_nodes]
  -> pure materialization
```

The fixed header contains protocol/schema versions, a two-word diagnostic
session summary, generation/rank/world, diagnostic config digest and exact
per-rank blob length,
explicit experts/top-k/M-tile/node/capacity fields, collective
dtype/index widths and row widths, local token/route requirements, and a local
error with original token/slot/expert coordinates. Its little-endian checksum
is validated before use. `num_tokens` may differ by rank; every rank must
agree on `max_tokens_per_rank` and stay within it.

Every row first proves that its declared rank equals its AllGather row. Rank 0
then supplies the common generation/session/world/schema baseline. Remaining
rows are examined by ascending origin rank and fields by fixed schema order.
The validator takes only gathered bytes, never local context, so all callers
produce byte-identical errors. The canonical config payload explicitly encodes,
without native-struct padding, the original length-prefixed stable session
identity, `rank_to_node`, `rank_to_local_rank`, `expert_to_rank`,
`expert_rows`, `workspace_rows_per_rank`, topology shape, experts/top-k/M-tile,
capacity, dtype/index widths, and descriptor schema. A 64-byte little-endian
envelope adds magic, wire version, header/payload lengths, two zero reserved
words, payload checksum, and normalized wire digest. Decoding rejects bad
constants, reserved bits, overflow, truncation, trailing bytes, checksum/digest
mismatch, and `ControlHeader.config_digest` mismatch before payload use,
including world=1. Different row lengths use the same maximum AllGather slot
width; each exact declared prefix is reconstructed before validation. Digest
equality never skips exact field and byte comparison.
Control failure resets the framework protocol gate to control-ready; count and
summary starts remain zero in the audit. A new session/generation view may share
the still-healthy native communicator and starts again at control.

### 1. Rank-local histogram

`build_rank_local_histogram` accepts an owned rank context and that rank's
routes. It returns an opaque immutable `LocalHistogramHandle` owning a config
snapshot, local outgoing route arrays, one typed expert-count row, one typed
destination-node packet-count row, and unique local provenance. It never reads
another rank's route. Its PIMPL cannot be constructed or mutated through the
public API.

The route scan is strictly `(local_token_id, topk_slot)`. Parallel device
implementations may parallelize histogram work, but their final offset must come
from deterministic partition prefixes, not an atomic return order.

### 2. Collective integer exchange

`CountExchangeRequest` explicitly supplies:

- an immutable `CollectiveSession` containing communicator identity, world
  size, generation, and a strongly owned backend-private state token;
- calling rank and an immutable `ExecutionToken` strongly owning any
  backend-private stream/context state;
- collective kind and row width;
- one inseparable `HostIntegerRow` typed immutable allocation and, when a
  device backend is selected, an optional backend-opaque `DeviceBufferLease`.

There is no public integer address/owner pair and no public native communicator
or stream integer. Backend subclasses can retrieve their own opaque state only
through protected `CountExchange` accessors. The NCCL backend consumes its own
opaque device lease and, after its CUDA event reaches completion, returns a
checked typed `HostCountMatrix`. Generic metadata never obtains or dereferences
a device address.

A device lease records `rows`, `row_width`, checked `element_count`, and the
backend-declared `capacity_elements`. Construction requires nonzero dimensions
and capacity and validates:

```text
element_count = checked_uint64(rows * row_width)
element_count <= SIZE_MAX
element_count <= capacity_elements
```

Overflow and short capacity fail before a lease is published.

The backend `start` returns a backend operation, immediately wrapped in a
move-only framework `CollectiveOperation`. The wrapper strongly retains the
local handle, input row, session, execution token, and backend operation until
terminal completion/cancellation. `wait()` validates every returned field and
is single-use. Explicit `cancel()` reports cleanup failures. Destruction of an
unwaited operation performs best-effort cancellation, catches every exception,
and is `noexcept`. Backend `NotifyDispatchError`, standard exceptions, unknown
exceptions, null operations, host-result construction errors, wait errors, and
cancel errors are all mapped to contextual `CountExchangeFailure`.

Pending operations own a shared internal `OperationResourceLease`. Successful
`wait()` creates an immutable `CompletionResourceLease` containing the exact
backend completion and a shared reference to that same operation resource.
Every `ValidatedCollectiveHandle` copy shares the completion lease. Therefore
the local histogram, typed input, session, execution token, backend operation,
and backend completion lease remain alive until the final validated copy is
destroyed. After the completed operation object is also released, destruction
of the final validated copy releases the chain exactly once through ordinary
shared ownership; it does not call wait or cancel again. Keeping the completed
operation object longer may conservatively extend the same resource lifetime.

After fixed control and exact config rounds, two data rounds are used:

- `ExpertCounts`: row width `num_experts`;
- `NotifyPacketSummary`: row width `num_nodes`, counting this source rank's
  deduplicated token-node payloads.

`HostFakeCountExchange` implements this contract for tests only. It is not a
simulation of NCCL, GIN, LSA, or RDMA and accepts host buffers only.

The coordinator-private NCCL backend is a correctness baseline using real
`ncclComm_t`, `cudaStream_t`, `ncclAllGather`, pinned D2H staging, a completion
event, and `ncclCommGetAsyncError`; all native objects stay in CUDA translation
units. Its fixed control slot is always allocated. The capacity constructor
also preallocates count and summary slots before the measured hot path; the
control-only constructor allocates agreed data slots after exact config.
Its internal starts enqueue H2D, byte AllGather, D2H, and an event; waits check
the event, NCCL async status, wire schema, configuration, and local failures.
If any pre-return enqueue step fails, it drains partially submitted stream work
before releasing the typed input owner. `wait` synchronizes the event, checks
the NCCL async error, then publishes typed host storage. NCCL has no
per-operation cancellation, so `cancel` safely drains that event without
aborting the shared communicator.
This backend is not the final same-rail/GIN implementation.

The backend state machine is auditable and ordered:
`ControlReady -> ControlPending -> ConfigReady -> ConfigPending -> DataReady -> CountStarted -> CountComplete
-> SummaryStarted -> Complete`. A rejected control returns to `ControlReady`.
Pending control operation destruction/cancel drains its event without throwing
from destruction; a partial enqueue is drained before the slot/input can be
released. One operation object is not safe for concurrent wait/cancel/move.
The successful audit is `control/config/count/summary = 1/1/1/1`; local/header
failure is `1/0/0/0`, and config integrity/exact-agreement failure is
`1/1/0/0`. A counter increments immediately after its `ncclAllGather` returns
success; later D2H, event record/wait, allocation, or completion failure cannot
roll it back.
Test-only hooks in `src/` inject control start/enqueue, event record/wait, D2H,
slot allocation, config corruption, identity-summary collision, and completion
failures. Lifecycle counters verify balanced slots/completion leases, no
validated publication, and no repeated wait/cancel/drain. They are absent from
public headers.
World-one or all-rank injection validates cleanup; a single-rank fatal NCCL or
CUDA failure is a backend failure and is not claimed recoverable by agreement.

### 3. Pure rank-local materialization

Only successful wrapper completion can create an opaque immutable
`ValidatedCollectiveHandle`. The framework, not the backend, copies the local
provenance into it. `materialize_rank_metadata` performs no communication and
requires exactly one expert-count and one packet-summary completion matching
the local handle's rank, world, session instance, communicator identity,
generation, execution-token instance, and provenance.

Before indexing any route or allocating from a derived size, materialization
revalidates route-array lengths, checked shape arithmetic, expert/rank/node/
local-rank/slot ranges, stable local offsets, combine permutations, node masks,
typed row widths, matrix shape, and the calling-rank count/summary rows. This
makes corruption from the dedicated internal test hook a controlled exception,
not an OOB read. From the validated inputs it constructs:

- local outgoing canonical rows and reverse-route identity;
- local owned-expert row counts, prefixes, M-tile valid rows, and incoming row
  geometry `(source_rank, expert, source_expert_offset, dst_row)`;
- node-deduplicated outgoing packet descriptors;
- outgoing and incoming-proxy signal cardinalities.

Remote token IDs/top-k slots are not copied into the destination handle.
Outgoing fan-out descriptors own that reverse identity and remain immutable for
the handle lifetime. A later real data/control transport must deliver the
descriptor with its payload or retain equivalent dispatch-context state.

## Ownership and lifetime

```text
CollectiveSession ─┐
ExecutionToken ────┼─> RankCollectiveContext
                   │          │
local routes ──────┴─> LocalHistogramHandle (typed rows + provenance)
                                  │ strong ownership
                                  v
                         CollectiveOperation
                                  │ owns while pending
                                  v
                     OperationResourceLease
                    (local + input + tokens + backend op)
                                  │ successful wait; shared by
                                  v
                    CompletionResourceLease
                 (backend completion + optional backend lease)
                                  │ shared by every copy
                                  v
                    ValidatedCollectiveHandle
                     (typed matrix + provenance)
                                  │ exact-context checks
                                  v
                       rank-local materialization
                                  │
                                  v
                           DispatchHandle
```

Handles are cheaply copyable immutable owners. Operations are move-only.
Descriptor/reverse metadata is owned by the final `DispatchHandle`; workspace
plans hold identity/provenance bindings rather than borrowed metadata pointers.
Materialization copies the validated integer metadata into `DispatchHandle`;
the dispatch handle deliberately does not retain collective resources. Callers
may release validated handles after materialization when no later consumer needs
their completion lifetime.

The NCCL stable identity source is the communicator's original shared
`ncclUniqueId` bytes plus an explicitly supplied protocol namespace. The
session owns this immutable bounded byte string across generation views. Only a
two-word safe summary enters fixed control; the raw bytes enter exact config
agreement and are never emitted in normal diagnostics.

For cross-host bootstrap, rank 0 creates the ID inside the runner and writes
exactly 128 binary bytes to an inherited, launcher-owned pipe descriptor. The
launcher forwards the bytes to each other local runner through that process's
private stdin and to each remote runner exclusively through SSH stdin. The
secret is never serialized as text and never enters argv, environment, a
remote command, process listings, stdout/stderr, or a temporary file. All
captured runner output and all launcher-generated exceptions/tails pass through
a defense-in-depth scrubber that removes the known raw bytes, lower/upper hex,
standard/URL-safe base64 (with or without padding), and the retired text-marker
line. Scrubbing is not the transport boundary; the dedicated pipes are.

## Benchmark accounting

The benchmark establishes every rank's NCCL session once before 10 warmups and
50 measured generations and reuses that communicator. Session setup is a
separate one-establishment sample across ranks. Per-generation setup includes
coordinator/backend creation and control/config/count/summary slot, device,
pinned, and event allocation. Setup-inclusive E2E is computed for each rank as
`rank_generation_setup + rank_steady_execution` before selecting the slowest
rank; it never subtracts maxima from different ranks.

Unique workspace totals count route input, the notify arena, pinned staging,
and collective slots once. Final device metadata is reported as a notify-arena
subview and is not added again. Event count is reported; event bytes remain
zero unless CUDA exposes a reliable allocation size.

## CUDA deterministic Notify and device workspace

The CUDA frontend takes only the calling rank's device route matrix. Runtime
shape, topology, calling rank, device, session generation, execution identity,
and a collective-wide `max_tokens_per_rank` capacity are bound into
`CudaNotifyWorkspacePlan`. Query arithmetic checks both integer and allocation
limits. Allocation creates one device arena, one pinned host staging arena, and
one completion event before the hot path; sort, scan, histograms, route fields,
count matrices, descriptors, and final metadata are aligned subregions.

For flat route index `i = token * top_k + slot`, the sort key is:

```text
key(i) = expert_id(i) * route_count + i
```

The explicit `i` makes it a total order even if the radix-sort implementation
does not promise stability. A key-group inclusive scan produces the occurrence
number, which is scattered back as `local_offset`. Atomics are limited to
commutative histogram/summary accumulation and deterministic minimum-invalid-
coordinate reporting; no atomic return order chooses a canonical row.

After checked host materialization, one stream-ordered staging step fills:

- route destination row plus all original route arrays and node masks;
- `[source_rank, expert]` counts, owned expert IDs/counts/prefixes, flattened
  M-tile rows, and incoming row geometry;
- reverse-route records, packet headers, fan-out offsets/targets, and outgoing
  and incoming-proxy signal records.

Every published `DeviceBufferLease` strongly owns the common workspace state.
Route input, workspace, output, execution stream, event, and session therefore
remain alive through pending work and through the final `CudaDispatchHandle`.
The controlled `validate_cuda_dispatch_metadata` diagnostic is implemented in
the CUDA module and downloads every device field for tests; the generic
metadata layer still cannot inspect a native pointer.

## Thread-safety contract

One `CollectiveOperation` does **not** support concurrent `wait()`, `cancel()`,
move construction, or move assignment. The current state machine has no internal
lock and must not be described as thread-safe. Different operation objects may
run in parallel when their backend supports it; the host fake supports parallel
waits on distinct operations. Shared immutable local/validated handles may be
read concurrently, but mutation is unavailable through the public API.

## Descriptor baseline

For each local token, routes are grouped by destination node. One packet is
created per non-empty token-node group. Deterministic order is destination node
first, then local token ID. Each fan-out is sorted by global expert ID and stores:

```text
topk_slot, destination_local_rank, expert_id, canonical dst_row
```

Channel is currently `0`. One packet produces one signal, so both
`expected_packets` and `expected_signals` equal the summary count. For a proxy
at `(destination_node, rail_local_rank)`, the expected count from source node
`s` is:

```text
summary[rank_at(s, rail_local_rank)][destination_node]
```

Packet aggregation/channel autotuning may replace this baseline only if it
also supplies a new explicit summary collective and leaves canonical mapping
unchanged.

## Handle/workspace safety

Every handle has a deterministic `(content_hash, collective_generation)` plus a
non-serialized shared provenance token unique to that handle instance.
`make_workspace_plan` binds requirements, runtime shape, epoch, identity, and a
weak reference to that token.

Before any future persistent launch can read metadata or workspace buffers,
`ForwardPersistentInputs::checked` / `BackwardPersistentInputs::checked` verify:

- non-empty handle and plan;
- exact handle-instance provenance;
- content identity and collective generation/epoch;
- calling rank, world size, experts, top-k, and M tile;
- canonical-row, M-tile, outgoing-route, and packet capacities.

Default plans, expired plans, and plans from another handle fail safely. No
persistent launch implementation exists in this phase.
