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

## Three-stage production API

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
- one inseparable `HostIntegerRow` typed immutable allocation.

There is no public integer address/owner pair and no public native communicator
or stream integer. Backend subclasses can retrieve their own opaque state only
through protected `CountExchange` accessors. A host backend returns a checked,
typed `HostCountMatrix` at completion. A future device backend may return only
an opaque `DeviceBufferLease`; the host materializer rejects it until that
backend produces a controlled typed host result. Generic metadata never obtains
or dereferences a device address.

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

Two explicit rounds are used:

- `ExpertCounts`: row width `num_experts`;
- `NotifyPacketSummary`: row width `num_nodes`, counting this source rank's
  deduplicated token-node payloads.

`HostFakeCountExchange` implements this contract for tests only. It is not a
simulation of NCCL, GIN, LSA, or RDMA and accepts host buffers only.

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
