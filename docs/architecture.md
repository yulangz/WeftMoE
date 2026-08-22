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

`build_rank_local_histogram` accepts one calling rank and that rank's routes.
It owns a config snapshot, local outgoing route arrays, one expert-count row,
and one destination-node packet-count row. It never reads another rank's route.

The route scan is strictly `(local_token_id, topk_slot)`. Parallel device
implementations may parallelize histogram work, but their final offset must come
from deterministic partition prefixes, not an atomic return order.

### 2. Collective integer exchange

`CountExchangeRequest` explicitly supplies:

- calling rank, world size, communicator identity, and generation;
- collective kind and row width;
- a type-erased immutable integer buffer with address, element count,
  host/device location, and shared allocation ownership;
- opaque execution stream and backend context handles.

`start` returns an operation in `Pending`, `Complete`, or `Failed` state. The
backend retains the input owner through completion. `wait` returns a similarly
owned type-erased output buffer plus matrix shape. Production wrappers validate
returned rank, world, communicator, generation, execution context, kind,
location, address, element count, and shape before host materialization.
Every backend exception is normalized to `CountExchangeFailure` with rank,
world, communicator, generation, and backend error text.

Two explicit rounds are used:

- `ExpertCounts`: row width `num_experts`;
- `NotifyPacketSummary`: row width `num_nodes`, counting this source rank's
  deduplicated token-node payloads.

`HostFakeCountExchange` implements this contract for tests only. It is not a
simulation of NCCL, GIN, LSA, or RDMA and accepts host buffers only.

### 3. Pure rank-local materialization

`materialize_rank_metadata` performs no communication. From the local histogram
and exchanged integer matrices it constructs:

- local outgoing canonical rows and reverse-route identity;
- local owned-expert row counts, prefixes, M-tile valid rows, and incoming row
  geometry `(source_rank, expert, source_expert_offset, dst_row)`;
- node-deduplicated outgoing packet descriptors;
- outgoing and incoming-proxy signal cardinalities.

Remote token IDs/top-k slots are not copied into the destination handle.
Outgoing fan-out descriptors own that reverse identity and remain immutable for
the handle lifetime. A later real data/control transport must deliver the
descriptor with its payload or retain equivalent dispatch-context state.

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
