# Phase-1 architecture and contracts

## Public/runtime boundary

`RuntimeConfig` owns only runtime values: expert count, top-k, M tile, rank/node
topology, arbitrary expert ownership, and optional capacities. Token counts are
runtime per-rank inputs. None of these values appears as a C++ template or
compile-time maximum in the Notify Dispatch implementation.

The public entry point is:

```cpp
DispatchHandle build_notify_dispatch(
    const RuntimeConfig&,
    const std::vector<RankRoutingInput>&,
    CountExchange&);
```

`DispatchHandle` owns `shared_ptr<const DispatchMetadata>`, so forward and
backward can cheaply retain exactly the same routing and row mapping without
re-running Notify or risking mutation.

## Determinism boundary

There are three ordered coordinate systems:

1. A source rank scans routes by `(local_token_id, topk_slot)` and assigns
   `local_offset` within each global expert.
2. Count exchange stores integer count payloads by canonical source rank.
   `rank_base` scans source ranks numerically, never by arrival order.
3. Every destination owns global experts in ascending expert-ID order.
   Its expert prefix plus `rank_base + local_offset` defines `dst_row`.

This creates a bijection between valid routes and destination-local rows. The
reverse-route array is indexed directly by `dst_row`; construction fails if a
collision or hole is observed. Combine uses ascending global expert ID and is
therefore independent of ownership and message delivery.

## Transport boundary

`CountExchange` exchanges only integer matrices and promises a source-indexed
result. `InMemoryCountExchange` is a test transport that deliberately accepts a
message arrival permutation. It is not RDMA. A later NCCL GIN implementation can
replace it without changing metadata semantics.

The eventual hierarchical transport should perform same-rail inter-node count
exchange plus node-local fan-out, while preserving source rank as the logical
matrix row. Staging ring arrival order remains outside canonical row allocation.

## Workspace and persistent kernels

`make_workspace_plan` consumes a handle and returns runtime row/tile cardinality
per destination. Device-specific byte layouts, epochs, ready/claim bitmaps,
release/acquire operations, and ingress credits remain future device-backend
work.

`ForwardPersistentInputs` and `BackwardPersistentInputs` establish that both
kernels consume the same handle and workspace plan. There is intentionally no
launch function in phase 1. This keeps incomplete kernels impossible to mistake
for implemented functionality.
