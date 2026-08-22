# MegaMOE

MegaMOE targets deterministic training MoE execution on H200/SM90. The current
vertical slice implements rank-local deterministic Notify Dispatch metadata in
production C++ plus an independent Python oracle. It does **not** implement
NCCL GIN/LSA, RDMA, activation movement, or persistent forward/backward kernels.

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

The production path never asks one rank to provide every rank's routes:

```cpp
auto local = build_rank_local_histogram(config, calling_rank, local_routes);

auto count_op = start_count_exchange(local, backend, collective_context);
auto summary_op = start_notify_summary_exchange(local, backend, collective_context);
// In a real collective all ranks start each corresponding round.
auto counts = complete_exchange(*count_op, collective_context,
                                ExchangeKind::ExpertCounts, config.num_experts);
auto summary = complete_exchange(*summary_op, collective_context,
                                 ExchangeKind::NotifyPacketSummary,
                                 config.topology.num_nodes);

auto handle = materialize_rank_metadata(
    local, CompletedCollectives{std::move(counts), std::move(summary)});
```

The three stages are:

1. Local stable histogram/scan over only `(local_token_id, topk_slot)`.
2. Asynchronous integer all-gathers for expert counts and packet summaries.
3. Pure rank-local metadata materialization with no backend calls.

`build_notify_dispatch_for_testing` is a host reference/test wrapper that
composes this path for all ranks. It is not a device entry point.

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
include/megamoe/scoreboard/   identity-bound workspace plan
include/megamoe/kernels/      checked future launch contracts
src/                          C++ implementation; no device kernel yet
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
```

`make test` runs C++ contract tests, Python reference tests, and a machine-readable
full-field C++/Python comparison. See [docs/build.md](docs/build.md) for
`-Werror`, sanitizers, and clean-export validation.
