# MegaMOE

MegaMOE is an H200/SM90-oriented deterministic training MoE runtime. The first
vertical slice implements deterministic Notify Dispatch metadata in production
C++ and an independent Python reference. It does **not** claim RDMA transport,
forward/backward persistent kernels, or full data-plane dispatch yet.

## Phase 1 result

`build_notify_dispatch` accepts runtime token counts, experts, top-k, EP world
size, node placement, local-rank placement, expert ownership, M-tile size, and
optional capacities. It returns an immutable, cheap-copy `DispatchHandle` with:

- stable per-source/per-expert `local_offset` generated in
  `(local_token_id, topk_slot)` order;
- source-indexed count matrix and source-rank-prefix canonical destination rows;
- destination rank/node/local-rank metadata and per-token destination node mask;
- local expert row counts, exclusive prefixes, and per-M-tile valid row counts;
- destination-row-indexed reverse routes;
- ascending-expert `combine_order`;
- the same metadata object reusable by forward and backward launch contracts.

Duplicate experts within a token, negative/out-of-range experts, malformed
runtime topology, arithmetic overflow, count exchange corruption, expert
capacity overflow, and per-rank workspace overflow fail explicitly.

Canonical mapping is:

```text
local_offset(t,k) = stable occurrence index for expert e in source rank r
rank_base(r,e)    = sum(count[q,e] for q < r)
dst_row(t,k)      = expert_row_prefix[e] + rank_base(r,e) + local_offset(t,k)
```

The transport may deliver count messages in any order, but it must place them
in rows keyed by source rank. Arrival-order atomics are never used to choose a
destination row.

## Layout

```text
include/megamoe/api/          public Notify Dispatch API and errors
include/megamoe/config/       runtime shape, topology, ownership, capacity
include/megamoe/metadata/     immutable forward/backward dispatch handle
include/megamoe/transport/    count-exchange abstraction
include/megamoe/scoreboard/   shape-derived workspace planning boundary
include/megamoe/kernels/      persistent forward/backward launch contracts
src/metadata/                 production deterministic mapping
src/transport/                development in-memory count transport
src/scoreboard/               workspace plan implementation
src/kernels/{forward,backward}/ reserved device backend modules
reference/                    independent clarity-first Python oracle
tests/                        C++ production and Python reference tests
benchmarks/                   host control-plane benchmark
docs/                         architecture, build, references, roadmap
```

## Build and test

Requirements are a C++17 compiler, GNU Make, and Python 3. No CUDA or third-party
Python package is required for phase 1.

```bash
make -j
make test
make bench
```

For undefined-behavior and memory checks:

```bash
make clean
make test CXXFLAGS='-O1 -g -std=c++17 -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer'
```

See [docs/build.md](docs/build.md), [docs/architecture.md](docs/architecture.md),
[docs/references.md](docs/references.md), and [docs/roadmap.md](docs/roadmap.md).
