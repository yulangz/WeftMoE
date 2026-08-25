# Rank-local Notify architecture

## Single-node Forward data plane

After Notify publishes `DeviceDispatchHandle`, the Forward adapter acquires an
owning dispatch consumer and passes its private kernel view directly to
`LocalCudaP2pForwardMegaKernel`; metadata is neither downloaded nor rebuilt.
The main-shape kernel uses one resident CTA per SM and a cooperative grid.
Dispatch is currently an initial bulk phase; after its peer release/acquire
boundary, compute/return/reduce are M-block pipelined rather than globally
phase-serialized:

```text
DispatchPut -> FC1(WGMMA/TMA) -> SwiGLU -> FC2(WGMMA/TMA)
                                      \-> batched peer return
                                                \-> gate-after TopKReduce
```

FC1 evaluates both `[A,B]` projections from each staged X tile, casts
`SiLU(A)*B` to BF16 once, and releases the M-block scoreboard only after every
lane's store is ordered. FC2 acquires that scoreboard, accumulates WGMMA in
FP32, casts the expert output to BF16, and writes canonical local rows. The CTA
that completes an M-block's last FC2 N tile immediately returns that block
while other CTAs continue FC2. Return completion is aggregated once per source
rank instead of signalled per route. Source warps reduce independent tokens,
wait on the destination-rank release signal, multiply by the local FP32 gate,
add routes in expert-ID order, and perform one final BF16 cast. The final local
grid completion is sufficient for reuse: every incoming peer has already
published all rows acquired by Reduce, and every outgoing store has completed
before its release signal.

Completion storage is initialized, not assumed. Pool creation clears each
physical slot once. Every Forward generation then launches a small owner-only
setup kernel that writes `ingress_published` and
`outgoing_sequence/source_ready` to `epoch-1`, system-fences those stores, and
performs a peer-visible ready/ack handshake before any put can publish. The
second acknowledgement prevents a fast rank from reinitializing a reused slot
before a slow peer observed the preceding epoch. Two live generations select
different deterministic physical slots; a slot is reusable only after the
last Forward handle releases it.

The benchmark reads first-start and last-complete device timestamps from the
same status lease. On the main world-8 fixture FC1/FC2 overlap by about
0.366 ms, FC2/Return by about 5.018 ms, and Return/Reduce by about 0.429 ms.
These full stage spans intentionally overlap and are not additive.

Release mode precomputes an immutable M-block-to-expert lookup in the
debug-only claim arena and acquires FC tasks in deterministic chunks of four.
Debug mode keeps per-task CAS claims and full validation. Both execute the same
WGMMA K order and produce identical output bits.

The cache policy affects lifetime, not arithmetic: the cache mode retains BF16
A/B checkpoints in the session pool; no-cache omits those stores. Both retain
canonical dispatch X through the Forward handle. Transient S and expert output
are generation-local. Cross-node Forward and backward are not implemented.

The launch requests a `2x1x1` cluster where supported and records requested and
actual sizes independently. On the current L20X performance path the runtime
audit is `requested=2, actual=1`; TMA multicast is disabled. Documentation and
benchmark JSON therefore make no 2-CTA formation or multicast claim.

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
  -> device prefix/descriptor materialization
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
backend-owned device completion lease. Host/fake backends return a checked
typed `HostCountMatrix`. Generic metadata never obtains or dereferences a
device address.

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
`ncclComm_t`, `cudaStream_t`, `ncclAllGather`, a completion event, and
`ncclCommGetAsyncError`; all native objects stay in CUDA translation units. Its
fixed control/config slots use bounded pinned staging because those two small
agreement rounds are host control-plane data. Expert-count and packet-summary
slots remain device-only and successful waits publish opaque device completion
leases rather than typed host matrices. The capacity constructor
also preallocates count and summary slots before the measured hot path; the
control-only constructor allocates agreed data slots after exact config.
Control/config starts enqueue H2D, byte AllGather, bounded D2H, and an event;
data starts enqueue device-to-device AllGather plus an event. Waits check the
event, NCCL async status, wire schema, configuration, and local failures.
If any pre-return enqueue step fails, it drains partially submitted stream work
before releasing the typed input owner. `wait` synchronizes the event, checks
the NCCL async error, then publishes the appropriate host or device lease. NCCL has no
per-operation cancellation, so `cancel` safely drains that event without
aborting the shared communicator.
This backend is not the final same-rail/GIN implementation.

The backend state machine is auditable and ordered:
`ControlReady -> ControlPending -> ConfigReady -> ConfigPending -> DataReady -> CountStarted -> CountComplete
-> SummaryStarted -> Complete`. A rejected control returns to `ControlReady`.
Pending control operation destruction/cancel drains its event without throwing
from destruction; a partial enqueue is drained before the slot/input can be
released. One operation object is not safe for concurrent wait/cancel/move.
The first successful agreement audit is
`control/config/cache/count/summary = 1/1/0/1/1`. In a trusted-participant
release session, later generations use the immutable verified wire and report
`1/0/1/1/1`; debug mode always repeats exact config agreement. Local/header
failure remains `1/0/0/0`, and config integrity/exact-agreement failure is
`1/1/0/0` in the original four collective fields. A counter increments
immediately after its `ncclAllGather` returns
success; later D2H, event record/wait, allocation, or completion failure cannot
roll it back.
Test-only hooks in `src/` inject control start/enqueue, event record/wait, D2H,
slot allocation, config corruption, identity-summary collision, and completion
failures. Lifecycle counters verify balanced slots/completion leases, no
validated publication, and no repeated wait/cancel/drain. They are absent from
public headers.
World-one or all-rank injection validates cleanup; a single-rank fatal NCCL or
CUDA failure is a backend failure and is not claimed recoverable by agreement.

### 3. Host oracle materialization

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

This function remains the independent C++ host oracle and host-only production
fallback. The CUDA generation coordinator does not call it.

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
a defense-in-depth scrubber that removes every known nonempty raw prefix,
case-insensitive hex derived with an escaped literal pattern, standard/URL-safe
base64 (with or without padding), and the retired text-marker line. Scrubbing is
not the transport boundary; the dedicated pipes are.

Bootstrap transport is an exact-frame protocol. Python loops on binary
`os.write` until all 128 bytes are accepted, retries interruption, rejects zero
progress or short capacity, and always closes stdin. C++ symmetrically retries
partial/EINTR reads and writes; after byte 128 the receiver requires EOF and
rejects byte 129. The sender closes immediately after the frame, while the
launcher's generation timeout bounds the receiver's EOF wait. C++ production
transfers the descriptor into a move-only RAII channel with explicit
`Open/Consumed/Failed/Closed` state. Success, short/oversize input, non-EINTR
I/O failure, cancellation, and destruction all relinquish the descriptor
exactly once. Oversize detection closes immediately at byte 129 without
draining; every terminal state rejects reuse, so residual bytes cannot become a
second bootstrap frame and a recycled descriptor number cannot be touched by
the old channel.

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
limits. Allocation creates one device arena, a bounded pinned status arena, and
one completion event before the hot path; sort, scan, histograms, route fields,
descriptors, and final metadata are aligned subregions. NCCL count matrices
remain in their own backend-owned device slots.

For flat route index `i = token * top_k + slot`, the sort key is:

```text
key(i) = expert_id(i) * route_count + i
```

The explicit `i` makes it a total order even if the radix-sort implementation
does not promise stability. A key-group inclusive scan produces the occurrence
number, which is scattered back as `local_offset`. Atomics are limited to
commutative histogram/summary accumulation and deterministic minimum-invalid-
coordinate reporting; no atomic return order chooses a canonical row.

After NCCL data completion, a checked stream-ordered materialization pipeline
directly consumes `[source_rank, expert]` and
`[source_rank, destination_node]` matrices. In release, expert/source totals,
owner prefixes, destination capacities, and node packet compaction are separate
parallel kernels; packet compaction uses a deterministic node-owned block scan.
Debug retains the serial checked planner as an oracle. Bulk records are emitted
by regular parallel per-route, per-expert/source, per-expert, per-packet, and
per-node kernels. The pipeline
fills:

- route destination row plus all original route arrays and node masks;
- `[source_rank, expert]` counts, owned expert IDs/counts/prefixes, flattened
  M-tile rows, and incoming row geometry;
- reverse-route records, packet headers, fan-out offsets/targets, and outgoing
  and incoming-proxy signal records.

The kernel first validates collective row bounds, the calling-rank rows,
checked 64-bit sums, expert/workspace capacities, and every output capacity.
Only after those checks does it write variable-length records. Source ranks,
destination ranks, experts, nodes, tokens, and top-k slots are traversed in
fixed ascending order. Packet order is `(destination_node, local_token_id)` and
fan-out consumes the device `combine_order`, which is expert-ID order.

No parallel phase uses an atomic return value to choose an output index. Every
writer consumes checked prefixes, and every route-sort key contains the flat
route index as an explicit tie-breaker. Block scheduling and stream enqueue
timing therefore cannot change any byte of the published layout.

The POD record schema is layout version 1. It places 64-bit fields before
32-bit fields and fixes every record size with compile-time assertions. The
arena stores aligned offset/count/capacity subviews; count/summary matrices are
separate backend-owned completion leases. Every published `DeviceBufferLease`
strongly owns the common workspace state.
Route input, workspace, output, execution stream, event, and session therefore
remain alive through pending work and through the final
`DeviceDispatchHandle`. Copies share those leases. Public code never receives
a detachable pointer view: `acquire_consumer` returns a move-only
`DeviceDispatchConsumerLease`, and only a trusted launch adapter may create the
internal `DeviceDispatchKernelView` POD. Submitted consumers record a pooled
event on their own stream; the slot cannot return until the last handle and
consumer have drained. The explicit
`download_device_metadata_for_validation` diagnostic downloads
every field for tests and reconstructs a host `DispatchHandle`; the generic
metadata layer still cannot inspect a native pointer and the production
coordinator never invokes this API.

## Session resource pool and slot lifecycle

`DeviceNotifyResourcePool` is sized from runtime maximum tokens, experts,
top-k, world, nodes, canonical-config bytes, and an explicit
`max_inflight_generations` and `max_inflight_consumers`. Session setup allocates, per slot:

- one host-convenience route-input device region;
- the maximum-shape Notify transient/final arena and CUB workspace;
- pinned status/config staging;
- fixed control plus bounded config/count/summary send/receive slots;
- mapping/planning/Notify-completion and route-ready events, the configured
  consumer-completion events, and five collective CUDA events (the fifth is
  the preallocated fused-Notify completion event).

Generation-slot state is `Available`, `InUse(epoch)`, or `Quarantined`.
Consumer-event state is tracked independently with the same three values.
Acquisition is serialized,
assigns a monotonic epoch, resets backend generation state, and never allocates.
The workspace and successful result share the same lease. A handle copy keeps
the slot alive; a moved-from/default handle cannot. When the last lease dies,
its noexcept cleanup records and waits on the precreated completion event
before making the slot available. An event record/synchronize failure
permanently quarantines that event. If the consumer stream can still be
drained, the generation slot remains reclaimable; if both event and stream
drain fail, the generation slot is also permanently quarantined. Every
download, launch adapter, later consumer acquire, and pool acquire checks this
state and fails closed. Cleanup failure never guesses that device work ended.
At pool teardown a quarantined slot performs a best-effort device-wide drain;
if even that fails, native allocations are deliberately abandoned to the
process rather than freed underneath possibly live device work.
Errors before publication release by the same path, and pool destruction is
deferred by outstanding leases.

The production route contract is
`SelectedExpertRouteLease<std::int32_t>`: immutable int32 dtype and layout
version, contiguous `[tokens,top_k]` shape, checked element capacity, device,
router/pipeline session and route generation, producer execution, allocation
owner, and a recorded producer-ready event. It is not constructible from a
generic `DeviceBufferLease`. Same-stream consumption uses stream order;
cross-stream consumption requires the explicit import API, which enqueues
`cudaStreamWaitEvent` without replacing producer provenance. The host-input
generation comparison is exact and unconditional. The coordinator host-input
overload and private test storage adapter are convenience/test paths,
not router-to-Notify training interfaces.

Consumer state is `Acquired -> Submitted -> Completed/Released`. An
unsubmitted lease releases its event token immediately. A submitted lease
waits its precreated completion event in `finish()` or its noexcept destructor.
If the public handle dies first, the consumer still owns the arena and epoch.
Multiple streams may acquire distinct tokens; exhaustion is
`ResourceExhausted`, never an event allocation fallback. A single consumer
object does not support concurrent move/submit/finish.

The coordinator also accepts a shared cancellation token and a relative
deadline. After fixed control/config agreement it starts the count operation;
an all-rank-consistent cancellation or expired deadline invokes that real
operation's `cancel()` exactly once, publishes no handle, and releases or
quarantines the pool slot through the normal lease path. Recovery is promised
for world=1 or consistent all-rank cancellation, not for an isolated process or
NCCL hard failure.

Production construction requires a `DeviceNotifyResourcePool`. The old
two-argument allocating constructor is private; an explicitly named
`testing::make_unpooled_coordinator_for_testing` remains solely for allocation
fault injection. A compile-negative probe prevents its accidental restoration
as a production constructor.

For pooled generations the forbidden steady APIs are `cudaMalloc/cudaFree`,
`cudaHostAlloc/cudaFreeHost`, `cudaEventCreate*/cudaEventDestroy`, and
`cudaDeviceSynchronize`. Default GPU tests snapshot backend counters across ten
generations, exercise two live handles, exhaustion, failure recovery, slot
reuse, and pool-owner destruction. Bounded control/config/status copies remain
visible in the audit; validation download is explicit and excluded from the
production audit interval.

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

Default plans, expired plans, and plans from another handle fail safely.

## Phase-4 activation ownership and worker

The router-to-Notify boundary is a pair of immutable typed leases:
`SelectedExpertRouteLease<int32_t>` and `SelectedExpertGateLease<float>`.
Both describe the same contiguous `[tokens,top_k]` order and are inseparably
bound to the same allocation provenance, session, exact generation, device,
producer execution, and ready event. Notify rejects a gate from another route
allocation even when shape and capacity happen to match. The generation
coordinator is the only production consumer and requires both leases; the
test upload helper allocates only during setup and reissues exact-generation
views for measured iterations.

The public router handoff is `SelectedActivationLease<BFloat16>`, not a generic
device buffer. Its immutable provenance is dtype/layout, `[tokens,hidden]`
shape and capacity, device, session identity, exact generation, producer
execution, allocation owner, and precreated ready event. Only the private
launch bridge sees its address. The host upload function is explicitly in the
`testing` namespace and acquires a preallocated input slot.

`ActivationTransportResourcePool` is session-level and capacity-planned from
maximum tokens, hidden width, top-k, world, nodes, destination rows, channels,
ring slots, and in-flight generations. Each slot contains input, bounded
ingress ring/header/sequence, canonical owned rows, route contribution/return,
combined output, device status, pinned status, and all events. Slot state is
`Available -> InUse(epoch) -> Available|Quarantined`; the shared slot lease is
retained by input and output handles, so destroying either public wrapper
cannot permit premature reuse. There is no steady CUDA allocation or event
creation fallback.

The simulation worker and local P2P worker are separate concrete types. The
former can only publish `DeviceSimulation`; the latter is constructible only
after all local rank processes exchange and validate versioned CUDA IPC arena
handles using the existing secret NCCL bootstrap/session. It validates rank,
world, device ordinal, session digest, generation, slot count, and arena bytes,
opens only peer-accessible devices, and closes each owned mapping exactly once.
Partial setup fails closed. No caller-supplied backend enum participates in an
audit record.

The release local worker is a fixed stream-ordered pipeline of bulk CUDA grids:
route CTAs write peer canonical rows directly, one tiny kernel publishes a
per-peer dispatch completion, destination-row CTAs apply the selected validation
expert body and return contributions, a second per-peer completion closes
return, and token CTAs reduce in fixed expert order. The sole formal gate
placement is source-side gate-after: each returned BF16 identity result is
multiplied by its route's FP32 gate, route contributions are accumulated in
expert-ID order with explicit FP32 rounding, then cast once to BF16. Gates
remain source-local (`tokens*top_k*sizeof(float)` bytes) and never enter IPC
payloads. `DestinationBeforeUnsupported` fails explicitly; unit gates use the
same pointer and reduce kernel without a constant specialization. Identity is the integrated
benchmark body; the older synthetic transform remains only for transport
validation. The grids expose all routes/rows/tokens to all SMs and avoid
per-route system atomics. Physical peer traffic counts every destination-GPU
fan-out; logical node dedup is reported separately.

Static IPC mappings, arena layout, and capacities are session setup. A release
generation does not clear the multi-gigabyte arena: its monotonically increasing
slot epoch makes prior signals stale, and only the used status record is reset.
After Notify, a 16-byte/rank device AllGather verifies that every process has
acquired the same transport slot and epoch before peer writes can start. This
small agreement is necessary because Notify completion alone does not prove the
transport slot/capacity boundary. The source waits for all return signals before
reducing, then records output completion; the normal release path does not add a
second terminal team barrier. Debug/cancel paths retain explicit draining.

`MEGAMOE_DEBUG_VALIDATE=1` selects the earlier persistent ring worker. There,
physical slot is `sequence % ring_slots`; every link has a monotonic 64-bit next
sequence and bounded credit. Payload stores and headers precede
`__threadfence_system` plus a system-scope signal; relay acquire-observes the
signal, validates the exact generation/source/destination/route/row tuple, and
writes only the Notify-provided
canonical row, and returns credit after consumption. Before launch a fixed
device AllGather confirms that every rank selected the same pool slot and
epoch. No activation is staged through host memory.

Two sequence modes share the same header invariant. Small bounded rings use a
per-slot system-scope release signal and return one credit only after consume;
this is the wrap/slow-relay correctness path. If the planned ring can hold the
entire checked per-link capacity, the producer cannot wrap or overwrite, so it
publishes the same contiguous slot sequences with a monotonically increasing
release watermark (every 64 entries and at tail). Consumers may read only
sequences below that acquire-observed watermark. Completion return is likewise
batched after the source-specific relay CTA has fenced all independent route
slots. Packet identity, canonical row and reduction order are unchanged.

Once exact dispatch and fan-out counters complete, CTAs transition to the
synthetic-return/TopKReduce phase. Each route has its own contribution slot.
Reduction traverses `combine_order`, which is global expert-ID order, performs
FP32 additions, and casts once to BF16. Synthetic contribution is a documented
transport-validation transform; no FC1, SwiGLU, or FC2 is present.

The ordering probe copies 64 MiB with multiple producer blocks. Each block
publishes its range before incrementing a completion counter; only the last
block fences and signals. A destination kernel on an independent stream waits
only on that signal and validates the whole payload before the source stream is
synchronized. Its early-signal negative seam must observe corruption.

GIN capability has five independent booleans: headers, full symbols,
distributed runtime initialization, project backend implementation, and
usable. Usable is their conjunction. The default 2.28.3 runtime fails the
symbol gate. The explicitly loaded NCCL 2.29.7 CUDA 12.9 runtime passes
world-2/world-16 device-communicator and registered-window initialization, but
the data-plane backend is not implemented yet; therefore `gin_usable` remains
false. The session probe is collective and returns no native NCCL/CUDA handle.
The opt-in Socket correctness backend is now concrete and separately typed. It
packs GPU route payloads, exchanges one aggregate per peer with host-enqueued
NCCL Send/Recv, performs destination device relay plus the validation expert
transform, returns aggregate contributions, and reduces with source-resident
dynamic gates. Only the small runtime count matrix and terminal status are
staged to host; activation bytes remain on device. Its current first slice is
direct rank-to-rank and therefore does not yet implement node dedup,
same-rail proxy plus local CUDA-P2P fan-out, or full FC Forward/Backward. It is
never described as persistent, GIN, or RDMA and cannot be confused with the
real one-node CUDA IPC backend.

### Backward ownership and dataflow

The Backward public boundary is owning on both sides. A
`SelectedOutputGradientLease` binds BF16 `[tokens, hidden]` dY to the same
session, generation, device, execution and Forward slot; a
`DeviceBackwardHandle` retains the Forward handle, dY lease, pool slot and
backend team. Consequently A/B checkpoints, canonical X, transient gradients
and returned dX cannot be recycled while any Backward result remains live.

The integrated cache path keeps that ownership chain asynchronous. Forward
records pooled `output_ready` and `complete` events and returns an owning,
pending handle. `transfer_consumer_to_backward()` moves the already acquired
dispatch consumer into Backward; Backward waits on `output_ready` in its CUDA
stream and finishes that same consumer only after its kernel completes. The
slot therefore observes one acquire and one release for the complete training
step. There is no intermediate host stream synchronization, cache
release/reacquire, repeated Notify, or route/metadata rebuild. Public Forward
audit access remains checked and rejects a pending handle until this handoff
reaches its terminal state.

After Backward has completed dX return and fixed-order source reduction, the
production kernel publishes its monotonic generation epoch into a dedicated
peer-visible terminal vector and acquire-waits for every rank. This final
device drain uses no host collective or allocation. Keeping it distinct from
the Forward/Backward setup ready/ack vector prevents an already-published
setup epoch from falsely satisfying terminal completion. It also closes each
formal benchmark generation before the next fused Notify, rather than charging
the previous generation's rank skew to the next AllGather.

For the 7168x3072 specialization one cooperative kernel executes dY peer put,
dV, source gate multiplication, SwiGLU backward, dX, dW2/dW1, peer dX/dGate
return, and top-k reduction. Cache mode consumes the Forward BF16 A/B planes;
no-cache mode recomputes those planes in-kernel from canonical X. dAB lives
until both dX and dW1 finish, and the BF16 U plane lives until dW2 finishes.
dW uses one owner per output tile and increasing canonical padded-M traversal;
dGate uses fixed intermediate-tile partials, and dX reduction follows the
Forward handle's immutable expert-ID `combine_order`.

The pool's `enable_backward` layout reserves dY, dAB/U, dX return/combined,
dGate partial/final, padded K-major transposed dW operands, dW1/dW2, scheduler
and status storage per in-flight generation. U/dAB transposes are fused into
the dV epilogue; dY/X use a coalesced 256x256 shared-memory transpose. Weight
transposes and TMA descriptors are one-time weight/session setup. The steady
Backward launch initializes only its small status/signal words and performs no
CUDA/pinned/event allocation or device-wide synchronization.

An owned expert with zero canonical rows is made ready during generation
setup. It has no dY/dV/dX producer task, but its dW1 and dW2 output tiles remain
real persistent-kernel tasks that vector-store BF16 zero and contribute to the
same expected/completed terminal accounting. This closes the case where a rank
owns only empty experts (common for world=8/top-k=1) without inventing a host
phase or a second kernel.

Each slot also owns a small mapped progress page. The kernel writes it only
when timeout diagnostics are explicitly enabled; after 30 seconds the host may
dump Forward terminal epoch, Backward phase counters and per-expert row state.
This is observation only and never participates in the completion predicate.
In debug-validation builds, Forward also clears the exact FC1/FC2 claim domain
at generation setup. Backward legitimately reuses that arena as prefix scratch;
without the generation-local clear, a scratch offset equal to the next epoch
could be mistaken for an already-claimed task and prevent terminal progress.
The clear is grid-stride parallel and is compiled out of the release path.
Complete-gradient validation hashes are computed on device with fixed chunk
and reduction order. Hash results are staged, not gradient tensors. A
`DeviceBackwardHandle` retains its slot, so two live handles have disjoint
gradient/checkpoint ranges and the slot cannot be reused until the last handle
lease is released.
