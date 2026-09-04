# Backward pipeline design

This document fixes the Backward dataflow and scheduling contract implemented
by the CUDA qualification path. `backward_pipeline_available()` remains false
until the selected world-8 and world-16 configurations complete final
performance qualification.

## Numeric contract and retained Forward state

For one canonical routed row, Forward is:

```text
[A, B] = bf16(FC1(X))
S      = SiLU(fp32(A)) * fp32(B)
U      = bf16(gate * S)
R      = FC2(U)
Y      = fixed-expert-order sum_fp32(R), followed by one BF16 cast
```

Backward receives the source-token gradient `dY`. Because Forward's final sum
is unweighted, every selected route initially has `dR = dY`. The expert-side
math is:

```text
dU     = dR * W2
dGate  = sum_i(dU[i] * S[i])
dS     = gate * dU
dA     = dS * B * SiLU'(A)
dB     = dS * SiLU(A)
dAB    = concat(dA, dB)
dX     = dAB * W1
dW1    = dAB^T * X
dW2    = dR^T * U
```

There is no FC1 recomputation. A training-enabled Forward specialization must
retain the immutable Dispatch context, canonical `X`, canonical gates, and its
`A/B` checkpoints. It must also keep Forward output storage separate from
canonical `X`; the inference-only alias that overwrites `X` cannot be used by
Backward. `S`, `U`, and FC2 output are not retained across Forward/Backward.

The `A/B` checkpoint precision is part of the numeric contract, not merely a
storage choice. Forward now rounds both projections to BF16 before SwiGLU and
stores those same BF16 values when `retain_backward_state` is enabled. Backward
must convert the checkpoints to FP32 for SwiGLU recomputation, so the operands
match Forward exactly without retaining `S`.

## Activation-gradient task chain

The activation path deliberately uses two dependency-local task types:

1. `(expert, M-block, intermediate-N tile)` TMA/WGMMA tasks materialize BF16
   `dU = dR * W2` into the reusable U workspace;
2. after all N tiles for that M-block complete, warp-row epilogue tasks load
   cached A/B and the FP32 gate, recompute S, write BF16 dA/dB, reduce dGate in
   a fixed column order, and overwrite the temporary dU row with the exact
   BF16 `U = gate * S` consumed by dW2.

The dU buffer is phase-local rather than retained state. No FP32 atomics or
cross-CTA dGate partial reduction are used. Splitting the GEMM and scalar
epilogue avoids extending every WGMMA task with the long dependency chain of
SwiGLU recomputation; a fused experiment regressed both local and two-node
latency and was removed.

All expert row ranges are physically padded to 128. Padding rows are zero in
`X`, `dR`, `A/B`, `U`, and `dAB`, so the GEMM inner loops do not need
divergent tail handling. A 128-row dependency M-block remains the canonical
identity even if autotuning chooses an internal M64 compute tile. The setup
kernel separately precomputes each expert's dW reduction extent as the valid
row count rounded up to the 64-row WGMMA K tile. dW therefore skips a final
all-zero 64-row K stage when the physical 128-row padding does not require it;
storage addresses, readiness accounting, and M-block identity remain padded
to 128.

## Four logical work queues

The task domains are known after Notify, so the queues should be precomputed
descriptor tables with readiness counters and CAS claim bitmaps, not wrapping
ring buffers. This preserves multiple independent scheduling domains without
runtime enqueue traffic or head-of-line blocking.

| logical queue | task | ready predicate |
| --- | --- | --- |
| `Q_DAB` | dU N tile, then warp-row SwiGLU backward batches | required dR rows have arrived; row batches wait for all dU N tiles |
| `Q_DX` | one dX output tile | every SwiGLU row batch for that M-block is complete |
| `Q_DW2` | one `dW2` output tile | all `dR` and `U` M-blocks for that expert are complete |
| `Q_DW1` | one `dW1` output tile | all `X` and `dAB` M-blocks for that expert are complete |

Readiness is per M-block or per expert. There is no rank-wide `dY dispatch
complete` prerequisite for an expert's `dW2`, and neither dW queue depends on
all `dX` being computed or sent.

Each `dW` output tile is owned by one CTA, uses FP32 accumulation, traverses
the expert's valid M rows plus at most 63 zero padding rows in canonical order,
and defaults to `split_k=1`.
The selected dW output tile is 128x256x64 with three stages; its M/N tile,
stage count, and register split remain compile-time tuning parameters.

## Work-conserving CTA policy

All compute CTAs belong to one work-conserving pool. At every task boundary a
CTA probes each logical queue at most once, in this fixed priority order:

```text
Q_DX > Q_DAB > Q_DW2 > Q_DW1
```

There is no second `Q_DX` probe before falling through to dW. Queue descriptors
and their readiness state persist, so a task published just after its probe is
not lost; a CTA observes it at its next task boundary. If neither activation
queue currently has an issuable task, the CTA may claim dW immediately. Once a
dW tile starts it is not preempted, so its tile duration bounds the wake-up
latency when a later `dR` wave arrives. There is no static CTA reservation, CTA
cohort, priority-burst policy, or age-based fairness. `dW2` is always probed
before `dW1`; tile shape and stage count remain compile-time tuning choices.
While activation work remains, each dW claim is deliberately limited to one
tile so that a CTA cannot hoard a long chunk across a newly published dX/dAB
task. Once dX is complete, the topology-specific compile-time claim size is
used; the qualified world-8 and world-16 settings both use four tiles.

Activation owners are stored as expert-major, contiguous M128 blocks. The
selected `ready_word_stagger=0` specialization therefore starts every CTA's
complete ready-bitmap probe at word zero. Atomic tile claims still distribute
work across CTAs. Within the first non-empty ready word, dAB/dX probes rotate
over an aligned group of four neighboring owners while each owner continues
to claim N tiles from zero upward. Neighboring M-blocks consequently request
the same W1/W2 N tile close together and reuse it from L2. dW probing keeps the
single-owner order because spreading it across experts destroys weight and
reduction-input locality. This is only a scan-order hint: a miss visits every
bitmap word, queue priority is unchanged, and there is no owner affinity, new
global cursor, or synchronization. `activation_owner_group` is a compile-time
JSON parameter; the narrow plans retain 1, 2, and 8 as nearby candidates.
`ready_word_stagger=1` remains representable by the explicit tuning surface
for deliberate experiments, but is not in the maintained narrow search.

The execution therefore has two conceptual periods but no physical phase
transition:

- while communication exposes `dR`, activation work has priority and dW fills
  otherwise idle SM intervals;
- after all activation descriptors are claimed and completed, every compute
  CTA naturally finds only dW work.

No CTA waits at a phase flag, no grid-wide synchronization occurs, and dW may
already be in flight while later experts are still producing or returning
`dX`. Communication CTAs continue draining `dX` after compute has shifted
mostly to dW.

The primary compute grid contains `SM count - transport CTAs` workers, which
guarantees that every RDMA and Relay CTA can remain resident. After the
persistent transport grid exits and the queued dX reduction drains, the now
free SMs receive a second, small launch of the same compute kernel. These late
workers join the existing global queues and terminal counters; they are not a
separate dW implementation and do not wait for or synchronize the primary
compute grid. In the steady benchmark they find the remaining dW tail, while
the scheduler remains correct if another task kind is still visible. This
stream-ordered handoff avoids leaving the former communication SMs idle
without relying on CUDA stream priority to reserve SM residency.
The qualified world-16 specialization reserves 8 RDMA and 16 Relay CTAs; the
qualified single-node specialization reserves 12 Relay CTAs and launches no
RDMA workers. These counts are explicit JIT configuration values rather than
runtime environment switches.

## Readiness and immediate return

The dependency chain is entirely release/acquire based:

```text
dR row stores
  -> release dy_arrived_rows[expert, mblock, rail]
  -> acquire and claim Q_DAB tiles
  -> store dAB, U, and row-complete dGate
  -> release dab_done_tiles[expert, mblock]
  -> acquire and claim Q_DX / per-expert Q_DW tasks
  -> store dX tiles and finalize dGate
  -> release return_mblock_ready
  -> Relay local copy or RDMA FULL-return put
  -> release source contribution readiness
  -> fixed-expert-order FP32 dX reduction
```

The last dX N tile for an M-block publishes its return readiness immediately;
dGate is already complete at that point. A system-scope fence orders those
stores before a Relay/RDMA worker observes the marker; it is local to that
producer-consumer handoff and is not a barrier. There is never an "all dX
output sent" event
that launches dW.

## Reuse of Forward communication

- `dY` dispatch reuses Forward's same-rail RAIL packetization, row geometry,
  Relay fanout, and M-block arrival counters. Gates and reverse metadata come
  from the retained Dispatch context.
- `dX` uses Forward's local Relay and cross-node FULL-return path. Since gate
  is already present in `dS`, source reduction is the same unweighted,
  fixed-expert-order FP32 sum used by Forward.
- `dGate` is one FP32 scalar per route. Local Relay scatters it directly;
  cross-node return should piggyback it on the corresponding FULL-return
  metadata/marker when that does not add a GIN put.

RDMA and Relay CTAs remain dedicated transport owners for the full generation.
They never enter the dW queues. The late workers are newly launched compute
CTAs after those transport CTAs have exited; only compute CTAs use
opportunistic dW work to fill communication dependency holes.

## Completion contract

An elected completion worker may publish the terminal epoch only after all
four compute task domains, all local Relay work, every GIN local completion,
all source `dX/dGate` contributions, and every fixed-order `dX` reduction are
accounted for. Other CTAs observe that epoch and exit independently. Terminal
accounting is not on the dataflow critical path and does not introduce a
`grid.sync()`.
