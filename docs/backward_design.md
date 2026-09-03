# Backward pipeline design

This document fixes the Backward dataflow and scheduling contract before CUDA
implementation begins. Backward remains unavailable in the current revision;
`backward_pipeline_available()` must continue to return `false` until the full
pipeline is correct and benchmarked.

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

## Fused activation-gradient task

One `DAB` task owns `(expert, M-block, intermediate-N tile)` and performs:

1. a TMA/WGMMA `dU = dR * W2` tile;
2. load cached `A/B` and the canonical FP32 gate;
3. recompute `S` in the epilogue;
4. form `dA/dB` and a fixed-slot `dGate` partial;
5. form and store the exact BF16 `U = gate * S` needed by `dW2`;
6. release-publish completion after every store is visible.

`dU` is never materialized as a long-lived global tensor. `dGate` partials use
one deterministic slot per intermediate tile; floating-point atomics are not
used. A small activation-priority finalize task reduces those slots in
increasing intermediate-tile order.

All expert row ranges are padded to 128. Padding rows are zero in `X`, `dR`,
`A/B`, `U`, and `dAB`, so the GEMM inner loops do not need divergent tail
handling. A 128-row dependency M-block remains the canonical identity even if
autotuning chooses an internal M64 compute tile.

## Four logical work queues

The task domains are known after Notify, so the queues should be precomputed
descriptor tables with readiness counters and CAS claim bitmaps, not wrapping
ring buffers. This preserves multiple independent scheduling domains without
runtime enqueue traffic or head-of-line blocking.

| logical queue | task | ready predicate |
| --- | --- | --- |
| `Q_DAB` | `dU` plus fused SwiGLU backward | required `dR` rows for the M-block have arrived |
| `Q_DX` | `dX` output tile or `dGate` finalize | every intermediate `DAB` tile for that M-block is complete |
| `Q_DW2` | one `dW2` output tile | all `dR` and `U` M-blocks for that expert are complete |
| `Q_DW1` | one `dW1` output tile | all `X` and `dAB` M-blocks for that expert are complete |

Readiness is per M-block or per expert. There is no rank-wide `dY dispatch
complete` prerequisite for an expert's `dW2`, and neither dW queue depends on
all `dX` being computed or sent.

Each `dW` output tile is owned by one CTA, uses FP32 accumulation, traverses
the expert's padded M rows in canonical order, and defaults to `split_k=1`.
The initial dW output tile candidate is 128x128; its M-step, TMA/WGMMA stages,
register split, and cluster policy remain compile-time autotuning parameters.

## Work-conserving CTA policy

Compute CTAs poll all four logical queues at every task boundary. They are
split into compile-time-tuned cohorts so production of new `dX` work cannot be
starved:

```text
dX-latency cohort:  Q_DX  > Q_DAB > Q_DW2 > Q_DW1
dAB-feeder cohort:  Q_DAB > Q_DX  > Q_DW2 > Q_DW1
```

If neither activation queue currently has an issuable task, either cohort may
claim dW immediately. Once a dW tile starts it is not preempted, so a small
activation-reserve cohort and the 128x128 dW tile duration bound the wake-up
latency when a later `dR` wave arrives. There is no age-based fairness between
`dW1` and `dW2`; their order is a throughput/locality tuning choice.

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

## Readiness and immediate return

The dependency chain is entirely release/acquire based:

```text
dR row stores
  -> release dy_arrived_rows[expert, mblock, rail]
  -> acquire and claim Q_DAB tiles
  -> store dAB, U, dGate partials
  -> release dab_done_tiles[expert, mblock]
  -> acquire and claim Q_DX / per-expert Q_DW tasks
  -> store dX tiles and finalize dGate
  -> release return_mblock_ready
  -> Relay local copy or RDMA FULL-return put
  -> release source contribution readiness
  -> fixed-expert-order FP32 dX reduction
```

The last `dX` N tile and `dGate` finalize for an M-block publish its return
readiness immediately. A system-scope fence orders those stores before a
Relay/RDMA worker observes the marker; it is local to that producer-consumer
handoff and is not a barrier. There is never an "all dX output sent" event
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
They never enter the dW queues. Only compute CTAs use opportunistic dW work to
fill communication dependency holes.

## Completion contract

An elected completion worker may publish the terminal epoch only after all
four compute task domains, all local Relay work, every GIN local completion,
all source `dX/dGate` contributions, and every fixed-order `dX` reduction are
accounted for. Other CTAs observe that epoch and exit independently. Terminal
accounting is not on the dataflow critical path and does not introduce a
`grid.sync()`.
