# GIN/RDMA critical-path baseline

This note freezes the evidence collected before changing the cross-node data
plane. The measured shape is 8192 tokens/rank, hidden 7168, intermediate 3072,
384 experts, top-k 6, BF16 inputs/outputs, FP32 accumulation. The physical
topology is two hosts with eight L20X GPUs per host. NCCL 2.29.7 built for CUDA
12.9 provides the device GIN API; the data path uses RoCE v2, GID index 3,
eight 200 Gb/s rails, and GDRDMA.

## Device timeline contract

`MEGAMOE_DEVICE_TIMELINE=1` builds a separate instrumented binary. Each CTA
owns a fixed 256-record slice in a session-preallocated device buffer and only
thread 0 writes that slice, so recording does not require a reservation
atomic. `%globaltimer` is read as a 64-bit value. Overflow increments a
bounded per-CTA counter and never writes beyond the slice. Trace download is
an explicit test operation after completion.

`MEGAMOE_DEVICE_TIMELINE=0` removes the context fields, branch state, calls,
and storage with preprocessing. The artifact gate reported exactly the
versioned release manifest hashes and resources:

| kernel | target | registers | stack | normalized SHA-256 |
|---|---:|---:|---:|---|
| Forward | sm90 | 61 | 0 B | `ffa5bd918c5d84bc90adc726b5ac3fe7cb92ebd9f5ad3c22089f54ea03dbd3c0` |
| Forward | sm90a | 168 | 16 B | `47f6f6ccade6ba27fbddd8719ef5fb4aa9af28a34307e911a8e6d9d8ae24559a` |
| Backward | sm90 | 64 | 0 B | `0de7b814fa516c3e99a8bd56a569bd772b111c2c9d7c8f23227bb10c90fe134d` |
| Backward | sm90a | 168 | 120 B | `206a5366d2e21ef703afd7bd83056963f8ccb7ac37eb9d0f5f95be741bab2bfa` |

Timestamps are compared only within one GPU. The summarizer first forms
durations by `(rank, CTA, channel)` and only then computes cross-rank
percentiles. It never subtracts clocks from different GPUs.

## World-16 critical path

The complete machine-readable summary is
`benchmarks/results/gin_device_timeline_world16_20260827.json`. A world-2
main-shape control is in
`benchmarks/results/gin_device_timeline_world2_20260827.json`.

Representative rank-local offsets from kernel entry are:

| Forward event | first (ms) | last (ms) |
|---|---:|---:|
| dispatch packet ready | 1.362 | 1.368 |
| dispatch GIN issue | 3.326 | 3.392 |
| dispatch local completion | 8.265 | 8.324 |
| proxy relay complete / X tile ready | 9.250 | 9.256 |
| FC1 first / last | 9.259 | 18.368 |
| FC2 tile ready | 17.888 | 24.251 |
| return pack start / end | 24.386 | 25.529 |
| return GIN issue / local completion | 25.575 | 40.381 |
| source relay complete | 44.703 | 44.703 |
| reduce complete / terminal | 45.142 | 45.434 |

| Backward event | first (ms) | last (ms) |
|---|---:|---:|
| dY packet ready | 1.345 | 1.354 |
| dY GIN issue / local completion | 3.400 | 8.402 |
| dY relay complete | 9.317 | 9.317 |
| dV first / last | 9.327 | 17.744 |
| dX ready / return pack start | 26.109 | 26.116 |
| dX GIN issue / local completion | 27.442 | 42.257 |
| source relay / dGate complete | 45.011 | 46.329 |
| dW2 start / end | 46.342 | 51.028 |
| dW1 start / end | 50.991 | 58.241 |
| reduce complete / terminal | 58.678 | 59.396 |

Across all 16 ranks, the P50/P95 spans are 4.971/5.199 ms for dispatch GIN
issue-to-local-completion, 14.787/15.074 ms for Forward return, 4.974/4.983 ms
for dY dispatch, and 14.781/14.853 ms for dX return. Only eight Forward
return puts and 24 Backward return puts carry roughly 352 MB/rank. The first
FC2 output is available roughly 7.69 ms before the first return issue. dW2 and
dW1 start only after dX return and source relay; measured overlap with dV/dX
is zero.

The critical path is therefore global phase granularity, not the frozen GEMM
core:

```text
pack all dispatch -> wait all dispatch -> compute all output
-> pack all return -> wait all return -> source relay/reduce
```

Backward additionally serializes `dW2 -> dW1` after the return path. The next
optimization must stream bounded ready chunks and allow dW to consume ready
experts while dX return is in flight.

## First pipeline fold: Return with dW

The first accepted pipeline change deliberately did not tune the GIN transfer
chunk.  The eight GIN context CTAs remain responsible for return progress;
after dX, every other CTA converts directly to split-K=1 dW work.  Both
branches join before source relay and deterministic token reduction.  Remote
GIN dW tasks use a shared claim counter because context CTAs join the work at
different times.  The local and Socket paths retain their static grid-stride
task ownership.

With the same NCCL 2.29.7 runtime and cache-mode fixture, a 10-warmup/20-sample
screen changed the rank-0 means as follows:

| release world-16 metric | phase-serial | Return/dW pipeline |
|---|---:|---:|
| Forward | 45.44 ms | 45.49 ms |
| Backward | 59.30 ms | 47.99 ms |
| Notify-inclusive F+B | 106.0 ms | 94.77 ms |

All actual dX/dW1/dW2/dGate hashes remained unchanged.  An instrumented
world-16 generation then located dX GIN issue/completion at 27.56/42.56 ms,
dW2 first/last activity at 27.18/45.86 ms, and dW1 first/last activity at
31.01/45.86 ms.  This is real compute/communication overlap rather than a
change in timing boundaries.  The overlap makes the dW spans longer because
TMA/HBM traffic competes with the registered return window, but removes about
11.3 ms from the Backward critical path.

The new critical path is the front of both kernels: dY dispatch/relay still
occupies about 9.3 ms before the first dV task, and Forward dispatch/relay
similarly precedes FC1.  Further work therefore targets per-M-tile readiness
and dedicated Dispatch/Relay/Compute roles.  Transfer chunk, channel and
credit values are compile-time policy candidates selected by full F+B
autotuning; isolated chunk latency is not the optimization objective.

## Nsight Compute boundary

The existing NCCL-free main Forward profile reports 55.97% SM throughput,
66.55% DRAM throughput, 56.29% Tensor Pipe active, 18.75% achieved occupancy,
9.78 warps/issue-active barrier stall, and 3.27 long-scoreboard stall; see
`benchmarks/results/forward_main_world1_ncu.csv`.

Nsight Compute 2025.2.1 cannot directly produce a valid replay report for the
live two-rank GIN persistent protocol. Kernel replay pauses one rank and backs
up tens of GiB while its peer continues; range replay replays CUDA state but
not the external GIN QP/window/signal state. Both modes were fail-closed: the
peer remained in protocol progress and no report was accepted. This is a
tool/replay boundary, not evidence of a kernel failure. World-16 GIN timing
therefore uses `%globaltimer`; NCU remains restricted to NCCL-free compute and
relay helpers where replay does not alter external state. No failed replay is
reported as performance evidence.

## Reproduction

Build the instrumented executable separately:

```bash
make BUILD_DIR=build-timeline MEGAMOE_DEVICE_TIMELINE=1 \
  build-timeline/test_cuda_notify
```

Set `MEGAMOE_DEVICE_TIMELINE_PREFIX` to a per-host writable prefix, launch the
ordinary GIN benchmark with the compatible NCCL runtime, then combine the
rank JSON files:

```bash
python3 tools/analyze_device_timeline.py TRACE.rank*.json \
  --redact-paths --json timeline-summary.json
```

The trace build is diagnostic only. All release performance measurements must
use `MEGAMOE_DEVICE_TIMELINE=0`.
