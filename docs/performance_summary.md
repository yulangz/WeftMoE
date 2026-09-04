# Forward performance summary

Latency is measured at the end-to-end Forward boundary and aggregated as the
slowest rank for each iteration. Ten warmups are discarded and the final
validation contains 100 measured iterations. Component intervals are
diagnostic overlap intervals; they are not added to manufacture a total.

## Post-cleanup result

| topology | before cleanup | tuned after cleanup | p95 after | change |
| --- | ---: | ---: | ---: | ---: |
| world 8, one H200 node | 15.0235 ms | 14.6285 ms | 14.9184 ms | -2.63% |
| world 16, two H200 nodes, GIN | 21.5492 ms | 21.2461 ms | 21.6320 ms | -1.41% |

The immediate pre-cleanup world-16 Megatron reference was 31.334 ms under the
same main-shape workload. The post-cleanup MegaMOE result is 1.475x faster.
Every compiled candidate and each final validation produced the same input,
route, gate, weight, and output correctness fingerprint.

The winners were rebuilt and measured again at final commit `a6cb123`. World 8
measured 14.6920 ms over 100 iterations, 2.21% below its pre-cleanup result.
World 16 was deliberately repeated because run-to-run jitter was larger than
the tuning margin: four independent 100-iteration intervals ranged from
21.2675 to 21.7369 ms and averaged 21.5508 ms. That aggregate differs from the
21.5492 ms pre-cleanup result by 0.007%, so the cleanup introduced no
measurable regression; even the conservative aggregate remains 1.454x faster
than the 31.334 ms Megatron reference. Both rebuilt winners retained the same
correctness fingerprints and compiled-kernel digests as the search winners.

The selected common compute shape is M128, FC1-N128, FC2-N256, K64, with four
TMA stages in each GEMM, a two-CTA cluster, 232 math registers, 40 producer
registers, FC2 TMA store, and queue-based reduction. Topology-specific role
choices are:

- world 8: zero RDMA CTAs, 16 Relay CTAs, four local-dispatch TMA warps;
- world 16: eight RDMA CTAs, eight Relay CTAs, twelve local-dispatch TMA
  warps.

The world-8 search compiled four candidates and took 225.790 seconds. The
world-16 search compiled seven candidates and took 570.583 seconds. Combined
dynamic compilation, candidate measurement, and winner validation took
796.374 seconds (13 minutes 16 seconds).

The compact source-controlled evidence is
[`benchmarks/results/forward_cleanup_20260902.json`](../benchmarks/results/forward_cleanup_20260902.json).
The explicit winner defaults and retained regression candidates are in
`configs/forward_h200_world8_autotune.json` and
`configs/forward_h200_autotune.json`. Historical cache, generic single-node
fallback, and Backward results were removed because those data paths and
measurement boundaries no longer exist.

## Gate-before-FC2 semantic alignment

The subsequent semantic correction applies each route's FP32 gate in the FC1
epilogue before the single BF16 cast consumed by FC2. Return reduction is now
an unweighted, fixed-expert-order FP32 sum. An independent NumPy FP64 oracle
matched all selected coordinates exactly (`max_abs=0`, `rmse=0`), and both GIN
and Socket two-host paths passed their correctness runs.

The selected tile/stage/CTA configuration was not retuned because the change
does not alter GEMM or transport geometry. Two world-8 intervals averaged
14.6616 ms. Three world-16 GIN intervals ranged from 21.3848 to 21.7787 ms and
averaged 21.5593 ms. The latter differs by 0.04% from the previous four-run
21.5508 ms aggregate and remains 1.453x faster than the 31.334 ms Megatron
reference. The hot GIN fanout record remains 32 bytes, Dispatch remains four
puts per wave, and the compiled compute kernel remains at 168 registers per
thread.

The compact evidence is
[`benchmarks/results/forward_gate_before_20260903.json`](../benchmarks/results/forward_gate_before_20260903.json).

## Backward target baseline

Before implementing the new Backward pipeline, Megatron-Core 0.17.0 was
measured with the same retained-state contract. Forward setup is outside the
timed region; selective `moe_act` recomputation retains routed X and BF16 FC1
A/B, releases U, and recomputes SwiGLU plus the gate multiply during Backward.
The timed region includes dY dispatch, dAB, dX, dGate, dW1, dW2, and dX
combine.

| topology | mean BWD-only host wall | p95 |
| --- | ---: | ---: |
| world 8, one H200 node | 30.2048 ms | 31.2478 ms |
| world 16, two H200 nodes, RDMA | 43.0583 ms | 43.4779 ms |

On both topologies, selective recomputation matched Megatron without activation
recomputation byte-for-byte for dX, dGate, dW1, and dW2, and a second cached
X/A/B Backward execution was byte-identical. The world-16 run used all eight
RoCE rails with GDRDMA and no Socket payload fallback. Compact evidence is in
[`benchmarks/results/backward_megatron_cache_x_ab_20260903.json`](../benchmarks/results/backward_megatron_cache_x_ab_20260903.json).

## Backward split-pipeline checkpoint

The qualification implementation now uses one persistent compute kernel and
one split-role transport kernel, with dX/dAB queues taking strict priority and
dW tiles filling otherwise idle compute CTAs. A narrow runtime-JIT search found
different scheduler points for the two maintained topologies:

| topology | selected dispatch rows | selected dW claim | MegaMOE mean | Megatron device mean | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| world 8, one H200 node | 32 | 8 | 29.8137 ms | 30.1810 ms | 1.012x |
| world 16, two H200 nodes, GIN | 64 | 4 | 32.1809 ms | 43.0339 ms | 1.337x |

Both final runs discarded ten warmups and measured 100 iterations using the
slowest rank per iteration. On world 16, compute averaged 31.4047 ms while the
transport interval averaged 27.7654 ms, so transport is hidden under compute
at the end-to-end mean boundary. The separately launched dX reduction still
adds a 0.7931 ms mean tail; folding this work into the work-conserving
scheduler is the next optimization target.

The selected compute kernel uses 168 registers and 80 stack bytes per thread;
the transport kernel uses 72 registers and 8 stack bytes. Candidate preflights
matched all four output fingerprints, repeated execution was byte-identical,
and an independent PyTorch oracle remained within BF16 accumulation tolerance.
Compact evidence is in
[`benchmarks/results/backward_python_jit_20260904.json`](../benchmarks/results/backward_python_jit_20260904.json).

## Backward late-worker checkpoint

The transport and reduction grids finish before the dW critical path. A
stream-ordered second launch of the same persistent compute kernel now fills
the SMs released by those dedicated roles and joins the existing global work
queues. It does not add a dW-only kernel, change a transport CTA into a compute
CTA, or introduce a grid-wide synchronization.

Paired 300-iteration runs measured the following slowest-rank means:

| topology | previous MegaMOE | late workers | latency reduction | Megatron speedup |
| --- | ---: | ---: | ---: | ---: |
| world 8, one H200 node | 29.1292 ms | 28.7763 ms | 1.21% | 1.049x |
| world 16, two H200 nodes, GIN | 31.2920 ms | 30.5813 ms | 2.27% | 1.407x |

After adding late workers, the world-8 JIT search was repeated over six narrow
candidates. It selected four-tile dW claims over the former eight-tile claims
by 1.20% under the confirmation policy. The independent final run measured
28.4960 ms mean and 30.5961 ms p95 over 100 iterations, or 1.059x over the
30.1810 ms Megatron device baseline. The full search took 175.8 seconds.

The corresponding eight-candidate world-16 search selected 16 Relay CTAs plus
8 RDMA CTAs, while retaining 64-row dispatch batches and four-tile dW claims.
Two interleaved confirmation runs improved the selection score by 2.39% over
the former 12-Relay point. The independent final run measured 30.0128 ms mean
and 31.6778 ms p95 over 100 iterations, or 1.434x over the 43.0339 ms Megatron
device baseline. The full compile, preflight, tune, confirmation, and final
sequence took 587.1 seconds.
Because 16 Relay CTAs was the upper search boundary, a follow-up search tested
20 and 24. Relay=20 was 0.08% slower after paired confirmation and relay=24
regressed further, so 16 remains the interior winner. Its follow-up final run
measured 29.8826 ms mean (1.440x over Megatron).

The compute kernel remains at 168 registers, 24 stack bytes, zero local bytes,
and 213104 dynamic shared-memory bytes per CTA. Small- and full-shape repeated
validation was byte-identical on all four outputs for both topologies. A first
attempt to over-subscribe the initial compute grid deadlocked because stream
priority does not reserve transport residency; the retained design launches
late workers only after transport and reduction have completed. Compact
evidence is in
[`benchmarks/results/backward_late_workers_20260904.json`](../benchmarks/results/backward_late_workers_20260904.json).

## Backward vector dX reduction

The source-token reducer now assigns one token to one warp and loads eight
BF16 values per `uint4`. Lanes 0--5 independently acquire the six route
markers, while eight FP32 accumulators retain the original route-position
order and round once to BF16. Both top-k=1 and top-k=6 validation produced the
same all-rank fingerprints as the scalar predecessor and remained
byte-identical across repeated executions. The kernel uses 38 registers and
has no stack or local-memory spill.

Two paired 300-iteration world-8 runs reduced the isolated reduction mean from
3.0950 ms to 1.6519 ms (46.63%) and the end-to-end mean from 28.4497 ms to
28.3038 ms (0.51%). Two paired 100-iteration world-16 runs reduced the
isolated reduction mean from 1.2506 ms to 0.4979 ms (60.18%); because the
primary compute grid remains the critical path, end-to-end improved only
0.19%, from 29.9085 ms to 29.8527 ms. A 128/256/512-thread sweep found 128
threads slightly faster for the isolated world-8 reducer, but less than 0.01%
apart end-to-end, so the existing 256-thread default remains selected under
the 0.5% promotion rule. Compact evidence is in
[`benchmarks/results/backward_vector_reduce_20260904.json`](../benchmarks/results/backward_vector_reduce_20260904.json).

## Backward dW reduction-tail trimming

Physical expert storage and dependency M-blocks remain padded to 128 rows, but
the setup kernel now records a separate dW reduction extent rounded to the
64-row WGMMA K tile. dW1 and dW2 read that immutable extent and skip the final
all-zero K stage where possible. The persistent loop gains no branch, grid
synchronization, or communication dependency.

Two 300-iteration world-8 candidate runs averaged 28.0376 ms, versus a
28.3056 ms same-binary-generation control, a 0.95% reduction. Two
100-iteration world-16 GIN candidate runs averaged 29.7221 ms, versus
29.8674 ms across two interleaved controls, a 0.49% reduction. The resulting
means are 1.076x and 1.448x faster than the corresponding retained-state
Megatron device baselines.

The compute kernel remains at 168 registers, 24 stack bytes, zero local bytes,
and 213104 dynamic shared-memory bytes. Its static HGMMA and dependency-barrier
counts remain exactly 72 and 30; earlier runtime-rounding variants increased
the latter to 80 and regressed, so they were discarded. Top-k=1 and top-k=6
repeated validation passed, top-k=6 fingerprints matched the qualified
predecessor exactly, and the two-node GIN path passed all-rank repeated-output
validation. Compact evidence is in
[`benchmarks/results/backward_dw_k64_20260904.json`](../benchmarks/results/backward_dw_k64_20260904.json).

## Backward expert-local queue probing

Activation M-block owners are laid out contiguously by expert. The scheduler
now defaults to starting every CTA's complete ready-bitmap scan at word zero,
so atomic claims consume neighboring M-blocks while the same expert's W1/W2
tiles are more likely to remain in L2. The old `blockIdx.x % word_count`
stagger remains an explicit compile-time autotune candidate. The change adds
no queue cursor, barrier, grid synchronization, or CTA affinity and preserves
the strict `dX > dAB > dW2 > dW1` probe order.

Paired long runs measured the following slowest-rank means:

| topology | staggered scan | expert-local scan | reduction | Megatron speedup |
| --- | ---: | ---: | ---: | ---: |
| world 8, one H200 node | 28.0461 ms | 27.3243 ms | 2.57% | 1.105x |
| world 16, two H200 nodes, GIN | 29.7332 ms | 28.9468 ms | 2.64% | 1.487x |

The generated final specializations independently measured 27.3165 ms and
28.9279 ms. A 30-iteration rank-0 Nsight profile shortened the primary
compute kernel from 27.3461 ms to 26.6242 ms. During its first 16 ms, Tensor
Pipe activity increased from 48.33% to 50.21% and L2 hit rate from 20.60% to
23.23%, while SM activity remained effectively 100%. The compute kernel still
uses 168 registers, 24 stack bytes, zero local bytes, and 213104 dynamic
shared-memory bytes. Both one-node and actual two-node GIN five-repeat
validation remained byte-stable for dX, dGate, dW1, and dW2. Compact evidence
is in
[`benchmarks/results/backward_owner_locality_20260904.json`](../benchmarks/results/backward_owner_locality_20260904.json).

## Backward activation-owner grouping

Starting every bitmap scan at the expert-major head improved locality, but it
also made many CTAs contend for all N tiles of one M-block at a time. The
activation scheduler now rotates CTAs over four adjacent ready M-blocks while
each M-block still claims N tiles in ascending order. This makes nearby CTAs
request the same W1/W2 tile close together. The rule applies only to dAB/dX;
dW retains the original single-expert order. It adds no atomic, barrier,
`grid.sync()`, queue pass, or communication dependency.

The explicit compile-time search compared owner groups 1, 2, 4, 8, and 16.
Group 4 won on both maintained topologies:

| topology | group 1 | group 4 final | reduction | Megatron speedup |
| --- | ---: | ---: | ---: | ---: |
| world 8, one H200 node | 27.3004 ms | 25.8043 ms | 5.48% | 1.170x |
| world 16, two H200 nodes, GIN | 29.0264 ms | 27.9878 ms | 3.58% | 1.538x |

The world-8 alternatives measured 26.0461 ms for group 2, 26.4988 ms for
group 8, and 27.4045 ms for group 16. World 16 measured 27.9982, 28.3402,
and 28.7694 ms respectively. Rank-0
Nsight shortened the primary compute kernel from 26.6242 ms to 25.8144 ms.
During its first 16 ms, Tensor Pipe activity rose from 50.21% to 57.12% and
L2 hit rate from 23.23% to 28.95%; SM activity remained 99.9% and L2-to-HBM
pressure fell from 82.78% to 82.02%. The compute resource footprint remains
168 registers, 24 stack bytes, zero local bytes, and 213104 dynamic shared
bytes. Top-k=1/6 local validation and actual two-node top-k=6 validation were
byte-stable across five repeats. Compact evidence is in
[`benchmarks/results/backward_activation_owner_group_20260904.json`](../benchmarks/results/backward_activation_owner_group_20260904.json).

## Backward single-node dispatch wavefront

With four-owner activation locality selected, the single-node Relay batch was
revisited below the former 32-row boundary.  A 16-row batch performs more
system fences and therefore moves the all-routes dispatch timestamp later,
but exposes partially filled expert M-blocks soon enough to shorten the
persistent compute critical path.  It changes neither the copy operation nor
the M-block arrival/release protocol and adds no synchronization with compute.

Two 210-iteration measurements in each order produced 25.6292 ms for the
16-row point and 25.8356 ms for the former 32-row point, a 0.80% reduction in
slowest-rank mean latency.  The primary compute interval improved by the same
0.80%, from 25.8169 ms to 25.6110 ms, while the transport interval shortened
from 15.3734 ms to 15.1080 ms.  The dAB/dX duration counters grow because
their start timestamp moves earlier; their completion relative to kernel
launch is what improves.  Top-k=1 and top-k=6 five-repeat validation passed
the independent oracle and remained byte-identical.  The world-16 GIN point
keeps its independently selected 64-row batch.  A 210-iteration lower-bound
control measured 25.6143 ms at 12 rows versus 25.5903 ms at 16 rows, so 12 is
retained only as a nearby autotune candidate.

A rank-0 Nsight profile independently reduced the primary compute interval
from 25.8144 ms to 25.5135 ms and transport from 15.3707 ms to 15.0505 ms.
During the first 2 ms, Tensor Pipe activity rose from 29.97% to 40.91% and L2
hit rate from 25.82% to 27.36%; subsequent activation buckets remain near 60%
Tensor activity.  This localizes the gain to the initial readiness wave rather
than a change in steady GEMM throughput.  Compact evidence is in
[`benchmarks/results/backward_dispatch_wavefront_20260904.json`](../benchmarks/results/backward_dispatch_wavefront_20260904.json).
