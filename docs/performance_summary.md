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
