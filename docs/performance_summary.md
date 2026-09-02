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
