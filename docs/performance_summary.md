# Forward performance summary

All reported latency is an end-to-end Forward boundary aggregated by the
slowest rank. Component durations are diagnostic and are never added to create
an artificial total.

The cleanup started from these current-code reference points:

| topology | MegaMOE mean | MegaMOE p95 | comparison |
| --- | ---: | ---: | --- |
| world 8, one H200 node | 15.0235 ms | 15.2690 ms | Megatron performance guardrail |
| world 16, two H200 nodes, GIN | 21.5492 ms | 22.2612 ms | Megatron mean 31.334 ms (1.454x) |

The final post-cleanup autotune and benchmark record should be checked against
these values with identical shape, routes, gates, weights, warmups, sample
count, runtime library, and slowest-rank aggregation. A source-only reorganization
is considered a regression if correctness hashes change or stable latency moves
outside normal run-to-run variance.

Current tuning plans and the latest measured winner are described by the JSON
artifacts emitted by `tools/autotune_forward.py`. Historical cache, generic
single-node fallback, and Backward measurements were removed because their
code and timing boundaries no longer exist.
