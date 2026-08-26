# Performance summary

All formal numbers below are checked-in machine-readable results, not estimates.
They use BF16 input/output, FP32 accumulation, 8192 tokens/rank, 384 experts,
top-k 6, hidden 7168 and intermediate 3072. Each formal comparison uses three
independent sessions, 10 warmups and 100 measurements, with the slowest rank as
the iteration result. Fixture activation, route, gate, dY and actual device
weight hashes are checked before timing.

## Single host, eight GPUs

From `benchmarks/results/forward_release_world8.json`:

| Metric | MegaMOE runs (ms) | Megatron runs (ms) | Improvement |
|---|---:|---:|---:|
| Forward, no cache | 14.9338 / 14.9344 / 14.9387 | 18.2190 / 18.3069 / 18.1783 | 18.03% / 18.42% / 17.82% |

The representative Forward P50/P95 are 14.9218/15.0187 ms. Cache Forward is
16.1698 ms mean and adds 1.2342 ms while retaining A/B for Backward. The actual
standalone launch in that result reports cluster size two, 168 registers/thread,
16 bytes stack/thread and no TMA multicast. The deferred F+B benchmark records
actual cluster size one; cluster audit is therefore reported per harness.

From `benchmarks/results/backward_integrated_world8.json`:

| Metric | MegaMOE runs (ms) | Megatron runs (ms) | Improvement |
|---|---:|---:|---:|
| Backward | 28.6502 / 28.6333 / 28.6909 | 33.7723 / 33.8729 / 33.9355 | 15.17% / 15.47% / 15.45% |
| Notify-inclusive F+B | 46.0079 / 45.9729 / 46.0677 | 54.1746 / 54.2635 / 54.2632 | 15.07% / 15.28% / 15.10% |

The formal training-step boundary is route/gate ready through fused Notify,
Forward, Backward and final gradient drain. Post-Notify timing is diagnostic
only. Steady forbidden allocation/event/device-sync calls are zero.

## Two hosts, sixteen GPUs

From `benchmarks/results/gin_world16_training_20260826.json`:

| Metric | MegaMOE GIN cross-run mean (ms) | Megatron NCCL Socket mean (ms) | Improvement range |
|---|---:|---:|---:|
| Forward | 41.0693 | 816.3075 | 94.70–94.72% |
| Backward | 52.7385 | 720.9331 | 92.52–92.55% |
| Notify-inclusive F+B | 94.0076 | 1465.2237 | 93.46–93.47% |

This comparison records the actual network distinction: MegaMOE uses NCCL
2.29.7 GIN/RDMA same-rail device put/signal, while the measured Megatron
baseline uses NCCL Socket. It is therefore a real end-to-end environment result,
not an assertion that GIN and Socket have equal transport semantics. The
separately measured MegaMOE Socket backend remains a correctness/lower-bound
result in `socket_world16_fallback_20260826.json`.

Per-rank Forward GIN storage is 4,382,039,368 ordinary compute bytes plus
1,766,522,888 registered-window bytes and 2,584 pinned bytes. Recorded per-rail
derived payload bandwidth is about 13.26 GB/s dispatch and 14.88–14.89 GB/s
return. Runtime counters, not formula-only fields, provide put/signal/payload
counts; derived bandwidth and fan-out values retain the `derived_` label.

## Regression policy

The main-shape hash is frozen. Refactoring is accepted only when output and
gradient hashes remain unchanged and Forward, Backward and F+B regress by no
more than 3%. The generic runtime-shape path is correctness-only and excluded
from the main-shape performance claim.

For the final module review, commit `5af8828` (before launch-contract/test
splitting) and the reviewed head were rebuilt and measured back-to-back on the
same host. With 10 warmups and 100 samples, the pre-refactor/current means were
29.5487/29.5287 ms Backward, 15.8548/15.8519 ms Forward, and
45.9335/45.9039 ms Notify-inclusive F+B. The reviewed build is respectively
0.068%, 0.018%, and 0.064% faster in that paired run. Normalized complete SASS
SHA-256 and all kernel resource records are identical; this is the performance
protection evidence for the source/module refactor, independent of older-run
machine variance.

A fresh two-host GIN 10+100 run measured 40.9664 ms Forward, 53.2206 ms
Backward and 95.4062 ms Notify-inclusive F+B. Relative to the checked
41.0693/52.7385/94.0076 ms cross-run means, the changes are -0.25%, +0.91% and
+1.49%, all inside the 3% guard. The corresponding P95 values were
41.0542/53.4385/95.6547 ms.
