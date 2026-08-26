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

The API-v2 release candidate was freshly remeasured after replacing the local
success-path terminal NCCL AllGather with the existing peer-visible status
matrix and terminal epoch. With the same actual fixture and explicitly loaded
NCCL 2.28.3 runtime, the three MegaMOE Backward means were
28.5997/28.5978/28.6096 ms and the formal Notify-inclusive F+B means were
45.0336/45.0559/45.1161 ms. Fresh Megatron-Core 0.17.0 means were
33.7057/33.8497/33.8405 ms Backward and 54.1241/54.2496/54.3071 ms F+B. The
per-run improvements are 15.15%/15.52%/15.46% and
16.80%/16.95%/16.92%; the fail-closed comparison gate passes every run.
Release local-rank success and structured-error agreement performs no NCCL
status collective; Debug, Socket and GIN retain their fixed-width terminal
AllGather. The CUDA-event Backward boundary and Notify-inclusive host-wall
boundary remain unchanged.

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

The correctness-only generic GIN benchmark at world 2, H/I=127/65, two
tokens/rank and top-k 6 used 10 warmups plus 10 measurements. Slowest-rank
means were 0.442 ms Notify, 0.368 ms Forward device span, 5.904 ms Backward
device span and 6.776 ms Notify-inclusive F+B wall. These small-shape numbers
only demonstrate that the real GIN path executes without a host payload phase;
they are not compared with Megatron and are not a tuning claim.

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

After the final generic-GIN/API-v2 commits, short 10-warmup/10-measurement
guards rechecked the untouched main specialization. Single-node world 8 means
were 15.543 ms Forward, 28.962 ms Backward and 45.039 ms Notify-inclusive F+B.
Two-node world 16 GIN means were 40.946/53.149/95.297 ms respectively (P95
40.984/53.321/95.543 ms). These short guards are not substituted for the
10+100 records above; together with the byte-identical normalized main-kernel
SASS they show that the final correctness/API work stayed within the 3% guard.
