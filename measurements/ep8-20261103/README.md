# EP8 measurement, 2026-11-03

This archive contains only this measurement campaign. Kernel/runtime source is
`24a20996be2c38191cef0d6cd41945f694c2d413`. The only test change releases unused
PyTorch cached memory before the inference communicator is created; it does not
change benchmark timing or kernels. Flash's first correctness preflight exhausted
memory during NCCL communicator creation. The full preflight passed after this
cleanup, before any performance measurement.

- `results.json`: 36 PF jobs (three shapes × three modes × ABBA) and six DT jobs
  (three shapes × two seeds), including all per-rank samples and workload hashes.
- `manifest.json`: native binary SHA256, source revision, environment and runner hash.
- `summary.json`: pooled medians and per-round medians; GPU telemetry summaries.
- `*.gpu.csv`: 500 ms GPU telemetry (timestamp, GPU index, utilization %, SM MHz,
  memory MHz, temperature C, power W, memory MiB); includes job startup/teardown.
- `correctness.json`: pass-marker counts from the eight-rank FP8 preflight; E=256, top-k=2,
  nominal 65 tokens/rank, including source FP8, checkpoint/optimizer/inference.
  DT timing jobs independently validate diff < 5e-3 and online/prequant bit equality.
- Absolute workspace paths in structured results are replaced by `.`. No timings,
  workload hashes, or other numerical results have been changed.

Hardware: 8× NVIDIA L20X, SM90, 132 SM, 143771 MiB per GPU, 700 W power limit,
driver 570.148.08. PyTorch 2.9.1+cu129, CUDA 12.9. Clocks were not locked.
This is WeftMoE BF16 versus WeftMoE FP8; no optimizer, complete RL job or EP16
measurement is included. See summary for both gains and regressions.

## Reproduce

On an otherwise idle eight-GPU node, from this archive's repository root:

```bash
mkdir -p build/perf_remeasure_20261103
cp measurements/ep8-20261103/{build.py,ep8.py,summarize.py} build/perf_remeasure_20261103/
python build/perf_remeasure_20261103/build.py
python build/perf_remeasure_20261103/ep8.py
python build/perf_remeasure_20261103/summarize.py
```

The build script uses `/opt/.venv/bin/python`; adjust it for your environment.
The archived runner is byte-identical to the measured runner (see manifest).
It reuses already completed Qwen preflights and runs the Flash preflight, then
runs every PF and DT performance job. To repeat both Qwen preflights too, remove
its `if model != 'flash': ... continue` block. Exact preflight settings are in
that loop. Reproduction changes the runner hash and records the new source commit.

PF measures online source quantization and output/tape release; fixed FP8 W/Wᵀ
preparation is excluded. Samples use per-sample maximum across ranks, then p50
across two rounds (60 samples). Separate FWD/BWD measure cache/gradient ready;
cycle uses release-inclusive wall time. DT captures 100 calls, alternates order,
discards its first sweep, and pools 20 replay samples across two routing seeds.
DT prequantized input results exclude upstream quantization.
