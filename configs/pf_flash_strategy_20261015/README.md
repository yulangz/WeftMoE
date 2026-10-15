# Flash PF dataflow profile

Current M19 configuration: use **B2048 on both EP8 and EP16**, contribution
CLI depth 2 and network credit4. `return_rails=1` keeps FC2 TX on the computing
GPU and reuses its compute-slot credit; Put workers now scan their statically
owned ready blocks instead of waiting on the first unfinished block.
Candidate digest `14727edb3f32`, compiled ID `3172e19487b0`.
Canonical cross-window M128, local row-striped bulk and overlapping 96-thread
BWD reduce are retained. See M19 in [the optimization record](../../docs/pf_backward_20261012.md)
for the paired measurements and the negative all-rail ablation. Do not apply
Qwen's all-rail choice to Flash without measuring it.

## Historical whole-profile experiment (before M17)

Update after M17: the current header/candidate have removed M-block-owned
pull and full-CTA BWD reduce (compiled ID `e8955a4e0df6`). Local pull is again
row-striped bulk; 96 communication threads reduce concurrently with dW.
The original experiment described below used `6f3b9992e498`; its exact
configuration is preserved in the linked result JSON and git history.
The historical timings below do **not** describe the updated implementation.

H4096/I2048/E256/top6 BF16. This is an explicit experimental profile, not a
replacement of Flash's default configuration or a new autotune winner.
Compiled ID: `6f3b9992e498`.

The Qwen profile's dataflow switches are enabled together: local three-warp
M-block vector pull and compact contributions; network ordered dispatch,
coalesced Get, window-ordered tape, pending4 return with precomputed spans and
striped completion signals, full-CTA BWD reduce, private dR and credit4.
Keep Flash's original M128/N256/K64, three stages, cluster-N2 and dW chunks 4/1.
Node deduplication already existed before these switches.

Use B2048 on EP8 (one eight-GPU node), B1024 on EP16 (two nodes), and pass
`--ring-depth 2`. Network contribution capacity covers the invocation; its
actual window count is derived from prepared token capacity and B, not this
CLI D. Compiled wire credit is 4. Do not override the latter with the old
profile's credit2 environment setting.

Build with the existing helper:

```bash
/opt/.venv/bin/python benchmarks/build_pf_main_timeline.py flash_strategy_20261015 \
  --profile configs/pf_flash_strategy_20261015
```

Run `benchmarks/run_pf_network.py` with this artifact and
`tools/benchmark_training.py --hidden 4096 --intermediate 2048 --experts 256
--top-k 6 --ring-tokens 2048 --ring-depth 2 --tokens 8192 20480
--route-distribution balanced --forward-only --prime-trace --warmups 4 --samples 16`.
For EP16 set `--nodes 2` on the launcher and `--ring-tokens 1024` on the benchmark.
Repeat for `random` and `--backward-only`. Numerical verification is separate
(`--verify --warmups 0 --samples 2 --tokens 129 8192 20480`).

The full 64-point comparison (old PF, current code/original Flash strategy,
this profile, and Megatron) does not support replacing Flash's defaults with
this profile. Against current code/original strategy, the primary run improves
balanced EP16 8K BWD by 14.0% in throughput, but reduces EP16 FWD throughput by
7.4%–17.6%. These are whole-profile comparisons, not individual-switch ablations.
The original Flash and validated Qwen configurations remain unchanged.

Numerical verification passed for both topologies, with two changing tapes
at each of 129/8K/20K tokens, checking output/dX/dGate/dW1/dW2. Both nodes were
requested to lock at 1620 MHz; sparse observations still include lower clocks.
Results, reverse-order checks and clock caveats are recorded in
[the PF optimization record](../../docs/pf_backward_20261012.md), with raw samples
and binary/workload hashes in
[the result JSON](../../benchmarks/results/flash_pf_strategy_20261015.json).

M16 in the same record subsequently isolates the regressions: M-block-owned
local pull, the smaller Forward window, and full-CTA Backward reduce. Compact
contributions also help Flash when row-striped bulk pull is retained. These
diagnostic counterfactuals do not change this frozen whole-profile experiment
or select a new production default.
