# Benchmarks

Historical V3/earlier-PF measurements are retained. Reproduce their removed
kernels and build profiles from tag `archive/pre-pf-dt-cleanup-20261012`.
Current operator benchmarks accept PF/DT, not the deleted V3 backend.

Use [tools/run.py](../tools/run.py) to benchmark FWD or BWD with an explicit
configuration, or to autotune before benchmarking. Native CUDA checks and rank
launchers are maintained in [tests](../tests).

The [1620 MHz saved-X report](saved_x_20260917.md) records FC1 checkpoint
stores and training buffer reuse, with paired FWD/BWD measurements against the
completion-latency checkpoint. It includes memory savings, batch/EP invariance,
retained-state reuse, gradient comparisons, and device-memory sanitizer checks.
All samples and validation evidence are in its
[machine record](saved_x_20260917.json).

The [1620 MHz combined FC1 experiment](fc1_combined_20260917.md) records exact
BF16 activation lookup, packed gate/up WGMMA, and TMA checkpoint stores, with
8,680 training/inference and supplemental samples and bitwise validation. The
Qwen cross-node regression persists. Longer Flash sampling improved P95/P99,
although the candidate maximum remained higher; this candidate is not enabled
in the production source.

The [1620 MHz FC1 layout controls](fc1_layout_controls_20260917.md) isolate
pool allocation, unused activation-table initialization, and parameter layout.
They include complete 128-bit machine-code comparisons, bitwise FWD/BWD
references, reverse-order timing, and continuous frequency sampling.

The [1620 MHz interleaved FC1 experiment](fc1_interleaved_20260917.md) compares
independent baseline and candidate kernels in the same process and memory pool.
It verifies each round's selected path, checks complete output equality across
paths, and reports adjacent timing differences in both orders. The Qwen EP16
64K regression remains visible with this control.

The [FC1 lookup/cache controls](fc1_lookup_cache_20260917.md) use five-call
blocks to separate switching effects from later calls. N256 alone is nearly
flat at Qwen EP16/64K, while the tested `__ldcg` lookup implementation regresses
on both topologies. Its exhaustive numerical checks passed, but it is rejected
on performance grounds.

The [BF16 activation reciprocal experiment](fc1_reciprocal_20260917.md) replaces
the lookup with a corrected reciprocal and preserves the tested FP32 activation
factor bits across all BF16 inputs on all 16 GPUs. Qwen 64K paired measurements
show small gains in the diagnostic layout. The subsequent
[full production-layout validation](fc1_reciprocal_full_20260917.md) passed its
bitwise and lifecycle checks but found repeatable Flash and Qwen EP16/64K
regressions, so the candidate is not adopted. Its 4,640 timing samples include
960 continuous BWD samples retained as diagnostic data: Flash 64K process
windows showed power throttling and cannot establish strict 1620 MHz timing.

The [Backward clock audit](backward_clock_audit_20260918.md) correlates native
call windows with NVML samples. A 500 ms idle interval passed one run, but
longer intervals still had sporadic deviations; no fixed-spacing protocol is
yet established as reliably sustaining 1620 MHz. The separate
[BF16 derivative probe](bwd_reciprocal_probe_20260917.md) found exact sigmoid,
SiLU, and derivative factors on all 16 GPUs; it is not evidence of an operator
speedup.

The [Backward reciprocal matrix](bwd_reciprocal_20260918.md) reports the
arithmetic-only candidate across both models, EP8/EP16, and 8K/64K tokens.
The candidate is adopted. It retains all 1,920 primary timing samples, separately identifies calls with
complete 1620 MHz observations, and includes gradient and lifecycle checks.
Flash improves about 0.8%; Qwen results are mixed. Actual frequency deviations,
missing observations, and host-wall tails are disclosed in the report. A fixed
800-call Qwen EP16/8K follow-up did not reproduce a candidate-specific tail
regression; its roughly 0.08% median slowdown remains recorded.

The [dX task-claim experiment](bwd_dx_claim_20260918.md) batches two adjacent
tiles without changing GEMM arithmetic. Qwen EP8 improves 2.50% at 8K and
2.06% at 64K, but EP16 regresses, so this version is not adopted as a uniform
default. All 960 samples, frequency coverage, bitwise gradients, lifecycle
checks, and concurrent scheduler tests are retained. The subsequent
[local-only specialization](bwd_dx_local_20260918.md) is adopted: Qwen EP8
improves 2.79% / 1.95% at 8K / 64K, and Flash EP8 improves 0.63% / 0.60%.
GIN and Forward main-kernel machine code remain identical to the baseline.
The report includes fresh 960-call timing, frequency-qualified comparisons,
local and remote lifecycle checks, an odd tile-group tail, and memcheck.
P95/P99 improve in all four local cases, while isolated 8K wall maxima increase.
That optimization round ended at the user's request.

The subsequent user-requested [Notify row-order experiment](notify_row_order_20260918.md)
tests whether atomic expert-row allocation can replace stable local sorting.
Qwen EP8 Forward, dX, and dgate remain bitwise equal in the tested cases, but
independent Notify executions change dW1/dW2. A stable run-start scan also
preserves all reference tensors. Neither candidate shows a consistent full-call
speedup in the 360-sample comparison, so the production implementation is retained.

The [1620 MHz FC1 N256 experiment](fc1_packed_20260917.md) records a separate
gate/up packing candidate, 3,200 paired samples, and bitwise validation. Its
small gains and remaining integration work are documented; it is not enabled
in the production source.

The [1620 MHz completion-latency report](completion_latency_20260917.md)
records the stream-ordered mapped completion change, its CPU polling tradeoff,
and a fresh eight-case matrix with repeat checks disabled. All raw samples,
validation evidence, and source/binary hashes are in its
[machine record](completion_latency_20260917.json).

The [1620 MHz Notify radix report](notify_radix_20260917.md) records stable-sort
optimization, two-order paired measurements, and batch/EP bitwise checks.
These historical timings had repeat checks enabled outside timing and are
diagnostic only; the report records that correction. All samples, including
regressions, are in its [machine record](notify_radix_20260917.json).

The [all-CTA worker report](full_cta_all_worker_20260914.md) records the unified
compute/communication layout, final FWD/BWD matrix, register diagnosis and
stability coverage. Its [machine record](full_cta_all_worker_20260914.json)
contains all timing samples and source/configuration evidence. Main FWD controls
are freshly paired; main BWD and Flex controls are explicitly historical.
BWD improves across the tested cases, while FWD still has regressions.

The preceding [1620 MHz GEMM report](gemm_optimization_20260914.md) records the
FC2 optimization, cuBLAS comparisons for FWD and BWD, and allocation-inclusive
FWD results against Flex. It includes [raw samples and validation records](gemm_optimization_20260914.json)
and uses the [2026-09-14 profiles](../configs/performance_20260914/profiles.json).
The earlier [Qwen diagnosis](qwen_forward_diagnosis_20260914.md) and
[Forward optimization report](forward_optimization_20260914.md) describe the
communication, scheduler, and epilogue work preceding that measurement. Their
clock conditions differ; use the GEMM report for the same-frequency cuBLAS
comparison.

The archived EP8/EP16 comparison is in
[the 2026-09-13 report](performance_20260913.md), with
[statistics and raw timing samples](performance_20260913.json) and
[exported FWD/BWD profiles](../configs/performance_20260913/profiles.json).
The report identifies its frozen source revision; use that revision's tools to
reproduce its historical runs. Current comparison manifests use schema 2 and
are measured independently of that archive.

- `megatron_dispatch_baseline.py` and `megatron_backward_baseline.py` measure
  Megatron/Flex FWD and cached-state BWD.
- `run_multinode_megatron_rdma.py` launches the multi-node IB/RoCE baseline;
  `run_multinode_megatron_socket.py` provides the Socket comparison and shared
  process management.
- `run_integrated_comparison.py` compares complete host-wall execution boundaries.
- `compare_forward_outputs.py`, `compare_backward_outputs.py`, and
  `compare_forward_ep_outputs.py` provide numerical and cross-EP checks.
- `forward_fp64_reference.py` is the independent numerical reference.
- `nccl_library.py` resolves the NCCL runtime. `dynamic_gate_fixture.json`
  supplies routing gates for reference checks.
- `notify_dispatch_bench.cpp` is built with
  `python3 build_native.py metadata-bench`.
- `run_gemm_diagnostics.py` builds `pure_forward_gemm.cu` or
  `pure_backward_gemm.cu` with an explicit tuning header. The forward benchmark
  extracts production WGMMA math and the FC2 epilogue; the backward benchmark
  calls the production GEMM routines. `gemm_benchmark.cuh` provides the shared
  cuBLAS comparison, full-output checks, CUDA Graph timing, and NVML sampling.
- `run_gemm_matrix.py` runs prebuilt diagnostics for the EP8/EP16 shapes. This
  measures isolated GEMMs with preallocated tensors; it observes GPU clocks
  and never changes them. `run_forward_gemm_diagnostics.py` retains the older
  forward build entry point.
- `forward_diagnostics.py` is enabled through the Flex baseline's
  `--forward-diagnostics-dir` option. It records communication-only and isolated
  expert measurements, stage events, HCA counters, and a Chrome trace after
  the formal timing has finished.

For a communication-only measurement of the production FWD protocol, build
`cuda-test` with `-DWEFTMOE_BENCH_TRANSPORT_ONLY=1` in `NVCCFLAGS`, alongside
the usual shape macros. Run the resulting binary with the existing FWD rank
launchers. This test specialization waits for Dispatch, publishes prefilled
Return rows, and runs the normal Return and Reduce workers. Output preparation
is outside the CUDA-event interval; Notify is reported separately. GEMMs are
skipped, so GEMM instruction counters and FLOPS derived from the model shape
are not meaningful in this mode. The flag is rejected by package builds.

Generated results belong in `results/`, which is ignored by Git.

`run_performance_matrix.py` builds and runs the Qwen3.5-35B-A3B and
DeepSeek-V4-Flash BF16 routed-expert shapes at EP8/EP16 and 8K/64K tokens **per
rank**. The matrix uses the ordinary SwiGLU expert body, without a router or
shared expert. This is a layer benchmark, not a complete model benchmark.

The allocation-inclusive mode requests fresh input, selected-route/gate,
output, saved-state, and gradient tensor ownership on every applicable call.
Input initialization uses timed device-to-device clones of identical fixtures.
The native benchmark uses a caller-owned caching allocator; Flex uses the
PyTorch caching allocator. Warmups populate both caches. Prior outputs are
released outside timing, and new outputs/checkpoints stay alive until the call
completes. Workspace, weights, peer mappings, GIN contexts, and DeepEP buffers
are reused. Native cache misses after warmup fail the benchmark. The primary
metric is host wall time, taking the maximum rank **for each iteration** before
computing statistics. Native Forward's CUDA interval excludes Notify and is
not directly comparable to Flex's complete Forward CUDA interval.
Both implementations synchronize ranks with an untimed NCCL collective before
each iteration. The matrix enables `WEFTMOE_BENCH_SYNC_RANKS=1` for native runs
and rejects records without that synchronization scope.

```sh
python3 benchmarks/run_performance_matrix.py build --output results/candidate
python3 benchmarks/run_performance_matrix.py weft --output results/candidate
python3 benchmarks/run_performance_matrix.py flex --output results/flex
```

Use `--remote`, `--port`, and `--socket-ifname` for the test hosts. The matrix
currently targets two nodes with eight Hopper GPUs each; the lower-level
autotuners retain their general model and topology interfaces. Build-time
`--override KEY=JSON` selects a candidate parameter. `--source` accepts a
frozen tree for an independently runnable baseline. `build-pair` with
`--forward-builds` and `--backward-builds` compiles both selected configurations
into one binary and exports a combined `.tuning.hpp` file. The exported header
also works with `NVCCFLAGS='-include /absolute/path/model.tuning.hpp'` when
running `build_native.py`.

`run_fair_comparison.py` interleaves frozen before/after binaries and a selected
Flex SM configuration, then reverses the implementation and case order.
It records raw samples, binary hashes, selected configurations, and execution
order. Finish compilation and all correctness jobs before running it.
`summarize_fair_comparison.py` pools every formal sample, preserves per-pass
means and P50s, and reports P50/mean ratios alongside P95 and dispersion.

Run correctness separately: `--check-repeats` compares every output bit,
including every dW element, after each warmup/measured call. `--dump-output`
writes the first `--dump-tokens` tokens and their inputs for
`compare_forward_ep_outputs.py`. `forward_dump_reference.py` evaluates selected
dump coordinates with NumPy FP64 dot products at arbitrary supported H/I/E
shapes. `run_backward_validation.py` checks native repeat stability and then
compares all gradients with PyTorch autograd, verifies retained X/A/B, and
fails on nonfinite values or excessive relative L2 error. Checks and dump
copies are outside the timed interval and are disabled in performance runs.
The native harness enables repeat checks whenever `WEFTMOE_BENCH_CHECK_REPEAT`
is present, including a value of `0`; unset it to disable checks. The matrix
runner verifies actual per-rank check counters for both new and reused records.
