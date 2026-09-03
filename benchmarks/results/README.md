# Benchmark results

Only results produced from current code and active measurement boundaries
belong in this directory. Historical single-node fallback, cache, and retired
Backward records were removed because their kernels no longer exist. A new
Backward record must identify whether Forward setup and retained-state creation
are inside or outside its timed boundary.

Use `tools/autotune_forward.py` for tuning and the Forward benchmark scenarios
in `tests/run_local_activation_p2p.py` or `tests/run_multinode_nccl.py` for
performance runs. Keep checked-in records compact and include the commit,
topology, tuning configuration, warmup/sample counts, correctness hashes, and
slowest-rank latency statistics.

The current records are:

- [`forward_cleanup_20260902.json`](forward_cleanup_20260902.json), the compact
  post-cleanup kernel baseline;
- [`forward_python_jit_20260903.json`](forward_python_jit_20260903.json), the
  cold/warm Python-JIT timings, noise-resistant selection, and final world-8
  and world-16 validation;
- [`forward_gate_before_20260903.json`](forward_gate_before_20260903.json), the
  gate-before-FC2 semantic change, independent oracle, and world-8/world-16
  performance regression evidence;
- [`forward_bf16_checkpoint_20260903.json`](forward_bf16_checkpoint_20260903.json),
  the BF16 FC1 checkpoint contract, retained-state correctness, compile-time
  inference/training split, and paired world-8/world-16 regression evidence;
- [`backward_megatron_cache_x_ab_20260903.json`](backward_megatron_cache_x_ab_20260903.json),
  the pre-implementation Megatron BWD target using selective SwiGLU
  recomputation from cached X and BF16 A/B;
- [`backward_local_scaffold_20260903.json`](backward_local_scaffold_20260903.json),
  the first local cached-state BWD implementation milestone and its independent
  PyTorch-autograd accuracy evidence. This is a correctness scaffold, not the
  persistent performance-qualified pipeline.

Full build and rank logs remain ignored artifacts and are not
source-controlled.
