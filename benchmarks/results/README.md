# Benchmark results

Only results produced from the current Forward-only implementation belong in
this directory. Historical single-node fallback, cache, and Backward records
were removed because their kernels and measurement boundaries no longer exist.

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
  performance regression evidence.

Full build and rank logs remain ignored artifacts and are not
source-controlled.
