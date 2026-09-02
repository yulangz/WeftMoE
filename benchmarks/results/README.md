# Benchmark results

Only results produced from the current Forward-only implementation belong in
this directory. Historical single-node fallback, cache, and Backward records
were removed because their kernels and measurement boundaries no longer exist.

Use `tools/autotune_forward.py` for tuning and the Forward benchmark scenarios
in `tests/run_local_activation_p2p.py` or `tests/run_multinode_nccl.py` for
performance runs. Keep checked-in records compact and include the commit,
topology, tuning configuration, warmup/sample counts, correctness hashes, and
slowest-rank latency statistics.
