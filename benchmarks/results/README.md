# Checked-in benchmark evidence

These files are compact machine-readable extracts from the clean release runs
documented in `docs/build.md`; they are evidence, not hard-coded benchmark
expectations.

- `forward_release_world8.json` records the three independent MegaMOE and
  Megatron-Core 0.17.0 Forward sessions and the formal per-run comparison,
  exact cross-implementation fixture hashes, cache/no-cache workspace and
  timing, the independent NumPy FP64 oracle, runtime cluster audit, and the
  remote clean-export reproduction.
- `backward_integrated_world8.json` records three true same-generation
  Forward-to-Backward event-handoff sessions and the native Megatron
  Backward/F+B comparison.
- `gin_world16_training_20260826.json` records three two-node world-16 GIN
  Forward and F+B sessions, exact fixture agreement, runtime put/signal/byte
  counters, and the per-run Megatron Socket comparison gate.
- `socket_world16_fallback_20260826.json` records the corresponding three-run
  full-FC NCCL Socket correctness/performance lower bound. Its explicit host
  payload phases are included and every GIN audit field is zero.
- `forward_main_world1_ncu.csv` records the available Nsight Compute metrics
  for the same main-shape kernel in world 1. Multi-process replay was not used
  because pausing one process independently invalidates the live peer data-plane
  timing and progress assumptions.

The optimized capture improves kernel duration from 25.349 to 19.432 ms, SM
throughput from 42.37% to 55.97%, and DRAM throughput from 48.87% to 66.55%.
The profiler still emits `ptxas` WGMMA dependency diagnostics; barrier and
long-scoreboard stalls are now the primary remaining leads. No Nsight Systems
executable was available on the inspected host.
