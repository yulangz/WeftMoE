# Checked-in benchmark evidence

These files are compact machine-readable extracts from the clean release runs
documented in `docs/build.md`; they are evidence, not hard-coded benchmark
expectations.

- `forward_release_world8.json` records the three independent MegaMOE and
  Megatron-Core 0.17.0 Forward sessions and the formal per-run comparison,
  plus the independent remote clean-export MegaMOE reproduction.
- `forward_main_world1_ncu.csv` records the available Nsight Compute metrics
  for the same main-shape kernel in world 1. Multi-process replay was not used
  because pausing one process independently invalidates the live peer data-plane
  timing and progress assumptions.

The optimized capture improves kernel duration from 25.349 to 19.432 ms, SM
throughput from 42.37% to 55.97%, and DRAM throughput from 48.87% to 66.55%.
The profiler still emits `ptxas` WGMMA dependency diagnostics; barrier and
long-scoreboard stalls are now the primary remaining leads. No Nsight Systems
executable was available on the inspected host.
