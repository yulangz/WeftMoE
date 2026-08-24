# Checked-in benchmark evidence

These files are compact machine-readable extracts from the clean release runs
documented in `docs/build.md`; they are evidence, not hard-coded benchmark
expectations.

- `forward_release_world8.json` records the three independent MegaMOE and
  Megatron-Core 0.17.0 Forward sessions and the formal per-run comparison.
- `forward_main_world1_ncu.csv` records the available Nsight Compute metrics
  for the same main-shape kernel in world 1. Multi-process replay was not used:
  independently pausing ranks at a peer-visible terminal barrier is not a
  valid profiling configuration.

The profiler also emitted `ptxas` diagnostics that WGMMA accumulator
dependencies were serialized. Together with 18.75% achieved occupancy, this
is the current primary performance lead. No Nsight Systems executable was
available on the inspected host.
