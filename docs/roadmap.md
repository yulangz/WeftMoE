# Roadmap

## Forward stabilization

- Keep world-8 and world-16 correctness/performance gates on the same split
  implementation.
- Tune only parameters with measured sensitivity; retain singleton defaults
  for fixed hardware/protocol constraints.
- Extend timeline coverage for dispatch-to-first-FC1 and last-FC2-to-terminal
  tails without perturbing release kernels.
- Validate three- and four-node GIN progress before generalizing topology.

## Backward redesign

Backward will not reuse the deleted monolithic implementation. Its new design
must start from the Forward invariants:

- fixed communication, relay, and compute ownership;
- generation-tagged counters and bounded queues instead of phase-wide
  `grid.sync()`;
- M blocks padded to 128 and compile-time TMA/WGMMA specializations;
- communication/computation overlap only where it shortens the critical path;
- independently optimized dV/dW compute kernels with no artificial folding
  requirement when all SMs are already busy;
- explicit JSON autotuning and the same slowest-rank correctness/performance
  evidence used by Forward.

The public availability query remains false until this path has end-to-end
correctness, lifecycle, and performance validation.
