# Roadmap

## Forward stabilization

- Keep world-8 and world-16 correctness/performance gates on the same split
  implementation.
- Tune only parameters with measured sensitivity; retain singleton defaults
  for fixed hardware/protocol constraints.
- Extend timeline coverage for dispatch-to-first-FC1 and last-FC2-to-terminal
  tails without perturbing release kernels.
- Validate three- and four-node GIN progress before generalizing topology.

## Backward maintenance

The replacement cached-state Backward pipeline is implemented and qualified;
its selected contract is specified in
[backward_design.md](backward_design.md). Continue to preserve these Forward
invariants:

- fixed communication, relay, and compute ownership;
- generation-tagged counters and fixed task tables instead of phase-wide
  `grid.sync()`;
- M blocks padded to 128 and compile-time TMA/WGMMA specializations;
- communication/computation overlap only where it shortens the critical path;
- activation-priority multi-queue scheduling where otherwise-idle compute CTAs
  may claim `dW1/dW2` without waiting for all `dX` to be sent;
- explicit JSON autotuning and the same slowest-rank correctness/performance
  evidence used by Forward.
- keep the CPU-only weak availability query false and the CUDA-backed strong
  query true, so a binary cannot advertise a data plane it does not contain.
