# Known limitations

- The optimized kernel accepts only hidden 7168 and intermediate 3072. The
  removed generic-shape path is not a supported fallback.
- Backward requires retained Forward X and BF16 A/B state and supports only the
  maintained cached-state shape and one-/two-node split-role topologies. It has
  no generic-shape or recompute-FC1 fallback.
- Production topology requires homogeneous nodes, dense node/local-rank IDs,
  and complete rails. Physical GIN testing currently covers one and two nodes.
- The target hosts contain eight H200 GPUs but the installed driver reports
  `NVIDIA L20X`; builds explicitly target SM90a.
- GIN depends on the external NCCL 2.29.7 CUDA 12.9 runtime and its matching
  network/plugin environment. Missing device symbols or registration support
  produce `BackendUnavailable`.
- NCCL Socket is a host-enqueued correctness backend with explicit payload
  phase boundaries. Its latency is not used as a device-overlap claim.
- BF16 is the only data type. FP8, heterogeneous types, alternate gate
  placement, sparse/heterogeneous topology, and arbitrary-shape tuning are out
  of scope.
- Compute Sanitizer cannot reliably freeze one process in a live GIN pipeline.
  NCCL-free kernel tests and separate real-network smoke tests cover those
  boundaries instead.
