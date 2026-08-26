# Known limitations

- The optimized WGMMA/TMA specialization is only for hidden 7168 and
  intermediate 3072. Other positive sizes use a correct one-kernel CUDA-core
  fallback and are not performance tuned. That fallback is currently wired for
  world 1 and LocalCudaP2p; non-main-shape GIN execution is rejected rather
  than silently selecting an incorrect transport path.
- Nodes must be homogeneous, node IDs dense, and every node must contain the
  complete local-rank set. Missing rails and heterogeneous node sizes are
  rejected.
- Physical GIN execution has been validated on two nodes (world 2 and 16).
  Three- and four-node topology/team construction is tested without a third
  physical host; physical 3+ node progress and performance remain unverified.
- The available runtime hosts report NVIDIA L20X and the documented virtual
  compute-capability behavior. SM90a fatbin compilation is checked, but this is
  not an H200 hardware performance or functional qualification.
- GIN requires a repository-external NCCL 2.29.7 CUDA-12.9 runtime and the
  matching network/plugin/registration environment. No third-party binary is
  committed. Missing prerequisites produce `BackendUnavailable`.
- `NcclSocketFallback` is host-enqueued NCCL Send/Recv with explicit payload
  phase boundaries. It is not GIN/RDMA and not a single persistent cross-host
  data plane.
- Compute Sanitizer cannot reliably instrument every NCCL/GIN internal kernel
  in the measured runtime. NCCL-free production-kernel harnesses validate the
  MegaMOE Forward/Backward kernels; backend lifecycle and real GIN are tested
  separately.
- Installed CUDA headers still contain legacy explicitly named `testing`,
  synthetic transport and `*_for_validation` entry points. Newly added
  runtime-shape harnesses are internal, and native backend handles remain
  hidden, but fully removing the older test/oracle surface is a pending public
  API compatibility cleanup.
- Heterogeneous dtype, FP8, alternative gate placement, cross-node topology
  heterogeneity, graph capture and automatic tuning of arbitrary H/I shapes are
  outside the frozen semantic contract.
