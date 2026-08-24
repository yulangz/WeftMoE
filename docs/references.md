# Reference implementation survey

The implementation was written from the MegaMOE design contract and inspected
the following files for design patterns. No source was mechanically copied.

## DeepEP

- `DeepEP/csrc/kernels/legacy/ibgda_device.cuh`: device-side WQE reservation,
  ordered doorbell publication, local quiet/fence, and remote atomic/signal
  mechanics. MegaMOE uses the ownership and completion distinctions, not the
  NVSHMEM-private implementation.
- `DeepEP/csrc/kernels/legacy/internode.cu` and `intranode.cu`: channel
  partitioning, same-local-rank scale-out, node-local forwarding, signal
  barriers, and bounded buffer reuse.
- `DeepEP/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh` and
  `hybrid_combine.cuh`: same-rail proxy/fan-out and reverse metadata flow.

- `DeepEP/csrc/kernels/backend/nccl.cu`: CUDA/NCCL compilation boundaries,
  communicator calls, stream enqueue style, and contextual backend checks. It
  informed the private backend boundary, not the generic public types.
- `DeepEP/deep_ep/include/deep_ep/common/exception.cuh`: CUDA/NCCL launch and
  runtime error checking patterns; MegaMOE maps errors into its own typed,
  phase/rank/session-aware exceptions.

- `DeepEP/deep_ep/include/deep_ep/impls/dispatch.cuh`: notify warp roles,
  rank/expert counting, peer count writes, count readiness, expert/rank prefix,
  and the existing atomic sender slot allocation that MegaMOE must not use for
  canonical expert rows.
- `DeepEP/deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh`: split scale-out /
  scale-up notify and forwarding structure for a future same-rail transport.
- `DeepEP/deep_ep/include/deep_ep/impls/combine.cuh` and
  `hybrid_combine.cuh`: source metadata carried back through combine.
- `DeepEP/deep_ep/include/deep_ep/common/layout.cuh`,
  `DeepEP/deep_ep/include/deep_ep/common/comm.cuh`, and
  `DeepEP/deep_ep/include/deep_ep/common/handle.cuh`: explicit workspace
  regions, signals, barriers, and GIN handle separation.
- `DeepEP/deep_ep/buffers/elastic.py`: reusable `EPHandle` and deterministic
  post-sort behavior. MegaMOE instead precomputes the canonical row before data
  movement.

## DeepGEMM

- `DeepGEMM/CMakeLists.txt` and
  `DeepGEMM/deep_gemm/include/deep_gemm/common/exception.cuh`: optional CUDA
  target layout, architecture selection, and checked launch boundaries.

- `DeepGEMM/deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`: explicit
  `BlockPhase`, persistent task production, L1 warmup/L2 interleave, and runtime
  valid-M propagation.
- `DeepGEMM/deep_gemm/include/deep_gemm/layout/mega_moe.cuh`: typed workspace
  regions, hot-counter isolation, token source metadata, and ring/full counters.
- `DeepGEMM/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh`:
  dispatch/compute/combine roles in one persistent kernel and acquire/release
  producer-consumer waits.

## DeepGEMM-bk

- `DeepGEMM-bk/CMakeLists.txt`: separate compile targets for experimental SM90
  paths and the need to distinguish compile coverage from runnable hardware.

- `DeepGEMM-bk/deep_gemm/include/deep_gemm/scheduler/sm90_mega_moe.cuh`:
  SM90-specific phase scheduling, logical pool versus physical ring, and
  generation semantics preventing ABA.
- `DeepGEMM-bk/deep_gemm/include/deep_gemm/impls/sm90_bf16_mega_moe.cuh`:
  SM90 dispatch/math role split, M-tail handling, full/empty counters, and
  workspace cleanup boundaries.

The backup tree contains experimental/local modifications; it was treated only
as an architecture reference, not as an upstream API or code source.

The Phase-4 survey additionally used
`DeepGEMM-bk/deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh`
for persistent dispatch/epilogue/combine role separation, logical-pool versus
physical-ring indexing, backpressure, and phase-tagged barriers.

## Triton-distributed

- `Triton-distributed/tutorials/03-inter-node-allgather.py`: hierarchical
  same-local-rank inter-node put followed by intra-node fan-out and signal
  waits.
- `Triton-distributed/tutorials/07-overlapping-allgather-gemm.py`: device
  put-with-signal producer semantics and a separately named copy-engine
  fallback. It reinforced the need to label backend mechanisms precisely.

- `Triton-distributed/tutorials/04-deepseek-infer-all2all.py`: expert split
  cumsums, explicit dispatch/combine offsets, put/fence/signal ordering, and
  stable gather-index construction.
- `Triton-distributed/python/triton_dist/kernels/nvidia/moe_utils.py`: stable
  gather/scatter helpers, histogram/scan/sort workspace shapes, explicit
  indices, padding, and a contrasting atomic scatter allocator whose return
  order is unsuitable for MegaMOE canonical destination rows. The MegaMOE CUDA
  key includes the flat route index to impose a complete order.
- `Triton-distributed/csrc/lib/moe_utils.cu`: deterministic per-thread shard
  prefix construction and expert-aligned token layout.
- `Triton-distributed/tutorials/01-distributed-notify-wait.py`: sequence-aware
  notify/wait concepts for future transport work.
