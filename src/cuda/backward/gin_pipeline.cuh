#pragma once

// Backward uses the same fixed wire rings as Forward. The retained Notify
// handle supplies routes; no data or metadata must survive in the GIN window.
#if __has_include(<nccl_device.h>)
namespace activation_ring_detail {
template <>
__device__ __forceinline__ ncclGin gin_for_cta(const BackwardKernelArgs &a,
                                               unsigned cta) {
  // Each unified CTA has one submitter. Context zero belongs to metadata;
  // use exclusive queues only when the actual allocation cannot alias CTAs.
  // GinExecution orders invocations that reuse this context pool.
  const bool exclusive =
      a.gin_total_ctas != 0U &&
      unsigned(a.gin_dev_comm.ginContextCount) > a.gin_total_ctas;
  return ncclGin(a.gin_dev_comm, context_for_cta(a, cta),
                 exclusive ? NCCL_GIN_RESOURCE_SHARING_THREAD
                           : NCCL_GIN_RESOURCE_SHARING_GPU);
}
} // namespace activation_ring_detail

#endif
