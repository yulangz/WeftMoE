#pragma once

#include "cuda/persistent_v3/roles/common.cuh"

namespace megamoe::cuda::persistent_v3 {

#if defined(__CUDACC__)
template <std::uint32_t Capacity>
MEGAMOE_V3_DEVICE inline bool
rdma_publish_to_relay(SpscRing<PipelineTask, Capacity> &queue,
                      GenerationState &state, const PipelineTask &task) {
  while (!try_push(queue, task)) {
    if (generation_cancelled(state, task.epoch))
      return false;
    __nanosleep(64U);
  }
  return true;
}

template <std::uint32_t Capacity>
MEGAMOE_V3_DEVICE inline bool
rdma_consume_from_relay(SpscRing<PipelineTask, Capacity> &queue,
                        GenerationState &state, unsigned long long epoch,
                        PipelineTask &task) {
  while (!try_pop(queue, task)) {
    if (generation_cancelled(state, epoch))
      return false;
    __nanosleep(64U);
  }
  return true;
}
#endif

} // namespace megamoe::cuda::persistent_v3
