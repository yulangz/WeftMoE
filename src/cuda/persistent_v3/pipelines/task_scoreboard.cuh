#pragma once

#include <cstdint>

#include "cuda/persistent_v3/queues.cuh"

namespace weftmoe::cuda::persistent_v3 {

enum class TaskStage : unsigned long long {
  Empty = 0ULL,
  Ready = 1ULL,
  Claimed = 2ULL,
  Complete = 3ULL,
};

WEFTMOE_V3_HOST_DEVICE constexpr unsigned long long
encode_task_state(unsigned long long epoch, TaskStage stage) {
  return (epoch << 2U) | static_cast<unsigned long long>(stage);
}

struct alignas(32) ScoreboardTask {
  unsigned long long state = 0;
  unsigned long long input = 0;
  unsigned long long output = 0;
  std::uint32_t context = 0;
  std::uint32_t reserved = 0;
};

static_assert(sizeof(ScoreboardTask) == 32U);

#if defined(__CUDACC__)
WEFTMOE_V3_DEVICE inline void publish_ready(ScoreboardTask &task,
                                             unsigned long long epoch) {
  store_release<::cuda::thread_scope_device>(
      task.state, encode_task_state(epoch, TaskStage::Ready));
}

WEFTMOE_V3_DEVICE inline bool try_claim(ScoreboardTask &task,
                                        unsigned long long epoch) {
  auto expected = encode_task_state(epoch, TaskStage::Ready);
  return compare_exchange_acq_rel<::cuda::thread_scope_device>(
      task.state, expected, encode_task_state(epoch, TaskStage::Claimed));
}

WEFTMOE_V3_DEVICE inline void publish_complete(ScoreboardTask &task,
                                                unsigned long long epoch) {
  store_release<::cuda::thread_scope_device>(
      task.state, encode_task_state(epoch, TaskStage::Complete));
}

WEFTMOE_V3_DEVICE inline bool task_complete(ScoreboardTask &task,
                                             unsigned long long epoch) {
  return load_acquire<::cuda::thread_scope_device>(task.state) ==
         encode_task_state(epoch, TaskStage::Complete);
}
#endif

} // namespace weftmoe::cuda::persistent_v3
