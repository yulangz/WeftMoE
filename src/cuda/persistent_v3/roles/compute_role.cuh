#pragma once

#include "cuda/persistent_v3/roles/common.cuh"

namespace megamoe::cuda::persistent_v3 {

#if defined(__CUDACC__)
MEGAMOE_V3_DEVICE inline std::uint32_t
try_claim_ready_task(ScoreboardTask *tasks, std::uint32_t task_count,
                     PaddedCounter &scan_cursor, unsigned long long epoch) {
  if (task_count == 0U)
    return kInvalidTask;
  const auto candidate = static_cast<std::uint32_t>(
      atomicAdd(&scan_cursor.value, 1ULL) % task_count);
  return try_claim(tasks[candidate], epoch) ? candidate : kInvalidTask;
}
#endif

} // namespace megamoe::cuda::persistent_v3
