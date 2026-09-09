#pragma once

#include "cuda/persistent_v3/pipelines/task_scoreboard.cuh"

namespace weftmoe::cuda::persistent_v3 {

#if defined(__CUDACC__)
WEFTMOE_V3_DEVICE inline bool generation_cancelled(GenerationState &state,
                                                    unsigned long long epoch) {
  return load_acquire<::cuda::thread_scope_device>(
             state.control.cancel_epoch.value) >= epoch ||
         load_acquire<::cuda::thread_scope_device>(
             state.control.error_code.value) != 0ULL;
}

WEFTMOE_V3_DEVICE inline void publish_error_once(GenerationState &state,
                                                  unsigned long long code) {
  auto expected = 0ULL;
  (void)compare_exchange_acq_rel<::cuda::thread_scope_device>(
      state.control.error_code.value, expected, code);
}

WEFTMOE_V3_DEVICE inline void wait_for_terminal(GenerationState &state,
                                                 unsigned long long epoch) {
  while (load_acquire<::cuda::thread_scope_device>(
             state.control.terminal_epoch.value) < epoch) {
    __nanosleep(64U);
  }
}
#endif

} // namespace weftmoe::cuda::persistent_v3
