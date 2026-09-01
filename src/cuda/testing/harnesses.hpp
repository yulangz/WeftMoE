#pragma once

#include <cstdint>

namespace megamoe::cuda::testing {

// Internal test entry points. These launch the production kernels with a
// controlled NCCL-free fixture; they are deliberately absent from the public
// CUDA API.
bool run_forward_world1_sanitizer_harness();
bool run_backward_world1_sanitizer_harness();
bool run_runtime_shape_world1_harness(std::uint64_t hidden,
                                      std::uint64_t intermediate,
                                      bool run_backward);

struct BackwardDw1CoreProfile {
  std::uint32_t sm_count = 0;
  std::uint32_t experts = 0;
  std::uint32_t rows_per_expert = 0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double tflops = 0.0;
};

// Runs the production V3 dW1 tile loop without transport or activation
// scheduling, using the main two-node per-rank grouped shape. This keeps NCU
// collection independent of communicator/bootstrap timing.
BackwardDw1CoreProfile profile_backward_v3_dw1_core(
    std::uint32_t warmup, std::uint32_t measured, bool k_major_x,
    bool cluster2, bool independent_tasks);

struct BackwardActivationCoreProfile {
  std::uint32_t sm_count = 0;
  std::uint32_t tasks_per_cta = 0;
  std::uint64_t tasks = 0;
  std::uint64_t mismatches = 0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double max_ms = 0.0;
  double tflops = 0.0;
};

// Runs the production dV 128x256x64, four-stage TMA/WGMMA mainloop on
// deterministic inputs.  The fixture deliberately omits communication,
// nonlinear work, and the production TMA epilogue so persistent pipeline
// reuse can be tested independently.
BackwardActivationCoreProfile profile_backward_v3_dv_core(
    std::uint32_t warmup, std::uint32_t measured,
    std::uint32_t tasks_per_cta);

} // namespace megamoe::cuda::testing
