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

} // namespace megamoe::cuda::testing
