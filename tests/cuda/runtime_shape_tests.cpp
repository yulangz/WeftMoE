#include "cuda/test_entrypoints.hpp"

#include <iostream>
#include <stdexcept>

#include "cuda/testing/harnesses.hpp"

namespace megamoe::tests {

void run_runtime_shape_test(std::uint64_t hidden,
                            std::uint64_t intermediate,
                            bool run_backward) {
  if (!cuda::testing::run_runtime_shape_world1_harness(
          hidden, intermediate, run_backward)) {
    throw std::runtime_error(
        "runtime-shape persistent Forward/Backward harness failed");
  }
  std::cout << "[PASS] runtime-shape persistent kernel hidden=" << hidden
            << " intermediate=" << intermediate
            << " backward=" << run_backward << '\n';
}

} // namespace megamoe::tests
