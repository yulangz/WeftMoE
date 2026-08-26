#pragma once

#include <cstdint>

namespace megamoe::tests {

void run_runtime_shape_test(std::uint64_t hidden,
                            std::uint64_t intermediate,
                            bool run_backward);

} // namespace megamoe::tests
