#pragma once
#include <cstdint>

#if defined(__CUDACC__)
#define WEFTMOE_READY_HD __host__ __device__
#else
#define WEFTMOE_READY_HD
#endif

namespace weftmoe::cuda::communication {
// Each producer owns disjoint rows. Its release OR publishes precisely those
// rows, so out-of-order producers never block one another or expose a gap.
WEFTMOE_READY_HD constexpr std::uint32_t row_ready_mask(unsigned offset,
                                                       unsigned count) {
  return count == 32U ? UINT32_MAX : ((std::uint32_t{1} << count) - 1U) << offset;
}

WEFTMOE_READY_HD inline unsigned row_ready_prefix(std::uint32_t bitmap,
                                                  unsigned offset,
                                                  unsigned limit) {
  const auto missing = ~(bitmap >> offset);
#if defined(__CUDA_ARCH__)
  const unsigned count = missing ? unsigned(__ffs(int(missing)) - 1) : 32U;
#else
  const unsigned count = missing ? unsigned(__builtin_ctz(missing)) : 32U;
#endif
  return count < limit ? count : limit;
}
} // namespace weftmoe::cuda::communication
#undef WEFTMOE_READY_HD
