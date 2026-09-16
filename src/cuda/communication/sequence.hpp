#pragma once
#include <cstdint>
#if defined(__CUDACC__)
#define WEFTMOE_SEQUENCE_HD __host__ __device__
#else
#define WEFTMOE_SEQUENCE_HD
#endif
namespace weftmoe::cuda::communication {
// Valid while outstanding distance is less than half the counter range.
WEFTMOE_SEQUENCE_HD constexpr bool reached(std::uint64_t value,
                                          std::uint64_t target,
                                          unsigned bits = 56U) {
  const auto mask = ~std::uint64_t{0} >> (64U - bits);
  return ((value - target) & mask) <= (mask >> 1U);
}
}
#undef WEFTMOE_SEQUENCE_HD
