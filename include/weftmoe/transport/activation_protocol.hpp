#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "weftmoe/metadata/dispatch_handle.hpp"

WEFTMOE_PUBLIC_API_BEGIN

namespace weftmoe::transport {

// Storage type used at the public boundary. It deliberately does not expose a
// CUDA-native half type, so host-only clients and public-header checks remain
// independent from the CUDA toolkit.
struct BFloat16 {
  std::uint16_t bits = 0;
};
static_assert(sizeof(BFloat16) == 2);
inline bool operator==(BFloat16 lhs, BFloat16 rhs) noexcept {
  return lhs.bits == rhs.bits;
}
inline bool operator!=(BFloat16 lhs, BFloat16 rhs) noexcept {
  return !(lhs == rhs);
}

float bfloat16_to_float(BFloat16 value) noexcept;
BFloat16 float_to_bfloat16(float value) noexcept;

} // namespace weftmoe::transport

WEFTMOE_PUBLIC_API_END
