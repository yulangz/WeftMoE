#pragma once

#include <cuda.h>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace weftmoe::cuda::internal {

// Expandable allocator segments can back one tensor with several adjacent
// cuMemMap regions. Validate the mapped span, not only its first region or the
// larger (possibly partly unmapped) cuMemAddressReserve reservation.
inline bool mapped_device_range(const void *pointer, std::size_t bytes) {
  auto cursor = reinterpret_cast<CUdeviceptr>(pointer);
  if (bytes > std::numeric_limits<CUdeviceptr>::max() - cursor)
    return false;
  while (bytes) {
    CUdeviceptr base = 0;
    std::size_t size = 0;
    if (cuMemGetAddressRange(&base, &size, cursor) != CUDA_SUCCESS ||
        cursor < base || cursor - base >= size)
      return false;
    const auto chunk = std::min(bytes, size - std::size_t(cursor - base));
    cursor += chunk;
    bytes -= chunk;
  }
  return true;
}

} // namespace weftmoe::cuda::internal
