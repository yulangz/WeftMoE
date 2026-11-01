#pragma once

#include "cuda/fp8/api.hpp"
#include "cuda/fp8/quantization.cuh"

namespace weftmoe::cuda::fp8 {

// Each source token owns one packet. Destination Pull converts only its layout;
// it never recalculates scales. Packet pitch fits the existing BF16 source pool.
// Forward X and Backward dY use the same [tokens, Hidden] packet format.
__host__ __device__ constexpr unsigned source_pitch(unsigned hidden) {
  return (hidden + hidden / 128U * sizeof(float) + 15U) & ~15U;
}

// Preserve caller quantization exactly: no amax, scale arithmetic or cast.
// Each warp copies one row into the packet layout used by local/GIN Pull.
template <unsigned Hidden>
__global__ __launch_bounds__(128) void copy_source_x(
    const std::uint8_t *input, const float *input_scales,
    std::uint8_t *output, std::uint64_t tokens) {
  constexpr unsigned Pitch = source_pitch(Hidden);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  for (auto row = std::uint64_t(blockIdx.x) * 4U + warp; row < tokens;
       row += std::uint64_t(gridDim.x) * 4U) {
    auto *packet = output + row * Pitch;
    for (unsigned col = lane * 16U; col < Hidden; col += 512U)
      *reinterpret_cast<uint4 *>(packet + col) =
          *reinterpret_cast<const uint4 *>(input + row * Hidden + col);
    auto *scales = reinterpret_cast<float *>(packet + Hidden);
    for (unsigned col = lane; col < Hidden / 128U; col += 32U)
      scales[col] = input_scales[row * (Hidden / 128U) + col];
  }
}

template <unsigned Hidden>
__global__ __launch_bounds__(128) void prepare_source_x(
    const __nv_bfloat16 *input, std::uint8_t *output, std::uint64_t tokens) {
  constexpr unsigned Pitch = source_pitch(Hidden);
  const auto warp = threadIdx.x / 32U, lane = threadIdx.x % 32U;
  for (auto row = std::uint64_t(blockIdx.x) * 4U + warp; row < tokens;
       row += std::uint64_t(gridDim.x) * 4U) {
    auto *packet = output + row * Pitch;
    auto *scales = reinterpret_cast<float *>(packet + Hidden);
    for (unsigned column = lane * 8U; column < Hidden; column += 256U) {
      // A partial last K256 group still uses complete 16-lane K128 groups.
      const auto value = *reinterpret_cast<const uint4 *>(input + row * Hidden + column);
      store_fp8_eight(value, packet, Hidden, column, scales, 1U);
    }
  }
}

} // namespace weftmoe::cuda::fp8
