#pragma once

#include "cuda/fp8/tile.cuh"

#ifndef WEFTMOE_FP8_STREAMED_WGRAD
#define WEFTMOE_FP8_STREAMED_WGRAD 1
#endif

namespace weftmoe::cuda::fp8 {

// BF16 magnitudes can be reduced as two packed unsigned integers. This
// avoids decoding all 128 values merely to find their column maxima.
__device__ __forceinline__ void column_cast_bf16(
    const unsigned (&values)[32], std::uint8_t *output, float *scales) {
  const unsigned lane = threadIdx.x % 32U;
  const unsigned channel = threadIdx.x / 32U * 16U + lane % 8U * 2U;
  unsigned maxima = 0;
#pragma unroll
  for (unsigned i = 0; i < 32U; ++i) {
    // Positive BF16 bit patterns are ordered by magnitude. Reduce both
    // channels in packed integer form before converting only the maxima.
    maxima = __vmaxu2(maxima, values[i] & 0x7fff7fffU);
  }
#pragma unroll
  for (unsigned delta = 8U; delta <= 16U; delta *= 2U)
    maxima = __vmaxu2(maxima, __shfl_xor_sync(0xffffffffU, maxima, delta));
  const auto amax = __uint_as_float(maxima << 16U);
  const auto bmax = __uint_as_float(maxima & 0xffff0000U);
  const unsigned invalid = unsigned((maxima & 0xffffU) >= 0x7f80U) |
      (unsigned((maxima >> 16U) >= 0x7f80U) << 1U);
  const auto sa = invalid & 1U ? nanf("") : block_scale(amax);
  const auto sb = invalid & 2U ? nanf("") : block_scale(bmax);
  const auto ia = invalid & 1U ? sa : __uint_as_float(0x7f000000U - __float_as_uint(sa));
  const auto ib = invalid & 2U ? sb : __uint_as_float(0x7f000000U - __float_as_uint(sb));
  if (lane < 8U) {
    scales[channel] = sa;
    scales[channel + 1U] = sb;
  }
#pragma unroll
  for (unsigned part = 0; part < 2U; ++part) {
    unsigned aq[4], bq[4];
#pragma unroll
    for (unsigned group = 0; group < 4U; ++group) {
      unsigned q[4];
#pragma unroll
      for (unsigned j = 0; j < 4U; ++j) {
        const auto bits = values[part * 16U + group * 4U + j];
        const auto value = make_float2(__uint_as_float(bits << 16U),
                                       __uint_as_float(bits & 0xffff0000U));
        q[j] = __nv_fp8x2_e4m3(make_float2(value.x * ia, value.y * ib)).__x;
      }
      const auto lo = q[0] | (q[1] << 16U), hi = q[2] | (q[3] << 16U);
      aq[group] = __byte_perm(lo, hi, 0x6420U);
      bq[group] = __byte_perm(lo, hi, 0x7531U);
    }
    *reinterpret_cast<uint4 *>(output + tile_offset(channel, lane / 8U * 32U + part * 16U)) =
        make_uint4(aq[0], aq[1], aq[2], aq[3]);
    *reinterpret_cast<uint4 *>(output + tile_offset(channel + 1U, lane / 8U * 32U + part * 16U)) =
        make_uint4(bq[0], bq[1], bq[2], bq[3]);
  }
}

__device__ __forceinline__ void row_to_column(
    const std::uint8_t *tile, const float *row_scales, unsigned rows,
    std::uint8_t *output, float *scales) {
  const unsigned lane = threadIdx.x % 32U;
  const unsigned column = threadIdx.x / 32U * 16U + lane % 8U * 2U;
  // The source was cast from BF16 with power-of-two row scales. Its decoded
  // finite values fit BF16 exactly. Keep 32 packed pairs, avoiding a second
  // shared-memory read and two scalar FP8 conversions per element.
  unsigned values[32];
#pragma unroll
  for (unsigned i = 0; i < 32U; ++i) {
    const auto row = lane / 8U * 32U + i;
    unsigned value = 0U;
    if (row < rows) {
      __nv_fp8x2_e4m3 q;
      q.__x = *reinterpret_cast<const unsigned short *>(tile + tile_offset(row, column));
      const auto decoded = static_cast<float2>(q);
      const auto scale = row_scales[row];
      const auto pair = __float22bfloat162_rn(make_float2(decoded.x * scale, decoded.y * scale));
      value = reinterpret_cast<const unsigned &>(pair);
    }
    values[i] = value;
  }
  column_cast_bf16(values, output, scales);
}

} // namespace weftmoe::cuda::fp8
