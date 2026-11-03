#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace weftmoe::cuda::fp8 {

// Keep each M128 slot independent: compact data first, then [K/128, M128]
// scales. storage_stride is a byte pitch; dR/dAB share the larger slot.
template <class Byte>
__device__ __forceinline__ Byte *ring_row(
    Byte *base, unsigned row, unsigned columns, unsigned storage_stride) {
  return base + std::uint64_t(row / 128U) * 128U * storage_stride +
      (row % 128U) * columns;
}

__device__ __forceinline__ float *ring_scales(
    std::uint8_t *base, unsigned row, unsigned columns, unsigned storage_stride) {
  return reinterpret_cast<float *>(ring_row(base, row / 128U * 128U,
      columns, storage_stride) + 128U * columns) + row % 128U;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (unsigned delta = 16; delta; delta >>= 1)
    value = fmaxf(value, __shfl_xor_sync(0xffffffffU, value, delta));
  return value;
}

// Round the dequantization multiplier upward to a power of two. Integer
// exponent extraction avoids libdevice frexp/ldexp and underflow. Nonfinite inputs retain
// a NaN scale, rather than silently becoming finite after saturation.
__device__ __forceinline__ float block_scale(float amax) {
  if (!isfinite(amax))
    return nanf("");
  if (amax == 0.0F)
    return 1.0F;
  const auto bits = __float_as_uint(amax);
  const int exponent = int(bits >> 23U) - 8 + ((bits & 0x7fffffU) > 0x600000U);
  return __uint_as_float(unsigned(max(exponent, 1)) << 23U);
}

// Sixteen lanes own a K128 block, with eight consecutive BF16 values per lane.
__device__ __forceinline__ void store_fp8_eight(
    uint4 value, std::uint8_t *base, unsigned columns, unsigned column,
    float *scales = nullptr, unsigned scale_stride = 1U) {
  // All callers own complete K128 groups (16 lanes), even when the other
  // half-warp has no tail block. Independent scheduling cannot change the
  // logical membership of this reduction.
  const unsigned mask = 0xffffU << (threadIdx.x & 16U);
  const unsigned words[4] = {value.x, value.y, value.z, value.w};
  float2 values[4];
  float maximum = 0;
  bool invalid = false;
#pragma unroll
  for (unsigned j = 0; j < 4U; ++j) {
    const __nv_bfloat162_raw raw{static_cast<unsigned short>(words[j]),
        static_cast<unsigned short>(words[j] >> 16U)};
    values[j] = __bfloat1622float2(raw);
    maximum = fmaxf(maximum, fmaxf(fabsf(values[j].x), fabsf(values[j].y)));
    invalid |= !isfinite(values[j].x) || !isfinite(values[j].y);
  }
#pragma unroll
  for (unsigned delta = 8; delta; delta >>= 1U) {
    maximum = fmaxf(maximum, __shfl_xor_sync(mask, maximum, delta, 16));
    invalid |= __shfl_xor_sync(mask, unsigned(invalid), delta, 16);
  }
  const auto scale = invalid ? nanf("") : block_scale(maximum);
  const auto inverse = invalid ? scale : __uint_as_float(0x7f000000U - __float_as_uint(scale));
  unsigned short q[4];
#pragma unroll
  for (unsigned j = 0; j < 4U; ++j)
    q[j] = __nv_fp8x2_e4m3(make_float2(values[j].x * inverse, values[j].y * inverse)).__x;
  *reinterpret_cast<uint2 *>(base + column) =
      make_uint2(unsigned(q[0]) | (unsigned(q[1]) << 16U), unsigned(q[2]) | (unsigned(q[3]) << 16U));
  if (!(threadIdx.x % 16U))
    (scales ? scales : reinterpret_cast<float *>(base + columns))[
        column / 128U * scale_stride] = scale;
}

template <class T, unsigned BlockRows, bool Transpose>
__global__ void quantize_kernel(const T *input, std::uint8_t *output,
                               float *scales, unsigned rows,
                               unsigned columns) {
  constexpr unsigned Threads = 256;
  const unsigned out_rows = Transpose ? columns : rows;
  const unsigned out_cols = Transpose ? rows : columns;
  const unsigned k_blocks = (out_cols + 127U) / 128U;
  const unsigned row_blocks = (out_rows + BlockRows - 1U) / BlockRows;
  const unsigned block_row = blockIdx.x / k_blocks;
  const unsigned block_col = blockIdx.x % k_blocks;
  const auto in_base = std::uint64_t(blockIdx.y) * rows * columns;
  const auto out_base = std::uint64_t(blockIdx.y) * out_rows * k_blocks * 128U;
  float values[(BlockRows * 128U + Threads - 1U) / Threads];
  float amax = 0.0F;
  unsigned invalid = 0;
#pragma unroll
  for (unsigned i = 0; i < (BlockRows * 128U + Threads - 1U) / Threads; ++i) {
    const unsigned index = threadIdx.x + i * Threads;
    const unsigned r = block_row * BlockRows + index / 128U;
    const unsigned c = block_col * 128U + index % 128U;
    float value = 0.0F;
    if (index < BlockRows * 128U && r < out_rows && c < out_cols) {
      const auto offset = Transpose ? std::uint64_t(c) * columns + r
                                    : std::uint64_t(r) * columns + c;
      value = static_cast<float>(input[in_base + offset]);
    }
    values[i] = value;
    invalid |= !isfinite(value);
    amax = fmaxf(amax, fabsf(value));
  }
  __shared__ float maxima[8];
  __shared__ float scale;
  const auto invalid_block = __syncthreads_or(invalid != 0U);
  amax = warp_max(amax);
  if (threadIdx.x % 32U == 0U)
    maxima[threadIdx.x / 32U] = amax;
  __syncthreads();
  if (threadIdx.x < 32U) {
    amax = warp_max(threadIdx.x < 8U ? maxima[threadIdx.x] : 0.0F);
    if (threadIdx.x == 0U) {
      scale = invalid_block ? nanf("") : block_scale(amax);
      scales[(std::uint64_t(blockIdx.y) * row_blocks + block_row) * k_blocks +
             block_col] = scale;
    }
  }
  __syncthreads();
#pragma unroll
  for (unsigned i = 0; i < (BlockRows * 128U + Threads - 1U) / Threads; ++i) {
    const unsigned index = threadIdx.x + i * Threads;
    const unsigned r = block_row * BlockRows + index / 128U;
    const unsigned c = block_col * 128U + index % 128U;
    if (index < BlockRows * 128U && r < out_rows) {
      output[out_base + std::uint64_t(r) * k_blocks * 128U + c] =
          c < out_cols ? __nv_fp8_e4m3(values[i] / scale).__x : 0;
    }
  }
}

__device__ __forceinline__ float4 load_weight_four(const float *input) {
  return *reinterpret_cast<const float4 *>(input);
}

__device__ __forceinline__ float4 load_weight_four(const __nv_bfloat16 *input) {
  const auto bits = *reinterpret_cast<const uint2 *>(input);
  const auto lo = __bfloat1622float2(__nv_bfloat162_raw{
      static_cast<unsigned short>(bits.x), static_cast<unsigned short>(bits.x >> 16U)});
  const auto hi = __bfloat1622float2(__nv_bfloat162_raw{
      static_cast<unsigned short>(bits.y), static_cast<unsigned short>(bits.y >> 16U)});
  return make_float4(lo.x, lo.y, hi.x, hi.y);
}

// One CTA owns one 128x128 scale block. Each lane reads four consecutive
// values; the quantized bytes go to both orientations without a global reread.
template <class T, bool Aligned>
__global__ void quantize_weight_kernel(const T *input, std::uint8_t *output,
    float *scales, std::uint8_t *output_t, float *scales_t,
    unsigned rows, unsigned columns) {
  const unsigned nr = (rows + 127U) / 128U, nc = (columns + 127U) / 128U;
  const unsigned br = blockIdx.x / nc, bc = blockIdx.x % nc;
  const unsigned lane = threadIdx.x % 32U, warp = threadIdx.x / 32U;
  const auto in_base = std::uint64_t(blockIdx.y) * rows * columns;
  float4 values[16];
  float amax = 0.0F;
  unsigned invalid = 0;
#pragma unroll
  for (unsigned i = 0; i < 16U; ++i) {
    const unsigned r = br * 128U + warp + i * 8U, c = bc * 128U + lane * 4U;
    auto &v = values[i];
    if constexpr (Aligned) {
      v = load_weight_four(input + in_base + std::uint64_t(r) * columns + c);
    } else {
      v = make_float4(0, 0, 0, 0);
      if (r < rows) {
        const auto *p = input + in_base + std::uint64_t(r) * columns;
        if (c < columns) v.x = static_cast<float>(p[c]);
        if (c + 1U < columns) v.y = static_cast<float>(p[c + 1U]);
        if (c + 2U < columns) v.z = static_cast<float>(p[c + 2U]);
        if (c + 3U < columns) v.w = static_cast<float>(p[c + 3U]);
      }
    }
    invalid |= !isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z) || !isfinite(v.w);
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
  }
  __shared__ float maxima[8];
  __shared__ float scale;
  // Swizzle the word column by row/4: both rowwise word stores and each
  // of the four transposed byte-gather loads have conflict-free bank access.
  __shared__ unsigned tile[128][32];
  const int invalid_block = __syncthreads_or(invalid != 0U);
  amax = warp_max(amax);
  if (lane == 0U) maxima[warp] = amax;
  __syncthreads();
  if (warp == 0U) {
    amax = warp_max(lane < 8U ? maxima[lane] : 0.0F);
    if (lane == 0U) {
      scale = invalid_block ? nanf("") : block_scale(amax);
      scales[(std::uint64_t(blockIdx.y) * nr + br) * nc + bc] = scale;
      scales_t[(std::uint64_t(blockIdx.y) * nc + bc) * nr + br] = scale;
    }
  }
  __syncthreads();
  const auto out_base = std::uint64_t(blockIdx.y) * rows * nc * 128U;
#pragma unroll
  for (unsigned i = 0; i < 16U; ++i) {
    const unsigned r = warp + i * 8U, c = lane * 4U;
    const auto &v = values[i];
    const unsigned lo = __nv_fp8x2_e4m3(make_float2(v.x / scale, v.y / scale)).__x;
    const unsigned hi = __nv_fp8x2_e4m3(make_float2(v.z / scale, v.w / scale)).__x;
    unsigned packed = lo | (hi << 16U);
    if constexpr (!Aligned) {
      if (br * 128U + r >= rows) packed = 0;
#pragma unroll
      for (unsigned j = 0; j < 4U; ++j)
        if (bc * 128U + c + j >= columns) packed &= ~(0xffU << (j * 8U));
    }
    tile[r][lane ^ (r / 4U)] = packed;
    if (Aligned || br * 128U + r < rows)
      *reinterpret_cast<unsigned *>(output + out_base +
          std::uint64_t(br * 128U + r) * nc * 128U + bc * 128U + c) = packed;
  }
  __syncthreads();
  const auto out_t_base = std::uint64_t(blockIdx.y) * columns * nr * 128U;
#pragma unroll
  for (unsigned i = 0; i < 16U; ++i) {
    const unsigned r = warp + i * 8U, c = lane * 4U;
    unsigned packed = 0;
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j)
      packed |= ((tile[c + j][(r / 4U) ^ lane] >> ((r % 4U) * 8U)) & 0xffU) << (j * 8U);
    if (Aligned || bc * 128U + r < columns)
      *reinterpret_cast<unsigned *>(output_t + out_t_base +
          std::uint64_t(bc * 128U + r) * nr * 128U + br * 128U + c) = packed;
  }
}

static __global__ void transpose_weight_kernel(Matrix input, std::uint8_t *output,
                                        float *scales) {
  __shared__ std::uint8_t tile[32][33];
  const auto row = blockIdx.y * 32U + threadIdx.y;
  const auto col = blockIdx.x * 32U + threadIdx.x;
  const auto in_base = std::uint64_t(blockIdx.z) * input.rows * input.stride;
  const auto out_stride = (input.rows + 127U) / 128U * 128U;
  const auto out_base = std::uint64_t(blockIdx.z) * input.columns * out_stride;
  for (unsigned i = 0; i < 32U; i += 8U)
    tile[threadIdx.y + i][threadIdx.x] =
        row + i < input.rows && col < input.columns
            ? input.data[in_base + std::uint64_t(row + i) * input.stride + col]
            : 0;
  __syncthreads();
  const auto out_row = blockIdx.x * 32U + threadIdx.y;
  const auto out_col = blockIdx.y * 32U + threadIdx.x;
  for (unsigned i = 0; i < 32U; i += 8U)
    if (out_row + i < input.columns && out_col < out_stride)
      output[out_base + std::uint64_t(out_row + i) * out_stride + out_col] =
          tile[threadIdx.x][threadIdx.y + i];
  if (threadIdx.x == 0U && threadIdx.y == 0U && blockIdx.x % 4U == 0U &&
      blockIdx.y % 4U == 0U) {
    const auto r = blockIdx.y / 4U, c = blockIdx.x / 4U;
    const auto nr = (input.rows + 127U) / 128U;
    const auto nc = (input.columns + 127U) / 128U;
    scales[(std::uint64_t(blockIdx.z) * nc + c) * nr + r] =
        input.scales[(std::uint64_t(blockIdx.z) * nr + r) * nc + c];
  }
}

} // namespace weftmoe::cuda::fp8
