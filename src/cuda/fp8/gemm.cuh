#pragma once

#include "cuda/primitives/hopper.cuh"
#include <cuda_bf16.h>

namespace weftmoe::cuda::fp8 {

// A 128-byte swizzled row contains 128 FP8 elements. XOR the 16-byte
// sectors by the row within an eight-row atom, matching GMMA's B128 layout.
__device__ __forceinline__ unsigned swizzle(unsigned row, unsigned col) {
  return row * 128U + (col ^ ((row & 7U) * 16U));
}

// This small standalone mainloop is also the numerical baseline for the PF
// pipelined implementation. It executes actual FP8 WGMMA, promoting each
// K128 partial sum into a separately scaled FP32 accumulator.
template <bool FloatOutput, bool Accumulate>
__global__ __launch_bounds__(128) void gemm_kernel(Matrix a, Matrix b,
                                                 void *output) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using Mma = hopper::FP8Mma<64>;
  __shared__ __align__(1024) std::uint8_t sa[64U * 128U];
  __shared__ __align__(1024) std::uint8_t sb[64U * 128U];
  const unsigned m_base = blockIdx.y * 64U, n_base = blockIdx.x * 64U;
  const unsigned lane = threadIdx.x % 32U;
  const unsigned r0 = threadIdx.x / 32U * 16U + lane / 4U;
  const unsigned c0 = lane % 4U * 2U;
  const unsigned k_blocks = (a.columns + 127U) / 128U;
  float total[Mma::kNumAccum] = {};
  for (unsigned kb = 0; kb < k_blocks; ++kb) {
    for (unsigned index = threadIdx.x; index < 64U * 128U; index += 128U) {
      const unsigned row = index / 128U, col = index % 128U;
      const unsigned k = kb * 128U + col;
      sa[swizzle(row, col)] = m_base + row < a.rows && k < a.columns
          ? a.data[std::uint64_t(m_base + row) * a.stride + k] : 0;
      sb[swizzle(row, col)] = n_base + row < b.rows && k < b.columns
          ? b.data[std::uint64_t(n_base + row) * b.stride + k] : 0;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    float partial[Mma::kNumAccum];
    // Initialize even though the first MMA uses ScaleOut::Zero, so register
    // fences never observe uninitialized C++ values.
#pragma unroll
    for (auto &value : partial) {
      value = 0.0F;
      cute::warpgroup_fence_operand(value);
    }
    cute::warpgroup_arrive();
#pragma unroll
    for (unsigned k = 0; k < 128U; k += Mma::K) {
      const auto da = hopper::make_gmma_desc<hopper::Major::K, 64, 128, 128>(sa, 0, k);
      const auto db = hopper::make_gmma_desc<hopper::Major::K, 64, 128, 128>(sb, 0, k);
      Mma::wgmma(da, db, partial, k != 0U);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
#pragma unroll
    for (auto &value : partial)
      cute::warpgroup_fence_operand(value);
#pragma unroll
    for (unsigned j = 0; j < Mma::kNumAccum / 4U; ++j) {
#pragma unroll
      for (unsigned e = 0; e < 4U; ++e) {
        const unsigned r = m_base + r0 + (e / 2U) * 8U;
        const unsigned c = n_base + c0 + j * 8U + e % 2U;
        if (r < a.rows && c < b.rows) {
          const float scale = a.scales[std::uint64_t(r / a.scale_rows) * k_blocks + kb] *
                              b.scales[std::uint64_t(c / b.scale_rows) * k_blocks + kb];
          total[j * 4U + e] = fmaf(partial[j * 4U + e], scale, total[j * 4U + e]);
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (unsigned j = 0; j < Mma::kNumAccum / 4U; ++j) {
#pragma unroll
    for (unsigned e = 0; e < 4U; ++e) {
      const unsigned r = m_base + r0 + (e / 2U) * 8U;
      const unsigned c = n_base + c0 + j * 8U + e % 2U;
      if (r < a.rows && c < b.rows) {
        const auto index = std::uint64_t(r) * b.rows + c;
        auto value = total[j * 4U + e];
        if constexpr (FloatOutput) {
          if constexpr (Accumulate)
            value += static_cast<float *>(output)[index];
          static_cast<float *>(output)[index] = value;
        } else {
          static_cast<__nv_bfloat16 *>(output)[index] = __float2bfloat16_rn(value);
        }
      }
    }
  }
#else
  __trap();
#endif
}

} // namespace weftmoe::cuda::fp8
