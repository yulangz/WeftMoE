#pragma once

#include "cuda/fp8/api.hpp"
#include "cuda/fp8/quantization.cuh"
#include "cuda/primitives/hopper.cuh"

namespace weftmoe::cuda::fp8 {

#ifndef WEFTMOE_FP8_MATH_REGISTERS
#define WEFTMOE_FP8_MATH_REGISTERS 216
#endif
inline constexpr unsigned kMathRegisters = WEFTMOE_FP8_MATH_REGISTERS;
inline constexpr unsigned kControlRegisters = 504U - 2U * kMathRegisters;
static_assert(kMathRegisters >= 168U && kMathRegisters <= 240U && kMathRegisters % 8U == 0U);

struct NoSync {
  __device__ static void sync() {}
};

struct MathSync {
  __device__ static void sync() {
    cutlass::arch::NamedBarrier::sync(256,
        static_cast<cutlass::arch::ReservedNamedBarriers>(1));
  }
};


__device__ __forceinline__ unsigned tile_offset(unsigned row, unsigned k) {
  return row * 128U + (k ^ ((row & 7U) * 16U));
}

struct WeightReader {
  const std::uint8_t *data;
  const float *scales;
  unsigned columns, output_begin, intermediate;
  bool interleave;
  __device__ unsigned logical_row(unsigned row) const {
    return interleave
        ? output_begin + row / 16U * 8U + row % 8U + ((row % 16U >= 8U) ? intermediate : 0U)
        : output_begin + row;
  }
  __device__ std::uint8_t value(unsigned row, unsigned k) const {
    return data[std::uint64_t(logical_row(row)) * columns + k];
  }
  __device__ float scale(unsigned row, unsigned k_block) const {
    return scales[std::uint64_t(logical_row(row) / 128U) * (columns / 128U) + k_block];
  }
};

template <unsigned N, class Sync = MathSync, bool BlockWeight = false,
          bool AsyncOperands = false>
__device__ __forceinline__ void multiply_tile(
    std::uint8_t *qa, std::uint8_t *qb, const float *sa, const float *sb,
    float *total) {
  // Keep only one N128 partial live alongside the final N-column result.
  // A native N256 partial plus the final accumulator exceeds the persistent
  // worker's register allocation and makes ptxas serialize/spill WGMMA.
  constexpr unsigned AtomN = N < 128U ? N : 128U;
  using Mma = hopper::FP8Mma<AtomN>;
  Sync::sync();
  // TMA already writes in the async proxy consumed by WGMMA. Ordinary
  // shared stores from the fallback quantizer still need the proxy fence.
  if constexpr (!AsyncOperands)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const auto r = threadIdx.x / 32U * 16U + threadIdx.x % 32U / 4U;
  const auto c = threadIdx.x % 4U * 2U;
  auto da = hopper::make_gmma_desc<hopper::Major::K, 128, 128, 128>(
      qa, threadIdx.x / 128U * 64U, 0U);
  auto db = hopper::make_gmma_desc<hopper::Major::K, N, 128, 128>(qb, 0U, 0U);
  const auto abase = __shfl_sync(0xffffffffU, da.reg32_[0], 0U);
  const auto bbase = __shfl_sync(0xffffffffU, db.reg32_[0], 0U);
  // The first WGMMA overwrites every accumulator (ScaleOut::Zero).
  // Reuse the fragment without emitting register clears for every K128.
  float partial[Mma::kNumAccum];
#pragma unroll
  for (unsigned segment = 0U; segment < N / AtomN; ++segment) {
    float factors[4];
    if constexpr (BlockWeight) {
      // A canonical N128 weight block has one scale. The packed gate/up
      // layout alternates two such blocks every eight output columns.
      const unsigned scale_block = segment * AtomN / 128U * 2U;
      const float b0 = sb[scale_block], b1 = sb[scale_block + 1U];
      factors[0] = sa[r] * b0;
      factors[1] = sa[r + 8U] * b0;
      factors[2] = sa[r] * b1;
      factors[3] = sa[r + 8U] * b1;
    }
#pragma unroll
    for (auto &v : partial)
      cute::warpgroup_fence_operand(v);
    cute::warpgroup_arrive();
#pragma unroll
    for (unsigned k = 0U; k < 128U; k += 32U) {
      da.reg32_[0] = abase + k / 16U;
      db.reg32_[0] = bbase + (segment * AtomN * 128U + k) / 16U;
      Mma::wgmma(da, db, partial, k != 0U);
    }
    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();
#pragma unroll
    for (auto &v : partial)
      cute::warpgroup_fence_operand(v);
#pragma unroll
    for (unsigned j = 0U; j < Mma::kNumAccum / 4U; ++j) {
#pragma unroll
      for (unsigned e = 0U; e < 4U; ++e) {
        const auto index = segment * Mma::kNumAccum + j * 4U + e;
        const auto scale = BlockWeight ? factors[(j % 2U) * 2U + e / 2U]
            : sa[r + e / 2U * 8U] * sb[segment * AtomN + c + j * 8U + e % 2U];
        total[index] = fmaf(partial[j * 4U + e], scale, total[index]);
      }
    }
  }
  Sync::sync();
}

// Two N64 fragments occupy the same registers as the usual N128 fragment.
// Retire one fragment while the next WGMMA group remains in flight. The
// caller keeps the operand stage alive until the final wait below.
template <class Sync = MathSync>
__device__ __forceinline__ void multiply_tile_pipelined_256(
    std::uint8_t *qa, std::uint8_t *qb, const float *sa, const float *sb,
    float *total) {
  using Mma = hopper::FP8Mma<64U>;
  Sync::sync();
  const auto r = threadIdx.x / 32U * 16U + threadIdx.x % 32U / 4U;
  const auto c = threadIdx.x % 4U * 2U;
  const auto da_base = hopper::make_gmma_desc<hopper::Major::K, 128, 128, 128>(
      qa, threadIdx.x / 128U * 64U, 0U);
  const auto db_base = hopper::make_gmma_desc<hopper::Major::K, 256, 128, 128>(qb, 0U, 0U);
  const auto abase = __shfl_sync(0xffffffffU, da_base.reg32_[0], 0U);
  const auto bbase = __shfl_sync(0xffffffffU, db_base.reg32_[0], 0U);
  float first[Mma::kNumAccum], second[Mma::kNumAccum];
  const auto issue = [&](float (&partial)[Mma::kNumAccum], unsigned segment) {
    auto da = da_base, db = db_base;
#pragma unroll
    for (auto &v : partial)
      cute::warpgroup_fence_operand(v);
    cute::warpgroup_arrive();
#pragma unroll
    for (unsigned k = 0U; k < 128U; k += 32U) {
      da.reg32_[0] = abase + k / 16U;
      db.reg32_[0] = bbase + (segment * 64U * 128U + k) / 16U;
      Mma::wgmma(da, db, partial, k != 0U);
    }
    cute::warpgroup_commit_batch();
  };
  const auto promote = [&](float (&partial)[Mma::kNumAccum], unsigned segment) {
#pragma unroll
    for (auto &v : partial)
      cute::warpgroup_fence_operand(v);
#pragma unroll
    for (unsigned j = 0U; j < Mma::kNumAccum / 4U; ++j) {
#pragma unroll
      for (unsigned e = 0U; e < 4U; ++e) {
        const auto index = segment * Mma::kNumAccum + j * 4U + e;
        const auto scale = sa[r + e / 2U * 8U] *
            sb[segment * 64U + c + j * 8U + e % 2U];
        total[index] = fmaf(partial[j * 4U + e], scale, total[index]);
      }
    }
  };
  issue(first, 0U);
  issue(second, 1U);
  cute::warpgroup_wait<1>();
  promote(first, 0U);
  issue(first, 2U);
  cute::warpgroup_wait<1>();
  promote(second, 1U);
  issue(second, 3U);
  cute::warpgroup_wait<1>();
  promote(first, 2U);
  cute::warpgroup_wait<0>();
  promote(second, 3U);
  Sync::sync();
}

} // namespace weftmoe::cuda::fp8
