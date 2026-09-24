#pragma once

// Included inside forward_compute_detail. FC1's first N tile owns each X
// checkpoint. Resolve immutable destinations once per task, then read each
// ready A stage before WGMMA wait/empty arrivals permit stage recycling.
template <unsigned TileK> struct StaticV4XCheckpoint {
  static constexpr auto kVectors =
      forward_tuning::kTileM * TileK / (8U * kMathThreads);
  transport::BFloat16 *destinations[kVectors];

  template <class Args>
  __device__ __forceinline__ void prepare(const Args &a, std::uint64_t mblock,
                                          unsigned valid_rows) {
    const auto tid = threadIdx.x % kMathThreads;
#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto m = (tid + index * kMathThreads) / (TileK / 8U);
      if (m < valid_rows) {
        const auto saved =
            a.checkpoint.rows[mblock * forward_tuning::kTileM + m];
        destinations[index] = a.saved_x + saved * a.hidden;
      }
    }
  }

  __device__ __forceinline__ void save(const transport::BFloat16 *stage,
                                       unsigned valid_rows,
                                       unsigned column_base,
                                       std::uint64_t hidden) const {
    const auto tid = threadIdx.x % kMathThreads;
#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto vector = tid + index * kMathThreads;
      const auto m = vector / (TileK / 8U);
      const auto column = (vector % (TileK / 8U)) * 8U;
      if (m < valid_rows && column_base + column + 8U <= hidden) {
        auto *target = destinations[index] + column_base + column;
        const auto *source = stage + m128_fragment_vector_offset(m, column);
        *reinterpret_cast<uint4 *>(target) =
            *reinterpret_cast<const uint4 *>(source);
      }
    }
  }
};
