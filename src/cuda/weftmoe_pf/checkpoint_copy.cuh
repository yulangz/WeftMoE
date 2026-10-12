#pragma once

// Included inside forward_compute_detail. One static FC1 pair owns each X
// checkpoint, with each CTA copying half of M from its ready multicast A.
// Both readers must retire before either CTA can recycle its ready A stage.
template <unsigned TileK, unsigned Rows = forward_tuning::kTileM>
struct StaticV4XCheckpoint {
  static constexpr auto kVectors = Rows * TileK / (8U * kMathThreads);
  transport::BFloat16 *destinations[kVectors];

  template <class Args>
  __device__ __forceinline__ void prepare(const Args &a, std::uint64_t mblock,
                                          unsigned valid_rows,
                                          unsigned row_begin = 0U) {
    const auto tid = threadIdx.x % kMathThreads;
    const auto *rows =
        a.checkpoint.rows + mblock * forward_tuning::kTileM + row_begin;

#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto m = (tid + index * kMathThreads) / (TileK / 8U);
      if (m < valid_rows)
        destinations[index] = a.saved_x + rows[m] * a.hidden;
    }
  }

  __device__ __forceinline__ void save(const transport::BFloat16 *stage,
                                       unsigned valid_rows,
                                       unsigned column_base,
                                       std::uint64_t hidden,
                                       const CUtensorMap &map) const {
    const auto tid = threadIdx.x % kMathThreads;
#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto vector = tid + index * kMathThreads;
      const auto m = vector / (TileK / 8U);
      const auto column = (vector % (TileK / 8U)) * 8U;
      if (m < valid_rows && column_base + column + 8U <= hidden) {
        auto *target = destinations[index] + column_base + column;
        const auto *source = stage + m_fragment_vector_offset(m, column);
        *reinterpret_cast<uint4 *>(target) =
            *reinterpret_cast<const uint4 *>(source);
      }
    }
  }
};
