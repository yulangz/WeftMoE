#pragma once

// Included inside forward_compute_detail. FC1's first N tile owns each X
// checkpoint. Read its ready A stage while WGMMA also reads it, before the
// warpgroup wait and empty arrivals permit either CTA to recycle the stage.
template <unsigned TileK, class Args>
__device__ __forceinline__ void static_v4_save_x_stage(
    const Args &a, const transport::BFloat16 *stage, std::uint64_t mblock,
    unsigned valid_rows, unsigned column_base) {
  constexpr auto TileM = forward_tuning::kTileM;
#pragma unroll
  for (unsigned vector = threadIdx.x; vector < TileM * TileK / 8U;
       vector += kMathThreads) {
    const auto m = vector / (TileK / 8U);
    const auto column = (vector % (TileK / 8U)) * 8U;
    if (m < valid_rows && column_base + column + 8U <= a.hidden) {
      const auto saved = a.checkpoint.rows[mblock * TileM + m];
      auto *target = a.saved_x + saved * a.hidden + column_base + column;
      const auto *source = stage + m128_fragment_vector_offset(m, column);
      *reinterpret_cast<uint4 *>(target) =
          *reinterpret_cast<const uint4 *>(source);
    }
  }
}
