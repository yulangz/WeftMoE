#pragma once

// Included inside forward_compute_detail. FC1's first N tile owns each X
// checkpoint. Single-node training orders canonical rows within each block:
// store contiguous M32 runs with TMA and scatter only the remainder. Network
// training retains round-robin geometry and vector copies.
// Both readers must retire before either CTA can recycle its ready A stage.
template <unsigned TileK, bool Contiguous> struct StaticV4XCheckpoint {
  static constexpr auto kChunkRows = 32U;
  static constexpr auto kChunks = forward_tuning::kTileM / kChunkRows;
  static constexpr auto kVectors =
      forward_tuning::kTileM * TileK / (8U * kMathThreads);
  unsigned canonical_begin[kChunks];
  unsigned tma_mask;
  transport::BFloat16 *destinations[kVectors];

  template <class Args>
  __device__ __forceinline__ void prepare(const Args &a, std::uint64_t mblock,
                                          unsigned valid_rows) {
    const auto tid = threadIdx.x % kMathThreads;
    const auto *rows = a.checkpoint.rows + mblock * forward_tuning::kTileM;
    if constexpr (Contiguous) {
      tma_mask = 0U;
#pragma unroll
      for (unsigned chunk = 0U; chunk < kChunks; ++chunk) {
        const auto first = chunk * kChunkRows;
        if (first + kChunkRows <= valid_rows) {
          const auto begin = rows[first];
          if (rows[first + kChunkRows - 1U] == begin + kChunkRows - 1U) {
            canonical_begin[chunk] = static_cast<unsigned>(begin);
            tma_mask |= 1U << chunk;
          }
        }
      }
    }
#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto m = (tid + index * kMathThreads) / (TileK / 8U);
      if (m < valid_rows)
        destinations[index] =
            (Contiguous && (tma_mask & (1U << (m / kChunkRows))))
                ? nullptr
                : a.saved_x + rows[m] * a.hidden;
    }
  }

  __device__ __forceinline__ void save(const transport::BFloat16 *stage,
                                       unsigned valid_rows,
                                       unsigned column_base,
                                       std::uint64_t hidden,
                                       const CUtensorMap &map) const {
    if (Contiguous && threadIdx.x == 0U && tma_mask) {
      cute::tma_store_fence();
#pragma unroll
      for (unsigned chunk = 0U; chunk < kChunks; ++chunk)
        if (tma_mask & (1U << chunk)) {
#pragma unroll
          for (unsigned column = 0U; column < TileK; column += 64U)
            if (column_base + column < hidden)
              cute::SM90_TMA_STORE_2D::copy(
                  &map, stage + (column / 64U) * forward_tuning::kTileM * 64U +
                            chunk * kChunkRows * 64U,
                  column_base + column, canonical_begin[chunk]);
        }
      cute::tma_store_arrive();
    }
    const auto tid = threadIdx.x % kMathThreads;
#pragma unroll
    for (unsigned index = 0U; index < kVectors; ++index) {
      const auto vector = tid + index * kMathThreads;
      const auto m = vector / (TileK / 8U);
      const auto column = (vector % (TileK / 8U)) * 8U;
      if (m < valid_rows && column_base + column + 8U <= hidden &&
          (!Contiguous || destinations[index])) {
        auto *target = destinations[index] + column_base + column;
        const auto *source = stage + m128_fragment_vector_offset(m, column);
        *reinterpret_cast<uint4 *>(target) =
            *reinterpret_cast<const uint4 *>(source);
      }
    }
  }

  __device__ __forceinline__ void finish() const {
    if (Contiguous && threadIdx.x < 32U && tma_mask) {
      if (threadIdx.x == 0U)
        // Full global completion also precedes FC1's checkpoint publication.
        asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
      // Hand completion to both CTA empty arrivals (lanes 0/1), without
      // relying on implicit warp convergence after the issuer-only wait.
      __syncwarp();
    }
  }
};
