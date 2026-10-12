#pragma once

#include <cstdint>

#include "cuda/pipeline/atomics.cuh"

namespace weftmoe::cuda::pipeline::backward {

// Prefer sharing the larger operand. An odd extent switches pairing direction
// at the boundary; only an odd-by-odd grid has a singleton tail.
struct ClusterTile {
  std::uint32_t m = 0U;
  std::uint32_t n = 0U;
  bool share_a = true;
  bool valid = true;
};

WEFTMOE_PIPELINE_HOST_DEVICE constexpr ClusterTile
cluster_tile(std::uint32_t pair, std::uint32_t rank, std::uint32_t m_tiles,
             std::uint32_t n_tiles, bool prefer_share_a = false) {
  const auto position = pair * 2U + rank;
  if ((n_tiles & 1U) == 0U && (prefer_share_a || (m_tiles & 1U) != 0U))
    return {position / n_tiles, position % n_tiles, true, true};
  const auto paired_m = m_tiles & ~1U;
  if (position < paired_m * n_tiles)
    return {(position / (2U * n_tiles)) * 2U + rank, (position / 2U) % n_tiles,
            false, true};
  const auto n = position - paired_m * n_tiles;
  return {m_tiles - 1U, n < n_tiles ? n : n_tiles - 1U, true, n < n_tiles};
}

} // namespace weftmoe::cuda::pipeline::backward
