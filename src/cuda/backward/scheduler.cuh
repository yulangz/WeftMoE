#pragma once

#include <cstdint>

#include "cuda/backward/tuning_config.hpp"
#include "cuda/pipeline/atomics.cuh"

namespace weftmoe::cuda::pipeline::backward {

enum class TaskKind : std::uint8_t {
  Dx = 0U,
  Dab = 1U,
  Dw2 = 2U,
  Dw1 = 3U,
  None = 4U,
};

inline constexpr std::uint32_t kTaskKindCount = 4U;

// Activation owners are M-blocks and retain their locality-aware ready bitmap.
// Weight-gradient owners are experts, published once after their inputs finish.
struct TileTask {
  std::uint32_t owner = 0U;
  std::uint32_t tile = 0U;
  TaskKind kind = TaskKind::None;
  std::uint32_t tile_count = 1U;
};

} // namespace weftmoe::cuda::pipeline::backward
