#pragma once

#include <cstdint>

#include "cuda/persistent_v3/queues.cuh"

namespace megamoe::cuda::persistent_v3::backward {

inline constexpr std::uint32_t kDabPhaseTileCapacity = 128U;
inline constexpr std::uint32_t kCounterWords =
    sizeof(PaddedCounter) / sizeof(unsigned long long);

struct BackwardMblockDescriptor {
  std::uint32_t compact_row = 0U;
  std::uint32_t padded_row = 0U;
  std::uint32_t expert = 0U;
  std::uint32_t valid_rows = 0U;
};

static_assert(sizeof(BackwardMblockDescriptor) == 16U);
static_assert(kCounterWords == 16U);

struct BackwardSchedulerLayout {
  std::uint64_t descriptors = 0ULL;
  std::uint64_t dispatch_arrived = 0ULL;
  std::uint64_t phase = 0ULL;
  std::uint64_t du_completed = 0ULL;
  std::uint64_t dab_completed = 0ULL;
  std::uint64_t dx_completed = 0ULL;
  std::uint64_t expert_mblocks_completed = 0ULL;
  std::uint64_t dx_ready = 0ULL;
  std::uint64_t dab_ready = 0ULL;
  std::uint64_t dw2_ready = 0ULL;
  std::uint64_t dw1_ready = 0ULL;
  std::uint64_t dx_claim = 0ULL;
  std::uint64_t dab_claim = 0ULL;
  std::uint64_t dw2_claim = 0ULL;
  std::uint64_t dw1_claim = 0ULL;
  std::uint64_t dx_scan = 0ULL;
  std::uint64_t dab_scan = 0ULL;
  std::uint64_t dw2_scan = 0ULL;
  std::uint64_t dw1_scan = 0ULL;
  std::uint64_t dispatch_scan = 0ULL;
  std::uint64_t dispatch_route_claim = 0ULL;
  std::uint64_t dispatched_routes = 0ULL;
  std::uint64_t return_scan = 0ULL;
  std::uint64_t returned_mblocks = 0ULL;
  std::uint64_t completed_dx_tiles = 0ULL;
  std::uint64_t completed_dw2_tiles = 0ULL;
  std::uint64_t completed_dw1_tiles = 0ULL;
  std::uint64_t total_words = 0ULL;
};

MEGAMOE_V3_HOST_DEVICE constexpr std::uint64_t
align_backward_scheduler_words(std::uint64_t words) {
  return (words + kCounterWords - 1ULL) & ~(kCounterWords - 1ULL);
}

MEGAMOE_V3_HOST_DEVICE constexpr BackwardSchedulerLayout
make_backward_scheduler_layout(std::uint64_t mblocks,
                               std::uint32_t dx_tiles,
                               std::uint64_t experts) {
  BackwardSchedulerLayout layout{};
  std::uint64_t cursor = 0ULL;
  layout.descriptors = cursor;
  cursor += 2ULL * mblocks;
  layout.dispatch_arrived = cursor;
  cursor += mblocks;
  layout.phase = cursor;
  cursor += mblocks;
  layout.du_completed = cursor;
  cursor += mblocks;
  layout.dab_completed = cursor;
  cursor += mblocks;
  layout.dx_completed = cursor;
  cursor += mblocks;
  layout.expert_mblocks_completed = cursor;
  cursor += experts;
  const auto ready_words = (mblocks + 63ULL) / 64ULL;
  const auto expert_ready_words = (experts + 63ULL) / 64ULL;
  layout.dx_ready = cursor;
  cursor += ready_words;
  layout.dab_ready = cursor;
  cursor += ready_words;
  layout.dw2_ready = cursor;
  cursor += expert_ready_words;
  layout.dw1_ready = cursor;
  cursor += expert_ready_words;
  layout.dx_claim = cursor;
  cursor += mblocks * ((dx_tiles + 63ULL) / 64ULL);
  layout.dab_claim = cursor;
  cursor += mblocks *
            ((kDabPhaseTileCapacity + 63ULL) / 64ULL);
  layout.dw2_claim = cursor;
  cursor += experts;
  layout.dw1_claim = cursor;
  cursor += experts;
  const auto counter = [&](std::uint64_t &field) constexpr {
    cursor = align_backward_scheduler_words(cursor);
    field = cursor;
    cursor += kCounterWords;
  };
  counter(layout.dx_scan);
  counter(layout.dab_scan);
  counter(layout.dw2_scan);
  counter(layout.dw1_scan);
  counter(layout.dispatch_scan);
  counter(layout.dispatch_route_claim);
  counter(layout.dispatched_routes);
  counter(layout.return_scan);
  counter(layout.returned_mblocks);
  counter(layout.completed_dx_tiles);
  counter(layout.completed_dw2_tiles);
  counter(layout.completed_dw1_tiles);
  layout.total_words = align_backward_scheduler_words(cursor);
  return layout;
}

} // namespace megamoe::cuda::persistent_v3::backward
