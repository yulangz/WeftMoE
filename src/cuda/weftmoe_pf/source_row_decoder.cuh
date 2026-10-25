#pragma once

#include <cstdint>

namespace weftmoe_pf {

// Return the source and source-local ordinal of an expert's interleaved row.
// Lanes stripe the rank domain, so this remains valid beyond one warp of ranks.
template <class Args>
__device__ __forceinline__ void
decode_source_row(const Args &a, unsigned expert, std::uint64_t row,
                  unsigned &source, unsigned &source_slot) {
  const auto lane = threadIdx.x & 31U;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  unsigned peeled = 0U;
  auto slot = static_cast<unsigned>(row);
  while (true) {
    unsigned active = 0U, layer = ~0U;
    for (unsigned rank = lane; rank < a.world; rank += 32U) {
      const auto count =
          a.fused_local_expert_count[std::uint64_t(rank) * owned + expert];
      if (count > peeled) {
        ++active;
        layer = min(layer, static_cast<unsigned>(count) - peeled);
      }
    }
    active = __reduce_add_sync(0xffffffffU, active);
    layer = __reduce_min_sync(0xffffffffU, layer);
    const auto rectangle = std::uint64_t(active) * layer;
    if (slot < rectangle) {
      auto ordinal = slot % active;
      source_slot = peeled + slot / active;
      for (unsigned base = 0; base < a.world; base += 32U) {
        const auto rank = base + lane;
        const bool present =
            rank < a.world &&
            a.fused_local_expert_count[std::uint64_t(rank) * owned + expert] >
                peeled;
        const auto mask = __ballot_sync(0xffffffffU, present);
        const auto count = __popc(mask);
        if (ordinal < count) {
          source = base + __fns(mask, 0U, ordinal + 1U);
          return;
        }
        ordinal -= count;
      }
    }
    slot -= static_cast<unsigned>(rectangle);
    peeled += layer;
  }
}

// Warp-cooperative cursor over immutable source counts. Each caller warp
// must visit valid, nondecreasing rows within an expert; changing expert
// resets the cursor. Pull traverses exactly this order for both X and dY.
// Retain each round-robin layer instead of reloading counts and rediscovering
// all preceding layers per row. Larger rank domains retain the generic path.
template <class Args> struct SourceRowCursor {
  const Args &a;
  unsigned expert = ~0U, count = 0U, peeled = 0U, next_peeled = 0U;
  unsigned mask = 0U, active = 0U;
  std::uint64_t begin = 0U, end = 0U;

  __device__ explicit SourceRowCursor(const Args &args) : a(args) {}

  __device__ __forceinline__ void decode(unsigned current_expert,
                                         std::uint64_t row, unsigned &source,
                                         unsigned &ordinal) {
    if (a.world > 32U) {
      decode_source_row(a, current_expert, row, source, ordinal);
      return;
    }
    if (expert != current_expert) {
      expert = current_expert;
      const auto lane = threadIdx.x % 32U;
      count = lane < a.world
                  ? unsigned(
                        a.fused_local_expert_count[std::uint64_t(lane) *
                                                       a.dispatch.sizes
                                                           .owned_expert_count +
                                                   expert])
                  : 0U;
      peeled = next_peeled = 0U;
      begin = end = 0U;
    }
    while (row >= end) {
      peeled = next_peeled;
      begin = end;
      mask = __ballot_sync(0xffffffffU, count > peeled);
      active = __popc(mask);
      next_peeled =
          __reduce_min_sync(0xffffffffU, count > peeled ? count : ~0U);
      end = begin + std::uint64_t(active) * (next_peeled - peeled);
    }
    const auto offset = unsigned(row - begin);
    unsigned position;
    if ((active & (active - 1U)) == 0U) {
      ordinal = peeled + (offset >> (__ffs(active) - 1));
      position = offset & (active - 1U);
    } else {
      ordinal = peeled + offset / active;
      position = offset % active;
    }
    source = __fns(mask, 0U, position + 1U);
  }
};

} // namespace weftmoe_pf
