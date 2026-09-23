#pragma once

#include "cuda/static_v4/source_plan.cuh"

namespace weftmoe::cuda::static_forward {

struct WindowGeometryView {
  WindowShape shape;
  std::uint64_t window;
  std::uint32_t owned_experts;
  const std::uint32_t
      *source_counts;           // [owned_experts, G], after count exchange.
  std::uint64_t *expert_rows;   // [owned_experts]
  std::uint64_t *mblock_prefix; // [owned_experts + 1]
  WindowMblock *mblocks;        // [shape.max_mblocks(owned_experts)]

  // Inverse of source-rank round-robin enumeration. At layer ordinal, all
  // sources with that many rows have one entry, ordered by source rank.
  __device__ std::uint64_t expert_row(unsigned expert, unsigned source,
                                      unsigned ordinal) const {
    const auto sources = shape.source_end(window) - shape.source_begin(window);
    const auto local_source = source - shape.source_begin(window);
    const auto *counts = source_counts + std::uint64_t(expert) * shape.sources;
    std::uint64_t row = 0U;
    for (unsigned s = 0; s < sources; ++s) {
      row += counts[s] < ordinal ? counts[s] : ordinal;
      row += s < local_source && counts[s] > ordinal;
    }
    return row;
  }
};

// One fixed CTA/control group owns a destination window. Whole-window X/S/C
// admission will reserve mblock_prefix[owned_experts] slots on every plane
// before this directory can be consumed by pull or compute.
template <unsigned Threads, unsigned BaseThread = 0U, unsigned Barrier = 0U>
__device__ void build_window_counts(const WindowGeometryView &a) {
  const auto tid = threadIdx.x - BaseThread;
  const auto sources =
      a.shape.source_end(a.window) - a.shape.source_begin(a.window);
  for (unsigned expert = tid; expert < a.owned_experts; expert += Threads) {
    std::uint64_t rows = 0U;
    for (unsigned source = 0; source < sources; ++source)
      rows += a.source_counts[std::uint64_t(expert) * a.shape.sources + source];
    a.expert_rows[expert] = rows;
  }
  source_plan_sync<Threads, Barrier>();
  if (tid == 0U) {
    std::uint64_t blocks = 0U;
    for (unsigned expert = 0; expert < a.owned_experts; ++expert) {
      a.mblock_prefix[expert] = blocks;
      blocks += ceil_div(a.expert_rows[expert], a.shape.tile_m);
    }
    a.mblock_prefix[a.owned_experts] = blocks;
  }
  source_plan_sync<Threads, Barrier>();
}

template <unsigned Threads, unsigned BaseThread = 0U, unsigned Barrier = 0U>
__device__ void build_window_geometry(const WindowGeometryView &a) {
  build_window_counts<Threads, BaseThread, Barrier>(a);
  const auto tid = threadIdx.x - BaseThread;
  for (unsigned expert = 0; expert < a.owned_experts; ++expert) {
    const auto blocks = a.mblock_prefix[expert + 1U] - a.mblock_prefix[expert];
    for (auto block = std::uint64_t(tid); block < blocks; block += Threads)
      a.mblocks[a.mblock_prefix[expert] + block] = {
          expert, static_cast<unsigned>(block * a.shape.tile_m),
          block_rows(a.expert_rows[expert], block, a.shape.tile_m)};
  }
  source_plan_sync<Threads, Barrier>();
}

} // namespace weftmoe::cuda::static_forward
