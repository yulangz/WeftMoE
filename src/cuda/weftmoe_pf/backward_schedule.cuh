#pragma once

#ifndef WEFTMOE_PF_BWD_DYNAMIC_DW
#define WEFTMOE_PF_BWD_DYNAMIC_DW 0
#endif

namespace weftmoe_pf::backward {
// After activation drains, clusters process disjoint dW chunks. Dynamic
// claiming balances experts with different reduction lengths; the default
// retains the static round-robin schedule.
struct Schedule {
  unsigned weight_partition;

  __device__ explicit Schedule(unsigned cluster) : weight_partition(cluster) {}

  __device__ __forceinline__ bool next(const BackwardKernelArgs &a,
                                       unsigned clusters,
                                       pipeline::backward::TileTask &task) {
    using namespace backward_pipeline_detail;
    // A small, fixed chunk keeps each hot expert active across the grid.
    // dW1/dW2 have different output domains and independently tuned chunk
    // sizes.
    constexpr auto claim2 = backward_tuning::kDw2ClaimTiles;
    constexpr auto claim1 = backward_tuning::kDw1ClaimTiles;
    constexpr auto chunks2 = (kDw2Pairs + claim2 - 1U) / claim2;
    constexpr auto chunks1 = (kDw1Pairs + claim1 - 1U) / claim1;
    const auto experts = a.dispatch.sizes.owned_expert_count;
#if WEFTMOE_PF_BWD_DYNAMIC_DW
    while (true) {
      // Reuse the scheduler's cache-line-padded dW2 head for both dW domains.
      // Only the cluster leader claims; existing cluster joins publish the
      // selected task to its peer. prepare_kernel resets the head each call.
      const auto layout = pipeline::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          backward_tuning::kMainHidden / backward_tuning::kDxTileN / 2U,
          a.scheduler_expert_capacity);
      auto *words = scheduler_words(a);
      const auto index = static_cast<unsigned>(atomicAdd(
          reinterpret_cast<unsigned long long *>(words + layout.dw2_head),
          1ULL));
      if (index >= experts * (chunks2 + chunks1))
        return false;
#else
    while (weight_partition < experts * (chunks2 + chunks1)) {
      const auto index = weight_partition;
      weight_partition += clusters;
#endif
      const bool first = index < experts * chunks2;
      const auto claim = first ? claim2 : claim1;
      const auto chunks = first ? chunks2 : chunks1;
      const auto relative = first ? index : index - experts * chunks2;
      const auto expert = relative / chunks;
      const auto tile = relative % chunks * claim;
      if (a.dispatch.expert_row_count[expert]) {
        task = {expert, tile, first ? TaskKind::Dw2 : TaskKind::Dw1,
                min(claim, (first ? kDw2Pairs : kDw1Pairs) - tile)};
        return true;
      }
    }
    return false;
  }

private:
  static constexpr unsigned kDw2Pairs =
      (backward_pipeline_detail::kDw2TilesPerExpert + 1U) / 2U;
  static constexpr unsigned kDw1Pairs =
      (backward_pipeline_detail::kDw1TilesPerExpert + 1U) / 2U;
};

} // namespace weftmoe_pf::backward
