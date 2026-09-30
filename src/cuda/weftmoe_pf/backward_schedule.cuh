#pragma once

namespace weftmoe_pf::backward {

// Keep an M-block's activation-gradient chain on one cluster. This preserves
// locality and avoids dependencies between independently striped N/row tasks.
struct Schedule {
  unsigned block, epilogue, dx, step = 0U;
  unsigned weight = 0U, weight_end = 0U;
  bool weights_initialized = false;

  __device__ explicit Schedule(unsigned cluster)
      : block(cluster), epilogue(cluster), dx(cluster) {}

  template <unsigned DuTiles, unsigned DxTiles, class Layout>
  __device__ __forceinline__ bool
  next(const BackwardKernelArgs &a, unsigned long long *words,
       const Layout &layout, unsigned mblocks, unsigned clusters,
       persistent_v3::backward::TileTask &task, unsigned &phase,
       bool &terminal) {
    using namespace backward_pipeline_detail;
    if (mblocks < clusters) {
      // An underfilled M-block grid needs N/epilogue residues to occupy the
      // remaining clusters. Ownership remains static in each dependency domain.
      const auto ready = [&](unsigned owner, unsigned required) {
        return ::cuda::atomic_ref<unsigned long long,
                                  ::cuda::thread_scope_device>(
                   words[layout.phase + owner])
                   .load(::cuda::memory_order_acquire) >= required;
      };
      if (dx < mblocks * DxTiles && ready(dx / DxTiles, 3U)) {
        task = {dx / DxTiles, dx % DxTiles, TaskKind::Dx, 1U};
        dx += clusters;
        return true;
      }
      if (epilogue < mblocks * kDabEpilogueTasks &&
          ready(epilogue / kDabEpilogueTasks, 2U)) {
        task = {epilogue / kDabEpilogueTasks, epilogue % kDabEpilogueTasks,
                TaskKind::Dab, 1U};
        phase = 2U;
        epilogue += clusters;
        return true;
      }
      if (block < mblocks * DuTiles && ready(block / DuTiles, 1U)) {
        task = {block / DuTiles, block % DuTiles, TaskKind::Dab, 1U};
        phase = 1U;
        block += clusters;
        return true;
      }
    } else if (block < mblocks) {
      if (step < DuTiles) {
        if (!step && !::cuda::atomic_ref<unsigned long long,
                                         ::cuda::thread_scope_device>(
                          words[layout.phase + block])
                          .load(::cuda::memory_order_acquire))
          return false;
        task = {block, step++, TaskKind::Dab, 1U};
        phase = 1U;
      } else if (step == DuTiles) {
        task = {block, 0U, TaskKind::Dab, kDabEpilogueTasks};
        phase = 2U;
        ++step;
      } else {
        task = {block, 0U, TaskKind::Dx, DxTiles};
        block += clusters;
        step = 0U;
      }
      return true;
    }
    const auto completed =
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
            reinterpret_cast<PaddedCounter *>(words + layout.completed_dx_tiles)
                ->value)
            .load(::cuda::memory_order_acquire);
    // An idle activation residue is NOT permission to start a long dW tile.
    // All local dX stores must be complete; Return/Reduce keep progressing in
    // independent communication warps while the clusters execute dW.
    if (completed != std::uint64_t(mblocks) * DxTiles)
      return false;
    const auto *rows = words + layout.dw_reduction_rows;
    const auto experts = a.dispatch.sizes.owned_expert_count;
    if (!weights_initialized) {
      std::uint64_t work = 0U;
      for (unsigned expert = 0; expert < experts; ++expert)
        work += rows[expert] * (kDw2Pairs + kDw1Pairs);
      const auto cluster = blockIdx.x / 2U;
      weight = weight_cut(rows, experts, work * cluster / clusters);
      weight_end = weight_cut(rows, experts, work * (cluster + 1U) / clusters);
      weights_initialized = true;
    }
    if (weight_task(a, task))
      return true;
    terminal = true;
    return false;
  }

private:
  static constexpr unsigned kDw2Pairs =
      (backward_pipeline_detail::kDw2TilesPerExpert + 1U) / 2U;
  static constexpr unsigned kDw1Pairs =
      (backward_pipeline_detail::kDw1TilesPerExpert + 1U) / 2U;

  // A pair costs one expert's padded K extent. Partition the combined dW2/dW1
  // stream by that work, rounding both adjacent cuts to the same tile boundary.
  // Empty experts cost nothing; skewed experts still spread across clusters.
  __device__ __forceinline__ static unsigned
  weight_cut(const unsigned long long *rows, unsigned experts,
             std::uint64_t work) {
    unsigned base = 0U;
    for (unsigned domain = 0; domain < 2U; ++domain) {
      const auto pairs = domain ? kDw1Pairs : kDw2Pairs;
      for (unsigned expert = 0; expert < experts; ++expert) {
        const auto extent = rows[expert] * pairs;
        if (work <= extent && rows[expert])
          return base + unsigned((work + rows[expert] - 1U) / rows[expert]);
        work -= extent;
        base += pairs;
      }
    }
    return base;
  }

  __device__ __forceinline__ bool
  weight_task(const BackwardKernelArgs &a,
              persistent_v3::backward::TileTask &task) {
    using namespace backward_pipeline_detail;
    const auto split = a.dispatch.sizes.owned_expert_count * kDw2Pairs;
    while (weight < weight_end) {
      const bool first = weight < split;
      const auto pairs = first ? kDw2Pairs : kDw1Pairs;
      const auto index = first ? weight : weight - split;
      const auto owner = index / pairs, tile = index % pairs;
      const auto batch = first ? backward_tuning::kDw2ClaimTiles
                               : backward_tuning::kDw1ClaimTiles;
      const auto count = min(batch, min(pairs - tile, weight_end - weight));
      weight += count;
      if (a.dispatch.expert_row_count[owner]) {
        task = {owner, tile, first ? TaskKind::Dw2 : TaskKind::Dw1, count};
        return true;
      }
    }
    return false;
  }
};

} // namespace weftmoe_pf::backward
