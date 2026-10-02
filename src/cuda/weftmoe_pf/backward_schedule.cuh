#pragma once

namespace weftmoe_pf::backward {

// Canonical dU/SwiGLU stays on one cluster. Ring dX uses window-major N pairs;
// both domains have immutable static ownership, without a shared claim queue.
struct Schedule {
  unsigned block, epilogue, dx;
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
    const auto dx_blocks = a.return_ring.depth ? a.return_ring.state->blocks : mblocks;
    const auto du_owner = [&](unsigned index) {
      return a.return_ring.depth ? a.return_ring.du_order[index] : index;
    };
    // A bounded window can underfill the M grid even for a long invocation.
    // Stripe N pairs as well; Return waits for every N tile of the M-block.
    constexpr auto dx_pairs = (DxTiles + 1U) / 2U;
    if (a.return_ring.depth && dx < dx_blocks * dx_pairs &&
        weftmoe_pf::backward_return::input_ready(a, dx / dx_pairs)) {
      const auto tile = (dx % dx_pairs) * 2U;
      task = {dx / dx_pairs, tile, TaskKind::Dx, min(2U, DxTiles - tile)};
      dx += clusters;
      return true;
    }
    if (mblocks < clusters) {
      // An underfilled M-block grid needs N/epilogue residues to occupy the
      // remaining clusters. Ownership remains static in each dependency domain.
      const auto ready = [&](unsigned owner, unsigned required) {
        return ::cuda::atomic_ref<unsigned long long,
                                  ::cuda::thread_scope_device>(
                   words[layout.phase + owner])
                   .load(::cuda::memory_order_acquire) >= required;
      };
      if (!a.return_ring.depth && dx < mblocks * DxTiles && ready(dx / DxTiles, 3U)) {
        task = {dx / DxTiles, dx % DxTiles, TaskKind::Dx, 1U};
        dx += clusters;
        return true;
      }
      if (epilogue < mblocks * kDabEpilogueTasks &&
          ready(du_owner(epilogue / kDabEpilogueTasks), 2U)) {
        task = {du_owner(epilogue / kDabEpilogueTasks), epilogue % kDabEpilogueTasks,
                TaskKind::Dab, 1U};
        phase = 2U;
        epilogue += clusters;
        return true;
      }
      if (block < mblocks * DuTiles && ready(du_owner(block / DuTiles), 1U)) {
        task = {du_owner(block / DuTiles), block % DuTiles, TaskKind::Dab, 1U};
        phase = 1U;
        block += clusters;
        return true;
      }
    } else if (block < mblocks) {
      const auto owner = du_owner(block);
      if (!::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
               words[layout.phase + owner])
               .load(::cuda::memory_order_acquire))
        return false;
      // Keep dU/SwiGLU together; the dense fallback also includes dX. No
      // task-mailbox exchange is needed between these dependent stages.
      task = {owner, 0U, TaskKind::Dab, 1U};
      phase = 1U;
      block += clusters;
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
    if (completed != std::uint64_t(dx_blocks) * DxTiles)
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
