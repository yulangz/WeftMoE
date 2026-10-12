#pragma once

namespace weftmoe_pf::backward {
// Canonical dU/SwiGLU/dX stays on one cluster when the source range fits.
// Bounded source windows use a separate, window-major dX N-pair domain.
// Both have immutable static ownership, without a shared claim queue.
struct Schedule {
  unsigned block, epilogue, dx;
  unsigned weight_partition;

  __device__ explicit Schedule(unsigned cluster)
      : block(cluster), epilogue(cluster), dx(cluster),
        weight_partition(cluster) {}

  template <unsigned DuTiles, unsigned DxTiles, bool WeightsOnly = false,
            class Layout>
  __device__ __forceinline__ bool
  next(const BackwardKernelArgs &a, unsigned long long *words,
       const Layout &layout, unsigned mblocks, unsigned clusters,
       pipeline::backward::TileTask &task, unsigned &phase, bool &terminal) {
    using namespace backward_pipeline_detail;
    if constexpr (!WeightsOnly) {
      const auto dx_blocks =
          a.return_ring.depth ? a.return_ring.state->blocks : mblocks;
      const auto du_owner = [&](unsigned index) {
        return a.return_ring.depth && !a.return_ring.canonical
                   ? a.return_ring.du_order[index]
                   : index;
      };
      // Batch adjacent N tiles over one gathered dAB M-block. Smaller batches
      // expose N parallelism when the bounded window underfills the M grid.
      constexpr auto dx_tiles = backward_tuning::kDxClaimTiles;
      constexpr auto dx_groups = (DxTiles + dx_tiles - 1U) / dx_tiles;
      if (a.return_ring.depth && !a.return_ring.canonical) {
        if (dx < dx_blocks * dx_groups &&
            weftmoe_pf::backward_return::input_ready(a, dx / dx_groups)) {
          const auto tile = (dx % dx_groups) * dx_tiles;
          task = {dx / dx_groups, tile, TaskKind::Dx,
                  min(dx_tiles, DxTiles - tile)};
          dx += clusters;
          return true;
        }
      }
      if (mblocks < clusters) {
        // An underfilled M-block grid needs N/epilogue residues to occupy the
        // remaining clusters. Ownership remains static in each dependency
        // domain.
        const auto ready = [&](unsigned owner, unsigned required) {
          return ::cuda::atomic_ref<unsigned long long,
                                    ::cuda::thread_scope_device>(
                     words[layout.phase + owner])
                     .load(::cuda::memory_order_acquire) >= required;
        };
        if ((!a.return_ring.depth || a.return_ring.canonical) &&
            dx < mblocks * DxTiles && ready(dx / DxTiles, 3U) &&
            (!a.return_ring.canonical ||
             weftmoe_pf::backward_return::slot_free(a, dx / DxTiles))) {
          task = {dx / DxTiles, dx % DxTiles, TaskKind::Dx, 1U};
          dx += clusters;
          return true;
        }
        if (epilogue < mblocks * kDabEpilogueTasks &&
            ready(du_owner(epilogue / kDabEpilogueTasks), 2U)) {
          task = {du_owner(epilogue / kDabEpilogueTasks),
                  epilogue % kDabEpilogueTasks, TaskKind::Dab, 1U};
          phase = 2U;
          epilogue += clusters;
          return true;
        }
        if (block < mblocks * DuTiles && ready(du_owner(block / DuTiles), 1U)) {
          task = {du_owner(block / DuTiles), block % DuTiles, TaskKind::Dab,
                  1U};
          phase = 1U;
          block += clusters;
          return true;
        }
      } else if (block < mblocks) {
        const auto owner = du_owner(block);
        if (a.return_ring.canonical &&
            !weftmoe_pf::backward_return::slot_free(a, owner))
          return false;
        if (!::cuda::atomic_ref<unsigned long long,
                                ::cuda::thread_scope_device>(
                 words[layout.phase + owner])
                 .load(::cuda::memory_order_acquire))
          return false;
        // Canonical Return and the dense fallback also include dX. No
        // task-mailbox exchange is needed between these dependent stages.
        task = {owner, 0U, TaskKind::Dab, 1U};
        phase = 1U;
        block += clusters;
        return true;
      }
      const auto completed =
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
              reinterpret_cast<PaddedCounter *>(words +
                                                layout.completed_dx_tiles)
                  ->value)
              .load(::cuda::memory_order_acquire);
      // An idle activation residue is NOT permission to start a long dW tile.
      // All local dX stores must be complete; Return/Reduce keep progressing in
      // independent communication warps while the clusters execute dW.
      if (completed != std::uint64_t(dx_blocks) * DxTiles)
        return false;
    }
    // A small, fixed chunk keeps each hot expert active across the grid.
    // dW1/dW2 have different output domains and independently tuned chunk
    // sizes.
    constexpr auto claim2 = backward_tuning::kDw2ClaimTiles;
    constexpr auto claim1 = backward_tuning::kDw1ClaimTiles;
    constexpr auto chunks2 = (kDw2Pairs + claim2 - 1U) / claim2;
    constexpr auto chunks1 = (kDw1Pairs + claim1 - 1U) / claim1;
    const auto experts = a.dispatch.sizes.owned_expert_count;
    while (weight_partition < experts * (chunks2 + chunks1)) {
      const auto index = weight_partition;
      weight_partition += clusters;
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
    terminal = true;
    return false;
  }

private:
  static constexpr unsigned kDw2Pairs =
      (backward_pipeline_detail::kDw2TilesPerExpert + 1U) / 2U;
  static constexpr unsigned kDw1Pairs =
      (backward_pipeline_detail::kDw1TilesPerExpert + 1U) / 2U;
};

} // namespace weftmoe_pf::backward
