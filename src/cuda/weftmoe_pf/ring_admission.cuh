#pragma once

namespace weftmoe_pf::ring {

// One static communication warp has no blocking copy or Reduce task.
// It services source and destination capacity independently; reserving a
// source window never depends on the destination's available X/S/C space.
// Lanes own disjoint slot checks/publications, not independent task cursors.
template <bool Network>
__device__ __noinline__ void admit(const ForwardKernelArgs &a, unsigned fc1,
                                   unsigned fc2) {
  const auto &r = a.ring;
  const auto lane = threadIdx.x & 31U;
  const auto source_windows = r.source.local_windows();
  std::uint64_t source_window = 0U, window = 0U;
  while (source_window < source_windows || window < r.state->windows) {
    if (source_window < source_windows) {
      const auto seq = r.state->source_base + source_window;
      bool reserved = false;
      if (lane == 0U) {
        auto &s = r.contributions[seq % r.contribution_depth];
        if (acquire(s.consumed) >= s.retire_target) {
          s.retire_target += static_forward::block_rows(
              r.source.tokens, source_window, r.source.shape.tokens);
          publish<::cuda::thread_scope_system>(s.generation,
                                               seq / r.contribution_depth + 1U);
          reserved = true;
        }
      }
      if (__any_sync(0xffffffffU, reserved)) {
        __syncwarp();
        // Import a monotone reservation head at each local destination. Pull
        // probes its own GPU, not the source GPU once for every activation row.
        // A head may advance past a reader's window, but that window cannot
        // retire until this reader's route has returned and been reduced.
        const auto first = Network ? a.node * a.local_world_size : 0U;
        const auto peers = Network ? a.local_world_size : a.world;
        for (unsigned destination = first + lane; destination < first + peers;
             destination += 32U)
          publish<::cuda::thread_scope_system>(
              peer(a, destination, r.source_reserved)[a.rank], seq + 1U);
        ++source_window;
      }
    }
    if (window < r.state->windows) {
      const auto *prefix = Network ? r.start_prefix : r.window_prefix;
      const auto begin = prefix[window];
      const auto end = prefix[window + 1U];
      bool free = true;
      for (auto index = begin + lane; index < end; index += 32U) {
        const auto block = Network ? r.start_blocks[index] : index;
        auto &s = slot(a, block);
        if (acquire(s.fc1_done) < s.fc1_target ||
            acquire(s.fc2_done) < s.fc2_target ||
            acquire<Network ? ::cuda::thread_scope_system
                            : ::cuda::thread_scope_device>(s.returned) < s.return_target) {
          free = false;
          break;
        }
      }
      if (__all_sync(0xffffffffU, free)) {
        for (auto index = begin + lane; index < end; index += 32U) {
          const auto block = Network ? r.start_blocks[index] : index;
          auto &s = slot(a, block);
          s.fc1_target += fc1;
          s.fc2_target += fc2;
          s.return_target += r.blocks[block].tile.rows;
          publish<Network ? ::cuda::thread_scope_system : ::cuda::thread_scope_device>(
              s.generation, r.blocks[block].generation);
        }
        __syncwarp();
        if (lane == 0U)
          publish(r.state->admitted_windows, window + 1U);
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, lane == 0U, WindowAdmitted,
            Kernel, 0U, window, 0U);
        ++window;
      }
    }
  }
}

} // namespace weftmoe_pf::ring
