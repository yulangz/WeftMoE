#pragma once

namespace static_v4::ring {

// CTA 0's third communication warp has no blocking copy or Reduce task.
// It services source and destination capacity independently; reserving a
// source window never depends on the destination's available X/S/C space.
__device__ __noinline__ void admit(const ForwardKernelArgs &a, unsigned fc1,
                                   unsigned fc2) {
  const auto &r = a.ring;
  std::uint64_t source_window = 0U, window = 0U;
  while (source_window < r.source.local_windows() ||
         window < r.state->windows) {
    if (source_window < r.source.local_windows()) {
      const auto seq = r.state->source_base + source_window;
      auto &s = r.contributions[seq % r.contribution_depth];
      if (acquire(s.consumed) >= s.retire_target) {
        s.retire_target += static_forward::block_rows(
            r.source.tokens, source_window, r.source.shape.tokens);
        publish<::cuda::thread_scope_system>(s.generation,
                                             seq / r.contribution_depth + 1U);
        ++source_window;
      }
    }
    if (window < r.state->windows) {
      const auto begin = r.window_prefix[window];
      const auto end = r.window_prefix[window + 1U];
      bool free = true;
      for (auto block = begin; block < end; ++block) {
        auto &s = slot(a, block);
        if (acquire(s.fc1_done) < s.fc1_target ||
            acquire(s.fc2_done) < s.fc2_target ||
            acquire(s.returned) < s.return_target) {
          free = false;
          break;
        }
      }
      if (free) {
        for (auto block = begin; block < end; ++block) {
          auto &s = slot(a, block);
          s.fc1_target += fc1;
          s.fc2_target += fc2;
          s.return_target += r.blocks[block].tile.rows;
          publish(s.generation, sequence(a, block) / r.depth + 1U);
        }
        ++window;
      }
    }
  }
}

} // namespace static_v4::ring
