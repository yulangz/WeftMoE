#pragma once

// Diagnostic only: preserve production dispatch, TX credit, return and reduce,
// replacing the expert Jacobian by identity. No GEMM or activation is executed.
// The extra GM read models a ready FC2 output; it is not free in this test.
namespace weftmoe_pf::backward_stream {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
template <bool Network>
__device__ __forceinline__ void identity(const ForwardKernelArgs &f,
                                        const BackwardKernelArgs &a) {
  using Scheduler = std::conditional_t<Network, NetworkSchedule, Schedule>;
  using Sync = backward_unified_compute_detail::UnifiedComputeSynchronization<true>;
  Scheduler schedule(f, a);
  Task t;
  while (schedule.next(t)) {
    if (t.phase != Phase::Dx)
      continue;
    auto &slot = f.ring.slots[t.row / kM];
    if (!threadIdx.x) {
      if constexpr (Network) {
        const auto *ready = backward_pipeline_detail::scheduler_words(a) +
                            directory_layout(a).dispatch_arrived;
        while (ring::acquire(*reinterpret_cast<const std::uint64_t *>(
                   ready + t.block)) < t.rows ||
               ring::acquire(slot.returned) < t.generation - 1U) {}
      } else {
        while (ring::acquire(slot.x_full) < std::uint64_t(t.generation) * kM) {}
      }
    }
    Sync::math_sync(256U, 0U);
    const auto input_row = Network ? directory(a)[t.block].compact_row : t.row;
    const auto *input = Network ? a.dr : peer_at<const transport::BFloat16>(
                                            f.local_base, f.canonical_rows_offset);
    for (unsigned v = threadIdx.x; v < t.rows * kDxN / 8U; v += 256U) {
      const auto row = v / (kDxN / 8U), column = v % (kDxN / 8U) * 8U;
      auto *output = Network ? dx_output(f, a, t, row)
                            : static_cast<transport::BFloat16 *>(
                                  f.ring.rows[t.block * kM + row].output);
      const auto col = t.column * kDxN + column;
      *reinterpret_cast<uint4 *>(output + col) = *reinterpret_cast<const uint4 *>(
          input + std::uint64_t(input_row + row) * f.hidden + col);
    }
    if (!t.column)
      for (unsigned row = threadIdx.x; row < t.rows; row += 256U) {
        if constexpr (Network)
          *gradient_output(f, a, t, row) = 0.0F;
        else {
          const auto id = f.ring.rows[t.block * kM + row];
          a.peer_payloads[id.source].dgate[id.route] = 0.0F;
        }
      }
    __threadfence_system();
    Sync::math_sync(256U, 0U);
    if (!threadIdx.x) {
      ring::arrive(slot.fc2_done, 1U);
      if constexpr (Network)
        ring::arrive(slot.fc2_published, 1U);
    }
  }
}
#endif
} // namespace weftmoe_pf::backward_stream
