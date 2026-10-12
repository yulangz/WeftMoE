#pragma once

// Diagnostic only: replace both GEMMs with data copies while retaining the
// real task stream, X/S lifetime, peer stores, ring admission and RDMA workers.
namespace weftmoe_pf::network {

__device__ __noinline__ void communication_copy(const ForwardKernelArgs &a) {
  using namespace local;
  static_assert(forward_tuning::kMainIntermediate <= forward_tuning::kMainHidden);
  static_assert(forward_tuning::kMainHidden % forward_tuning::kMainIntermediate == 0);
  auto *x = peer_at<weftmoe::transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *s = peer_at<weftmoe::transport::BFloat16>(a.local_base, a.forward_s_offset);
  Schedule schedule(a);
  Task t;
  while (schedule.next(t)) {
    auto &slot = a.ring.slots[t.row / kM];
    while (!ring::admitted(a, t.block)) {}
    const auto &ready = t.fc2 ? slot.fc1_done : slot.x_full;
    const auto target = std::uint64_t(t.generation) * (t.fc2 ? kFc1Tiles : kM);
    while (ring::acquire(ready) < target) {}
    if (!t.fc2)
      while (ring::acquire(slot.fc2_done) <
             std::uint64_t(t.generation - 1U) * kFc2Tiles) {}

    const unsigned width = t.fc2 ? kN : kN / 2U;
    const unsigned vectors = width / 8U;
    for (unsigned i = threadIdx.x; i < t.rows * vectors; i += kMathThreads) {
      const auto row = i / vectors;
      const auto column = t.column * width + (i % vectors) * 8U;
      const weftmoe::transport::BFloat16 *source;
      weftmoe::transport::BFloat16 *destination;
      if (t.fc2) {
        source = s + (t.row + row) * a.intermediate + column % a.intermediate;
        destination = static_cast<weftmoe::transport::BFloat16 *>(
            a.ring.rows[t.block * kM + row].output) + column;
      } else {
        source = x + (t.row + row) * a.hidden + column;
        destination = s + (t.row + row) * a.intermediate + column;
      }
      *reinterpret_cast<uint4 *>(destination) =
          *reinterpret_cast<const uint4 *>(source);
    }
    if (t.fc2)
      __threadfence_system();
    else
      __threadfence();
    collective_scope_sync<kMathThreads, 7U>();
    if (threadIdx.x == 0U) {
      if (t.fc2) {
        arrive(slot.fc2_done, 1U);
        arrive(slot.fc2_published, 1U);
      } else {
        arrive(slot.fc1_done, 1U);
      }
    }
    collective_scope_sync<kMathThreads, 7U>();
  }
}

} // namespace weftmoe_pf::network
