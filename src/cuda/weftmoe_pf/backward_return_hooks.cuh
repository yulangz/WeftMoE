#pragma once

// The common GIN sender still owns TX reservations and both NIC credits.
// Only Backward's immutable row interval and RX admission differ.
namespace activation_ring_detail {
template <> struct Hooks<BackwardKernelArgs> : DefaultHooks {
  __device__ __forceinline__ static Transfer returning(
      const BackwardKernelArgs &a, unsigned channel,
      unsigned long long sequence, unsigned long long position) {
    return a.return_ring.depth
        ? weftmoe_pf::backward_return::packet(a, channel, sequence, position)
        : DefaultHooks::returning(a, channel, sequence, position);
  }

  __device__ __forceinline__ static bool receive(
      const BackwardKernelArgs &a, unsigned channel,
      const communication::ChunkHeader *header) {
    return weftmoe_pf::backward_return::receive_reserved(a, channel, header);
  }
};
} // namespace activation_ring_detail
