#pragma once

#include "cuda/communication/sequence.hpp"
#include "cuda/static_v4/plan.hpp"
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define WEFTMOE_STATIC_RING_HD __host__ __device__
#else
#define WEFTMOE_STATIC_RING_HD
#endif

namespace weftmoe::cuda::static_forward {

WEFTMOE_STATIC_RING_HD constexpr std::size_t align_payload(std::size_t bytes) {
  return (bytes + 127U) & ~std::size_t{127U};
}

struct PacketLayout {
  std::size_t payload;
  std::size_t bytes;

  // Frozen directories already identify every token/route. The wire packet
  // only needs its generation/retirement header and the actual activation rows.
  WEFTMOE_STATIC_RING_HD static constexpr PacketLayout
  planned(std::uint32_t rows, std::uint32_t hidden) {
    const auto payload = align_payload(sizeof(PacketHeader));
    return {payload,
            align_payload(payload + std::size_t(rows) * hidden * 2U)};
  }
};

// GIN signals are accessed through GIN only. visible_generation imports NIC
// readiness for node-local consumers; their release RMWs target consumed_total.
// Submit/ACK owners keep their own targets here, outside the wire payload.
struct alignas(64) NetworkSlot {
  std::uint64_t ready_signal = 0;
  std::uint64_t credit_signal = 0;
  std::uint64_t visible_generation = 0;
  std::uint64_t consumed_total = 0;
  std::uint64_t retire_target = 0;
  std::uint64_t local_target = 0;
  std::uint64_t acknowledged_generation = 0;
};
static_assert(sizeof(NetworkSlot) == 64U);

// Bases are immutable during one invocation. Separate fixed owners advance
// TX, NIC-import and Return-scatter cursors; only terminal drain advances bases.
struct alignas(64) NetworkChannel {
  std::uint64_t tx_base;
  std::uint64_t rx_base;
  std::uint64_t tx_cursor;
  std::uint64_t rx_cursor;
  std::uint64_t consume_cursor;
};

struct SlotSequence {
  std::uint64_t sequence;
  std::uint32_t depth;

  WEFTMOE_STATIC_RING_HD constexpr std::uint32_t slot() const {
    return static_cast<std::uint32_t>(sequence % depth);
  }
  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t generation() const {
    return sequence / depth;
  }
  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t ready() const {
    return generation() + 1U;
  }
  WEFTMOE_STATIC_RING_HD constexpr bool
  reusable(std::uint64_t credit, std::uint64_t local_done,
           std::uint64_t local_target) const {
    return communication::reached(credit, generation(), 64U) &&
           communication::reached(local_done, local_target);
  }
};

// Private to the unique publisher of this physical slot. No atomic expected
// count, no reset on wrap, and no assumption that successive packets have the
// same number of readers. The returned target travels with the new generation.
struct Retirement {
  std::uint64_t target = 0;

  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t
  publish(std::uint32_t expected) {
    target += expected;
    return target;
  }
};

// These are owner-private capacity cursors, not task-claim atomics. The owner
// checks every required plane before advancing any head, and advances tail
// only across a contiguous prefix of individually retired slots.
struct RingCapacity {
  std::uint64_t head = 0;
  std::uint64_t tail = 0;
  std::uint32_t depth;

  WEFTMOE_STATIC_RING_HD constexpr bool fits(std::uint64_t count) const {
    return count <= depth && head - tail <= depth - count;
  }
};

// All offsets are relative to one rail's registered window. The host validates
// dimensions/byte bounds before creating it. Self traffic bypasses this layout.
struct NetworkRings {
  std::uint32_t nodes;
  std::uint32_t lanes;
  std::uint32_t depth;
  std::size_t dispatch_stride;
  std::size_t return_stride;

  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t channels() const {
    return std::uint64_t(nodes - 1U) * lanes;
  }
  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t slots() const {
    return channels() * depth;
  }
  WEFTMOE_STATIC_RING_HD constexpr std::uint64_t
  channel(std::uint32_t self, std::uint32_t peer, std::uint32_t lane) const {
    const auto compact_peer = peer - (peer > self ? 1U : 0U);
    return std::uint64_t(compact_peer) * lanes + lane;
  }
  WEFTMOE_STATIC_RING_HD constexpr std::uint32_t
  peer(std::uint32_t self, std::uint64_t channel_id) const {
    const auto compact = static_cast<std::uint32_t>(channel_id / lanes);
    return compact + (compact >= self);
  }
  WEFTMOE_STATIC_RING_HD constexpr std::size_t
  state(bool returning, std::uint64_t channel_id) const {
    return ((returning ? channels() : 0U) + channel_id) * sizeof(NetworkChannel);
  }
  WEFTMOE_STATIC_RING_HD constexpr std::size_t
  control(bool returning, std::uint64_t channel_id,
          std::uint64_t sequence) const {
    return 2U * channels() * sizeof(NetworkChannel) +
           ((returning ? slots() : 0U) + channel_id * depth +
            sequence % depth) * sizeof(NetworkSlot);
  }
  WEFTMOE_STATIC_RING_HD constexpr std::size_t data_begin() const {
    return align_payload(2U * channels() * sizeof(NetworkChannel) +
                         2U * slots() * sizeof(NetworkSlot));
  }
  WEFTMOE_STATIC_RING_HD constexpr std::size_t
  payload(bool returning, bool receive, std::uint64_t channel_id,
          std::uint64_t sequence) const {
    const auto stride = returning ? return_stride : dispatch_stride;
    const auto base = data_begin() +
                      (returning ? 2U * slots() * dispatch_stride : 0U) +
                      (receive ? slots() * stride : 0U);
    return base + (channel_id * depth + sequence % depth) * stride;
  }
  WEFTMOE_STATIC_RING_HD constexpr std::size_t bytes() const {
    return data_begin() + 2U * slots() * (dispatch_stride + return_stride);
  }
};

} // namespace weftmoe::cuda::static_forward

#undef WEFTMOE_STATIC_RING_HD
