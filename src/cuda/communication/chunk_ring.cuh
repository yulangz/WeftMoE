#pragma once

#include <cstddef>
#include <cstdint>
#include "cuda/communication/sequence.hpp"
#include <cuda/atomic>
#include "cuda/communication/tuning_config.hpp"
#include "cuda/communication/nccl_device_assert_trap.cuh"

namespace weftmoe::cuda::communication {

inline constexpr unsigned kChunkRows = WEFTMOE_GIN_CHUNK_ROWS;
inline constexpr unsigned kRingDepth = WEFTMOE_GIN_RING_DEPTH;
inline constexpr unsigned kDispatchChannels = WEFTMOE_GIN_DISPATCH_CHANNELS;
inline constexpr unsigned kReturnPorts = 2;

// A credit releases exactly one slot. Independent consumers may finish in
// any order: neither credit nor local completion is interpreted as a prefix.
struct alignas(64) ChunkSlot {
  std::uint64_t ready_signal;
  std::uint64_t credit_signal;
  unsigned long long consumed;
  unsigned long long acknowledged;
  std::uint64_t local_target; // This slot only, on its current sender context.
};

struct alignas(128) ChunkChannel {
  unsigned long long sent;
  unsigned long long claimed;
  unsigned long long tx_cursor;
  unsigned long long tx_end;
  unsigned int tx_terminal;
  unsigned int rx_terminal;
  // Reuses header padding without changing slot/control offsets. Like the
  // sequence counters, this bitmap survives operation prepare and Graph
  // replay. New windows are zeroed once; successful drain consumes every bit.
  unsigned long long consumed_slots;
  ChunkSlot slots[kRingDepth];
};
static_assert(kRingDepth <= 64U);
static_assert(sizeof(ChunkSlot) == 64U);
static_assert(offsetof(ChunkChannel, consumed_slots) == 40U);
static_assert(offsetof(ChunkChannel, slots) == 64U);
static_assert(sizeof(ChunkChannel) == 128U + kRingDepth * sizeof(ChunkSlot));

struct alignas(16) ChunkHeader {
  unsigned int rows;
  unsigned int terminal;
  std::uint64_t reserved;
};

struct alignas(16) DispatchRow {
  std::uint64_t token;
  unsigned int fanouts;
  unsigned int reserved;
};

struct alignas(16) ReturnRow {
  std::uint64_t route;
  float dgate;
  unsigned int reserved;
};

// Offsets depend only on topology, hidden width, top-k and ring depth. Token
// counts, routing skew and the number of retained compute arenas are absent.
struct ChunkRingLayout {
  std::size_t control = 0;
  std::size_t dispatch_tx = 0;
  std::size_t dispatch_rx = 0;
  std::size_t return_tx = 0;
  std::size_t return_rx = 0;
  std::size_t dispatch_stride = 0;
  std::size_t return_stride = 0;
  std::size_t dispatch_row_bytes = 0;
  std::size_t return_row_bytes = 0;
  unsigned int counter_base = 2;
  unsigned int dispatch_channels = 0;
  unsigned int return_channels = 0;

  __host__ __device__ unsigned int channels() const {
    return dispatch_channels + return_channels;
  }
  __host__ __device__ std::size_t slot_control(unsigned int channel,
                                               unsigned int slot) const {
    return control + channel * sizeof(ChunkChannel) +
           offsetof(ChunkChannel, slots) + slot * sizeof(ChunkSlot);
  }
  __host__ __device__ std::size_t payload(unsigned int channel,
                                          unsigned long long sequence,
                                          bool receive) const {
    const bool dispatch = channel < dispatch_channels;
    const auto index = dispatch ? channel : channel - dispatch_channels;
    const auto stride = dispatch ? dispatch_stride : return_stride;
    const auto base = dispatch ? (receive ? dispatch_rx : dispatch_tx)
                               : (receive ? return_rx : return_tx);
    return base + (index * kRingDepth + sequence % kRingDepth) * stride;
  }
};

__device__ __forceinline__ ChunkChannel *channels(void *base,
                                                  ChunkRingLayout layout) {
  return reinterpret_cast<ChunkChannel *>(static_cast<char *>(base) +
                                          layout.control);
}

__device__ __forceinline__ unsigned counter(ChunkRingLayout layout,
                                            unsigned channel, unsigned slot) {
  // Counters 0/1 belong to bootstrap and terminal-status collectives.
  return layout.counter_base + channel * kRingDepth + slot;
}

__device__ __forceinline__ bool
can_send(const ncclGin &gin, ncclWindow_t window, void *base, ChunkRingLayout layout,
         unsigned channel, unsigned long long sequence) {
  const auto slot = unsigned(sequence % kRingDepth);
  const auto previous = sequence / kRingDepth;
  return reached(gin.readSignal(window, layout.slot_control(channel, slot) +
                                    offsetof(ChunkSlot, credit_signal)), previous, 64U) &&
         reached(gin.readCounter(counter(layout, channel, slot)),
                 channels(base, layout)[channel].slots[slot].local_target);
}

__device__ __forceinline__ void send(const ncclGin &gin, ncclWindow_t window,
                                     void *base, ChunkRingLayout layout,
                                     unsigned channel, unsigned remote_channel,
                                     unsigned peer, std::size_t bytes) {
  auto &state = channels(base, layout)[channel];
  const auto sequence = state.sent;
  const auto slot = unsigned(sequence % kRingDepth);
  gin.put(ncclTeamWorld(gin.comm), peer, window,
          layout.payload(remote_channel, sequence, true), window,
          layout.payload(channel, sequence, false), bytes,
          ncclGin_StrongVASignalAdd{window,
                                    layout.slot_control(remote_channel, slot) +
                                        offsetof(ChunkSlot, ready_signal),
                                    1},
          ncclGin_WeakCounterInc{counter(layout, channel, slot)},
          ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_block,
          ::cuda::thread_scope_system);
  ++state.slots[slot].local_target;
  ++state.sent;
}

// Only the claim CAS is shared between consumers, and it addresses local
// memory. Acquire the NIC signal before publishing ownership to the CTA.
__device__ __forceinline__ bool claim(const ncclGin &gin, ncclWindow_t window,
                                      void *base, ChunkRingLayout layout,
                                      unsigned channel,
                                      unsigned long long &sequence) {
  auto &state = channels(base, layout)[channel];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
      state.claimed);
  sequence = head.load(::cuda::memory_order_relaxed);
  const auto slot = unsigned(sequence % kRingDepth);
  if (gin.readSignal(window, layout.slot_control(channel, slot) +
                                 offsetof(ChunkSlot, ready_signal)) <
      sequence / kRingDepth + 1ULL)
    return false;
  return head.compare_exchange_strong(sequence, sequence + 1ULL,
                                      ::cuda::memory_order_relaxed);
}

__device__ __forceinline__ void consumed(void *base, ChunkRingLayout layout,
                                         unsigned channel,
                                         unsigned long long sequence) {
  auto &state = channels(base, layout)[channel];
  auto &slot = state.slots[sequence % kRingDepth];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
      slot.consumed)
      .store(sequence / kRingDepth + 1ULL, ::cuda::memory_order_release);
  // Every RX/TMA reader has retired before this call. Publish its slot only
  // after consumed; the ACK owner's acquire exchange imports that completion.
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
      state.consumed_slots)
      .fetch_or(1ULL << (sequence % kRingDepth), ::cuda::memory_order_release);
}

// Called by the unique legacy GIN owner or under the unified context lease.
// Copy workers cannot return credit until every TMA/NVLink reader has retired.
__device__ __forceinline__ bool
acknowledge(const ncclGin &gin, ncclWindow_t window, void *base,
            ChunkRingLayout layout, unsigned channel, unsigned remote_channel,
            unsigned peer) {
  auto &state = channels(base, layout)[channel];
  bool progressed = false;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
      state.consumed_slots);
  // Empty polls are read-only. A completion racing this zero observation is
  // retained in the bitmap and serviced on a later pass.
  if (ready.load(::cuda::memory_order_relaxed) == 0ULL)
    return false;
  auto dirty = ready.exchange(0ULL, ::cuda::memory_order_acquire);
  while (dirty != 0ULL) {
    const unsigned i = unsigned(__ffsll(static_cast<long long>(dirty)) - 1);
    dirty &= dirty - 1ULL;
    auto &slot = state.slots[i];
    // One bit represents one credit. Before this credit is issued, the
    // sender cannot reuse this physical slot, so its next completion cannot
    // coalesce into the captured bit. A completion after the exchange ORs
    // into the next global mask, even while this ACK batch is still running.
    gin.signal(
        ncclTeamWorld(gin.comm), peer,
        ncclGin_StrongVASignalAdd{window,
                                  layout.slot_control(remote_channel, i) +
                                      offsetof(ChunkSlot, credit_signal),
                                  1},
        ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_thread,
        ::cuda::thread_scope_system);
    ++slot.acknowledged;
    progressed = true;
  }
  return progressed;
}

__device__ __forceinline__ bool drained(const ncclGin &gin, ncclWindow_t window,
                                        void *base, ChunkRingLayout layout,
                                        unsigned channel) {
  const auto &state = channels(base, layout)[channel];
  if (!state.tx_terminal ||
      !::cuda::atomic_ref<unsigned, ::cuda::thread_scope_device>(
           const_cast<unsigned &>(state.rx_terminal))
           .load(::cuda::memory_order_acquire))
    return false;
  const auto claimed =
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
          const_cast<unsigned long long &>(state.claimed))
          .load(::cuda::memory_order_acquire);
  for (unsigned i = 0; i < kRingDepth; ++i) {
    const auto sent = (state.sent + kRingDepth - 1U - i) / kRingDepth;
    const auto received = (claimed + kRingDepth - 1U - i) / kRingDepth;
    if (gin.readSignal(window, layout.slot_control(channel, i) +
                                   offsetof(ChunkSlot, credit_signal)) < sent ||
        state.slots[i].acknowledged < received)
      return false;
  }
  return true;
}

static __global__ void initialize(ncclDevComm_t dev, ncclWindow_t window,
                                  ChunkRingLayout layout, unsigned contexts) {
  const auto context = blockIdx.x;
  if (threadIdx.x || context >= contexts)
    return;
  const ncclGin gin(dev, context);
  for (unsigned c = 0; c < layout.channels(); ++c)
    for (unsigned slot = 0; slot < kRingDepth; ++slot) {
      gin.resetCounter(counter(layout, c, slot));
      if (c % contexts == context) {
        gin.resetSignal(window, layout.slot_control(c, slot) +
                                    offsetof(ChunkSlot, ready_signal));
        gin.resetSignal(window, layout.slot_control(c, slot) +
                                    offsetof(ChunkSlot, credit_signal));
      }
    }
}

} // namespace weftmoe::cuda::communication
