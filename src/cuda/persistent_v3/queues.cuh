#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuda/persistent_v3/epoch_state.cuh"

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

namespace weftmoe::cuda::persistent_v3 {

template <class T> struct SpscSlot {
  unsigned long long sequence = 0;
  T payload{};
};

template <class T, std::uint32_t Capacity> struct alignas(128) SpscRing {
  static_assert(Capacity >= 2U);
  static_assert((Capacity & (Capacity - 1U)) == 0U,
                "SPSC capacity must be a power of two");

  PaddedCounter producer_ticket;
  PaddedCounter consumer_ticket;
  SpscSlot<T> slots[Capacity];
};

#if defined(__CUDACC__)
template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline unsigned long long
load_acquire(unsigned long long &word) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.load(::cuda::memory_order_acquire);
}

template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline unsigned long long
load_relaxed(unsigned long long &word) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.load(::cuda::memory_order_relaxed);
}

template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline void store_release(unsigned long long &word,
                                             unsigned long long next) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  value.store(next, ::cuda::memory_order_release);
}

template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline void store_relaxed(unsigned long long &word,
                                             unsigned long long next) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  value.store(next, ::cuda::memory_order_relaxed);
}

template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline unsigned long long
fetch_add_release(unsigned long long &word, unsigned long long increment) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.fetch_add(increment, ::cuda::memory_order_release);
}

template <::cuda::thread_scope Scope>
WEFTMOE_V3_DEVICE inline bool compare_exchange_acq_rel(
    unsigned long long &word, unsigned long long &expected,
    unsigned long long desired) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.compare_exchange_strong(expected, desired,
                                       ::cuda::memory_order_acq_rel,
                                       ::cuda::memory_order_acquire);
}

template <class T, std::uint32_t Capacity>
WEFTMOE_V3_DEVICE inline bool try_push(SpscRing<T, Capacity> &ring,
                                       const T &payload) {
  const auto ticket = load_relaxed<::cuda::thread_scope_device>(
      ring.producer_ticket.value);
  auto &slot = ring.slots[ticket & (Capacity - 1U)];
  if (load_acquire<::cuda::thread_scope_device>(slot.sequence) != ticket)
    return false;
  slot.payload = payload;
  store_release<::cuda::thread_scope_device>(slot.sequence, ticket + 1ULL);
  store_relaxed<::cuda::thread_scope_device>(ring.producer_ticket.value,
                                           ticket + 1ULL);
  return true;
}

template <class T, std::uint32_t Capacity>
WEFTMOE_V3_DEVICE inline bool try_pop(SpscRing<T, Capacity> &ring,
                                      T &payload) {
  const auto ticket = load_relaxed<::cuda::thread_scope_device>(
      ring.consumer_ticket.value);
  auto &slot = ring.slots[ticket & (Capacity - 1U)];
  if (load_acquire<::cuda::thread_scope_device>(slot.sequence) != ticket + 1ULL)
    return false;
  payload = slot.payload;
  store_release<::cuda::thread_scope_device>(slot.sequence,
                                            ticket + Capacity);
  store_relaxed<::cuda::thread_scope_device>(ring.consumer_ticket.value,
                                           ticket + 1ULL);
  return true;
}

template <class T, std::uint32_t Capacity>
WEFTMOE_V3_DEVICE inline bool ring_drained(SpscRing<T, Capacity> &ring) {
  return load_acquire<::cuda::thread_scope_device>(
             ring.producer_ticket.value) ==
         load_acquire<::cuda::thread_scope_device>(
             ring.consumer_ticket.value);
}
#endif

} // namespace weftmoe::cuda::persistent_v3
