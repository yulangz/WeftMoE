#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuda/pipeline/state.cuh"

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

namespace weftmoe::cuda::pipeline {

#if defined(__CUDACC__)
template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline unsigned long long
load_acquire(unsigned long long &word) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.load(::cuda::memory_order_acquire);
}

template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline unsigned long long
load_relaxed(unsigned long long &word) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.load(::cuda::memory_order_relaxed);
}

template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline void store_release(unsigned long long &word,
                                                  unsigned long long next) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  value.store(next, ::cuda::memory_order_release);
}

template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline void store_relaxed(unsigned long long &word,
                                                  unsigned long long next) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  value.store(next, ::cuda::memory_order_relaxed);
}

template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline unsigned long long
fetch_add_release(unsigned long long &word, unsigned long long increment) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.fetch_add(increment, ::cuda::memory_order_release);
}

template <::cuda::thread_scope Scope>
WEFTMOE_PIPELINE_DEVICE inline bool
compare_exchange_acq_rel(unsigned long long &word, unsigned long long &expected,
                         unsigned long long desired) {
  ::cuda::atomic_ref<unsigned long long, Scope> value(word);
  return value.compare_exchange_strong(expected, desired,
                                       ::cuda::memory_order_acq_rel,
                                       ::cuda::memory_order_acquire);
}

#endif

} // namespace weftmoe::cuda::pipeline
