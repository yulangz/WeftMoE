#pragma once

namespace static_v4::ring {

template <::cuda::thread_scope Scope = ::cuda::thread_scope_device>
__device__ __forceinline__ std::uint64_t acquire(const std::uint64_t &word) {
  return ::cuda::atomic_ref<std::uint64_t, Scope>(
             const_cast<std::uint64_t &>(word))
      .load(::cuda::memory_order_acquire);
}

template <::cuda::thread_scope Scope = ::cuda::thread_scope_device>
__device__ __forceinline__ void publish(std::uint64_t &word,
                                        std::uint64_t value) {
  ::cuda::atomic_ref<std::uint64_t, Scope>(word).store(
      value, ::cuda::memory_order_release);
}

template <::cuda::thread_scope Scope = ::cuda::thread_scope_device>
__device__ __forceinline__ std::uint64_t arrive(std::uint64_t &word,
                                                std::uint64_t value) {
  return ::cuda::atomic_ref<std::uint64_t, Scope>(word).fetch_add(
      value, ::cuda::memory_order_acq_rel);
}

template <class T>
__device__ __forceinline__ T *peer(const ForwardKernelArgs &a, unsigned rank,
                                   T *local_pointer) {
  const auto offset = reinterpret_cast<const char *>(local_pointer) -
                      static_cast<const char *>(a.local_base);
  return reinterpret_cast<T *>(static_cast<char *>(a.peer_bases[rank]) +
                               offset);
}

__device__ __forceinline__ std::uint64_t sequence(const ForwardKernelArgs &a,
                                                  std::uint64_t block) {
  return a.ring.state->block_base + block;
}

__device__ __forceinline__ static_forward::ComputeSlot &
slot(const ForwardKernelArgs &a, std::uint64_t block) {
  return a.ring.slots[sequence(a, block) % a.ring.depth];
}

__device__ __forceinline__ std::uint64_t row(const ForwardKernelArgs &a,
                                             std::uint64_t block) {
  return sequence(a, block) % a.ring.depth * forward_tuning::kTileM;
}

__device__ __forceinline__ bool admitted(const ForwardKernelArgs &a,
                                         std::uint64_t block) {
  return acquire(slot(a, block).generation) ==
         sequence(a, block) / a.ring.depth + 1U;
}

__device__ __forceinline__ std::uint64_t
source_sequence(const ForwardKernelArgs &a, unsigned source,
                std::uint64_t token) {
  return a.ring.source_epochs[source].base + token / a.ring.source.shape.tokens;
}

__device__ __forceinline__ std::uint64_t
contribution_row(const ForwardKernelArgs &a, unsigned source,
                 std::uint64_t route) {
  const auto &shape = a.ring.source.shape;
  const auto token = route / shape.top_k;
  return source_sequence(a, source, token) % a.ring.contribution_depth *
             a.ring.contribution_stride +
         (token % shape.tokens) * shape.top_k + route % shape.top_k;
}

__device__ __forceinline__ std::uint64_t
source_count(const ForwardKernelArgs &a, unsigned source,
             std::uint64_t local_window, unsigned expert) {
  const auto &plan = a.ring.source;
  if (local_window * plan.shape.tokens >= a.ring.source_epochs[source].tokens)
    return 0U;
  const auto *prefix =
      peer(a, source, plan.expert_prefix) + local_window * (plan.experts + 1U);
  return prefix[expert + 1U] - prefix[expert];
}

} // namespace static_v4::ring
