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

// Only immutable source metadata uses rail replicas. Payload and slot control
// addresses retain their own transport-specific visibility/lifetime contracts.
template <class T>
__device__ __forceinline__ T *source_metadata(const ForwardKernelArgs &a,
                                              unsigned rank, T *local_pointer) {
  if (a.use_gin) {
    const auto offset = reinterpret_cast<const char *>(local_pointer) -
                        reinterpret_cast<const char *>(a.ring.source_header);
    return static_v4::source_pointer<T>(a, rank, offset);
  }
  return peer(a, rank, local_pointer);
}

__device__ __forceinline__ static_forward::ComputeSlot &
slot(const ForwardKernelArgs &a, std::uint64_t block) {
  return a.ring.slots[a.ring.blocks[block].slot];
}

__device__ __forceinline__ std::uint64_t row(const ForwardKernelArgs &a,
                                             std::uint64_t block) {
  return std::uint64_t(a.ring.blocks[block].slot) * forward_tuning::kTileM;
}

__device__ __forceinline__ bool admitted(const ForwardKernelArgs &a,
                                         std::uint64_t block) {
  return acquire(slot(a, block).generation) ==
         a.ring.blocks[block].generation;
}

__device__ __forceinline__ const static_forward::WindowSource &
window_source(const ForwardKernelArgs &a, std::uint64_t window, unsigned source) {
  const auto &shape = a.ring.source.shape;
  return a.ring.window_sources[window * shape.sources + source -
                               shape.source_begin(window)];
}

__device__ __forceinline__ std::uint64_t
source_count(const ForwardKernelArgs &a, unsigned source,
             std::uint64_t local_window, unsigned expert) {
  const auto &plan = a.ring.source;
  if (local_window * plan.shape.tokens >= a.ring.source_epochs[source].tokens)
    return 0U;
  const auto *prefix =
      source_metadata(a, source, plan.expert_prefix) +
      local_window * (plan.experts + 1U);
  return prefix[expert + 1U] - prefix[expert];
}

__device__ __forceinline__ void save_identity(const ForwardKernelArgs &a,
                                              std::uint64_t index) {
  const auto &identity = a.ring.rows[index];
  const auto tape_row = a.checkpoint.rows[index];
  const auto expert = a.rank * a.dispatch.sizes.owned_expert_count +
                      a.ring.blocks[index / forward_tuning::kTileM].tile.expert;
  a.checkpoint.gates[tape_row] = identity.gate;
  a.checkpoint.headers[tape_row] = {
      transport::kActivationTransportSchemaVersion, identity.source, a.rank, 0U,
      a.dispatch.identity.generation, execution_epoch(a), identity.route,
      identity.route, identity.token, tape_row, expert,
      identity.route % a.ring.source.shape.top_k};
}

} // namespace static_v4::ring
