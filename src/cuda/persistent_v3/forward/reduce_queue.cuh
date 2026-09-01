#pragma once

// Route completion is naturally distributed: local FC2 producers publish
// directly into the source rank while remote FULL consumers publish after the
// payload has arrived. This helper turns those independent route releases into
// one source-local token-ready queue without a grid rendezvous.

namespace forward_reduce_detail {

__device__ __forceinline__ void publish_ready_route(
    const ForwardKernelArgs &a, void *source_base, std::uint64_t route) {
  if (!a.forward_dynamic_reduce_queue)
    return;

  const auto top_k = a.dispatch.identity.top_k;
  const auto token = route / top_k;
  auto *route_count = peer_at<unsigned int>(
      source_base, a.forward_reduce_route_count_offset);
  ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_system> count(
      route_count[token]);

  // acq_rel RMWs form a release sequence for all route payload/map stores.
  // The final route publisher therefore acquires every earlier publisher
  // before releasing the token queue entry to Reduce.
  const auto previous = count.fetch_add(1U, ::cuda::memory_order_acq_rel);
  if (previous + 1U != top_k)
    return;

  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      source_base, a.forward_v3_control_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> tail(
      control->reduce_ready_tail.value);
  const auto slot = tail.fetch_add(1ULL, ::cuda::memory_order_relaxed);
  if (slot >= a.dispatch.identity.num_tokens)
    return;

  auto *ready_token = peer_at<unsigned long long>(
      source_base, a.forward_reduce_ready_token_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> token_word(
      ready_token[slot]);
  // Zero means unpublished. Generation setup clears the active range before
  // setup_ready releases producers, so token+1 is both the payload and the
  // release marker without a second queue-sized epoch array.
  token_word.store(token + 1ULL, ::cuda::memory_order_release);
}

} // namespace forward_reduce_detail
