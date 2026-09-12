#pragma once

// Source-local receive workers publish after the contribution has arrived.
// These independent route releases feed a token-ready queue without a grid
// rendezvous or a returned-value atomic on a peer GPU.

namespace forward_reduce_detail {

__device__ __forceinline__ void
publish_ready_route(const ForwardKernelArgs &a, std::uint64_t route,
                    std::uint32_t publisher_kind) {
  if (!a.forward_dynamic_reduce_queue) {
    auto *ready =
        peer_at<unsigned long long>(a.local_base, a.return_ready_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        ready[route])
        .store(execution_epoch(a), ::cuda::memory_order_release);
    return;
  }

  const auto top_k = a.dispatch.identity.top_k;
  const auto token = route / top_k;
  auto *route_count =
      peer_at<unsigned int>(a.local_base, a.forward_reduce_route_count_offset);
  ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device> count(
      route_count[token]);

  // acq_rel RMWs form a release sequence for all route payload/map stores.
  // The final route publisher therefore acquires every earlier publisher
  // before releasing the token queue entry to Reduce.
  const auto previous = count.fetch_add(1U, ::cuda::memory_order_acq_rel);
  const auto lane = threadIdx.x % 32U;
  const auto completed = __ballot_sync(__activemask(), previous + 1U == top_k);
  if ((completed & (1U << lane)) == 0U)
    return;

  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> tail(
      control->reduce_ready_tail.value);
  // All completed tokens in this warp reserve a contiguous range together.
  // Each lane still releases its own entry after acquiring its top-k inputs.
  const auto leader = unsigned(__ffs(completed) - 1);
  auto begin = lane == leader ? tail.fetch_add(unsigned(__popc(completed)),
                                               ::cuda::memory_order_relaxed)
                              : 0ULL;
  begin = __shfl_sync(completed, begin, leader);
  const auto slot = begin + __popc(completed & ((1U << lane) - 1U));
  if (slot >= a.dispatch.identity.num_tokens)
    asm volatile("trap;");

  auto *ready_token = peer_at<unsigned long long>(
      a.local_base, a.forward_reduce_ready_token_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      token_word(ready_token[slot]);
  // Zero means unpublished. Generation setup clears the active range before
  // setup_ready releases producers, so token+1 is both the payload and the
  // release marker without a second queue-sized epoch array.
  auto encoded = token + 1ULL;
#if WEFTMOE_DEVICE_TIMELINE
  // Profiling builds tag the otherwise-unused high byte so the source-local
  // Reduce consumer can attribute the exact token-ready observation without
  // comparing clocks from the peer GPU that published it.
  encoded |= static_cast<unsigned long long>(publisher_kind) << 56U;
#else
  (void)publisher_kind;
#endif
  token_word.store(encoded, ::cuda::memory_order_release);
}

} // namespace forward_reduce_detail
