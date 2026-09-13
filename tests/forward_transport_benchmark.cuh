#pragma once

// Test-only communication measurement. Keep Dispatch, Return and Reduce on
// their production paths; replace GEMMs with producers waiting for completed
// Dispatch. This measures sequential Dispatch then Return/Reduce, like a
// communication API. With backward state disabled, the received canonical
// rows stand in for FC2 output without adding another data copy. Every compute
// CTA retains its production reduction role and launch resources.
template <bool StaticLocal>
__device__ void transport_benchmark_role(const ForwardKernelArgs &a,
                                         unsigned char *storage) {
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  const auto *claims = peer_at<const unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *ready = reinterpret_cast<unsigned int *>(storage);
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  if (threadIdx.x == 0U) {
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               control->relay_dispatch_roles_complete.value) <
               a.forward_relay_cta_count &&
           !forward_cancelled(a))
      __nanosleep(64U);
  }
  __syncthreads();
  // Cluster placement can leave some CTAs pending. Claim work dynamically,
  // as the production scheduler does; static block-strided ownership could
  // leave resident reducers waiting for a producer that has not been launched.
  auto *claimed = reinterpret_cast<unsigned long long *>(storage + 256U);
  while (true) {
    if (threadIdx.x == 0U)
      *claimed = atomicAdd(scheduler, 1ULL);
    __syncthreads();
    const auto mblock = *claimed;
    if (mblock >= scheduler[3])
      break;
    const auto map = claims[mblock];
    const auto expert = static_cast<unsigned int>(map >> 32U);
    const auto local = static_cast<unsigned int>(map);
    const auto local_row = forward_mblock_local_row(a, expert, local);
    const auto rows = forward_mblock_valid_rows(a, expert, local);
    const auto row = a.dispatch.expert_row_prefix[expert] + local_row;
    // Local completion includes every destination-side FC1 publisher. It
    // already acquires all input tiles; GIN's independent owner still needs
    // the per-tile dependency check.
    if (!a.forward_stream_local_dispatch && threadIdx.x == 0U) {
      while (
          !forward_mblock_rows_ready<StaticLocal>(a, expert, local_row, rows) &&
          !forward_cancelled(a))
        __nanosleep(64U);
    }
    __syncthreads();
    __threadfence_system();

    if (threadIdx.x < kMathThreads)
      forward_return_detail::publish_rows<kMathThreads>(a, scheduler, row, rows,
                                                        ready);
    __syncthreads();
    if (threadIdx.x == 0U)
      atomicAdd(&scheduler[4U + a.world], 1ULL);
  }
}
