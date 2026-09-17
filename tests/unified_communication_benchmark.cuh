#pragma once

// Test-only sequential Dispatch -> Return/Reduce measurement. Keep the real
// communication workers and their full shared-memory reservation. Canonical
// received rows stand in for FC2 output, as in the split communication probe.
template <bool StaticLocal, bool RetainBackwardState>
__global__ __launch_bounds__(forward_tuning::kUnifiedThreads, 1)
void unified_communication_benchmark_kernel(
    const __grid_constant__ ForwardKernelArgs a,
    const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  programmatic_launch::wait();
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *scheduler = peer_at<unsigned long long>(
      a.local_base, a.forward_scheduler_offset);
  auto &control = *peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  auto &comm = *reinterpret_cast<forward_unified_detail::CommunicationShared *>(
      storage + ((forward_tuning::compute_smem_bytes(a.world) + 127U) & ~127ULL));
  forward_compute_detail::observe_compute_cluster(a);
  cute::cluster_sync();
  if (threadIdx.x < 256U) {
    forward_compute_detail::transport_benchmark_role<StaticLocal, 256U>(a,
                                                                       storage);
    forward_compute_detail::reduce_role<StaticLocal, 256U>(
        a, blockIdx.x, a.forward_reduce_cta_count);
  } else if (threadIdx.x >= forward_tuning::kUnifiedCommBase &&
             threadIdx.x < forward_tuning::kUnifiedCommBase +
                               forward_tuning::kUnifiedCommThreads) {
    forward_unified_detail::Communication<StaticLocal>{
        a, control, scheduler, comm}();
  }
  __syncthreads();
  cute::cluster_sync();
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if constexpr (StaticLocal)
    forward_finalize_detail::local_split_transport_finalize(
        a, control, scheduler, status);
  else
    forward_finalize_detail::split_transport_finalize<false>(
        a, control,
        static_cast<transport_state::GenerationCompletionVector *>(
            a.peer_bases[a.world]),
        scheduler, status);
#endif
}
