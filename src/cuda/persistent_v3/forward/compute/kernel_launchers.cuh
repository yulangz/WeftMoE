
#include "cuda/persistent_v3/forward/reduce_pipeline.cuh"

template <bool StaticLocal, bool M64 = false,
          ForwardTaskPartition Partition = ForwardTaskPartition::All,
          bool ReduceStage = true, bool Fc2Only = false,
          bool Fc1Only = false,
          std::uint32_t Fc1StageCount = forward_tuning::kFc1Stages,
          std::uint32_t Fc2StageCount = forward_tuning::kFc2Stages>
__global__ __launch_bounds__(forward_tuning::kComputeThreads, 1) void
local_p2p_forward_compute_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_kind_shared;
  __shared__ unsigned int return_mblock_shared;
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        setup_ready(scheduler[8ULL + 2ULL * a.world]);
    while (setup_ready.load(::cuda::memory_order_acquire) < a.epoch &&
           !forward_cancelled(a))
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
  __syncthreads();
  static_assert(!(Fc1Only && Fc2Only),
                "a Forward compute kernel cannot be both FC1-only and FC2-only");
  if constexpr (Fc1Only) {
    forward_compute_detail::persistent_role<
        StaticLocal, true, false, M64, Partition, false, false, Fc1StageCount,
        Fc2StageCount>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  } else if constexpr (Fc2Only) {
    forward_compute_detail::persistent_role<
        StaticLocal, false, true, M64, Partition, false, false, Fc1StageCount,
        Fc2StageCount>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  } else {
    // Every compute CTA is work-conserving. At each tile boundary it probes
    // FC2 first and borrows FC1 only when no FC2 tile is currently published.
    forward_compute_detail::persistent_role<
        StaticLocal, true, true, M64, Partition, false, false, Fc1StageCount,
        Fc2StageCount>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  }
  if constexpr (ReduceStage) {
    const auto reduce_role_index =
        a.forward_reduce_cta_offset + static_cast<std::uint32_t>(blockIdx.x);
    const auto reduce_role_count =
        a.forward_reduce_cta_count != 0U
            ? a.forward_reduce_cta_count
            : static_cast<std::uint32_t>(gridDim.x);
    if (reduce_role_index < reduce_role_count) {
      forward_compute_detail::rebalance_registers_for_reduce();
      forward_compute_detail::reduce_role<StaticLocal>(
          a, reduce_role_index, reduce_role_count);
    }
  }
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
#endif
}

// Compile-time STSM/TMA FC2 epilogue. DoFc1 selects the production
// FC2-priority mixed scheduler or the retained FC2-only tuning candidate
// without duplicating the kernel wrapper.
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
template <bool StaticLocal, bool DoFc1, bool StoreToStaging, bool Cluster2,
          std::uint32_t Fc1StageCount, std::uint32_t Fc2StageCount,
          bool RetainBackwardState>
__device__ __forceinline__ void local_p2p_forward_tma_store_role(
    const ForwardKernelArgs &a, const CUtensorMap &x_map,
    const CUtensorMap &s_map, const CUtensorMap &w1_map,
    const CUtensorMap &w2_map, const CUtensorMap &output_store_map,
    unsigned char *storage, unsigned long long *task_shared,
    unsigned int *task_kind_shared, unsigned int *return_mblock_shared,
    unsigned int *pair_operand_shared) {
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        setup_ready(scheduler[8ULL + 2ULL * a.world]);
    while (setup_ready.load(::cuda::memory_order_acquire) < a.epoch &&
           !forward_cancelled(a))
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
  __syncthreads();
  forward_compute_detail::persistent_role<
      StaticLocal, DoFc1, true, false, ForwardTaskPartition::All, true,
      StoreToStaging, Fc1StageCount, Fc2StageCount, Cluster2,
      RetainBackwardState>(
      a, x_map, s_map, w1_map, w2_map, storage, task_shared,
      task_kind_shared, return_mblock_shared, pair_operand_shared,
      &output_store_map);
  const auto reduce_role_index =
      a.forward_reduce_cta_offset + static_cast<std::uint32_t>(blockIdx.x);
  const auto reduce_role_count =
      a.forward_reduce_cta_count != 0U
          ? a.forward_reduce_cta_count
          : static_cast<std::uint32_t>(gridDim.x);
  if (reduce_role_index < reduce_role_count) {
    forward_compute_detail::rebalance_registers_for_reduce();
    forward_compute_detail::reduce_role<StaticLocal>(
        a, reduce_role_index, reduce_role_count);
  }
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
}
#endif

template <bool StaticLocal, bool DoFc1, bool StoreToStaging = false,
          bool Cluster2 = false,
          std::uint32_t Fc1StageCount = forward_tuning::kFc1Stages,
          std::uint32_t Fc2StageCount = forward_tuning::kFc2Stages>
__global__ __launch_bounds__(forward_tuning::kComputeThreads, 1) void
local_p2p_forward_tma_store_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_kind_shared;
  __shared__ unsigned int pair_operand_shared;
  __shared__ unsigned int return_mblock_shared;
  local_p2p_forward_tma_store_role<
      StaticLocal, DoFc1, StoreToStaging, Cluster2, Fc1StageCount,
      Fc2StageCount, false>(
      a, x_map, s_map, w1_map, w2_map, output_store_map, storage,
      &task_shared, &task_kind_shared, &return_mblock_shared,
      &pair_operand_shared);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
  (void)output_store_map;
#endif
}

// Training has a separate entry symbol so the ordinary inference kernel is
// compiled without checkpoint addressing, branches, or stores. Both entries
// execute the same mixed scheduler and differ only in the compile-time FC1
// epilogue policy.
template <bool StaticLocal, bool StoreToStaging = false,
          bool Cluster2 = false,
          std::uint32_t Fc1StageCount = forward_tuning::kFc1Stages,
          std::uint32_t Fc2StageCount = forward_tuning::kFc2Stages>
__global__ __launch_bounds__(forward_tuning::kComputeThreads, 1) void
local_p2p_forward_training_tma_store_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_kind_shared;
  __shared__ unsigned int pair_operand_shared;
  __shared__ unsigned int return_mblock_shared;
  local_p2p_forward_tma_store_role<
      StaticLocal, true, StoreToStaging, Cluster2, Fc1StageCount,
      Fc2StageCount, true>(
      a, x_map, s_map, w1_map, w2_map, output_store_map, storage,
      &task_shared, &task_kind_shared, &return_mblock_shared,
      &pair_operand_shared);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
  (void)output_store_map;
#endif
}

// Reduce follows the FC1/FC2 grid on the same compute stream.  It remains
// fully overlapped with Return/RDMA, while keeping its route-wait and FP32
// combine control flow out of the WGMMA entry's register allocation.
template <bool StaticLocal>
__global__
__launch_bounds__(256,
                  1) void local_p2p_forward_reduce_kernel(ForwardKernelArgs a) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        setup_ready(scheduler[8ULL + 2ULL * a.world]);
    while (setup_ready.load(::cuda::memory_order_acquire) < a.epoch &&
           !forward_cancelled(a))
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
  __syncthreads();
  forward_compute_detail::reduce_role<StaticLocal>(
      a, static_cast<std::uint32_t>(blockIdx.x),
      static_cast<std::uint32_t>(gridDim.x));
#else
  (void)a;
#endif
}
