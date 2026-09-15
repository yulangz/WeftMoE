#pragma once

#include "cuda/persistent_v3/forward/unified_transport.cuh"

namespace forward_unified_detail {
#if __has_include(<nccl_device.h>)
using CommunicationShared =
    forward_unified_transport::Shared<forward_tuning::kUnifiedCommThreads,
                                      forward_tuning::kUnifiedSegmentBytes,
                                      forward_tuning::kUnifiedCopyBuffers>;
#endif

inline constexpr std::size_t shared_bytes(unsigned world) {
#if __has_include(<nccl_device.h>)
  return ((forward_tuning::compute_smem_bytes(world) + 127U) & ~127ULL) +
         sizeof(CommunicationShared);
#else
  // The unavailable backend still needs a visible name in discarded host
  // launch branches; no unified kernel is launched without device NCCL.
  (void)world;
  return 0U;
#endif
}

// A zero-world layout is a lower bound: it includes every fixed compute
// operand/barrier, alignment gap and communication byte, but does not guess
// the runtime peer metadata or ptxas's static shared allocation. The host
// configures shared_bytes(actual_world); CUDA also charges the actual entry's
// static shared bytes when validating its opt-in dynamic allocation.
static_assert(!forward_tuning::kUnifiedWorkers ||
                  shared_bytes(0U) <= 227U * 1024U,
              "Unified Forward fixed compute/communication storage exceeds "
              "Hopper shared memory; reduce staging or the FC2 claim size");

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
template <bool StaticLocal> struct Communication {
  const ForwardKernelArgs &a;
  persistent_v3::ForwardPipelineControl &control;
  unsigned long long *scheduler;
  CommunicationShared &storage;
  // The entry separates the control role from math before calling transport.
  // Inlining preserves shared-memory address information for copy/barrier
  // operations without passing transport state through the math routine.
  __device__ __forceinline__ void operator()() const {
    if (threadIdx.x >= forward_tuning::kUnifiedCommBase &&
        threadIdx.x < forward_tuning::kUnifiedCommBase +
                          forward_tuning::kUnifiedCommThreads) {
      if constexpr (StaticLocal) {
        if constexpr (forward_tuning::kUnifiedSplitNvlink)
          forward_unified_transport::run_local_warps<
              forward_tuning::kUnifiedCommThreads,
              forward_tuning::kUnifiedCommBase>(a, control, scheduler, storage);
        else
          forward_unified_transport::run<forward_tuning::kUnifiedCommThreads,
                                         forward_tuning::kUnifiedCommBase>(
              a, control, scheduler, storage);
      } else
        forward_unified_transport::run_gin<forward_tuning::kUnifiedCommThreads,
                                           forward_tuning::kUnifiedCommBase>(
            a, control, scheduler, storage);
    }
  }
};
#endif
} // namespace forward_unified_detail

template <bool StaticLocal, bool RetainBackwardState>
__global__ __launch_bounds__(
    forward_tuning::kUnifiedThreads,
    1) void unified_forward_kernel(const __grid_constant__ ForwardKernelArgs a,
                                   const __grid_constant__ CUtensorMap x_map,
                                   const __grid_constant__ CUtensorMap s_map,
                                   const __grid_constant__ CUtensorMap w1_map,
                                   const __grid_constant__ CUtensorMap w2_map,
                                   const __grid_constant__ CUtensorMap
                                       output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task;
  __shared__ unsigned kind, return_mblock, pair_operand;
  __shared__ forward_compute_detail::ComputePairState pair;
  forward_compute_detail::observe_compute_cluster(a);
  using forward_compute_detail::ComputeRole;
  forward_compute_detail::persistent_role<
      StaticLocal, RetainBackwardState, forward_tuning::kFc1Stages,
      forward_tuning::kFc2Stages, true, ComputeRole::Initialize>(
      a, x_map, s_map, w1_map, w2_map, storage, &task, &kind, &return_mblock,
      &pair_operand, &output_store_map, pair);
  if (threadIdx.x >= forward_compute_detail::kMathThreads) {
    // Every lane of each control warpgroup releases registers before its
    // producer and communication warps enter independent progress loops.
    if constexpr (forward_tuning::kUnifiedProducerRegisters !=
                  forward_tuning::kUnifiedControlRegisters) {
      // A dedicated producer warpgroup can return more registers to math.
      // Both branches are uniform over all 128 lanes of each warpgroup.
      if (threadIdx.x < 384U)
        cutlass::arch::warpgroup_reg_dealloc<
            forward_tuning::kUnifiedProducerRegisters>();
      else
        cutlass::arch::warpgroup_reg_dealloc<
            forward_tuning::kUnifiedControlRegisters>();
    } else {
      cutlass::arch::warpgroup_reg_dealloc<
          forward_tuning::kUnifiedControlRegisters>();
    }
    if (threadIdx.x < 288U) {
      forward_compute_detail::persistent_role<
          StaticLocal, RetainBackwardState, forward_tuning::kFc1Stages,
          forward_tuning::kFc2Stages, true, ComputeRole::Producer>(
          a, x_map, s_map, w1_map, w2_map, storage, &task, &kind,
          &return_mblock, &pair_operand, &output_store_map, pair);
    } else {
      auto *scheduler =
          peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
      auto &control = *peer_at<persistent_v3::ForwardPipelineControl>(
          a.local_base, a.forward_v3_control_offset);
      auto &comm =
          *reinterpret_cast<forward_unified_detail::CommunicationShared *>(
              storage +
              ((forward_tuning::compute_smem_bytes(a.world) + 127U) & ~127ULL));
      forward_unified_detail::Communication<StaticLocal>{
          a, control, scheduler, comm}();
    }
  } else {
    forward_compute_detail::persistent_role<
        StaticLocal, RetainBackwardState, forward_tuning::kFc1Stages,
        forward_tuning::kFc2Stages, true, ComputeRole::Math>(
        a, x_map, s_map, w1_map, w2_map, storage, &task, &kind, &return_mblock,
        &pair_operand, &output_store_map, pair);
  }
  // Communication remains live while the math warpgroups drain fixed-order
  // token sums. Each domain owns its own barriers and storage until it exits.
  if (threadIdx.x < forward_compute_detail::kMathThreads)
    forward_compute_detail::reduce_role<StaticLocal, 256U>(
        a, blockIdx.x, a.forward_reduce_cta_count);
  __syncthreads();
  // Every role is finished before either CTA can retire its DSM mailbox.
  cute::cluster_sync();
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto &control = *peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if constexpr (StaticLocal)
    forward_finalize_detail::local_split_transport_finalize(a, control,
                                                            scheduler, status);
  else {
    auto *completion =
        static_cast<transport_state::GenerationCompletionVector *>(
            a.peer_bases[a.world]);
    forward_finalize_detail::split_transport_finalize<false>(
        a, control, completion, scheduler, status);
  }
#endif
}

#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY) && __has_include(<nccl_device.h>)
#include "unified_communication_benchmark.cuh"
#endif

template <bool StaticLocal, bool RetainBackwardState>
constexpr auto unified_forward_entry() {
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY) && __has_include(<nccl_device.h>)
  return unified_communication_benchmark_kernel<StaticLocal,
                                                RetainBackwardState>;
#else
  return unified_forward_kernel<StaticLocal, RetainBackwardState>;
#endif
}
