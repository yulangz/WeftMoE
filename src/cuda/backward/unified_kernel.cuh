#pragma once

#include "cuda/backward/unified_transport.cuh"

namespace backward_unified_detail {
static_assert(!backward_tuning::kUnifiedWorkers ||
                  backward_tuning::kUnifiedProducerRegisters ==
                      backward_tuning::kUnifiedControlRegisters,
              "Unified Backward currently shares the producer register quota "
              "with communication; independent producer quota is Forward-only");
#if __has_include(<nccl_device.h>)
using CommunicationShared =
    backward_unified_transport::Shared<backward_tuning::kUnifiedCommThreads,
                                       backward_tuning::kUnifiedSegmentBytes,
                                       backward_tuning::kUnifiedCopyBuffers>;
#endif
inline constexpr unsigned kStorageTileN =
    backward_tuning::kDabTileN > backward_tuning::kDxTileN
        ? (backward_tuning::kDabTileN > backward_tuning::kDwTileN
               ? backward_tuning::kDabTileN
               : backward_tuning::kDwTileN)
        : (backward_tuning::kDxTileN > backward_tuning::kDwTileN
               ? backward_tuning::kDxTileN
               : backward_tuning::kDwTileN);
inline constexpr std::size_t communication_offset() {
  return (backward_compute_detail::dynamic_shared_bytes<kStorageTileN>() +
          127U) &
         ~std::size_t(127U);
}
inline constexpr std::size_t shared_bytes() {
#if __has_include(<nccl_device.h>)
  return communication_offset() + sizeof(CommunicationShared);
#else
  // Keep the disabled host launch branch well-formed without device NCCL.
  return 0U;
#endif
}
} // namespace backward_unified_detail

template <std::uint32_t DabTileN, std::uint32_t DxTileN, std::uint32_t DwTileN,
          bool UseGin = false>
__global__ __launch_bounds__(
    backward_tuning::kUnifiedThreads,
    1) void unified_backward_kernel(const __grid_constant__ BackwardKernelArgs
                                        a,
                                    const __grid_constant__ CUtensorMap dr_map,
                                    const __grid_constant__ CUtensorMap dab_map,
                                    const __grid_constant__ CUtensorMap w2_map,
                                    const __grid_constant__ CUtensorMap w1_map,
                                    const __grid_constant__ CUtensorMap
                                        dr_mn_map,
                                    const __grid_constant__ CUtensorMap
                                        u_mn_map,
                                    const __grid_constant__ CUtensorMap
                                        dab_mn_map,
                                    const __grid_constant__ CUtensorMap
                                        x_mn_map,
                                    const __grid_constant__ CUtensorMap
                                        u_store_map,
                                    const __grid_constant__ CUtensorMap
                                        dx_store_map,
                                    const __grid_constant__ CUtensorMap
                                        dw2_store_map,
                                    const __grid_constant__ CUtensorMap
                                        dw1_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  // The host selects a topology specialization so local workers do not carry
  // the remote state machine's register live ranges.
  if (a.use_gin != UseGin)
    __trap();
  static_assert(DabTileN == backward_tuning::kDabTileN &&
                DxTileN == backward_tuning::kDxTileN &&
                DwTileN == backward_tuning::kDwTileN);
  static_assert(!backward_tuning::kUnifiedWorkers ||
                    backward_unified_detail::shared_bytes() + 2048U <=
                        227U * 1024U,
                "Unified Backward shared memory exceeds Hopper CTA capacity");
  const auto geometry = backward_unified_compute_detail::load_row_geometry(a);
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ backward_unified_compute_detail::SharedState state;
  auto &comm =
      *reinterpret_cast<backward_unified_detail::CommunicationShared *>(
          storage + backward_unified_detail::communication_offset());
  backward_unified_compute_detail::initialize_compute<
      backward_unified_detail::kStorageTileN>(storage, state);
  cute::cluster_sync();
  // Each register domain enters one persistent loop and then returns. Joining
  // math and producer at a common outer task loop lets ptxas discard setmaxnreg
  // and spill the N256 accumulator despite the available CTA register pool.
  if (threadIdx.x >= backward_compute_detail::kMathThreads) {
    cutlass::arch::warpgroup_reg_dealloc<
        backward_tuning::kUnifiedControlRegisters>();
    if (threadIdx.x < backward_unified_compute_detail::kComputeThreads) {
      backward_unified_compute_detail::UnifiedComputeSynchronization<false>
          sync{state};
      backward_unified_compute_detail::run_compute<DabTileN, DxTileN, DwTileN>(
          a, dr_map, dab_map, w2_map, w1_map, dr_mn_map, u_mn_map, dab_mn_map,
          x_mn_map, u_store_map, dx_store_map, dw2_store_map, dw1_store_map,
          geometry, storage, state, sync);
    } else if (threadIdx.x >= backward_tuning::kUnifiedCommBase &&
               threadIdx.x < backward_tuning::kUnifiedCommBase +
                                 backward_tuning::kUnifiedCommThreads) {
      constexpr unsigned kReduceWarps =
          backward_tuning::kUnifiedCommThreads / 32U;
      const unsigned reduce_tid =
          threadIdx.x - backward_tuning::kUnifiedCommBase;
      const unsigned reduce_warp = reduce_tid / 32U;
      const unsigned lane = reduce_tid % 32U;
      const std::uint64_t first_token =
          weftmoe_pf::backward_return::reduce_first(a, kReduceWarps,
                                                    reduce_warp);
      auto token = first_token;
      backward_unified_transport::run<UseGin,
                                      backward_tuning::kUnifiedCommBase>(
          a, geometry, comm, token);
      // run() returns after this CTA has retired every notification channel
      // and GIN has drained its contexts. Other CTAs can still finish their
      // local notification rows, but no remaining message needs these warps.
      // Static token ownership therefore needs no shared queue or head wait.
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      if (reduce_tid == 0U)
        atomicMax(&status->backward_transport_complete_clock,
                  device_globaltimer_ns());
      if (first_token < a.dispatch.identity.num_tokens && lane == 0U)
        atomicMin(&status->backward_reduce_start_clock,
                  device_globaltimer_ns());
      for (; token < a.dispatch.identity.num_tokens;
           token +=
           std::uint64_t(weftmoe_pf::backward_return::reduce_workers(a)) *
           kReduceWarps) {
        backward_reduce_dx_token(a, token);
        __syncwarp();
      }
      __threadfence();
      __syncwarp();
      if (first_token < a.dispatch.identity.num_tokens && lane == 0U)
        atomicMax(&status->backward_reduce_complete_clock,
                  device_globaltimer_ns());
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<
        backward_tuning::kUnifiedMathRegisters>();
    backward_unified_compute_detail::UnifiedComputeSynchronization<true> sync{
        state};
    backward_unified_compute_detail::run_compute<DabTileN, DxTileN, DwTileN>(
        a, dr_map, dab_map, w2_map, w1_map, dr_mn_map, u_mn_map, dab_mn_map,
        x_mn_map, u_store_map, dx_store_map, dw2_store_map, dw1_store_map,
        geometry, storage, state, sync);
  }
  // No role exits while a peer can still access this CTA's DSM mailbox. This
  // full-cluster lifetime boundary runs only after compute and transport drain;
  // per-task joins remain confined to the compute domain.
  __syncthreads();
  cute::cluster_sync();

#endif
}
