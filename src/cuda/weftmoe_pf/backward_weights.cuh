#pragma once

#include "cuda/weftmoe_pf/backward_schedule.cuh"

// Shared dW math and cluster synchronization used after PF activation drains.
namespace weftmoe_pf::backward_weights {

inline constexpr unsigned kStorageTileN =
    backward_tuning::kDabTileN > backward_tuning::kDxTileN
        ? (backward_tuning::kDabTileN > backward_tuning::kDwTileN
               ? backward_tuning::kDabTileN
               : backward_tuning::kDwTileN)
        : (backward_tuning::kDxTileN > backward_tuning::kDwTileN
               ? backward_tuning::kDxTileN
               : backward_tuning::kDwTileN);

inline constexpr std::uint32_t kComputeThreads =
    backward_compute_detail::kMathThreads + 32U;
inline constexpr std::uint32_t kComputeBarrier = 3U;

struct RowGeometry {
  std::uint32_t mblocks;
  std::uint32_t active_experts;
};

// Keep mutable geometry separate so the launch contract remains in read-only
// grid parameter memory throughout the compute and communication roles.
__device__ __forceinline__ RowGeometry
load_row_geometry(const BackwardKernelArgs &a) {
  __shared__ RowGeometry geometry;
  if (threadIdx.x == 0U) {
    const auto experts = dispatch_sizes(a.dispatch).owned_expert_count;
    const auto *offsets = peer_at<const unsigned long long>(
        a.local_base, a.padded_expert_offsets_offset);
    geometry.mblocks =
        static_cast<std::uint32_t>(offsets[experts] / backward_tuning::kTileM);
    unsigned active = 0U;
    for (unsigned expert = 0U; expert < experts; ++expert)
      active += a.dispatch.expert_row_count[expert] != 0ULL;
    geometry.active_experts = active;
  }
  __syncthreads();
  return geometry;
}

struct alignas(16) SharedState {
  std::uint32_t task_kind;
  std::uint32_t task_owner;
  std::uint32_t task_tile;
  std::uint32_t task_count;
  alignas(8) std::uint64_t cluster_join;
};

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
template <bool MathRole> struct UnifiedComputeSynchronization {
  static constexpr bool kMathEnabled = MathRole;
  static constexpr bool kProducerEnabled = !MathRole;
  static constexpr bool kUnifiedReturnQueue = true;
  static constexpr std::uint32_t kActivationWaitGroup = 0U;
  static constexpr std::uint32_t kThreads = kComputeThreads;
  static constexpr std::uint32_t kWorkWarps =
      backward_compute_detail::kMathThreads / 32U;
  SharedState &state;
  std::uint32_t phase = 0U;

  __device__ __forceinline__ static void cta_sync() {
    cutlass::arch::NamedBarrier::sync(
        kThreads,
        static_cast<cutlass::arch::ReservedNamedBarriers>(kComputeBarrier));
  }

  __device__ __forceinline__ static void math_sync(std::uint32_t threads,
                                                   std::uint32_t barrier) {
    // Select physical barrier 1/2, without CUTLASS's user-barrier offset of 8.
    // Physical 8/9 belong to the concurrent communication domains.
    cutlass::arch::NamedBarrier::sync(
        threads,
        static_cast<cutlass::arch::ReservedNamedBarriers>(barrier + 1U));
  }

  __device__ __forceinline__ void cluster_sync() {
    // Only the compute domain joins. Communication may be supplying the very
    // data that a compute task awaits and must never participate here.
    cta_sync();
    if (MathRole && threadIdx.x == 0U) {
      // Two arrivals (local and peer) are essential: a fast CTA cannot flip
      // the next phase before the peer has observed and left this phase.
      // DSM mailbox stores/loads require cluster scope. CUTLASS's unqualified
      // mbarrier arrive/wait defaults to CTA scope, even for a remote address.
      const auto local = cute::cast_smem_ptr_to_uint(&state.cluster_join);
      const auto peer = cute::block_rank_in_cluster() ^ 1U;
      asm volatile(
          "{\n"
          ".reg .b32 remote;\n"
          ".reg .pred ready;\n"
          "mapa.shared::cluster.u32 remote, %0, %1;\n"
          "mbarrier.arrive.release.cluster.shared::cta.b64 _, [%0];\n"
          "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote];\n"
          "wait_phase:\n"
          "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 "
          "ready, [%0], %2;\n"
          "@!ready bra wait_phase;\n"
          "}\n"
          :
          : "r"(local), "r"(peer), "r"(phase)
          : "memory");
      phase ^= 1U;
    }
    cta_sync();
  }

  __device__ __forceinline__ void mailbox_read_done() {
    // A NoTask iteration can finish immediately. Acknowledge the peer's DSM
    // copy before CTA 0 is allowed to overwrite the next task descriptor.
    cluster_sync();
  }
};

// Call during common CTA startup, before the one full-cluster initialization
// join. The caller subsequently partitions registers for entire warpgroups,
// then calls run_compute only for threadIdx.x < kComputeThreads. The caller
// must also load RowGeometry before splitting roles.
template <std::uint32_t StorageTileN>
__device__ __forceinline__ void initialize_compute(unsigned char *storage,
                                                   SharedState &state) {
  backward_compute_detail::initialize_k_mn_barriers<StorageTileN>(storage);
  if (threadIdx.x == 0U) {
    state.task_kind = ~0U;
    state.task_owner = 0U;
    state.task_tile = 0U;
    state.task_count = 0U;
    auto &barrier =
        *reinterpret_cast<cutlass::arch::ClusterBarrier *>(&state.cluster_join);
    barrier.init(2U);
    cutlass::arch::fence_barrier_init();
  }
}

template <std::uint32_t DwTileN, class Synchronization>
__device__ __forceinline__ void
run_compute(const BackwardKernelArgs &a, const CUtensorMap &dr_mn_map,
            const CUtensorMap &u_mn_map, const CUtensorMap &dab_mn_map,
            const CUtensorMap &x_mn_map, const CUtensorMap &dw2_store_map,
            const CUtensorMap &dw1_store_map, const RowGeometry &geometry,
            unsigned char *storage, SharedState &state,
            Synchronization &synchronization) {
  using namespace backward_pipeline_detail;
  static_assert(DwTileN == backward_tuning::kDwTileN);
  constexpr auto kDxTiles =
      backward_tuning::kMainHidden / backward_tuning::kDxTileN / 2U;
  constexpr std::uint32_t kDwStoreIssuers =
      (backward_tuning::kDwTileM / 128U) * (DwTileN / 64U);
  auto &task_kind_shared = state.task_kind;
  auto &task_owner_shared = state.task_owner;
  auto &task_tile_shared = state.task_tile;
  auto &task_count_shared = state.task_count;
  std::uint32_t dw_pipeline = 0U;
  auto *words = scheduler_words(a);
  const auto layout =
      weftmoe::cuda::pipeline::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity, kDxTiles, a.scheduler_expert_capacity);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  const auto cluster = cooperative_groups::this_cluster();
  const auto cluster_rank = cluster.block_rank();
  weftmoe_pf::backward::Schedule schedule(blockIdx.x / 2U);

  while (true) {
    if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
        cluster_rank == 0U) {
      TileTask task{};
      const bool claimed = schedule.next(a, gridDim.x / 2U, task);

      if (claimed) {
        task_kind_shared = static_cast<std::uint32_t>(task.kind);
        task_owner_shared = task.owner;
        task_tile_shared = task.tile;
        task_count_shared = task.tile_count;
      } else {
        task_kind_shared = kTerminalTask;
      }
    }
    // This join drains the previous task before either CTA reads the new
    // claim. Each CTA then keeps its own mailbox copy until the next join.
    synchronization.cluster_sync();
    if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
        cluster_rank != 0U) {
      task_kind_shared = *cluster.map_shared_rank(&task_kind_shared, 0U);
      task_owner_shared = *cluster.map_shared_rank(&task_owner_shared, 0U);
      task_tile_shared = *cluster.map_shared_rank(&task_tile_shared, 0U);
      task_count_shared = *cluster.map_shared_rank(&task_count_shared, 0U);
    }
    synchronization.mailbox_read_done();
    if (task_kind_shared == kTerminalTask) {
      if (threadIdx.x < kDwStoreIssuers)
        asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
      synchronization.cluster_sync();
      break;
    }

    const auto owner = task_owner_shared;

    if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dw2)) {
      if (Synchronization::kMathEnabled && threadIdx.x == 0U)
        atomicMin(&status->backward_dw2_start_clock, device_globaltimer_ns());
      const auto *padded_offsets = peer_at<const unsigned long long>(
          a.local_base, a.padded_expert_offsets_offset);
      const auto reduction_rows =
          static_cast<std::uint32_t>(words[layout.dw_reduction_rows + owner]);
      const auto compact_reduction_start =
          static_cast<std::uint32_t>(a.dispatch.expert_row_prefix[owner]);
      backward_compute_detail::run_mn_mn_chunk<
          backward_tuning::kDwTileM, DwTileN, kStorageTileN, Synchronization>(
          dr_mn_map, u_mn_map, dw2_store_map, task_tile_shared,
          task_count_shared, kDw2MTiles, kDw2NTiles, compact_reduction_start,
          static_cast<std::uint32_t>(padded_offsets[owner]), reduction_rows,
          static_cast<std::uint32_t>(a.intermediate),
          owner * static_cast<std::uint32_t>(a.hidden), storage, dw_pipeline);
      const auto last = pipeline::backward::cluster_tile(
          task_tile_shared + task_count_shared - 1U, cluster_rank, kDw2MTiles,
          kDw2NTiles);
      const auto completed_tiles = task_count_shared - (last.valid ? 0U : 1U);
      if (Synchronization::kMathEnabled && threadIdx.x == 0U) {
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw2_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done =
            completed.fetch_add(completed_tiles, ::cuda::memory_order_acq_rel) +
            completed_tiles;
        if (done == static_cast<unsigned long long>(geometry.active_experts) *
                        kDw2TilesPerExpert)
          atomicMax(&status->backward_dw2_complete_clock,
                    device_globaltimer_ns());
      }
    } else if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dw1)) {
      if (Synchronization::kMathEnabled && threadIdx.x == 0U)
        atomicMin(&status->backward_dw1_start_clock, device_globaltimer_ns());
      const auto *padded_offsets = peer_at<const unsigned long long>(
          a.local_base, a.padded_expert_offsets_offset);
      const auto reduction_rows =
          static_cast<std::uint32_t>(words[layout.dw_reduction_rows + owner]);
      const auto compact_reduction_start =
          static_cast<std::uint32_t>(a.dispatch.expert_row_prefix[owner]);
      backward_compute_detail::run_mn_mn_chunk<
          backward_tuning::kDwTileM, DwTileN, kStorageTileN, Synchronization>(
          dab_mn_map, x_mn_map, dw1_store_map, task_tile_shared,
          task_count_shared, kDw1MTiles, kDw1NTiles,
          static_cast<std::uint32_t>(padded_offsets[owner]),
          compact_reduction_start, reduction_rows,
          static_cast<std::uint32_t>(a.hidden),
          owner * static_cast<std::uint32_t>(2ULL * a.intermediate), storage,
          dw_pipeline);
      const auto last = pipeline::backward::cluster_tile(
          task_tile_shared + task_count_shared - 1U, cluster_rank, kDw1MTiles,
          kDw1NTiles);
      const auto completed_tiles = task_count_shared - (last.valid ? 0U : 1U);
      if (Synchronization::kMathEnabled && threadIdx.x == 0U) {
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw1_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done =
            completed.fetch_add(completed_tiles, ::cuda::memory_order_acq_rel) +
            completed_tiles;
        if (done == static_cast<unsigned long long>(geometry.active_experts) *
                        kDw1TilesPerExpert)
          atomicMax(&status->backward_dw1_complete_clock,
                    device_globaltimer_ns());
      }
    }
    // The next iteration joins the cluster. Only a local join is needed
    // here to protect this CTA's mailbox from its leader's next claim.
    Synchronization::cta_sync();
  }
}
#endif

} // namespace weftmoe_pf::backward_weights
