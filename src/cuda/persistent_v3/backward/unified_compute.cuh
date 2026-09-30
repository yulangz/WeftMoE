#pragma once

#include "cuda/weftmoe_pf/backward_schedule.cuh"

// Included after backward_pipeline_detail and the SwiGLU epilogue helpers.
// Both kernel layouts use this scheduler and the same arithmetic routines.
namespace backward_unified_compute_detail {

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
  std::uint32_t phase;
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
      auto &barrier = *reinterpret_cast<cutlass::arch::ClusterBarrier *>(
          &state.cluster_join);
      // Two arrivals (local and peer) are essential: a fast CTA cannot flip
      // the next phase before the peer has observed and left this phase.
      barrier.arrive();
      barrier.arrive(cute::block_rank_in_cluster() ^ 1U);
      barrier.wait(phase);
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
    state.phase = 0U;
    auto &barrier =
        *reinterpret_cast<cutlass::arch::ClusterBarrier *>(&state.cluster_join);
    barrier.init(2U);
    cutlass::arch::fence_barrier_init();
  }
}

template <std::uint32_t DabTileN, std::uint32_t DxTileN, std::uint32_t DwTileN,
          std::uint32_t DxClaimTiles = 1U, bool StaticSchedule = false,
          class Synchronization>
__device__ __forceinline__ void
run_compute(const BackwardKernelArgs &a, const CUtensorMap &dr_map,
            const CUtensorMap &dab_map, const CUtensorMap &w2_map,
            const CUtensorMap &w1_map, const CUtensorMap &dr_mn_map,
            const CUtensorMap &u_mn_map, const CUtensorMap &dab_mn_map,
            const CUtensorMap &x_mn_map, const CUtensorMap &u_store_map,
            const CUtensorMap &dx_store_map, const CUtensorMap &dw2_store_map,
            const CUtensorMap &dw1_store_map, const RowGeometry &geometry,
            unsigned char *storage, SharedState &state,
            Synchronization &synchronization) {
  using namespace backward_pipeline_detail;
  static_assert(backward_tuning::kMainIntermediate % DabTileN == 0U);
  static_assert(backward_tuning::kMainHidden % DxTileN == 0U);
  static_assert(DwTileN == backward_tuning::kDwTileN);
  constexpr std::uint32_t kActivationStorageTileN =
      DabTileN > DxTileN ? DabTileN : DxTileN;
  constexpr std::uint32_t kStorageTileN =
      kActivationStorageTileN > DwTileN ? kActivationStorageTileN : DwTileN;
  constexpr std::uint32_t kDuTiles = backward_tuning::kMainIntermediate /
                                     DabTileN /
                                     backward_tuning::kActivationClusterN;
  constexpr std::uint32_t kDxTiles = backward_tuning::kMainHidden / DxTileN /
                                     backward_tuning::kActivationClusterN;
  constexpr std::uint32_t kDwStoreIssuers =
      (backward_tuning::kDwTileM / 128U) * (DwTileN / 64U);
  auto &task_kind_shared = state.task_kind;
  auto &task_owner_shared = state.task_owner;
  auto &task_tile_shared = state.task_tile;
  auto &task_count_shared = state.task_count;
  auto &phase_shared = state.phase;
  std::uint32_t activation_pipeline = 0U;
  std::uint32_t dw_pipeline = 0U;
  auto *words = scheduler_words(a);
  const auto layout =
      weftmoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity, kDxTiles, a.scheduler_expert_capacity);
  auto queues = make_work_queues(a, words, layout, geometry.mblocks);
  queues.dx.claim_granularity = DxClaimTiles;
  static_assert(DxClaimTiles == 1U || DxClaimTiles == 2U);
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + layout.descriptors);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  bool activation_complete = false;
  const auto cluster = cooperative_groups::this_cluster();
  const auto cluster_rank = cluster.block_rank();
  weftmoe_pf::backward::Schedule schedule(blockIdx.x / 2U);

  while (true) {
    if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
        cluster_rank == 0U) {
      TileTask task{};
      phase_shared = 0U;
      bool claimed = false;
      bool terminal = false;
      if constexpr (StaticSchedule) {
        claimed = schedule.next<kDuTiles, kDxTiles>(
            a, words, layout, geometry.mblocks, gridDim.x / 2U, task,
            phase_shared, terminal);
      } else {
        if (!activation_complete) {
          claimed =
              weftmoe::cuda::persistent_v3::backward::try_claim_domain_once(
                  queues.dx, task, blockIdx.x / 2U) ||
              weftmoe::cuda::persistent_v3::backward::try_claim_domain_once(
                  queues.dab, task, blockIdx.x / 2U);
          if (!claimed) {
            auto *dx_counter = reinterpret_cast<PaddedCounter *>(
                words + layout.completed_dx_tiles);
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                completed_dx(dx_counter->value);
            activation_complete =
                completed_dx.load(::cuda::memory_order_acquire) ==
                static_cast<unsigned long long>(geometry.mblocks) * kDxTiles;
          }
        }
        if (!claimed) {
          // Do not reserve a private dW backlog while activation work can still
          // arrive.  Once dX is globally complete, restore the tuned multi-tile
          // claims so each CTA can keep the dW K pipeline hot.
          auto dw2 = queues.dw2;
          if (!activation_complete)
            dw2.claim_granularity = 1U;
          claimed =
              weftmoe::cuda::persistent_v3::backward::try_claim_domain_once(
                  dw2, task);
        }
        if (!claimed) {
          auto dw1 = queues.dw1;
          if (!activation_complete)
            dw1.claim_granularity = 1U;
          claimed =
              weftmoe::cuda::persistent_v3::backward::try_claim_domain_once(
                  dw1, task);
        }
      }
      if (claimed) {
        task_kind_shared = static_cast<std::uint32_t>(task.kind);
        task_owner_shared = task.owner;
        task_tile_shared = task.tile;
        task_count_shared = task.tile_count;
        if (!StaticSchedule && task.kind == TaskKind::Dab) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              phase(words[layout.phase + task.owner]);
          phase_shared = static_cast<std::uint32_t>(
              phase.load(::cuda::memory_order_acquire));
        }
      } else {
        if constexpr (StaticSchedule) {
          task_kind_shared = terminal ? kTerminalTask : kNoTask;
        } else {
          auto *dx_counter = reinterpret_cast<PaddedCounter *>(
              words + layout.completed_dx_tiles);
          auto *dw2_counter = reinterpret_cast<PaddedCounter *>(
              words + layout.completed_dw2_tiles);
          auto *dw1_counter = reinterpret_cast<PaddedCounter *>(
              words + layout.completed_dw1_tiles);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed_dx(dx_counter->value);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed_dw2(dw2_counter->value);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed_dw1(dw1_counter->value);
          task_kind_shared =
              completed_dx.load(::cuda::memory_order_acquire) ==
                          static_cast<unsigned long long>(geometry.mblocks) *
                              kDxTiles &&
                      completed_dw2.load(::cuda::memory_order_acquire) ==
                          static_cast<unsigned long long>(
                              geometry.active_experts) *
                              kDw2TilesPerExpert &&
                      completed_dw1.load(::cuda::memory_order_acquire) ==
                          static_cast<unsigned long long>(
                              geometry.active_experts) *
                              kDw1TilesPerExpert
                  ? kTerminalTask
                  : kNoTask;
        }
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
      phase_shared = *cluster.map_shared_rank(&phase_shared, 0U);
    }
    synchronization.mailbox_read_done();
    if (task_kind_shared == kTerminalTask) {
      if (threadIdx.x < kDwStoreIssuers)
        asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
      synchronization.cluster_sync();
      break;
    }
    if (task_kind_shared == kNoTask) {
      __nanosleep(backward_tuning::kQueueIdleNanosleep);
      Synchronization::cta_sync();
      continue;
    }

    const auto owner = task_owner_shared;
    const auto tile = task_tile_shared;
    if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dab)) {
      const auto descriptor = descriptors[owner];
      if (phase_shared == 1U) {
        if (Synchronization::kMathEnabled && threadIdx.x == 0U)
          atomicMin(&status->fc1_start_clock, device_globaltimer_ns());
        backward_compute_detail::run_k_mn_tile_tma_store<
            DabTileN, kStorageTileN, Synchronization>(
            dr_map, w2_map, u_store_map, descriptor.compact_row,
            descriptor.expert * static_cast<std::uint32_t>(a.hidden),
            static_cast<std::uint32_t>(a.hidden), descriptor.padded_row,
            tile * backward_tuning::kActivationClusterN * DabTileN, storage,
            activation_pipeline);
        // A queue entry covers one complete cluster task. Publish it only
        // after both CTAs have stored their disjoint output regions.
        synchronization.cluster_sync();
        if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
            cluster_rank == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed(words[layout.du_completed + owner]);
          const auto done =
              completed.fetch_add(1ULL, ::cuda::memory_order_acq_rel) + 1ULL;
          if (done == kDuTiles) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                phase(words[layout.phase + owner]);
            phase.store(2ULL, ::cuda::memory_order_release);
            if constexpr (!StaticSchedule)
              weftmoe::cuda::persistent_v3::backward::publish_owner_prefix(
                  queues.dab, owner, kDabEpilogueTasks);
          }
        }
      } else if (phase_shared == 2U) {
        const bool whole_block = StaticSchedule && task_count_shared > 1U;
        const auto rows_per_task =
            whole_block ? kDabRowsPerMblock : kDabRowsPerTask;
        const auto first_row = tile * rows_per_task;
        const auto task_rows = kDabRowsPerMblock - first_row < rows_per_task
                                   ? kDabRowsPerMblock - first_row
                                   : rows_per_task;
        const auto first_half = (task_rows + 1U) / 2U;
        const auto local_row = first_row + cluster_rank * first_half;
        const auto rows =
            cluster_rank == 0U ? first_half : task_rows - first_half;
        const auto valid_rows = descriptor.valid_rows > local_row
                                    ? (descriptor.valid_rows - local_row < rows
                                           ? descriptor.valid_rows - local_row
                                           : rows)
                                    : 0U;
        if constexpr (Synchronization::kMathEnabled)
          backward_swiglu_epilogue_warp_rows<
              StaticSchedule ? kDabRowsPerMblock / 2U
                             : backward_tuning::kEpilogueRowsPerTask,
              Synchronization::kWorkWarps>(
              a, static_cast<std::uint64_t>(descriptor.padded_row) + local_row,
              static_cast<std::uint64_t>(descriptor.compact_row) + local_row,
              valid_rows, rows);
        __threadfence();
        synchronization.cluster_sync();
        if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
            cluster_rank == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed(words[layout.dab_completed + owner]);
          const auto epilogue_count = StaticSchedule ? task_count_shared : 1U;
          const auto done =
              completed.fetch_add(epilogue_count, ::cuda::memory_order_acq_rel) +
              epilogue_count;
          if (done == kDabEpilogueTasks) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                phase(words[layout.phase + owner]);
            phase.store(3ULL, ::cuda::memory_order_release);
            if constexpr (!StaticSchedule) {
              weftmoe::cuda::persistent_v3::backward::publish_owner(queues.dx,
                                                                    owner);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_device>
                  expert_completed(words[layout.expert_mblocks_completed +
                                         descriptor.expert]);
              const auto expert_done = expert_completed.fetch_add(
                                           1ULL, ::cuda::memory_order_acq_rel) +
                                       1ULL;
              const auto *padded_rows = peer_at<const unsigned long long>(
                  a.local_base, a.padded_expert_rows_offset);
              if (expert_done ==
                  padded_rows[descriptor.expert] / backward_tuning::kTileM) {
                weftmoe::cuda::persistent_v3::backward::publish_owner(
                    queues.dw2, descriptor.expert);
                weftmoe::cuda::persistent_v3::backward::publish_owner(
                    queues.dw1, descriptor.expert);
              }
            }
            atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
          }
        }
      }
    } else if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dx)) {
      const auto descriptor = descriptors[owner];
      const auto dx_count =
          !StaticSchedule && DxClaimTiles == 1U ? 1U : task_count_shared;
      if (Synchronization::kMathEnabled && threadIdx.x == 0U)
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
      // All N tiles of this owner were published together. Each original
      // GEMM still drains its store before the next one reuses the arena.
      for (std::uint32_t offset = 0U; offset < dx_count; ++offset) {
        // CTA 0 multicasts into the peer arena. Wait for both CTAs' previous
        // stores before allowing either producer to reuse that arena.
        if (offset != 0U)
          synchronization.cluster_sync();
        backward_compute_detail::run_k_mn_tile_tma_store<DxTileN, kStorageTileN,
                                                         Synchronization>(
            dab_map, w1_map, dx_store_map, descriptor.padded_row,
            descriptor.expert * static_cast<std::uint32_t>(2ULL * a.intermediate),
            static_cast<std::uint32_t>(2ULL * a.intermediate),
            descriptor.padded_row,
            (tile + offset) * backward_tuning::kActivationClusterN * DxTileN, storage,
            activation_pipeline);
      }
      if (a.use_gin)
        __threadfence_system();
      else
        __threadfence();
      synchronization.cluster_sync();
      if (Synchronization::kMathEnabled && threadIdx.x == 0U &&
          cluster_rank == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            owner_completed(words[layout.dx_completed + owner]);
        const auto owner_done =
            owner_completed.fetch_add(dx_count, ::cuda::memory_order_acq_rel) +
            dx_count;
        if (owner_done == kDxTiles) {
          owner_completed.store(
              weftmoe::cuda::persistent_v3::backward::backward_dx_ready_value(
                  kDxTiles),
              ::cuda::memory_order_release);
          if (!StaticSchedule &&
              (!a.use_gin || Synchronization::kUnifiedReturnQueue))
            weftmoe::cuda::persistent_v3::backward::publish_owner(
                make_return_domain(a, words, layout, geometry.mblocks), owner);
        }
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dx_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done =
            completed.fetch_add(dx_count, ::cuda::memory_order_acq_rel) + dx_count;
        if (done ==
            static_cast<unsigned long long>(geometry.mblocks) * kDxTiles)
          atomicMax(&status->fc2_complete_clock, device_globaltimer_ns());
      }
    } else if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dw2)) {
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
      const auto last = persistent_v3::backward::cluster_tile(
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
      const auto last = persistent_v3::backward::cluster_tile(
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

} // namespace backward_unified_compute_detail
