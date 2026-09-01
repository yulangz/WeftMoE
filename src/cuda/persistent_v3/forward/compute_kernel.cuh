#pragma once

// Hopper-only Forward compute grid.  Transport owns generation setup and the
// immutable Dispatch/Return directories; this grid owns only FC1/FC2.  Keeping
// the WGMMA role in a separate kernel and an out-of-line device function is
// intentional: ptxas can now assign the math warpgroup and the TMA
// producer independent dynamic register quotas without carrying any GIN or
// relay live ranges through the WGMMA pipeline.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

namespace forward_compute_detail {

constexpr std::uint32_t kThreads = 384U;
constexpr std::uint32_t kMathThreads = 256U;
constexpr std::uint32_t kMathWarpGroups = 2U;
constexpr std::uint32_t kTaskReadyBarrier = 0U;
constexpr std::uint32_t kTaskDoneBarrier = 1U;
constexpr std::uint32_t kMathReadyBarrier = 2U;
constexpr std::uint32_t kMathDoneBarrier = 3U;
constexpr std::uint32_t kReduceRegisterReleaseBarrier = 4U;
constexpr std::uint32_t kReduceRegisterAcquireBarrier = 5U;
constexpr std::uint32_t kFc1Stages = 3U;
constexpr std::uint32_t kFc2Stages = 4U;
constexpr std::uint32_t kFc2Iterations = 3072U / 64U;
constexpr std::uint32_t kAStageElements = 128U * 64U;
constexpr std::uint32_t kFc1BStageElements = 128U * 64U;
constexpr std::uint32_t kFc2BStageElements = 256U * 64U;

enum ForwardComputeTaskKind : std::uint32_t {
  kComputeTaskNone = 0U,
  kComputeTaskFc1 = 1U,
  kComputeTaskFc2 = 2U,
  kComputeTaskDone = 3U,
};

// FC1/FC2 lends most of the CTA register file to the two WGMMA warpgroups:
// 2 * 128 * 232 + 128 * 40 == 384 * 168. Once the final GEMM tile has
// retired, those asymmetric roles no longer overlap with Reduce. Rebalance
// the same fixed register budget so every warp can run the token combiner
// without carrying the producer warpgroup's 40-register constraint into the
// tail. The named barriers are CTA-local; this is not a grid-wide phase
// barrier and Return/RDMA/Relay continue concurrently on their own CTAs.
__device__ __forceinline__ void rebalance_registers_for_reduce() {
  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup < kMathWarpGroups)
    cutlass::arch::warpgroup_reg_dealloc<168>();
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    kReduceRegisterReleaseBarrier);
  if (warpgroup >= kMathWarpGroups)
    cutlass::arch::warpgroup_reg_alloc<168>();
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    kReduceRegisterAcquireBarrier);
}

template <ForwardTaskPartition Partition>
__device__ __forceinline__ unsigned long long partition_mblock_begin(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  if constexpr (Partition == ForwardTaskPartition::Body)
    return *forward_mixed_scheduler_word(a, scheduler, kForwardHeadMblocks);
  return 0ULL;
}

template <ForwardTaskPartition Partition>
__device__ __forceinline__ unsigned long long partition_mblock_count(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  const auto total = scheduler[3];
  if constexpr (Partition == ForwardTaskPartition::All)
    return total;
  const auto head =
      *forward_mixed_scheduler_word(a, scheduler, kForwardHeadMblocks);
  if constexpr (Partition == ForwardTaskPartition::Head)
    return head;
  return total - head;
}

// M=64 maps the two resident math warpgroups across adjacent N slices instead
// of leaving the second warpgroup idle.  The callback sees the CTA-wide output
// column (WG0 followed by WG1), while both groups address the same 64 rows.
template <typename TileMma, class Callback>
__device__ __forceinline__ void visit_m64_accumulator(float *accum,
                                                       Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = warpgroup * TileMma::N + (lane % 4U) * 2U +
                        index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

template <typename TileMma, class Callback>
__device__ __forceinline__ void visit_m64_accumulator_pair(
    float *accum_a, float *accum_b, Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = warpgroup * TileMma::N + (lane % 4U) * 2U +
                        index * 8U;
    callback(warpgroup, row0, column, accum_a[index * 4U],
             accum_a[index * 4U + 1U], accum_b[index * 4U],
             accum_b[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum_a[index * 4U + 2U],
             accum_a[index * 4U + 3U], accum_b[index * 4U + 2U],
             accum_b[index * 4U + 3U]);
  }
}

template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc1_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long fc1_tasks,
    std::uint32_t fc1_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin =
      partition_mblock_begin<Partition>(a, scheduler) * fc1_ntiles;
  const auto partition_tasks =
      partition_mblock_count<Partition>(a, scheduler) * fc1_ntiles;
  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc1Claim)
          : &scheduler[0];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  auto slot = cursor.load(::cuda::memory_order_relaxed);
  if (slot >= partition_tasks) {
    exhausted = true;
    return false;
  }

  auto selected = partition_begin + slot;
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      auto *ready_queue = claims + forward_mblocks;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          published(ready_queue[partition_begin + slot]);
      const auto encoded = published.load(::cuda::memory_order_acquire);
      if (encoded == 0ULL) {
        exhausted = false;
        return false;
      }
      selected = encoded - 1ULL;
    }
  }

  if (!cursor.compare_exchange_strong(slot, slot + 1ULL,
                                      ::cuda::memory_order_acq_rel,
                                      ::cuda::memory_order_relaxed)) {
    exhausted = false;
    return false;
  }
  task = selected;
  exhausted = false;
#if MEGAMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      claim_completion_tasks(completion, transport_state::TaskClass::Fc1OrDv);
      completion->fc1_scheduler_value = slot;
      completion->bulk_local_ready = 11ULL;
    }
  }
#endif
  (void)fc1_tasks;
  return true;
}

template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc2_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *ready_queue, unsigned long long fc2_tasks,
    std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin = partition_mblock_begin<Partition>(a, scheduler);
  const auto partition_mblocks =
      partition_mblock_count<Partition>(a, scheduler);

  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc2Claim)
          : &scheduler[1];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  while (true) {
    auto slot = cursor.load(::cuda::memory_order_relaxed);
    if (slot >= partition_mblocks) {
      exhausted = true;
      return false;
    }

    const auto queue_slot = partition_begin + slot;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(ready_queue[queue_slot]);
    const auto encoded = published.load(::cuda::memory_order_acquire);
    if (encoded == 0ULL) {
      exhausted = false;
      return false;
    }

    // One immutable descriptor represents the ready M-block.  A fetch-add on
    // its private counter gives the first fc2_ntiles consumers unique N tiles
    // without making all compute CTAs CAS the same global head for every tile.
    // The first surplus consumer advances the M-block head; nobody reserves
    // work from an unpublished descriptor.
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        tile_cursor(ready_queue[forward_mblocks + queue_slot]);
    const auto n_tile = tile_cursor.fetch_add(
        1ULL, ::cuda::memory_order_relaxed);
    if (n_tile >= fc2_ntiles) {
      auto expected = slot;
      cursor.compare_exchange_strong(expected, slot + 1ULL,
                                     ::cuda::memory_order_relaxed,
                                     ::cuda::memory_order_relaxed);
      continue;
    }

    task = (encoded - 1ULL) * fc2_ntiles + n_tile;
    exhausted = false;
#if MEGAMOE_DEBUG_VALIDATE
    if constexpr (!StaticLocal) {
      if (a.use_gin)
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc2OrDx);
    }
    if (a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas) {
      completion->fc2_scheduler_value = slot * fc2_ntiles + n_tile;
      completion->forward_fc2_cta_tiles_started[blockIdx.x] = task + 1ULL;
      completion->forward_fc2_cta_last_task[blockIdx.x] = task;
      completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] = 1ULL << 32U;
    }
#endif
    (void)fc2_tasks;
    return true;
  }
}

// A CTA never reserves an unpublished slot.  It first tries FC2, then FC1,
// and repeats the FC2 probe before sleeping.  Thus every scheduling boundary
// favors an already-ready FC2 tile while all CTAs remain available to FC1
// during the initial Dispatch wavefront.
template <bool StaticLocal, bool DoFc1, bool DoFc2,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ void select_priority_compute_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long *fc2_ready_queue,
    unsigned long long fc1_tasks, unsigned long long fc2_tasks,
    std::uint32_t fc1_ntiles, std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long *task_shared, unsigned int *task_kind_shared) {
  if (threadIdx.x != 0U)
    return;
  while (!forward_cancelled(a)) {
    bool fc1_exhausted = !DoFc1;
    bool fc2_exhausted = !DoFc2;
    unsigned long long task = 0ULL;
    if constexpr (DoFc2) {
      if (try_claim_fc2_task<StaticLocal, Partition>(
              a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles,
              completion, task, fc2_exhausted)) {
        *task_shared = task;
        *task_kind_shared = kComputeTaskFc2;
        return;
      }
    }
    if constexpr (DoFc1) {
      if (try_claim_fc1_task<StaticLocal, Partition>(
              a, scheduler, claims, fc1_tasks, fc1_ntiles, completion, task,
              fc1_exhausted)) {
        *task_shared = task;
        *task_kind_shared = kComputeTaskFc1;
        return;
      }
    }
    if (fc1_exhausted && fc2_exhausted) {
      *task_kind_shared = kComputeTaskDone;
      return;
    }
    __nanosleep(64U);
  }
  *task_kind_shared = kComputeTaskDone;
}

template <bool StaticLocal, bool DoFc1, bool DoFc2, bool M64 = false,
          ForwardTaskPartition Partition = ForwardTaskPartition::All,
          bool TmaFc2Store = false, bool TmaFc2StoreToStaging = false,
          std::uint32_t Fc1StageCount = kFc1Stages,
          std::uint32_t Fc2StageCount = kFc2Stages>
__device__ __forceinline__ void
persistent_role(const ForwardKernelArgs &a, const CUtensorMap &x_map,
                const CUtensorMap &s_map, const CUtensorMap &w1_map,
                const CUtensorMap &w2_map, unsigned char *storage,
                unsigned long long *task_shared,
                unsigned int *task_kind_shared,
                unsigned int *return_mblock_shared,
                const CUtensorMap *output_store_map = nullptr) {
  static_assert(!TmaFc2StoreToStaging || TmaFc2Store,
                "staging specialization requires the TMA FC2 epilogue");
  static_assert(Fc1StageCount >= 2U && Fc1StageCount <= kFc2Stages,
                "FC1 supports two to four TMA stages");
  static_assert(Fc2StageCount >= 2U && Fc2StageCount <= kFc2Stages,
                "FC2 supports two to four TMA stages");
  static_assert(kFc2Iterations % Fc2StageCount == 0U,
                "the fixed FC2 reduction must divide evenly into stages");
  constexpr auto kConfiguredFc2Waves =
      kFc2Iterations / Fc2StageCount;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  auto *ab_cache = a.forward_ab_cache_offset == 0
                       ? static_cast<transport::BFloat16 *>(nullptr)
                       : peer_at<transport::BFloat16>(
                             a.local_base, a.forward_ab_cache_offset);
  auto *forward_output =
      a.use_gin && a.forward_registered_output
          ? reinterpret_cast<transport::BFloat16 *>(
                static_cast<std::byte *>(a.gin_window_base) +
                a.gin_return_staging_offset)
          : peer_at<transport::BFloat16>(a.local_base,
                                         a.forward_output_rows_offset);
  auto *direct_return_staging =
      a.use_gin
          ? reinterpret_cast<transport::BFloat16 *>(
                static_cast<std::byte *>(a.gin_window_base) +
                a.gin_return_staging_offset)
          : static_cast<transport::BFloat16 *>(nullptr);
  auto *headers =
      peer_at<LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  auto *return_rows_by_source = reinterpret_cast<unsigned int *>(
      storage +
      (4U * kAStageElements + 4U * kFc2BStageElements) *
          sizeof(transport::BFloat16) +
      20U * sizeof(std::uint64_t));
  // This naturally aligned scratch word is dead until the compatibility local
  // Return copy starts. Reuse it to build the immutable descriptor's remote
  // and local context masks in parallel; this is the explicit shared-memory
  // union for the two non-overlapping roles.
  auto *return_context_masks_shared =
      reinterpret_cast<unsigned long long *>(return_rows_by_source);

  transport_state::GenerationCompletionVector *completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin)
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
  }

  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  const auto fc2_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *fc2_ready_queue = fc1_ready_queue + fc1_tasks;

  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + 4U * kAStageElements;
  auto *fc1_b2 = smem_b + Fc1StageCount * kFc1BStageElements;
  auto *fc1_full =
      reinterpret_cast<ForwardBarrier *>(smem_b + 4U * kFc2BStageElements);
  auto *fc1_empty = fc1_full + Fc1StageCount;
  auto *fc2_full = fc1_empty + Fc1StageCount;
  auto *fc2_empty = fc2_full + Fc2StageCount;
  auto *fc1_b2_full = fc2_empty + Fc2StageCount;

  if (threadIdx.x == kMathThreads) {
    for (std::uint32_t stage = 0; stage < Fc1StageCount; ++stage) {
      fc1_full[stage].init(1);
      fc1_empty[stage].init(kMathThreads / 32U);
      fc1_b2_full[stage].init(1);
    }
    for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
      fc2_full[stage].init(1);
      fc2_empty[stage].init(kMathThreads / 32U);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup >= kMathWarpGroups) {
    // Only the first producer-warpgroup lane issues TMA; the remaining lanes carry
    // the CTA task/barrier protocol with a deliberately small register set.
    cutlass::arch::warpgroup_reg_dealloc<40>();
    std::uint32_t fc1_pipeline = 0U;
    std::uint32_t fc2_pipeline = 0U;
    while (true) {
      if constexpr (DoFc1) {
      while (true) {
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
        const auto task = *task_shared;
        if (*task_kind_shared != kComputeTaskFc1)
          break;
        const auto mblock = task / fc1_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        const auto row = a.dispatch.expert_row_prefix[expert] +
                         forward_mblock_local_row(a, expert, local_mblock);
        if (threadIdx.x == kMathThreads) {
          // The readiness acquire is performed by the scalar scheduler lane and
          // handed to this producer through the CTA named barrier.  Make the
          // generic/system writes visible to the TMA async proxy before issuing
          // the first load for the claimed M block.
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto iterations =
              static_cast<std::uint32_t>((a.hidden + 63ULL) / 64ULL);
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto current = fc1_pipeline + iteration;
            const auto stage = current % Fc1StageCount;
            const auto phase = (current / Fc1StageCount) & 1U;
            fc1_empty[stage].wait(phase ^ 1U);
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements);
            auto *stage_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b + stage * kFc1BStageElements);
            auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
                fc1_b2 + stage * kFc1BStageElements);
            const auto k_base = iteration * 64U;
            const auto n = n_tile * 128U;
            if constexpr (M64) {
              deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                  &x_map, &fc1_full[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row));
            } else {
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &x_map, &fc1_full[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row));
            }
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &w1_map, &fc1_full[stage], stage_b1, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n));
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &w1_map, &fc1_b2_full[stage], stage_b2, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                           a.intermediate + n));
            constexpr auto a_stage_elements =
                M64 ? 64U * 64U : kAStageElements;
            fc1_full[stage].arrive_and_expect_tx(
                (a_stage_elements + kFc1BStageElements) *
                sizeof(transport::BFloat16));
            fc1_b2_full[stage].arrive_and_expect_tx(
                kFc1BStageElements * sizeof(transport::BFloat16));
          }
          fc1_pipeline += iterations;
        }
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      }
      }

      if (*task_kind_shared == kComputeTaskDone)
        break;

      if constexpr (DoFc2) {
      while (true) {
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
        const auto task = *task_shared;
        if (*task_kind_shared != kComputeTaskFc2)
          break;
        const auto mblock = task / fc2_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        const auto row = a.dispatch.expert_row_prefix[expert] +
                         forward_mblock_local_row(a, expert, local_mblock);
        if (threadIdx.x == kMathThreads) {
          // The production shape has exactly twelve four-stage K waves.  Keep
          // the stage ordinal compile-time-unrolled so ptxas does not clone the
          // WGMMA path behind runtime modulo/parity branches.
          for (std::uint32_t wave = 0; wave < kConfiguredFc2Waves; ++wave) {
#pragma unroll
            for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
              const auto phase = wave & 1U;
              fc2_empty[stage].wait(phase ^ 1U);
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements);
              auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b + stage * kFc2BStageElements);
              const auto k_base = (wave * Fc2StageCount + stage) * 64U;
              const auto n = n_tile * 256U;
              if constexpr (M64) {
                deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                    &s_map, &fc2_full[stage], stage_a, k_base,
                    static_cast<std::uint32_t>(row));
              } else {
                deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                    &s_map, &fc2_full[stage], stage_a, k_base,
                    static_cast<std::uint32_t>(row));
              }
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &w2_map, &fc2_full[stage], stage_b, k_base,
                  static_cast<std::uint32_t>(expert * a.hidden + n));
              constexpr auto a_stage_elements =
                  M64 ? 64U * 64U : kAStageElements;
              fc2_full[stage].arrive_and_expect_tx(
                  (a_stage_elements + kFc2BStageElements) *
                  sizeof(transport::BFloat16));
            }
          }
          fc2_pipeline += kFc2Iterations;
        }
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      }
      }

      if (*task_kind_shared == kComputeTaskDone)
        break;
    }
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<232>();
  std::uint32_t fc1_pipeline = 0U;
  std::uint32_t fc2_pipeline = 0U;
#if MEGAMOE_DEVICE_TIMELINE
  bool timeline_fc1_started = false;
  bool timeline_fc2_started = false;
#endif
  select_priority_compute_task<StaticLocal, DoFc1, DoFc2, Partition>(
      a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks, fc1_ntiles,
      fc2_ntiles, completion, task_shared, task_kind_shared);
  while (true) {
    if constexpr (DoFc1) {
    while (true) {
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
      const auto task = *task_shared;
      if (*task_kind_shared != kComputeTaskFc1)
        break;
#if MEGAMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U && !timeline_fc1_started) {
        MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstMtileReady, Compute, 0U, task,
                                a.epoch);
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1First, Compute, 0U, task,
                                a.epoch);
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1ClaimStart, Compute, 0U, task,
                                a.epoch);
        timeline_fc1_started = true;
      }
#endif
      if (threadIdx.x == 0U)
        atomicMin(&status->fc1_start_clock, device_globaltimer_ns());
      const auto mblock = task / fc1_ntiles;
      const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto row = a.dispatch.expert_row_prefix[expert] +
                       forward_mblock_local_row(a, expert, local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      const auto n = n_tile * 128U;
      using Fc1TileMma =
          typename std::conditional<M64, ForwardMmaNarrow, ForwardMma>::type;
      float accum_a[Fc1TileMma::kNumAccum];
      float accum_b[Fc1TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc1TileMma::kNumAccum; ++index) {
        accum_a[index] = 0.0F;
        accum_b[index] = 0.0F;
      }
      const auto iterations =
          static_cast<std::uint32_t>((a.hidden + 63ULL) / 64ULL);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto current = fc1_pipeline + iteration;
        const auto stage = current % Fc1StageCount;
        const auto phase = (current / Fc1StageCount) & 1U;
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_a + stage * kAStageElements +
            (M64 ? 0U : warpgroup * 64U * 64U));
        auto *stage_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_b + stage * kFc1BStageElements +
            (M64 ? warpgroup * 64U * 64U : 0U));
        auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
            fc1_b2 + stage * kFc1BStageElements +
            (M64 ? warpgroup * 64U * 64U : 0U));
        // A and gate share the first completion barrier, while up has its own.
        // Start the gate WGMMA group as soon as 32 KiB is ready so its issue
        // overlaps the tail of the independent 16 KiB up-weight TMA.
        fc1_full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0; k < 64U / Fc1TileMma::K; ++k) {
          const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
              stage_a + k * Fc1TileMma::K, 1);
          const auto desc_b1 = deep_gemm::mma::sm90::make_smem_desc(
              stage_b1 + k * Fc1TileMma::K, 1);
          Fc1TileMma::wgmma(desc_a, desc_b1, accum_a, 1);
        }
        deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);

        fc1_b2_full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
        deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0; k < 64U / Fc1TileMma::K; ++k) {
          const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
              stage_a + k * Fc1TileMma::K, 1);
          const auto desc_b2 = deep_gemm::mma::sm90::make_smem_desc(
              stage_b2 + k * Fc1TileMma::K, 1);
          Fc1TileMma::wgmma(desc_a, desc_b2, accum_b, 1);
        }
        deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
        deep_gemm::ptx::warpgroup_wait<0>();
        if ((threadIdx.x & 31U) == 0U)
          fc1_empty[stage].arrive();
      }
      fc1_pipeline += iterations;

      const auto store_fc1 =
          [&](std::uint32_t wg, std::uint32_t m, std::uint32_t column,
              float av0, float av1, float bv0, float bv1) {
            const auto block_m = M64 ? m : wg * 64U + m;
            const auto output_column = static_cast<std::uint64_t>(n) + column;
            if (block_m < valid_rows && output_column + 1ULL < a.intermediate &&
                (a.intermediate & 1ULL) == 0ULL) {
              const auto output_index =
                  (row + block_m) * a.intermediate + output_column;
              const auto sigmoid0 = 1.0F / (1.0F + __expf(-av0));
              const auto sigmoid1 = 1.0F / (1.0F + __expf(-av1));
              reinterpret_cast<std::uint32_t *>(s)[output_index / 2ULL] =
                  float2_to_bf16x2_bits((av0 * sigmoid0) * bv0,
                                        (av1 * sigmoid1) * bv1);
              if (a.cache_fc1 && ab_cache) {
                reinterpret_cast<std::uint32_t *>(
                    ab_cache)[output_index / 2ULL] =
                    float2_to_bf16x2_bits(av0, av1);
                const auto b_index =
                    a.max_destination_rows * a.intermediate + output_index;
                reinterpret_cast<std::uint32_t *>(ab_cache)[b_index / 2ULL] =
                    float2_to_bf16x2_bits(bv0, bv1);
              }
            } else if (block_m < valid_rows) {
              const float av[2] = {av0, av1};
              const float bv[2] = {bv0, bv1};
#pragma unroll
              for (std::uint32_t value = 0; value < 2U; ++value) {
                const auto tail_column = output_column + value;
                if (tail_column < a.intermediate) {
                  const auto output_index =
                      (row + block_m) * a.intermediate + tail_column;
                  const auto sigmoid = 1.0F / (1.0F + __expf(-av[value]));
                  s[output_index] = transport::BFloat16{
                      float_to_bf16_bits((av[value] * sigmoid) * bv[value])};
                  if (a.cache_fc1 && ab_cache) {
                    ab_cache[output_index] =
                        transport::BFloat16{float_to_bf16_bits(av[value])};
                    ab_cache[a.max_destination_rows * a.intermediate +
                             output_index] =
                        transport::BFloat16{float_to_bf16_bits(bv[value])};
                  }
                }
              }
            }
          };
      if constexpr (M64)
        visit_m64_accumulator_pair<Fc1TileMma>(accum_a, accum_b, store_fc1);
      else
        visit_forward_accumulator_pair(accum_a, accum_b, store_fc1);
      __threadfence();
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            ready(fc1_done[mblock]);
        const auto old = ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if MEGAMOE_DEBUG_VALIDATE
        if constexpr (!StaticLocal) {
          if (a.use_gin)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc1OrDv);
        }
#endif
        if ((old & 0xffffULL) + 1ULL == fc1_ntiles) {
          const auto fc1_mblocks_complete =
              atomicAdd(&scheduler[11ULL + 2ULL * a.world], 1ULL) + 1ULL;
          if (fc1_mblocks_complete == forward_mblocks) {
            status->fc1_complete_clock = device_globaltimer_ns();
            MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1Last, Compute, 0U,
                                    fc1_tasks, a.epoch);
            forward_publish_progress(a, kForwardFc1Complete, fc1_tasks);
          }
          const auto head_mblocks = a.forward_m64_head_wavefront
                                        ? *forward_mixed_scheduler_word(
                                              a, scheduler,
                                              kForwardHeadMblocks)
                                        : forward_mblocks;
          const bool body_task =
              a.forward_m64_head_wavefront && mblock >= head_mblocks;
          auto *publish_tail =
              body_task
                  ? forward_mixed_scheduler_word(a, scheduler,
                                                 kForwardBodyFc2Publish)
                  : &scheduler[2];
          const auto partition_base = body_task ? head_mblocks : 0ULL;
          const auto queue_slot =
              partition_base + atomicAdd(publish_tail, 1ULL);
#if MEGAMOE_DEBUG_VALIDATE
          if (a.use_gin) {
            atomicAdd(&completion->progress_ctas_at_bulk_return, 1ULL);
            atomicMax(&completion->compute_ctas_at_bulk_return,
                      queue_slot + 1ULL);
          }
#endif
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              queue_entry(fc2_ready_queue[queue_slot]);
          queue_entry.store(mblock + 1ULL, ::cuda::memory_order_release);
        }
      }
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      select_priority_compute_task<StaticLocal, DoFc1, DoFc2, Partition>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared);
    }
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;

    if constexpr (DoFc2) {
    while (true) {
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
      if (*task_kind_shared != kComputeTaskFc2)
        break;
#if MEGAMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U && !timeline_fc2_started) {
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2First, Compute, 0U,
                                *task_shared, a.epoch);
        timeline_fc2_started = true;
      }
#endif
      if (threadIdx.x == 0U)
        atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
      using Fc2TileMma =
          typename std::conditional<M64, ForwardMma, ForwardMmaWide>::type;
      float accum[Fc2TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
        accum[index] = 0.0F;
      for (std::uint32_t wave = 0; wave < kConfiguredFc2Waves; ++wave) {
#pragma unroll
        for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
          const auto phase = wave & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements +
              (M64 ? 0U : warpgroup * 64U * 64U));
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kFc2BStageElements +
              (M64 ? warpgroup * 128U * 64U : 0U));
          fc2_full[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0; k < 64U / Fc2TileMma::K; ++k) {
            const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                stage_a + k * Fc2TileMma::K, 1);
            const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                stage_b + k * Fc2TileMma::K, 1);
            Fc2TileMma::wgmma(desc_a, desc_b, accum, 1);
          }
          deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          // TMA remains four-stage asynchronous in the producer warpgroup.  The
          // math side completes one K=64 group before recycling that stage; the
          // previous wait<1> needed a second live accumulator generation and
          // made ptxas serialize every individual WGMMA instruction.
          deep_gemm::ptx::warpgroup_wait<0>();
          if ((threadIdx.x & 31U) == 0U)
            fc2_empty[stage].arrive();
        }
      }
      fc2_pipeline += kFc2Iterations;

      // Reload immutable task metadata only after the WGMMA mainloop.  This is
      // the register analogue of a role union: task/return bookkeeping reuses
      // registers that the asynchronous accumulator pipeline no longer needs.
      const auto task = *task_shared;
      const auto mblock = task / fc2_ntiles;
      const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto row = a.dispatch.expert_row_prefix[expert] +
                       forward_mblock_local_row(a, expert, local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      const auto n = n_tile * 256U;

      const bool full_tma_tile = TmaFc2Store && !M64 && valid_rows == 128U &&
                                 n + 256ULL <= a.hidden;
      if constexpr (TmaFc2Store) {
        if (full_tma_tile) {
          // Reuse the 64 KiB FC2 input-stage union after its final consumer.
          // STSM converts the accumulator fragments directly to the 128-byte
          // swizzle expected by four 64x128 TMA stores.  Partial expert tails
          // deliberately stay on the checked scalar path: expert prefixes are
          // compact rather than padded, so a 128-row store there would clobber
          // the following expert.
          auto *output_tile = smem_a;
          const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
          const auto lane = threadIdx.x & 31U;
#pragma unroll
          for (std::uint32_t index = 0;
               index < ForwardMmaWide::kNumAccum / 4U; ++index) {
            constexpr std::uint32_t kBankGroupBytes = 16U;
            const auto atom = index / (64U / 8U);
            const auto in_atom = index % (64U / 8U);
            const auto swizzle_row = lane;
            auto swizzle_column = in_atom;
            swizzle_column ^= swizzle_row % (128U / kBankGroupBytes);
            auto *smem_pointer =
                reinterpret_cast<std::uint8_t *>(output_tile) +
                warp * 16U * 128U + atom * 128U * 128U +
                swizzle_row * 128U + swizzle_column * kBankGroupBytes;
            deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
                __float22bfloat162_rn(
                    make_float2(accum[index * 4U], accum[index * 4U + 1U])),
                __float22bfloat162_rn(make_float2(accum[index * 4U + 2U],
                                                   accum[index * 4U + 3U])),
                smem_pointer);
          }
          cute::tma_store_fence();
          cutlass::arch::NamedBarrier::sync(kMathThreads,
                                            kMathReadyBarrier);
          if (threadIdx.x < 4U) {
            const auto atom = threadIdx.x;
            cute::SM90_TMA_STORE_2D::copy(
                output_store_map, output_tile + atom * 128U * 64U,
                static_cast<std::uint32_t>(n + atom * 64ULL),
                static_cast<std::uint32_t>(row));
            cute::tma_store_arrive();
            // The unqualified wait both protects the shared-memory union from
            // the next task and completes the async-to-generic proxy handoff
            // needed before FC2 publishes its completion counter.
            asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
          }
          cutlass::arch::NamedBarrier::sync(kMathThreads,
                                            kMathDoneBarrier);
        }
      }
      // When TMA targets a separate Return staging plane, retain the scalar
      // canonical store for local/backward consumers. Production aliases its
      // canonical output with registered staging, so a full tile needs only
      // the TMA store; partial expert tails remain on the checked scalar path.
      if (!full_tma_tile || TmaFc2StoreToStaging) {
        const auto store_fc2 =
            [&](std::uint32_t wg, std::uint32_t m, std::uint32_t column,
                float v0, float v1) {
              const auto output_column = static_cast<std::uint64_t>(n) + column;
              const auto block_m = M64 ? m : wg * 64U + m;
              const auto output_row = row + block_m;
              if (block_m < valid_rows && output_column + 1ULL < a.hidden) {
                const auto packed = float2_to_bf16x2_bits(v0, v1);
                reinterpret_cast<std::uint32_t *>(
                    forward_output + output_row * a.hidden + output_column)[0] =
                    packed;
                if (!full_tma_tile && direct_return_staging != nullptr &&
                    direct_return_staging != forward_output &&
                    headers[output_row].source_rank / a.local_world_size !=
                        a.node)
                  reinterpret_cast<std::uint32_t *>(
                      direct_return_staging + output_row * a.hidden +
                      output_column)[0] = packed;
              } else if (block_m < valid_rows && output_column < a.hidden) {
                const auto value =
                    transport::BFloat16{float_to_bf16_bits(v0)};
                forward_output[output_row * a.hidden + output_column] = value;
                if (!full_tma_tile && direct_return_staging != nullptr &&
                    direct_return_staging != forward_output &&
                    headers[output_row].source_rank / a.local_world_size !=
                        a.node)
                  direct_return_staging[output_row * a.hidden + output_column] =
                      value;
              }
            };
        if constexpr (M64)
          visit_m64_accumulator<Fc2TileMma>(accum, store_fc2);
        else
          visit_forward_accumulator_wide(accum, store_fc2);
      }
      // N-tiles synchronize only through this GPU-local completion chain.
      // A device fence is sufficient here; the final N-tile acquires the
      // chain and performs one system fence for the whole M-block below.
      __threadfence();
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            complete(fc1_done[mblock]);
        const auto old =
            complete.fetch_add(1ULL << 16U, ::cuda::memory_order_acq_rel);
        *return_mblock_shared =
            (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
        if constexpr (!StaticLocal) {
          if (a.use_gin)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc2OrDx);
        }
#endif
      }
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);

      if (threadIdx.x == 0U)
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                                a.epoch);

      if (*return_mblock_shared != 0U) {
        // Thread 0's acq_rel RMW above acquired every earlier N-tile release.
        // Publish that transitive chain, plus this CTA's stores, to NVLink/GIN
        // once per M-block before any Return-ready counter becomes visible.
        __threadfence_system();
        if (threadIdx.x == 0U)
          *return_context_masks_shared = 0ULL;
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);

        // Exact masks prevent seven unrelated context owners from rescanning
        // this M-block. Build them with one header load per row and one shared
        // OR per warp instead of a 128-load scalar loop on the last FC2 tile.
        std::uint32_t remote_context_bit = 0U;
        std::uint32_t local_context_bit = 0U;
        if (threadIdx.x < valid_rows) {
          const auto source = headers[row + threadIdx.x].source_rank;
          if (source < a.world) {
            const auto context = source % a.local_world_size;
            if (source / a.local_world_size != a.node)
              remote_context_bit = 1U << context;
            else if (a.forward_relay_local_return)
              local_context_bit = 1U << context;
          }
        }
        const auto warp_remote =
            __reduce_or_sync(0xffffffffU, remote_context_bit);
        const auto warp_local =
            __reduce_or_sync(0xffffffffU, local_context_bit);
        if ((threadIdx.x & 31U) == 0U)
          atomicOr(return_context_masks_shared,
                   static_cast<unsigned long long>(warp_remote) |
                       (static_cast<unsigned long long>(warp_local) << 32U));
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);

        if (threadIdx.x == 0U) {
          const auto finished =
              atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
          if (finished == forward_mblocks) {
            status->fc2_complete_clock = device_globaltimer_ns();
            forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
          }
          if constexpr (!StaticLocal) {
            if (a.use_gin) {
              // Lane 0 publishes one maximal contiguous descriptor to the
              // fixed FULL GIN owners. No context state or network register
              // live range enters this WGMMA kernel.
              forward_return_detail::issue_direct_full_mblock(
                  a, scheduler, row, valid_rows,
                  *return_context_masks_shared);
              MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2MblockReturnReady,
                                      Compute, 0U, mblock, a.epoch);
            }
          }
        }
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);

        // In the split GIN path Relay CTAs consume the same immutable M-block
        // descriptor as RDMA and own all node-local NVLink Return work.  Keep
        // this compatibility copy only for the non-GIN local P2P path; a math
        // CTA must become schedulable again immediately after publication.
        if (!a.use_socket &&
            (!a.use_gin || !a.forward_relay_local_return)) {
          for (std::uint32_t source = threadIdx.x; source < a.world;
               source += kMathThreads)
            return_rows_by_source[source] = 0U;
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
          for (std::uint32_t local_row = 0; local_row < valid_rows;
               ++local_row) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (a.use_gin && header.source_rank / a.local_world_size != a.node)
              continue;
            auto *source_contribution = peer_at<transport::BFloat16>(
                a.peer_bases[header.source_rank], a.contribution_offset);
            const auto *source128 = reinterpret_cast<const uint4 *>(
                forward_output + return_row * a.hidden);
            auto *target128 = reinterpret_cast<uint4 *>(
                source_contribution + header.route_index * a.hidden);
            const auto vector_count = a.hidden / 8ULL;
            for (std::uint64_t vector = threadIdx.x; vector < vector_count;
                 vector += kMathThreads)
              target128[vector] = source128[vector];
            if (threadIdx.x == 0U)
              ++return_rows_by_source[header.source_rank];
          }
          __threadfence_system();
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
          // Preserve the rank-wide signal below as the terminal protocol, but
          // expose each local-node contribution independently for Reduce.
          for (std::uint32_t local_row = threadIdx.x;
               local_row < valid_rows; local_row += kMathThreads) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (!a.use_gin ||
                header.source_rank / a.local_world_size == a.node) {
              auto *source_route_ready = peer_at<unsigned long long>(
                  a.peer_bases[header.source_rank],
                  a.return_ready_epoch_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  route_ready(source_route_ready[header.route_index]);
              if (a.forward_dynamic_reduce_queue)
                forward_reduce_detail::publish_ready_route(
                    a, a.peer_bases[header.source_rank], header.route_index,
                    3U);
              else
                route_ready.store(a.epoch, ::cuda::memory_order_release);
            }
          }
          for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
               source_rank += kMathThreads) {
            if (a.use_gin && source_rank / a.local_world_size != a.node)
              continue;
            const auto rows = return_rows_by_source[source_rank];
            if (rows == 0U)
              continue;
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                completed(scheduler[4U + source_rank]);
            const auto old =
                completed.fetch_add(rows, ::cuda::memory_order_acq_rel);
            const auto expected = old >> 32U;
            const auto actual = (old & 0xffffffffULL) + rows;
            if (actual == expected) {
              auto *source_signal = peer_at<unsigned long long>(
                  a.peer_bases[source_rank], a.outgoing_sequence_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  ready(source_signal[a.rank * a.channels]);
              ready.store(a.epoch, ::cuda::memory_order_release);
            }
          }
          cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
        }
      }

#if MEGAMOE_DEBUG_VALIDATE
      if (a.use_gin && threadIdx.x == 0U &&
          blockIdx.x < transport_state::kForwardDebugCtas) {
        completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
            (5ULL << 32U) | fc2_pipeline;
        completion->forward_fc2_cta_tiles_completed[blockIdx.x] = task + 1ULL;
      }
#endif
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      select_priority_compute_task<StaticLocal, DoFc1, DoFc2, Partition>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared);
    }
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;
  }
#if MEGAMOE_DEVICE_TIMELINE
  if (threadIdx.x == 0U && timeline_fc2_started)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2Last, Compute, 0U, fc2_tasks,
                            a.epoch);
#endif
}

} // namespace forward_compute_detail

#endif // CUTE_ARCH_MMA_SM90A_ENABLED

#include "cuda/persistent_v3/forward/reduce_pipeline.cuh"

template <bool StaticLocal, bool SplitStages = false, bool M64 = false,
          ForwardTaskPartition Partition = ForwardTaskPartition::All,
          bool ReduceStage = true, bool Fc2Only = false,
          bool Fc1Only = false, std::uint32_t Fc1StageCount = 3U,
          std::uint32_t Fc2StageCount = 4U>
__global__ __launch_bounds__(384, 1) void local_p2p_forward_compute_kernel(
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
      __nanosleep(256U);
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
  } else if constexpr (SplitStages) {
    // Every compute CTA is work-conserving. At each tile boundary it probes
    // FC2 first and borrows FC1 only when no FC2 tile is currently published.
    forward_compute_detail::persistent_role<
        StaticLocal, true, true, M64, Partition, false, false, Fc1StageCount,
        Fc2StageCount>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  } else {
    forward_compute_detail::persistent_role<
        StaticLocal, true, true, M64, Partition, false, false, Fc1StageCount,
        Fc2StageCount>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  }
  if constexpr (ReduceStage) {
    forward_compute_detail::rebalance_registers_for_reduce();
    const auto reduce_role_index =
        a.forward_reduce_cta_offset + static_cast<std::uint32_t>(blockIdx.x);
    forward_compute_detail::reduce_role<StaticLocal>(
        a, reduce_role_index,
        a.forward_reduce_cta_count != 0U
            ? a.forward_reduce_cta_count
            : static_cast<std::uint32_t>(gridDim.x));
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
template <bool StaticLocal, bool DoFc1,
          bool StoreToStaging = false>
__global__ __launch_bounds__(384, 1) void
local_p2p_forward_tma_store_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap output_store_map) {
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
      __nanosleep(256U);
  }
  __syncthreads();
  forward_compute_detail::persistent_role<
      StaticLocal, DoFc1, true, false, ForwardTaskPartition::All, true,
      StoreToStaging, 4U, 4U>(
      a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
      &task_kind_shared, &return_mblock_shared, &output_store_map);
  forward_compute_detail::rebalance_registers_for_reduce();
  const auto reduce_role_index =
      a.forward_reduce_cta_offset + static_cast<std::uint32_t>(blockIdx.x);
  forward_compute_detail::reduce_role<StaticLocal>(
      a, reduce_role_index,
      a.forward_reduce_cta_count != 0U
          ? a.forward_reduce_cta_count
          : static_cast<std::uint32_t>(gridDim.x));
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
  (void)output_store_map;
#endif
}

// A small concurrent grid owns only the two M64 halves of each expert's first
// physical M128 block.  It exits after FC2 publication, releasing its SMs to
// queued M128 body CTAs; Return and Reduce retain their existing owners.
template <bool StaticLocal>
__global__ __launch_bounds__(384, 1) void local_p2p_forward_m64_head_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 64ULL,
                            a.epoch);
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
      __nanosleep(256U);
  }
  __syncthreads();
  forward_compute_detail::persistent_role<
      StaticLocal, true, true, true,
      ForwardTaskPartition::Head>(
      a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
      &task_kind_shared, &return_mblock_shared);
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 64ULL, a.epoch);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
#endif
}

// FC1 and FC2 have no useful overlap in the production wavefront: all FC1
// tasks drain before the Return directory admits FC2.  Separate stream-ordered
// entries give ptxas independent WGMMA register/control-flow scopes without
// changing communication concurrency or adding a device-wide grid barrier.
template <bool StaticLocal, bool Fc1Stage>
__global__ __launch_bounds__(256, 1) void local_p2p_forward_stage_kernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U,
                            Fc1Stage ? 1ULL : 2ULL, a.epoch);
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
      __nanosleep(256U);
  }
  __syncthreads();
  if constexpr (Fc1Stage) {
    forward_compute_detail::persistent_role<StaticLocal, true, false>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
  } else {
    forward_compute_detail::persistent_role<StaticLocal, false, true>(
        a, x_map, s_map, w1_map, w2_map, storage, &task_shared,
        &task_kind_shared, &return_mblock_shared);
    // Preserve the stable persistent lifecycle: transport finalization waits
    // for one reduction release from every compute CTA.  FC2 and Reduce can
    // have separate compiler scopes, but must remain the same resident role.
    forward_compute_detail::reduce_role<StaticLocal>(
        a, static_cast<std::uint32_t>(blockIdx.x),
        static_cast<std::uint32_t>(gridDim.x));
  }
  if (threadIdx.x == 0U)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U,
                            Fc1Stage ? 1ULL : 2ULL, a.epoch);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
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
      __nanosleep(256U);
  }
  __syncthreads();
  forward_compute_detail::reduce_role<StaticLocal>(
      a, static_cast<std::uint32_t>(blockIdx.x),
      static_cast<std::uint32_t>(gridDim.x));
#else
  (void)a;
#endif
}
