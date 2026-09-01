#pragma once

// Compile-isolated persistent dW roles.  Production launches activation, dW2
// and dW1 phases serially on one compute stream; each phase owns the complete
// Compute budget while the fixed transport grid keeps progressing.  Each role
// therefore owns one WGMMA layout, never performs a grid-wide barrier, and
// cannot contend with another Tensor Core kernel.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
enum class BackwardV3WeightKind : std::uint32_t { Dw1, Dw2 };

template <BackwardV3WeightKind Kind, std::uint32_t TileN,
          bool KMajorRows = false>
struct BackwardV3WeightMma;

template <std::uint32_t TileN, bool KMajorRows>
struct BackwardV3WeightMma<BackwardV3WeightKind::Dw1, TileN,
                           KMajorRows> {
  using Type = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN, cute::UMMA::Major::K,
      KMajorRows ? cute::UMMA::Major::K
                 : cute::UMMA::Major::MN>::type;
};

template <std::uint32_t TileN, bool KMajorRows>
struct BackwardV3WeightMma<BackwardV3WeightKind::Dw2, TileN,
                           KMajorRows> {
  using Type = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN,
      KMajorRows ? cute::UMMA::Major::K
                 : cute::UMMA::Major::MN,
      cute::UMMA::Major::K>::type;
};

template <BackwardV3WeightKind Kind>
__device__ __forceinline__ unsigned long long
backward_v3_claim_weight_role_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    unsigned long long *expert_dv_complete,
    unsigned long long *expert_cursor, std::uint32_t dv_ntiles,
    std::uint64_t per_expert) {
  if (backward_v3_generation_cancelled(a, control))
    return kBackwardV3DoneTask;
  if constexpr (Kind == BackwardV3WeightKind::Dw1) {
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.dw1_tasks_complete.value) == control.dw1_tasks)
      return kBackwardV3DoneTask;
    return backward_v3_try_claim_dw1_task(
        a, control, expert_dv_complete, expert_cursor, dv_ntiles,
        per_expert);
  } else {
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.dw2_tasks_complete.value) == control.dw2_tasks)
      return kBackwardV3DoneTask;
    return backward_v3_try_claim_dw2_task(
        a, control, expert_dv_complete, expert_cursor, dv_ntiles,
        per_expert);
  }
}

template <BackwardV3WeightKind Kind>
__device__ __forceinline__ unsigned long long
backward_v3_claim_cluster_weight_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t pair_task, std::uint32_t mtiles,
    std::uint32_t ntiles) {
  if (backward_v3_generation_cancelled(a, control))
    return kBackwardV3DoneTask;
  const auto pair_mtiles = mtiles / 2U;
  const auto pairs_per_expert = pair_mtiles * ntiles;
  const auto total_pairs =
      static_cast<std::uint32_t>(a.dispatch.sizes.owned_expert_count) *
      pairs_per_expert;
  if (pair_task >= total_pairs)
    return kBackwardV3DoneTask;
  const auto expert = pair_task / pairs_per_expert;
  auto local_pair = pair_task % pairs_per_expert;
  const auto ntile = local_pair / pair_mtiles;
  const auto mtile =
      (local_pair % pair_mtiles) * 2U + cute::block_rank_in_cluster();
  const auto local_task = ntile * mtiles + mtile;
  constexpr auto task_kind =
      Kind == BackwardV3WeightKind::Dw1 ? kBackwardV3Dw1TaskKind
                                        : kBackwardV3Dw2TaskKind;
  return task_kind |
         (expert * mtiles * ntiles + local_task);
}

template <BackwardV3WeightKind Kind>
__device__ __forceinline__ unsigned long long
backward_v3_static_weight_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t task, std::uint32_t total_tasks) {
  if (backward_v3_generation_cancelled(a, control) || task >= total_tasks)
    return kBackwardV3DoneTask;
  const auto experts =
      static_cast<std::uint32_t>(a.dispatch.sizes.owned_expert_count);
  const auto per_expert = total_tasks / experts;
  // Keep each grid wave on as few (expert, N-tile) B operands as possible.
  // The old expert-interleaved order balanced unequal reductions but reduced
  // a 128x256 B tile's concurrent L2 reuse to roughly gridDim/experts CTAs.
  // Expert-contiguous waves let dozens of independent CTAs reuse that tile;
  // every CTA still owns one output tile and no cross-CTA synchronization is
  // introduced.  Expert row counts are padded to 128 and all experts expose
  // the same output task count, so the persistent strided loop retains a
  // naturally balanced number of tasks per CTA.
  const auto expert = task / per_expert;
  const auto local_task = task % per_expert;
  constexpr auto task_kind =
      Kind == BackwardV3WeightKind::Dw1 ? kBackwardV3Dw1TaskKind
                                        : kBackwardV3Dw2TaskKind;
  return task_kind | (expert * per_expert + local_task);
}

// One dynamic claim owns all M tiles for an (expert, N-tile) pair.  The
// activation phase no longer needs mblock_mapping when this role starts, so
// its first gridDim/2 entries form per-cluster mailboxes without increasing
// the persistent workspace.  A device release/acquire around the mailbox and
// a cluster barrier distribute the leader's single atomic claim to both CTAs.
template <BackwardV3WeightKind Kind>
__device__ __forceinline__ unsigned long long
backward_v3_claim_weight_macro(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    std::uint32_t ntiles) {
  auto *mailboxes = peer_at<unsigned long long>(
      a.local_base, a.v3_mblock_mapping_offset);
  const auto pair = static_cast<std::uint32_t>(blockIdx.x) / 2U;
  const auto macro_count =
      static_cast<unsigned long long>(
          a.dispatch.sizes.owned_expert_count) *
      ntiles;
  if (threadIdx.x == 0U && cute::block_rank_in_cluster() == 0U) {
    auto &cursor = Kind == BackwardV3WeightKind::Dw1
                       ? control.dw1_scan_cursor.value
                       : control.dw2_scan_cursor.value;
    const auto macro =
        backward_v3_generation_cancelled(a, control)
            ? kBackwardV3DoneTask
            : atomicAdd(&cursor, 1ULL);
    persistent_v3::store_release<::cuda::thread_scope_device>(
        mailboxes[pair],
        macro < macro_count ? macro : kBackwardV3DoneTask);
  }
  cute::cluster_sync();
  return persistent_v3::load_acquire<::cuda::thread_scope_device>(
      mailboxes[pair]);
}

template <BackwardV3WeightKind Kind>
__device__ __forceinline__ unsigned long long
backward_v3_weight_macro_tile(std::uint32_t macro,
                              std::uint32_t pair_mtile,
                              std::uint32_t experts,
                              std::uint32_t mtiles,
                              std::uint32_t ntiles) {
  // Interleave experts so the first wave does not inherit one expert's K on
  // every cluster.  Dynamic claims then naturally balance later waves.
  const auto expert = macro % experts;
  const auto ntile = macro / experts;
  const auto mtile =
      pair_mtile * 2U + cute::block_rank_in_cluster();
  constexpr auto task_kind =
      Kind == BackwardV3WeightKind::Dw1 ? kBackwardV3Dw1TaskKind
                                        : kBackwardV3Dw2TaskKind;
  return task_kind |
         (static_cast<unsigned long long>(expert) * mtiles * ntiles +
          ntile * mtiles + mtile);
}

template <BackwardV3WeightKind Kind, std::uint32_t TileN,
          bool KMajorRows = false, bool Cluster2 = false,
          bool IndependentTasks = false, bool DynamicMacros = false>
__device__ __forceinline__ void backward_v3_weight_compute_role(
    const BackwardKernelArgs &a, const CUtensorMap &operand_a_map,
    const CUtensorMap &operand_b_map, const CUtensorMap &output_store_map,
    unsigned char *storage) {
  static_assert(TileN == 128U || TileN == 256U);
  using Mma = typename BackwardV3WeightMma<Kind, TileN, KMajorRows>::Type;
  static_assert(Mma::kNumAccum == TileN / 2U);

  // DeepGEMM's matching 128x256 kernel reaches its compute roofline with
  // three stages. Keep the compile-isolated weight pipeline equally shallow;
  // the activation kernel retains its independently compiled four stages.
  constexpr std::uint32_t kStages = 3U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = TileN * 64U;
  constexpr std::uint32_t kTaskReadyBarrier = 0U;
  constexpr std::uint32_t kTaskDoneBarrier = 1U;
  constexpr std::uint32_t kMathOutputBarrier = 2U;

  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *expert_padded_prefix = peer_at<unsigned long long>(
      a.local_base, a.v3_expert_padded_prefix_offset);
  auto *expert_dv_complete = peer_at<unsigned long long>(
      a.local_base, a.v3_expert_dv_complete_offset);
  auto *expert_cursor = peer_at<unsigned long long>(
      a.local_base,
      Kind == BackwardV3WeightKind::Dw1
          ? a.v3_expert_dw1_cursor_offset
          : a.v3_expert_dw2_cursor_offset);
  const auto dv_ntiles = static_cast<std::uint32_t>(
      (a.intermediate + kBackwardV3DvN - 1ULL) / kBackwardV3DvN);
  const auto mtiles = static_cast<std::uint32_t>(
      Kind == BackwardV3WeightKind::Dw1
          ? (2ULL * a.intermediate + 127ULL) / 128ULL
          : (a.hidden + 127ULL) / 128ULL);
  const auto ntiles = static_cast<std::uint32_t>(
      Kind == BackwardV3WeightKind::Dw1
          ? (a.hidden + TileN - 1ULL) / TileN
          : (a.intermediate + TileN - 1ULL) / TileN);
  const auto per_expert = mtiles * ntiles;
  const auto total_tasks =
      static_cast<unsigned long long>(a.dispatch.sizes.owned_expert_count) *
      per_expert;
  if constexpr (Kind == BackwardV3WeightKind::Dw1)
    MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1Ready, Compute, 0U, total_tasks,
                            a.epoch);
  else
    MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2Ready, Compute, 0U, total_tasks,
                            a.epoch);

  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kStages * kAStageElements;
  // The production-compatible path aliases the epilogue tile with the A
  // pipeline and therefore needs a producer/consumer task-boundary barrier.
  // The cluster microbenchmark can instead spend the otherwise unused Hopper
  // shared-memory capacity on a distinct output tile; that lets the producer
  // prefetch the next GEMM while the math warpgroups store the previous one.
  auto *pipeline_end = smem_b + kStages * kBStageElements;
  auto *output_tile = IndependentTasks ? pipeline_end : smem_a;
  auto *barrier_storage =
      IndependentTasks ? output_tile + 128U * TileN : pipeline_end;
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(
      barrier_storage);
  auto *empty_barriers = full_barriers + kStages;
  __shared__ unsigned long long task_shared;
  static_assert(!Cluster2 || TileN == 256U);
  static_assert(!DynamicMacros || IndependentTasks);
  static_assert(!DynamicMacros || Cluster2);

  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(Cluster2 ? 16 : 8);
    }
    cutlass::arch::fence_barrier_init();
  }
  auto cluster_pair_task = static_cast<std::uint32_t>(blockIdx.x) / 2U;
  auto independent_task = static_cast<std::uint32_t>(blockIdx.x);
  if (threadIdx.x == 0U) {
    if constexpr (!IndependentTasks) {
      if constexpr (Cluster2)
        task_shared = backward_v3_claim_cluster_weight_task<Kind>(
            a, *control, cluster_pair_task, mtiles, ntiles);
      else
        task_shared = backward_v3_claim_weight_role_task<Kind>(
            a, *control, expert_dv_complete, expert_cursor, dv_ntiles,
            per_expert);
    }
  }
  if constexpr (Cluster2)
    cute::cluster_sync();
  else
    __syncthreads();

  // Broadcasting the role index from lane zero encourages NVCC to keep the
  // value uniform across each warp instead of inserting warpgroup dependency
  // bookkeeping around WGMMA descriptor construction.
  const auto warpgroup =
      __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup >= 2U) {
    // Leave two 128-thread math warpgroups at 224 registers without consuming
    // the final register-file allocation unit; an exact 64/224/224 split can
    // make setmaxnreg.inc wait indefinitely when all three groups arrive in an
    // unlucky scheduler order.
    cutlass::arch::warpgroup_reg_dealloc<48>();
    // Keep the phase of each physical stage independently so every access to
    // the stage storage and barrier below has a compile-time index.  A linear
    // `pipeline % kStages` leaves the shared-memory/barrier address dynamic;
    // on Hopper that adds enough integer/address work to starve WGMMA between
    // TMA arrivals.  Expert reductions are padded to 128 rows, hence there are
    // always an integral number of 64-row K tiles to distribute over stages.
    std::uint32_t stage_epochs[kStages] = {0U, 0U, 0U};
    const auto pair_mtiles = mtiles / 2U;
    std::uint32_t macro_task = 0U;
    std::uint32_t pair_mtile = pair_mtiles;
    if constexpr (IndependentTasks) {
      // The activation phase publishes every dV tile before entering this
      // deterministic weight phase.  The TMA issuer performs the acquire and
      // generic->async proxy fence once per role rather than once per output
      // tile; macro-task production will preserve this same boundary.
      if (threadIdx.x == 256U) {
        while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                   control->dv_tasks_complete.value) != control->dv_tasks)
          __nanosleep(64U);
        asm volatile("fence.proxy.async.global;" : : : "memory");
      }
    }
    while (true) {
      unsigned long long encoded_task = kBackwardV3DoneTask;
      if constexpr (IndependentTasks) {
        if constexpr (DynamicMacros) {
          if (pair_mtile == pair_mtiles) {
            if (threadIdx.x == 256U) {
              empty_barriers[0].wait((stage_epochs[0] & 1U) ^ 1U);
              empty_barriers[1].wait((stage_epochs[1] & 1U) ^ 1U);
              empty_barriers[2].wait((stage_epochs[2] & 1U) ^ 1U);
            }
            const auto claimed = backward_v3_claim_weight_macro<Kind>(
                a, *control, ntiles);
            if (claimed == kBackwardV3DoneTask) {
              encoded_task = kBackwardV3DoneTask;
            } else {
              macro_task = static_cast<std::uint32_t>(claimed);
              pair_mtile = 0U;
              encoded_task = backward_v3_weight_macro_tile<Kind>(
                  macro_task, pair_mtile,
                  static_cast<std::uint32_t>(
                      a.dispatch.sizes.owned_expert_count),
                  mtiles, ntiles);
            }
          } else {
            encoded_task = backward_v3_weight_macro_tile<Kind>(
                macro_task, pair_mtile,
                static_cast<std::uint32_t>(
                    a.dispatch.sizes.owned_expert_count),
                mtiles, ntiles);
          }
        } else {
          if (threadIdx.x != 256U)
            break;
          if constexpr (Cluster2)
            encoded_task = backward_v3_claim_cluster_weight_task<Kind>(
                a, *control, cluster_pair_task, mtiles, ntiles);
          else
            encoded_task = backward_v3_static_weight_task<Kind>(
                a, *control, independent_task,
                static_cast<std::uint32_t>(total_tasks));
        }
      } else {
        cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
        encoded_task = task_shared;
      }
      if (encoded_task == kBackwardV3DoneTask)
        break;
      if (encoded_task != kBackwardV3IdleTask && threadIdx.x == 256U) {
        auto task = static_cast<std::uint32_t>(
            encoded_task & kBackwardV3TaskPayloadMask);
        const auto expert = static_cast<std::uint32_t>(task / per_expert);
        task %= per_expert;
        // Both gradients are N-major: hold the larger B tile in L2 while
        // sweeping every M tile for the expert.
        const auto mtile = task % mtiles;
        const auto ntile = task / mtiles;
        if constexpr (!IndependentTasks) {
          const auto expected =
              persistent_v3::expert_mblocks(
                  a.dispatch.expert_row_count[expert]) *
              dv_ntiles;
          while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                     expert_dv_complete[expert]) !=
                 backward_v3_expert_ready_value(expected))
            __nanosleep(64U);
          asm volatile("fence.proxy.async.global;" : : : "memory");
        }
        const auto reduction = persistent_v3::padded_expert_rows(
            a.dispatch.expert_row_count[expert]);
        const auto reduction_start = expert_padded_prefix[expert];
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        auto load_stage = [&](auto stage_constant,
                              std::uint32_t iteration) {
          constexpr auto stage = decltype(stage_constant)::value;
          const auto phase = stage_epochs[stage] & 1U;
          empty_barriers[stage].wait(phase ^ 1U);
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kBStageElements);
          if constexpr (Kind == BackwardV3WeightKind::Dw1) {
            const auto a_k = static_cast<std::uint32_t>(
                reduction_start + iteration * 64ULL);
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &operand_a_map, &full_barriers[stage], stage_a, a_k,
                static_cast<std::uint32_t>(mtile * 128ULL));
            if constexpr (KMajorRows) {
              deep_gemm::tma::copy<64, TileN, 128,
                                   cutlass::bfloat16_t>(
                  &operand_b_map, &full_barriers[stage], stage_b, a_k,
                  static_cast<std::uint32_t>(ntile * TileN),
                  Cluster2 ? 2U : 1U);
            } else {
              const auto b_k = static_cast<std::uint32_t>(
                  a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
#pragma unroll
              for (std::uint32_t atom = 0; atom < TileN / 64U; ++atom)
                deep_gemm::tma::copy<64, 64, 128,
                                     cutlass::bfloat16_t>(
                    &operand_b_map, &full_barriers[stage],
                    stage_b + atom * 64U * 64U,
                    static_cast<std::uint32_t>(ntile * TileN + atom * 64ULL),
                    b_k, Cluster2 ? 2U : 1U);
            }
          } else {
            const auto b_k = static_cast<std::uint32_t>(
                reduction_start + iteration * 64ULL);
            if constexpr (KMajorRows) {
              deep_gemm::tma::copy<64, 128, 128,
                                   cutlass::bfloat16_t>(
                  &operand_a_map, &full_barriers[stage], stage_a, b_k,
                  static_cast<std::uint32_t>(mtile * 128ULL));
            } else {
              const auto a_k = static_cast<std::uint32_t>(
                  a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
#pragma unroll
              for (std::uint32_t atom = 0; atom < 2U; ++atom)
                deep_gemm::tma::copy<64, 64, 128,
                                     cutlass::bfloat16_t>(
                    &operand_a_map, &full_barriers[stage],
                    stage_a + atom * 64U * 64U,
                    static_cast<std::uint32_t>(mtile * 128ULL +
                                               atom * 64ULL),
                    a_k);
            }
#pragma unroll
            for (std::uint32_t atom = 0; atom < TileN / 128U; ++atom)
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &operand_b_map, &full_barriers[stage],
                  stage_b + atom * 128U * 64U, b_k,
                  static_cast<std::uint32_t>(ntile * TileN + atom * 128ULL),
                  Cluster2 ? 2U : 1U);
          }
          full_barriers[stage].arrive_and_expect_tx(
              (kAStageElements + kBStageElements) *
              sizeof(transport::BFloat16));
          ++stage_epochs[stage];
        };
        const auto full_stage_groups = iterations / kStages;
        for (std::uint32_t group = 0; group < full_stage_groups; ++group) {
          const auto iteration = group * kStages;
          load_stage(cute::Int<0>{}, iteration);
          load_stage(cute::Int<1>{}, iteration + 1U);
          load_stage(cute::Int<2>{}, iteration + 2U);
        }
        const auto remainder_start = full_stage_groups * kStages;
        switch (iterations - remainder_start) {
        case 2U:
          load_stage(cute::Int<0>{}, remainder_start);
          load_stage(cute::Int<1>{}, remainder_start + 1U);
          break;
        case 1U:
          load_stage(cute::Int<0>{}, remainder_start);
          break;
        default:
          break;
        }
      } else if (encoded_task == kBackwardV3IdleTask &&
                 threadIdx.x == 256U) {
        __nanosleep(64U);
      }
      if constexpr (IndependentTasks) {
        if constexpr (DynamicMacros)
          ++pair_mtile;
        else if constexpr (Cluster2)
          cluster_pair_task += static_cast<std::uint32_t>(gridDim.x) / 2U;
        else
          independent_task += static_cast<std::uint32_t>(gridDim.x);
      } else {
        cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
      }
    }
    if constexpr (IndependentTasks) {
      empty_barriers[0].wait((stage_epochs[0] & 1U) ^ 1U);
      empty_barriers[1].wait((stage_epochs[1] & 1U) ^ 1U);
      empty_barriers[2].wait((stage_epochs[2] & 1U) ^ 1U);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<TileN == 128U ? 192 : 224>();
    std::uint32_t stage_epochs[kStages] = {0U, 0U, 0U};
    std::uint32_t local_tasks_complete = 0U;
#if MEGAMOE_DEVICE_TIMELINE
    bool timeline_weight_started = false;
    std::uint32_t timeline_weight_tasks = 0U;
#endif
    const auto pair_mtiles = mtiles / 2U;
    std::uint32_t macro_task = 0U;
    std::uint32_t pair_mtile = pair_mtiles;
    if constexpr (IndependentTasks) {
      if (threadIdx.x == 0U) {
        const auto now = device_globaltimer_ns();
        if constexpr (Kind == BackwardV3WeightKind::Dw1)
          atomicMin(&status->dw1_first_start_clock, now);
        else
          atomicMin(&status->dw2_first_start_clock, now);
      }
    }
    while (true) {
      unsigned long long encoded_task = kBackwardV3DoneTask;
      if constexpr (IndependentTasks) {
        if constexpr (DynamicMacros) {
          if (pair_mtile == pair_mtiles) {
            if (threadIdx.x < TileN / 64U)
              cute::tma_store_wait<0>();
            const auto claimed = backward_v3_claim_weight_macro<Kind>(
                a, *control, ntiles);
            if (claimed == kBackwardV3DoneTask) {
              encoded_task = kBackwardV3DoneTask;
            } else {
              macro_task = static_cast<std::uint32_t>(claimed);
              pair_mtile = 0U;
              encoded_task = backward_v3_weight_macro_tile<Kind>(
                  macro_task, pair_mtile,
                  static_cast<std::uint32_t>(
                      a.dispatch.sizes.owned_expert_count),
                  mtiles, ntiles);
            }
          } else {
            encoded_task = backward_v3_weight_macro_tile<Kind>(
                macro_task, pair_mtile,
                static_cast<std::uint32_t>(
                    a.dispatch.sizes.owned_expert_count),
                mtiles, ntiles);
          }
        } else {
          if constexpr (Cluster2)
            encoded_task = backward_v3_claim_cluster_weight_task<Kind>(
                a, *control, cluster_pair_task, mtiles, ntiles);
          else
            encoded_task = backward_v3_static_weight_task<Kind>(
                a, *control, independent_task,
                static_cast<std::uint32_t>(total_tasks));
        }
      } else {
        cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
        encoded_task = task_shared;
      }
      if (encoded_task == kBackwardV3DoneTask)
        break;
      if (encoded_task != kBackwardV3IdleTask) {
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U && !timeline_weight_started) {
          if constexpr (Kind == BackwardV3WeightKind::Dw1)
            MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1Start, Compute, 0U,
                                    encoded_task, a.epoch);
          else
            MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2Start, Compute, 0U,
                                    encoded_task, a.epoch);
          timeline_weight_started = true;
        }
#endif
        auto task = static_cast<std::uint32_t>(
            encoded_task & kBackwardV3TaskPayloadMask);
        const auto expert = static_cast<std::uint32_t>(task / per_expert);
        task %= per_expert;
        const auto mtile = task % mtiles;
        const auto ntile = task / mtiles;
        const auto reduction = persistent_v3::padded_expert_rows(
            a.dispatch.expert_row_count[expert]);
        if constexpr (!IndependentTasks) {
          if (threadIdx.x == 0U) {
            const auto now = device_globaltimer_ns();
            if constexpr (Kind == BackwardV3WeightKind::Dw1)
              atomicMin(&status->dw1_first_start_clock, now);
            else
              atomicMin(&status->dw2_first_start_clock, now);
          }
        }

        float accum[Mma::kNumAccum];
#pragma unroll
        for (int index = 0; index < Mma::kNumAccum; ++index)
          accum[index] = 0.0F;
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        auto compute_stage = [&](auto stage_constant) {
          constexpr auto stage = decltype(stage_constant)::value;
          const auto phase = stage_epochs[stage] & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kBStageElements);
          full_barriers[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < Mma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0; k < 64U / Mma::K; ++k) {
            if constexpr (Kind == BackwardV3WeightKind::Dw1) {
              const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
                  cute::UMMA::Major::K, 128, 64, 128,
                  cutlass::bfloat16_t>(stage_a, warpgroup * 64U,
                                       k * Mma::K);
              if constexpr (KMajorRows) {
                const auto desc_b =
                    deep_gemm::mma::sm90::make_gmma_desc<
                        cute::UMMA::Major::K, TileN, 64, 128,
                        cutlass::bfloat16_t>(stage_b, 0U, k * Mma::K);
                Mma::wgmma(desc_a, desc_b, accum, 1);
              } else {
                const auto desc_b =
                    deep_gemm::mma::sm90::make_gmma_desc<
                        cute::UMMA::Major::MN, TileN, 64, 128,
                        cutlass::bfloat16_t>(stage_b, 0U, k * Mma::K);
                Mma::wgmma(desc_a, desc_b, accum, 1);
              }
            } else {
              const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
                  cute::UMMA::Major::K, TileN, 64, 128,
                  cutlass::bfloat16_t>(stage_b, 0U, k * Mma::K);
              if constexpr (KMajorRows) {
                const auto desc_a =
                    deep_gemm::mma::sm90::make_gmma_desc<
                        cute::UMMA::Major::K, 128, 64, 128,
                        cutlass::bfloat16_t>(stage_a, warpgroup * 64U,
                                             k * Mma::K);
                Mma::wgmma(desc_a, desc_b, accum, 1);
              } else {
                const auto desc_a =
                    deep_gemm::mma::sm90::make_gmma_desc<
                        cute::UMMA::Major::MN, 128, 64, 128,
                        cutlass::bfloat16_t>(stage_a, warpgroup * 64U,
                                             k * Mma::K);
                Mma::wgmma(desc_a, desc_b, accum, 1);
              }
            }
          }
          deep_gemm::ptx::warpgroup_commit_batch();
          if constexpr (TileN == 128U)
            backward_v3_wgmma_wait0_accum64(accum);
          else
            backward_v3_wgmma_wait0_accum128(accum);
          if constexpr (Cluster2) {
            const auto lane = threadIdx.x & 31U;
            if (lane < 2U)
              empty_barriers[stage].arrive(lane);
          } else if ((threadIdx.x & 31U) == 0U) {
            empty_barriers[stage].arrive();
          }
          ++stage_epochs[stage];
        };
        const auto full_stage_groups = iterations / kStages;
        for (std::uint32_t group = 0; group < full_stage_groups; ++group) {
          compute_stage(cute::Int<0>{});
          compute_stage(cute::Int<1>{});
          compute_stage(cute::Int<2>{});
        }
        const auto remainder_start = full_stage_groups * kStages;
        switch (iterations - remainder_start) {
        case 2U:
          compute_stage(cute::Int<0>{});
          compute_stage(cute::Int<1>{});
          break;
        case 1U:
          compute_stage(cute::Int<0>{});
          break;
        default:
          break;
        }

        // Wait for the previous tile's asynchronous stores only when this
        // output buffer is about to be reused.  This deliberately overlaps a
        // tile's global epilogue with the following tile's TMA/WGMMA work.
        if constexpr (IndependentTasks) {
          if (threadIdx.x < TileN / 64U)
            cute::tma_store_wait<0>();
          cutlass::arch::NamedBarrier::sync(256U, kMathOutputBarrier);
        }

        // Convert accumulator fragments directly with STSM into the 128-byte
        // swizzle consumed by TMA. This is the same epilogue organization as
        // the DeepGEMM reference and avoids the scalar global-store loop.
        const auto warp =
            __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
        const auto lane = threadIdx.x & 31U;
#pragma unroll
        for (std::uint32_t index = 0; index < Mma::kNumAccum / 4U; ++index) {
          constexpr std::uint32_t kBankGroupBytes = 16U;
          const auto atom = index / (64U / 8U);
          const auto in_atom = index % (64U / 8U);
          const auto row = lane;
          auto column = in_atom;
          column ^= row % (128U / kBankGroupBytes);
          auto *smem_pointer =
              reinterpret_cast<std::uint8_t *>(output_tile) +
              warp * 16U * 128U + atom * 128U * 128U + row * 128U +
              column * kBankGroupBytes;
          deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U], accum[index * 4U + 1U])),
              __float22bfloat162_rn(make_float2(accum[index * 4U + 2U],
                                                 accum[index * 4U + 3U])),
              smem_pointer);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(256U, kMathOutputBarrier);
        const auto output_rows = Kind == BackwardV3WeightKind::Dw1
                                     ? 2ULL * a.intermediate
                                     : a.hidden;
        if (threadIdx.x < TileN / 64U) {
          const auto atom = threadIdx.x;
          cute::SM90_TMA_STORE_2D::copy(
              &output_store_map, output_tile + atom * 128U * 64U,
              static_cast<std::uint32_t>(ntile * TileN + atom * 64ULL),
              static_cast<std::uint32_t>(expert * output_rows +
                                         mtile * 128ULL));
          cute::tma_store_arrive();
          if constexpr (!IndependentTasks)
            backward_v3_tma_store_wait_complete();
        }
        if constexpr (IndependentTasks)
          __syncwarp();
        else
          cutlass::arch::NamedBarrier::sync(256U, kMathOutputBarrier);
        if constexpr (!IndependentTasks) {
          if (threadIdx.x == 0U) {
            const auto now = device_globaltimer_ns();
            if constexpr (Kind == BackwardV3WeightKind::Dw1) {
              const auto completed =
                  persistent_v3::fetch_add_release<
                      ::cuda::thread_scope_device>(
                      control->dw1_tasks_complete.value, 1ULL) +
                  1ULL;
              atomicMax(&status->dw1_last_complete_clock, now);
              if (completed == total_tasks)
                status->dw1_complete_clock = now;
            } else {
              const auto completed =
                  persistent_v3::fetch_add_release<
                      ::cuda::thread_scope_device>(
                      control->dw2_tasks_complete.value, 1ULL) +
                  1ULL;
              atomicMax(&status->dw2_last_complete_clock, now);
              if (completed == total_tasks)
                status->dw2_complete_clock = now;
            }
          }
        }
        if constexpr (IndependentTasks)
          ++local_tasks_complete;
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U)
          ++timeline_weight_tasks;
#endif
      } else if (threadIdx.x == 0U) {
        __nanosleep(64U);
      }

      if constexpr (IndependentTasks) {
        if constexpr (DynamicMacros)
          ++pair_mtile;
        else if constexpr (Cluster2)
          cluster_pair_task += static_cast<std::uint32_t>(gridDim.x) / 2U;
        else
          independent_task += static_cast<std::uint32_t>(gridDim.x);
      } else {
        cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
        if (threadIdx.x == 0U) {
          if constexpr (Cluster2) {
            cluster_pair_task +=
                static_cast<std::uint32_t>(gridDim.x) / 2U;
            task_shared = backward_v3_claim_cluster_weight_task<Kind>(
                a, *control, cluster_pair_task, mtiles, ntiles);
          } else {
            task_shared = backward_v3_claim_weight_role_task<Kind>(
                a, *control, expert_dv_complete, expert_cursor, dv_ntiles,
                per_expert);
          }
        }
      }
    }
    if constexpr (IndependentTasks) {
      if (threadIdx.x < TileN / 64U)
        backward_v3_tma_store_wait_complete();
      // Every issuing lane owns a distinct 64-column TMA store group. Lane
      // zero must not publish the role's aggregate completion merely because
      // its own group has drained: the host may hash/reuse the output while
      // lanes 1..3 are still writing it. This is only the once-per-role tail
      // rendezvous; steady-state epilogues remain overlapped with the next
      // tile's TMA/WGMMA pipeline.
      cutlass::arch::NamedBarrier::sync(256U, kMathOutputBarrier);
      if (threadIdx.x == 0U) {
        const auto now = device_globaltimer_ns();
        if constexpr (Kind == BackwardV3WeightKind::Dw1) {
          const auto completed =
              persistent_v3::fetch_add_release<
                  ::cuda::thread_scope_device>(
                  control->dw1_tasks_complete.value,
                  static_cast<unsigned long long>(local_tasks_complete)) +
              local_tasks_complete;
          atomicMax(&status->dw1_last_complete_clock, now);
          if (completed == total_tasks)
            status->dw1_complete_clock = now;
        } else {
          const auto completed =
              persistent_v3::fetch_add_release<
                  ::cuda::thread_scope_device>(
                  control->dw2_tasks_complete.value,
                  static_cast<unsigned long long>(local_tasks_complete)) +
              local_tasks_complete;
          atomicMax(&status->dw2_last_complete_clock, now);
          if (completed == total_tasks)
            status->dw2_complete_clock = now;
        }
      }
    }
#if MEGAMOE_DEVICE_TIMELINE
    if (threadIdx.x == 0U && timeline_weight_started) {
      if constexpr (Kind == BackwardV3WeightKind::Dw1)
        MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1End, Compute, 0U,
                                timeline_weight_tasks, a.epoch);
      else
        MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2End, Compute, 0U,
                                timeline_weight_tasks, a.epoch);
    }
#endif
  }
  if constexpr (Cluster2)
    cute::cluster_sync();
  else
    __syncthreads();
}
#endif

template <std::uint32_t TileN, bool Cluster2 = false,
          bool IndependentTasks = false>
__global__ __launch_bounds__(384, 1) void backward_v3_dw1_compute_kernel(
    BackwardKernelArgs a,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dw1_store_map,
    std::uint32_t global_ordinal_base, std::uint32_t total_compute_ctas,
    bool terminal_owner) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  backward_v3_wait_role_start(a, *control);
  backward_v3_weight_compute_role<BackwardV3WeightKind::Dw1, TileN, false,
                                  Cluster2, IndependentTasks>(
      a, dab_transpose_map, x_mn_map, dw1_store_map, storage);
  backward_v3_finish_compute_role(
      a, *control, *status,
      global_ordinal_base + static_cast<std::uint32_t>(blockIdx.x),
      total_compute_ctas, terminal_owner);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dab_transpose_map;
  (void)x_mn_map;
  (void)dw1_store_map;
  (void)global_ordinal_base;
  (void)total_compute_ctas;
  (void)terminal_owner;
#endif
}

template <std::uint32_t TileN, bool Cluster2 = false,
          bool IndependentTasks = false>
__global__ __launch_bounds__(384, 1) void backward_v3_dw2_compute_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    std::uint32_t global_ordinal_base, std::uint32_t total_compute_ctas,
    bool terminal_owner) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  backward_v3_wait_role_start(a, *control);
  backward_v3_weight_compute_role<BackwardV3WeightKind::Dw2, TileN, false,
                                  Cluster2, IndependentTasks>(
      a, dy_mn_map, u_transpose_map, dw2_store_map, storage);
  backward_v3_finish_compute_role(
      a, *control, *status,
      global_ordinal_base + static_cast<std::uint32_t>(blockIdx.x),
      total_compute_ctas, terminal_owner);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dy_mn_map;
  (void)u_transpose_map;
  (void)dw2_store_map;
  (void)global_ordinal_base;
  (void)total_compute_ctas;
  (void)terminal_owner;
#endif
}

// Cluster-paired production compute path.  All compute CTAs first drain dV
// and dX.  Only a small paired cohort remains on the communication-dependent
// Return/Reduce tail; every other resident CTA immediately turns otherwise
// idle SM time into dW work.  This overlaps compute with communication, while
// avoiding pointless dV/dW Tensor Core contention.  The same resident CTAs
// execute dW1 and dW2 schedules in pairs, so B multicast, independent
// TMA/WGMMA progress, and epilogue overlap need neither another global kernel
// nor a grid-wide rendezvous.
template <bool ActivationCluster2 = false>
__global__ __launch_bounds__(384, 1)
void backward_v3_cluster_compute_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    const __grid_constant__ CUtensorMap dab_dx_map,
    const __grid_constant__ CUtensorMap w1_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dw1_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    std::uint32_t reduce_ctas, std::uint32_t total_compute_ctas) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  backward_v3_wait_role_start(a, *control);
  const auto ordinal = static_cast<std::uint32_t>(blockIdx.x);
  const auto compute_ctas = static_cast<std::uint32_t>(gridDim.x);
  backward_v3_dv_dw2_compute_role_inline<true, ActivationCluster2>(
      a, dy_map, w2_transpose_map, dab_dx_map, w1_transpose_map,
      x_mn_map, dab_transpose_map, dy_mn_map, u_transpose_map,
      dw1_store_map, dw2_store_map, storage, ordinal, compute_ctas,
      reduce_ctas);
  backward_v3_weight_compute_role<BackwardV3WeightKind::Dw1, 256U, false,
                                  true, true, true>(
      a, dab_transpose_map, x_mn_map, dw1_store_map, storage);
  backward_v3_weight_compute_role<BackwardV3WeightKind::Dw2, 256U, false,
                                  true, true, true>(
      a, dy_mn_map, u_transpose_map, dw2_store_map, storage);
  backward_v3_finish_compute_role(a, *control, *status, ordinal,
                                  total_compute_ctas);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dy_map;
  (void)w2_transpose_map;
  (void)dab_dx_map;
  (void)w1_transpose_map;
  (void)x_mn_map;
  (void)dab_transpose_map;
  (void)dy_mn_map;
  (void)u_transpose_map;
  (void)dw1_store_map;
  (void)dw2_store_map;
  (void)reduce_ctas;
  (void)total_compute_ctas;
#endif
}

#if MEGAMOE_BUILD_TESTING
// NCU-friendly entry point for the exact production dW1 tile loop. The test
// harness initializes every expert as ready, so no communicator, transport
// grid, or terminal coordinator participates in the measurement.
template <std::uint32_t TileN, bool KMajorRows, bool Cluster2,
          bool IndependentTasks>
__global__ __launch_bounds__(384, 1) void backward_v3_dw1_profile_kernel(
    BackwardKernelArgs a,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dw1_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  backward_v3_weight_compute_role<BackwardV3WeightKind::Dw1, TileN,
                                  KMajorRows, Cluster2, IndependentTasks>(
      a, dab_transpose_map, x_mn_map, dw1_store_map, storage);
#else
  (void)a;
  (void)dab_transpose_map;
  (void)x_mn_map;
  (void)dw1_store_map;
#endif
}
#endif
