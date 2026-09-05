
template <bool StaticLocal, bool DoFc1, bool DoFc2, bool M64 = false,
          ForwardTaskPartition Partition = ForwardTaskPartition::All,
          bool TmaFc2Store = false, bool TmaFc2StoreToStaging = false,
          std::uint32_t Fc1StageCount = kFc1Stages,
          std::uint32_t Fc2StageCount = kFc2Stages, bool Cluster2 = false,
          bool RetainBackwardState = false>
__device__ __forceinline__ void
persistent_role(const ForwardKernelArgs &a, const CUtensorMap &x_map,
                const CUtensorMap &s_map, const CUtensorMap &w1_map,
                const CUtensorMap &w2_map, unsigned char *storage,
                unsigned long long *task_shared,
                unsigned int *task_kind_shared,
                unsigned int *return_mblock_shared,
                unsigned int *pair_operand_shared = nullptr,
                const CUtensorMap *output_store_map = nullptr) {
  static_assert(!TmaFc2StoreToStaging || TmaFc2Store,
                "staging specialization requires the TMA FC2 epilogue");
  static_assert(!Cluster2 || !M64,
                "cluster multicast requires the production M128 tile");
  static_assert(Fc1StageCount >= 2U && Fc1StageCount <= 4U,
                "FC1 supports two to four TMA stages");
  static_assert(Fc2StageCount >= 2U && Fc2StageCount <= 4U,
                "FC2 supports two to four TMA stages");
  static_assert(kFc2Iterations % Fc2StageCount == 0U,
                "the fixed FC2 reduction must divide evenly into stages");
  constexpr auto kConfiguredFc2Waves =
      kFc2Iterations / Fc2StageCount;
  constexpr auto kConfiguredAStorageElements =
      (Fc1StageCount > Fc2StageCount ? Fc1StageCount : Fc2StageCount) *
      kAStageElements;
  constexpr auto kConfiguredBStorageElements =
      2U * Fc1StageCount * kFc1BStageElements >
              Fc2StageCount * kFc2BStageElements
          ? 2U * Fc1StageCount * kFc1BStageElements
          : Fc2StageCount * kFc2BStageElements;
  constexpr auto kConfiguredGemmStorageElements =
      kConfiguredAStorageElements + kConfiguredBStorageElements;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  transport::BFloat16 *ab_cache = nullptr;
  if constexpr (RetainBackwardState)
    ab_cache = peer_at<transport::BFloat16>(a.local_base,
                                            a.forward_ab_cache_offset);
  const auto *canonical_gates =
      peer_at<const float>(a.local_base, a.canonical_gates_offset);
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
      storage + kConfiguredGemmStorageElements *
                    sizeof(transport::BFloat16) +
      (3U * Fc1StageCount + 2U * Fc2StageCount) * sizeof(std::uint64_t));
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
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  const auto fc2_ntiles =
      static_cast<std::uint32_t>(
          (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
          forward_tuning::kFc2TileN);
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *fc2_ready_queue = fc1_ready_queue + fc1_tasks;

  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kConfiguredAStorageElements;
  auto *fc1_b2 = smem_b + Fc1StageCount * kFc1BStageElements;
  auto *fc1_full =
      reinterpret_cast<ForwardBarrier *>(smem_b +
                                         kConfiguredBStorageElements);
  auto *fc1_empty = fc1_full + Fc1StageCount;
  auto *fc2_full = fc1_empty + Fc1StageCount;
  auto *fc2_empty = fc2_full + Fc2StageCount;
  auto *fc1_b2_full = fc2_empty + Fc2StageCount;

  if (threadIdx.x == kMathThreads) {
    for (std::uint32_t stage = 0; stage < Fc1StageCount; ++stage) {
      fc1_full[stage].init(1);
      fc1_empty[stage].init(Cluster2 ? 2U * kMathThreads / 32U
                                    : kMathThreads / 32U);
      fc1_b2_full[stage].init(1);
    }
    for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
      fc2_full[stage].init(1);
      fc2_empty[stage].init(Cluster2 ? 2U * kMathThreads / 32U
                                    : kMathThreads / 32U);
    }
    cutlass::arch::fence_barrier_init();
  }
  if constexpr (Cluster2)
    cute::cluster_sync();
  else
    __syncthreads();

  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup >= kMathWarpGroups) {
    // Only the first producer-warpgroup lane issues TMA; the remaining lanes carry
    // the CTA task/barrier protocol with a deliberately small register set.
    cutlass::arch::warpgroup_reg_dealloc<
        forward_tuning::kProducerRegisters>();
    std::uint32_t fc1_pipeline = 0U;
    std::uint32_t fc2_pipeline = 0U;
    // The math scheduler publishes the first task through this one-time
    // barrier.  At every later boundary kTaskDoneBarrier also publishes the
    // next task, avoiding a second CTA-wide rendezvous per GEMM tile.
    cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
    while (true) {
      if constexpr (DoFc1) {
      while (true) {
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
              static_cast<std::uint32_t>(
                  (a.hidden + forward_tuning::kTileK - 1ULL) /
                  forward_tuning::kTileK);
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
            const auto k_base = iteration * forward_tuning::kTileK;
            const auto n = n_tile * forward_tuning::kFc1TileN;
            const auto multicast_a =
                Cluster2 && *pair_operand_shared == kComputePairShareA ? 2U
                                                                       : 1U;
            const auto multicast_b =
                Cluster2 && *pair_operand_shared == kComputePairShareB ? 2U
                                                                       : 1U;
            if constexpr (M64) {
              deep_gemm::tma::copy<forward_tuning::kTileK, 64, 128,
                                   cutlass::bfloat16_t>(
                  &x_map, &fc1_full[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row), multicast_a);
            } else {
              deep_gemm::tma::copy<forward_tuning::kTileK,
                                   forward_tuning::kTileM, 128,
                                   cutlass::bfloat16_t>(
                  &x_map, &fc1_full[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row), multicast_a);
            }
            deep_gemm::tma::copy<forward_tuning::kTileK,
                                 forward_tuning::kFc1TileN, 128,
                                 cutlass::bfloat16_t>(
                &w1_map, &fc1_full[stage], stage_b1, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
                multicast_b);
            deep_gemm::tma::copy<forward_tuning::kTileK,
                                 forward_tuning::kFc1TileN, 128,
                                 cutlass::bfloat16_t>(
                &w1_map, &fc1_b2_full[stage], stage_b2, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                           a.intermediate + n),
                multicast_b);
            constexpr auto a_stage_elements =
                M64 ? 64U * forward_tuning::kTileK : kAStageElements;
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
              const auto k_base =
                  (wave * Fc2StageCount + stage) * forward_tuning::kTileK;
              const auto n = n_tile * forward_tuning::kFc2TileN;
              const auto multicast_a =
                  Cluster2 && *pair_operand_shared == kComputePairShareA ? 2U
                                                                         : 1U;
              if constexpr (M64) {
                deep_gemm::tma::copy<forward_tuning::kTileK, 64, 128,
                                     cutlass::bfloat16_t>(
                    &s_map, &fc2_full[stage], stage_a, k_base,
                    static_cast<std::uint32_t>(row), multicast_a);
              } else {
                deep_gemm::tma::copy<forward_tuning::kTileK,
                                     forward_tuning::kTileM, 128,
                                     cutlass::bfloat16_t>(
                    &s_map, &fc2_full[stage], stage_a, k_base,
                    static_cast<std::uint32_t>(row), multicast_a);
              }
              deep_gemm::tma::copy<forward_tuning::kTileK,
                                   forward_tuning::kFc2TileN, 128,
                                   cutlass::bfloat16_t>(
                  &w2_map, &fc2_full[stage], stage_b, k_base,
                  static_cast<std::uint32_t>(expert * a.hidden + n));
              constexpr auto a_stage_elements =
                  M64 ? 64U * forward_tuning::kTileK : kAStageElements;
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
    if constexpr (Cluster2)
      cute::cluster_sync();
    return;
  }
