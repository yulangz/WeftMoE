
  cutlass::arch::warpgroup_reg_alloc<forward_tuning::kMathRegisters>();
  std::uint32_t fc1_pipeline = 0U;
  std::uint32_t fc2_pipeline = 0U;
  unsigned long long pair_round = 0ULL;
  unsigned int fc2_streak = 0U;
#if MEGAMOE_DEVICE_TIMELINE
  bool timeline_fc1_started = false;
  bool timeline_fc2_started = false;
#endif
  select_compute_task<Cluster2, StaticLocal, DoFc1, DoFc2, Partition>(
      a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks, fc1_ntiles,
      fc2_ntiles, completion, task_shared, task_kind_shared,
      pair_operand_shared, pair_round, fc2_streak);
  cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
  while (true) {
    if constexpr (DoFc1) {
    while (true) {
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
      const auto n = n_tile * forward_tuning::kFc1TileN;
      using Fc1TileMma =
          typename std::conditional<M64, TunedForwardFc1M64Mma,
                                    TunedForwardFc1Mma>::type;
      float accum_a[Fc1TileMma::kNumAccum];
      float accum_b[Fc1TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc1TileMma::kNumAccum; ++index) {
        accum_a[index] = 0.0F;
        accum_b[index] = 0.0F;
      }
      const auto iterations =
          static_cast<std::uint32_t>(
              (a.hidden + forward_tuning::kTileK - 1ULL) /
              forward_tuning::kTileK);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto current = fc1_pipeline + iteration;
        const auto stage = current % Fc1StageCount;
        const auto phase = (current / Fc1StageCount) & 1U;
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_a + stage * kAStageElements +
            (M64 ? 0U : warpgroup * 64U * forward_tuning::kTileK));
        auto *stage_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_b + stage * kFc1BStageElements +
            (M64 ? warpgroup * (forward_tuning::kFc1TileN / 2U) *
                           forward_tuning::kTileK
                 : 0U));
        auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
            fc1_b2 + stage * kFc1BStageElements +
            (M64 ? warpgroup * (forward_tuning::kFc1TileN / 2U) *
                           forward_tuning::kTileK
                 : 0U));
        // A and gate share the first completion barrier, while up has its own.
        // Start the gate WGMMA group as soon as 32 KiB is ready so its issue
        // overlaps the tail of the independent 16 KiB up-weight TMA.
        fc1_full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0;
             k < forward_tuning::kTileK / Fc1TileMma::K; ++k) {
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
        for (std::uint32_t k = 0;
             k < forward_tuning::kTileK / Fc1TileMma::K; ++k) {
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
        deep_gemm::ptx::warpgroup_wait<
            forward_tuning::kWgmmaWaitGroup>();
        if constexpr (Cluster2) {
          const auto lane = threadIdx.x & 31U;
          if (lane < 2U)
            fc1_empty[stage].arrive(lane);
        } else if ((threadIdx.x & 31U) == 0U) {
          fc1_empty[stage].arrive();
        }
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
              const auto gate = canonical_gates[row + block_m];
              const auto a_pair = float2_to_bf16x2_bits(av0, av1);
              const auto b_pair = float2_to_bf16x2_bits(bv0, bv1);
              const auto a_values = bf16x2_bits_to_float2(a_pair);
              const auto b_values = bf16x2_bits_to_float2(b_pair);
              if constexpr (RetainBackwardState) {
                reinterpret_cast<std::uint32_t *>(
                    ab_cache)[output_index / 2ULL] = a_pair;
                const auto b_index =
                    a.max_destination_rows * a.intermediate + output_index;
                reinterpret_cast<std::uint32_t *>(
                    ab_cache)[b_index / 2ULL] = b_pair;
              }
              reinterpret_cast<std::uint32_t *>(s)[output_index / 2ULL] =
                  float2_to_bf16x2_bits(
                      gated_swiglu_fc2_input(a_values.x, b_values.x, gate),
                      gated_swiglu_fc2_input(a_values.y, b_values.y, gate));
            } else if (block_m < valid_rows) {
              const float av[2] = {av0, av1};
              const float bv[2] = {bv0, bv1};
              const auto gate = canonical_gates[row + block_m];
#pragma unroll
              for (std::uint32_t value = 0; value < 2U; ++value) {
                const auto tail_column = output_column + value;
                if (tail_column < a.intermediate) {
                  const auto output_index =
                      (row + block_m) * a.intermediate + tail_column;
                  const auto a_bits = float_to_bf16_bits(av[value]);
                  const auto b_bits = float_to_bf16_bits(bv[value]);
                  if constexpr (RetainBackwardState) {
                    ab_cache[output_index] =
                        transport::BFloat16{a_bits};
                    ab_cache[a.max_destination_rows * a.intermediate +
                             output_index] = transport::BFloat16{b_bits};
                  }
                  s[output_index] = transport::BFloat16{
                      float_to_bf16_bits(gated_swiglu_fc2_input_from_bf16(
                          a_bits, b_bits, gate))};
                }
              }
            }
          };
      if constexpr (M64)
        visit_m64_accumulator_pair<Fc1TileMma>(accum_a, accum_b, store_fc1);
      else
        visit_m128_accumulator_pair<Fc1TileMma>(accum_a, accum_b, store_fc1);
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
                                      transport_state::TaskClass::Fc1);
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
          const auto queue_slot = atomicAdd(&scheduler[2], 1ULL);
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
      select_compute_task<Cluster2, StaticLocal, DoFc1, DoFc2, Partition>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
          pair_operand_shared, pair_round, fc2_streak);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
    }
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;
