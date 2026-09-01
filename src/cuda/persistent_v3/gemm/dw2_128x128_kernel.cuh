#pragma once

// This file is deliberately included into activation_transport.cu after the
// private Backward launch contract and WGMMA helpers are visible. Keeping the
// Hopper implementation in the aggregation TU preserves descriptor inlining
// while the source remains isolated from the legacy megakernel.

template <std::uint32_t Stages>
__global__ __launch_bounds__(384, 1)
void backward_dw2_128x128_diagnostic_kernel(
    BackwardKernelArgs a,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  static_assert(Stages == 4U || Stages == 5U);
  using Mma = typename deep_gemm::mma::sm90::BF16MMASelector<
      128, cute::UMMA::Major::MN, cute::UMMA::Major::K>::type;
  static_assert(Mma::kNumAccum == 64,
                "128-column dW must use 64 FP32 accumulators per math thread");

  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *claims = peer_at<unsigned long long>(a.local_base, a.claim_offset);
  auto *dw2 = peer_at<transport::BFloat16>(a.local_base, a.dw2_offset);
#if MEGAMOE_DEBUG_VALIDATE
  auto *dw_timeline = peer_at<BackwardExpertDwTimeline>(
      a.local_base, a.dw_timeline_offset);
#endif

  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 128U * 64U;
  constexpr std::uint64_t kDwN = 128ULL;
  constexpr std::uint32_t kTaskReadyBarrier = 0U;
  constexpr std::uint32_t kTaskDoneBarrier = 1U;
  constexpr std::uint32_t kMathOutputReadyBarrier = 2U;
  constexpr std::uint32_t kMathOutputDoneBarrier = 3U;

  const auto dw2_mtiles = (a.hidden + 127ULL) / 128ULL;
  const auto dw2_ntiles = (a.intermediate + kDwN - 1ULL) / kDwN;
  const auto dw2_tasks = static_cast<unsigned long long>(
      a.dispatch.sizes.owned_expert_count) * dw2_mtiles * dw2_ntiles;
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + Stages * kAStageElements;
  auto *output_tile = smem_a;
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(
      smem_b + Stages * kBStageElements);
  auto *empty_barriers = full_barriers + Stages;

  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    status->dw2_first_start_clock = ~0ULL;
    status->dw2_last_complete_clock = 0ULL;
    status->dw2_complete_clock = 0ULL;
  }
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < Stages; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (threadIdx.x == 0U)
    task_shared = blockIdx.x;
  __syncthreads();
  // Diagnostic-only launch boundary. The production V3 role loop invokes the
  // task body below without either of these cooperative grid barriers.
  grid.sync();
  backward_publish_progress(a, kBackwardDw2Started, dw2_tasks);

  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup >= 2U) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    std::uint32_t pipeline = 0U;
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      auto task = task_shared;
      if (task >= dw2_tasks)
        break;
      const auto per_expert = dw2_mtiles * dw2_ntiles;
      const auto expert = static_cast<std::uint32_t>(task / per_expert);
      task %= per_expert;
      const auto ntile = task / dw2_mtiles;
      const auto mtile = task % dw2_mtiles;
      const auto reduction_start = claims[expert];
      const auto reduction = persistent_v3::padded_expert_rows(
          a.dispatch.expert_row_count[expert]);
      if (threadIdx.x == 256U) {
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        for (std::uint32_t iteration = 0; iteration < iterations;
             ++iteration) {
          const auto current = pipeline + iteration;
          const auto stage = current % Stages;
          const auto phase = (current / Stages) & 1U;
          empty_barriers[stage].wait(phase ^ 1U);
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kBStageElements);
          const auto a_k = static_cast<std::uint32_t>(
              a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
          const auto b_k = static_cast<std::uint32_t>(
              reduction_start + iteration * 64ULL);
          for (std::uint32_t atom = 0; atom < 2U; ++atom)
            deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                &dy_mn_map, &full_barriers[stage],
                stage_a + atom * 64U * 64U,
                static_cast<std::uint32_t>(mtile * 128ULL + atom * 64ULL),
                a_k);
          deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
              &u_transpose_map, &full_barriers[stage], stage_b, b_k,
              static_cast<std::uint32_t>(ntile * kDwN));
          full_barriers[stage].arrive_and_expect_tx(
              (kAStageElements + kBStageElements) *
              sizeof(transport::BFloat16));
        }
        pipeline += iterations;
      }
      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<192>();
    std::uint32_t pipeline = 0U;
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      auto task = task_shared;
      if (task >= dw2_tasks)
        break;
      const auto per_expert = dw2_mtiles * dw2_ntiles;
      const auto expert = static_cast<std::uint32_t>(task / per_expert);
      task %= per_expert;
      const auto ntile = task / dw2_mtiles;
      const auto mtile = task % dw2_mtiles;
      const auto reduction = persistent_v3::padded_expert_rows(
          a.dispatch.expert_row_count[expert]);
      if (threadIdx.x == 0U)
        atomicMin(&status->dw2_first_start_clock, device_globaltimer_ns());

      float accum[Mma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Mma::kNumAccum; ++index)
        accum[index] = 0.0F;
      const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto current = pipeline + iteration;
        const auto stage = current % Stages;
        const auto phase = (current / Stages) & 1U;
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
          const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
              cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
              stage_a, warpgroup * 64U, k * Mma::K);
          const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
              cute::UMMA::Major::K, 128, 64, 128, cutlass::bfloat16_t>(
              stage_b, 0U, k * Mma::K);
          Mma::wgmma(desc_a, desc_b, accum, 1);
        }
        deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Mma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
        deep_gemm::ptx::warpgroup_wait<0>();
        if ((threadIdx.x & 31U) == 0U)
          empty_barriers[stage].arrive();
      }
      pipeline += iterations;

      visit_forward_accumulator(
          accum, [&](std::uint32_t wg, std::uint32_t m,
                     std::uint32_t column, float v0, float v1) {
            const auto local_row = wg * 64U + m;
            output_tile[local_row * kDwN + column] =
                transport::BFloat16{float_to_bf16_bits(v0)};
            output_tile[local_row * kDwN + column + 1U] =
                transport::BFloat16{float_to_bf16_bits(v1)};
          });
      cutlass::arch::NamedBarrier::sync(256U, kMathOutputReadyBarrier);
      constexpr std::uint32_t kVectorsPerRow = kDwN / 8U;
      constexpr std::uint32_t kOutputVectors = 128U * kVectorsPerRow;
      const auto output_rows = a.hidden;
      const auto output_columns = a.intermediate;
      const auto expert_stride = output_rows * output_columns;
      for (std::uint32_t vector = threadIdx.x; vector < kOutputVectors;
           vector += 256U) {
        const auto local_row = vector / kVectorsPerRow;
        const auto local_vector = vector % kVectorsPerRow;
        const auto mma_row = mtile * 128ULL + local_row;
        const auto mma_column = ntile * kDwN + local_vector * 8ULL;
        if (mma_row < output_rows && mma_column + 7ULL < output_columns) {
          reinterpret_cast<uint4 *>(
              dw2 + expert * expert_stride + mma_row * output_columns +
              mma_column)[0] = reinterpret_cast<const uint4 *>(
                  output_tile + local_row * kDwN)[local_vector];
        }
      }
      cutlass::arch::NamedBarrier::sync(256U, kMathOutputDoneBarrier);
#if MEGAMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0U) {
        const auto complete =
            atomicAdd(&dw_timeline[expert].dw2_tasks_complete, 1ULL) + 1ULL;
        if (complete == per_expert)
          dw_timeline[expert].dw2_complete_clock = device_globaltimer_ns();
      }
#endif
      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
      if (threadIdx.x == 0U)
        task_shared += gridDim.x;
    }
  }

  grid.sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    status->dw2_last_complete_clock = device_globaltimer_ns();
    status->dw2_complete_clock = status->dw2_last_complete_clock;
    auto *scheduler = peer_at<unsigned long long>(a.local_base,
                                                   a.scheduler_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        complete(scheduler[11]);
    complete.store(a.epoch, ::cuda::memory_order_release);
  }
  backward_publish_progress(a, kBackwardDw2Complete, dw2_tasks);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2End, Compute, 0U, dw2_tasks,
                          a.epoch);
#else
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    auto *status = peer_at<BackwardDeviceStatus>(a.local_base,
                                                 a.status_offset);
    publish_local_structured_error(status, a.rank, 90U, 1U, 1ULL, 0ULL);
  }
#endif
}
