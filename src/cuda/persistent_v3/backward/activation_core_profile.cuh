#pragma once

// NCU/correctness fixture for the exact production dV TMA/WGMMA mainloop.
// It intentionally excludes routing, GIN, task claims, nonlinear epilogues,
// and inter-kernel publication so failures can be attributed to the compute
// pipeline itself.  This header is included only by the testing build of the
// aggregation translation unit after the SM90 aliases/helpers are visible.

#if MEGAMOE_BUILD_TESTING

__global__ void backward_v3_fill_bf16_profile_kernel(
    transport::BFloat16 *values, std::uint64_t elements,
    std::uint16_t bits) {
  const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < elements; index += stride)
    values[index] = transport::BFloat16{bits};
}

__global__ void backward_v3_validate_bf16_profile_kernel(
    const transport::BFloat16 *values, std::uint64_t elements,
    std::uint16_t expected, unsigned long long *mismatches) {
  unsigned long long local = 0ULL;
  const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
       index < elements; index += stride)
    local += values[index].bits != expected;
  if (local != 0ULL)
    atomicAdd(mismatches, local);
}

__global__ __launch_bounds__(384, 1)
void backward_v3_dv_core_profile_kernel(
    const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    transport::BFloat16 *output, std::uint32_t tasks) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 128U * 64U;
  constexpr std::uint32_t kTaskReadyBarrier = 0U;
  constexpr std::uint32_t kTaskDoneBarrier = 1U;
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kStages * kAStageElements;
  auto *smem_b2 = smem_b + kStages * kBStageElements;
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(
      smem_b2 + kStages * kBStageElements);
  auto *empty_barriers = full_barriers + kStages;

  if (threadIdx.x == 256U) {
#pragma unroll
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();

  const auto warpgroup =
      __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0U);
  if (warpgroup >= 2U) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    std::uint32_t stage_epochs[kStages] = {};
    for (std::uint32_t task = static_cast<std::uint32_t>(blockIdx.x);
         task < tasks; task += static_cast<std::uint32_t>(gridDim.x)) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      if (threadIdx.x == 256U) {
        constexpr std::uint32_t kIterations = 7168U / 64U;
        for (std::uint32_t group = 0; group < kIterations / kStages;
             ++group) {
#pragma unroll
          for (std::uint32_t stage = 0; stage < kStages; ++stage) {
            const auto iteration = group * kStages + stage;
            const auto phase = stage_epochs[stage] & 1U;
            empty_barriers[stage].wait(phase ^ 1U);
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements);
            auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b + stage * kBStageElements);
            auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b2 + stage * kBStageElements);
            const auto k_base = iteration * 64U;
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &dy_map, &full_barriers[stage], stage_a, k_base, 0U);
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &w2_transpose_map, &full_barriers[stage], stage_b, k_base,
                0U);
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &w2_transpose_map, &full_barriers[stage], stage_b2, k_base,
                128U);
            full_barriers[stage].arrive_and_expect_tx(
                3U * kAStageElements * sizeof(transport::BFloat16));
            ++stage_epochs[stage];
          }
        }
#pragma unroll
        for (std::uint32_t stage = 0; stage < kStages; ++stage)
          empty_barriers[stage].wait((stage_epochs[stage] + 1U) & 1U);
      }
      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<224>();
    std::uint32_t stage_epochs[kStages] = {};
    for (std::uint32_t task = static_cast<std::uint32_t>(blockIdx.x);
         task < tasks; task += static_cast<std::uint32_t>(gridDim.x)) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      float accum_a[ForwardMma::kNumAccum];
      float accum_b[ForwardMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
        accum_a[index] = 0.0F;
        accum_b[index] = 0.0F;
      }
      constexpr std::uint32_t kIterations = 7168U / 64U;
      for (std::uint32_t group = 0; group < kIterations / kStages;
           ++group) {
#pragma unroll
        for (std::uint32_t stage = 0; stage < kStages; ++stage) {
          const auto phase = stage_epochs[stage] & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements +
              warpgroup * 64U * 64U);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kBStageElements);
          auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b2 + stage * kBStageElements);
          full_barriers[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
            deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
            deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
          }
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t atom = 0; atom < 4U; ++atom) {
            const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
                stage_a + atom * ForwardMma::K, 1);
            const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
                stage_b + atom * ForwardMma::K, 1);
            const auto desc_b2 = deep_gemm::mma::sm90::make_smem_desc(
                stage_b2 + atom * ForwardMma::K, 1);
            ForwardMma::wgmma(desc_a, desc_b, accum_a, 1);
            ForwardMma::wgmma(desc_a, desc_b2, accum_b, 1);
          }
          deep_gemm::ptx::warpgroup_commit_batch();
          backward_v3_wgmma_wait0_accum64x2(accum_a, accum_b);
          if ((threadIdx.x & 31U) == 0U)
            empty_barriers[stage].arrive();
          ++stage_epochs[stage];
        }
      }

      const auto task_base =
          static_cast<std::uint64_t>(task) * 128ULL * 256ULL;
#pragma unroll
      for (std::uint32_t half = 0; half < 2U; ++half) {
        auto *half_accum = half == 0U ? accum_a : accum_b;
        visit_forward_accumulator(
            half_accum,
            [&](std::uint32_t wg, std::uint32_t m,
                std::uint32_t column, float value0, float value1) {
              const auto row = wg * 64U + m;
              const auto output_column = half * 128U + column;
              output[task_base + row * 256ULL + output_column] =
                  transport::BFloat16{float_to_bf16_bits(value0)};
              output[task_base + row * 256ULL + output_column + 1U] =
                  transport::BFloat16{float_to_bf16_bits(value1)};
            });
      }
      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
    }
  }
  __syncthreads();
#else
  (void)dy_map;
  (void)w2_transpose_map;
  (void)output;
  (void)tasks;
#endif
}

#endif
