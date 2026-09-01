#pragma once

// Included inside activation_transport.cu's private CUDA namespace. This is
// the production-shaped task body: no grid synchronization and no global task
// ownership. The V3 compute scheduler owns readiness/claiming around it.
__device__ void backward_wgmma_dw_mn_k_tile_128(
    const CUtensorMap &a_mn_map, const CUtensorMap &b_k_map,
    std::uint32_t a_reduction_start, std::uint32_t b_reduction_start,
    std::uint32_t output_row, std::uint32_t output_column,
    std::uint32_t reduction, transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration) {
  using Mma = typename deep_gemm::mma::sm90::BF16MMASelector<
      128, cute::UMMA::Major::MN, cute::UMMA::Major::K>::type;
  static_assert(Mma::kNumAccum == 64);
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kMathWarpGroups = 2U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 128U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < Mma::kNumAccum; ++index)
      accum[index] = 0.0F;
  }
  const auto iterations = reduction / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      const auto a_k = a_reduction_start + iteration * 64U;
      const auto b_k = b_reduction_start + iteration * 64U;
      for (std::uint32_t atom = 0; atom < 2U; ++atom)
        deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
            &a_mn_map, &full_barriers[stage],
            smem_a + atom * 64U * 64U, output_row + atom * 64U, a_k);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &b_k_map, &full_barriers[stage], smem_b, b_k, output_column);
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < Mma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / Mma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
            smem_a, warpgroup * 64U, k * Mma::K);
        const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::K, 128, 64, 128, cutlass::bfloat16_t>(
            smem_b, 0U, k * Mma::K);
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
  }
  __syncthreads();
  pipeline_iteration += iterations;
}
