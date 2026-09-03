#pragma once

namespace backward_compute_detail {

constexpr std::uint32_t kStages = 4U;
constexpr std::uint32_t kMathWarpGroups = 2U;
constexpr std::uint32_t kMathThreads = kMathWarpGroups * 128U;
constexpr std::uint32_t kStageK = backward_tuning::kTileK;
constexpr std::uint32_t kStageM = backward_tuning::kTileM;

template <std::uint32_t TileN>
constexpr std::size_t dynamic_shared_bytes() {
  return static_cast<std::size_t>(kStages) *
             (kStageM * kStageK + TileN * kStageK) *
             sizeof(transport::BFloat16) +
         2U * kStages *
             sizeof(cutlass::arch::ClusterTransactionBarrier);
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
__device__ __forceinline__ std::uint32_t
find_padded_expert(const BackwardKernelArgs &a, std::uint64_t row) {
  const auto *offsets = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  for (std::uint32_t expert = 0U;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    if (row >= offsets[expert] && row < offsets[expert + 1U])
      return expert;
  }
  return a.dispatch.sizes.owned_expert_count;
}

template <std::uint32_t TileN>
__device__ __forceinline__ void store_m128_accumulator(
    float *accum, transport::BFloat16 *output, std::uint64_t output_row,
    std::uint64_t output_column, std::uint64_t output_width) {
  const auto store = [&](std::uint32_t warpgroup, std::uint32_t m,
                         std::uint32_t column, float value0, float value1) {
    const auto row = output_row + warpgroup * 64ULL + m;
    const auto column0 = output_column + column;
    reinterpret_cast<std::uint32_t *>(
        output + row * output_width + column0)[0] =
        float2_to_bf16x2_bits(value0, value1);
  };
  if constexpr (TileN == 128U)
    visit_forward_accumulator(accum, store);
  else
    visit_forward_accumulator_wide(accum, store);
}

template <std::uint32_t TileN>
__device__ __forceinline__ void initialize_k_mn_pipeline(
    unsigned char *storage) {
  constexpr std::uint32_t kAStageElements = kStageM * kStageK;
  constexpr std::uint32_t kBStageElements = TileN * kStageK;
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kStages * kAStageElements;
  auto *full = reinterpret_cast<ForwardBarrier *>(
      smem_b + kStages * kBStageElements);
  auto *empty = full + kStages;
  if (threadIdx.x == kMathThreads) {
#pragma unroll
    for (std::uint32_t stage = 0U; stage < kStages; ++stage) {
      full[stage].init(1);
      empty[stage].init(kMathThreads / 32U);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup < kMathWarpGroups)
    cutlass::arch::warpgroup_reg_alloc<backward_tuning::kMathRegisters>();
  else
    cutlass::arch::warpgroup_reg_dealloc<
        backward_tuning::kProducerRegisters>();
}

template <std::uint32_t TileN, std::uint32_t StorageTileN = TileN>
__device__ __forceinline__ void run_k_mn_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_mn_map,
    std::uint32_t activation_row, std::uint32_t weight_reduction_start,
    std::uint32_t reduction, transport::BFloat16 *output,
    std::uint64_t output_width, std::uint32_t output_column,
    unsigned char *storage, std::uint32_t &pipeline) {
  static_assert(TileN == 128U || TileN == 256U);
  using TileMma = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN, cute::UMMA::Major::K, cute::UMMA::Major::MN>::type;
  constexpr std::uint32_t kAStageElements = kStageM * kStageK;
  static_assert(StorageTileN >= TileN);
  constexpr std::uint32_t kBStageElements = StorageTileN * kStageK;
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kStages * kAStageElements;
  auto *full = reinterpret_cast<ForwardBarrier *>(
      smem_b + kStages * kBStageElements);
  auto *empty = full + kStages;

  const auto warpgroup = threadIdx.x / 128U;

  // The scheduler/dispatch acquire is in the generic proxy. Make its dR/dAB
  // payload writes visible before the producer issues the first TMA load.
  if (threadIdx.x == kMathThreads)
    asm volatile("fence.proxy.async.global;" : : : "memory");

  float accum[TileMma::kNumAccum];
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < TileMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  }
  const auto iterations = reduction / kStageK;
  if (threadIdx.x == kMathThreads) {
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto current = pipeline + iteration;
      const auto stage = current % kStages;
      const auto phase = (current / kStages) & 1U;
      empty[stage].wait(phase ^ 1U);
      auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a + stage * kAStageElements);
      auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b + stage * kBStageElements);
      const auto k = iteration * kStageK;
      deep_gemm::tma::copy<kStageK, kStageM, 128, cutlass::bfloat16_t>(
          &activation_map, &full[stage], stage_a, k, activation_row);
      deep_gemm::tma::copy<TileN, kStageK, 128, cutlass::bfloat16_t>(
          &weight_mn_map, &full[stage], stage_b, output_column,
          weight_reduction_start + k);
      full[stage].arrive_and_expect_tx(
          (kAStageElements + TileN * kStageK) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto current = pipeline + iteration;
      const auto stage = current % kStages;
      const auto phase = (current / kStages) & 1U;
      auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a + stage * kAStageElements);
      auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b + stage * kBStageElements);
      full[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0U; k < kStageK / TileMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::K, kStageM, kStageK, 128,
            cutlass::bfloat16_t>(stage_a, warpgroup * 64U,
                                 k * TileMma::K);
        const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, TileN, kStageK, 128,
            cutlass::bfloat16_t>(stage_b, 0U, k * TileMma::K);
        TileMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty[stage].arrive();
    }
  }
  __syncthreads();
  store_m128_accumulator<TileN>(accum, output, activation_row, output_column,
                                output_width);
  __syncthreads();
  pipeline += iterations;
}
#endif

} // namespace backward_compute_detail

template <std::uint32_t TileN>
__global__ __launch_bounds__(backward_tuning::kComputeThreads, 1) void
backward_du_wgmma_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dr_map,
    const __grid_constant__ CUtensorMap w2_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  backward_compute_detail::initialize_k_mn_pipeline<TileN>(storage);
  std::uint32_t pipeline = 0U;
  const auto ntiles = static_cast<std::uint32_t>(a.intermediate / TileN);
  const auto task = static_cast<std::uint64_t>(blockIdx.x);
  const auto mblock = task / ntiles;
  const auto ntile = static_cast<std::uint32_t>(task % ntiles);
  const auto row = mblock * backward_tuning::kTileM;
  if (row >= a.padded_rows)
    return;
  const auto expert = backward_compute_detail::find_padded_expert(a, row);
  if (expert >= a.dispatch.sizes.owned_expert_count)
    return;
  auto *du = peer_at<transport::BFloat16>(a.local_base,
                                          a.recomputed_u_offset);
  backward_compute_detail::run_k_mn_tile<TileN>(
      dr_map, w2_map, static_cast<std::uint32_t>(row),
      static_cast<std::uint32_t>(expert * a.hidden),
      static_cast<std::uint32_t>(a.hidden), du, a.intermediate,
      ntile * TileN, storage, pipeline);
#else
  (void)a;
  (void)dr_map;
  (void)w2_map;
#endif
}

template <std::uint32_t TileN>
__global__ __launch_bounds__(backward_tuning::kComputeThreads, 1) void
backward_dx_wgmma_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dab_map,
    const __grid_constant__ CUtensorMap w1_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  backward_compute_detail::initialize_k_mn_pipeline<TileN>(storage);
  std::uint32_t pipeline = 0U;
  const auto ntiles = static_cast<std::uint32_t>(a.hidden / TileN);
  const auto task = static_cast<std::uint64_t>(blockIdx.x);
  const auto mblock = task / ntiles;
  const auto ntile = static_cast<std::uint32_t>(task % ntiles);
  const auto row = mblock * backward_tuning::kTileM;
  if (row >= a.padded_rows)
    return;
  const auto expert = backward_compute_detail::find_padded_expert(a, row);
  if (expert >= a.dispatch.sizes.owned_expert_count)
    return;
  auto *dx = peer_at<transport::BFloat16>(a.local_base, a.dx_rows_offset);
  backward_compute_detail::run_k_mn_tile<TileN>(
      dab_map, w1_map, static_cast<std::uint32_t>(row),
      static_cast<std::uint32_t>(expert * 2ULL * a.intermediate),
      static_cast<std::uint32_t>(2ULL * a.intermediate), dx, a.hidden,
      ntile * TileN, storage, pipeline);
#else
  (void)a;
  (void)dab_map;
  (void)w1_map;
#endif
}
