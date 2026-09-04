#pragma once

namespace backward_compute_detail {

constexpr std::uint32_t kActivationStages = backward_tuning::kDabStages;
constexpr std::uint32_t kDwStages = backward_tuning::kDw1Stages;
constexpr std::uint32_t kMathWarpGroups = 2U;
constexpr std::uint32_t kMathThreads = kMathWarpGroups * 128U;
constexpr std::uint32_t kStageK = backward_tuning::kTileK;
constexpr std::uint32_t kStageM = backward_tuning::kTileM;
constexpr std::uint32_t kStoreReadyBarrier = 0U;
constexpr std::uint32_t kStoreDoneBarrier = 1U;

template <std::uint32_t TileN>
constexpr std::uint32_t stage_storage_elements() {
  constexpr auto activation_elements = (kStageM + TileN) * kStageK;
  constexpr auto dw_elements =
      (backward_tuning::kDwTileM + backward_tuning::kDwTileN) * kStageK;
  return activation_elements > dw_elements ? activation_elements
                                           : dw_elements;
}

template <std::uint32_t TileN>
constexpr std::uint32_t shared_arena_elements() {
  constexpr auto activation_elements =
      kActivationStages * stage_storage_elements<TileN>();
  constexpr auto dw_elements =
      kDwStages * stage_storage_elements<TileN>() +
      backward_tuning::kDwTileM * backward_tuning::kDwTileN;
  return activation_elements > dw_elements ? activation_elements
                                           : dw_elements;
}

template <std::uint32_t TileN>
constexpr std::size_t dynamic_shared_bytes() {
  return static_cast<std::size_t>(shared_arena_elements<TileN>()) *
             sizeof(transport::BFloat16) +
         2U * (kActivationStages + kDwStages) *
             sizeof(cutlass::arch::ClusterTransactionBarrier);
}

static_assert(backward_tuning::kDabStages == backward_tuning::kDxStages);
static_assert(backward_tuning::kDw1Stages == backward_tuning::kDw2Stages);

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
  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *activation_full = reinterpret_cast<ForwardBarrier *>(
      smem + shared_arena_elements<TileN>());
  auto *activation_empty = activation_full + kActivationStages;
  auto *dw_full = activation_empty + kActivationStages;
  auto *dw_empty = dw_full + kDwStages;
  if (threadIdx.x == kMathThreads) {
#pragma unroll
    for (std::uint32_t stage = 0U; stage < kActivationStages; ++stage) {
      activation_full[stage].init(1);
      activation_empty[stage].init(kMathThreads / 32U);
    }
#pragma unroll
    for (std::uint32_t stage = 0U; stage < kDwStages; ++stage) {
      dw_full[stage].init(1);
      dw_empty[stage].init(kMathThreads / 32U);
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

template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t StorageTileN,
          std::uint32_t OutputOffsetElements>
__device__ __forceinline__ void store_tma_accumulator(
    float *accum, const CUtensorMap &output_store_map,
    std::uint32_t output_row, std::uint32_t output_column,
    unsigned char *storage, bool wait_previous, bool wait_current) {
  using TileMma = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN, cute::UMMA::Major::MN, cute::UMMA::Major::MN>::type;
  constexpr std::uint32_t kMWaves = TileM / kStageM;
  constexpr std::uint32_t kNAtoms = TileN / 64U;
  constexpr std::uint32_t kMHalves = TileM / 128U;
  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup >= kMathWarpGroups)
    return;

  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *output_tile = smem + OutputOffsetElements;
  if (wait_previous && threadIdx.x < kNAtoms * kMHalves)
    asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
  cutlass::arch::NamedBarrier::sync(kMathThreads, kStoreReadyBarrier);

  const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
  const auto lane = threadIdx.x & 31U;
#pragma unroll
  for (std::uint32_t wave = 0U; wave < kMWaves; ++wave) {
    auto *wave_accum = accum + wave * TileMma::kNumAccum;
#pragma unroll
    for (std::uint32_t index = 0U;
         index < TileMma::kNumAccum / 4U; ++index) {
      constexpr std::uint32_t kStoreAtomN = 64U;
      constexpr std::uint32_t kSwizzleBytes = 128U;
      constexpr std::uint32_t kBankGroupBytes = 16U;
      const auto atom = index / (kStoreAtomN / 8U);
      const auto in_atom = index % (kStoreAtomN / 8U);
      const auto swizzle_row = lane;
      auto swizzle_column = in_atom;
      swizzle_column ^= swizzle_row %
                        (kSwizzleBytes / kBankGroupBytes);
      auto *smem_pointer =
          reinterpret_cast<std::uint8_t *>(output_tile) +
          atom * TileM * kSwizzleBytes +
          wave * kStageM * kSwizzleBytes +
          warp * 16U * kSwizzleBytes +
          swizzle_row * kSwizzleBytes +
          swizzle_column * kBankGroupBytes;
      deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
          __float22bfloat162_rn(make_float2(
              wave_accum[index * 4U], wave_accum[index * 4U + 1U])),
          __float22bfloat162_rn(make_float2(
              wave_accum[index * 4U + 2U],
              wave_accum[index * 4U + 3U])),
          smem_pointer);
    }
  }
  cute::tma_store_fence();
  cutlass::arch::NamedBarrier::sync(kMathThreads, kStoreDoneBarrier);
  if (threadIdx.x < kNAtoms * kMHalves) {
    const auto atom = threadIdx.x % kNAtoms;
    const auto m_half = threadIdx.x / kNAtoms;
    cute::SM90_TMA_STORE_2D::copy(
        &output_store_map,
        output_tile + atom * TileM * 64U + m_half * 128U * 64U,
        output_column + atom * 64U, output_row + m_half * 128U);
    cute::tma_store_arrive();
    if (wait_current)
      asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
  }
  cutlass::arch::NamedBarrier::sync(kMathThreads, kStoreReadyBarrier);
}

template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t StorageTileN,
          cute::UMMA::Major MajorA, cute::UMMA::Major MajorB,
          bool TmaOutput, std::uint32_t StageCount,
          std::uint32_t BarrierOffset,
          std::uint32_t OutputOffsetElements>
__device__ __forceinline__ void run_wgmma_tile(
    const CUtensorMap &a_map, const CUtensorMap &b_map,
    const CUtensorMap &output_store_map,
    std::uint32_t a_m, std::uint32_t b_n, std::uint32_t a_k,
    std::uint32_t b_k, std::uint32_t reduction,
    transport::BFloat16 *output, std::uint64_t output_width,
    std::uint32_t output_row, std::uint32_t output_column,
    unsigned char *storage, std::uint32_t &pipeline) {
  static_assert(TileM == 128U || TileM == 256U);
  static_assert(TileN == 128U || TileN == 256U);
  constexpr std::uint32_t kMWaves = TileM / kStageM;
  using TileMma = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN, MajorA, MajorB>::type;
  constexpr std::uint32_t kAStageElements = TileM * kStageK;
  constexpr std::uint32_t kBStageElements = TileN * kStageK;
  constexpr auto kStageStorageElements =
      stage_storage_elements<StorageTileN>();
  static_assert(kAStageElements + kBStageElements <=
                kStageStorageElements);
  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *barriers = reinterpret_cast<ForwardBarrier *>(
      smem + shared_arena_elements<StorageTileN>());
  auto *full = barriers + BarrierOffset;
  auto *empty = full + StageCount;

  const auto warpgroup = threadIdx.x / 128U;

  // The scheduler/dispatch acquire is in the generic proxy. Make its dR/dAB
  // payload writes visible before the producer issues the first TMA load.
  if (threadIdx.x == kMathThreads)
    asm volatile("fence.proxy.async.global;" : : : "memory");

  float accum[TileMma::kNumAccum * kMWaves];
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
      accum[index] = 0.0F;
  }
  auto a_desc = deep_gemm::mma::sm90::make_gmma_desc<
      MajorA, TileM, kStageK, 128, cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(smem),
      warpgroup * TileMma::M, 0U);
  auto b_desc = deep_gemm::mma::sm90::make_gmma_desc<
      MajorB, TileN, kStageK, 128, cutlass::bfloat16_t>(
      reinterpret_cast<cutlass::bfloat16_t *>(smem) + kAStageElements,
      0U, 0U);
  const auto a_desc_lo = __shfl_sync(0xffffffffU, a_desc.reg32_[0], 0);
  const auto b_desc_lo = __shfl_sync(0xffffffffU, b_desc.reg32_[0], 0);
  const auto iterations = reduction / kStageK;
  if (threadIdx.x == kMathThreads) {
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto current = pipeline + iteration;
      const auto stage = current % StageCount;
      const auto phase = (current / StageCount) & 1U;
      empty[stage].wait(phase ^ 1U);
      auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem + stage * kStageStorageElements);
      auto *stage_b = stage_a + kAStageElements;
      const auto k = iteration * kStageK;
      if constexpr (MajorA == cute::UMMA::Major::K)
        deep_gemm::tma::copy<kStageK, TileM, 128,
                             cutlass::bfloat16_t>(
            &a_map, &full[stage], stage_a, a_k + k, a_m);
      else
        deep_gemm::tma::copy<TileM, kStageK, 128,
                             cutlass::bfloat16_t>(
            &a_map, &full[stage], stage_a, a_m, a_k + k);
      if constexpr (MajorB == cute::UMMA::Major::K)
        deep_gemm::tma::copy<kStageK, TileN, 128,
                             cutlass::bfloat16_t>(
            &b_map, &full[stage], stage_b, b_k + k, b_n);
      else
        deep_gemm::tma::copy<TileN, kStageK, 128,
                             cutlass::bfloat16_t>(
            &b_map, &full[stage], stage_b, b_n, b_k + k);
      full[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto current = pipeline + iteration;
      const auto stage = current % StageCount;
      const auto phase = (current / StageCount) & 1U;
      full[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
      const auto stage_desc_offset =
          stage * kStageStorageElements * sizeof(transport::BFloat16) / 16U;
#pragma unroll
      for (std::uint32_t wave = 0U; wave < kMWaves; ++wave)
#pragma unroll
        for (std::uint32_t k = 0U; k < kStageK / TileMma::K; ++k) {
          a_desc.reg32_[0] =
              deep_gemm::mma::sm90::advance_gmma_desc_lo<
                  MajorA, TileM, kStageK, 128,
                  cutlass::bfloat16_t>(
                  a_desc_lo + stage_desc_offset, wave * kStageM,
                  k * TileMma::K);
          b_desc.reg32_[0] =
              deep_gemm::mma::sm90::advance_gmma_desc_lo<
                  MajorB, TileN, kStageK, 128,
                  cutlass::bfloat16_t>(
                  b_desc_lo + stage_desc_offset, 0U, k * TileMma::K);
          TileMma::wgmma(a_desc, b_desc,
                         accum + wave * TileMma::kNumAccum, 1);
        }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      if (iteration != 0U) {
        deep_gemm::ptx::warpgroup_wait<1>();
        const auto completed = pipeline + iteration - 1U;
        if ((threadIdx.x & 31U) == 0U)
          empty[completed % StageCount].arrive();
      }
    }
    if (iterations != 0U) {
      deep_gemm::ptx::warpgroup_wait<0>();
      const auto completed = pipeline + iterations - 1U;
      if ((threadIdx.x & 31U) == 0U)
        empty[completed % StageCount].arrive();
    }
  }
  __syncthreads();
  if constexpr (TmaOutput) {
    static_assert(TileM % 128U == 0U && TileN % 64U == 0U);
    static_assert(OutputOffsetElements + TileM * TileN <=
                  shared_arena_elements<StorageTileN>());
    store_tma_accumulator<TileM, TileN, StorageTileN,
                          OutputOffsetElements>(
        accum, output_store_map, output_row, output_column, storage, false,
        true);
  } else {
#pragma unroll
    for (std::uint32_t wave = 0U; wave < kMWaves; ++wave)
      store_m128_accumulator<TileN>(
          accum + wave * TileMma::kNumAccum, output,
          output_row + wave * kStageM, output_column, output_width);
  }
  __syncthreads();
  pipeline += iterations;
}

template <std::uint32_t TileN, std::uint32_t StorageTileN = TileN>
__device__ __forceinline__ void run_k_mn_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_mn_map,
    std::uint32_t activation_row, std::uint32_t weight_reduction_start,
    std::uint32_t reduction, transport::BFloat16 *output,
    std::uint64_t output_width, std::uint32_t output_column,
    unsigned char *storage, std::uint32_t &pipeline) {
  run_wgmma_tile<kStageM, TileN, StorageTileN, cute::UMMA::Major::K,
                 cute::UMMA::Major::MN, false, kActivationStages, 0U, 0U>(
      activation_map, weight_mn_map, activation_map, activation_row,
      output_column, 0U,
      weight_reduction_start, reduction, output, output_width,
      activation_row, output_column, storage, pipeline);
}

template <std::uint32_t TileN, std::uint32_t StorageTileN = TileN>
__device__ __forceinline__ void run_k_mn_tile_tma_store(
    const CUtensorMap &activation_map, const CUtensorMap &weight_mn_map,
    const CUtensorMap &output_store_map, std::uint32_t activation_row,
    std::uint32_t weight_reduction_start, std::uint32_t reduction,
    transport::BFloat16 *output, std::uint64_t output_width,
    std::uint32_t output_row, std::uint32_t output_column,
    unsigned char *storage,
    std::uint32_t &pipeline) {
  run_wgmma_tile<kStageM, TileN, StorageTileN, cute::UMMA::Major::K,
                 cute::UMMA::Major::MN, true, kActivationStages, 0U, 0U>(
      activation_map, weight_mn_map, output_store_map, activation_row,
      output_column, 0U, weight_reduction_start, reduction, output,
      output_width, output_row, output_column, storage, pipeline);
}

template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t StorageTileN = TileN>
__device__ __forceinline__ void run_mn_mn_tile(
    const CUtensorMap &left_map, const CUtensorMap &right_map,
    std::uint32_t output_row, std::uint32_t output_column,
    std::uint32_t reduction_start, std::uint32_t reduction,
    transport::BFloat16 *output, std::uint64_t output_width,
    std::uint32_t global_output_row, const CUtensorMap &output_store_map,
    unsigned char *storage,
    std::uint32_t &pipeline) {
  run_wgmma_tile<TileM, TileN, StorageTileN, cute::UMMA::Major::MN,
                 cute::UMMA::Major::MN, true, kDwStages,
                 2U * kActivationStages,
                 kDwStages * stage_storage_elements<StorageTileN>()>(
      left_map, right_map, output_store_map, output_row, output_column,
      reduction_start, reduction_start, reduction, output, output_width,
      global_output_row, output_column, storage, pipeline);
}

// Once all activation work has completed, a claimed dW range cannot be
// preempted by a higher-priority queue.  Run that short contiguous range as
// one producer/consumer pipeline: the TMA warp may start the next tile while
// the math warps STSM/TMA-store the previous result.  This mirrors
// DeepGEMM's persistent block loop without weakening Q_DX > Q_DAB priority.
template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t StorageTileN = TileN>
__device__ __forceinline__ void run_mn_mn_chunk(
    const CUtensorMap &left_map, const CUtensorMap &right_map,
    const CUtensorMap &output_store_map, std::uint32_t first_tile,
    std::uint32_t tile_count, std::uint32_t n_tiles,
    std::uint32_t reduction_start, std::uint32_t reduction,
    std::uint32_t output_width, std::uint32_t output_owner_row,
    unsigned char *storage, std::uint32_t &pipeline) {
  using TileMma = typename deep_gemm::mma::sm90::BF16MMASelector<
      TileN, cute::UMMA::Major::MN, cute::UMMA::Major::MN>::type;
  constexpr std::uint32_t kMWaves = TileM / kStageM;
  constexpr std::uint32_t kAStageElements = TileM * kStageK;
  constexpr std::uint32_t kBStageElements = TileN * kStageK;
  constexpr auto kStageStorageElements =
      stage_storage_elements<StorageTileN>();
  static_assert(kAStageElements + kBStageElements <=
                kStageStorageElements);
  (void)output_width;

  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *barriers = reinterpret_cast<ForwardBarrier *>(
      smem + shared_arena_elements<StorageTileN>());
  auto *full = barriers + 2U * kActivationStages;
  auto *empty = full + kDwStages;
  const auto warpgroup = threadIdx.x / 128U;
  const auto iterations = reduction / kStageK;

  if (threadIdx.x == kMathThreads) {
    asm volatile("fence.proxy.async.global;" : : : "memory");
    for (std::uint32_t offset = 0U; offset < tile_count; ++offset) {
      const auto tile = first_tile + offset;
      const auto m_tile = tile / n_tiles;
      const auto n_tile = tile % n_tiles;
      for (std::uint32_t iteration = 0U; iteration < iterations;
           ++iteration) {
        const auto current = pipeline + offset * iterations + iteration;
        const auto stage = current % kDwStages;
        const auto phase = (current / kDwStages) & 1U;
        empty[stage].wait(phase ^ 1U);
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            smem + stage * kStageStorageElements);
        auto *stage_b = stage_a + kAStageElements;
        const auto k = reduction_start + iteration * kStageK;
        deep_gemm::tma::copy<TileM, kStageK, 128,
                             cutlass::bfloat16_t>(
            &left_map, &full[stage], stage_a, m_tile * TileM, k);
        deep_gemm::tma::copy<TileN, kStageK, 128,
                             cutlass::bfloat16_t>(
            &right_map, &full[stage], stage_b, n_tile * TileN, k);
        full[stage].arrive_and_expect_tx(
            (kAStageElements + kBStageElements) *
            sizeof(transport::BFloat16));
      }
    }
  } else if (warpgroup < kMathWarpGroups) {
    auto a_desc = deep_gemm::mma::sm90::make_gmma_desc<
        cute::UMMA::Major::MN, TileM, kStageK, 128,
        cutlass::bfloat16_t>(
        reinterpret_cast<cutlass::bfloat16_t *>(smem),
        warpgroup * TileMma::M, 0U);
    auto b_desc = deep_gemm::mma::sm90::make_gmma_desc<
        cute::UMMA::Major::MN, TileN, kStageK, 128,
        cutlass::bfloat16_t>(
        reinterpret_cast<cutlass::bfloat16_t *>(smem) + kAStageElements,
        0U, 0U);
    const auto a_desc_lo =
        __shfl_sync(0xffffffffU, a_desc.reg32_[0], 0);
    const auto b_desc_lo =
        __shfl_sync(0xffffffffU, b_desc.reg32_[0], 0);

    for (std::uint32_t offset = 0U; offset < tile_count; ++offset) {
      float accum[TileMma::kNumAccum * kMWaves];
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
        accum[index] = 0.0F;
      for (std::uint32_t iteration = 0U; iteration < iterations;
           ++iteration) {
        const auto current = pipeline + offset * iterations + iteration;
        const auto stage = current % kDwStages;
        const auto phase = (current / kDwStages) & 1U;
        full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
        deep_gemm::ptx::warpgroup_arrive();
        const auto stage_desc_offset =
            stage * kStageStorageElements * sizeof(transport::BFloat16) /
            16U;
#pragma unroll
        for (std::uint32_t wave = 0U; wave < kMWaves; ++wave)
#pragma unroll
          for (std::uint32_t k = 0U; k < kStageK / TileMma::K; ++k) {
            a_desc.reg32_[0] =
                deep_gemm::mma::sm90::advance_gmma_desc_lo<
                    cute::UMMA::Major::MN, TileM, kStageK, 128,
                    cutlass::bfloat16_t>(
                    a_desc_lo + stage_desc_offset, wave * kStageM,
                    k * TileMma::K);
            b_desc.reg32_[0] =
                deep_gemm::mma::sm90::advance_gmma_desc_lo<
                    cute::UMMA::Major::MN, TileN, kStageK, 128,
                    cutlass::bfloat16_t>(
                    b_desc_lo + stage_desc_offset, 0U,
                    k * TileMma::K);
            TileMma::wgmma(a_desc, b_desc,
                           accum + wave * TileMma::kNumAccum, 1);
          }
        deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
        deep_gemm::ptx::warpgroup_wait<0>();
        if ((threadIdx.x & 31U) == 0U)
          empty[stage].arrive();
      }

      const auto tile = first_tile + offset;
      const auto m_tile = tile / n_tiles;
      const auto n_tile = tile % n_tiles;
      store_tma_accumulator<
          TileM, TileN, StorageTileN,
          kDwStages * stage_storage_elements<StorageTileN>()>(
          accum, output_store_map,
          output_owner_row + m_tile * TileM, n_tile * TileN, storage,
          offset != 0U, offset + 1U == tile_count);
    }
  }
  __syncthreads();
  pipeline += tile_count * iterations;
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
