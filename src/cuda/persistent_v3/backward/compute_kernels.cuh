#pragma once

namespace backward_compute_detail {

constexpr std::uint32_t kActivationStages = backward_tuning::kDabStages;
constexpr std::uint32_t kDwStages = backward_tuning::kDw1Stages;
constexpr std::uint32_t kMathWarpGroups = 2U;
constexpr std::uint32_t kMathThreads = kMathWarpGroups * 128U;
constexpr std::uint32_t kActivationMathThreads =
    backward_tuning::kActivationTileM == 128U ? kMathThreads : 128U;
constexpr std::uint32_t kStageK = backward_tuning::kTileK;
constexpr std::uint32_t kStageM = backward_tuning::kTileM;
constexpr std::uint32_t kStoreReadyBarrier = 0U;
constexpr std::uint32_t kStoreDoneBarrier = 1U;

// The standalone baseline still uses every thread in its compute CTA. The
// unified kernel supplies a narrower synchronization domain, keeping its
// communication warps independent while sharing the exact GEMM implementation.
struct FullCtaComputeSynchronization {
  static constexpr std::uint32_t kThreads = backward_tuning::kComputeThreads;
  static constexpr std::uint32_t kWorkWarps = kThreads / 32U;
  static constexpr bool kMathEnabled = true;
  static constexpr bool kProducerEnabled = true;
  static constexpr bool kUnifiedReturnQueue = false;
  static constexpr std::uint32_t kActivationWaitGroup = 1U;

  __device__ __forceinline__ static void cta_sync() { __syncthreads(); }
  __device__ __forceinline__ static void math_sync(std::uint32_t threads,
                                                   std::uint32_t barrier) {
    cutlass::arch::NamedBarrier::sync(threads, barrier);
  }
  __device__ __forceinline__ void cluster_sync() { cute::cluster_sync(); }
  __device__ __forceinline__ void mailbox_read_done() { cta_sync(); }
};

template <std::uint32_t TileN>
constexpr std::uint32_t stage_storage_elements() {
  constexpr auto activation_elements =
      (backward_tuning::kActivationTileM + TileN) * kStageK;
  constexpr auto dw_elements =
      (backward_tuning::kDwTileM + backward_tuning::kDwTileN) * kStageK;
  return activation_elements > dw_elements ? activation_elements : dw_elements;
}

template <std::uint32_t TileN> constexpr std::uint32_t shared_arena_elements() {
  constexpr auto activation_elements =
      kActivationStages * stage_storage_elements<TileN>();
  constexpr auto dw_elements =
      kDwStages * stage_storage_elements<TileN>() +
      backward_tuning::kDwTileM * backward_tuning::kDwTileN;
  return activation_elements > dw_elements ? activation_elements : dw_elements;
}

template <std::uint32_t TileN> constexpr std::size_t dynamic_shared_bytes() {
  return static_cast<std::size_t>(shared_arena_elements<TileN>()) *
             sizeof(transport::BFloat16) +
         2U * (kActivationStages + kDwStages) *
             sizeof(cutlass::arch::ClusterTransactionBarrier);
}

static_assert(backward_tuning::kDabStages == backward_tuning::kDxStages);
static_assert(backward_tuning::kDw1Stages == backward_tuning::kDw2Stages);

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
__device__ __forceinline__ void release_stage(ForwardBarrier &empty) {
  // Only CTA 0 issues multicast, so it waits for both consumers. CTA 1
  // overwrites only its own unicast operand and needs just its local readers.
  const auto lane = threadIdx.x & 31U;
  if (lane == 0U)
    empty.arrive();
  if (lane == 1U && cute::block_rank_in_cluster() != 0U)
    empty.arrive(0U);
}

template <std::uint32_t TileN>
__device__ __forceinline__ void
initialize_k_mn_barriers(unsigned char *storage) {
  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *activation_full =
      reinterpret_cast<ForwardBarrier *>(smem + shared_arena_elements<TileN>());
  auto *activation_empty = activation_full + kActivationStages;
  auto *dw_full = activation_empty + kActivationStages;
  auto *dw_empty = dw_full + kDwStages;
  const auto readers =
      (cute::block_rank_in_cluster() == 0U ? 2U : 1U) * (kMathThreads / 32U);
  if (threadIdx.x == kMathThreads) {
#pragma unroll
    for (std::uint32_t stage = 0U; stage < kActivationStages; ++stage) {
      activation_full[stage].init(1);
      activation_empty[stage].init(readers * kActivationMathThreads /
                                   kMathThreads);
    }
#pragma unroll
    for (std::uint32_t stage = 0U; stage < kDwStages; ++stage) {
      dw_full[stage].init(1);
      dw_empty[stage].init(readers);
    }
    cutlass::arch::fence_barrier_init();
  }
}

template <std::uint32_t TileN>
__device__ __forceinline__ void
initialize_k_mn_pipeline(unsigned char *storage) {
  initialize_k_mn_barriers<TileN>(storage);
  cute::cluster_sync();
  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup < kMathWarpGroups)
    cutlass::arch::warpgroup_reg_alloc<backward_tuning::kMathRegisters>();
  else
    cutlass::arch::warpgroup_reg_dealloc<backward_tuning::kProducerRegisters>();
}

struct TmaOutput {
  template <unsigned TileM, unsigned TileN>
  __device__ __forceinline__ bool store(transport::BFloat16 *, unsigned,
                                       unsigned) const {
    return false;
  }
};

template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t OutputOffsetElements,
          class Synchronization = FullCtaComputeSynchronization,
          class Output = TmaOutput>
__device__ __forceinline__ void
store_tma_accumulator(float *accum, const CUtensorMap &output_store_map,
                      std::uint32_t output_row, std::uint32_t output_column,
                      unsigned char *storage, bool wait_previous,
                      bool wait_current, const Output &output = {}) {
  using TileMma =
      typename hopper::BF16MMASelector<TileN, cute::SM90::GMMA::Major::MN,
                                       cute::SM90::GMMA::Major::MN, kStageK,
                                       128U>::type;
  constexpr std::uint32_t kMWaves = TileM == 64U ? 1U : TileM / kStageM;
  constexpr std::uint32_t kNAtoms = TileN / 64U;
  constexpr std::uint32_t kStoreRows = TileM == 64U ? 64U : 128U;
  constexpr std::uint32_t kStoreTilesM = TileM / kStoreRows;
  constexpr std::uint32_t kStoreThreads = TileM == 64U ? 128U : kMathThreads;
  if (threadIdx.x >= kStoreThreads)
    return;

  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *output_tile = smem + OutputOffsetElements;
  if (wait_previous && threadIdx.x < kNAtoms * kStoreTilesM)
    asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
  Synchronization::math_sync(kStoreThreads, kStoreReadyBarrier);

  const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
  const auto lane = threadIdx.x & 31U;
#pragma unroll
  for (std::uint32_t wave = 0U; wave < kMWaves; ++wave) {
    auto *wave_accum = accum + wave * TileMma::kNumAccum;
#pragma unroll
    for (std::uint32_t index = 0U; index < TileMma::kNumAccum / 4U; ++index) {
      constexpr std::uint32_t kStoreAtomN = 64U;
      constexpr std::uint32_t kSwizzleBytes = 128U;
      constexpr std::uint32_t kBankGroupBytes = 16U;
      const auto atom = index / (kStoreAtomN / 8U);
      const auto in_atom = index % (kStoreAtomN / 8U);
      const auto swizzle_row = lane;
      auto swizzle_column = in_atom;
      swizzle_column ^= swizzle_row % (kSwizzleBytes / kBankGroupBytes);
      auto *smem_pointer =
          reinterpret_cast<std::uint8_t *>(output_tile) +
          atom * TileM * kSwizzleBytes + wave * kStageM * kSwizzleBytes +
          warp * 16U * kSwizzleBytes + swizzle_row * kSwizzleBytes +
          swizzle_column * kBankGroupBytes;
      hopper::StoreMatrixX2<__nv_bfloat162>::copy(
          __float22bfloat162_rn(
              make_float2(wave_accum[index * 4U], wave_accum[index * 4U + 1U])),
          __float22bfloat162_rn(make_float2(wave_accum[index * 4U + 2U],
                                            wave_accum[index * 4U + 3U])),
          smem_pointer);
    }
  }
  cute::tma_store_fence();
  Synchronization::math_sync(kStoreThreads, kStoreDoneBarrier);
  if (output.template store<TileM, TileN>(output_tile, output_row, output_column))
    return;
  if (threadIdx.x < kNAtoms * kStoreTilesM) {
    const auto atom = threadIdx.x % kNAtoms;
    const auto m_half = threadIdx.x / kNAtoms;
    cute::SM90_TMA_STORE_2D::copy(
        &output_store_map,
        output_tile + atom * TileM * 64U + m_half * kStoreRows * 64U,
        output_column + atom * 64U, output_row + m_half * kStoreRows);
    cute::tma_store_arrive();
    if (wait_current)
      asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
  }
  // The next epilogue waits for this store before reusing output_tile.  Do
  // not hold both math warpgroups here: after their STSM writes have crossed
  // kStoreDoneBarrier, they may start the next WGMMA tile while TMA drains.
  __syncwarp();
}

template <std::uint32_t TileN, std::uint32_t StorageTileN = TileN,
          class Synchronization = FullCtaComputeSynchronization,
          class Output = TmaOutput>
__device__ __forceinline__ void run_k_mn_tile_tma_store(
    const CUtensorMap &activation_map, const CUtensorMap &weight_mn_map,
    const CUtensorMap &output_store_map, std::uint32_t activation_row,
    std::uint32_t weight_reduction_start, std::uint32_t reduction,
    std::uint32_t output_row, std::uint32_t output_column,
    unsigned char *storage, std::uint32_t &pipeline, const Output &output = {}) {
  constexpr auto TileM = backward_tuning::kActivationTileM;
  constexpr bool kPairN = backward_tuning::kActivationClusterN == 2U;
  constexpr auto MajorA = cute::SM90::GMMA::Major::K;
  constexpr auto MajorB = cute::SM90::GMMA::Major::MN;
  constexpr auto StageCount = kActivationStages;
  static_assert(Synchronization::kActivationWaitGroup <= 1U,
                "Activation WGMMA supports zero or one outstanding group");
  static_assert(TileM == 64U || TileM == 128U);
  const auto cluster_rank = cute::block_rank_in_cluster();
  const auto row_half = kPairN ? 0U : cluster_rank * TileM;
  const auto a_m = activation_row + row_half;
  output_row += row_half;
  if constexpr (kPairN)
    output_column += cluster_rank * TileN;
  static_assert(TileN == 128U || TileN == 256U);
  using TileMma = typename hopper::BF16MMASelector<TileN, MajorA, MajorB,
                                                   kStageK, 128U>::type;
  constexpr std::uint32_t kAStageElements = TileM * kStageK;
  constexpr std::uint32_t kBStageElements = TileN * kStageK;
  constexpr auto kStageStorageElements = stage_storage_elements<StorageTileN>();
  static_assert(kAStageElements + kBStageElements <= kStageStorageElements);
  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *barriers = reinterpret_cast<ForwardBarrier *>(
      smem + shared_arena_elements<StorageTileN>());
  auto *full = barriers;
  auto *empty = full + StageCount;

  // The scheduler/dispatch acquire is in the generic proxy. Make its dR/dAB
  // payload writes visible before the producer issues the first TMA load.
  if (Synchronization::kProducerEnabled && threadIdx.x == kMathThreads)
    asm volatile("fence.proxy.async.global;" : : : "memory");

  const auto iterations = reduction / kStageK;
  if (Synchronization::kProducerEnabled && threadIdx.x == kMathThreads) {
    for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
      const auto current = pipeline + iteration;
      const auto stage = current % StageCount;
      const auto phase = (current / StageCount) & 1U;
      empty[stage].wait(phase ^ 1U);
      auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem + stage * kStageStorageElements);
      auto *stage_b = stage_a + kAStageElements;
      const auto k = iteration * kStageK;
      hopper::tma_load<kStageK, TileM, 128, cutlass::bfloat16_t>(
          &activation_map, &full[stage], stage_a, k, a_m, kPairN ? 2U : 1U);
      hopper::tma_load<TileN, kStageK, 128, cutlass::bfloat16_t>(
          &weight_mn_map, &full[stage], stage_b, output_column,
          weight_reduction_start + k, kPairN ? 1U : 2U);
      full[stage].arrive_and_expect_tx((kAStageElements + kBStageElements) *
                                       sizeof(transport::BFloat16));
    }
  }
  if constexpr (Synchronization::kMathEnabled) {
    if (threadIdx.x < kActivationMathThreads) {
      // Keep accumulators entirely in the math role. Conditional initialization
      // outside this branch lets ptxas carry unused producer-side values across
      // task kinds, spilling N128 accumulators when dW uses N256.
      float accum[TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum; ++index)
        accum[index] = 0.0F;
      auto a_desc = hopper::make_gmma_desc<MajorA, TileM, kStageK, 128,
                                           cutlass::bfloat16_t>(
          reinterpret_cast<cutlass::bfloat16_t *>(smem),
          (threadIdx.x / 128U) * TileMma::M, 0U);
      auto b_desc = hopper::make_gmma_desc<MajorB, TileN, kStageK, 128,
                                           cutlass::bfloat16_t>(
          reinterpret_cast<cutlass::bfloat16_t *>(smem) + kAStageElements, 0U,
          0U);
      const auto a_desc_lo = __shfl_sync(0xffffffffU, a_desc.reg32_[0], 0);
      const auto b_desc_lo = __shfl_sync(0xffffffffU, b_desc.reg32_[0], 0);

      for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto current = pipeline + iteration;
        const auto stage = current % StageCount;
        const auto phase = (current / StageCount) & 1U;
        full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum[index]);
        cute::warpgroup_arrive();
        const auto stage_desc_offset =
            stage * kStageStorageElements * sizeof(transport::BFloat16) / 16U;
#pragma unroll
        for (std::uint32_t k = 0U; k < kStageK / TileMma::K; ++k) {
          a_desc.reg32_[0] =
              hopper::advance_gmma_desc_lo<MajorA, TileM, kStageK, 128,
                                           cutlass::bfloat16_t>(
                  a_desc_lo + stage_desc_offset, 0U, k * TileMma::K);
          b_desc.reg32_[0] =
              hopper::advance_gmma_desc_lo<MajorB, TileN, kStageK, 128,
                                           cutlass::bfloat16_t>(
                  b_desc_lo + stage_desc_offset, 0U, k * TileMma::K);
          TileMma::wgmma(a_desc, b_desc, accum, 1);
        }
        cute::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum[index]);
        if constexpr (Synchronization::kActivationWaitGroup == 0U) {
          // Complete this K64 group before releasing its operand stage. This
          // avoids conservative WGMMA serialization in the unified entry.
          cute::warpgroup_wait<0>();
          release_stage(empty[stage]);
        } else if (iteration != 0U) {
          cute::warpgroup_wait<1>();
          const auto completed = pipeline + iteration - 1U;
          release_stage(empty[completed % StageCount]);
        }
      }
      if constexpr (Synchronization::kActivationWaitGroup == 1U) {
        if (iterations != 0U) {
          cute::warpgroup_wait<0>();
          const auto completed = pipeline + iterations - 1U;
          release_stage(empty[completed % StageCount]);
        }
      }
      // The final WGMMA wait has consumed the last TMA stage. The TMA store
      // helper synchronizes the math warpgroup before reusing the arena;
      // producer threads must not carry accumulators across a CTA-wide join.
      static_assert(TileM * TileN <= shared_arena_elements<StorageTileN>());
      store_tma_accumulator<TileM, TileN, 0U, Synchronization>(
          accum, output_store_map, output_row, output_column, storage, false,
          true, output);
    }
  }
  // The compute roles rejoin before the next task reuses the pipeline. In a
  // unified CTA this excludes the independently progressing communication
  // warps.
  Synchronization::cta_sync();
  pipeline += iterations;
}

// Once all activation work has completed, a claimed dW range cannot be
// preempted by a higher-priority queue.  Run that short contiguous range as
// one producer/consumer pipeline: the TMA warp may start the next tile while
// the math warps STSM/TMA-store the previous result.  This mirrors
// DeepGEMM's persistent block loop without weakening Q_DX > Q_DAB priority.
template <std::uint32_t TileM, std::uint32_t TileN,
          std::uint32_t StorageTileN = TileN,
          class Synchronization = FullCtaComputeSynchronization>
__device__ __forceinline__ void
run_mn_mn_chunk(const CUtensorMap &left_map, const CUtensorMap &right_map,
                const CUtensorMap &output_store_map, std::uint32_t first_tile,
                std::uint32_t tile_count, std::uint32_t m_tiles,
                std::uint32_t n_tiles, std::uint32_t left_reduction_start,
                std::uint32_t right_reduction_start, std::uint32_t reduction,
                std::uint32_t output_width, std::uint32_t output_owner_row,
                unsigned char *storage, std::uint32_t &pipeline) {
  using TileMma =
      typename hopper::BF16MMASelector<TileN, cute::SM90::GMMA::Major::MN,
                                       cute::SM90::GMMA::Major::MN, kStageK,
                                       128U>::type;
  constexpr std::uint32_t kMWaves = TileM / kStageM;
  constexpr std::uint32_t kAStageElements = TileM * kStageK;
  constexpr std::uint32_t kBStageElements = TileN * kStageK;
  constexpr auto kStageStorageElements = stage_storage_elements<StorageTileN>();
  static_assert(kAStageElements + kBStageElements <= kStageStorageElements);
  (void)output_width;
  const auto rank = cute::block_rank_in_cluster();
  using persistent_v3::backward::cluster_tile;

  auto *smem = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *barriers = reinterpret_cast<ForwardBarrier *>(
      smem + shared_arena_elements<StorageTileN>());
  auto *full = barriers + 2U * kActivationStages;
  auto *empty = full + kDwStages;
  const auto warpgroup = threadIdx.x / 128U;
  const auto iterations = reduction / kStageK;

  if (Synchronization::kProducerEnabled && threadIdx.x == kMathThreads) {
    asm volatile("fence.proxy.async.global;" : : : "memory");
    for (std::uint32_t offset = 0U; offset < tile_count; ++offset) {
      const auto tile = cluster_tile(first_tile + offset, rank, m_tiles,
                                     n_tiles, TileM > TileN);
      const auto m_tile = tile.m;
      const auto n_tile = tile.n;
      for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration) {
        const auto current = pipeline + offset * iterations + iteration;
        const auto stage = current % kDwStages;
        const auto phase = (current / kDwStages) & 1U;
        empty[stage].wait(phase ^ 1U);
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            smem + stage * kStageStorageElements);
        auto *stage_b = stage_a + kAStageElements;
        const auto left_k = left_reduction_start + iteration * kStageK;
        const auto right_k = right_reduction_start + iteration * kStageK;
        hopper::tma_load<TileM, kStageK, 128, cutlass::bfloat16_t>(
            &left_map, &full[stage], stage_a, m_tile * TileM, left_k,
            tile.share_a ? 2U : 1U);
        hopper::tma_load<TileN, kStageK, 128, cutlass::bfloat16_t>(
            &right_map, &full[stage], stage_b, n_tile * TileN, right_k,
            tile.share_a ? 1U : 2U);
        full[stage].arrive_and_expect_tx((kAStageElements + kBStageElements) *
                                         sizeof(transport::BFloat16));
      }
    }
  }
  if constexpr (Synchronization::kMathEnabled) {
    if (warpgroup < kMathWarpGroups) {
      auto a_desc = hopper::make_gmma_desc<cute::SM90::GMMA::Major::MN, TileM,
                                           kStageK, 128, cutlass::bfloat16_t>(
          reinterpret_cast<cutlass::bfloat16_t *>(smem), warpgroup * TileMma::M,
          0U);
      auto b_desc = hopper::make_gmma_desc<cute::SM90::GMMA::Major::MN, TileN,
                                           kStageK, 128, cutlass::bfloat16_t>(
          reinterpret_cast<cutlass::bfloat16_t *>(smem) + kAStageElements, 0U,
          0U);
      const auto a_desc_lo = __shfl_sync(0xffffffffU, a_desc.reg32_[0], 0);
      const auto b_desc_lo = __shfl_sync(0xffffffffU, b_desc.reg32_[0], 0);

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
            cute::warpgroup_fence_operand(accum[index]);
          cute::warpgroup_arrive();
          const auto stage_desc_offset =
              stage * kStageStorageElements * sizeof(transport::BFloat16) / 16U;
#pragma unroll
          for (std::uint32_t wave = 0U; wave < kMWaves; ++wave)
#pragma unroll
            for (std::uint32_t k = 0U; k < kStageK / TileMma::K; ++k) {
              a_desc.reg32_[0] =
                  hopper::advance_gmma_desc_lo<cute::SM90::GMMA::Major::MN,
                                               TileM, kStageK, 128,
                                               cutlass::bfloat16_t>(
                      a_desc_lo + stage_desc_offset, wave * kStageM,
                      k * TileMma::K);
              b_desc.reg32_[0] =
                  hopper::advance_gmma_desc_lo<cute::SM90::GMMA::Major::MN,
                                               TileN, kStageK, 128,
                                               cutlass::bfloat16_t>(
                      b_desc_lo + stage_desc_offset, 0U, k * TileMma::K);
              TileMma::wgmma(a_desc, b_desc, accum + wave * TileMma::kNumAccum,
                             1);
            }
          cute::warpgroup_commit_batch();
#pragma unroll
          for (int index = 0; index < TileMma::kNumAccum * kMWaves; ++index)
            cute::warpgroup_fence_operand(accum[index]);
          // Release promptly to keep all TMA stages available to the multicast
          // producer. Holding a completed stage reduces memory prefetch depth.
          cute::warpgroup_wait<0>();
          release_stage(empty[stage]);
        }

        const auto tile = cluster_tile(first_tile + offset, rank, m_tiles,
                                       n_tiles, TileM > TileN);
        const auto m_tile = tile.m;
        const auto n_tile = tile.n;
        if (tile.valid)
          store_tma_accumulator<
              TileM, TileN, kDwStages * stage_storage_elements<StorageTileN>(),
              Synchronization>(accum, output_store_map,
                               output_owner_row + m_tile * TileM,
                               n_tile * TileN, storage, true, false);
      }
    }
  }
  Synchronization::cta_sync();
  // A following activation task may reuse the dW output staging area. Keep
  // stores pipelined within this range, then drain before the cluster rejoins.
  if (threadIdx.x < (TileM / 128U) * (TileN / 64U))
    asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
  Synchronization::cta_sync();
  pipeline += tile_count * iterations;
}
#endif

} // namespace backward_compute_detail
