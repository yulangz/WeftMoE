#include "cuda/forward/kernel_support.cuh"

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) // compiled production MMA path only
using ForwardMmaNarrow =
    typename deep_gemm::mma::sm90::BF16MMASelector<64>::type;
using ForwardMma = typename deep_gemm::mma::sm90::BF16MMASelector<128>::type;
using ForwardMmaWide =
    typename deep_gemm::mma::sm90::BF16MMASelector<256>::type;
// The split Forward kernel is compiled once per explicit tuning record.  Keep
  // Give the production FC1/FC2 paths independent N shapes.
using TunedForwardFc1Mma =
    typename deep_gemm::mma::sm90::BF16MMASelector<
        forward_tuning::kFc1TileN>::type;
using TunedForwardFc1M64Mma =
    typename deep_gemm::mma::sm90::BF16MMASelector<
        forward_tuning::kFc1TileN / 2U>::type;
using TunedForwardFc2Mma =
    typename deep_gemm::mma::sm90::BF16MMASelector<
        forward_tuning::kFc2TileN>::type;
using TunedForwardFc2M64Mma =
    typename deep_gemm::mma::sm90::BF16MMASelector<
        forward_tuning::kFc2TileN / 2U>::type;
using ForwardBarrier = cutlass::arch::ClusterTransactionBarrier;


// FC1 consumes the same activation tile twice (the A and B projections of
// SwiGLU).  Keep both projection accumulators live and issue the two WGMMA
// groups from one TMA-loaded activation tile. Three pipeline stages fill the
// Hopper shared-memory union and hide the long 7168-element reduction loads;
// FC2 reinitializes its independent four-stage pipeline below.
__device__ void forward_wgmma_swiglu_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_map,
    std::uint32_t activation_row, std::uint32_t weight_a_row,
    std::uint32_t weight_b_row, std::uint32_t reduction,
    transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum_a, float *accum_b,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 3;
  constexpr std::uint32_t kMathWarpGroups = 2;
  constexpr std::uint32_t kStageElements = 128U * 64U;
  auto *smem_b2_public = smem_b_public + kStages * kStageElements;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
      accum_a[index] = 0.0F;
      accum_b[index] = 0.0F;
    }
  }
  const auto iterations = (reduction + 63U) / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      const auto k_base = iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements);
      auto *smem_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      auto *smem_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b2_public + stage * kStageElements);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &activation_map, &full_barriers[stage], smem_a, k_base,
          activation_row);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b1, k_base, weight_a_row);
      deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b2, k_base, weight_b_row);
      full_barriers[stage].arrive_and_expect_tx(
          3U * kStageElements * sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements +
          warpgroup * 64U * 64U);
      auto *smem_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      auto *smem_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b2_public + stage * kStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
      }
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / ForwardMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
            smem_a + k * ForwardMma::K, 1);
        const auto desc_b1 = deep_gemm::mma::sm90::make_smem_desc(
            smem_b1 + k * ForwardMma::K, 1);
        const auto desc_b2 = deep_gemm::mma::sm90::make_smem_desc(
            smem_b2 + k * ForwardMma::K, 1);
        ForwardMma::wgmma(desc_a, desc_b1, accum_a, 1);
        ForwardMma::wgmma(desc_a, desc_b2, accum_b, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
      }
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

template <class TileMma, std::uint32_t TileColumns>
__device__ void forward_wgmma_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_map,
    std::uint32_t activation_row, std::uint32_t weight_row,
    std::uint32_t reduction, transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration, std::uint32_t reduction_start = 0U
#if MEGAMOE_DEBUG_VALIDATE
    ,
    unsigned long long *debug_producer = nullptr,
    unsigned long long *debug_math0 = nullptr,
    unsigned long long *debug_math1 = nullptr
#endif
) {
  constexpr std::uint32_t kStages = 4;
  constexpr std::uint32_t kMathWarpGroups = 2;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = TileColumns * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups)
#pragma unroll
    for (int index = 0; index < TileMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  const auto iterations = (reduction + 63U) / 64U;
#if MEGAMOE_DEBUG_VALIDATE
  const bool debug_first_tile = pipeline_iteration == 0U;
#endif
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && debug_producer)
        *debug_producer =
            (static_cast<unsigned long long>(iteration) << 8U) | 1ULL;
#endif
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && debug_producer)
        *debug_producer =
            (static_cast<unsigned long long>(iteration) << 8U) | 2ULL;
#endif
      const auto k_base = reduction_start + iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      deep_gemm::tma::copy<64, TileColumns, 128, cutlass::bfloat16_t>(
          &activation_map, &full_barriers[stage], smem_a, k_base,
          activation_row);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b, k_base, weight_row);
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && debug_producer)
        *debug_producer =
            (static_cast<unsigned long long>(iteration) << 8U) | 3ULL;
#endif
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) * sizeof(transport::BFloat16));
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && debug_producer)
        *debug_producer =
            (static_cast<unsigned long long>(iteration) << 8U) | 4ULL;
#endif
    }
  } else if (warpgroup < kMathWarpGroups) {
#if MEGAMOE_DEBUG_VALIDATE
    auto *debug_math = warpgroup == 0U ? debug_math0 : debug_math1;
#endif
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && (threadIdx.x & 127U) == 0U && debug_math)
        *debug_math = (static_cast<unsigned long long>(iteration) << 8U) | 1ULL;
#endif
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements +
          warpgroup * 64U * 64U);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && (threadIdx.x & 127U) == 0U && debug_math)
        *debug_math = (static_cast<unsigned long long>(iteration) << 8U) | 2ULL;
#endif
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / TileMma::K; ++k) {
        const auto desc_a =
            deep_gemm::mma::sm90::make_smem_desc(smem_a + k * TileMma::K, 1);
        const auto desc_b =
            deep_gemm::mma::sm90::make_smem_desc(smem_b + k * TileMma::K, 1);
        TileMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && (threadIdx.x & 127U) == 0U && debug_math)
        *debug_math = (static_cast<unsigned long long>(iteration) << 8U) | 3ULL;
#endif
#pragma unroll
      for (int index = 0; index < TileMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      if (iteration != 0U) {
        deep_gemm::ptx::warpgroup_wait<1>();
#if MEGAMOE_DEBUG_VALIDATE
        if (debug_first_tile && (threadIdx.x & 127U) == 0U && debug_math)
          *debug_math =
              (static_cast<unsigned long long>(iteration) << 8U) | 4ULL;
#endif
        const auto completed_pipeline = pipeline_iteration + iteration - 1U;
        if ((threadIdx.x & 31U) == 0U)
          empty_barriers[completed_pipeline % kStages].arrive();
      }
    }
    if (iterations != 0U) {
      deep_gemm::ptx::warpgroup_wait<0>();
#if MEGAMOE_DEBUG_VALIDATE
      if (debug_first_tile && (threadIdx.x & 127U) == 0U && debug_math)
        *debug_math =
            (static_cast<unsigned long long>(iterations) << 8U) | 5ULL;
#endif
      const auto final_pipeline = pipeline_iteration + iterations - 1U;
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[final_pipeline % kStages].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

template <class Callback>
__device__ void visit_forward_accumulator(float *accum,
                                          Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMma::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
      callback(warpgroup, row0, column, accum[index * 4U],
               accum[index * 4U + 1U]);
      callback(warpgroup, row1, column, accum[index * 4U + 2U],
               accum[index * 4U + 3U]);
  }
}

template <class Callback>
__device__ void visit_forward_accumulator_pair(float *accum_a, float *accum_b,
                                               Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMma::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum_a[index * 4U],
             accum_a[index * 4U + 1U], accum_b[index * 4U],
             accum_b[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum_a[index * 4U + 2U],
             accum_a[index * 4U + 3U], accum_b[index * 4U + 2U],
             accum_b[index * 4U + 3U]);
  }
}

template <class Callback>
__device__ void visit_forward_accumulator_wide(float *accum,
                                               Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMmaWide::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

#endif

// Forward normally uses uniform M128 blocks.  The latency-wavefront variant
// preserves the same 128-row physical expert padding but exposes the first
// physical block as two M64 compute tasks: [0,64), [64,128), followed by
// aligned M128 tasks.  Keeping this address calculation in one helper prevents
// Dispatch, compute, and Return from silently disagreeing about block identity.
__device__ __forceinline__ std::uint64_t forward_expert_mblock_count(
    const ForwardKernelArgs &a, std::uint32_t expert) {
  const auto rows = a.dispatch.expert_row_count[expert];
  const bool split_head = a.forward_m64_head_wavefront &&
                          expert < a.forward_m64_head_expert_count;
  if (!split_head)
    return (rows + a.forward_mblock_rows - 1ULL) / a.forward_mblock_rows;
  if (rows == 0ULL)
    return 0ULL;
  const auto head_blocks = rows <= 64ULL ? 1ULL : 2ULL;
  const auto body_rows = rows > 128ULL ? rows - 128ULL : 0ULL;
  return head_blocks + (body_rows + 127ULL) / 128ULL;
}

__device__ __forceinline__ std::uint64_t forward_expert_head_mblock_count(
    const ForwardKernelArgs &a, std::uint32_t expert) {
  if (!a.forward_m64_head_wavefront ||
      expert >= a.forward_m64_head_expert_count)
    return 0ULL;
  const auto rows = a.dispatch.expert_row_count[expert];
  return rows == 0ULL ? 0ULL : (rows <= 64ULL ? 1ULL : 2ULL);
}

__device__ __forceinline__ std::uint64_t forward_mblock_local_row(
    const ForwardKernelArgs &a, std::uint32_t expert,
    std::uint32_t local_mblock) {
  const bool split_head = a.forward_m64_head_wavefront &&
                          expert < a.forward_m64_head_expert_count;
  if (!split_head)
    return static_cast<std::uint64_t>(local_mblock) * a.forward_mblock_rows;
  if (local_mblock < 2U)
    return static_cast<std::uint64_t>(local_mblock) * 64ULL;
  return 128ULL + static_cast<std::uint64_t>(local_mblock - 2U) * 128ULL;
}

__device__ __forceinline__ std::uint32_t forward_mblock_extent(
    const ForwardKernelArgs &a, std::uint32_t expert,
    std::uint32_t local_mblock) {
  return a.forward_m64_head_wavefront &&
                 expert < a.forward_m64_head_expert_count &&
                 local_mblock < 2U
             ? 64U
             : a.forward_mblock_rows;
}

__device__ __forceinline__ std::uint32_t forward_mblock_valid_rows(
    const ForwardKernelArgs &a, std::uint32_t expert,
    std::uint32_t local_mblock) {
  const auto local_row = forward_mblock_local_row(a, expert, local_mblock);
  const auto remaining = a.dispatch.expert_row_count[expert] - local_row;
  const auto extent = forward_mblock_extent(a, expert, local_mblock);
  return static_cast<std::uint32_t>(remaining < extent ? remaining : extent);
}

enum ForwardMixedSchedulerWord : std::uint32_t {
  kForwardHeadMblocks = 0U,
  kForwardBodyFc1Publish = 1U,
  kForwardBodyFc1Claim = 2U,
  kForwardBodyFc2Publish = 3U,
  kForwardBodyFc2Claim = 4U,
  // The last three words in the aligned scheduler gap form a transport-only
  // startup rendezvous.  Keeping one immutable epoch publication and two
  // monotonic arrival counters avoids resetting a reusable barrier while a
  // slower CTA can still be observing the previous phase.
  kForwardTransportInitEpoch = 5U,
  kForwardTransportClearArrivals = 6U,
  kForwardTransportMapArrivals = 7U,
};

enum class ForwardTaskPartition : std::uint32_t {
  All = 0U,
  Head = 1U,
  Body = 2U,
};

__device__ __forceinline__ unsigned long long *forward_mixed_scheduler_word(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    ForwardMixedSchedulerWord word) {
  // Return-directory construction owns two arrays of one word per
  // (remote-node, GIN-context), beginning at 16 + 2*world.  Mixed scheduling
  // state must live after both arrays; placing it in the apparent gap before
  // that base aliases the arrays for world=2/16 once directory scans begin.
  const auto partitions = a.dispatch.identity.num_nodes - 1ULL;
  const auto mixed_base = 16ULL + 2ULL * a.world +
                          2ULL * partitions * a.gin_context_count;
  return scheduler + mixed_base +
         static_cast<std::uint32_t>(word);
}

__device__ __forceinline__ void forward_transport_wait_epoch(
    unsigned long long *word, unsigned long long epoch) {
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(*word);
    while (published.load(::cuda::memory_order_acquire) != epoch)
      __nanosleep(256U);
  }
  __syncthreads();
}

// A fixed transport grid needs only a CTA-arrival barrier, not cooperative
// launch semantics.  Every CTA first releases all of its lanes' stores through
// one acq_rel RMW.  Waiting on the terminal counter value then acquires the
// complete RMW release sequence before the next initialization phase begins.
__device__ __forceinline__ void forward_transport_cta_barrier(
    unsigned long long *arrivals, unsigned int expected_ctas) {
  __syncthreads();
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> count(
        *arrivals);
    count.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
    while (count.load(::cuda::memory_order_acquire) < expected_ctas)
      __nanosleep(256U);
  }
  __syncthreads();
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
// Map the immutable M-block task space in the same wavefront order used by
// Notify packet priorities: local M tile first, then expert.  This lets the
// bounded compute scheduler consume the earliest completed tiles without a
// global Dispatch barrier or an O(N-task) atomic scan.
__device__ std::uint32_t forward_wavefront_mblock_expert(
    const ForwardKernelArgs &a, std::uint64_t ordinal,
    std::uint64_t &local_mblock) {
  if (a.forward_m64_head_wavefront) {
    const auto head_experts =
        min(a.forward_m64_head_expert_count,
            a.dispatch.sizes.owned_expert_count);
    // Keep every M64 task contiguous at the head of the immutable mapping so
    // the head and body kernels can use independent monotonic queue cursors.
    for (std::uint32_t expert = 0; expert < head_experts; ++expert) {
      const auto blocks = forward_expert_head_mblock_count(a, expert);
      for (std::uint64_t tile = 0; tile < blocks; ++tile) {
        if (ordinal == 0ULL) {
          local_mblock = tile;
          return expert;
        }
        --ordinal;
      }
    }

    // The body returns to M128 wavefront order. Split experts start at logical
    // block 2 (physical row 128); unsplit experts start at block 0.
    std::uint64_t max_body_blocks = 0ULL;
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto blocks = forward_expert_mblock_count(a, expert);
      const auto skipped = expert < head_experts
                               ? forward_expert_head_mblock_count(a, expert)
                               : 0ULL;
      const auto body_blocks = blocks - skipped;
      max_body_blocks = body_blocks > max_body_blocks ? body_blocks
                                                       : max_body_blocks;
    }
    for (std::uint64_t body_tile = 0; body_tile < max_body_blocks;
         ++body_tile) {
      for (std::uint32_t expert = 0;
           expert < a.dispatch.sizes.owned_expert_count; ++expert) {
        const auto blocks = forward_expert_mblock_count(a, expert);
        const auto skipped = expert < head_experts
                                 ? forward_expert_head_mblock_count(a, expert)
                                 : 0ULL;
        if (body_tile >= blocks - skipped)
          continue;
        if (ordinal == 0ULL) {
          local_mblock = body_tile + skipped;
          return expert;
        }
        --ordinal;
      }
    }
    local_mblock = 0ULL;
    return a.dispatch.sizes.owned_expert_count;
  }

  std::uint64_t max_blocks = 0ULL;
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const auto blocks = forward_expert_mblock_count(a, expert);
    max_blocks = blocks > max_blocks ? blocks : max_blocks;
  }
  for (std::uint64_t tile = 0; tile < max_blocks; ++tile) {
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto blocks = forward_expert_mblock_count(a, expert);
      if (tile >= blocks)
        continue;
      if (ordinal == 0ULL) {
        local_mblock = tile;
        return expert;
      }
      --ordinal;
    }
  }
  local_mblock = 0ULL;
  return a.dispatch.sizes.owned_expert_count;
}

// The readiness queue, rather than this immutable lookup, now determines
// when a GIN FC1 task becomes runnable. Keep two tile-first waves for startup
// latency, then place each expert's body M-blocks contiguously so a readiness
// publisher can release small weight-local groups without changing the wire
// order or waiting for all Dispatch traffic.
__device__ std::uint32_t forward_locality_mblock_expert(
    const ForwardKernelArgs &a, std::uint64_t ordinal,
    std::uint64_t &local_mblock) {
  if (a.forward_fc1_locality_group <= 1U ||
      a.forward_m64_head_wavefront)
    return forward_wavefront_mblock_expert(a, ordinal, local_mblock);

  constexpr std::uint64_t kStartupMblocks = 2ULL;
  for (std::uint64_t tile = 0; tile < kStartupMblocks; ++tile) {
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      if (tile >= forward_expert_mblock_count(a, expert))
        continue;
      if (ordinal == 0ULL) {
        local_mblock = tile;
        return expert;
      }
      --ordinal;
    }
  }
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const auto blocks = forward_expert_mblock_count(a, expert);
    for (std::uint64_t tile = kStartupMblocks; tile < blocks; ++tile) {
      if (ordinal == 0ULL) {
        local_mblock = tile;
        return expert;
      }
      --ordinal;
    }
  }
  local_mblock = 0ULL;
  return a.dispatch.sizes.owned_expert_count;
}

#endif

__device__ bool forward_cancelled(const ForwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(const_cast<unsigned long long *>(a.cancel_epoch),
                          0ULL) == a.epoch;
}

template <bool StaticLocal>
__device__ __forceinline__ bool
forward_mblock_rows_ready(const ForwardKernelArgs &a, std::uint32_t expert,
                          std::uint64_t local_row, std::uint32_t valid_rows) {
  if constexpr (StaticLocal) {
    if (!a.forward_stream_local_dispatch)
      return true;
  }
  if ((!a.use_gin && !a.forward_stream_local_dispatch) || valid_rows == 0U)
    return true;
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived = peer_at<unsigned long long>(a.local_base,
                                               a.x_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    unsigned long long total = 0ULL;
    for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          arrived[tile * a.local_world_size + rail]);
      total += ready.load(::cuda::memory_order_acquire);
    }
    if (total < expected)
      return false;
    tile_row = tile_row_end;
  }
  return true;
}

template <bool StaticLocal>
__device__ void wait_forward_mblock_rows(
    const ForwardKernelArgs &a, std::uint32_t expert, std::uint64_t local_row,
    std::uint32_t valid_rows,
    transport_state::GenerationCompletionVector *completion) {
  if constexpr (StaticLocal)
    return;
  if (!a.use_gin || valid_rows == 0U)
    return;
  // One elected lane polls the peer-visible rail counters; the CTA-level
  // synchronization at the call site releases the remaining lanes. Polling
  // from all 384 threads only multiplies L2/system traffic while conveying no
  // additional readiness information.
  if (threadIdx.x != 0U)
    return;
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived =
      peer_at<unsigned long long>(a.local_base, a.x_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    auto load_arrived = [&]() {
      unsigned long long total = 0ULL;
      for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(arrived[tile * a.local_world_size + rail]);
        total += ready.load(::cuda::memory_order_acquire);
      }
      return total;
    };
    auto current = load_arrived();
#if MEGAMOE_DEBUG_VALIDATE
    if (current < expected && threadIdx.x == 0U) {
      atomicAdd(&completion->tile_waiter_count, 1ULL);
      (void)atomicCAS(&completion->first_waiting_tile, ~0ULL, tile);
      if (completion->first_waiting_tile == tile) {
        completion->first_wait_expected = expected;
        completion->first_wait_arrived = current;
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
      }
    }
#endif
    while (current < expected && !forward_cancelled(a)) {
#if MEGAMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        completion->first_wait_arrived = current;
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
#endif
      current = load_arrived();
    }
    tile_row = tile_row_end;
  }
}

__device__ transport_state::TaskCompletionRecord &completion_task(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind) {
  return completion->tasks[static_cast<std::size_t>(kind)];
}

__device__ void mark_task_space_complete(
    transport_state::TaskCompletionRecord &task) {
  task.claimed = task.expected;
  task.completed = task.expected;
  task.inflight = 0ULL;
}

__device__ void claim_completion_tasks(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind, unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.claimed, count);
  atomicAdd(&task.inflight, count);
}

__device__ void complete_completion_tasks(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind, unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.completed, count);
  atomicAdd(&task.inflight, 0ULL - count);
}

__device__ bool completion_vector_ready(
    const transport_state::GenerationCompletionVector &completion) {
  for (std::size_t index = 0; index < transport_state::kTaskClassCount;
       ++index) {
    const auto task = completion.tasks[index];
    if (task.claimed != task.expected || task.completed != task.expected ||
        task.inflight != 0ULL)
      return false;
  }
  return completion.directory_ready == completion.directory_expected &&
         completion.dispatch_local_completion_completed ==
             completion.dispatch_local_completion_expected &&
         completion.return_local_completion_completed ==
             completion.return_local_completion_expected &&
         completion.ring_credit_restored == completion.ring_credit_expected;
}

__device__ void publish_elected_terminal(
    transport_state::GenerationCompletionVector *completion,
    unsigned long long epoch, bool error) {
  if (!error && !completion_vector_ready(*completion))
    return;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      published(completion->kernel_complete_epoch);
  published.store(epoch, ::cuda::memory_order_release);
  completion->terminal_publish_count += 1ULL;
}

__device__ void forward_publish_progress(const ForwardKernelArgs &a,
                                         std::uint32_t word,
                                         unsigned long long value) {
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
  if (a.progress != nullptr && blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> field(
        a.progress[word]);
    field.store(value, ::cuda::memory_order_release);
  }
#else
  (void)a;
  (void)word;
  (void)value;
#endif
}

template <bool StaticLocal>
__device__ __forceinline__ bool remote_backend_enabled(bool enabled) {
  if constexpr (StaticLocal)
    return false;
  return enabled;
}

#include "cuda/persistent_v3/forward/reduce_queue.cuh"
#include "cuda/persistent_v3/forward/return_pipeline.cuh"
#include "cuda/persistent_v3/forward/dispatch_pipeline.cuh"
#include "cuda/persistent_v3/forward/finalize_pipeline.cuh"

enum class ForwardKernelRole : std::uint8_t {
  Combined,
  TransportOnly,
  ComputeOnly,
};

#include "cuda/persistent_v3/forward/compute_kernel.cuh"

template <bool StaticLocal,
          ForwardKernelRole KernelRole = ForwardKernelRole::Combined>
__global__ __maxnreg__(160) void local_p2p_forward_megakernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  extern __shared__ __align__(1024) unsigned char storage[];
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = reinterpret_cast<transport::BFloat16 *>(
      storage + 4U * 128U * 64U * sizeof(transport::BFloat16));
#endif
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  // FC1 and FC2 reuse the large tile storage, but they must not reuse an
  // mbarrier object. Reinitializing a transaction barrier after a long TMA /
  // WGMMA phase is not a valid lifetime transition while an async-proxy
  // completion may still reference its address. Keep the tiny control objects
  // disjoint and initialize both sets once.
  auto *fc1_full_barriers = reinterpret_cast<ForwardBarrier *>(
      storage +
      (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(transport::BFloat16));
  auto *fc1_empty_barriers = fc1_full_barriers + 3;
  auto *fc2_full_barriers = fc1_empty_barriers + 3;
  auto *fc2_empty_barriers = fc2_full_barriers + 4;
#endif
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  forward_publish_progress(a, kForwardEntered, a.epoch);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const auto cluster = cg::this_cluster();
#endif
  auto *canonical =
      peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *forward_output =
      peer_at<transport::BFloat16>(a.local_base, a.forward_output_rows_offset);
  auto *headers =
      peer_at<LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  const auto forward_rdma_ctas =
      StaticLocal ? 0U
                  : (a.gin_context_count < gridDim.x ? a.gin_context_count
                                                     : gridDim.x);
  const auto configured_relay_ctas = a.forward_relay_cta_count;
  const auto relay_cta_capacity = gridDim.x - forward_rdma_ctas;
  const auto forward_relay_ctas = configured_relay_ctas < relay_cta_capacity
                                      ? configured_relay_ctas
                                      : relay_cta_capacity;
  const auto forward_relay_cta_begin = forward_rdma_ctas;
  const auto forward_compute_cta_begin =
      forward_relay_cta_begin + forward_relay_ctas;
  const bool stream_remote_dispatch =
      KernelRole == ForwardKernelRole::TransportOnly &&
      remote_backend_enabled<StaticLocal>(a.use_gin) &&
      a.dispatch.identity.num_nodes == 2U &&
      forward_rdma_ctas == persistent_v3::kRdmaCtaCount &&
      forward_relay_ctas == configured_relay_ctas &&
      forward_relay_ctas >= persistent_v3::kRdmaCtaCount &&
      forward_relay_ctas % persistent_v3::kRdmaCtaCount == 0U;
  const bool stream_local_dispatch =
      KernelRole == ForwardKernelRole::TransportOnly && StaticLocal &&
      a.forward_stream_local_dispatch &&
      gridDim.x == a.forward_relay_cta_count;
  const bool stream_remote_return_pipeline = stream_remote_dispatch;
  auto *forward_v3_control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  transport_state::GenerationCompletionVector *completion = nullptr;
  transport_state::ContextCompletionRecord *context_completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
      context_completion =
          static_cast<transport_state::ContextCompletionRecord *>(
              a.peer_bases[a.world + 1U]);
    }
  }
  constexpr std::size_t kForwardGemmSmemBytes =
      (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(transport::BFloat16) +
      16U * sizeof(std::uint64_t);
  [[maybe_unused]] auto *return_rows_by_source =
      reinterpret_cast<unsigned int *>(storage + kForwardGemmSmemBytes);
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *fc1_published = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
  auto *forward_compute_pair_mailboxes =
      peer_at<persistent_v3::ForwardComputePairMailbox>(
          a.local_base, a.forward_compute_pair_mailboxes_offset);
  auto *forward_reduce_route_count = peer_at<unsigned int>(
      a.local_base, a.forward_reduce_route_count_offset);
  auto *forward_reduce_ready_token = peer_at<unsigned long long>(
      a.local_base, a.forward_reduce_ready_token_offset);
  auto *forward_output_ready_epoch = peer_at<unsigned long long>(
      a.local_base, a.forward_output_ready_epoch_offset);
  auto *forward_return_chunks = peer_at<ForwardReturnChunkRecord>(
      a.local_base, a.forward_return_chunk_directory_offset);
  auto *forward_return_chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  auto *forward_return_chunk_ready_count = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_ready_count_offset);
  auto *forward_return_fragment_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_fragment_index_offset);
  auto *forward_return_source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_compact_index_offset);
  auto *forward_return_source_route_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_route_index_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  auto *ab_cache =
      a.retain_backward_state
          ? peer_at<transport::BFloat16>(a.local_base,
                                         a.forward_ab_cache_offset)
          : static_cast<transport::BFloat16 *>(nullptr);
  const auto *canonical_gates =
      peer_at<const float>(a.local_base, a.canonical_gates_offset);
  if constexpr (KernelRole != ForwardKernelRole::ComputeOnly) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      status->start_clock = device_globaltimer_ns();
      // Transport-only counters are unused by Forward and serve as first-start
      // clocks without growing the pooled status allocation.
      status->dispatch_blocks_done = ~0ULL; // first FC2 tile
      status->relay_blocks_done = ~0ULL;    // first returned M-block
      status->reduced_tokens = ~0ULL;       // first reduced token
      status->fc1_start_clock = ~0ULL;
      scheduler[0] = scheduler[1] = scheduler[2] = scheduler[3] = 0ULL;
      auto *head_mblocks = forward_mixed_scheduler_word(
          a, scheduler, kForwardHeadMblocks);
      *head_mblocks = 0ULL;
      *forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc1Publish) =
          0ULL;
      *forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc1Claim) = 0ULL;
      *forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc2Publish) =
          0ULL;
      *forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc2Claim) = 0ULL;
      *forward_mixed_scheduler_word(a, scheduler,
                                    kForwardTransportInitEpoch) = 0ULL;
      *forward_mixed_scheduler_word(a, scheduler,
                                    kForwardTransportClearArrivals) = 0ULL;
      *forward_mixed_scheduler_word(a, scheduler,
                                    kForwardTransportMapArrivals) = 0ULL;
      for (std::uint32_t expert = 0;
           expert < a.dispatch.sizes.owned_expert_count; ++expert) {
        scheduler[3] += forward_expert_mblock_count(a, expert);
        *head_mblocks += forward_expert_head_mblock_count(a, expert);
      }
      scheduler[4U + a.world] = 0ULL;
      scheduler[5U + 2ULL * a.world] = 0ULL;  // GIN headers ready contexts
      scheduler[6U + 2ULL * a.world] = 0ULL;  // Return directory ready epoch
      scheduler[7U + 2ULL * a.world] = 0ULL;  // Relay copy barrier arrivals
      scheduler[8U + 2ULL * a.world] = 0ULL;  // Compute setup ready epoch
      scheduler[9U + 2ULL * a.world] = 0ULL;  // Return-directory totals barrier
      scheduler[10U + 2ULL * a.world] = 0ULL; // Return-directory map barrier
      scheduler[11U + 2ULL * a.world] = 0ULL; // FC1-complete M-block count
      scheduler[12U + 2ULL * a.world] = 0ULL; // Published FC1 queue tail
      scheduler[13U + 2ULL * a.world] = 0ULL; // Completed M-block queue head
      scheduler[14U + 2ULL * a.world] = 0ULL; // Directory fragments barrier
      scheduler[15U + 2ULL * a.world] = 0ULL; // Reserved directory scratch
      auto *direct_issued =
          forward_return_detail::direct_issued_counts(a, scheduler);
      for (std::uint32_t context = 0; context < a.gin_context_count;
           ++context)
        direct_issued[context] = 0ULL;
      forward_return_detail::direct_fragment_tail(a, scheduler) = 0ULL;
      forward_return_detail::direct_mblocks_complete(a, scheduler) = 0ULL;
      if constexpr (!StaticLocal) {
        if (a.use_gin) {
          completion->generation = a.dispatch.identity.generation;
          completion->epoch = a.epoch;
#if MEGAMOE_DEBUG_VALIDATE
          completion->first_waiting_tile = ~0ULL;
#endif
          unsigned long long dispatch_waves = 0ULL;
          for (std::uint64_t wave = 0; wave < a.dispatch_waves.wave_count;
               ++wave)
            if (a.dispatch_waves.destination_node[wave] != a.node)
              ++dispatch_waves;
          const auto fc1_ntiles =
              (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
              forward_tuning::kFc1TileN;
          const auto fc2_ntiles =
              (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
              forward_tuning::kFc2TileN;
          completion_task(completion, transport_state::TaskClass::DispatchWave)
              .expected = dispatch_waves;
          unsigned long long relay_packets = 0ULL;
          for (std::uint32_t index = 0;
               index < a.dispatch.sizes.incoming_signal_count; ++index) {
            const auto signal = a.dispatch.incoming_proxy_signals[index];
            if (signal.source_node != a.node)
              relay_packets += signal.expected_packets;
          }
          completion_task(completion, transport_state::TaskClass::RelayWave)
              .expected = relay_packets;
          completion_task(completion, transport_state::TaskClass::Fc1)
              .expected = scheduler[3] * fc1_ntiles;
          completion_task(completion, transport_state::TaskClass::Fc2)
              .expected = scheduler[3] * fc2_ntiles;
          completion_task(completion, transport_state::TaskClass::SourceArrival)
              .expected = a.dispatch.sizes.route_count;
          completion_task(completion, transport_state::TaskClass::Reduce)
              .expected = a.dispatch.identity.num_tokens;
          completion_task(completion, transport_state::TaskClass::Return)
              .expected = 0ULL;
          completion->return_local_completion_expected = 0ULL;
          completion->directory_expected =
              static_cast<unsigned long long>(a.dispatch.identity.num_nodes -
                                              1U) *
              a.gin_context_count;
          completion->dispatch_local_completion_expected =
              completion->directory_expected + dispatch_waves;
        }
      }
      // This release replaces the first transport-grid barrier.  No other CTA
      // can consume scheduler bounds or phase counters until all scalar
      // generation initialization above is visible.
      if constexpr (KernelRole == ForwardKernelRole::TransportOnly) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            init_epoch(*forward_mixed_scheduler_word(
                a, scheduler, kForwardTransportInitEpoch));
        init_epoch.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    if (threadIdx.x == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
      const auto cluster_size = cluster.dim_blocks().x *
                                cluster.dim_blocks().y * cluster.dim_blocks().z;
      atomicMax(&status->runtime_cluster_size, cluster_size);
      const auto cluster_rank = cluster.block_rank();
      if (cluster_rank < 64U)
        atomicOr(&status->cluster_rank_mask, 1ULL << cluster_rank);
#else
      atomicMax(&status->runtime_cluster_size, 1U);
      atomicOr(&status->cluster_rank_mask, 1ULL);
#endif
    }
    // Block 0 publishes the generation-local scheduler bounds and resets all
    // queue cursors above.  Every transport CTA must observe those values
    // before using scheduler[3] as the extent of the parallel initialization
    // below; otherwise a recycled generation can leave stale ready-queue
    // entries when the forward M-block shape changes (notably M128 -> M64).
    // This rendezvous is confined to the fixed transport grid and precedes
    // compute setup publication, so it is not an FC1/FC2 pipeline barrier.
    if constexpr (KernelRole == ForwardKernelRole::TransportOnly)
      forward_transport_wait_epoch(
          forward_mixed_scheduler_word(a, scheduler,
                                       kForwardTransportInitEpoch),
          a.epoch);
    else
      grid.sync();
    // One completion word per destination->source rank replaces a system-scope
    // signal for every route.  The high word is immutable expected rows; return
    // CTAs add to the low word after their peer stores have been fenced.
    if (blockIdx.x == 0 && threadIdx.x < a.world) {
      unsigned long long expected = 0;
      for (std::uint32_t local_expert = 0;
           local_expert < a.dispatch.sizes.owned_expert_count; ++local_expert) {
        const auto expert = a.dispatch.owned_expert_ids[local_expert];
        expected +=
            a.dispatch
                .source_expert_count[static_cast<std::uint64_t>(threadIdx.x) *
                                         a.dispatch.identity.num_experts +
                                     expert];
      }
      scheduler[4U + threadIdx.x] = expected << 32U;
    }
    for (std::uint64_t mtile =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         mtile < scheduler[3];
         mtile += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
      fc1_done[mtile] = a.epoch << 32U;
      fc1_published[mtile] = 0ULL;
    }
    for (std::uint64_t pair =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < persistent_v3::kForwardComputePairCapacity;
         pair += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
      forward_compute_pair_mailboxes[pair].sequence = 0ULL;
      forward_compute_pair_mailboxes[pair].tasks[0] = 0ULL;
      forward_compute_pair_mailboxes[pair].tasks[1] = 0ULL;
      forward_compute_pair_mailboxes[pair].metadata = 0ULL;
    }
    // The same allocation is a generation-local row->(proxy,chunk) map for
    // remote Return. Clear only the actual canonical rows; peer proxies fill
    // remote-route entries after consuming the immutable Dispatch directory.
    for (std::uint64_t row =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         row < a.dispatch.sizes.incoming_row_count;
         row += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
      forward_output_ready_epoch[row] = 0ULL;
    if (a.forward_dynamic_reduce_queue || a.forward_dynamic_reduce_scan) {
      for (std::uint64_t token =
               static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
               threadIdx.x;
           token < a.dispatch.identity.num_tokens;
           token += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        forward_reduce_route_count[token] = 0U;
        if (a.forward_dynamic_reduce_queue)
          forward_reduce_ready_token[token] = 0ULL;
      }
    }
    if (stream_remote_dispatch || stream_local_dispatch) {
      auto *control_words =
          reinterpret_cast<unsigned long long *>(forward_v3_control);
      constexpr auto control_words_count =
          sizeof(persistent_v3::ForwardPipelineControl) /
          sizeof(unsigned long long);
      for (std::uint64_t word =
               static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
               threadIdx.x;
           word < control_words_count;
           word += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
        control_words[word] = 0ULL;
    }
    if constexpr (KernelRole == ForwardKernelRole::TransportOnly)
      forward_transport_cta_barrier(
          forward_mixed_scheduler_word(a, scheduler,
                                       kForwardTransportClearArrivals),
          gridDim.x);
    else
      grid.sync();
    forward_publish_progress(a, kForwardSetupComplete, scheduler[3]);
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    const auto setup_fc1_ntiles =
        static_cast<std::uint32_t>(
            (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
            forward_tuning::kFc1TileN);
    const auto setup_fc2_ntiles =
        static_cast<std::uint32_t>(
            (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
            forward_tuning::kFc2TileN);
    const auto setup_mblocks = scheduler[3];
    // The first words are an immutable expert/local-M lookup in Notify's M-tile
    // wavefront order.  The following regions are generation-local,
    // readiness-ordered FC1 and FC2 task queues.
    for (std::uint64_t mblock =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         mblock < setup_mblocks;
         mblock += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
      std::uint64_t local_mblock = 0;
      const auto expert =
          forward_locality_mblock_expert(a, mblock, local_mblock);
      claims[mblock] = (static_cast<unsigned long long>(expert) << 32U) |
                       static_cast<std::uint32_t>(local_mblock);
    }
    // FC1-ready tasks, then FC2 M-block descriptors and their private N-tile
    // ticket counters, live after the immutable M-block mapping.  The FC2
    // region retains fc2_ntiles words per M-block, so the two M-block-sized
    // arrays fit without changing the pooled ABI. FULL Return has a separate
    // append-only descriptor array.
    const auto setup_task_count =
        setup_mblocks * (setup_fc1_ntiles + setup_fc2_ntiles);
    for (std::uint64_t task =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < setup_task_count;
         task += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
      claims[setup_mblocks + task] = 0ULL;
    if constexpr (KernelRole == ForwardKernelRole::TransportOnly)
      forward_transport_cta_barrier(
          forward_mixed_scheduler_word(a, scheduler,
                                       kForwardTransportMapArrivals),
          gridDim.x);
    else
      grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          setup_ready(scheduler[8ULL + 2ULL * a.world]);
      setup_ready.store(a.epoch, ::cuda::memory_order_release);
    }
    if constexpr (KernelRole == ForwardKernelRole::TransportOnly)
      forward_transport_wait_epoch(
          &scheduler[8ULL + 2ULL * a.world], a.epoch);
#endif

    // Socket correctness mode has already placed every incoming activation and
    // canonical header through host-enqueued NCCL Send/Recv. It deliberately
    // bypasses the peer/G​IN dispatch phase while retaining the exact FC
    // core.
    if (!remote_backend_enabled<StaticLocal>(a.use_socket)) {
#if __has_include(<nccl_device.h>)
      if (stream_local_dispatch) {
#if defined(MEGAMOE_FORWARD_TRANSPORT_TU)
        forward_dispatch_detail::local_relay_role(
            a, *forward_v3_control, static_cast<std::uint32_t>(blockIdx.x),
            scheduler, storage);
        forward_publish_progress(a, kForwardDispatchStored,
                                 a.dispatch.sizes.route_count);
        forward_publish_progress(a, kForwardDispatchObserved,
                                 a.dispatch.sizes.incoming_row_count);
        forward_finalize_detail::local_split_transport_finalize(
            a, *forward_v3_control, scheduler, status);
#endif
      } else if (stream_remote_dispatch) {
        forward_dispatch_detail::dispatch_role(
            a, *forward_v3_control, forward_relay_cta_begin, completion,
            context_completion, scheduler, storage);
        forward_publish_progress(a, kForwardDispatchStored,
                                 a.dispatch.sizes.route_count);
        forward_publish_progress(a, kForwardDispatchObserved,
                                 a.dispatch.sizes.incoming_row_count);
      } else
#endif
      {
        // Local rows use the frozen CUDA-IPC fast path. Cross-node rows are
        // packed once per (token,node) by the same-rail proxy path below.
        unsigned long long dispatch_peer_rows = 0ULL;
#if MEGAMOE_DEVICE_TIMELINE
        bool timeline_x_local_recorded = false;
#endif
        for (std::uint64_t route = blockIdx.x;
             route < a.dispatch.sizes.route_count; route += gridDim.x) {
          const auto destination = a.dispatch.route_dst_rank[route];
          if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
              destination / a.local_world_size != a.node)
            continue;
          const auto row = a.dispatch.route_dst_row[route];
          const auto token = route / a.dispatch.identity.top_k;
          auto *destination_base = a.peer_bases[destination];
          auto *destination_rows = peer_at<transport::BFloat16>(
              destination_base, a.canonical_rows_offset);
          const auto *source = a.input + token * a.hidden;
          auto *target = destination_rows + row * a.hidden;
          const auto *source128 = reinterpret_cast<const uint4 *>(source);
          auto *target128 = reinterpret_cast<uint4 *>(target);
          for (std::uint64_t vector = threadIdx.x; vector < a.hidden / 8ULL;
               vector += blockDim.x)
            target128[vector] = source128[vector];
          if (threadIdx.x == 0) {
            auto *destination_gates = peer_at<float>(
                destination_base, a.canonical_gates_offset);
            destination_gates[row] = a.gates[route];
          }
          if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
            // Every lane's row stores precede the exact destination M-tile row
            // increment.  The destination compute worker acquires this counter;
            // no packet/wave signal is itself interpreted as tile readiness.
            __threadfence_system();
            __syncthreads();
            if (threadIdx.x == 0) {
              auto *arrived = peer_at<unsigned long long>(
                  destination_base, a.x_arrived_rows_offset);
              const auto rail = a.rank % a.local_world_size;
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  counter(arrived[a.dispatch_waves.route_dst_mtile[route] *
                                      a.local_world_size +
                                  rail]);
              counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
              atomicAdd(&completion->local_rows_published, 1ULL);
              const auto tile = a.dispatch_waves.route_dst_mtile[route];
              if (tile < 64U)
                atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
            }
            __syncthreads();
          }
#if MEGAMOE_DEVICE_TIMELINE
          __syncthreads();
          if (!timeline_x_local_recorded) {
            MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible,
                                    GinDispatch, 0U, route, a.epoch);
            timeline_x_local_recorded = true;
          }
#endif
          if (threadIdx.x == 0) {
            auto *destination_headers = peer_at<LocalP2pRouteHeader>(
                destination_base, a.direct_headers_offset);
            destination_headers[row] = LocalP2pRouteHeader{
                transport::kActivationTransportSchemaVersion,
                a.rank,
                destination,
                0U,
                a.dispatch.identity.generation,
                a.epoch,
                route,
                route,
                token,
                row,
                a.dispatch.expert_id[route],
                static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
            if (destination != a.rank)
              ++dispatch_peer_rows;
          }
        }
#if __has_include(<nccl_device.h>)
        if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
          grid.sync();
          auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
          auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
              window_bytes + a.gin_send_header_offset);
          auto *send_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
              window_bytes + a.gin_send_fanout_offset);
          auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
              window_bytes + a.gin_send_payload_offset);
          const auto vector_count = a.hidden / 8ULL;
          const auto global_thread =
              static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
          const auto global_threads =
              static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
#if MEGAMOE_DEVICE_TIMELINE
          unsigned long long dispatch_pack_thread_bytes = 0ULL;
          __shared__ unsigned long long dispatch_pack_cta_bytes;
          if (threadIdx.x == 0U)
            dispatch_pack_cta_bytes = 0ULL;
          __syncthreads();
          MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchPackStart, GinDispatch,
                                  0U, 0ULL, a.epoch);
#endif
          for (std::uint64_t item = global_thread;
               item < a.dispatch.sizes.packet_count * vector_count;
               item += global_threads) {
            const auto wave_position = item / vector_count;
            const auto vector = item % vector_count;
            const auto packet = a.dispatch_waves.packet_order[wave_position];
            const auto descriptor = a.dispatch.packet_headers[packet];
            if (descriptor.destination_node == a.node)
              continue;
            reinterpret_cast<uint4 *>(
                send_payload)[wave_position * vector_count + vector] =
                reinterpret_cast<const uint4 *>(
                    a.input)[descriptor.local_token_id * vector_count + vector];
#if MEGAMOE_DEVICE_TIMELINE
            dispatch_pack_thread_bytes += sizeof(uint4);
#endif
          }
          for (std::uint64_t packed = global_thread;
               packed <
               a.dispatch.sizes.packet_count * a.dispatch.identity.top_k;
               packed += global_threads) {
            const auto wave_position = packed / a.dispatch.identity.top_k;
            const auto packet = a.dispatch_waves.packet_order[wave_position];
            const auto descriptor = a.dispatch.packet_headers[packet];
            const auto fanout_slot = packed % a.dispatch.identity.top_k;
            const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
            const auto fanout_end =
                a.dispatch.packet_fanout_offsets[packet + 1U];
            if (fanout_begin + fanout_slot < fanout_end) {
              const auto target =
                  a.dispatch.packet_fanout[fanout_begin + fanout_slot];
              const auto route = descriptor.local_token_id *
                                     a.dispatch.identity.top_k +
                                 target.topk_slot;
              send_fanout[packed] = ForwardDispatchFanoutRecord{
                  target.dst_row,
                  target.return_compact_row,
                  target.topk_slot,
                  static_cast<std::uint16_t>(target.destination_local_rank),
                  static_cast<std::uint16_t>(target.expert_id),
                  target.reserved,
                  a.gates[route]};
            }
#if MEGAMOE_DEVICE_TIMELINE
            if (fanout_begin + fanout_slot < fanout_end)
              dispatch_pack_thread_bytes +=
                  sizeof(ForwardDispatchFanoutRecord);
#endif
          }
          for (std::uint64_t wave_position = global_thread;
               wave_position < a.dispatch.sizes.packet_count;
               wave_position += global_threads) {
            const auto packet = a.dispatch_waves.packet_order[wave_position];
            const auto descriptor = a.dispatch.packet_headers[packet];
            if (descriptor.destination_node == a.node)
              continue;
            const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
            const auto fanout_end =
                a.dispatch.packet_fanout_offsets[packet + 1U];
            send_headers[wave_position] = LocalP2pRouteHeader{
                transport::kActivationTransportSchemaVersion,
                a.rank,
                descriptor.proxy_rank,
                descriptor.channel,
                a.dispatch.identity.generation,
                a.epoch,
                wave_position,
                fanout_end - fanout_begin,
                descriptor.local_token_id,
                wave_position * a.dispatch.identity.top_k,
                0,
                0};
#if MEGAMOE_DEVICE_TIMELINE
            dispatch_pack_thread_bytes += sizeof(LocalP2pRouteHeader);
#endif
          }
#if MEGAMOE_DEVICE_TIMELINE
          atomicAdd(&dispatch_pack_cta_bytes, dispatch_pack_thread_bytes);
          __syncthreads();
          MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchPackCtaBytes, GinDispatch,
                                  0U, dispatch_pack_cta_bytes, a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchPackEnd, GinDispatch, 0U,
                                  dispatch_pack_cta_bytes, a.epoch);
#endif
          // GIN's device DMA reads this window after the grid-wide producers
          // have populated it. A cooperative grid barrier orders CUDA threads
          // but does not by itself publish their stores to a system-scope
          // device backend.
          MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchPacketReady, GinDispatch,
                                  blockIdx.x < a.gin_context_count ? blockIdx.x
                                                                   : 0U,
                                  a.dispatch.sizes.packet_count, a.epoch);
          __threadfence_system();
          grid.sync();
        }
        if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
            blockIdx.x < a.gin_context_count) {
          const auto context = static_cast<std::uint32_t>(blockIdx.x);
          const ncclGin gin(a.gin_dev_comm, context);
          const auto rail = ncclTeamRail(a.gin_dev_comm);
          auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
          auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
              window_bytes + a.gin_send_dispatch_directory_offset);
          auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
              window_bytes + a.gin_send_header_offset);
#if MEGAMOE_DEBUG_VALIDATE
          auto *send_fanout =
              reinterpret_cast<ForwardDispatchFanoutRecord *>(
              window_bytes + a.gin_send_fanout_offset);
#endif
          // The owner-side initialization kernel only rendezvous with CUDA-IPC
          // peers on this node.  A remote same-rail producer can otherwise
          // enter this kernel first, publish an exact M-tile row increment, and
          // have that increment erased when the slower destination initializes
          // its local x_arrived_rows storage.  Signal 2 is reserved for a
          // fixed-width, per-context generation-ready handshake.  It runs once
          // before any Dispatch payload put and therefore adds no per-wave or
          // per-task work.
          //
          // Reuse each context's immutable directory slot as the eight-byte DMA
          // source/destination: the real directory record is written only after
          // the handshake completes, and Dispatch signal 0 orders that later
          // record.
          if (threadIdx.x == 0U) {
            unsigned long long setup_peers = 0ULL;
            for (std::uint32_t remote_node = 0;
                 remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
              if (remote_node == a.node)
                continue;
              const auto remote_partition =
                  remote_node < a.node ? remote_node : remote_node - 1U;
              const auto source_partition =
                  a.node < remote_node ? a.node : a.node - 1U;
              const auto directory_index =
                  remote_partition * a.gin_context_count + context;
              const auto remote_directory_index =
                  source_partition * a.gin_context_count + context;
              send_directory[directory_index] = GinDispatchContextRecord{
                  a.dispatch.identity.generation, a.epoch, 0ULL, 0U, 0U};
              __threadfence_system();
              gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                      a.gin_receive_dispatch_directory_offset +
                          remote_directory_index *
                              sizeof(GinDispatchContextRecord),
                      a.gin_window,
                      a.gin_send_dispatch_directory_offset +
                          directory_index * sizeof(GinDispatchContextRecord),
                      sizeof(GinDispatchContextRecord), ncclGin_SignalInc{2},
                      ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                      ::cuda::thread_scope_block, ::cuda::thread_scope_device);
              ++setup_peers;
            }
            if (setup_peers != 0ULL) {
              gin.waitCounter(ncclCoopThread{}, 0, setup_peers);
              gin.resetCounter(0);
              gin.waitSignal(ncclCoopThread{}, 2, setup_peers);
            }
          }
          __syncthreads();
          unsigned long long issued_completions = 0ULL;
          unsigned long long issued_packets = 0ULL;
          for (std::uint32_t remote_node = 0;
               remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
            if (remote_node == a.node)
              continue;
            std::uint64_t node_wave_begin = 0;
            while (node_wave_begin < a.dispatch_waves.wave_count &&
                   a.dispatch_waves.destination_node[node_wave_begin] <
                       remote_node)
              ++node_wave_begin;
            auto node_wave_end = node_wave_begin;
            while (node_wave_end < a.dispatch_waves.wave_count &&
                   a.dispatch_waves.destination_node[node_wave_end] ==
                       remote_node)
              ++node_wave_end;
            const auto node_waves = node_wave_end - node_wave_begin;
            const auto context_wave_begin =
                node_wave_begin + node_waves * context / a.gin_context_count;
            const auto context_wave_end =
                node_wave_begin +
                node_waves * (context + 1U) / a.gin_context_count;
            const auto node_packet_begin =
                a.dispatch_waves.offsets[node_wave_begin];
            const auto packet_begin =
                a.dispatch_waves.offsets[context_wave_begin] -
                node_packet_begin;
            const auto packet_end =
                a.dispatch_waves.offsets[context_wave_end] - node_packet_begin;
            const auto remote_partition =
                remote_node < a.node ? remote_node : remote_node - 1U;
            const auto source_partition =
                a.node < remote_node ? a.node : a.node - 1U;
            const auto directory_index =
                remote_partition * a.gin_context_count + context;
            if (threadIdx.x == 0U) {
              send_directory[directory_index] = GinDispatchContextRecord{
                  a.dispatch.identity.generation, a.epoch, packet_begin,
                  static_cast<std::uint32_t>(packet_end - packet_begin),
                  static_cast<std::uint32_t>(context_wave_end -
                                             context_wave_begin)};
              __threadfence_system();
              const auto remote_directory_index =
                  source_partition * a.gin_context_count + context;
              gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                      a.gin_receive_dispatch_directory_offset +
                          remote_directory_index *
                              sizeof(GinDispatchContextRecord),
                      a.gin_window,
                      a.gin_send_dispatch_directory_offset +
                          directory_index * sizeof(GinDispatchContextRecord),
                      sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
                      ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                      ::cuda::thread_scope_block, ::cuda::thread_scope_device);
              ++issued_completions;
            }
            __syncthreads();
            // Every context publishes exactly one fixed-width directory record
            // for every remote node that participates in this generation,
            // including an empty record.  Receivers therefore never have to
            // infer which sender contexts own waves before waiting for the
            // directory phase.
            for (auto wave = context_wave_begin; wave < context_wave_end;
                 ++wave) {
              const auto wave_begin = a.dispatch_waves.offsets[wave];
              const auto wave_end = a.dispatch_waves.offsets[wave + 1U];
              const auto wire_begin = wave_begin - node_packet_begin;
              const auto wave_packets = wave_end - wave_begin;
              for (auto position = wave_begin + threadIdx.x;
                   position < wave_end; position += blockDim.x) {
                send_headers[position].sequence = position - node_packet_begin;
                send_headers[position].dst_row =
                    (position - node_packet_begin) * a.dispatch.identity.top_k;
                send_headers[position].topk_slot = 0U;
              }
              __syncthreads();
              if (threadIdx.x == 0U) {
                send_headers[wave_begin].topk_slot =
                    static_cast<std::uint32_t>(wave_packets);
                __threadfence_system();
                const auto receive_packet =
                    source_partition * a.max_tokens_per_rank + wire_begin;
                MEGAMOE_TIMELINE_RECORD(
                    a.timeline, GinPutIssue, GinDispatch, context,
                    wave_packets * a.hidden * sizeof(transport::BFloat16),
                    a.epoch);
                gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                        a.gin_receive_header_offset +
                            receive_packet * sizeof(LocalP2pRouteHeader),
                        a.gin_window,
                        a.gin_send_header_offset +
                            wave_begin * sizeof(LocalP2pRouteHeader),
                        wave_packets * sizeof(LocalP2pRouteHeader));
                gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                        a.gin_receive_fanout_offset +
                            (source_partition * a.max_tokens_per_rank +
                             wire_begin) *
                                a.dispatch.identity.top_k *
                                sizeof(ForwardDispatchFanoutRecord),
                        a.gin_window,
                        a.gin_send_fanout_offset +
                            wave_begin * a.dispatch.identity.top_k *
                                sizeof(ForwardDispatchFanoutRecord),
                        wave_packets * a.dispatch.identity.top_k *
                            sizeof(ForwardDispatchFanoutRecord));
                gin.put(
                    rail, static_cast<int>(remote_node), a.gin_window,
                    a.gin_receive_payload_offset +
                        receive_packet * a.hidden * sizeof(transport::BFloat16),
                    a.gin_window,
                    a.gin_send_payload_offset +
                        wave_begin * a.hidden * sizeof(transport::BFloat16),
                    wave_packets * a.hidden * sizeof(transport::BFloat16),
                    ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                    ncclCoopThread{}, ncclGin_None{},
                    ::cuda::thread_scope_block, ::cuda::thread_scope_device);
                ++issued_completions;
                issued_packets += wave_packets;
                atomicAdd(&status->gin_dispatch_put_count, 3ULL);
                atomicAdd(&status->gin_signal_count, 1ULL);
                atomicAdd(&status->gin_metadata_bytes,
                          wave_packets * (sizeof(LocalP2pRouteHeader) +
                                          a.dispatch.identity.top_k *
                                              sizeof(
                                                  ForwardDispatchFanoutRecord)));
              }
              __syncthreads();
            }
          }
          if (threadIdx.x == 0U && issued_completions != 0ULL) {
            gin.waitCounter(ncclCoopThread{}, 0, issued_completions);
            MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion,
                                    GinDispatch, context, issued_packets,
                                    a.epoch);
            atomicAdd(&status->deduplicated_cross_node_payload_bytes,
                      issued_packets * a.hidden * sizeof(transport::BFloat16));
            atomicAdd(&status->actual_dispatch_peer_bytes,
                      issued_packets * a.hidden * sizeof(transport::BFloat16));
            gin.resetCounter(0);
            for (std::uint32_t remote_node = 0;
                 remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
              if (remote_node == a.node)
                continue;
              const auto remote_partition =
                  remote_node < a.node ? remote_node : remote_node - 1U;
              const auto record_index =
                  remote_partition * a.gin_context_count + context;
              const auto directory = send_directory[record_index];
              auto &record = context_completion[record_index];
              record.directory_ready = 1ULL;
              record.dispatch_expected = directory.wave_count;
              record.dispatch_claimed = directory.wave_count;
              record.dispatch_completed = directory.wave_count;
              record.local_completion = 1ULL + directory.wave_count;
              record.credit = 1ULL;
              record.last_cursor =
                  directory.packet_begin + directory.packet_count;
              record.last_progress_globaltimer = device_globaltimer_ns();
#if MEGAMOE_DEBUG_VALIDATE
              std::uint64_t first_node_wave = 0;
              while (first_node_wave < a.dispatch_waves.wave_count &&
                     a.dispatch_waves.destination_node[first_node_wave] <
                         remote_node)
                ++first_node_wave;
              const auto node_packet_begin =
                  a.dispatch_waves.offsets[first_node_wave];
              for (std::uint64_t position = directory.packet_begin;
                   position < directory.packet_begin + directory.packet_count;
                   ++position) {
                const auto header = send_headers[node_packet_begin + position];
                record.dispatch_fanout_rows += header.route_index;
                for (std::uint64_t fanout = 0; fanout < header.route_index;
                     ++fanout)
                  record.dispatch_tile0_rows +=
                      send_fanout[(node_packet_begin + position) *
                                      a.dispatch.identity.top_k +
                                  fanout]
                          .mtile == 0U;
              }
#endif
              atomicAdd(&completion->directory_ready, 1ULL);
              atomicAdd(&completion->dispatch_local_completion_completed,
                        1ULL + directory.wave_count);
              if (directory.wave_count != 0ULL) {
                claim_completion_tasks(completion,
                                       transport_state::TaskClass::DispatchWave,
                                       directory.wave_count);
                complete_completion_tasks(
                    completion, transport_state::TaskClass::DispatchWave,
                    directory.wave_count);
              }
            }
          }
          __syncthreads();
          unsigned long long incoming_sources = 0ULL;
          if (threadIdx.x == 0U) {
            for (std::uint32_t index = 0;
                 index < a.dispatch.sizes.incoming_signal_count; ++index) {
              const auto signal = a.dispatch.incoming_proxy_signals[index];
              if (signal.source_node != a.node &&
                  signal.expected_packets != 0ULL)
                ++incoming_sources;
            }
            if (incoming_sources != 0ULL)
              gin.waitSignal(ncclCoopThread{}, 0, incoming_sources);
            auto *receive_directory =
                reinterpret_cast<GinDispatchContextRecord *>(
                    static_cast<std::byte *>(a.gin_window_base) +
                    a.gin_receive_dispatch_directory_offset);
            unsigned long long incoming_waves = 0ULL;
            for (std::uint32_t index = 0;
                 index < a.dispatch.sizes.incoming_signal_count; ++index) {
              const auto signal = a.dispatch.incoming_proxy_signals[index];
              if (signal.source_node == a.node ||
                  signal.expected_packets == 0ULL)
                continue;
              const auto source_partition = signal.source_node < a.node
                                                ? signal.source_node
                                                : signal.source_node - 1U;
              const auto directory =
                  receive_directory[source_partition * a.gin_context_count +
                                    context];
              if (directory.generation == a.dispatch.identity.generation &&
                  directory.epoch == a.epoch) {
                incoming_waves += directory.wave_count;
                const auto record_index =
                    source_partition * a.gin_context_count + context;
                context_completion[record_index].relay_expected =
                    directory.packet_count;
              }
            }
            if (incoming_waves != 0ULL)
              gin.waitSignal(ncclCoopThread{}, 0,
                             incoming_sources + incoming_waves);
            MEGAMOE_TIMELINE_RECORD(a.timeline, RemoteSignalObserved,
                                    GinDispatch, context, incoming_sources,
                                    a.epoch);
            MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstRemoteSignal, GinDispatch,
                                    context, incoming_sources, a.epoch);
            // The GIN signal orders the network operation, but the next
            // consumer is now a different CTA/SM.  Bridge async-proxy writes
            // into the generic memory domain before publishing the CUDA atomic
            // hand-off.
            asm volatile("fence.proxy.async.global;" : : : "memory");
            __threadfence();
            atomicAdd(&scheduler[5U + 2ULL * a.world], 1ULL);
          }
          __syncthreads();

          // All GIN contexts publish an immutable receive directory before
          // Relay. Packet waves are divided between contexts by wave count,
          // which can be very uneven in bytes/fan-out.  Relaying only the
          // directory owned by the matching context left one CTA copying nearly
          // the entire remote payload while the other progress CTAs were idle.
          // Wait for the fixed directory set, then stripe every directory
          // across the fixed context CTA team.  No task space is extended here
          // and the exact destination-row counters stay the sole FC readiness
          // condition.
          if (threadIdx.x == 0U) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                directories_ready(scheduler[5U + 2ULL * a.world]);
            while (directories_ready.load(::cuda::memory_order_acquire) <
                       a.gin_context_count &&
                   !forward_cancelled(a)) {
              __nanosleep(256U);
            }
          }
          __syncthreads();
        }
        if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
            blockIdx.x >= forward_relay_cta_begin &&
            blockIdx.x < forward_compute_cta_begin) {
          // Relay work is row-owned by a static transport team. Wait until the
          // fixed context CTAs have observed every immutable directory before
          // dereference.
          if (threadIdx.x == 0U) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                directories_ready(scheduler[5U + 2ULL * a.world]);
            while (directories_ready.load(::cuda::memory_order_acquire) <
                       a.gin_context_count &&
                   !forward_cancelled(a)) {
              __nanosleep(256U);
            }
          }
          __syncthreads();
          // Acquire the cross-CTA hand-off before this Relay team dereferences
          // the registered receive window.  Keeping this fence at the consumer
          // as well makes the proxy-domain transition explicit for every Relay
          // SM.
          asm volatile("fence.proxy.async.global;" : : : "memory");
          __syncthreads();
          auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
          auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
              window_bytes + a.gin_receive_header_offset);
          auto *receive_fanout =
              reinterpret_cast<ForwardDispatchFanoutRecord *>(
              window_bytes + a.gin_receive_fanout_offset);
          auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
              window_bytes + a.gin_receive_payload_offset);
          auto *receive_directory =
              reinterpret_cast<GinDispatchContextRecord *>(
                  window_bytes + a.gin_receive_dispatch_directory_offset);
          MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, 0U,
                                  a.dispatch.sizes.incoming_signal_count,
                                  a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstProxyRelay, ProxyRelay, 0U,
                                  a.dispatch.sizes.incoming_signal_count,
                                  a.epoch);
          const auto vector_count = a.hidden / 8ULL;
          const auto relay_cta =
              static_cast<std::uint64_t>(blockIdx.x - forward_relay_cta_begin);
          const auto relay_thread = relay_cta * blockDim.x + threadIdx.x;
          const auto relay_threads =
              static_cast<std::uint64_t>(forward_relay_ctas) * blockDim.x;
          // Stable control path: stripe the complete immutable receive set
          // across the fixed Relay team, join the team once, then publish row
          // readiness. This deliberately gives up chunk-level FC1 overlap while
          // the dedicated Relay queue protocol is being rebuilt, but it
          // preserves broad vector transactions and makes
          // payload-before-counter ordering exact.
          for (std::uint32_t signal_index = 0;
               signal_index < a.dispatch.sizes.incoming_signal_count;
               ++signal_index) {
            const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
            if (signal.source_node == a.node)
              continue;
            const auto source_partition = signal.source_node < a.node
                                              ? signal.source_node
                                              : signal.source_node - 1U;
            const auto packet_vectors =
                a.dispatch.identity.top_k * vector_count;
            for (std::uint32_t directory_context = 0;
                 directory_context < a.gin_context_count; ++directory_context) {
              const auto directory =
                  receive_directory[source_partition * a.gin_context_count +
                                    directory_context];
              for (std::uint64_t item = relay_thread;
                   item < static_cast<std::uint64_t>(directory.packet_count) *
                              packet_vectors;
                   item += relay_threads) {
                const auto sequence =
                    directory.packet_begin + item / packet_vectors;
                const auto within_packet = item % packet_vectors;
                const auto fanout = within_packet / vector_count;
                const auto vector = within_packet % vector_count;
                const auto packet =
                    source_partition * a.max_tokens_per_rank + sequence;
                const auto header = receive_headers[packet];
                if (fanout >= header.route_index)
                  continue;
                const auto target =
                    receive_fanout[source_partition * a.max_tokens_per_rank *
                                       a.dispatch.identity.top_k +
                                   header.dst_row + fanout];
                const auto destination =
                    a.node * a.local_world_size + target.destination_local_rank;
                auto *destination_rows = peer_at<transport::BFloat16>(
                    a.peer_bases[destination], a.canonical_rows_offset);
                reinterpret_cast<uint4 *>(
                    destination_rows)[target.dst_row * vector_count + vector] =
                    reinterpret_cast<const uint4 *>(
                        receive_payload)[packet * vector_count + vector];
              }
              for (std::uint64_t slot = relay_thread;
                   slot < static_cast<std::uint64_t>(directory.packet_count) *
                              a.dispatch.identity.top_k;
                   slot += relay_threads) {
                const auto sequence =
                    directory.packet_begin + slot / a.dispatch.identity.top_k;
                const auto fanout = slot % a.dispatch.identity.top_k;
                const auto packet =
                    source_partition * a.max_tokens_per_rank + sequence;
                const auto header = receive_headers[packet];
                if (fanout >= header.route_index)
                  continue;
                const auto target =
                    receive_fanout[source_partition * a.max_tokens_per_rank *
                                       a.dispatch.identity.top_k +
                                   header.dst_row + fanout];
                const auto destination =
                    a.node * a.local_world_size + target.destination_local_rank;
                const auto gate = target.gate;
                auto *destination_headers = peer_at<LocalP2pRouteHeader>(
                    a.peer_bases[destination], a.direct_headers_offset);
                const auto route =
                    header.local_token_id * a.dispatch.identity.top_k +
                    target.topk_slot;
                destination_headers[target.dst_row] = LocalP2pRouteHeader{
                    transport::kActivationTransportSchemaVersion,
                    header.source_rank,
                    destination,
                    header.channel,
                    header.generation,
                    header.source_epoch,
                    header.sequence,
                    route,
                    header.local_token_id,
                    target.dst_row,
                    target.expert_id,
                    target.topk_slot};
                auto *destination_gates = peer_at<float>(
                    a.peer_bases[destination], a.canonical_gates_offset);
                destination_gates[target.dst_row] = gate;
              }
            }
          }
          // This is the only inter-CTA rendezvous in the temporary stable
          // relay. Every Relay CTA has finished all of its vector/header
          // stripes before any CTA is allowed to release a destination-row
          // counter.
          __threadfence_system();
          asm volatile("fence.proxy.async.global;" : : : "memory");
          __syncthreads();
          if (threadIdx.x == 0U) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
                copy_complete(scheduler[7ULL + 2ULL * a.world]);
            copy_complete.fetch_add(1ULL, ::cuda::memory_order_release);
            while (copy_complete.load(::cuda::memory_order_acquire) <
                       forward_relay_ctas &&
                   !forward_cancelled(a))
              __nanosleep(256U);
          }
          __syncthreads();
          for (std::uint32_t signal_index = 0;
               signal_index < a.dispatch.sizes.incoming_signal_count;
               ++signal_index) {
            const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
            if (signal.source_node == a.node)
              continue;
            const auto source_partition = signal.source_node < a.node
                                              ? signal.source_node
                                              : signal.source_node - 1U;
            for (std::uint32_t directory_context = 0;
                 directory_context < a.gin_context_count; ++directory_context) {
              const auto directory =
                  receive_directory[source_partition * a.gin_context_count +
                                    directory_context];
              for (std::uint64_t slot = relay_thread;
                   slot < static_cast<std::uint64_t>(directory.packet_count) *
                              a.dispatch.identity.top_k;
                   slot += relay_threads) {
                const auto sequence =
                    directory.packet_begin + slot / a.dispatch.identity.top_k;
                const auto fanout = slot % a.dispatch.identity.top_k;
                const auto packet =
                    source_partition * a.max_tokens_per_rank + sequence;
                const auto header = receive_headers[packet];
                if (fanout >= header.route_index)
                  continue;
                const auto target =
                    receive_fanout[source_partition * a.max_tokens_per_rank *
                                       a.dispatch.identity.top_k +
                                   header.dst_row + fanout];
                const auto destination =
                    a.node * a.local_world_size + target.destination_local_rank;
                auto *arrived = peer_at<unsigned long long>(
                    a.peer_bases[destination], a.x_arrived_rows_offset);
                const auto rail = a.rank % a.local_world_size;
                ::cuda::atomic_ref<unsigned long long,
                                   ::cuda::thread_scope_system>
                    counter(
                        arrived[target.mtile * a.local_world_size + rail]);
                counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
                const auto record_index =
                    source_partition * a.gin_context_count + directory_context;
                atomicAdd(&context_completion[record_index].relay_fanout_rows,
                          1ULL);
                if (target.mtile == 0U)
                  atomicAdd(&context_completion[record_index].relay_tile0_rows,
                            1ULL);
                atomicAdd(&completion->remote_rows_published, 1ULL);
                if (target.mtile < 64U)
                  atomicOr(&completion->published_tile_mask,
                           1ULL << target.mtile);
#endif
              }
            }
          }
          __syncthreads();
          if (relay_cta < a.gin_context_count && threadIdx.x == 0U) {
            const auto context = static_cast<std::uint32_t>(relay_cta);
            unsigned long long relayed = 0ULL;
            for (std::uint32_t remote = 0;
                 remote < a.dispatch.identity.num_nodes - 1U; ++remote) {
              const auto index = remote * a.gin_context_count + context;
              auto &record = context_completion[index];
              record.relay_claimed = record.relay_expected;
              record.relay_completed = record.relay_expected;
              record.last_progress_globaltimer = device_globaltimer_ns();
              relayed += record.relay_expected;
            }
            if (relayed != 0ULL) {
              claim_completion_tasks(
                  completion, transport_state::TaskClass::RelayWave, relayed);
              complete_completion_tasks(
                  completion, transport_state::TaskClass::RelayWave, relayed);
            }
          }
          MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, 0U,
                                  a.dispatch.sizes.incoming_row_count, a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, 0U,
                                  a.dispatch.sizes.incoming_row_count, a.epoch);
        }
#endif
        if (threadIdx.x == 0 && dispatch_peer_rows != 0ULL)
          atomicAdd(&status->actual_dispatch_peer_bytes,
                    dispatch_peer_rows * a.hidden *
                        sizeof(transport::BFloat16));
        // Every participating lane orders its peer payload stores before the
        // team rendezvous. A fence issued only by the signal lane would not
        // cover stores performed by the other lanes in the CTA.
        __threadfence_system();
        forward_publish_progress(a, kForwardDispatchStored,
                                 a.dispatch.sizes.route_count);
        if (!remote_backend_enabled<StaticLocal>(a.use_gin)) {
          grid.sync();
          if (blockIdx.x == 0) {
            const auto first_local =
                remote_backend_enabled<StaticLocal>(a.use_gin)
                    ? a.node * a.local_world_size
                    : 0U;
            const auto local_count =
                remote_backend_enabled<StaticLocal>(a.use_gin)
                    ? a.local_world_size
                    : a.world;
            if (threadIdx.x < local_count) {
              const auto peer = first_local + threadIdx.x;
              auto *signal = peer_at<unsigned long long>(
                  a.peer_bases[peer], a.ingress_published_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  published(signal[a.rank * a.channels]);
              published.store(a.epoch, ::cuda::memory_order_release);
            }
            for (std::uint32_t local_source = threadIdx.x;
                 local_source < local_count; local_source += blockDim.x) {
              const auto source = first_local + local_source;
              auto *signals = peer_at<unsigned long long>(
                  a.local_base, a.ingress_published_offset);
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  published(signals[source * a.channels]);
              while (published.load(::cuda::memory_order_acquire) < a.epoch &&
                     !forward_cancelled(a)) {
              }
            }
          }
          grid.sync();
        }
        forward_publish_progress(a, kForwardDispatchObserved,
                                 remote_backend_enabled<StaticLocal>(a.use_gin)
                                     ? a.local_world_size
                                     : a.world);
      }
    } else {
      grid.sync();
      forward_publish_progress(a, kForwardDispatchStored,
                               a.dispatch.sizes.route_count);
      forward_publish_progress(a, kForwardDispatchObserved,
                               a.dispatch.sizes.incoming_row_count);
    }
    if (!stream_remote_dispatch && blockIdx.x == 0 && threadIdx.x == 0) {
      status->dispatch_complete_clock = device_globaltimer_ns();
      status->fc1_start_clock = status->dispatch_complete_clock;
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, XTileReady, Compute, 0U,
                            a.dispatch.sizes.mtile_count, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, XDispatchComplete, Compute, 0U,
                            a.dispatch.sizes.mtile_count, a.epoch);

    if constexpr (KernelRole == ForwardKernelRole::TransportOnly) {
      if (stream_local_dispatch)
        return;
      if (!stream_remote_return_pipeline) {
        if (blockIdx.x == 0U && threadIdx.x == 0U)
          publish_local_structured_error(status, a.rank, 133U, 1U, 1ULL, 0ULL);
        return;
      }
      forward_finalize_detail::split_transport_finalize<StaticLocal>(
          a, *forward_v3_control, completion, scheduler, status);
      return;
    }

  } else {
    // The transport grid owns all generation initialization.  Compute can be
    // launched immediately on its independent stream and waits only for this
    // release publication; Dispatch and FC1 then overlap through exact
    // per-M-block row counters.
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          setup_ready(scheduler[8ULL + 2ULL * a.world]);
      while (setup_ready.load(::cuda::memory_order_acquire) < a.epoch &&
             !forward_cancelled(a))
        __nanosleep(256U);
    }
    __syncthreads();
  }

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  const auto forward_mblocks = scheduler[3];
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_ntiles =
      static_cast<std::uint32_t>(
          (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
          forward_tuning::kFc2TileN);
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  const bool rdma_worker_cta = KernelRole != ForwardKernelRole::ComputeOnly &&
                               remote_backend_enabled<StaticLocal>(a.use_gin) &&
                               blockIdx.x < forward_rdma_ctas;
  [[maybe_unused]] const bool relay_worker_cta =
      KernelRole != ForwardKernelRole::ComputeOnly &&
      remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x >= forward_relay_cta_begin &&
      blockIdx.x < forward_compute_cta_begin;
  // Keep transport and WGMMA roles disjoint for the complete persistent
  // interval.  In particular, a Relay CTA must not execute generic/proxy
  // traffic and then enter the dynamic WGMMA register-allocation protocol.
  const bool transport_worker_cta = rdma_worker_cta || relay_worker_cta;

  // Roles are immutable for the Dispatch/FC/streaming-Return interval:
  // [0,8) owns GIN contexts, [8,16) owns NVLink Relay/directory work, and the
  // remaining CTAs own WGMMA.  In particular no Relay CTA enters the GEMM
  // path and no RDMA CTA copies the Dispatch payload into expert arenas.
  if constexpr (KernelRole != ForwardKernelRole::TransportOnly) {
    if (!transport_worker_cta) {
      if (threadIdx.x == 256U) {
        for (std::uint32_t stage = 0; stage < 3U; ++stage) {
          fc1_full_barriers[stage].init(1);
          fc1_empty_barriers[stage].init(8);
        }
        for (std::uint32_t stage = 0; stage < 4U; ++stage) {
          fc2_full_barriers[stage].init(1);
          fc2_empty_barriers[stage].init(8);
        }
        cutlass::arch::fence_barrier_init();
      }
      __syncthreads();
      // Do not emit SETMAXNREG in the monolithic compatibility kernel. Its
      // static allocation is already 168 registers/thread (64,512
      // registers/CTA), and ptxas cannot lower the transport-capable producer
      // warpgroup to 48. Keeping the ignored alloc/dealloc pair still
      // introduces warpgroup dependency edges and can strand the TMA producer
      // behind math warpgroups waiting on its barrier. The split compute kernel
      // restores the intended 224/48 redistribution once its producer path has
      // a genuinely small live set.
      std::uint32_t pipeline_iteration = 0;
      __shared__ unsigned long long task_shared;
      __shared__ unsigned int claim_shared;
      auto *fc1_claims = claims + forward_mblocks;
      auto *fc2_claims = fc1_claims + fc1_tasks;
      constexpr unsigned int kTaskChunk = MEGAMOE_DEBUG_VALIDATE ? 1U : 4U;
      unsigned int task_in_chunk = kTaskChunk;
#if MEGAMOE_DEVICE_TIMELINE
      bool timeline_fc1_started = false;
#endif
#if MEGAMOE_DEBUG_VALIDATE
      if (completion != nullptr && threadIdx.x == 0U) {
        completion->fc1_scheduler_value =
            a.cancel_epoch
                ? atomicAdd_system(
                      const_cast<unsigned long long *>(a.cancel_epoch), 0ULL)
                : ~0ULL;
        completion->bulk_local_ready = 10ULL;
      }
      __syncthreads();
#endif
      while (!forward_cancelled(a)) {
        if (task_in_chunk == kTaskChunk) {
          if (threadIdx.x == 0U)
            task_shared = atomicAdd(
                &scheduler[0], static_cast<unsigned long long>(kTaskChunk));
          __syncthreads();
          task_in_chunk = 0U;
        }
        const auto task = task_shared + task_in_chunk++;
        if (task >= fc1_tasks)
          break;
        const auto mblock = task / fc1_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        if (threadIdx.x == 0U) {
          claim_shared =
              atomicCAS(&fc1_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
          if (completion != nullptr) {
            completion->fc1_scheduler_value = task;
            completion->bulk_local_ready = 11ULL;
          }
#endif
        }
        __syncthreads();
        if (claim_shared == 0U)
          continue;
        if (expert >= a.dispatch.sizes.owned_expert_count)
          break;
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            claim_completion_tasks(completion,
                                   transport_state::TaskClass::Fc1);
        const auto row =
            a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
        const auto remaining_rows =
            a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            remaining_rows < 128ULL ? remaining_rows : 128ULL);
        wait_forward_mblock_rows<StaticLocal>(
            a, expert, static_cast<std::uint64_t>(local_mblock) * 128ULL,
            valid_rows, completion);
        __syncthreads();
#if MEGAMOE_DEVICE_TIMELINE
        if (!timeline_fc1_started) {
          MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstMtileReady, Compute, 0U,
                                  task, a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1First, Compute, 0U, task,
                                  a.epoch);
          MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1ClaimStart, Compute, 0U, task,
                                  a.epoch);
          timeline_fc1_started = true;
        }
#endif
        const auto n = n_tile * 128U;
        {
          float accum_a[ForwardMma::kNumAccum];
          float accum_b[ForwardMma::kNumAccum];
          forward_wgmma_swiglu_tile(
              x_map, w1_map, static_cast<std::uint32_t>(row),
              static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
              static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                         a.intermediate + n),
              static_cast<std::uint32_t>(a.hidden), smem_a, smem_b,
              fc1_full_barriers, fc1_empty_barriers, accum_a, accum_b,
              pipeline_iteration);
          visit_forward_accumulator_pair(
              accum_a, accum_b,
              [&](std::uint32_t warpgroup, std::uint32_t m,
                  std::uint32_t column, float av0, float av1, float bv0,
                  float bv1) {
                const auto block_m = warpgroup * 64U + m;
                const auto output_column =
                    static_cast<std::uint64_t>(n) + column;
                if (block_m < valid_rows &&
                    output_column + 1ULL < a.intermediate &&
                    (a.intermediate & 1ULL) == 0ULL) {
                  const auto output_index =
                      (row + block_m) * a.intermediate + output_column;
                  const auto gate = canonical_gates[row + block_m];
                  // Each WGMMA lane owns two adjacent BF16 outputs.  Publish
                  // the pair as one naturally aligned store for gated U.
                  const auto a_pair = float2_to_bf16x2_bits(av0, av1);
                  const auto b_pair = float2_to_bf16x2_bits(bv0, bv1);
                  const auto a_values = bf16x2_bits_to_float2(a_pair);
                  const auto b_values = bf16x2_bits_to_float2(b_pair);
                  if (ab_cache != nullptr) {
                    reinterpret_cast<std::uint32_t *>(
                        ab_cache)[output_index / 2ULL] = a_pair;
                    const auto b_index =
                        a.max_destination_rows * a.intermediate +
                        output_index;
                    reinterpret_cast<std::uint32_t *>(
                        ab_cache)[b_index / 2ULL] = b_pair;
                  }
                  reinterpret_cast<std::uint32_t *>(s)[output_index / 2ULL] =
                      float2_to_bf16x2_bits(
                          forward_compute_detail::
                              gated_swiglu_fc2_input(a_values.x, b_values.x,
                                                    gate),
                          forward_compute_detail::
                              gated_swiglu_fc2_input(a_values.y, b_values.y,
                                                    gate));
                } else if (block_m < valid_rows) {
                  // Retain checked scalar stores for non-main odd-N fixtures.
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
                      if (ab_cache != nullptr) {
                        ab_cache[output_index] =
                            transport::BFloat16{a_bits};
                        ab_cache[a.max_destination_rows * a.intermediate +
                                 output_index] =
                            transport::BFloat16{b_bits};
                      }
                      s[output_index] = transport::BFloat16{float_to_bf16_bits(
                          forward_compute_detail::
                              gated_swiglu_fc2_input_from_bf16(
                                  a_bits, b_bits, gate))};
                    }
                  }
                }
              });
        }
        // FC2 may be claimed by another CTA as soon as this tile increments the
        // m-block counter. Order every lane's S/checkpoint stores first; a
        // fence from only the counter lane would not publish the other lanes'
        // writes.
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              ready(fc1_done[mblock]);
          ready.fetch_add(1ULL, ::cuda::memory_order_release);
        }
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc1);
        if (threadIdx.x == 0U) {
          atomicExch(&fc1_claims[task], 2ULL);
        }
        __syncthreads();
      }
      if (threadIdx.x == 0)
        atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
      MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1Last, Compute, 0U, fc1_tasks,
                              a.epoch);
      forward_publish_progress(a, kForwardFc1Complete, fc1_tasks);

      // FC1 transactions issued by this CTA are complete. FC2 consumes its
      // separately initialized barrier set, so no transaction-barrier object is
      // reinitialized in place. There is deliberately no grid-wide FC1/FC2
      // barrier: late FC1 tiles and ready FC2 tiles overlap across CTAs.
      __syncthreads();
      pipeline_iteration = 0;

      task_in_chunk = kTaskChunk;
      if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            directory_ready(scheduler[6ULL + 2ULL * a.world]);
        while (directory_ready.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
          if (threadIdx.x == 0U)
            __nanosleep(256U);
          __syncthreads();
        }
      }
      while (!forward_cancelled(a)) {
        if (task_in_chunk == kTaskChunk) {
          if (threadIdx.x == 0U)
            task_shared = atomicAdd(
                &scheduler[1], static_cast<unsigned long long>(kTaskChunk));
          __syncthreads();
          task_in_chunk = 0U;
        }
        const auto task = task_shared + task_in_chunk++;
        if (task >= fc2_tasks)
          break;
        const auto mblock = task / fc2_ntiles;
        const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
        if (threadIdx.x == 0U) {
          claim_shared =
              atomicCAS(&fc2_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
          if (completion != nullptr)
            completion->fc2_scheduler_value = task;
#endif
        }
        __syncthreads();
        if (claim_shared == 0U)
          continue;
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            claim_completion_tasks(completion,
                                   transport_state::TaskClass::Fc2);
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas) {
          completion->forward_fc2_cta_tiles_started[blockIdx.x] = task + 1ULL;
          completion->forward_fc2_cta_last_task[blockIdx.x] = task;
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (1ULL << 32U) | pipeline_iteration;
        }
#endif
        if (threadIdx.x == 0)
          atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
        const auto mapping = claims[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            fc1_ready(fc1_done[mblock]);
        while ((fc1_ready.load(::cuda::memory_order_acquire) & 0xffffffffULL) <
               fc1_ntiles) {
        }
        __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (2ULL << 32U) | pipeline_iteration;
#endif
        const auto row =
            a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
        const auto remaining_rows =
            a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            remaining_rows < 128ULL ? remaining_rows : 128ULL);
        const auto n = n_tile * 256U;
        float accum[ForwardMmaWide::kNumAccum];
        forward_wgmma_tile<ForwardMmaWide, 256U>(
            s_map, w2_map, static_cast<std::uint32_t>(row),
            static_cast<std::uint32_t>(expert * a.hidden + n),
            static_cast<std::uint32_t>(a.intermediate), smem_a, smem_b,
            fc2_full_barriers, fc2_empty_barriers, accum, pipeline_iteration, 0U
#if MEGAMOE_DEBUG_VALIDATE
            ,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_producer_checkpoint[blockIdx.x]
                : nullptr,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_math0_checkpoint[blockIdx.x]
                : nullptr,
            a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas
                ? &completion->forward_fc2_cta_math1_checkpoint[blockIdx.x]
                : nullptr
#endif
        );
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (3ULL << 32U) | pipeline_iteration;
#endif
        visit_forward_accumulator_wide(accum, [&](std::uint32_t warpgroup,
                                                  std::uint32_t m,
                                                  std::uint32_t column,
                                                  float v0, float v1) {
          const auto output_column = static_cast<std::uint64_t>(n) + column;
          const auto block_m = warpgroup * 64U + m;
          if (block_m < valid_rows && output_column < a.hidden)
            forward_output[(row + block_m) * a.hidden + output_column] =
                transport::BFloat16{float_to_bf16_bits(v0)};
          if (block_m < valid_rows && output_column + 1U < a.hidden)
            forward_output[(row + block_m) * a.hidden + output_column + 1U] =
                transport::BFloat16{float_to_bf16_bits(v1)};
        });
        // Publish this N tile's canonical stores before exposing completion.
        // The CTA that observes the final N tile owns the M-block return; other
        // CTAs can continue FC2 for unrelated blocks in parallel.
        __threadfence();
        __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas)
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (4ULL << 32U) | pipeline_iteration;
#endif
        __shared__ unsigned int return_mblock_shared;
        if (threadIdx.x == 0) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              complete(fc1_done[mblock]);
          const auto old =
              complete.fetch_add(1ULL << 16U, ::cuda::memory_order_acq_rel);
          return_mblock_shared =
              (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
        }
        __syncthreads();
        if (return_mblock_shared != 0U) {
          // Preserve the monolithic local/socket completion accounting. The
          // cross-node build always uses the split FULL descriptor path.
          __threadfence_system();
          __syncthreads();
          if (threadIdx.x == 0U) {
            const auto finished =
                atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
            if (finished == forward_mblocks) {
              status->fc2_complete_clock = device_globaltimer_ns();
              forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
            }
          }
          __syncthreads();
        }
        MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                                a.epoch);
        if (return_mblock_shared != 0U &&
            !remote_backend_enabled<StaticLocal>(a.use_socket)) {
          // Only lane 0 produces and consumes the peer-row audit count. Keeping
          // it in CTA shared storage made independent CTAs alias through the
          // cluster-shared address space under racecheck. A thread-local scalar
          // has the same audit semantics and no cross-thread lifetime.
          std::uint32_t return_peer_rows = 0U;
          if (threadIdx.x == 0)
            atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
          for (std::uint32_t source = threadIdx.x; source < a.world;
               source += blockDim.x)
            return_rows_by_source[source] = 0U;
          __syncthreads();
          for (std::uint32_t local_row = 0; local_row < valid_rows;
               ++local_row) {
            const auto return_row = row + local_row;
            const auto header = headers[return_row];
            if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
                header.source_rank / a.local_world_size != a.node)
              continue;
            auto *source_contribution = peer_at<transport::BFloat16>(
                a.peer_bases[header.source_rank], a.contribution_offset);
            const auto *source128 = reinterpret_cast<const uint4 *>(
                forward_output + return_row * a.hidden);
            auto *target128 = reinterpret_cast<uint4 *>(
                source_contribution + header.route_index * a.hidden);
            const auto vector_count = a.hidden / 8ULL;
            for (std::uint64_t vector = threadIdx.x; vector < vector_count;
                 vector += blockDim.x * 2ULL) {
              target128[vector] = source128[vector];
              const auto second = vector + blockDim.x;
              if (second < vector_count)
                target128[second] = source128[second];
            }
            if (threadIdx.x == 0) {
              ++return_rows_by_source[header.source_rank];
              if (header.source_rank != a.rank)
                ++return_peer_rows;
            }
          }
          if (threadIdx.x == 0 && return_peer_rows != 0U)
            atomicAdd(&status->actual_return_peer_bytes,
                      static_cast<unsigned long long>(return_peer_rows) *
                          a.hidden * sizeof(transport::BFloat16));
          __threadfence_system();
          __syncthreads();
          for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
               source_rank += blockDim.x) {
            if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
                source_rank / a.local_world_size != a.node)
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
          __syncthreads();
        }
        if constexpr (!StaticLocal)
          if (a.use_gin && threadIdx.x == 0U)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc2);
#if MEGAMOE_DEBUG_VALIDATE
        if (a.use_gin && threadIdx.x == 0U &&
            blockIdx.x < transport_state::kForwardDebugCtas) {
          completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
              (5ULL << 32U) | pipeline_iteration;
          completion->forward_fc2_cta_tiles_completed[blockIdx.x] = task + 1ULL;
        }
#endif
        if (threadIdx.x == 0U) {
          atomicExch(&fc2_claims[task], 2ULL);
        }
        __syncthreads();
      }
      // Return chunks have a single owner: their GIN context CTA. Allowing
      // completed compute CTAs to race those owners for WAITING -> PACKING made
      // the release kernel timing-dependent and could strand a context in its
      // terminal wait. The fixed progress team is sufficient to gather while
      // FC2 continues and keeps pack, issue, and completion ordered in one CTA.
    }
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    publish_local_structured_error(status, a.rank, 90U, 1U, 1ULL, 0ULL);
  }
#endif
  if constexpr (KernelRole != ForwardKernelRole::ComputeOnly) {
    if (remote_backend_enabled<StaticLocal>(a.use_socket)) {
      grid.sync();
      if (blockIdx.x == 0 && threadIdx.x == 0 &&
          a.injected_status_error_rank == static_cast<int>(a.rank))
        publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
      grid.sync();
      publish_production_terminal_status<StaticLocal>(
          a, status, ProductionStatusPhase::Forward,
          a.production_error_records_offset,
          a.gin_production_error_records_offset, 0ULL);
      grid.sync();
      if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
        status->fc1_tasks_done = fc1_tasks;
        status->fc2_tasks_done = fc2_tasks;
        const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
        const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
        status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                            fc2_tasks * fc2_k_blocks * 2ULL;
        status->wgmma_instructions =
            (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) *
            2ULL * (64ULL / ForwardMma::K);
#endif
        status->return_complete_clock = device_globaltimer_ns();
        status->complete_clock = status->return_complete_clock;
      }
      forward_publish_progress(a, kForwardFc2Complete, scheduler[3]);
      forward_publish_progress(a, kForwardTerminal, a.epoch);
      return;
    }
    if (threadIdx.x == 0)
      atomicMax(&status->return_complete_clock, device_globaltimer_ns());
    forward_publish_progress(a, kForwardReturnObserved, scheduler[3]);

    // A CTA may begin reducing tokens whose destination-rank bulk signals are
    // already visible while other CTAs finish returning later M-blocks.  The
    // per-destination acquire below is the dependency; a grid barrier here only
    // serialized independent return and reduce work.
    auto *contribution =
        peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
    auto *source_ready =
        peer_at<unsigned long long>(a.local_base, a.outgoing_sequence_offset);
    const auto *direct_source_compact_index =
        peer_at<const unsigned long long>(
            a.local_base, a.forward_direct_source_compact_index_offset);
    const auto *direct_return_receive =
        a.use_gin
            ? reinterpret_cast<const transport::BFloat16 *>(
                  static_cast<const std::byte *>(a.gin_window_base) +
                  a.gin_return_receive_offset)
            : nullptr;
    auto *combined =
        peer_at<transport::BFloat16>(a.local_base, a.combined_offset);
    constexpr std::uint32_t kWarpsPerCta = 12U;
    constexpr std::uint32_t kWarpsPerToken = 1U;
    constexpr std::uint32_t kTokensPerCta = kWarpsPerCta / kWarpsPerToken;
    const auto warp = threadIdx.x / 32U;
    const auto lane = threadIdx.x & 31U;
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    const auto token_warp = warp % kWarpsPerToken;
    for (std::uint64_t token =
             static_cast<std::uint64_t>(blockIdx.x) * kTokensPerCta +
             warp / kWarpsPerToken;
         token < a.dispatch.identity.num_tokens && !forward_cancelled(a);
         token += static_cast<std::uint64_t>(gridDim.x) * kTokensPerCta) {
      if constexpr (!StaticLocal)
        if (a.use_gin && lane == 0U)
          claim_completion_tasks(completion,
                                 transport_state::TaskClass::Reduce);
      if (lane == 0U && status->reduced_tokens == ~0ULL)
        atomicMin(&status->reduced_tokens, device_globaltimer_ns());
      unsigned long long lane_route = 0ULL;
      unsigned long long lane_row = 0ULL;
      unsigned int lane_direct = 0U;
      if (lane < a.dispatch.identity.top_k) {
        const auto slot =
            a.dispatch.combine_order[token * a.dispatch.identity.top_k + lane];
        lane_route = token * a.dispatch.identity.top_k + slot;
        lane_row = lane_route;
        const auto destination = a.dispatch.route_dst_rank[lane_route];
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(source_ready[destination * a.channels]);
        while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
        }
        if (a.use_gin &&
            a.dispatch.route_dst_node[lane_route] != a.node) {
          lane_row = direct_source_compact_index[lane_route];
          lane_direct = 1U;
        }
      }
      __syncwarp();
      auto *output8 = reinterpret_cast<uint4 *>(combined + token * a.hidden);
      for (std::uint64_t vector = lane + token_warp * 32U;
           vector < a.hidden / 8ULL; vector += kWarpsPerToken * 32ULL) {
        float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
        for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
             ++order) {
          const auto row = __shfl_sync(0xffffffffU, lane_row, order);
          const auto direct =
              __shfl_sync(0xffffffffU, lane_direct, order) != 0U;
          const auto *source = direct ? direct_return_receive : contribution;
          const auto packed = reinterpret_cast<const uint4 *>(
              source + row * a.hidden)[vector];
          const unsigned int words[4] = {packed.x, packed.y, packed.z,
                                         packed.w};
#pragma unroll
          for (std::uint32_t word = 0; word < 4U; ++word) {
            sums[word * 2U] = __fadd_rn(
                sums[word * 2U],
                bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
            sums[word * 2U + 1U] = __fadd_rn(
                sums[word * 2U + 1U], bf16_bits_to_float(
                                          static_cast<std::uint16_t>(
                                              words[word] >> 16U)));
          }
        }
        unsigned int words[4];
#pragma unroll
        for (std::uint32_t word = 0; word < 4U; ++word)
          words[word] =
              static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U])) |
              (static_cast<unsigned int>(
                   float_to_bf16_bits(sums[word * 2U + 1U]))
               << 16U);
        output8[vector] = make_uint4(words[0], words[1], words[2], words[3]);
      }
      __syncwarp();
      if constexpr (!StaticLocal)
        if (a.use_gin && lane == 0U)
          complete_completion_tasks(completion,
                                    transport_state::TaskClass::Reduce);
    }
    if (lane == 0U)
      atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    __syncthreads();
    grid.sync();
    forward_publish_progress(a, kForwardReduceComplete,
                             a.dispatch.identity.num_tokens);
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        a.injected_status_error_rank == static_cast<int>(a.rank))
      publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
    grid.sync();
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        if (blockIdx.x == 0U && threadIdx.x == 0U) {
          completion->ring_credit_restored = completion->ring_credit_expected;
          publish_elected_terminal(completion, a.epoch, status->error != 0U);
        }
        __syncthreads();
        if (blockIdx.x != 0U) {
          // Elect one lane to poll.  forward_cancelled() contains warp-level
          // collectives used by the normal worker loops, so evaluating it from
          // independently reconverging lanes at terminal can strand the rest of
          // warp 0 in WARPSYNC even after the completion epoch is visible.
          if (threadIdx.x == 0U) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
                terminal(completion->kernel_complete_epoch);
            while (terminal.load(::cuda::memory_order_acquire) < a.epoch) {
            }
          }
          __syncthreads();
          return;
        }
      }
    }
    // Only the elected terminal CTA performs the production status agreement.
    // GIN progress CTAs and compute CTAs have already proved their static task
    // spaces complete and now leave after observing kernel_complete_epoch.  In
    // particular, they never participate in a second grid-wide rendezvous whose
    // progress would otherwise depend on CTA-local control-flow state.
    publish_production_terminal_status<StaticLocal>(
        a, status, ProductionStatusPhase::Forward,
        a.production_error_records_offset,
        a.gin_production_error_records_offset,
        a.use_gin ? a.dispatch.identity.num_nodes - 1ULL : 0ULL);
    if constexpr (StaticLocal) {
      grid.sync();
    } else if (!a.use_gin) {
      grid.sync();
    } else {
      __syncthreads();
    }
    // Every local return CTA has fenced before publishing its destination
    // signal, and every local Reduce warp acquired all of its destination
    // signals before reaching this grid completion.  Consequently no peer can
    // still write this slot and no local worker can still read it; a second
    // all-rank terminal rendezvous would only duplicate those dependencies.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
      status->fc1_tasks_done = fc1_tasks;
      status->fc2_tasks_done = fc2_tasks;
      const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
      const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
      status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                          fc2_tasks * fc2_k_blocks * 2ULL;
      status->wgmma_instructions =
          (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) * 2ULL *
          (64ULL / ForwardMma::K);
#endif
      status->complete_clock = device_globaltimer_ns();
    }
    forward_publish_progress(a, kForwardTerminal, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
  }
}
