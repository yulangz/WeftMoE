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

