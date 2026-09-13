#pragma once

// Hopper-only Forward compute grid.  Transport owns generation setup and the
// immutable Dispatch/Return directories; this grid owns only FC1/FC2.  Keeping
// the WGMMA role in a separate kernel and an out-of-line device function is
// intentional: ptxas can now assign the math warpgroup and the TMA
// producer independent dynamic register quotas without carrying any GIN or
// relay live ranges through the WGMMA pipeline.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

namespace forward_compute_detail {

// The transport grid launches with one CTA per cluster. Observe the
// compute grid too so split-kernel diagnostics report its actual launch.
__device__ __forceinline__ void
observe_compute_cluster(const ForwardKernelArgs &a) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (threadIdx.x == 0U) {
    const auto cluster = cooperative_groups::this_cluster();
    const auto dimensions = cluster.dim_blocks();
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicMax(&status->runtime_cluster_size,
              dimensions.x * dimensions.y * dimensions.z);
    atomicOr(&status->cluster_rank_mask, 1ULL << cluster.block_rank());
  }
#endif
}

constexpr std::uint32_t kThreads = forward_tuning::kComputeThreads;
constexpr std::uint32_t kMathWarpGroups = forward_tuning::kMathWarpgroups;
constexpr std::uint32_t kMathThreads = kMathWarpGroups * 128U;
constexpr std::uint32_t kTaskReadyBarrier = 0U;
constexpr std::uint32_t kTaskDoneBarrier = 1U;
constexpr std::uint32_t kMathReadyBarrier = 2U;
constexpr std::uint32_t kMathDoneBarrier = 3U;
constexpr std::uint32_t kReduceRegisterReleaseBarrier = 4U;
constexpr std::uint32_t kReduceRegisterAcquireBarrier = 5U;
constexpr std::uint32_t kFc1Stages = forward_tuning::kFc1Stages;
constexpr std::uint32_t kFc2Stages = forward_tuning::kFc2Stages;
constexpr std::uint32_t kFc2Iterations =
    forward_tuning::kMainIntermediate / forward_tuning::kTileK;
constexpr std::uint32_t kAStageElements =
    static_cast<std::uint32_t>(forward_tuning::kAStageElements);
constexpr std::uint32_t kFc1BStageElements =
    static_cast<std::uint32_t>(forward_tuning::kFc1BStageElements);
constexpr std::uint32_t kFc2BStageElements =
    static_cast<std::uint32_t>(forward_tuning::kFc2BStageElements);

enum ForwardComputeTaskKind : std::uint32_t {
  kComputeTaskNone = 0U,
  kComputeTaskFc1 = 1U,
  kComputeTaskFc2 = 2U,
  kComputeTaskDone = 3U,
  kComputeTaskReduce = 4U,
};

enum ForwardComputePairOperand : std::uint32_t {
  kComputePairShareA = 1U,
  kComputePairShareB = 2U,
};

// FC1's WGMMA accumulator is FP32, but the training contract checkpoints A/B
// in BF16. Forward must consume those exact checkpoint values as well, or a
// later SwiGLU recomputation would silently use different operands. FC2 then
// consumes the separately rounded BF16 U = gate*S.
__device__ __forceinline__ float gated_swiglu_fc2_input(float a, float b,
                                                        float gate) {
  const auto sigmoid = 1.0F / (1.0F + __expf(-a));
  return __fmul_rn((a * sigmoid) * b, gate);
}

__device__ __forceinline__ float
gated_swiglu_fc2_input_from_bf16(std::uint16_t a_bits, std::uint16_t b_bits,
                                 float gate) {
  return gated_swiglu_fc2_input(bf16_bits_to_float(a_bits),
                                bf16_bits_to_float(b_bits), gate);
}

#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
#if !defined(WEFTMOE_BUILD_TESTING)
#error "WEFTMOE_BENCH_TRANSPORT_ONLY requires the cuda-test build target"
#endif
#include "forward_transport_benchmark.cuh"
#endif

template <typename TileMma, class Callback>
__device__ __forceinline__ void visit_m128_accumulator(float *accum,
                                                       Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

template <typename TileMma, class Callback>
__device__ __forceinline__ void
visit_m128_accumulator_pair(float *accum_a, float *accum_b, Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum_a[index * 4U],
             accum_a[index * 4U + 1U], accum_b[index * 4U],
             accum_b[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum_a[index * 4U + 2U],
             accum_a[index * 4U + 3U], accum_b[index * 4U + 2U],
             accum_b[index * 4U + 3U]);
  }
}

// FC1/FC2 lends most of the CTA register file to the WGMMA warpgroups, with a
// separately tuned quota for the TMA producer. Once the final GEMM tile has
// retired, those asymmetric roles no longer overlap with Reduce. Rebalance
// the same physical register file to the tuned uniform Reduce quota instead
// of carrying either compute role's live-range budget into the terminal
// phase. The named barriers are CTA-local; this is not a grid-wide phase
// barrier and Return/RDMA/Relay continue concurrently on their own CTAs.
__device__ __forceinline__ void rebalance_registers_for_reduce() {
  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup < kMathWarpGroups)
    cutlass::arch::warpgroup_reg_dealloc<forward_tuning::kReduceRegisters>();
  cutlass::arch::NamedBarrier::sync(kThreads, kReduceRegisterReleaseBarrier);
  if (warpgroup >= kMathWarpGroups)
    cutlass::arch::warpgroup_reg_alloc<forward_tuning::kReduceRegisters>();
  cutlass::arch::NamedBarrier::sync(kThreads, kReduceRegisterAcquireBarrier);
}

// The locality publisher reserves even-sized queue regions. A cluster can
// therefore claim two adjacent entries without crossing an independently
// published group. Depending on the ordering, the pair shares either its A
// activation tile or its expert/N weight tile.
template <bool StaticLocal>
__device__ __forceinline__ bool
try_claim_fc1_task_pair(const ForwardKernelArgs &a,
                        unsigned long long *scheduler,
                        unsigned long long *claims,
                        unsigned long long fc1_tasks, std::uint32_t fc1_ntiles,
                        transport_state::GenerationCompletionVector *completion,
                        unsigned long long &task0, unsigned long long &task1,
                        std::uint32_t &pair_operand, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto all_tasks = forward_mblocks * fc1_ntiles;
  auto *claim_word = &scheduler[0];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  auto slot = cursor.load(::cuda::memory_order_relaxed);
  if (slot >= all_tasks) {
    exhausted = true;
    return false;
  }
  // Cluster mode is admitted only for an even FC1 task extent. Keep this
  // defensive guard so a future shape cannot consume a singleton as a pair.
  if (slot + 1ULL >= all_tasks) {
    exhausted = false;
    return false;
  }

  auto selected0 = slot;
  auto selected1 = selected0 + 1ULL;
  if (a.use_gin || a.forward_stream_local_dispatch) {
    auto *ready_queue = claims + forward_mblocks;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published0(ready_queue[slot]);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published1(ready_queue[slot + 1ULL]);
    const auto encoded0 = published0.load(::cuda::memory_order_acquire);
    const auto encoded1 = published1.load(::cuda::memory_order_acquire);
    if (encoded0 == 0ULL || encoded1 == 0ULL) {
      exhausted = false;
      return false;
    }
    selected0 = encoded0 - 1ULL;
    selected1 = encoded1 - 1ULL;
  }

  if (!cursor.compare_exchange_strong(slot, slot + 2ULL,
                                      ::cuda::memory_order_acq_rel,
                                      ::cuda::memory_order_relaxed)) {
    exhausted = false;
    return false;
  }
  task0 = selected0;
  task1 = selected1;
  const auto mblock0 = selected0 / fc1_ntiles;
  const auto mblock1 = selected1 / fc1_ntiles;
  const auto ntile0 = selected0 % fc1_ntiles;
  const auto ntile1 = selected1 % fc1_ntiles;
  if (mblock0 == mblock1) {
    pair_operand = kComputePairShareA;
  } else if (ntile0 == ntile1 &&
             static_cast<std::uint32_t>(claims[mblock0] >> 32U) ==
                 static_cast<std::uint32_t>(claims[mblock1] >> 32U)) {
    pair_operand = kComputePairShareB;
  } else {
    asm volatile("trap;"); // Queue publication must form multicast pairs.
  }
  exhausted = false;
#if WEFTMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      claim_completion_tasks(completion, transport_state::TaskClass::Fc1);
      claim_completion_tasks(completion, transport_state::TaskClass::Fc1);
      completion->fc1_scheduler_value = slot + 1ULL;
      completion->bulk_local_ready = 11ULL;
    }
  }
#endif
  (void)fc1_tasks;
  return true;
}

template <bool StaticLocal>
__device__ __forceinline__ bool
try_claim_fc2_task_pair(const ForwardKernelArgs &a,
                        unsigned long long *scheduler,
                        unsigned long long *ready_queue,
                        unsigned long long fc2_tasks, std::uint32_t fc2_ntiles,
                        transport_state::GenerationCompletionVector *completion,
                        unsigned long long &task0, unsigned long long &task1,
                        std::uint32_t &pair_operand, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  auto *claim_word = &scheduler[1];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  while (true) {
    auto slot = cursor.load(::cuda::memory_order_relaxed);
    if (slot >= forward_mblocks) {
      exhausted = true;
      return false;
    }
    const auto queue_slot = slot;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(ready_queue[queue_slot]);
    const auto encoded = published.load(::cuda::memory_order_acquire);
    if (encoded == 0ULL) {
      exhausted = false;
      return false;
    }
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        tile_cursor(ready_queue[forward_mblocks + queue_slot]);
    const auto n_tile =
        tile_cursor.fetch_add(2ULL, ::cuda::memory_order_relaxed);
    if (n_tile + 1ULL >= fc2_ntiles) {
      auto expected = slot;
      cursor.compare_exchange_strong(expected, slot + 1ULL,
                                     ::cuda::memory_order_relaxed,
                                     ::cuda::memory_order_relaxed);
      continue;
    }
    const auto mblock = encoded - 1ULL;
    task0 = mblock * fc2_ntiles + n_tile;
    task1 = task0 + 1ULL;
    pair_operand = kComputePairShareA;
    exhausted = false;
#if WEFTMOE_DEBUG_VALIDATE
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        claim_completion_tasks(completion, transport_state::TaskClass::Fc2);
        claim_completion_tasks(completion, transport_state::TaskClass::Fc2);
      }
    }
    if (a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas) {
      completion->fc2_scheduler_value = slot * fc2_ntiles + n_tile + 1ULL;
      const auto local_task =
          cute::block_rank_in_cluster() == 0U ? task0 : task1;
      completion->forward_fc2_cta_tiles_started[blockIdx.x] = local_task + 1ULL;
      completion->forward_fc2_cta_last_task[blockIdx.x] = local_task;
      completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] = 1ULL << 32U;
    }
#endif
    (void)fc2_tasks;
    return true;
  }
}

// A CTA never reserves an unpublished slot.  It first tries FC2, then FC1,
// and repeats the FC2 probe before sleeping.  Thus every scheduling boundary
// favors an already-ready FC2 tile while all CTAs remain available to FC1
// during the initial Dispatch wavefront.
template <bool StaticLocal>
__device__ __forceinline__ void select_compute_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long *fc2_ready_queue,
    unsigned long long fc1_tasks, unsigned long long fc2_tasks,
    std::uint32_t fc1_ntiles, std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long *task_shared, unsigned int *task_kind_shared,
    unsigned int *pair_operand_shared, unsigned long long &pair_round,
    unsigned int &fc2_streak) {
  if (threadIdx.x != 0U)
    return;
  const auto cluster_rank = cute::block_rank_in_cluster();
  auto *mailboxes = peer_at<persistent_v3::ForwardComputePairMailbox>(
      a.local_base, a.forward_compute_pair_mailboxes_offset);
  auto &mailbox = mailboxes[static_cast<std::uint32_t>(blockIdx.x) / 2U];

  // Acknowledge the previous pair before the leader overwrites its mailbox.
  // Only one lane per CTA participates; the ordinary CTA named barrier below
  // distributes the resulting local task to its producer and math warpgroups.
  if (pair_round != 0ULL) {
    const auto complete_sequence = pair_round * 2ULL;
    if (cluster_rank == 1U) {
      persistent_v3::store_release<::cuda::thread_scope_device>(
          mailbox.sequence, complete_sequence);
    } else {
      while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                 mailbox.sequence) != complete_sequence)
        __nanosleep(forward_tuning::kTaskIdleNanosleep);
    }
  }

  const auto ready_sequence = pair_round * 2ULL + 1ULL;
  if (cluster_rank == 0U) {
    unsigned long long task0 = 0ULL;
    unsigned long long task1 = 0ULL;
    std::uint32_t task_kind = kComputeTaskDone;
    std::uint32_t pair_operand = kComputePairShareA;
    while (!forward_cancelled(a)) {
      bool fc1_exhausted = false;
      bool fc2_exhausted = false;
      const bool fc2_first = forward_tuning::kFc2PriorityBurst == 0U ||
                             fc2_streak < forward_tuning::kFc2PriorityBurst;

      if (fc2_first &&
          try_claim_fc2_task_pair<StaticLocal>(
              a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles, completion,
              task0, task1, pair_operand, fc2_exhausted)) {
        ++fc2_streak;
        task_kind = kComputeTaskFc2;
        break;
      }

      if (try_claim_fc1_task_pair<StaticLocal>(
              a, scheduler, claims, fc1_tasks, fc1_ntiles, completion, task0,
              task1, pair_operand, fc1_exhausted)) {
        fc2_streak = 0U;
        task_kind = kComputeTaskFc1;
        break;
      }

      if (!fc2_first &&
          try_claim_fc2_task_pair<StaticLocal>(
              a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles, completion,
              task0, task1, pair_operand, fc2_exhausted)) {
        ++fc2_streak;
        task_kind = kComputeTaskFc2;
        break;
      }

      if (const auto encoded =
              forward_reduce_detail::try_claim_reduce_token(a)) {
        task0 = encoded;
        task1 = forward_reduce_detail::try_claim_reduce_token(a);
        task_kind = kComputeTaskReduce;
        break;
      }
      if (fc1_exhausted && fc2_exhausted)
        break;
      __nanosleep(forward_tuning::kTaskIdleNanosleep);
    }
    mailbox.tasks[0] = task0;
    mailbox.tasks[1] = task1;
    mailbox.metadata = static_cast<unsigned long long>(task_kind) |
                       (static_cast<unsigned long long>(pair_operand) << 32U);
    persistent_v3::store_release<::cuda::thread_scope_device>(mailbox.sequence,
                                                              ready_sequence);
    *task_shared = task0;
    *task_kind_shared = task_kind;
    *pair_operand_shared = pair_operand;
  } else {
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               mailbox.sequence) != ready_sequence)
      __nanosleep(forward_tuning::kTaskIdleNanosleep);
    const auto metadata = mailbox.metadata;
    *task_shared = mailbox.tasks[1];
    *task_kind_shared = static_cast<std::uint32_t>(metadata);
    *pair_operand_shared = static_cast<std::uint32_t>(metadata >> 32U);
  }
  ++pair_round;
}

template <bool StaticLocal, bool RetainBackwardState,
          std::uint32_t Fc1StageCount, std::uint32_t Fc2StageCount>
__device__ __forceinline__ void
persistent_role(const ForwardKernelArgs &a, const CUtensorMap &x_map,
                const CUtensorMap &s_map, const CUtensorMap &w1_map,
                const CUtensorMap &w2_map, unsigned char *storage,
                unsigned long long *task_shared, unsigned int *task_kind_shared,
                unsigned int *return_mblock_shared,
                unsigned int *pair_operand_shared,
                const CUtensorMap *output_store_map) {
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
  transport_benchmark_role<StaticLocal>(a, storage);
  return;
#endif
  static_assert(Fc1StageCount >= 2U && Fc1StageCount <= 4U,
                "FC1 supports two to four TMA stages");
  static_assert(Fc2StageCount >= 2U && Fc2StageCount <= 4U,
                "FC2 supports two to four TMA stages");
  static_assert(kFc2Iterations % Fc2StageCount == 0U,
                "the fixed FC2 reduction must divide evenly into stages");
  constexpr auto kConfiguredFc2Waves = kFc2Iterations / Fc2StageCount;
  constexpr auto kConfiguredAStorageElements =
      (Fc1StageCount > Fc2StageCount ? Fc1StageCount : Fc2StageCount) *
      kAStageElements;
  constexpr auto kConfiguredBStorageElements =
      2U * Fc1StageCount * kFc1BStageElements >
              Fc2StageCount * kFc2BStageElements
          ? 2U * Fc1StageCount * kFc1BStageElements
          : Fc2StageCount * kFc2BStageElements;
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
    ab_cache = a.saved_ab;
  const auto *canonical_gates =
      peer_at<const float>(a.local_base, a.canonical_gates_offset);
  auto *forward_output =
      peer_at<transport::BFloat16>(a.local_base, a.forward_output_rows_offset);

  transport_state::GenerationCompletionVector *completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin)
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
  }

  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles = static_cast<std::uint32_t>(
      (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
      forward_tuning::kFc1TileN);
  const auto fc2_ntiles =
      static_cast<std::uint32_t>((a.hidden + forward_tuning::kFc2TileN - 1ULL) /
                                 forward_tuning::kFc2TileN);
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *fc2_ready_queue = fc1_ready_queue + fc1_tasks;

  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kConfiguredAStorageElements;
  auto *fc1_b2 = smem_b + Fc1StageCount * kFc1BStageElements;
  auto *fc1_full =
      reinterpret_cast<ForwardBarrier *>(smem_b + kConfiguredBStorageElements);
  auto *fc1_empty = fc1_full + Fc1StageCount;
  auto *fc2_full = fc1_empty + Fc1StageCount;
  auto *fc2_empty = fc2_full + Fc2StageCount;
  auto *fc1_b2_full = fc2_empty + Fc2StageCount;

  if (threadIdx.x == kMathThreads) {
    for (std::uint32_t stage = 0; stage < Fc1StageCount; ++stage) {
      fc1_full[stage].init(1);
      fc1_empty[stage].init(2U * kMathThreads / 32U);
      fc1_b2_full[stage].init(1);
    }
    for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
      fc2_full[stage].init(1);
      fc2_empty[stage].init(2U * kMathThreads / 32U);
    }
    cutlass::arch::fence_barrier_init();
  }
  cute::cluster_sync();

  const auto warpgroup = threadIdx.x / 128U;
  if (warpgroup >= kMathWarpGroups) {
    // Only the first producer-warpgroup lane issues TMA; the remaining lanes
    // carry the CTA task/barrier protocol with a deliberately small register
    // set.
    cutlass::arch::warpgroup_reg_dealloc<forward_tuning::kProducerRegisters>();
    std::uint32_t fc1_pipeline = 0U;
    std::uint32_t fc2_pipeline = 0U;
    // The math scheduler publishes the first task through this one-time
    // barrier.  At every later boundary kTaskDoneBarrier also publishes the
    // next task, avoiding a second CTA-wide rendezvous per GEMM tile.
    cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
    while (true) {
      if (*task_kind_shared == kComputeTaskReduce) {
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
        continue;
      }

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
                         forward_mblock_local_row(local_mblock);
        if (threadIdx.x == kMathThreads) {
          // The readiness acquire is performed by the scalar scheduler lane and
          // handed to this producer through the CTA named barrier.  Make the
          // generic/system writes visible to the TMA async proxy before issuing
          // the first load for the claimed M block.
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto iterations = static_cast<std::uint32_t>(
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
                *pair_operand_shared == kComputePairShareA ? 2U : 1U;
            const auto multicast_b =
                *pair_operand_shared == kComputePairShareA ? 1U : 2U;

            hopper::tma_load<forward_tuning::kTileK, forward_tuning::kTileM,
                             128, cutlass::bfloat16_t>(
                &x_map, &fc1_full[stage], stage_a, k_base,
                static_cast<std::uint32_t>(row), multicast_a);

            hopper::tma_load<forward_tuning::kTileK, forward_tuning::kFc1TileN,
                             128, cutlass::bfloat16_t>(
                &w1_map, &fc1_full[stage], stage_b1, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
                multicast_b);
            hopper::tma_load<forward_tuning::kTileK, forward_tuning::kFc1TileN,
                             128, cutlass::bfloat16_t>(
                &w1_map, &fc1_b2_full[stage], stage_b2, k_base,
                static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                           a.intermediate + n),
                multicast_b);
            constexpr auto a_stage_elements = kAStageElements;
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

      if (*task_kind_shared == kComputeTaskDone)
        break;

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
                         forward_mblock_local_row(local_mblock);
        if (threadIdx.x == kMathThreads) {
          // Keep stage ordinals unrolled, but carry barrier parity between
          // tasks when the reduction has an odd number of pipeline waves.
          // Even-wave shapes keep the original compile-time phase expression.
          for (std::uint32_t wave = 0; wave < kConfiguredFc2Waves; ++wave) {
#pragma unroll
            for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
              const auto phase = ((kConfiguredFc2Waves % 2U != 0U
                                       ? fc2_pipeline / Fc2StageCount
                                       : 0U) +
                                  wave) &
                                 1U;
              fc2_empty[stage].wait(phase ^ 1U);
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements);
              auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b + stage * kFc2BStageElements);
              const auto k_base =
                  (wave * Fc2StageCount + stage) * forward_tuning::kTileK;
              const auto n = n_tile * forward_tuning::kFc2TileN;
              hopper::tma_load<forward_tuning::kTileK, forward_tuning::kTileM,
                               128, cutlass::bfloat16_t>(
                  &s_map, &fc2_full[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row), 2U);

              hopper::tma_load<forward_tuning::kTileK,
                               forward_tuning::kFc2TileN, 128,
                               cutlass::bfloat16_t>(
                  &w2_map, &fc2_full[stage], stage_b, k_base,
                  static_cast<std::uint32_t>(expert * a.hidden + n));
              constexpr auto a_stage_elements = kAStageElements;
              fc2_full[stage].arrive_and_expect_tx(
                  (a_stage_elements + kFc2BStageElements) *
                  sizeof(transport::BFloat16));
            }
          }
          fc2_pipeline += kFc2Iterations;
        }
        cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      }

      if (*task_kind_shared == kComputeTaskDone)
        break;
    }
    cute::cluster_sync();
    return;
  }

  cutlass::arch::warpgroup_reg_alloc<forward_tuning::kMathRegisters>();
  std::uint32_t fc1_pipeline = 0U;
  std::uint32_t fc2_pipeline = 0U;
  unsigned long long pair_round = 0ULL;
  unsigned int fc2_streak = 0U;
#if WEFTMOE_DEVICE_TIMELINE
  bool timeline_fc1_started = false;
  bool timeline_fc2_started = false;
#endif
  select_compute_task<StaticLocal>(a, scheduler, claims, fc2_ready_queue,
                                   fc1_tasks, fc2_tasks, fc1_ntiles, fc2_ntiles,
                                   completion, task_shared, task_kind_shared,
                                   pair_operand_shared, pair_round, fc2_streak);
  cutlass::arch::NamedBarrier::sync(kThreads, kTaskReadyBarrier);
  while (true) {
    if (*task_kind_shared == kComputeTaskReduce) {
      if (threadIdx.x < 32U)
        forward_reduce_detail::reduce_claimed_token(a, *task_shared);
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
      select_compute_task<StaticLocal>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
          pair_operand_shared, pair_round, fc2_streak);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
      continue;
    }

    while (true) {
      const auto task = *task_shared;
      if (*task_kind_shared != kComputeTaskFc1)
        break;
#if WEFTMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U && !timeline_fc1_started) {
        WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstMtileReady, Compute, 0U, task,
                                execution_epoch(a));
        WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1First, Compute, 0U, task,
                                execution_epoch(a));
        WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1ClaimStart, Compute, 0U, task,
                                execution_epoch(a));
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
                       forward_mblock_local_row(local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      const auto n = n_tile * forward_tuning::kFc1TileN;
      using Fc1TileMma = TunedForwardFc1Mma;
      float accum_a[Fc1TileMma::kNumAccum];
      float accum_b[Fc1TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc1TileMma::kNumAccum; ++index) {
        accum_a[index] = 0.0F;
        accum_b[index] = 0.0F;
      }
      const auto iterations = static_cast<std::uint32_t>(
          (a.hidden + forward_tuning::kTileK - 1ULL) / forward_tuning::kTileK);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto current = fc1_pipeline + iteration;
        const auto stage = current % Fc1StageCount;
        const auto phase = (current / Fc1StageCount) & 1U;
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_a + stage * kAStageElements +
            warpgroup * 64U * forward_tuning::kTileK);
        auto *stage_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
            smem_b + stage * kFc1BStageElements);
        auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
            fc1_b2 + stage * kFc1BStageElements);
        // A and gate share the first completion barrier, while up has its own.
        // Start the gate WGMMA group as soon as 32 KiB is ready so its issue
        // overlaps the tail of the independent 16 KiB up-weight TMA.
        fc1_full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum_a[index]);
        cute::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0; k < forward_tuning::kTileK / Fc1TileMma::K;
             ++k) {
          const auto desc_a =
              hopper::make_smem_desc(stage_a + k * Fc1TileMma::K, 1);
          const auto desc_b1 =
              hopper::make_smem_desc(stage_b1 + k * Fc1TileMma::K, 1);
          Fc1TileMma::wgmma(desc_a, desc_b1, accum_a, 1);
        }
        cute::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum_a[index]);

        fc1_b2_full[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum_b[index]);
        cute::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0; k < forward_tuning::kTileK / Fc1TileMma::K;
             ++k) {
          const auto desc_a =
              hopper::make_smem_desc(stage_a + k * Fc1TileMma::K, 1);
          const auto desc_b2 =
              hopper::make_smem_desc(stage_b2 + k * Fc1TileMma::K, 1);
          Fc1TileMma::wgmma(desc_a, desc_b2, accum_b, 1);
        }
        cute::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < Fc1TileMma::kNumAccum; ++index)
          cute::warpgroup_fence_operand(accum_b[index]);
        cute::warpgroup_wait<forward_tuning::kWgmmaWaitGroup>();

        const auto lane = threadIdx.x & 31U;
        if (lane < 2U)
          fc1_empty[stage].arrive(lane);
      }
      fc1_pipeline += iterations;

      const auto store_fc1 = [&](std::uint32_t wg, std::uint32_t m,
                                 std::uint32_t column, float av0, float av1,
                                 float bv0, float bv1) {
        const auto block_m = wg * 64U + m;
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
            reinterpret_cast<std::uint32_t *>(ab_cache)[output_index / 2ULL] =
                a_pair;
            const auto b_index = a.saved_rows * a.intermediate + output_index;
            reinterpret_cast<std::uint32_t *>(ab_cache)[b_index / 2ULL] =
                b_pair;
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
                ab_cache[output_index] = transport::BFloat16{a_bits};
                ab_cache[a.saved_rows * a.intermediate + output_index] =
                    transport::BFloat16{b_bits};
              }
              s[output_index] = transport::BFloat16{float_to_bf16_bits(
                  gated_swiglu_fc2_input_from_bf16(a_bits, b_bits, gate))};
            }
          }
        }
      };
      visit_m128_accumulator_pair<Fc1TileMma>(accum_a, accum_b, store_fc1);
      __threadfence();
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            ready(fc1_done[mblock]);
        const auto old = ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if WEFTMOE_DEBUG_VALIDATE
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
            WEFTMOE_TIMELINE_RECORD(a.timeline, Fc1Last, Compute, 0U, fc1_tasks,
                                    execution_epoch(a));
            forward_publish_progress(a, kForwardFc1Complete, fc1_tasks);
          }
          const auto queue_slot = atomicAdd(&scheduler[2], 1ULL);
#if WEFTMOE_DEBUG_VALIDATE
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
      select_compute_task<StaticLocal>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
          pair_operand_shared, pair_round, fc2_streak);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;

    while (true) {
      if (*task_kind_shared != kComputeTaskFc2)
        break;
#if WEFTMOE_DEVICE_TIMELINE
      if (threadIdx.x == 0U && !timeline_fc2_started) {
        WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2First, Compute, 0U, *task_shared,
                                execution_epoch(a));
        timeline_fc2_started = true;
      }
#endif
      if (threadIdx.x == 0U)
        atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
      using Fc2TileMma = TunedForwardFc2Mma;
      float accum[Fc2TileMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
        accum[index] = 0.0F;
      for (std::uint32_t wave = 0; wave < kConfiguredFc2Waves; ++wave) {
#pragma unroll
        for (std::uint32_t stage = 0; stage < Fc2StageCount; ++stage) {
          // Match the producer's persistent barrier phase across FC2 tasks.
          const auto phase =
              ((kConfiguredFc2Waves % 2U != 0U ? fc2_pipeline / Fc2StageCount
                                               : 0U) +
               wave) &
              1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements +
              warpgroup * 64U * forward_tuning::kTileK);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * kFc2BStageElements);
          fc2_full[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            cute::warpgroup_fence_operand(accum[index]);
          cute::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0; k < forward_tuning::kTileK / Fc2TileMma::K;
               ++k) {
            const auto desc_a =
                hopper::make_smem_desc(stage_a + k * Fc2TileMma::K, 1);
            const auto desc_b =
                hopper::make_smem_desc(stage_b + k * Fc2TileMma::K, 1);
            Fc2TileMma::wgmma(desc_a, desc_b, accum, 1);
          }
          cute::warpgroup_commit_batch();
#pragma unroll
          for (int index = 0; index < Fc2TileMma::kNumAccum; ++index)
            cute::warpgroup_fence_operand(accum[index]);
          // TMA remains asynchronous in the producer warpgroup.  The
          // math side completes one K=64 group before recycling that stage; the
          // previous wait<1> needed a second live accumulator generation and
          // made ptxas serialize every individual WGMMA instruction.
          cute::warpgroup_wait<forward_tuning::kWgmmaWaitGroup>();

          const auto lane = threadIdx.x & 31U;
          if (lane < 2U)
            fc2_empty[stage].arrive(lane);
        }
      }
      fc2_pipeline += kFc2Iterations;

      // Reload immutable task metadata only after the WGMMA mainloop.  This is
      // the register analogue of a role union: task/return bookkeeping reuses
      // registers that the asynchronous accumulator pipeline no longer needs.
      const auto task = *task_shared;
      const auto mblock = task / fc2_ntiles;
      const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto row = a.dispatch.expert_row_prefix[expert] +
                       forward_mblock_local_row(local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      const auto n = n_tile * forward_tuning::kFc2TileN;

      const bool full_tma_tile = valid_rows == forward_tuning::kTileM &&
                                 n + forward_tuning::kFc2TileN <= a.hidden;

      if (full_tma_tile) {
        // Reuse the 64 KiB FC2 input-stage union after its final consumer.
        // STSM converts the accumulator fragments directly to the 128-byte
        // swizzle expected by four 64x128 TMA stores.  Partial expert tails
        // deliberately stay on the checked scalar path: expert prefixes are
        // compact rather than padded, so a 128-row store there would clobber
        // the following expert.
        auto *output_tile = smem_a;
        const auto warp = __shfl_sync(0xffffffffU, threadIdx.x / 32U, 0);
        const auto lane = threadIdx.x & 31U;
#pragma unroll
        for (std::uint32_t index = 0; index < Fc2TileMma::kNumAccum / 4U;
             ++index) {
          constexpr std::uint32_t kBankGroupBytes = 16U;
          const auto atom = index / (64U / 8U);
          const auto in_atom = index % (64U / 8U);
          const auto swizzle_row = lane;
          auto swizzle_column = in_atom;
          swizzle_column ^= swizzle_row % (128U / kBankGroupBytes);
          auto *smem_pointer = reinterpret_cast<std::uint8_t *>(output_tile) +
                               warp * 16U * 128U + atom * 128U * 128U +
                               swizzle_row * 128U +
                               swizzle_column * kBankGroupBytes;
          hopper::StoreMatrixX2<__nv_bfloat162>::copy(
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U], accum[index * 4U + 1U])),
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U + 2U], accum[index * 4U + 3U])),
              smem_pointer);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
        if (threadIdx.x <
            forward_tuning::kFc2TileN / forward_tuning::kFc2StoreAtomN) {
          const auto atom = threadIdx.x;
          cute::SM90_TMA_STORE_2D::copy(
              output_store_map, output_tile + atom * 128U * 64U,
              static_cast<std::uint32_t>(n +
                                         atom * forward_tuning::kFc2StoreAtomN),
              static_cast<std::uint32_t>(row));
          cute::tma_store_arrive();
          // The unqualified wait both protects the shared-memory union from
          // the next task and completes the async-to-generic proxy handoff
          // needed before FC2 publishes its completion counter.
          asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
        }
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);
      }

      // Compact expert tails use bounds-checked stores to avoid overwriting
      // the next expert; full tiles use the TMA epilogue above.
      if (!full_tma_tile) {
        const auto store_fc2 = [&](std::uint32_t wg, std::uint32_t m,
                                   std::uint32_t column, float v0, float v1) {
          const auto output_column = static_cast<std::uint64_t>(n) + column;
          const auto block_m = wg * 64U + m;
          const auto output_row = row + block_m;
          if (block_m < valid_rows && output_column + 1ULL < a.hidden) {
            const auto packed = float2_to_bf16x2_bits(v0, v1);
            reinterpret_cast<std::uint32_t *>(
                forward_output + output_row * a.hidden + output_column)[0] =
                packed;

          } else if (block_m < valid_rows && output_column < a.hidden) {
            const auto value = transport::BFloat16{float_to_bf16_bits(v0)};
            forward_output[output_row * a.hidden + output_column] = value;
          }
        };
        visit_m128_accumulator<Fc2TileMma>(accum, store_fc2);
      }
      // N-tiles synchronize only through this GPU-local completion chain.
      // A device fence is sufficient here; the final N-tile acquires the
      // chain and performs one system fence for the whole M-block below.
      __threadfence();
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            complete(fc1_done[mblock]);
        const auto old =
            complete.fetch_add(1ULL << 16U, ::cuda::memory_order_acq_rel);
        *return_mblock_shared =
            (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
#if WEFTMOE_DEBUG_VALIDATE
        if constexpr (!StaticLocal) {
          if (a.use_gin)
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Fc2);
        }
#endif
      }
      cutlass::arch::NamedBarrier::sync(kMathThreads, kMathDoneBarrier);

      if (threadIdx.x == 0U)
        WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                                execution_epoch(a));

      if (*return_mblock_shared != 0U) {
        // Thread 0's acq_rel RMW above acquired every earlier N-tile release.
        // Publish that transitive chain, plus this CTA's stores, to NVLink/GIN
        // once per M-block before any Return-ready counter becomes visible.
        __threadfence_system();
        forward_return_detail::publish_rows<kMathThreads>(
            a, row, valid_rows, reinterpret_cast<unsigned int *>(smem_a));
        if (threadIdx.x == 0U) {
          const auto finished =
              atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
          if (finished == forward_mblocks) {
            status->fc2_complete_clock = device_globaltimer_ns();
            forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
          }
          WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2MblockReturnReady, Compute, 0U,
                                  mblock, execution_epoch(a));
        }
        cutlass::arch::NamedBarrier::sync(kMathThreads, kMathReadyBarrier);

        // Relay owns node-local Return; compute only publishes descriptors.
      }

#if WEFTMOE_DEBUG_VALIDATE
      if (a.use_gin && threadIdx.x == 0U &&
          blockIdx.x < transport_state::kForwardDebugCtas) {
        completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] =
            (5ULL << 32U) | fc2_pipeline;
        completion->forward_fc2_cta_tiles_completed[blockIdx.x] = task + 1ULL;
      }
#endif
      select_compute_task<StaticLocal>(
          a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
          fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
          pair_operand_shared, pair_round, fc2_streak);
      cutlass::arch::NamedBarrier::sync(kThreads, kTaskDoneBarrier);
    }

    if (*task_kind_shared == kComputeTaskDone)
      break;
  }
#if WEFTMOE_DEVICE_TIMELINE
  if (threadIdx.x == 0U && timeline_fc2_started)
    WEFTMOE_TIMELINE_RECORD(a.timeline, Fc2Last, Compute, 0U, fc2_tasks,
                            execution_epoch(a));
#endif
  cute::cluster_sync();
}

} // namespace forward_compute_detail

#endif // CUTE_ARCH_MMA_SM90A_ENABLED

#include "cuda/persistent_v3/forward/reduce_pipeline.cuh"

// Both entry points use the paired multicast scheduler and TMA epilogue.
// Training additionally retains the BF16 FC1 projections for backward.
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
template <bool StaticLocal, bool RetainBackwardState,
          std::uint32_t Fc1StageCount, std::uint32_t Fc2StageCount>
__device__ __forceinline__ void local_p2p_forward_tma_store_role(
    const ForwardKernelArgs &a, const CUtensorMap &x_map,
    const CUtensorMap &s_map, const CUtensorMap &w1_map,
    const CUtensorMap &w2_map, const CUtensorMap &output_store_map,
    unsigned char *storage, unsigned long long *task_shared,
    unsigned int *task_kind_shared, unsigned int *return_mblock_shared,
    unsigned int *pair_operand_shared) {
  if (threadIdx.x == 0U)
    WEFTMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                            execution_epoch(a));
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        setup_ready(scheduler[8ULL + 2ULL * a.world]);
    while (setup_ready.load(::cuda::memory_order_acquire) <
               execution_epoch(a) &&
           !forward_cancelled(a))
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
  __syncthreads();
  forward_compute_detail::observe_compute_cluster(a);
  forward_compute_detail::persistent_role<StaticLocal, RetainBackwardState,
                                          Fc1StageCount, Fc2StageCount>(
      a, x_map, s_map, w1_map, w2_map, storage, task_shared, task_kind_shared,
      return_mblock_shared, pair_operand_shared, &output_store_map);
  const auto reduce_role_index = static_cast<std::uint32_t>(blockIdx.x);
  const auto reduce_role_count = a.forward_reduce_cta_count != 0U
                                     ? a.forward_reduce_cta_count
                                     : static_cast<std::uint32_t>(gridDim.x);
  if (reduce_role_index < reduce_role_count) {
    forward_compute_detail::rebalance_registers_for_reduce();
    forward_compute_detail::reduce_role<StaticLocal>(a, reduce_role_index,
                                                     reduce_role_count);
  }
  if (threadIdx.x == 0U)
    WEFTMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL,
                            execution_epoch(a));
}
#endif

template <bool StaticLocal,
          std::uint32_t Fc1StageCount = forward_tuning::kFc1Stages,
          std::uint32_t Fc2StageCount = forward_tuning::kFc2Stages>
__global__ __launch_bounds__(
    forward_tuning::kComputeThreads,
    1) void local_p2p_forward_tma_store_kernel(const __grid_constant__
                                                   ForwardKernelArgs a,
                                               const __grid_constant__
                                                   CUtensorMap x_map,
                                               const __grid_constant__
                                                   CUtensorMap s_map,
                                               const __grid_constant__
                                                   CUtensorMap w1_map,
                                               const __grid_constant__
                                                   CUtensorMap w2_map,
                                               const __grid_constant__
                                                   CUtensorMap
                                                       output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_kind_shared;
  __shared__ unsigned int pair_operand_shared;
  __shared__ unsigned int return_mblock_shared;
  local_p2p_forward_tma_store_role<StaticLocal, false, Fc1StageCount,
                                   Fc2StageCount>(
      a, x_map, s_map, w1_map, w2_map, output_store_map, storage, &task_shared,
      &task_kind_shared, &return_mblock_shared, &pair_operand_shared);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
  (void)output_store_map;
#endif
}

// Training has a separate entry symbol so the ordinary inference kernel is
// compiled without checkpoint addressing, branches, or stores. Both entries
// execute the same mixed scheduler and differ only in the compile-time FC1
// epilogue policy.
template <bool StaticLocal,
          std::uint32_t Fc1StageCount = forward_tuning::kFc1Stages,
          std::uint32_t Fc2StageCount = forward_tuning::kFc2Stages>
__global__ __launch_bounds__(
    forward_tuning::kComputeThreads,
    1) void local_p2p_forward_training_tma_store_kernel(const __grid_constant__
                                                            ForwardKernelArgs a,
                                                        const __grid_constant__
                                                            CUtensorMap x_map,
                                                        const __grid_constant__
                                                            CUtensorMap s_map,
                                                        const __grid_constant__
                                                            CUtensorMap w1_map,
                                                        const __grid_constant__
                                                            CUtensorMap w2_map,
                                                        const __grid_constant__
                                                            CUtensorMap
                                                                output_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_kind_shared;
  __shared__ unsigned int pair_operand_shared;
  __shared__ unsigned int return_mblock_shared;
  local_p2p_forward_tma_store_role<StaticLocal, true, Fc1StageCount,
                                   Fc2StageCount>(
      a, x_map, s_map, w1_map, w2_map, output_store_map, storage, &task_shared,
      &task_kind_shared, &return_mblock_shared, &pair_operand_shared);
#else
  (void)a;
  (void)x_map;
  (void)s_map;
  (void)w1_map;
  (void)w2_map;
  (void)output_store_map;
#endif
}
