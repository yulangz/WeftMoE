namespace forward_compute_detail {

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
};

enum ForwardComputePairOperand : std::uint32_t {
  kComputePairIndependent = 0U,
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

__device__ __forceinline__ float gated_swiglu_fc2_input_from_bf16(
    std::uint16_t a_bits, std::uint16_t b_bits, float gate) {
  return gated_swiglu_fc2_input(bf16_bits_to_float(a_bits),
                                bf16_bits_to_float(b_bits), gate);
}

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
__device__ __forceinline__ void visit_m128_accumulator_pair(
    float *accum_a, float *accum_b, Callback callback) {
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
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    kReduceRegisterReleaseBarrier);
  if (warpgroup >= kMathWarpGroups)
    cutlass::arch::warpgroup_reg_alloc<forward_tuning::kReduceRegisters>();
  cutlass::arch::NamedBarrier::sync(kThreads,
                                    kReduceRegisterAcquireBarrier);
}

template <ForwardTaskPartition Partition>
__device__ __forceinline__ unsigned long long partition_mblock_begin(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  if constexpr (Partition == ForwardTaskPartition::Body)
    return *forward_mixed_scheduler_word(a, scheduler, kForwardHeadMblocks);
  return 0ULL;
}

template <ForwardTaskPartition Partition>
__device__ __forceinline__ unsigned long long partition_mblock_count(
    const ForwardKernelArgs &a, unsigned long long *scheduler) {
  const auto total = scheduler[3];
  if constexpr (Partition == ForwardTaskPartition::All)
    return total;
  const auto head =
      *forward_mixed_scheduler_word(a, scheduler, kForwardHeadMblocks);
  if constexpr (Partition == ForwardTaskPartition::Head)
    return head;
  return total - head;
}

// M=64 maps the two resident math warpgroups across adjacent N slices instead
// of leaving the second warpgroup idle.  The callback sees the CTA-wide output
// column (WG0 followed by WG1), while both groups address the same 64 rows.
template <typename TileMma, class Callback>
__device__ __forceinline__ void visit_m64_accumulator(float *accum,
                                                       Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = warpgroup * TileMma::N + (lane % 4U) * 2U +
                        index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

template <typename TileMma, class Callback>
__device__ __forceinline__ void visit_m64_accumulator_pair(
    float *accum_a, float *accum_b, Callback callback) {
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < TileMma::kNumAccum / 4U; ++index) {
    const auto column = warpgroup * TileMma::N + (lane % 4U) * 2U +
                        index * 8U;
    callback(warpgroup, row0, column, accum_a[index * 4U],
             accum_a[index * 4U + 1U], accum_b[index * 4U],
             accum_b[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum_a[index * 4U + 2U],
             accum_a[index * 4U + 3U], accum_b[index * 4U + 2U],
             accum_b[index * 4U + 3U]);
  }
}

template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc1_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long fc1_tasks,
    std::uint32_t fc1_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin =
      partition_mblock_begin<Partition>(a, scheduler) * fc1_ntiles;
  const auto partition_tasks =
      partition_mblock_count<Partition>(a, scheduler) * fc1_ntiles;
  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc1Claim)
          : &scheduler[0];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  auto slot = cursor.load(::cuda::memory_order_relaxed);
  if (slot >= partition_tasks) {
    exhausted = true;
    return false;
  }

  auto selected = partition_begin + slot;
  if (a.use_gin || a.forward_stream_local_dispatch) {
    auto *ready_queue = claims + forward_mblocks;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(ready_queue[partition_begin + slot]);
    const auto encoded = published.load(::cuda::memory_order_acquire);
    if (encoded == 0ULL) {
      exhausted = false;
      return false;
    }
    selected = encoded - 1ULL;
  }

  if (!cursor.compare_exchange_strong(slot, slot + 1ULL,
                                      ::cuda::memory_order_acq_rel,
                                      ::cuda::memory_order_relaxed)) {
    exhausted = false;
    return false;
  }
  task = selected;
  exhausted = false;
#if MEGAMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      claim_completion_tasks(completion, transport_state::TaskClass::Fc1);
      completion->fc1_scheduler_value = slot;
      completion->bulk_local_ready = 11ULL;
    }
  }
#endif
  (void)fc1_tasks;
  return true;
}

// The locality publisher reserves even-sized queue regions. A cluster can
// therefore claim two adjacent entries without crossing an independently
// published group. Depending on the ordering, the pair shares either its A
// activation tile or its expert/N weight tile.
template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc1_task_pair(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long fc1_tasks,
    std::uint32_t fc1_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task0, unsigned long long &task1,
    std::uint32_t &pair_operand, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin =
      partition_mblock_begin<Partition>(a, scheduler) * fc1_ntiles;
  const auto partition_tasks =
      partition_mblock_count<Partition>(a, scheduler) * fc1_ntiles;
  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc1Claim)
          : &scheduler[0];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  auto slot = cursor.load(::cuda::memory_order_relaxed);
  if (slot >= partition_tasks) {
    exhausted = true;
    return false;
  }
  // Cluster mode is admitted only for an even FC1 task extent. Keep this
  // defensive guard so a future shape cannot consume a singleton as a pair.
  if (slot + 1ULL >= partition_tasks) {
    exhausted = false;
    return false;
  }

  auto selected0 = partition_begin + slot;
  auto selected1 = selected0 + 1ULL;
  if (a.use_gin || a.forward_stream_local_dispatch) {
    auto *ready_queue = claims + forward_mblocks;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published0(ready_queue[partition_begin + slot]);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published1(ready_queue[partition_begin + slot + 1ULL]);
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
  pair_operand = kComputePairIndependent;
  if (mblock0 == mblock1) {
    pair_operand = kComputePairShareA;
  } else if (ntile0 == ntile1 &&
             static_cast<std::uint32_t>(claims[mblock0] >> 32U) ==
                 static_cast<std::uint32_t>(claims[mblock1] >> 32U)) {
    pair_operand = kComputePairShareB;
  }
  exhausted = false;
#if MEGAMOE_DEBUG_VALIDATE
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

template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc2_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *ready_queue, unsigned long long fc2_tasks,
    std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin = partition_mblock_begin<Partition>(a, scheduler);
  const auto partition_mblocks =
      partition_mblock_count<Partition>(a, scheduler);

  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc2Claim)
          : &scheduler[1];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  while (true) {
    auto slot = cursor.load(::cuda::memory_order_relaxed);
    if (slot >= partition_mblocks) {
      exhausted = true;
      return false;
    }

    const auto queue_slot = partition_begin + slot;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(ready_queue[queue_slot]);
    const auto encoded = published.load(::cuda::memory_order_acquire);
    if (encoded == 0ULL) {
      exhausted = false;
      return false;
    }

    // One immutable descriptor represents the ready M-block.  A fetch-add on
    // its private counter gives the first fc2_ntiles consumers unique N tiles
    // without making all compute CTAs CAS the same global head for every tile.
    // The first surplus consumer advances the M-block head; nobody reserves
    // work from an unpublished descriptor.
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        tile_cursor(ready_queue[forward_mblocks + queue_slot]);
    const auto n_tile = tile_cursor.fetch_add(
        1ULL, ::cuda::memory_order_relaxed);
    if (n_tile >= fc2_ntiles) {
      auto expected = slot;
      cursor.compare_exchange_strong(expected, slot + 1ULL,
                                     ::cuda::memory_order_relaxed,
                                     ::cuda::memory_order_relaxed);
      continue;
    }

    task = (encoded - 1ULL) * fc2_ntiles + n_tile;
    exhausted = false;
#if MEGAMOE_DEBUG_VALIDATE
    if constexpr (!StaticLocal) {
      if (a.use_gin)
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc2);
    }
    if (a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas) {
      completion->fc2_scheduler_value = slot * fc2_ntiles + n_tile;
      completion->forward_fc2_cta_tiles_started[blockIdx.x] = task + 1ULL;
      completion->forward_fc2_cta_last_task[blockIdx.x] = task;
      completion->forward_fc2_cta_pipeline_iteration[blockIdx.x] = 1ULL << 32U;
    }
#endif
    (void)fc2_tasks;
    return true;
  }
}

template <bool StaticLocal,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ bool try_claim_fc2_task_pair(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *ready_queue, unsigned long long fc2_tasks,
    std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long &task0, unsigned long long &task1,
    std::uint32_t &pair_operand, bool &exhausted) {
  const auto forward_mblocks = scheduler[3];
  const auto partition_begin = partition_mblock_begin<Partition>(a, scheduler);
  const auto partition_mblocks =
      partition_mblock_count<Partition>(a, scheduler);
  auto *claim_word =
      Partition == ForwardTaskPartition::Body
          ? forward_mixed_scheduler_word(a, scheduler, kForwardBodyFc2Claim)
          : &scheduler[1];
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cursor(
      *claim_word);
  while (true) {
    auto slot = cursor.load(::cuda::memory_order_relaxed);
    if (slot >= partition_mblocks) {
      exhausted = true;
      return false;
    }
    const auto queue_slot = partition_begin + slot;
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
#if MEGAMOE_DEBUG_VALIDATE
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc2);
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc2);
      }
    }
    if (a.use_gin && blockIdx.x < transport_state::kForwardDebugCtas) {
      completion->fc2_scheduler_value = slot * fc2_ntiles + n_tile + 1ULL;
      const auto local_task = cute::block_rank_in_cluster() == 0U ? task0
                                                                  : task1;
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
template <bool StaticLocal, bool DoFc1, bool DoFc2,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ void select_priority_compute_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long *fc2_ready_queue,
    unsigned long long fc1_tasks, unsigned long long fc2_tasks,
    std::uint32_t fc1_ntiles, std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long *task_shared, unsigned int *task_kind_shared,
    unsigned int &fc2_streak) {
  if (threadIdx.x != 0U)
    return;
  while (!forward_cancelled(a)) {
    bool fc1_exhausted = !DoFc1;
    bool fc2_exhausted = !DoFc2;
    unsigned long long task = 0ULL;
    const bool fc2_first =
        forward_tuning::kFc2PriorityBurst == 0U ||
        fc2_streak < forward_tuning::kFc2PriorityBurst;
    if constexpr (DoFc2) {
      if (fc2_first &&
          try_claim_fc2_task<StaticLocal, Partition>(
              a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles,
              completion, task, fc2_exhausted)) {
        ++fc2_streak;
        *task_shared = task;
        *task_kind_shared = kComputeTaskFc2;
        return;
      }
    }
    if constexpr (DoFc1) {
      if (try_claim_fc1_task<StaticLocal, Partition>(
              a, scheduler, claims, fc1_tasks, fc1_ntiles, completion, task,
              fc1_exhausted)) {
        fc2_streak = 0U;
        *task_shared = task;
        *task_kind_shared = kComputeTaskFc1;
        return;
      }
    }
    if constexpr (DoFc2) {
      if (!fc2_first &&
          try_claim_fc2_task<StaticLocal, Partition>(
              a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles,
              completion, task, fc2_exhausted)) {
        ++fc2_streak;
        *task_shared = task;
        *task_kind_shared = kComputeTaskFc2;
        return;
      }
    }
    if (fc1_exhausted && fc2_exhausted) {
      *task_kind_shared = kComputeTaskDone;
      return;
    }
    __nanosleep(forward_tuning::kTaskIdleNanosleep);
  }
  *task_kind_shared = kComputeTaskDone;
}

template <bool StaticLocal, bool DoFc1, bool DoFc2,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ void select_priority_compute_task_pair(
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
    std::uint32_t pair_operand = kComputePairIndependent;
    while (!forward_cancelled(a)) {
      bool fc1_exhausted = !DoFc1;
      bool fc2_exhausted = !DoFc2;
      const bool fc2_first =
          forward_tuning::kFc2PriorityBurst == 0U ||
          fc2_streak < forward_tuning::kFc2PriorityBurst;
      if constexpr (DoFc2) {
        if (fc2_first &&
            try_claim_fc2_task_pair<StaticLocal, Partition>(
                a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles,
                completion, task0, task1, pair_operand, fc2_exhausted)) {
          ++fc2_streak;
          task_kind = kComputeTaskFc2;
          break;
        }
      }
      if constexpr (DoFc1) {
        if (try_claim_fc1_task_pair<StaticLocal, Partition>(
                a, scheduler, claims, fc1_tasks, fc1_ntiles, completion,
                task0, task1, pair_operand, fc1_exhausted)) {
          fc2_streak = 0U;
          task_kind = kComputeTaskFc1;
          break;
        }
      }
      if constexpr (DoFc2) {
        if (!fc2_first &&
            try_claim_fc2_task_pair<StaticLocal, Partition>(
                a, scheduler, fc2_ready_queue, fc2_tasks, fc2_ntiles,
                completion, task0, task1, pair_operand, fc2_exhausted)) {
          ++fc2_streak;
          task_kind = kComputeTaskFc2;
          break;
        }
      }
      if (fc1_exhausted && fc2_exhausted)
        break;
      __nanosleep(forward_tuning::kTaskIdleNanosleep);
    }
    mailbox.tasks[0] = task0;
    mailbox.tasks[1] = task1;
    mailbox.metadata = static_cast<unsigned long long>(task_kind) |
                       (static_cast<unsigned long long>(pair_operand) << 32U);
    persistent_v3::store_release<::cuda::thread_scope_device>(
        mailbox.sequence, ready_sequence);
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

template <bool Cluster2, bool StaticLocal, bool DoFc1, bool DoFc2,
          ForwardTaskPartition Partition = ForwardTaskPartition::All>
__device__ __forceinline__ void select_compute_task(
    const ForwardKernelArgs &a, unsigned long long *scheduler,
    unsigned long long *claims, unsigned long long *fc2_ready_queue,
    unsigned long long fc1_tasks, unsigned long long fc2_tasks,
    std::uint32_t fc1_ntiles, std::uint32_t fc2_ntiles,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long *task_shared, unsigned int *task_kind_shared,
    unsigned int *pair_operand_shared, unsigned long long &pair_round,
    unsigned int &fc2_streak) {
  if constexpr (Cluster2) {
    select_priority_compute_task_pair<StaticLocal, DoFc1, DoFc2, Partition>(
        a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
        fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
        pair_operand_shared, pair_round, fc2_streak);
  } else {
    select_priority_compute_task<StaticLocal, DoFc1, DoFc2, Partition>(
        a, scheduler, claims, fc2_ready_queue, fc1_tasks, fc2_tasks,
        fc1_ntiles, fc2_ntiles, completion, task_shared, task_kind_shared,
        fc2_streak);
  }
}
