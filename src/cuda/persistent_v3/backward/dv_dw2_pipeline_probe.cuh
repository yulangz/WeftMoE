#pragma once

// Included by activation_transport.cu after BackwardKernelArgs, the GIN role
// functions and SM90 WGMMA helpers are visible.  The kernel has fixed lifetime
// roles: 8 RDMA, 8 Relay and every remaining resident CTA Compute.  There is
// no grid.sync(); dY M-block release counters and expert U completion counters
// are the only data-flow dependencies.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
constexpr std::uint32_t kBackwardV3TaskKindShift = 61U;
constexpr unsigned long long kBackwardV3TaskKindMask = 7ULL << 61U;
constexpr unsigned long long kBackwardV3DxTaskKind = 1ULL << 61U;
constexpr unsigned long long kBackwardV3Dw1TaskKind = 2ULL << 61U;
constexpr unsigned long long kBackwardV3Dw2TaskKind = 3ULL << 61U;
constexpr unsigned long long kBackwardV3ReduceTaskKind = 4ULL << 61U;
constexpr unsigned long long kBackwardV3TaskPayloadMask = (1ULL << 61U) - 1ULL;
constexpr unsigned long long kBackwardV3TokenClaimedBit = 1ULL << 63U;
constexpr unsigned long long kBackwardV3TokenCompleteBit = 1ULL << 62U;
constexpr unsigned long long kBackwardV3ExpertReadyBit = 1ULL << 63U;
constexpr unsigned long long kBackwardV3IdleTask = ~1ULL;
constexpr unsigned long long kBackwardV3DoneTask = ~0ULL;
constexpr std::uint32_t kBackwardV3DvN = 256U;

// Physical BF16 index used by a 128-byte-swizzled TMA tile. The logical
// coordinates are [outer, inner], with 64 contiguous inner elements per TMA
// atom. This lets the nonlinear dV epilogue feed TMA stores directly instead
// of materializing a linear tile and then issuing strided global stores.
__device__ __forceinline__ std::uint32_t
backward_v3_tma_128b_bf16_index(std::uint32_t outer, std::uint32_t inner) {
  const auto atom = inner / 64U;
  const auto in_atom = inner % 64U;
  const auto bank_group = (in_atom / 8U) ^ (outer % 8U);
  return atom * 128U * 64U + outer * 64U + bank_group * 8U + in_atom % 8U;
}

// A WGMMA row is distributed over four physical lanes, while the historical
// dGate reduction assigned one whole warp to the row. Each physical lane can
// first form the eight virtual-lane partials that it owns. These operations
// then reproduce the old shfl-down 16,8,4,2,1 tree exactly, including FP32
// addition order, without round-tripping dV and S through shared memory.
__device__ __forceinline__ float
backward_v3_reduce_dgate_fragment(const float (&partial)[8]) {
  auto residue0 = __fadd_rn(__fadd_rn(partial[0], partial[4]),
                            __fadd_rn(partial[2], partial[6]));
  auto residue1 = __fadd_rn(__fadd_rn(partial[1], partial[5]),
                            __fadd_rn(partial[3], partial[7]));
  residue0 = __fadd_rn(residue0, __shfl_down_sync(0xffffffffU, residue0, 2U));
  residue1 = __fadd_rn(residue1, __shfl_down_sync(0xffffffffU, residue1, 2U));
  residue0 = __fadd_rn(residue0, __shfl_down_sync(0xffffffffU, residue0, 1U));
  residue1 = __fadd_rn(residue1, __shfl_down_sync(0xffffffffU, residue1, 1U));
  return __fadd_rn(residue0, residue1);
}

__device__ __forceinline__ unsigned long long
backward_v3_expert_ready_value(unsigned long long expected) {
  return expected == 0ULL ? 0ULL : kBackwardV3ExpertReadyBit | expected;
}

// Transport metadata may use a finer M-tile for early packet release.  V3
// compute has one independent, invariant scheduling quantum: a 128-row block
// per expert, including one physically padded tail block.  Never derive the
// compute task count or row extent from dispatch.mtile_*.
__device__ __forceinline__ std::uint64_t
backward_v3_mblock_count(const BackwardKernelArgs &a) {
  std::uint64_t result = 0ULL;
  for (std::uint32_t expert = 0U; expert < a.dispatch.sizes.owned_expert_count;
       ++expert)
    result +=
        persistent_v3::expert_mblocks(a.dispatch.expert_row_count[expert]);
  return result;
}

__device__ __forceinline__ std::uint32_t
backward_v3_mblock_expected_rows(const BackwardKernelArgs &a,
                                 std::uint32_t expert,
                                 std::uint32_t local_mblock) {
  const auto first =
      static_cast<std::uint64_t>(local_mblock) * persistent_v3::kMBlockRows;
  const auto rows = a.dispatch.expert_row_count[expert];
  return first >= rows
             ? 0U
             : static_cast<std::uint32_t>(
                   min(static_cast<std::uint64_t>(persistent_v3::kMBlockRows),
                       rows - first));
}

// Test exact row identities rather than an aggregate arrival count.  The old
// counter could reach 128 when one row was credited twice and another had not
// arrived, releasing a partially stale M block.  Once all row epochs match,
// cache the result in device-local scheduler storage so the 12 dV N tiles do
// not repeat 128 system-scope loads.
__device__ __forceinline__ unsigned long long
backward_v3_load_dy_arrived_rows(const BackwardKernelArgs &a,
                                 std::uint64_t mblock, std::uint32_t expert,
                                 std::uint32_t local_mblock) {
  const auto expected =
      backward_v3_mblock_expected_rows(a, expert, local_mblock);
  auto *ready_cache =
      peer_at<unsigned long long>(a.local_base, a.scheduler_offset) + 64ULL;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> cached(
      ready_cache[mblock]);
  if (cached.load(::cuda::memory_order_acquire) == a.epoch)
    return expected;

  const auto row =
      a.dispatch.expert_row_prefix[expert] +
      static_cast<std::uint64_t>(local_mblock) * persistent_v3::kMBlockRows;
  auto *row_ready =
      peer_at<unsigned long long>(a.local_base, a.dy_arrived_rows_offset);
  unsigned long long arrived = 0ULL;
  std::uint32_t first_missing = expected;
  for (std::uint32_t local_row = 0; local_row < expected; ++local_row) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        row_ready[row + local_row]);
    const auto present = ready.load(::cuda::memory_order_acquire) != 0ULL;
    arrived += present;
    if (!present && a.progress == nullptr)
      return arrived;
    if (!present && first_missing == expected)
      first_missing = local_row;
  }
  if (arrived == expected)
    cached.store(a.epoch, ::cuda::memory_order_release);
  else if (a.progress != nullptr && threadIdx.x == 0U) {
    const auto missing_row = row + first_missing;
    const auto incoming = a.dispatch.incoming_rows[missing_row];
    const auto diagnostic =
        (mblock << 32U) |
        (static_cast<unsigned long long>(first_missing + 1U) << 16U) |
        min(arrived, 0xffffULL);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
        a.progress[kBackwardDyComplete]);
    word.store(diagnostic, ::cuda::memory_order_release);
    const auto source_diagnostic =
        (missing_row << 32U) |
        (static_cast<unsigned long long>(incoming.source_rank) << 24U) |
        (static_cast<unsigned long long>(incoming.expert_id) << 8U) |
        min(first_missing, 0xffU);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        source_word(a.progress[kBackwardSetupComplete]);
    source_word.store(source_diagnostic, ::cuda::memory_order_release);
  }
  return arrived;
}

// Testing-only strong boundary for separating per-M-block publication bugs
// from activation-kernel bugs.  A destination GPU receives peer-global dY
// stores from all eight Relay grids on its node, so waiting only on its own
// grid (the old implementation) was not a full-dispatch wait.  Production
// leaves v3_wait_full_dy_dispatch disabled and continues to consume the exact
// M-block counters above.
__device__ __forceinline__ void
backward_v3_wait_full_dy_dispatch(const BackwardKernelArgs &a) {
  if (!a.v3_wait_full_dy_dispatch)
    return;
  const auto first_rank = a.node * a.local_world_size;
  for (std::uint32_t local_rank = 0; local_rank < a.local_world_size;
       ++local_rank) {
    auto *peer_control = peer_at<persistent_v3::BackwardPipelineControl>(
        a.peer_bases[first_rank + local_rank], a.v3_control_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        local_complete(peer_control->local_route_roles_complete.value);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        inbound_complete(peer_control->inbound_dy_roles_complete.value);
    while (local_complete.load(::cuda::memory_order_acquire) !=
               persistent_v3::kDefaultRelayCtaCount ||
           inbound_complete.load(::cuda::memory_order_acquire) !=
               persistent_v3::kDefaultRelayCtaCount)
      __nanosleep(64U);
  }
}

// Bridge peer-global generic stores into a same-GPU allocation before TMA
// consumes dY.  One CTA wins the per-M-block claim; all other dV tasks for
// that block wait on its device-local release.  The testing-only Forward
// output plane retains the complete snapshot so its final hash can distinguish
// peer-publication errors from activation-core errors.
__device__ __forceinline__ void backward_v3_localize_dy_task(
    const BackwardKernelArgs &a, unsigned long long encoded_task,
    std::uint32_t dv_ntiles, const unsigned long long *mblock_mapping,
    unsigned long long *mblock_localized, unsigned int &owner_shared) {
  if (!a.v3_localize_dy || encoded_task == kBackwardV3IdleTask ||
      (encoded_task & kBackwardV3TaskKindMask) != 0ULL)
    return;

  const auto mblock = encoded_task / dv_ntiles;
  constexpr unsigned long long kClaimedBit = 1ULL << 63U;
  const auto claimed = kClaimedBit | a.epoch;
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> state(
        mblock_localized[mblock]);
    auto expected = 0ULL;
    owner_shared = state.compare_exchange_strong(expected, claimed,
                                                 ::cuda::memory_order_acq_rel,
                                                 ::cuda::memory_order_acquire)
                       ? 1U
                       : 0U;
  }
  __syncthreads();

  if (owner_shared != 0U) {
    const auto mapping = mblock_mapping[mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    const auto expected =
        backward_v3_mblock_expected_rows(a, expert, local_mblock);
    backward_v3_wait_full_dy_dispatch(a);
    // Every copying thread performs the system acquires itself before its
    // peer-global loads.  This avoids assuming that a CTA barrier transfers a
    // remote-memory acquire from one CUDA thread to another.
    while (backward_v3_load_dy_arrived_rows(a, mblock, expert, local_mblock) <
           expected)
      __nanosleep(64U);
    const auto row = a.dispatch.expert_row_prefix[expert] +
                     static_cast<std::uint64_t>(local_mblock) * 128ULL;
    const auto valid_rows = static_cast<std::uint64_t>(
        min(128ULL, a.dispatch.expert_row_count[expert] -
                        static_cast<std::uint64_t>(local_mblock) * 128ULL));
    constexpr std::uint64_t kBf16PerWord =
        sizeof(unsigned long long) / sizeof(transport::BFloat16);
    const auto words_per_row = a.hidden / kBf16PerWord;
    const auto words = valid_rows * words_per_row;
    const auto *source =
        peer_at<const unsigned long long>(a.local_base, a.dy_canonical_offset);
    auto *destination =
        peer_at<unsigned long long>(a.local_base, a.v3_local_dy_offset);
    for (std::uint64_t item = threadIdx.x; item < words; item += blockDim.x) {
      const auto local_row = item / words_per_row;
      const auto word = item % words_per_row;
      destination[(row + local_row) * words_per_row + word] =
          source[(row + local_row) * words_per_row + word];
    }
    __threadfence();
    asm volatile("fence.proxy.async.global;" : : : "memory");
    __syncthreads();
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> state(
          mblock_localized[mblock]);
      state.store(a.epoch, ::cuda::memory_order_release);
    }
  }
  __syncthreads();
  // The TMA issuer as well as both math warp-groups acquire the local copy's
  // publication directly.  This is cheap (one word per dV task) and keeps the
  // proxy transition independent of warp scheduling.
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> state(
      mblock_localized[mblock]);
  while (state.load(::cuda::memory_order_acquire) != a.epoch)
    __nanosleep(64U);
  asm volatile("fence.proxy.async.global;" : : : "memory");
  __syncthreads();
}

__device__ __forceinline__ unsigned long long
backward_v3_try_claim_dv_task(const BackwardKernelArgs &a,
                              persistent_v3::BackwardPipelineControl &control,
                              const unsigned long long *mblock_mapping,
                              unsigned long long *mblock_dv_cursor,
                              std::uint32_t dv_ntiles) {
  if (control.mblocks == 0ULL)
    return kBackwardV3IdleTask;
  const auto attempts = control.mblocks < 16ULL ? control.mblocks : 16ULL;
  for (std::uint64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto candidate =
        atomicAdd(&control.dv_scan_cursor.value, 1ULL) % control.mblocks;
    const auto mapping = mblock_mapping[candidate];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    const auto expected =
        backward_v3_mblock_expected_rows(a, expert, local_mblock);
    if (backward_v3_load_dy_arrived_rows(a, candidate, expert, local_mblock) <
        expected)
      continue;
    unsigned long long ntile = 0ULL;
    if (persistent_v3::bounded_ticket_claim(mblock_dv_cursor[candidate],
                                            dv_ntiles, ntile))
      return candidate * dv_ntiles + ntile;
  }
  return kBackwardV3IdleTask;
}

__device__ __forceinline__ unsigned long long
backward_v3_try_claim_dx_task(const BackwardKernelArgs &a,
                              persistent_v3::BackwardPipelineControl &control,
                              unsigned long long *mblock_ready,
                              unsigned long long *mblock_dx_cursor,
                              std::uint32_t dx_ntiles) {
  if (control.mblocks == 0ULL)
    return kBackwardV3IdleTask;
  const auto attempts = control.mblocks < 16ULL ? control.mblocks : 16ULL;
  for (std::uint64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto candidate =
        atomicAdd(&control.dx_scan_cursor.value, 1ULL) % control.mblocks;
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            mblock_ready[candidate]) != a.epoch)
      continue;
    unsigned long long ntile = 0ULL;
    if (persistent_v3::bounded_ticket_claim(mblock_dx_cursor[candidate],
                                            dx_ntiles, ntile))
      return kBackwardV3DxTaskKind | candidate * dx_ntiles + ntile;
  }
  return kBackwardV3IdleTask;
}

// Return readiness is published at 128-row M-block granularity, not at the
// individual 128x256 dX tile granularity.  Let one scalar activation CTA own
// all N tiles of a ready M block so completed rows become visible to the
// put/relay pipeline early.  The old per-tile round-robin claimant spread the
// 28 main-shape N tiles over the whole M-block population, leaving nearly
// every ReturnChunk one missing tile until the global dX tail.
__device__ __forceinline__ unsigned long long
backward_v3_try_claim_dx_mblock_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    unsigned long long *mblock_ready, unsigned long long *mblock_dx_cursor,
    std::uint32_t dx_ntiles, const unsigned long long *dx_mblock_order,
    unsigned long long &owned_mblock, std::uint32_t &next_ntile) {
  if (owned_mblock != kBackwardV3IdleTask) {
    const auto task =
        kBackwardV3DxTaskKind | owned_mblock * dx_ntiles + next_ntile;
    ++next_ntile;
    if (next_ntile == dx_ntiles) {
      owned_mblock = kBackwardV3IdleTask;
      next_ntile = 0U;
    }
    return task;
  }
  if (control.mblocks == 0ULL)
    return kBackwardV3IdleTask;
  const auto attempts = control.mblocks < 16ULL ? control.mblocks : 16ULL;
  for (std::uint64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto priority =
        atomicAdd(&control.dx_scan_cursor.value, 1ULL) % control.mblocks;
    const auto candidate = dx_mblock_order[priority];
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            mblock_ready[candidate]) != a.epoch)
      continue;
    unsigned long long first_ntile = 0ULL;
    if (!persistent_v3::bounded_ticket_claim_span(
            mblock_dx_cursor[candidate], dx_ntiles, dx_ntiles, first_ntile))
      continue;
    owned_mblock = candidate;
    next_ntile = 1U;
    if (dx_ntiles == 1U) {
      owned_mblock = kBackwardV3IdleTask;
      next_ntile = 0U;
    }
    return kBackwardV3DxTaskKind | candidate * dx_ntiles + first_ntile;
  }
  return kBackwardV3IdleTask;
}

__device__ __forceinline__ bool backward_v3_try_claim_dv_pair(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    const unsigned long long *mblock_mapping,
    unsigned long long *mblock_dv_cursor, std::uint32_t dv_ntiles,
    unsigned long long &task0, unsigned long long &task1) {
  if (control.mblocks == 0ULL)
    return false;
  const auto attempts = control.mblocks < 16ULL ? control.mblocks : 16ULL;
  for (std::uint64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto candidate =
        atomicAdd(&control.dv_scan_cursor.value, 1ULL) % control.mblocks;
    const auto mapping = mblock_mapping[candidate];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    const auto expected =
        backward_v3_mblock_expected_rows(a, expert, local_mblock);
    if (backward_v3_load_dy_arrived_rows(a, candidate, expert, local_mblock) <
        expected)
      continue;
    unsigned long long first_ntile = 0ULL;
    if (persistent_v3::bounded_ticket_claim_span(
            mblock_dv_cursor[candidate], dv_ntiles, 2ULL, first_ntile)) {
      task0 = candidate * dv_ntiles + first_ntile;
      task1 = task0 + 1ULL;
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ bool backward_v3_try_claim_dx_pair(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    unsigned long long *mblock_ready, unsigned long long *mblock_dx_cursor,
    std::uint32_t dx_ntiles, unsigned long long &task0,
    unsigned long long &task1) {
  if (control.mblocks == 0ULL)
    return false;
  const auto attempts = control.mblocks < 16ULL ? control.mblocks : 16ULL;
  for (std::uint64_t attempt = 0; attempt < attempts; ++attempt) {
    const auto candidate =
        atomicAdd(&control.dx_scan_cursor.value, 1ULL) % control.mblocks;
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            mblock_ready[candidate]) != a.epoch)
      continue;
    unsigned long long first_ntile = 0ULL;
    if (persistent_v3::bounded_ticket_claim_span(
            mblock_dx_cursor[candidate], dx_ntiles, 2ULL, first_ntile)) {
      task0 = kBackwardV3DxTaskKind | (candidate * dx_ntiles + first_ntile);
      task1 = task0 + 1ULL;
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ unsigned long long
backward_v3_try_claim_dw1_task(const BackwardKernelArgs &a,
                               persistent_v3::BackwardPipelineControl &control,
                               unsigned long long *expert_dv_complete,
                               unsigned long long *expert_dw1_cursor,
                               std::uint32_t dv_ntiles,
                               std::uint64_t per_expert) {
  const auto experts = a.dispatch.sizes.owned_expert_count;
  if (experts == 0U)
    return kBackwardV3IdleTask;
  const auto base = atomicAdd(&control.dw1_scan_cursor.value, 1ULL);
  const auto attempts = experts;
  for (std::uint32_t attempt = 0; attempt < attempts; ++attempt) {
    const auto expert = static_cast<std::uint32_t>((base + attempt) % experts);
    const auto expected =
        persistent_v3::expert_mblocks(a.dispatch.expert_row_count[expert]) *
        dv_ntiles;
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            expert_dv_complete[expert]) !=
        backward_v3_expert_ready_value(expected))
      continue;
    unsigned long long local_task = 0ULL;
    if (persistent_v3::bounded_ticket_claim(expert_dw1_cursor[expert],
                                            per_expert, local_task))
      return kBackwardV3Dw1TaskKind |
             (static_cast<unsigned long long>(expert) * per_expert +
              local_task);
  }
  return kBackwardV3IdleTask;
}

__device__ __forceinline__ unsigned long long
backward_v3_try_claim_dw2_task(const BackwardKernelArgs &a,
                               persistent_v3::BackwardPipelineControl &control,
                               unsigned long long *expert_dv_complete,
                               unsigned long long *expert_dw2_cursor,
                               std::uint32_t dv_ntiles,
                               std::uint64_t per_expert) {
  const auto experts = a.dispatch.sizes.owned_expert_count;
  if (experts == 0U)
    return kBackwardV3IdleTask;
  const auto base = atomicAdd(&control.dw2_scan_cursor.value, 1ULL);
  const auto attempts = experts;
  for (std::uint32_t attempt = 0; attempt < attempts; ++attempt) {
    const auto expert = static_cast<std::uint32_t>((base + attempt) % experts);
    const auto expected =
        persistent_v3::expert_mblocks(a.dispatch.expert_row_count[expert]) *
        dv_ntiles;
    if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
            expert_dv_complete[expert]) !=
        backward_v3_expert_ready_value(expected))
      continue;
    unsigned long long local_task = 0ULL;
    if (persistent_v3::bounded_ticket_claim(expert_dw2_cursor[expert],
                                            per_expert, local_task))
      return kBackwardV3Dw2TaskKind |
             (static_cast<unsigned long long>(expert) * per_expert +
              local_task);
  }
  return kBackwardV3IdleTask;
}

__device__ __forceinline__ unsigned long long backward_v3_try_claim_reduce_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    unsigned long long *token_state) {
  const auto tokens = a.dispatch.identity.num_tokens;
  if (tokens == 0ULL)
    return kBackwardV3IdleTask;
  constexpr std::uint32_t kAttempts = 16U;
  for (std::uint32_t attempt = 0; attempt < kAttempts; ++attempt) {
    const auto token =
        atomicAdd(&control.reduce_scan_cursor.value, 1ULL) % tokens;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> state(
        token_state[token]);
    auto expected = static_cast<unsigned long long>(a.dispatch.identity.top_k);
    if (state.compare_exchange_strong(
            expected, kBackwardV3TokenClaimedBit | expected,
            ::cuda::memory_order_acq_rel, ::cuda::memory_order_acquire))
      return kBackwardV3ReduceTaskKind | token;
  }
  return kBackwardV3IdleTask;
}

// A Reduce task owns one token.  All twelve warps split the hidden vectors;
// each lane performs the same ordered top-k FP32 accumulation as the legacy
// oracle, so the final BF16 output stays bit-for-bit deterministic while the
// task is small enough to fill dW bubbles.
template <bool DirectInputsAcquired = false>
__device__ __forceinline__ void
backward_v3_reduce_token(const BackwardKernelArgs &a, std::uint64_t token,
                         const transport::BFloat16 *dx_return,
                         transport::BFloat16 *dx_combined) {
  const auto *route_compact_index = peer_at<const unsigned long long>(
      a.local_base, a.v3_return_route_compact_index_offset);
  const auto *return_receive = reinterpret_cast<const transport::BFloat16 *>(
      static_cast<const std::byte *>(a.gin_window_base) +
      a.gin_return_receive_offset);
  if constexpr (!DirectInputsAcquired) {
    if (a.v3_direct_return_consume) {
      // The claimant's acquire does not transfer through a CTA barrier to
      // the threads that load the put-delivered payload. Every consumer first
      // acquires this token's release sequence and crosses Hopper's
      // async->generic proxy boundary itself.
      auto *token_state =
          peer_at<unsigned long long>(a.local_base, a.v3_token_state_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          token_state[token]);
      (void)ready.load(::cuda::memory_order_acquire);
      asm volatile("fence.proxy.async.global;" : : : "memory");
    }
  }
  const auto vector_count = a.hidden / 8ULL;
  auto *output = reinterpret_cast<uint4 *>(dx_combined + token * a.hidden);
  for (std::uint64_t vector = threadIdx.x; vector < vector_count;
       vector += blockDim.x) {
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t order = 0; order < a.dispatch.identity.top_k; ++order) {
      const auto slot =
          a.dispatch.combine_order[token * a.dispatch.identity.top_k + order];
      const auto route = token * a.dispatch.identity.top_k + slot;
      const auto compact =
          a.v3_direct_return_consume ? route_compact_index[route] : 0ULL;
      const auto *route_values =
          compact == 0ULL ? dx_return + route * a.hidden
                          : return_receive + (compact - 1ULL) * a.hidden;
      const auto packed = reinterpret_cast<const uint4 *>(route_values)[vector];
      const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (std::uint32_t word = 0; word < 4U; ++word) {
        sums[2U * word] = __fadd_rn(
            sums[2U * word],
            bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
        sums[2U * word + 1U] = __fadd_rn(
            sums[2U * word + 1U],
            bf16_bits_to_float(static_cast<std::uint16_t>(words[word] >> 16U)));
      }
    }
    unsigned int words[4];
#pragma unroll
    for (std::uint32_t word = 0; word < 4U; ++word)
      words[word] =
          static_cast<unsigned int>(float_to_bf16_bits(sums[2U * word])) |
          (static_cast<unsigned int>(float_to_bf16_bits(sums[2U * word + 1U]))
           << 16U);
    output[vector] = make_uint4(words[0], words[1], words[2], words[3]);
  }
}

__device__ __forceinline__ unsigned long long backward_v3_try_claim_weight_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    unsigned long long *expert_dv_complete,
    unsigned long long *expert_dw1_cursor,
    unsigned long long *expert_dw2_cursor, std::uint32_t dv_ntiles,
    std::uint64_t dw1_per_expert, std::uint64_t dw2_per_expert,
    bool dw1_pending, bool dw2_pending, std::uint32_t weight_phase) {
  const bool prefer_dw1 = weight_phase % 3U != 2U;
  if (prefer_dw1 && dw1_pending) {
    const auto task = backward_v3_try_claim_dw1_task(
        a, control, expert_dv_complete, expert_dw1_cursor, dv_ntiles,
        dw1_per_expert);
    if (task != kBackwardV3IdleTask)
      return task;
  }
  if (dw2_pending) {
    const auto task = backward_v3_try_claim_dw2_task(
        a, control, expert_dv_complete, expert_dw2_cursor, dv_ntiles,
        dw2_per_expert);
    if (task != kBackwardV3IdleTask)
      return task;
  }
  if (!prefer_dw1 && dw1_pending)
    return backward_v3_try_claim_dw1_task(a, control, expert_dv_complete,
                                          expert_dw1_cursor, dv_ntiles,
                                          dw1_per_expert);
  return kBackwardV3IdleTask;
}

// Production activation CTAs never steal dW work.  Keeping their completion
// condition independent lets the return-critical dV -> dX -> Reduce chain
// retire while a compile-isolated weight grid drains dW concurrently.
__device__ __forceinline__ unsigned long long backward_v3_claim_activation_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    const unsigned long long *mblock_mapping,
    unsigned long long *mblock_dv_cursor, unsigned long long *mblock_ready,
    unsigned long long *mblock_dx_cursor,
    const unsigned long long *dx_mblock_order, unsigned long long *token_state,
    std::uint32_t dv_ntiles, std::uint32_t dx_ntiles,
    std::uint32_t activation_phase, std::uint32_t compute_ordinal,
    std::uint32_t reduce_ctas, unsigned long long &owned_dx_mblock,
    std::uint32_t &next_dx_ntile) {
  if (backward_v3_generation_cancelled(a, control))
    return kBackwardV3DoneTask;
  const auto dv_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dv_tasks_complete.value);
  const auto dx_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dx_tasks_complete.value);
  const auto reduce_done =
      persistent_v3::load_acquire<::cuda::thread_scope_device>(
          control.reduce_tasks_complete.value);
  const bool activation_math_done =
      dv_done == control.dv_tasks && dx_done == control.dx_tasks;
  if (activation_math_done &&
      (reduce_done == control.reduce_tasks || compute_ordinal >= reduce_ctas))
    return kBackwardV3DoneTask;

  const bool dv_pending = dv_done != control.dv_tasks;
  const bool dx_pending = dx_done != control.dx_tasks;
  const auto activation_period = a.v3_dv_phase_slots + a.v3_dx_phase_slots;
  const bool prefer_dv =
      activation_phase % activation_period < a.v3_dv_phase_slots;
  if (prefer_dv && dv_pending) {
    const auto task = backward_v3_try_claim_dv_task(
        a, control, mblock_mapping, mblock_dv_cursor, dv_ntiles);
    if (task != kBackwardV3IdleTask)
      return task;
  } else if (!prefer_dv && dx_pending) {
    const auto task = backward_v3_try_claim_dx_mblock_task(
        a, control, mblock_ready, mblock_dx_cursor, dx_ntiles, dx_mblock_order,
        owned_dx_mblock, next_dx_ntile);
    if (task != kBackwardV3IdleTask)
      return task;
  }
  if (dv_pending && !prefer_dv) {
    const auto task = backward_v3_try_claim_dv_task(
        a, control, mblock_mapping, mblock_dv_cursor, dv_ntiles);
    if (task != kBackwardV3IdleTask)
      return task;
  }
  if (dx_pending && prefer_dv) {
    const auto task = backward_v3_try_claim_dx_mblock_task(
        a, control, mblock_ready, mblock_dx_cursor, dx_ntiles, dx_mblock_order,
        owned_dx_mblock, next_dx_ntile);
    if (task != kBackwardV3IdleTask)
      return task;
  }
  if (reduce_done != control.reduce_tasks && compute_ordinal < reduce_ctas)
    return backward_v3_try_claim_reduce_task(a, control, token_state);
  return kBackwardV3IdleTask;
}

// Cluster-paired activation scheduling is deliberately separate from the
// scalar claimant above.  One CAS reserves two adjacent N tiles from a single
// M block, which is the invariant required for sharing the A operand with a
// two-CTA TMA multicast.  Main-shape dV and dX have 12 and 28 N tiles, so no
// singleton tail is possible.
__device__ __forceinline__ void backward_v3_claim_activation_pair(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    const unsigned long long *mblock_mapping,
    unsigned long long *mblock_dv_cursor, unsigned long long *mblock_ready,
    unsigned long long *mblock_dx_cursor, unsigned long long *token_state,
    std::uint32_t dv_ntiles, std::uint32_t dx_ntiles,
    std::uint32_t activation_phase, std::uint32_t compute_ordinal,
    std::uint32_t reduce_ctas, unsigned long long &task0,
    unsigned long long &task1) {
  task0 = kBackwardV3IdleTask;
  task1 = kBackwardV3IdleTask;
  if (backward_v3_generation_cancelled(a, control)) {
    task0 = kBackwardV3DoneTask;
    task1 = kBackwardV3DoneTask;
    return;
  }
  const auto dv_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dv_tasks_complete.value);
  const auto dx_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dx_tasks_complete.value);
  const auto reduce_done =
      persistent_v3::load_acquire<::cuda::thread_scope_device>(
          control.reduce_tasks_complete.value);
  const bool activation_math_done =
      dv_done == control.dv_tasks && dx_done == control.dx_tasks;
  if (activation_math_done &&
      (reduce_done == control.reduce_tasks || compute_ordinal >= reduce_ctas)) {
    task0 = kBackwardV3DoneTask;
    task1 = kBackwardV3DoneTask;
    return;
  }

  const bool dv_pending = dv_done != control.dv_tasks;
  const bool dx_pending = dx_done != control.dx_tasks;
  const auto activation_period = a.v3_dv_phase_slots + a.v3_dx_phase_slots;
  const bool prefer_dv =
      activation_phase % activation_period < a.v3_dv_phase_slots;
  if (prefer_dv && dv_pending &&
      backward_v3_try_claim_dv_pair(a, control, mblock_mapping,
                                    mblock_dv_cursor, dv_ntiles, task0, task1))
    return;
  if (!prefer_dv && dx_pending &&
      backward_v3_try_claim_dx_pair(a, control, mblock_ready, mblock_dx_cursor,
                                    dx_ntiles, task0, task1))
    return;
  if (dv_pending && !prefer_dv &&
      backward_v3_try_claim_dv_pair(a, control, mblock_mapping,
                                    mblock_dv_cursor, dv_ntiles, task0, task1))
    return;
  if (dx_pending && prefer_dv &&
      backward_v3_try_claim_dx_pair(a, control, mblock_ready, mblock_dx_cursor,
                                    dx_ntiles, task0, task1))
    return;
  if (reduce_done != control.reduce_tasks && compute_ordinal < reduce_ctas) {
    task0 = backward_v3_try_claim_reduce_task(a, control, token_state);
    task1 = backward_v3_try_claim_reduce_task(a, control, token_state);
  }
}

__device__ __forceinline__ unsigned long long
backward_v3_publish_or_wait_activation_pair(
    persistent_v3::BackwardActivationPairMailbox &mailbox,
    std::uint32_t cluster_rank, unsigned long long round,
    unsigned long long task0, unsigned long long task1) {
  const auto ready_sequence = round * 2ULL + 1ULL;
  if (cluster_rank == 0U) {
    const auto reusable_sequence = round * 2ULL;
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               mailbox.sequence) != reusable_sequence)
      __nanosleep(64U);
    mailbox.tasks[0] = task0;
    mailbox.tasks[1] = task1;
    persistent_v3::store_release<::cuda::thread_scope_device>(mailbox.sequence,
                                                              ready_sequence);
    return task0;
  }
  while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
             mailbox.sequence) != ready_sequence)
    __nanosleep(64U);
  return mailbox.tasks[1];
}

__device__ __forceinline__ void backward_v3_finish_activation_pair(
    persistent_v3::BackwardActivationPairMailbox &mailbox,
    std::uint32_t cluster_rank, unsigned long long round) {
  const auto complete_sequence = (round + 1ULL) * 2ULL;
  if (cluster_rank == 1U) {
    persistent_v3::store_release<::cuda::thread_scope_device>(
        mailbox.sequence, complete_sequence);
    return;
  }
  while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
             mailbox.sequence) != complete_sequence)
    __nanosleep(64U);
}

__device__ __forceinline__ unsigned long long backward_v3_claim_compute_task(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    const unsigned long long *mblock_mapping,
    unsigned long long *mblock_dv_cursor, unsigned long long *mblock_ready,
    unsigned long long *mblock_dx_cursor, unsigned long long *token_state,
    unsigned long long *expert_dv_complete,
    unsigned long long *expert_dw1_cursor,
    unsigned long long *expert_dw2_cursor, std::uint32_t dv_ntiles,
    std::uint32_t dx_ntiles, std::uint64_t dw1_per_expert,
    std::uint64_t dw2_per_expert, bool prefer_dw_role,
    std::uint32_t activation_phase, std::uint32_t weight_phase) {
  if (backward_v3_generation_cancelled(a, control))
    return kBackwardV3DoneTask;
  const auto dv_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dv_tasks_complete.value);
  const auto dw_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dw2_tasks_complete.value);
  const auto dw1_done =
      persistent_v3::load_acquire<::cuda::thread_scope_device>(
          control.dw1_tasks_complete.value);
  const auto dx_done = persistent_v3::load_acquire<::cuda::thread_scope_device>(
      control.dx_tasks_complete.value);
  const auto reduce_done =
      persistent_v3::load_acquire<::cuda::thread_scope_device>(
          control.reduce_tasks_complete.value);
  if (dv_done == control.dv_tasks && dx_done == control.dx_tasks &&
      dw1_done == control.dw1_tasks && dw_done == control.dw2_tasks &&
      reduce_done == control.reduce_tasks)
    return kBackwardV3DoneTask;

  // Once a task class is complete, do not keep probing all of its already
  // exhausted per-object cursors.  In particular, the old alternating policy
  // performed sixteen contended dv_scan_cursor atomics before every other dW2
  // tile throughout the dW2-only tail.
  const bool dv_pending = dv_done != control.dv_tasks;
  const bool dx_pending = dx_done != control.dx_tasks;
  const bool dw1_pending = dw1_done != control.dw1_tasks;
  const bool dw_pending = dw_done != control.dw2_tasks;
  const bool reduce_pending = reduce_done != control.reduce_tasks;
  // Keep most CTAs sticky on the return-critical activation chain and a small
  // cohort sticky on weight gradients. This avoids changing the WGMMA body on
  // every tile while preserving work stealing whenever a preferred stream is
  // temporarily empty. dV:dX activation slots are 1:2, matching task volume
  // and their similar per-tile math.
  if (prefer_dw_role && (dw1_pending || dw_pending)) {
    const auto dw = backward_v3_try_claim_weight_task(
        a, control, expert_dv_complete, expert_dw1_cursor, expert_dw2_cursor,
        dv_ntiles, dw1_per_expert, dw2_per_expert, dw1_pending, dw_pending,
        weight_phase);
    if (dw != kBackwardV3IdleTask)
      return dw;
  }
  const auto activation_period = a.v3_dv_phase_slots + a.v3_dx_phase_slots;
  const bool prefer_dv =
      activation_phase % activation_period < a.v3_dv_phase_slots;
  if (prefer_dv && dv_pending) {
    const auto dv = backward_v3_try_claim_dv_task(a, control, mblock_mapping,
                                                  mblock_dv_cursor, dv_ntiles);
    if (dv != kBackwardV3IdleTask)
      return dv;
  } else if (!prefer_dv && dx_pending) {
    const auto dx = backward_v3_try_claim_dx_task(a, control, mblock_ready,
                                                  mblock_dx_cursor, dx_ntiles);
    if (dx != kBackwardV3IdleTask)
      return dx;
  }
  if (dv_pending && !prefer_dv) {
    const auto dv = backward_v3_try_claim_dv_task(a, control, mblock_mapping,
                                                  mblock_dv_cursor, dv_ntiles);
    if (dv != kBackwardV3IdleTask)
      return dv;
  }
  if (dx_pending && prefer_dv) {
    const auto dx = backward_v3_try_claim_dx_task(a, control, mblock_ready,
                                                  mblock_dx_cursor, dx_ntiles);
    if (dx != kBackwardV3IdleTask)
      return dx;
  }
  if (reduce_pending) {
    const auto reduce =
        backward_v3_try_claim_reduce_task(a, control, token_state);
    if (reduce != kBackwardV3IdleTask)
      return reduce;
  }
  if ((dw1_pending || dw_pending) && !prefer_dw_role)
    return backward_v3_try_claim_weight_task(
        a, control, expert_dv_complete, expert_dw1_cursor, expert_dw2_cursor,
        dv_ntiles, dw1_per_expert, dw2_per_expert, dw1_pending, dw_pending,
        weight_phase);
  return kBackwardV3IdleTask;
}

template <bool ActivationOnly = false, bool ActivationCluster2 = false>
__device__ __forceinline__ void backward_v3_dv_dw2_compute_role_inline(
    const BackwardKernelArgs &a, const CUtensorMap &dy_map,
    const CUtensorMap &w2_transpose_map, const CUtensorMap &dab_dx_map,
    const CUtensorMap &w1_transpose_map, const CUtensorMap &x_mn_map,
    const CUtensorMap &dab_transpose_map, const CUtensorMap &dy_mn_map,
    const CUtensorMap &u_transpose_map, const CUtensorMap &dw1_store_map,
    const CUtensorMap &dw2_store_map, unsigned char *storage,
    std::uint32_t compute_ordinal, std::uint32_t compute_ctas,
    std::uint32_t reduce_ctas) {
  static_assert(ForwardMma::kNumAccum == 64);
  static_assert(!ActivationCluster2 || ActivationOnly);
  // Keep four stages for both scalar and cluster-2 paths.  The activation
  // task scheduler restarts each task at stage zero, and all supported K
  // extents are exact multiples of four K64 iterations.
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 128U * 64U;
  constexpr std::uint32_t kDvEpilogueElements = 6U * 128U * 128U;
  constexpr std::uint64_t kDwN = 256ULL;
  constexpr std::uint32_t kTaskReadyBarrier = 0U;
  constexpr std::uint32_t kTaskDoneBarrier = 1U;
  constexpr std::uint32_t kMathBarrier = 2U;
  constexpr std::uint32_t kReduceBarrier = 3U;
  constexpr std::uint32_t kReduceClaimBarrier = 4U;
  // dX writes one scattered plane through system/async proxy fences; a
  // smaller writer cohort wins there by reducing fence warps.
  constexpr std::uint32_t kDxEpilogueThreads = 128U;

  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *mblock_mapping =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_mapping_offset);
  auto *activation_pair_mailboxes =
      peer_at<persistent_v3::BackwardActivationPairMailbox>(
          a.local_base, a.v3_activation_pair_mailboxes_offset);
  auto *mblock_dv_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dv_cursor_offset);
  auto *mblock_ready =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_ready_offset);
  auto *mblock_dx_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dx_cursor_offset);
  auto *mblock_localized =
      peer_at<unsigned long long>(a.local_base, a.scheduler_offset) + 64ULL +
      backward_v3_mblock_count(a);
  const auto *dx_mblock_order = mblock_localized + backward_v3_mblock_count(a);
  auto *token_state =
      peer_at<unsigned long long>(a.local_base, a.v3_token_state_offset);
  auto *expert_dv_complete =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dv_complete_offset);
  auto *expert_dw1_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dw1_cursor_offset);
  auto *expert_dw2_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dw2_cursor_offset);
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = smem_a + kStages * kAStageElements;
  auto *smem_b2 = smem_b + kStages * kBStageElements;
  auto *output_tile = smem_a;
  // Two sets of three swizzled 128x128 output planes form a lifetime union
  // with the four-stage pipeline.  This lets half-0's async TMA stores fly
  // while the math threads construct half-1 in disjoint shared storage.
  constexpr auto kPipelineElements = 3U * kStages * kAStageElements;
  constexpr auto kSharedUnionElements = kDvEpilogueElements > kPipelineElements
                                            ? kDvEpilogueElements
                                            : kPipelineElements;
  auto *barrier_storage = smem_a + kSharedUnionElements;
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(barrier_storage);
  auto *empty_barriers = full_barriers + kStages;
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int task_flags_shared;
  __shared__ unsigned int dy_localize_owner_shared;
  __shared__ unsigned long long owned_dx_mblock_shared;
  __shared__ unsigned int next_dx_ntile_shared;
  __shared__ unsigned long long reduce_tokens_shared[12];
  __shared__ unsigned int reduce_count_shared;
  __shared__ float dv_gate_shared[128];

  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < kStages; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(ActivationCluster2 ? 16 : 8);
    }
    cutlass::arch::fence_barrier_init();
  }
  if (threadIdx.x == 0U) {
    owned_dx_mblock_shared = kBackwardV3IdleTask;
    next_dx_ntile_shared = 0U;
  }
  if constexpr (ActivationCluster2)
    cute::cluster_sync();
  const auto mblocks = control->mblocks;
  const auto dv_ntiles = static_cast<std::uint32_t>(
      (a.intermediate + kBackwardV3DvN - 1ULL) / kBackwardV3DvN);
  const auto dv_tasks = mblocks * dv_ntiles;
  const auto dx_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto dx_tasks = mblocks * dx_ntiles;
  const auto dw1_mtiles = (2ULL * a.intermediate + 127ULL) / 128ULL;
  const auto dw1_ntiles = (a.hidden + kDwN - 1ULL) / kDwN;
  const auto dw1_per_expert = dw1_mtiles * dw1_ntiles;
  const auto dw1_tasks =
      static_cast<unsigned long long>(a.dispatch.sizes.owned_expert_count) *
      dw1_per_expert;
  const auto dw2_mtiles = (a.hidden + 127ULL) / 128ULL;
  const auto dw2_ntiles = (a.intermediate + kDwN - 1ULL) / kDwN;
  const auto dw2_per_expert = dw2_mtiles * dw2_ntiles;
  const auto dw2_tasks =
      static_cast<unsigned long long>(a.dispatch.sizes.owned_expert_count) *
      dw2_per_expert;
  const bool prefer_dw_role = compute_ordinal % a.v3_weight_cta_divisor == 0U;
  unsigned long long activation_pair_round = 0ULL;
  if (threadIdx.x == 0U) {
    if constexpr (ActivationCluster2) {
      const auto cluster_rank = cute::block_rank_in_cluster();
      auto &mailbox = activation_pair_mailboxes[compute_ordinal / 2U];
      unsigned long long task0 = kBackwardV3IdleTask;
      unsigned long long task1 = kBackwardV3IdleTask;
      if (cluster_rank == 0U)
        backward_v3_claim_activation_pair(
            a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
            mblock_dx_cursor, token_state, dv_ntiles, dx_ntiles, 0U,
            compute_ordinal, reduce_ctas, task0, task1);
      task_shared = backward_v3_publish_or_wait_activation_pair(
          mailbox, cluster_rank, activation_pair_round, task0, task1);
    } else if constexpr (ActivationOnly) {
      task_shared = backward_v3_claim_activation_task(
          a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
          mblock_dx_cursor, dx_mblock_order, token_state, dv_ntiles, dx_ntiles,
          0U, compute_ordinal, reduce_ctas, owned_dx_mblock_shared,
          next_dx_ntile_shared);
    } else {
      task_shared = backward_v3_claim_compute_task(
          a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
          mblock_dx_cursor, token_state, expert_dv_complete, expert_dw1_cursor,
          expert_dw2_cursor, dv_ntiles, dx_ntiles, dw1_per_expert,
          dw2_per_expert, prefer_dw_role, 0U, 0U);
    }
  }
  __syncthreads();

  // Match DeepGEMM's uniform-register idiom.  NVCC can otherwise carry the
  // lane-derived quotient as a divergent value through the large activation
  // role and insert warp-group dependencies around WGMMA.
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0U);
  if (warpgroup >= 2U) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    std::uint32_t pipeline = 0U;
    std::uint32_t activation_stage_epochs[kStages] = {0U, 0U, 0U, 0U};
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      const auto encoded_task = task_shared;
      if (encoded_task == kBackwardV3DoneTask)
        break;
      const auto task_kind = encoded_task & kBackwardV3TaskKindMask;
      backward_v3_localize_dy_task(a, encoded_task, dv_ntiles, mblock_mapping,
                                   mblock_localized, dy_localize_owner_shared);
      if (encoded_task != kBackwardV3IdleTask &&
          task_kind == kBackwardV3ReduceTaskKind) {
        auto *dx_return =
            peer_at<transport::BFloat16>(a.local_base, a.dx_return_offset);
        auto *dx_combined =
            peer_at<transport::BFloat16>(a.local_base, a.dx_combined_offset);
        cutlass::arch::NamedBarrier::sync(384U, kReduceClaimBarrier);
        for (std::uint32_t item = 0; item < reduce_count_shared; ++item)
          backward_v3_reduce_token(a, reduce_tokens_shared[item], dx_return,
                                   dx_combined);
        cutlass::arch::NamedBarrier::sync(384U, kReduceBarrier);
      } else if (encoded_task != kBackwardV3IdleTask && threadIdx.x == 256U) {
        if (task_kind == 0ULL) {
          const auto mblock = encoded_task / dv_ntiles;
          const auto ntile =
              static_cast<std::uint32_t>(encoded_task % dv_ntiles);
          const auto mapping = mblock_mapping[mblock];
          const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
          const auto local_mblock = static_cast<std::uint32_t>(mapping);
          // As with dX and dW, the thread issuing TMA must itself acquire the
          // data-plane publication. The scheduling thread's acquire plus a
          // CTA barrier transfers task identity, but does not establish this
          // thread's system-acquire -> async-proxy chain for remote dY/gates.
          const auto expected =
              backward_v3_mblock_expected_rows(a, expert, local_mblock);
          backward_v3_wait_full_dy_dispatch(a);
          // Acquire every actual producer rail, then validate the exact
          // aggregate row count.  Remote rows are published by the relay GPU
          // that fans them into canonical storage; that rail is not encoded
          // by IncomingRowRecord::source_rank, so reconstructing a per-rail
          // expectation from source_rank can wait on a counter that will
          // never be incremented.
          while (backward_v3_load_dy_arrived_rows(a, mblock, expert,
                                                  local_mblock) < expected)
            __nanosleep(64U);
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto row = a.dispatch.expert_row_prefix[expert] +
                           static_cast<std::uint64_t>(local_mblock) * 128ULL;
          const auto n = ntile * kBackwardV3DvN;
          const auto iterations =
              static_cast<std::uint32_t>((a.hidden + 63ULL) / 64ULL);
          if constexpr (ActivationOnly) {
            for (std::uint32_t group = 0; group < iterations / kStages;
                 ++group) {
#pragma unroll
              for (std::uint32_t stage = 0; stage < kStages; ++stage) {
                const auto iteration = group * kStages + stage;
                const auto phase = activation_stage_epochs[stage] & 1U;
                empty_barriers[stage].wait(phase ^ 1U);
                auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                    smem_a + stage * kAStageElements);
                auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                    smem_b + stage * kBStageElements);
                auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
                    smem_b2 + stage * kBStageElements);
                const auto k_base = iteration * 64U;
                deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                    &dy_map, &full_barriers[stage], stage_a, k_base,
                    static_cast<std::uint32_t>(row),
                    ActivationCluster2 ? 2U : 1U);
                deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                    &w2_transpose_map, &full_barriers[stage], stage_b, k_base,
                    static_cast<std::uint32_t>(expert * a.intermediate + n));
                deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                    &w2_transpose_map, &full_barriers[stage], stage_b2, k_base,
                    static_cast<std::uint32_t>(expert * a.intermediate + n +
                                               128U));
                full_barriers[stage].arrive_and_expect_tx(
                    3U * kAStageElements * sizeof(transport::BFloat16));
                ++activation_stage_epochs[stage];
              }
            }
          } else {
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) {
              const auto current = pipeline + iteration;
              const auto stage = current % kStages;
              const auto phase = (current / kStages) & 1U;
              empty_barriers[stage].wait(phase ^ 1U);
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements);
              auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b + stage * kBStageElements);
              auto *stage_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b2 + stage * kBStageElements);
              const auto k_base = iteration * 64U;
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &dy_map, &full_barriers[stage], stage_a, k_base,
                  static_cast<std::uint32_t>(row),
                  ActivationCluster2 ? 2U : 1U);
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &w2_transpose_map, &full_barriers[stage], stage_b, k_base,
                  static_cast<std::uint32_t>(expert * a.intermediate + n));
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &w2_transpose_map, &full_barriers[stage], stage_b2, k_base,
                  static_cast<std::uint32_t>(expert * a.intermediate + n +
                                             128U));
              full_barriers[stage].arrive_and_expect_tx(
                  3U * kAStageElements * sizeof(transport::BFloat16));
            }
            pipeline += iterations;
          }
        } else if (task_kind == kBackwardV3DxTaskKind) {
          const auto task = encoded_task & kBackwardV3TaskPayloadMask;
          const auto mblock = task / dx_ntiles;
          const auto ntile = static_cast<std::uint32_t>(task % dx_ntiles);
          // The TMA issuer itself acquires the dV tile publication.  dAB is
          // now produced and consumed through the async proxy, so unlike the
          // old generic row-major stores no generic->async proxy fence is
          // required here.
          while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                     mblock_ready[mblock]) != a.epoch)
            __nanosleep(64U);
          const auto mapping = mblock_mapping[mblock];
          const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
          const auto local_mblock = static_cast<std::uint32_t>(mapping);
          auto *expert_padded_prefix = peer_at<unsigned long long>(
              a.local_base, a.v3_expert_padded_prefix_offset);
          const auto padded_row =
              expert_padded_prefix[expert] +
              static_cast<std::uint64_t>(local_mblock) * 128ULL;
          const auto n = ntile * 256U;
          const auto iterations = static_cast<std::uint32_t>(
              (2ULL * a.intermediate + 63ULL) / 64ULL);
          if constexpr (ActivationOnly) {
            for (std::uint32_t group = 0; group < iterations / kStages;
                 ++group) {
#pragma unroll
              for (std::uint32_t stage = 0; stage < kStages; ++stage) {
                const auto iteration = group * kStages + stage;
                const auto phase = activation_stage_epochs[stage] & 1U;
                empty_barriers[stage].wait(phase ^ 1U);
                auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                    smem_a + stage * kAStageElements);
                // dX uses both 128-column B slices as one contiguous 256-column
                // stage. The allocation is a storage union; no extra smem is
                // needed beyond the dV b1/b2 planes.
                auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                    smem_b + stage * (2U * kBStageElements));
                const auto k_base = iteration * 64U;
                for (std::uint32_t atom = 0; atom < 2U; ++atom)
                  deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                      &dab_dx_map, &full_barriers[stage],
                      stage_a + atom * 64U * 64U,
                      static_cast<std::uint32_t>(padded_row + atom * 64ULL),
                      k_base, ActivationCluster2 ? 2U : 1U);
                deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                    &w1_transpose_map, &full_barriers[stage], stage_b, k_base,
                    static_cast<std::uint32_t>(expert * a.hidden + n));
                full_barriers[stage].arrive_and_expect_tx(
                    3U * kAStageElements * sizeof(transport::BFloat16));
                ++activation_stage_epochs[stage];
              }
            }
          } else {
            for (std::uint32_t iteration = 0; iteration < iterations;
                 ++iteration) {
              const auto current = pipeline + iteration;
              const auto stage = current % kStages;
              const auto phase = (current / kStages) & 1U;
              empty_barriers[stage].wait(phase ^ 1U);
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements);
              auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b + stage * (2U * kBStageElements));
              const auto k_base = iteration * 64U;
              for (std::uint32_t atom = 0; atom < 2U; ++atom)
                deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                    &dab_dx_map, &full_barriers[stage],
                    stage_a + atom * 64U * 64U,
                    static_cast<std::uint32_t>(padded_row + atom * 64ULL),
                    k_base, ActivationCluster2 ? 2U : 1U);
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &w1_transpose_map, &full_barriers[stage], stage_b, k_base,
                  static_cast<std::uint32_t>(expert * a.hidden + n));
              full_barriers[stage].arrive_and_expect_tx(
                  3U * kAStageElements * sizeof(transport::BFloat16));
            }
            pipeline += iterations;
          }
        } else if (!ActivationOnly && task_kind == kBackwardV3Dw1TaskKind) {
          auto task = encoded_task & kBackwardV3TaskPayloadMask;
          const auto expert = static_cast<std::uint32_t>(task / dw1_per_expert);
          task %= dw1_per_expert;
          // Keep the 1 MiB X/B tile hot while sweeping all dW1 M tiles.
          // The old M-major order reloaded the full expert X plane once per
          // output M tile and was the opposite of the dW2/DeepGEMM schedule.
          const auto ntile = task / dw1_mtiles;
          const auto mtile = task % dw1_mtiles;
          auto *expert_padded_prefix = peer_at<unsigned long long>(
              a.local_base, a.v3_expert_padded_prefix_offset);
          const auto reduction_start = expert_padded_prefix[expert];
          const auto reduction = persistent_v3::padded_expert_rows(
              a.dispatch.expert_row_count[expert]);
          const auto expert_expected =
              persistent_v3::expert_mblocks(
                  a.dispatch.expert_row_count[expert]) *
              dv_ntiles;
          while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                     expert_dv_complete[expert]) !=
                 backward_v3_expert_ready_value(expert_expected))
            __nanosleep(64U);
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto current = pipeline + iteration;
            const auto stage = current % kStages;
            const auto phase = (current / kStages) & 1U;
            empty_barriers[stage].wait(phase ^ 1U);
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements);
            auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b + stage * (2U * kBStageElements));
            const auto a_k =
                static_cast<std::uint32_t>(reduction_start + iteration * 64ULL);
            const auto b_k = static_cast<std::uint32_t>(
                a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
            deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                &dab_transpose_map, &full_barriers[stage], stage_a, a_k,
                static_cast<std::uint32_t>(mtile * 128ULL));
            for (std::uint32_t atom = 0; atom < 4U; ++atom)
              deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                  &x_mn_map, &full_barriers[stage], stage_b + atom * 64U * 64U,
                  static_cast<std::uint32_t>(ntile * kDwN + atom * 64ULL), b_k);
            full_barriers[stage].arrive_and_expect_tx(
                3U * kAStageElements * sizeof(transport::BFloat16));
          }
          pipeline += iterations;
        } else if (!ActivationOnly && task_kind == kBackwardV3Dw2TaskKind) {
          auto task = encoded_task & kBackwardV3TaskPayloadMask;
          const auto expert = static_cast<std::uint32_t>(task / dw2_per_expert);
          task %= dw2_per_expert;
          const auto ntile = task / dw2_mtiles;
          const auto mtile = task % dw2_mtiles;
          auto *expert_padded_prefix = peer_at<unsigned long long>(
              a.local_base, a.v3_expert_padded_prefix_offset);
          const auto reduction_start = expert_padded_prefix[expert];
          const auto reduction = persistent_v3::padded_expert_rows(
              a.dispatch.expert_row_count[expert]);
          const auto expert_expected =
              persistent_v3::expert_mblocks(
                  a.dispatch.expert_row_count[expert]) *
              dv_ntiles;
          while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                     expert_dv_complete[expert]) !=
                 backward_v3_expert_ready_value(expert_expected))
            __nanosleep(64U);
          asm volatile("fence.proxy.async.global;" : : : "memory");
          const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto current = pipeline + iteration;
            const auto stage = current % kStages;
            const auto phase = (current / kStages) & 1U;
            empty_barriers[stage].wait(phase ^ 1U);
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements);
            auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b + stage * (2U * kBStageElements));
            const auto a_k = static_cast<std::uint32_t>(
                a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
            const auto b_k =
                static_cast<std::uint32_t>(reduction_start + iteration * 64ULL);
            for (std::uint32_t atom = 0; atom < 2U; ++atom)
              deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                  &dy_mn_map, &full_barriers[stage], stage_a + atom * 64U * 64U,
                  static_cast<std::uint32_t>(mtile * 128ULL + atom * 64ULL),
                  a_k);
            for (std::uint32_t atom = 0; atom < 2U; ++atom)
              deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
                  &u_transpose_map, &full_barriers[stage],
                  stage_b + atom * 128U * 64U, b_k,
                  static_cast<std::uint32_t>(ntile * kDwN + atom * 128ULL));
            full_barriers[stage].arrive_and_expect_tx(
                3U * kAStageElements * sizeof(transport::BFloat16));
          }
          pipeline += iterations;
        }
      } else if (encoded_task == kBackwardV3IdleTask && threadIdx.x == 256U) {
        __nanosleep(64U);
      }
      if constexpr (ActivationOnly) {
        // The producer's pre-load wait acquires the previous empty phase, but
        // the last issued stage has no following load to perform that acquire.
        // Drain it explicitly before the task boundary.  This is also the
        // required deconstruction protocol for distributed (multicast)
        // mbarriers and keeps their state valid for the next persistent task.
        if (encoded_task != kBackwardV3IdleTask &&
            (task_kind == 0ULL || task_kind == kBackwardV3DxTaskKind) &&
            threadIdx.x == 256U) {
#pragma unroll
          for (std::uint32_t stage = 0; stage < kStages; ++stage)
            empty_barriers[stage].wait((activation_stage_epochs[stage] + 1U) &
                                       1U);
        }
      }
      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<224>();
    std::uint32_t pipeline = 0U;
    std::uint32_t activation_stage_epochs[kStages] = {0U, 0U, 0U, 0U};
    std::uint32_t activation_phase = 0U;
    std::uint32_t weight_phase = 0U;
#if MEGAMOE_DEVICE_TIMELINE
    bool timeline_dv_started = false;
    bool timeline_dv_sample = false;
    bool timeline_dx_started = false;
    bool timeline_dx_sample = false;
#endif
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kTaskReadyBarrier);
      const auto encoded_task = task_shared;
      if (encoded_task == kBackwardV3DoneTask)
        break;
      const auto task_kind = encoded_task & kBackwardV3TaskKindMask;
      backward_v3_localize_dy_task(a, encoded_task, dv_ntiles, mblock_mapping,
                                   mblock_localized, dy_localize_owner_shared);
      if (encoded_task != kBackwardV3IdleTask &&
          task_kind == kBackwardV3ReduceTaskKind) {
        auto *dx_return =
            peer_at<transport::BFloat16>(a.local_base, a.dx_return_offset);
        auto *dx_combined =
            peer_at<transport::BFloat16>(a.local_base, a.dx_combined_offset);
        if (threadIdx.x == 0U) {
          reduce_count_shared = 1U;
          reduce_tokens_shared[0] = encoded_task & kBackwardV3TaskPayloadMask;
          while (reduce_count_shared < 12U) {
            const auto next =
                backward_v3_try_claim_reduce_task(a, *control, token_state);
            if (next == kBackwardV3IdleTask)
              break;
            reduce_tokens_shared[reduce_count_shared++] =
                next & kBackwardV3TaskPayloadMask;
          }
        }
        cutlass::arch::NamedBarrier::sync(384U, kReduceClaimBarrier);
        for (std::uint32_t item = 0; item < reduce_count_shared; ++item)
          backward_v3_reduce_token(a, reduce_tokens_shared[item], dx_return,
                                   dx_combined);
        cutlass::arch::NamedBarrier::sync(384U, kReduceBarrier);
        if (threadIdx.x == 0U) {
          for (std::uint32_t item = 0; item < reduce_count_shared; ++item) {
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
                state(token_state[reduce_tokens_shared[item]]);
            state.store(
                kBackwardV3TokenCompleteBit |
                    static_cast<unsigned long long>(a.dispatch.identity.top_k),
                ::cuda::memory_order_release);
          }
          const auto completed =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->reduce_tasks_complete.value, reduce_count_shared) +
              reduce_count_shared;
          if (completed == control->reduce_tasks)
            status->dx_reduce_complete_clock = device_globaltimer_ns();
        }
      } else if (encoded_task != kBackwardV3IdleTask && task_kind == 0ULL) {
        if (threadIdx.x == 0U) {
          atomicMin(&status->dv_start_clock, device_globaltimer_ns());
#if MEGAMOE_DEVICE_TIMELINE
          timeline_dv_sample = false;
          if (!timeline_dv_started) {
            MEGAMOE_TIMELINE_RECORD(a.timeline, DyFirstMtileReady, Compute, 0U,
                                    encoded_task, a.epoch);
            MEGAMOE_TIMELINE_RECORD(a.timeline, DvFirst, Compute, 0U,
                                    encoded_task, a.epoch);
            MEGAMOE_TIMELINE_RECORD(a.timeline, DvClaimStart, Compute, 0U,
                                    encoded_task, a.epoch);
            timeline_dv_started = true;
            timeline_dv_sample = true;
          }
#endif
        }
        const auto mblock = encoded_task / dv_ntiles;
        const auto ntile = static_cast<std::uint32_t>(encoded_task % dv_ntiles);
        const auto mapping = mblock_mapping[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        const auto row = a.dispatch.expert_row_prefix[expert] +
                         static_cast<std::uint64_t>(local_mblock) * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            min(128ULL, a.dispatch.expert_row_count[expert] -
                            static_cast<std::uint64_t>(local_mblock) * 128ULL));
        const auto n = ntile * kBackwardV3DvN;

        // The TMA producer acquired every dY rail before issuing the mainloop,
        // but canonical gates are consumed through the generic proxy by the
        // two math warp-groups.  Keep the system acquire and the peer-global
        // gate loads in the same CUDA thread, then hand the values to all math
        // lanes through shared memory.  A CTA barrier transfers shared-memory
        // writes; it does not make thread 0's system acquire apply directly to
        // peer-global loads subsequently issued by unrelated lanes.
        if (threadIdx.x == 0U) {
          const auto expected =
              backward_v3_mblock_expected_rows(a, expert, local_mblock);
          while (backward_v3_load_dy_arrived_rows(a, mblock, expert,
                                                  local_mblock) < expected)
            __nanosleep(64U);
          auto *canonical_gate =
              peer_at<float>(a.local_base, a.canonical_gate_offset);
          for (std::uint32_t local_row = 0; local_row < 128U; ++local_row)
            dv_gate_shared[local_row] =
                local_row < valid_rows ? canonical_gate[row + local_row] : 0.0F;
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);

        float accum_a[ForwardMma::kNumAccum];
        float accum_b[ForwardMma::kNumAccum];
#pragma unroll
        for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
          accum_a[index] = 0.0F;
          accum_b[index] = 0.0F;
        }
        const auto iterations =
            static_cast<std::uint32_t>((a.hidden + 63ULL) / 64ULL);
        if constexpr (ActivationOnly) {
          for (std::uint32_t group = 0; group < iterations / kStages; ++group) {
#pragma unroll
            for (std::uint32_t stage = 0; stage < kStages; ++stage) {
              const auto phase = activation_stage_epochs[stage] & 1U;
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements + warpgroup * 64U * 64U);
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
              if constexpr (ActivationCluster2) {
                const auto lane = threadIdx.x & 31U;
                if (lane < 2U)
                  empty_barriers[stage].arrive(lane);
              } else if ((threadIdx.x & 31U) == 0U) {
                empty_barriers[stage].arrive();
              }
              ++activation_stage_epochs[stage];
            }
          }
        } else {
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto current = pipeline + iteration;
            const auto stage = current % kStages;
            const auto phase = (current / kStages) & 1U;
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements + warpgroup * 64U * 64U);
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
            if constexpr (ActivationCluster2) {
              const auto lane = threadIdx.x & 31U;
              if (lane < 2U)
                empty_barriers[stage].arrive(lane);
            } else if ((threadIdx.x & 31U) == 0U) {
              empty_barriers[stage].arrive();
            }
          }
          pipeline += iterations;
        }
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U && timeline_dv_sample)
          MEGAMOE_TIMELINE_RECORD(a.timeline, DvMathEnd, Compute, 0U,
                                  encoded_task, a.epoch);
#endif

        auto *expert_padded_prefix = peer_at<unsigned long long>(
            a.local_base, a.v3_expert_padded_prefix_offset);
        auto *ab = a.use_forward_cache
                       ? peer_at<transport::BFloat16>(a.local_base,
                                                      a.forward_ab_cache_offset)
                       : peer_at<transport::BFloat16>(a.local_base,
                                                      a.recompute_ab_offset);
        const auto *sigmoid_lut =
            peer_at<const float>(a.local_base, a.v3_sigmoid_lut_offset);
        auto *dgate_partial =
            peer_at<float>(a.local_base, a.dgate_partial_offset);
        auto *dgate_canonical =
            peer_at<float>(a.local_base, a.dgate_canonical_offset);
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_BUILD_TESTING
        auto *debug_dv =
            peer_at<transport::BFloat16>(a.local_base, a.debug_dv_offset);
        auto *debug_dab =
            peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
#endif
        auto *mblock_dv_complete = peer_at<unsigned long long>(
            a.local_base, a.v3_mblock_dv_complete_offset);
        const auto padded_row_base =
            expert_padded_prefix[expert] +
            static_cast<std::uint64_t>(local_mblock) * 128ULL;
        constexpr std::uint32_t kEpilogueTileElements = 128U * 128U;
        // U/dA/dB are emitted only in the padded reduction-major layout used
        // directly by dX, dW1 and dW2.  This removes the canonical dAB global
        // write/read round trip from the production data flow.
        // Each lane owns 32 columns for two rows. Accumulate dV*S in registers
        // while visiting the fragments, then reduce each four-lane row group
        // once after both N128 halves. This removes two shared planes and the
        // separate shared-memory reduction pass.
        float dgate_tile_row0 = 0.0F;
        float dgate_tile_row1 = 0.0F;
        const auto epilogue_lane = threadIdx.x & 31U;
        const auto epilogue_lane_in_row = epilogue_lane & 3U;

        // Double-buffer the two output halves so the second nonlinear pass can
        // run while the first set of TMA stores drains through the async proxy.
#pragma unroll
        for (std::uint32_t half = 0; half < 2U; ++half) {
          auto *epilogue_u_t = smem_a + half * 3U * kEpilogueTileElements;
          auto *epilogue_da_t = epilogue_u_t + kEpilogueTileElements;
          auto *epilogue_db_t = epilogue_da_t + kEpilogueTileElements;
          float dgate_virtual_row0[8];
          float dgate_virtual_row1[8];
#pragma unroll
          for (std::uint32_t index = 0; index < 8U; ++index) {
            dgate_virtual_row0[index] = 0.0F;
            dgate_virtual_row1[index] = 0.0F;
          }
          auto *half_accum = half == 0U ? accum_a : accum_b;
          if (threadIdx.x < 256U) {
            const auto warpgroup = threadIdx.x / 128U;
            const auto warpgroup_thread = threadIdx.x % 128U;
            const auto warp = warpgroup_thread / 32U;
            const auto lane = warpgroup_thread % 32U;
            const auto fragment_row0 =
                warpgroup * 64U + warp * 16U + lane / 4U;
            const auto fragment_row1 = fragment_row0 + 8U;
#pragma unroll
            for (std::uint32_t fragment = 0;
                 fragment < ForwardMma::kNumAccum / 4U; ++fragment) {
              const auto column = (lane % 4U) * 2U + fragment * 8U;
              const auto output_column =
                  static_cast<std::uint64_t>(n) + half * 128ULL + column;
              std::uint32_t u_words[2] = {0U, 0U};
              std::uint32_t da_words[2] = {0U, 0U};
              std::uint32_t db_words[2] = {0U, 0U};
#pragma unroll
              for (std::uint32_t row_slot = 0; row_slot < 2U; ++row_slot) {
                const auto block_m =
                    row_slot == 0U ? fragment_row0 : fragment_row1;
                const auto value0 = half_accum[fragment * 4U + row_slot * 2U];
                const auto value1 =
                    half_accum[fragment * 4U + row_slot * 2U + 1U];
                if (block_m < valid_rows &&
                    output_column + 1ULL < a.intermediate) {
                  const auto index =
                      (row + block_m) * a.intermediate + output_column;
                  // Load A/B as aligned BF16x2 words and keep every nonlinear
                  // conversion packed.  The two row fragments are retained
                  // just long enough for one x2 STSM per U/dA/dB plane.
                  const auto dv_pair = static_cast<__nv_bfloat162_raw>(
                      __float22bfloat162_rn(make_float2(value0, value1)));
                  const auto av_pair =
                      __ldg(reinterpret_cast<const uint32_t *>(ab + index));
                  const auto bv_pair = __ldg(
                      reinterpret_cast<const uint32_t *>(
                          ab + a.max_destination_rows * a.intermediate +
                          index));
                  const std::uint16_t dv_bits[2] = {dv_pair.x, dv_pair.y};
                  const std::uint16_t av_bits[2] = {
                      static_cast<std::uint16_t>(av_pair),
                      static_cast<std::uint16_t>(av_pair >> 16U)};
                  const std::uint16_t bv_bits[2] = {
                      static_cast<std::uint16_t>(bv_pair),
                      static_cast<std::uint16_t>(bv_pair >> 16U)};
                  float silu_values[2];
                  float sigmoid_values[2];
                  float s_values[2];
#pragma unroll
                  for (std::uint32_t value = 0; value < 2U; ++value) {
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_BUILD_TESTING
                    if (a.capture_stage_storage)
                      debug_dv[(row + block_m) * a.intermediate +
                               output_column + value] =
                          transport::BFloat16{dv_bits[value]};
#endif
                    const auto av = bf16_bits_to_float(av_bits[value]);
                    const auto bv = bf16_bits_to_float(bv_bits[value]);
                    sigmoid_values[value] =
                        __ldg(sigmoid_lut + av_bits[value]);
                    silu_values[value] = av * sigmoid_values[value];
                    s_values[value] = silu_values[value] * bv;
                  }
                  const auto s_pair = static_cast<__nv_bfloat162_raw>(
                      __float22bfloat162_rn(
                          make_float2(s_values[0], s_values[1])));
                  const std::uint16_t s_bits[2] = {s_pair.x, s_pair.y};
                  float u_values[2];
                  float da_values[2];
                  float db_values[2];
#pragma unroll
                  for (std::uint32_t value = 0; value < 2U; ++value) {
                    const auto av = bf16_bits_to_float(av_bits[value]);
                    const auto bv = bf16_bits_to_float(bv_bits[value]);
                    const auto gate = dv_gate_shared[block_m];
                    u_values[value] =
                        __fmul_rn(gate, bf16_bits_to_float(s_bits[value]));
                    const auto ds =
                        __fmul_rn(gate, bf16_bits_to_float(dv_bits[value]));
                    da_values[value] = __fmul_rn(
                        ds, __fmul_rn(
                                bv, sigmoid_values[value] *
                                        (1.0F +
                                         av * (1.0F - sigmoid_values[value]))));
                    db_values[value] = __fmul_rn(ds, silu_values[value]);
                    const auto product =
                        __fmul_rn(bf16_bits_to_float(dv_bits[value]),
                                  bf16_bits_to_float(s_bits[value]));
                    const auto dgate_slot = (fragment % 4U) * 2U + value;
                    auto *dgate_virtual = row_slot == 0U
                                              ? dgate_virtual_row0
                                              : dgate_virtual_row1;
                    dgate_virtual[dgate_slot] =
                        __fadd_rn(dgate_virtual[dgate_slot], product);
                  }
                  const auto u_pair = static_cast<__nv_bfloat162_raw>(
                      __float22bfloat162_rn(
                          make_float2(u_values[0], u_values[1])));
                  const auto da_pair = static_cast<__nv_bfloat162_raw>(
                      __float22bfloat162_rn(
                          make_float2(da_values[0], da_values[1])));
                  const auto db_pair = static_cast<__nv_bfloat162_raw>(
                      __float22bfloat162_rn(
                          make_float2(db_values[0], db_values[1])));
                  u_words[row_slot] = static_cast<std::uint32_t>(u_pair.x) |
                                      (static_cast<std::uint32_t>(u_pair.y)
                                       << 16U);
                  da_words[row_slot] =
                      static_cast<std::uint32_t>(da_pair.x) |
                      (static_cast<std::uint32_t>(da_pair.y) << 16U);
                  db_words[row_slot] =
                      static_cast<std::uint32_t>(db_pair.x) |
                      (static_cast<std::uint32_t>(db_pair.y) << 16U);
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_BUILD_TESTING
                  if (a.capture_stage_storage) {
                    const auto debug_row =
                        (row + block_m) * 2ULL * a.intermediate;
                    debug_dab[debug_row + output_column] =
                        transport::BFloat16{da_pair.x};
                    debug_dab[debug_row + output_column + 1ULL] =
                        transport::BFloat16{da_pair.y};
                    debug_dab[debug_row + a.intermediate + output_column] =
                        transport::BFloat16{db_pair.x};
                    debug_dab[debug_row + a.intermediate + output_column +
                              1ULL] = transport::BFloat16{db_pair.y};
                  }
#endif
                }
              }

              // In the reduction-major TMA layout, eight consecutive rows
              // form one aligned 16-byte vector for each output column.  The
              // transposed x2 STSM consumes the two WGMMA row fragments and
              // writes both 8x8 row blocks directly into that swizzle.
              const auto destination_column = fragment * 8U + lane % 8U;
              const auto destination_row =
                  warpgroup * 64U + warp * 16U +
                  ((lane / 8U) & 1U) * 8U;
              const auto destination_index = backward_v3_tma_128b_bf16_index(
                  destination_column, destination_row);
              deep_gemm::ptx::SM90_U32x2_STSM_T<std::uint32_t>::copy(
                  u_words[0], u_words[1], epilogue_u_t + destination_index);
              deep_gemm::ptx::SM90_U32x2_STSM_T<std::uint32_t>::copy(
                  da_words[0], da_words[1], epilogue_da_t + destination_index);
              deep_gemm::ptx::SM90_U32x2_STSM_T<std::uint32_t>::copy(
                  db_words[0], db_words[1], epilogue_db_t + destination_index);
            }
          }
          const auto dgate_half_row0 =
              backward_v3_reduce_dgate_fragment(dgate_virtual_row0);
          const auto dgate_half_row1 =
              backward_v3_reduce_dgate_fragment(dgate_virtual_row1);
          if (half == 0U) {
            dgate_tile_row0 = dgate_half_row0;
            dgate_tile_row1 = dgate_half_row1;
          } else {
            dgate_tile_row0 = __fadd_rn(dgate_tile_row0, dgate_half_row0);
            dgate_tile_row1 = __fadd_rn(dgate_tile_row1, dgate_half_row1);
          }
          cute::tma_store_fence();
          cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
          if (threadIdx.x == 0U) {
            const auto output_feature =
                static_cast<std::uint32_t>(n + half * 128ULL);
#pragma unroll
            for (std::uint32_t atom = 0; atom < 2U; ++atom) {
              const auto output_row =
                  static_cast<std::uint32_t>(padded_row_base + atom * 64ULL);
              cute::SM90_TMA_STORE_2D::copy(&u_transpose_map,
                                            epilogue_u_t + atom * 128U * 64U,
                                            output_row, output_feature);
              cute::SM90_TMA_STORE_2D::copy(&dab_transpose_map,
                                            epilogue_da_t + atom * 128U * 64U,
                                            output_row, output_feature);
              cute::SM90_TMA_STORE_2D::copy(
                  &dab_transpose_map, epilogue_db_t + atom * 128U * 64U,
                  output_row,
                  static_cast<std::uint32_t>(a.intermediate) + output_feature);
            }
            // Commit the six independent tensor stores as one bulk group.
            // Splitting them across lanes/groups did not change dW1
            // determinism and only enlarged the epilogue protocol.
            cute::tma_store_arrive();
          }
        }
        if (epilogue_lane_in_row == 0U) {
          const auto warpgroup_thread = threadIdx.x & 127U;
          const auto warp = warpgroup_thread / 32U;
          const auto row0 =
              (threadIdx.x / 128U) * 64U + warp * 16U + epilogue_lane / 4U;
          const auto row1 = row0 + 8U;
          if (row0 < valid_rows)
            dgate_partial[(row + row0) * dv_ntiles + ntile] = dgate_tile_row0;
          if (row1 < valid_rows)
            dgate_partial[(row + row1) * dv_ntiles + ntile] = dgate_tile_row1;
        }
        if (threadIdx.x == 0U)
          backward_v3_tma_store_wait_complete();
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        // Only the 64 row-fragment lanes write dGate through the generic
        // proxy, while thread 0 owns every async TMA store and its wait.  Fence
        // the actual producers instead of making all 256 math threads execute
        // both fences for every tile.  The following CTA rendezvous precedes
        // the release-counter chain consumed by dX/dW.
        if (epilogue_lane_in_row == 0U || a.capture_stage_storage)
          __threadfence();
        if (threadIdx.x == 0U)
          asm volatile("fence.proxy.async.global;" : : : "memory");
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              mblock_complete(mblock_dv_complete[mblock]);
          const auto mblock_completed =
              mblock_complete.fetch_add(1ULL, ::cuda::memory_order_acq_rel) +
              1ULL;
          task_flags_shared = mblock_completed == dv_ntiles ? 1U : 0U;
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              expert_complete(expert_dv_complete[expert]);
          // Every RMW acquires the preceding release sequence before adding
          // this task's release.  Consequently the final value represents a
          // transitive publication of every dV tile, not only the CTA that
          // happened to perform the last increment.
          const auto expert_completed =
              expert_complete.fetch_add(1ULL, ::cuda::memory_order_acq_rel) +
              1ULL;
          const auto expert_expected =
              persistent_v3::expert_mblocks(
                  a.dispatch.expert_row_count[expert]) *
              dv_ntiles;
          if (expert_completed == expert_expected) {
            // The final acq_rel RMW has acquired every preceding dV tile's
            // release.  Cross the generic -> async proxy once more from this
            // aggregate participant, then publish a state distinct from the
            // transient numeric count.  dW TMA producers never observe the
            // count as ready before this proxy publication completes.
            __threadfence();
            asm volatile("fence.proxy.async.global;" : : : "memory");
            expert_complete.store(
                backward_v3_expert_ready_value(expert_expected),
                ::cuda::memory_order_release);
            const auto now = device_globaltimer_ns();
            atomicMin(&status->dw1_first_ready_clock, now);
            atomicMin(&status->dw2_first_ready_clock, now);
          }
          const auto completed =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->dv_tasks_complete.value, 1ULL) +
              1ULL;
          if (completed == dv_tasks) {
            status->dv_complete_clock = device_globaltimer_ns();
            MEGAMOE_TIMELINE_RECORD(a.timeline, DvLast, Compute, 0U, completed,
                                    a.epoch);
          }
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (task_flags_shared != 0U) {
          for (std::uint32_t local_row = threadIdx.x; local_row < valid_rows;
               local_row += 256U) {
            float sum = 0.0F;
            for (std::uint32_t tile = 0; tile < dv_ntiles; ++tile)
              sum = __fadd_rn(
                  sum, dgate_partial[(row + local_row) * dv_ntiles + tile]);
            dgate_canonical[row + local_row] = sum;
          }
          // Remote proxy gather reads dGate directly through the peer mapping.
          // Make the row plane system-visible before dX can eventually release
          // the corresponding ReturnChunk producer count.
          if (threadIdx.x < valid_rows)
            __threadfence_system();
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x == 0U && task_flags_shared != 0U) {
          persistent_v3::store_release<::cuda::thread_scope_device>(
              mblock_ready[mblock], a.epoch);
          const auto ready =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->mblocks_ready.value, 1ULL) +
              1ULL;
          if (ready == mblocks) {
            const auto now = device_globaltimer_ns();
            status->dgate_complete_clock = now;
            status->transpose_complete_clock = now;
          }
        }
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U && timeline_dv_sample)
          MEGAMOE_TIMELINE_RECORD(a.timeline, DvTileEnd, Compute, 0U,
                                  encoded_task, a.epoch);
#endif
      } else if (encoded_task != kBackwardV3IdleTask &&
                 task_kind == kBackwardV3DxTaskKind) {
        auto task = encoded_task & kBackwardV3TaskPayloadMask;
        const auto mblock = task / dx_ntiles;
        const auto ntile = static_cast<std::uint32_t>(task % dx_ntiles);
        const auto mapping = mblock_mapping[mblock];
        const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
        const auto local_mblock = static_cast<std::uint32_t>(mapping);
        const auto row = a.dispatch.expert_row_prefix[expert] +
                         static_cast<std::uint64_t>(local_mblock) * 128ULL;
        const auto valid_rows = static_cast<std::uint32_t>(
            min(128ULL, a.dispatch.expert_row_count[expert] -
                            static_cast<std::uint64_t>(local_mblock) * 128ULL));
        const auto n = ntile * 256U;
        if (threadIdx.x == 0U) {
          atomicMin(&status->dx_start_clock, device_globaltimer_ns());
#if MEGAMOE_DEVICE_TIMELINE
          timeline_dx_sample = false;
          if (!timeline_dx_started) {
            MEGAMOE_TIMELINE_RECORD(a.timeline, DxFirst, Compute, 0U,
                                    encoded_task, a.epoch);
            timeline_dx_started = true;
            timeline_dx_sample = true;
          }
#endif
        }

        static_assert(BackwardDwDirectMma::kNumAccum ==
                      ForwardMmaWide::kNumAccum);
        float accum[BackwardDwDirectMma::kNumAccum];
#pragma unroll
        for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
          accum[index] = 0.0F;
        const auto iterations =
            static_cast<std::uint32_t>((2ULL * a.intermediate + 63ULL) / 64ULL);
        if constexpr (ActivationOnly) {
          for (std::uint32_t group = 0; group < iterations / kStages; ++group) {
#pragma unroll
            for (std::uint32_t stage = 0; stage < kStages; ++stage) {
              const auto phase = activation_stage_epochs[stage] & 1U;
              auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_a + stage * kAStageElements);
              auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                  smem_b + stage * (2U * kBStageElements));
              full_barriers[stage].wait(phase);
#pragma unroll
              for (int index = 0; index < BackwardDwDirectMma::kNumAccum;
                   ++index)
                deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
              deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
              for (std::uint32_t k = 0; k < 64U / BackwardDwDirectMma::K; ++k) {
                const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
                    cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
                    stage_a, warpgroup * 64U, k * BackwardDwDirectMma::K);
                const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
                    cute::UMMA::Major::K, 256, 64, 128, cutlass::bfloat16_t>(
                    stage_b, 0U, k * BackwardDwDirectMma::K);
                BackwardDwDirectMma::wgmma(desc_a, desc_b, accum, 1);
              }
              deep_gemm::ptx::warpgroup_commit_batch();
              backward_v3_wgmma_wait0_accum128(accum);
              if constexpr (ActivationCluster2) {
                const auto lane = threadIdx.x & 31U;
                if (lane < 2U)
                  empty_barriers[stage].arrive(lane);
              } else if ((threadIdx.x & 31U) == 0U) {
                empty_barriers[stage].arrive();
              }
              ++activation_stage_epochs[stage];
            }
          }
        } else {
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto current = pipeline + iteration;
            const auto stage = current % kStages;
            const auto phase = (current / kStages) & 1U;
            auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_a + stage * kAStageElements);
            auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
                smem_b + stage * (2U * kBStageElements));
            full_barriers[stage].wait(phase);
#pragma unroll
            for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
              deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
            deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
            for (std::uint32_t k = 0; k < 64U / BackwardDwDirectMma::K; ++k) {
              const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
                  cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
                  stage_a, warpgroup * 64U, k * BackwardDwDirectMma::K);
              const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
                  cute::UMMA::Major::K, 256, 64, 128, cutlass::bfloat16_t>(
                  stage_b, 0U, k * BackwardDwDirectMma::K);
              BackwardDwDirectMma::wgmma(desc_a, desc_b, accum, 1);
            }
            deep_gemm::ptx::warpgroup_commit_batch();
            backward_v3_wgmma_wait0_accum128(accum);
            if constexpr (ActivationCluster2) {
              const auto lane = threadIdx.x & 31U;
              if (lane < 2U)
                empty_barriers[stage].arrive(lane);
            } else if ((threadIdx.x & 31U) == 0U) {
              empty_barriers[stage].arrive();
            }
          }
          pipeline += iterations;
        }
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U && timeline_dx_sample)
          MEGAMOE_TIMELINE_RECORD(a.timeline, DxMathEnd, Compute, 0U,
                                  encoded_task, a.epoch);
#endif

        auto *dx_canonical =
            peer_at<transport::BFloat16>(a.local_base, a.dx_canonical_offset);
        auto *headers =
            peer_at<LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
        auto *dgate_canonical =
            peer_at<float>(a.local_base, a.dgate_canonical_offset);
        auto *forward_output_ready_epoch = peer_at<unsigned long long>(
            a.local_base, a.forward_output_ready_epoch_offset);
        auto *mblock_dx_complete = peer_at<unsigned long long>(
            a.local_base, a.v3_mblock_dx_complete_offset);
        visit_forward_accumulator_wide(
            accum, [&](std::uint32_t wg, std::uint32_t m, std::uint32_t column,
                       float v0, float v1) {
              const auto local_row = wg * 64U + m;
              const auto pair = static_cast<__nv_bfloat162_raw>(
                  __float22bfloat162_rn(make_float2(v0, v1)));
              reinterpret_cast<std::uint32_t *>(output_tile + local_row * 256U +
                                                column)[0] =
                  static_cast<std::uint32_t>(pair.x) |
                  (static_cast<std::uint32_t>(pair.y) << 16U);
            });
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        // The WGMMA stages are dead after wait_group 0. Reuse their shared
        // storage for one row mapping and one direct staging pointer per row,
        // avoiding hundreds of system-scope mapping loads in the vector loop.
        auto *return_mapping_tile =
            reinterpret_cast<unsigned long long *>(smem_b);
        auto *return_staging_tile = return_mapping_tile + 128U;
        for (std::uint32_t local_row = threadIdx.x; local_row < valid_rows;
             local_row += 256U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published_mapping(forward_output_ready_epoch[row + local_row]);
          const auto mapping =
              published_mapping.load(::cuda::memory_order_acquire);
          return_mapping_tile[local_row] = mapping;
          if (a.v3_direct_return_staging && mapping != 0ULL) {
            const auto proxy_rank = return_mapping_proxy(mapping);
            return_staging_tile[local_row] =
                reinterpret_cast<unsigned long long>(ncclGetPeerPointer(
                    a.gin_window, a.gin_return_staging_offset, proxy_rank));
          } else {
            return_staging_tile[local_row] = 0ULL;
          }
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        constexpr std::uint32_t kDxVectorsPerRow = 256U / 8U;
        constexpr std::uint32_t kDxOutputVectors = 128U * kDxVectorsPerRow;
        for (std::uint32_t vector = threadIdx.x;
             threadIdx.x < kDxEpilogueThreads && vector < kDxOutputVectors;
             vector += kDxEpilogueThreads) {
          const auto local_row = vector / kDxVectorsPerRow;
          const auto local_vector = vector % kDxVectorsPerRow;
          const auto output_column =
              static_cast<std::uint64_t>(n) + local_vector * 8ULL;
          if (local_row < valid_rows && output_column + 7ULL < a.hidden) {
            const auto packed = reinterpret_cast<const uint4 *>(
                output_tile + local_row * 256U)[local_vector];
            bool preserve_canonical = !a.v3_direct_return_staging;
#if MEGAMOE_DEBUG_VALIDATE
            preserve_canonical = preserve_canonical || a.capture_stage_storage;
#endif
            if (preserve_canonical)
              reinterpret_cast<uint4 *>(dx_canonical +
                                        (row + local_row) * a.hidden +
                                        output_column)[0] = packed;
            const auto mapping = return_mapping_tile[local_row];
            if (mapping == 0ULL) {
              const auto header = headers[row + local_row];
              if (header.source_rank / a.local_world_size == a.node) {
                auto *source_dx = peer_at<transport::BFloat16>(
                    a.peer_bases[header.source_rank], a.dx_return_offset);
                reinterpret_cast<uint4 *>(source_dx +
                                          header.route_index * a.hidden +
                                          output_column)[0] = packed;
              }
            } else if (a.v3_direct_return_staging) {
              auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
                  return_staging_tile[local_row]);
              const auto packed_row = return_mapping_compact_row(mapping);
              reinterpret_cast<uint4 *>(return_staging + packed_row * a.hidden +
                                        output_column)[0] = packed;
            }
          }
        }
        if (threadIdx.x < kDxEpilogueThreads)
          __threadfence_system();
        if (a.v3_direct_return_staging && threadIdx.x < kDxEpilogueThreads)
          asm volatile("fence.proxy.async.global;" : : : "memory");
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              mblock_complete(mblock_dx_complete[mblock]);
          const auto mblock_completed =
              mblock_complete.fetch_add(1ULL, ::cuda::memory_order_acq_rel) +
              1ULL;
          task_flags_shared = mblock_completed == dx_ntiles ? 1U : 0U;
          const auto completed =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->dx_tasks_complete.value, 1ULL) +
              1ULL;
          if (completed == dx_tasks) {
            status->dx_complete_clock = device_globaltimer_ns();
            MEGAMOE_TIMELINE_RECORD(a.timeline, DxTileReady, Compute, 0U,
                                    completed, a.epoch);
          }
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (task_flags_shared != 0U) {
          // Thread 0 is the final mblock counter participant, but up to 128
          // different threads publish row/chunk readiness below.  A CTA
          // barrier alone does not give each publishing thread a device-scope
          // acquire of the counter's release sequence.  Make every publisher
          // acquire the final value so its subsequent system-scope RMW
          // transitively releases every N-tile payload store, independent of
          // which CTA produced that tile.
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed_tiles(mblock_dx_complete[mblock]);
          const auto visible_tiles =
              completed_tiles.load(::cuda::memory_order_acquire);
          if (visible_tiles != dx_ntiles)
            persistent_v3::publish_error_once(control->generation, 602ULL);
          // The final N-tile publishes one row completion. Local rows already
          // reside in their source rank's route buffer; remote rows release
          // the immutable proxy/chunk count consumed by a Relay CTA.
          for (std::uint32_t local_row = threadIdx.x; local_row < valid_rows;
               local_row += 256U) {
            const auto canonical_row = row + local_row;
            const auto header = headers[canonical_row];
            if (header.source_rank / a.local_world_size == a.node) {
              auto *source_dgate = peer_at<float>(
                  a.peer_bases[header.source_rank], a.dgate_offset);
              source_dgate[header.route_index] = dgate_canonical[canonical_row];
              if (header.source_rank != a.rank)
                atomicAdd(&status->actual_dx_peer_bytes,
                          a.hidden * sizeof(transport::BFloat16));
            } else {
              const auto mapping = return_mapping_tile[local_row];
              if (mapping == 0ULL) {
                persistent_v3::publish_error_once(control->generation, 601ULL);
                continue;
              }
              const auto proxy_rank = return_mapping_proxy(mapping);
              const auto chunk = return_mapping_chunk(mapping);
              if (a.v3_direct_return_staging) {
                auto *return_gate_staging =
                    static_cast<float *>(ncclGetPeerPointer(
                        a.gin_window, a.gin_return_gate_staging_offset,
                        proxy_rank));
                return_gate_staging[return_mapping_compact_row(mapping)] =
                    dgate_canonical[canonical_row];
                __threadfence_system();
                asm volatile("fence.proxy.async.global;" : : : "memory");
              }
              auto *peer_ready_count = peer_at<unsigned long long>(
                  a.peer_bases[proxy_rank],
                  a.forward_return_chunk_ready_count_offset);
              const auto producer_rail = a.rank % a.local_world_size;
              ::cuda::atomic_ref<unsigned long long,
                                 ::cuda::thread_scope_system>
                  ready(peer_ready_count[chunk * a.local_world_size +
                                         producer_rail]);
              // Rows for one chunk/rail can be completed by different dX
              // CTAs.  Carry all preceding payload publications through the
              // atomic modification order to the Relay CTA that observes the
              // final count.
              ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
            }
          }
          if (threadIdx.x < valid_rows)
            __threadfence_system();
          cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
          for (std::uint32_t local_row = threadIdx.x; local_row < valid_rows;
               local_row += 256U) {
            const auto header = headers[row + local_row];
            if (header.source_rank / a.local_world_size != a.node)
              continue;
            auto *source_token_state = peer_at<unsigned long long>(
                a.peer_bases[header.source_rank], a.v3_token_state_offset);
            const auto token = header.route_index / a.dispatch.identity.top_k;
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
                ready(source_token_state[token]);
            // Same-node routes for one token can be produced by independent
            // dX CTAs.  The final top-k value must transitively publish every
            // route before the reducer claims it.
            ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
          }
          cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        }
#if MEGAMOE_DEVICE_TIMELINE
        if (threadIdx.x == 0U && timeline_dx_sample)
          MEGAMOE_TIMELINE_RECORD(a.timeline, DxTileEnd, Compute, 0U,
                                  encoded_task, a.epoch);
#endif
      } else if (!ActivationOnly && encoded_task != kBackwardV3IdleTask &&
                 task_kind == kBackwardV3Dw1TaskKind) {
        auto task = encoded_task & kBackwardV3TaskPayloadMask;
        const auto expert = static_cast<std::uint32_t>(task / dw1_per_expert);
        task %= dw1_per_expert;
        const auto ntile = task / dw1_mtiles;
        const auto mtile = task % dw1_mtiles;
        const auto reduction = persistent_v3::padded_expert_rows(
            a.dispatch.expert_row_count[expert]);
        if (threadIdx.x == 0U)
          atomicMin(&status->dw1_first_start_clock, device_globaltimer_ns());
        float accum[BackwardDwMixedMma::kNumAccum];
#pragma unroll
        for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
          accum[index] = 0.0F;
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto current = pipeline + iteration;
          const auto stage = current % kStages;
          const auto phase = (current / kStages) & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * (2U * kBStageElements));
          full_barriers[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0; k < 64U / BackwardDwMixedMma::K; ++k) {
            const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, 128, 64, 128, cutlass::bfloat16_t>(
                stage_a, warpgroup * 64U, k * BackwardDwMixedMma::K);
            const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
                cute::UMMA::Major::MN, 256, 64, 128, cutlass::bfloat16_t>(
                stage_b, 0U, k * BackwardDwMixedMma::K);
            BackwardDwMixedMma::wgmma(desc_a, desc_b, accum, 1);
          }
          deep_gemm::ptx::warpgroup_commit_batch();
          backward_v3_wgmma_wait0_accum128(accum);
          if ((threadIdx.x & 31U) == 0U)
            empty_barriers[stage].arrive();
        }
        pipeline += iterations;

        // Match the mature DeepGEMM epilogue: convert accumulator fragments
        // directly with STSM, then let four TMA stores move the swizzled
        // 128x256 tile.  The previous scalar global epilogue issued sixteen
        // uint4 stores per math thread for every dW1 task.
        const auto warp = threadIdx.x / 32U;
        const auto lane = threadIdx.x & 31U;
#pragma unroll
        for (std::uint32_t index = 0;
             index < BackwardDwMixedMma::kNumAccum / 4U; ++index) {
          constexpr std::uint32_t kBankGroupBytes = 16U;
          const auto atom = index / (64U / 8U);
          const auto in_atom = index % (64U / 8U);
          const auto row = lane;
          auto column = in_atom;
          column ^= row % (128U / kBankGroupBytes);
          auto *smem_pointer = reinterpret_cast<std::uint8_t *>(output_tile) +
                               warp * 16U * 128U + atom * 128U * 128U +
                               row * 128U + column * kBankGroupBytes;
          deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U], accum[index * 4U + 1U])),
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U + 2U], accum[index * 4U + 3U])),
              smem_pointer);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x < 4U) {
          const auto atom = threadIdx.x;
          cute::SM90_TMA_STORE_2D::copy(
              &dw1_store_map, output_tile + atom * 128U * 64U,
              static_cast<std::uint32_t>(ntile * kDwN + atom * 64ULL),
              static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                         mtile * 128ULL));
          cute::tma_store_arrive();
          backward_v3_tma_store_wait_complete();
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x == 0U) {
          const auto completed =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->dw1_tasks_complete.value, 1ULL) +
              1ULL;
          const auto now = device_globaltimer_ns();
          atomicMax(&status->dw1_last_complete_clock, now);
          if (completed == dw1_tasks)
            status->dw1_complete_clock = now;
        }
      } else if (!ActivationOnly && encoded_task != kBackwardV3IdleTask &&
                 task_kind == kBackwardV3Dw2TaskKind) {
        auto task = encoded_task & kBackwardV3TaskPayloadMask;
        const auto expert = static_cast<std::uint32_t>(task / dw2_per_expert);
        task %= dw2_per_expert;
        const auto ntile = task / dw2_mtiles;
        const auto mtile = task % dw2_mtiles;
        const auto reduction = persistent_v3::padded_expert_rows(
            a.dispatch.expert_row_count[expert]);
        if (threadIdx.x == 0U)
          atomicMin(&status->dw2_first_start_clock, device_globaltimer_ns());
        float accum[BackwardDwDirectMma::kNumAccum];
#pragma unroll
        for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
          accum[index] = 0.0F;
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto current = pipeline + iteration;
          const auto stage = current % kStages;
          const auto phase = (current / kStages) & 1U;
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_a + stage * kAStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              smem_b + stage * (2U * kBStageElements));
          full_barriers[stage].wait(phase);
#pragma unroll
          for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
            deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
          deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
          for (std::uint32_t k = 0; k < 64U / BackwardDwDirectMma::K; ++k) {
            const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
                cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
                stage_a, warpgroup * 64U, k * BackwardDwDirectMma::K);
            const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
                cute::UMMA::Major::K, 256, 64, 128, cutlass::bfloat16_t>(
                stage_b, 0U, k * BackwardDwDirectMma::K);
            BackwardDwDirectMma::wgmma(desc_a, desc_b, accum, 1);
          }
          deep_gemm::ptx::warpgroup_commit_batch();
          backward_v3_wgmma_wait0_accum128(accum);
          if ((threadIdx.x & 31U) == 0U)
            empty_barriers[stage].arrive();
        }
        pipeline += iterations;

        const auto warp = threadIdx.x / 32U;
        const auto lane = threadIdx.x & 31U;
#pragma unroll
        for (std::uint32_t index = 0;
             index < BackwardDwDirectMma::kNumAccum / 4U; ++index) {
          constexpr std::uint32_t kBankGroupBytes = 16U;
          const auto atom = index / (64U / 8U);
          const auto in_atom = index % (64U / 8U);
          const auto row = lane;
          auto column = in_atom;
          column ^= row % (128U / kBankGroupBytes);
          auto *smem_pointer = reinterpret_cast<std::uint8_t *>(output_tile) +
                               warp * 16U * 128U + atom * 128U * 128U +
                               row * 128U + column * kBankGroupBytes;
          deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U], accum[index * 4U + 1U])),
              __float22bfloat162_rn(
                  make_float2(accum[index * 4U + 2U], accum[index * 4U + 3U])),
              smem_pointer);
        }
        cute::tma_store_fence();
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x < 4U) {
          const auto atom = threadIdx.x;
          cute::SM90_TMA_STORE_2D::copy(
              &dw2_store_map, output_tile + atom * 128U * 64U,
              static_cast<std::uint32_t>(ntile * kDwN + atom * 64ULL),
              static_cast<std::uint32_t>(expert * a.hidden + mtile * 128ULL));
          cute::tma_store_arrive();
          backward_v3_tma_store_wait_complete();
        }
        cutlass::arch::NamedBarrier::sync(256U, kMathBarrier);
        if (threadIdx.x == 0U) {
          const auto completed =
              persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
                  control->dw2_tasks_complete.value, 1ULL) +
              1ULL;
          const auto now = device_globaltimer_ns();
          atomicMax(&status->dw2_last_complete_clock, now);
          if (completed == dw2_tasks)
            status->dw2_complete_clock = now;
        }
      } else if (threadIdx.x == 0U) {
        __nanosleep(64U);
      }

      cutlass::arch::NamedBarrier::sync(384U, kTaskDoneBarrier);
      if (threadIdx.x == 0U) {
        if (encoded_task != kBackwardV3IdleTask) {
          const auto completed_kind = encoded_task & kBackwardV3TaskKindMask;
          if (completed_kind == 0ULL || completed_kind == kBackwardV3DxTaskKind)
            activation_phase = (activation_phase + 1U) %
                               (a.v3_dv_phase_slots + a.v3_dx_phase_slots);
          else if (!ActivationOnly &&
                   (completed_kind == kBackwardV3Dw1TaskKind ||
                    completed_kind == kBackwardV3Dw2TaskKind))
            weight_phase = (weight_phase + 1U) % 3U;
        }
        if constexpr (ActivationCluster2) {
          const auto cluster_rank = cute::block_rank_in_cluster();
          auto &mailbox = activation_pair_mailboxes[compute_ordinal / 2U];
          backward_v3_finish_activation_pair(mailbox, cluster_rank,
                                             activation_pair_round);
          ++activation_pair_round;
          unsigned long long task0 = kBackwardV3IdleTask;
          unsigned long long task1 = kBackwardV3IdleTask;
          if (cluster_rank == 0U)
            backward_v3_claim_activation_pair(
                a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
                mblock_dx_cursor, token_state, dv_ntiles, dx_ntiles,
                activation_phase, compute_ordinal, reduce_ctas, task0, task1);
          task_shared = backward_v3_publish_or_wait_activation_pair(
              mailbox, cluster_rank, activation_pair_round, task0, task1);
        } else if constexpr (ActivationOnly) {
          task_shared = backward_v3_claim_activation_task(
              a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
              mblock_dx_cursor, dx_mblock_order, token_state, dv_ntiles,
              dx_ntiles, activation_phase, compute_ordinal, reduce_ctas,
              owned_dx_mblock_shared, next_dx_ntile_shared);
        } else {
          task_shared = backward_v3_claim_compute_task(
              a, *control, mblock_mapping, mblock_dv_cursor, mblock_ready,
              mblock_dx_cursor, token_state, expert_dv_complete,
              expert_dw1_cursor, expert_dw2_cursor, dv_ntiles, dx_ntiles,
              dw1_per_expert, dw2_per_expert, prefer_dw_role, activation_phase,
              weight_phase);
        }
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void backward_v3_initialize_pipeline(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control) {
  auto *activation_pair_mailboxes =
      peer_at<persistent_v3::BackwardActivationPairMailbox>(
          a.local_base, a.v3_activation_pair_mailboxes_offset);
  auto *mblock_mapping =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_mapping_offset);
  auto *mblock_dv_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dv_cursor_offset);
  auto *mblock_dv_complete =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dv_complete_offset);
  auto *mblock_ready =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_ready_offset);
  auto *mblock_dx_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dx_cursor_offset);
  auto *mblock_dx_complete =
      peer_at<unsigned long long>(a.local_base, a.v3_mblock_dx_complete_offset);
  auto *dy_ready_cache =
      peer_at<unsigned long long>(a.local_base, a.scheduler_offset) + 64ULL;
  const auto mblocks = backward_v3_mblock_count(a);
  auto *mblock_localized = dy_ready_cache + mblocks;
  auto *dx_mblock_order = mblock_localized + mblocks;
  auto *token_state =
      peer_at<unsigned long long>(a.local_base, a.v3_token_state_offset);
  auto *expert_padded_prefix = peer_at<unsigned long long>(
      a.local_base, a.v3_expert_padded_prefix_offset);
  auto *expert_dv_complete =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dv_complete_offset);
  auto *expert_dw1_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dw1_cursor_offset);
  auto *expert_dw2_cursor =
      peer_at<unsigned long long>(a.local_base, a.v3_expert_dw2_cursor_offset);
  auto *sigmoid_lut = peer_at<float>(a.local_base, a.v3_sigmoid_lut_offset);
  auto *forward_return_chunks = peer_at<ForwardReturnChunkRecord>(
      a.local_base, a.forward_return_chunk_directory_offset);
  auto *forward_return_chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  auto *forward_return_chunk_ready_count = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_ready_count_offset);
  auto *forward_return_source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_compact_index_offset);
  auto *forward_return_source_route_index =
      peer_at<unsigned long long>(a.local_base, a.claim_offset);
  auto *return_route_compact_index = peer_at<unsigned long long>(
      a.local_base, a.v3_return_route_compact_index_offset);
  auto *forward_scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  const auto experts = a.dispatch.sizes.owned_expert_count;
  const auto dv_ntiles =
      (a.intermediate + kBackwardV3DvN - 1ULL) / kBackwardV3DvN;
  const auto dx_ntiles = (a.hidden + 255ULL) / 256ULL;
  const auto dw1_mtiles = (2ULL * a.intermediate + 127ULL) / 128ULL;
  const auto dw_tile_n = static_cast<unsigned long long>(a.v3_dw_tile_n);
  const auto dw1_ntiles = (a.hidden + dw_tile_n - 1ULL) / dw_tile_n;
  const auto dw1_per_expert = dw1_mtiles * dw1_ntiles;
  const auto dw2_mtiles = (a.hidden + 127ULL) / 128ULL;
  const auto dw2_ntiles = (a.intermediate + dw_tile_n - 1ULL) / dw_tile_n;
  const auto dw2_per_expert = dw2_mtiles * dw2_ntiles;
  const auto chunk_count = forward_scheduler[4ULL + 2ULL * a.world];

  // Include generation setup in the device phase clock.  This makes it
  // directly comparable with the CUDA-event wall time and prevents startup
  // work from disappearing from the phase breakdown.
  if (threadIdx.x == 0U)
    status->start_clock = device_globaltimer_ns();

  // The production contract is exactly two nodes.  Return task sizing used
  // to be done by thread 0, which serialized a full packet-directory walk
  // before any of the other 131 resident role CTAs could leave setup.  At
  // 8192 tokens that alone costs multiple milliseconds on Hopper because the
  // dependent global loads cannot hide their latency.  Size the one remote
  // node cooperatively and reduce only twelve warp partials.
  constexpr std::uint32_t kSetupWarps = 384U / 32U;
  __shared__ unsigned long long setup_inbound_rows_by_warp[kSetupWarps];
  __shared__ unsigned long long setup_outbound_rows_by_warp[kSetupWarps];
  __shared__ unsigned long long setup_inbound_rows;
  __shared__ unsigned long long setup_outbound_rows;
  const auto remote_node = a.node == 0U ? 1U : 0U;
  unsigned long long thread_inbound_rows = 0ULL;
  for (std::uint64_t packet = threadIdx.x;
       packet < a.dispatch.sizes.packet_count; packet += blockDim.x) {
    if (a.dispatch.packet_headers[packet].destination_node != remote_node)
      continue;
    thread_inbound_rows += a.dispatch.packet_fanout_offsets[packet + 1ULL] -
                           a.dispatch.packet_fanout_offsets[packet];
  }
  unsigned long long thread_outbound_rows = 0ULL;
  for (std::uint64_t chunk = threadIdx.x; chunk < chunk_count;
       chunk += blockDim.x) {
    const auto record = forward_return_chunks[chunk];
    if (record.source_node != remote_node)
      continue;
    const auto end = static_cast<unsigned long long>(record.row_begin) +
                     static_cast<unsigned long long>(record.row_count);
    thread_outbound_rows = max(thread_outbound_rows, end);
  }
  for (std::uint32_t offset = 16U; offset != 0U; offset >>= 1U) {
    thread_inbound_rows += __shfl_down_sync(0xffffffffU, thread_inbound_rows,
                                            static_cast<int>(offset));
    thread_outbound_rows =
        max(thread_outbound_rows,
            __shfl_down_sync(0xffffffffU, thread_outbound_rows,
                             static_cast<int>(offset)));
  }
  if ((threadIdx.x & 31U) == 0U) {
    setup_inbound_rows_by_warp[threadIdx.x / 32U] = thread_inbound_rows;
    setup_outbound_rows_by_warp[threadIdx.x / 32U] = thread_outbound_rows;
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    unsigned long long inbound_rows = 0ULL;
    unsigned long long outbound_rows = 0ULL;
    for (std::uint32_t warp = 0U; warp < kSetupWarps; ++warp) {
      inbound_rows += setup_inbound_rows_by_warp[warp];
      outbound_rows = max(outbound_rows, setup_outbound_rows_by_warp[warp]);
    }
    setup_inbound_rows = inbound_rows;
    setup_outbound_rows = outbound_rows;
  }
  __syncthreads();

  if (threadIdx.x == 0U) {
    control.generation.control.terminal_epoch.value = 0ULL;
    control.generation.control.cancel_epoch.value = 0ULL;
    control.generation.control.error_code.value = 0ULL;
    control.generation.control.terminal_publish_count.value = 0ULL;
    control.roles_start_epoch.value = 0ULL;
    control.rdma_roles_complete.value = 0ULL;
    control.relay_roles_complete.value = 0ULL;
    control.compute_roles_complete.value = 0ULL;
    control.local_route_roles_complete.value = 0ULL;
    control.outbound_pack_roles_complete.value = 0ULL;
    control.inbound_dy_roles_complete.value = 0ULL;
    control.dv_scan_cursor.value = 0ULL;
    control.dx_scan_cursor.value = 0ULL;
    control.dw1_scan_cursor.value = 0ULL;
    control.dw2_scan_cursor.value = 0ULL;
    control.direct_dy_payload_roles_complete.value = 0ULL;
    control.dv_tasks_complete.value = 0ULL;
    control.mblocks_ready.value = 0ULL;
    control.dx_tasks_complete.value = 0ULL;
    control.dw1_tasks_complete.value = 0ULL;
    control.dw2_tasks_complete.value = 0ULL;
    control.return_chunks_packed.value = 0ULL;
    control.outbound_return_roles_complete.value = 0ULL;
    control.inbound_return_roles_complete.value = 0ULL;
    control.return_scatter_tasks_complete.value = 0ULL;
    control.reduce_scan_cursor.value = 0ULL;
    control.reduce_tasks_complete.value = 0ULL;
    control.mblocks = mblocks;
    control.dv_tasks = mblocks * dv_ntiles;
    control.dx_tasks = mblocks * dx_ntiles;
    control.dw1_tasks =
        static_cast<unsigned long long>(experts) * dw1_per_expert;
    control.dw2_tasks =
        static_cast<unsigned long long>(experts) * dw2_per_expert;
    control.return_chunks = chunk_count;
    control.reduce_tasks = a.dispatch.identity.num_tokens;
    control.outbound_return_tasks =
        (setup_outbound_rows + a.v3_return_group_rows - 1ULL) /
        a.v3_return_group_rows;
    control.inbound_return_tasks =
        (setup_inbound_rows + a.v3_return_group_rows - 1ULL) /
        a.v3_return_group_rows;
    if (a.dispatch.identity.num_nodes != 2U ||
        a.gin_context_count != persistent_v3::kRdmaCtaCount ||
        (a.v3_dw_tile_n != 128U && a.v3_dw_tile_n != 256U))
      persistent_v3::publish_error_once(control.generation, 701ULL);
    if (control.outbound_return_tasks > 64ULL * persistent_v3::kRdmaCtaCount ||
        control.inbound_return_tasks > 64ULL * persistent_v3::kRdmaCtaCount)
      persistent_v3::publish_error_once(control.generation, 702ULL);
    unsigned long long padded = 0ULL;
    unsigned long long dw1_reduction_iterations = 0ULL;
    unsigned long long dw2_reduction_iterations = 0ULL;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      expert_padded_prefix[expert] = padded;
      const auto expert_padded_rows = persistent_v3::padded_expert_rows(
          a.dispatch.expert_row_count[expert]);
      padded += expert_padded_rows;
      dw1_reduction_iterations += (expert_padded_rows / 64ULL) * dw1_per_expert;
      dw2_reduction_iterations += (expert_padded_rows / 64ULL) * dw2_per_expert;
    }
    expert_padded_prefix[experts] = padded;
    control.padded_rows = padded;

    status->dv_start_clock = ~0ULL;
    status->dy_dispatch_complete_clock = 0ULL;
    status->dv_complete_clock = 0ULL;
    status->dgate_complete_clock = 0ULL;
    status->transpose_complete_clock = 0ULL;
    status->dx_complete_clock = 0ULL;
    status->dx_start_clock = ~0ULL;
    status->dx_reduce_complete_clock = 0ULL;
    status->dw1_complete_clock = 0ULL;
    status->dw2_complete_clock = 0ULL;
    status->complete_clock = 0ULL;
    status->dw1_first_ready_clock = ~0ULL;
    status->dw1_first_start_clock = ~0ULL;
    status->dw1_last_complete_clock = 0ULL;
    status->dw2_first_ready_clock = ~0ULL;
    status->dw2_first_start_clock = ~0ULL;
    status->dw2_last_complete_clock = 0ULL;
    status->dv_tasks = control.dv_tasks;
    status->dx_tasks = control.dx_tasks;
    status->dw1_tasks = control.dw1_tasks;
    status->dw2_tasks = control.dw2_tasks;
    status->padded_reduction_rows = padded;
    const auto dv_reduction_iterations = control.dv_tasks * (a.hidden / 64ULL);
    const auto dx_reduction_iterations =
        control.dx_tasks * ((2ULL * a.intermediate) / 64ULL);
    status->wgmma_instructions =
        (dv_reduction_iterations + dx_reduction_iterations +
         dw1_reduction_iterations + dw2_reduction_iterations) *
        2ULL * (64ULL / ForwardMma::K);
    const auto dw1_tma_loads_per_iteration =
        1ULL + static_cast<unsigned long long>(a.v3_dw_tile_n) / 64ULL;
    const auto dw2_tma_loads_per_iteration =
        2ULL + static_cast<unsigned long long>(a.v3_dw_tile_n) / 128ULL;
    status->tma_loads =
        (dv_reduction_iterations + dx_reduction_iterations) * 2ULL +
        dw1_reduction_iterations * dw1_tma_loads_per_iteration +
        dw2_reduction_iterations * dw2_tma_loads_per_iteration;
    status->actual_dy_peer_bytes = 0ULL;
    status->gin_dispatch_put_count = 0ULL;
    status->gin_signal_count = 0ULL;
    status->gin_metadata_bytes = 0ULL;
    status->raw_cross_node_dy_bytes = 0ULL;
    status->gin_dispatch_payload_bytes = 0ULL;
    status->actual_dx_peer_bytes = 0ULL;
    status->gin_return_put_count = 0ULL;
    status->gin_return_payload_bytes = 0ULL;
    status->v3_local_delivery_complete_clock = 0ULL;
    status->v3_outbound_pack_complete_clock = 0ULL;
    status->v3_rdma_complete_clock = 0ULL;
    status->error = 0U;
    status->error_field = 0U;
    status->error_origin_rank = 0U;
    status->error_code = 0U;
    status->error_expected = 0ULL;
    status->error_actual = 0ULL;
  }
  __syncthreads();

  for (std::uint32_t pair = threadIdx.x;
       pair < persistent_v3::kBackwardActivationPairCapacity;
       pair += blockDim.x) {
    activation_pair_mailboxes[pair].sequence = 0ULL;
    activation_pair_mailboxes[pair].tasks[0] = kBackwardV3IdleTask;
    activation_pair_mailboxes[pair].tasks[1] = kBackwardV3IdleTask;
    activation_pair_mailboxes[pair].reserved = 0ULL;
  }
  for (std::uint32_t context = 0; context < persistent_v3::kRdmaCtaCount;
       ++context) {
    persistent_v3::initialize_spsc_ring(control.relay_to_rdma[context]);
    persistent_v3::initialize_spsc_ring(control.rdma_to_relay[context]);
    if (threadIdx.x == 0U) {
      control.generation.contexts[context].rx_published.value = 0ULL;
      control.generation.contexts[context].rx_consumed.value = 0ULL;
      control.generation.contexts[context].tx_published.value = 0ULL;
      control.generation.contexts[context].tx_local_complete.value = 0ULL;
    }
  }
  for (std::uint32_t kind = threadIdx.x; kind < persistent_v3::kTaskKindCount;
       kind += blockDim.x) {
    control.generation.pipeline.expected[kind].value = 0ULL;
    control.generation.pipeline.completed[kind].value = 0ULL;
  }
  for (std::uint64_t mblock = threadIdx.x; mblock < mblocks;
       mblock += blockDim.x) {
    std::uint64_t local_mblock = 0ULL;
    const auto expert =
        backward_wavefront_mblock_expert(a, mblock, local_mblock);
    mblock_mapping[mblock] = (static_cast<unsigned long long>(expert) << 32U) |
                             static_cast<std::uint32_t>(local_mblock);
    std::uint64_t expert_major = local_mblock;
    for (std::uint32_t prior = 0U; prior < expert; ++prior)
      expert_major +=
          persistent_v3::expert_mblocks(a.dispatch.expert_row_count[prior]);
    dx_mblock_order[expert_major] = mblock;
    mblock_dv_cursor[mblock] = 0ULL;
    mblock_dv_complete[mblock] = 0ULL;
    mblock_ready[mblock] = 0ULL;
    mblock_dx_cursor[mblock] = 0ULL;
    mblock_dx_complete[mblock] = 0ULL;
    dy_ready_cache[mblock] = 0ULL;
    mblock_localized[mblock] = 0ULL;
  }
  for (std::uint32_t expert = threadIdx.x; expert < experts;
       expert += blockDim.x) {
    expert_dv_complete[expert] = 0ULL;
    expert_dw1_cursor[expert] = 0ULL;
    expert_dw2_cursor[expert] = 0ULL;
  }
  // dV consumes BF16 A, so all possible sigmoid inputs fit in one compact
  // table.  Building it inside the existing setup CTA adds no launch and
  // preserves the exact __expf/divide expression previously executed for
  // every element of every dV tile.
  for (std::uint32_t bits = threadIdx.x; bits < kBackwardV3SigmoidLutEntries;
       bits += blockDim.x) {
    const auto value = bf16_bits_to_float(static_cast<std::uint16_t>(bits));
    sigmoid_lut[bits] = 1.0F / (1.0F + __expf(-value));
  }
  for (std::uint64_t token = threadIdx.x;
       token < a.dispatch.identity.num_tokens; token += blockDim.x)
    token_state[token] = 0ULL;
  for (std::uint64_t chunk = threadIdx.x; chunk < chunk_count;
       chunk += blockDim.x)
    forward_return_chunk_state[chunk] = kReturnChunkWaiting;
  for (std::uint64_t producer = threadIdx.x;
       producer < chunk_count * a.local_world_size; producer += blockDim.x)
    forward_return_chunk_ready_count[producer] = 0ULL;
  if (a.v3_direct_return_consume) {
    for (std::uint64_t route = threadIdx.x;
         route < a.dispatch.sizes.route_count; route += blockDim.x)
      return_route_compact_index[route] = 0ULL;
  }
  __syncthreads();
  // Materialize compact-row -> canonical route once for direct Return
  // scatter. In direct-consume mode also build the inverse route -> compact
  // map in Forward's now-dead compact->route allocation. Packets are
  // disjoint, so setup threads can fill both directions without atomics.
  for (std::uint64_t packet = threadIdx.x;
       packet < a.dispatch.sizes.packet_count; packet += blockDim.x) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
    for (std::uint64_t fanout = fanout_begin; fanout < fanout_end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = descriptor.local_token_id * a.dispatch.identity.top_k +
                         target.topk_slot;
      const auto compact = forward_return_source_compact_index[fanout];
      forward_return_source_route_index[compact] = route;
      if (a.v3_direct_return_consume)
        return_route_compact_index[route] = compact + 1ULL;
    }
  }
  __syncthreads();
  __threadfence();
  if (threadIdx.x == 0U)
    persistent_v3::store_release<::cuda::thread_scope_system>(
        control.setup_epoch.value, a.epoch | (1ULL << 63U));
  __syncthreads();
}

// The transport and compute grids are launched on independent streams.  One
// transport CTA owns generation setup and turns the peer-visible setup epochs
// into a device-local release for every resident role CTA.
__device__ __forceinline__ void backward_v3_release_role_start(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control) {
  if (threadIdx.x != 0U)
    return;
  const auto token = a.epoch | (1ULL << 63U);
  const auto first_local = a.node * a.local_world_size;
  for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
    const auto peer = first_local + local;
    auto *peer_control = peer_at<persistent_v3::BackwardPipelineControl>(
        a.peer_bases[peer], a.v3_control_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        peer_control->setup_epoch.value);
    while (ready.load(::cuda::memory_order_acquire) != token)
      __nanosleep(64U);
  }
  persistent_v3::store_release<::cuda::thread_scope_device>(
      control.roles_start_epoch.value, token);
}

__device__ __forceinline__ void
backward_v3_wait_role_start(const BackwardKernelArgs &a,
                            persistent_v3::BackwardPipelineControl &control) {
  if (threadIdx.x == 0U) {
    const auto token = a.epoch | (1ULL << 63U);
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               control.roles_start_epoch.value) != token)
      __nanosleep(64U);
  }
  __syncthreads();
}

__device__ __forceinline__ void backward_v3_finish_compute_role(
    const BackwardKernelArgs &a,
    persistent_v3::BackwardPipelineControl &control,
    BackwardDeviceStatus &status, std::uint32_t ordinal,
    std::uint32_t compute_ctas, bool terminal_owner = true) {
  if (threadIdx.x == 0U)
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.compute_roles_complete.value, 1ULL);
  __syncthreads();
  // Only ordinal zero owns terminal publication. Every other compute CTA may
  // retire as soon as it has published role completion. In the specialized
  // elastic launch this lets queued weight CTAs replace completed activation
  // CTAs one SM at a time, without a grid barrier or resident rendezvous.
  if (!terminal_owner || ordinal != 0U)
    return;
  if (ordinal == 0U && threadIdx.x == 0U) {
    while (!backward_v3_generation_cancelled(a, control) &&
           (persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.compute_roles_complete.value) != compute_ctas ||
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.rdma_roles_complete.value) !=
                persistent_v3::kRdmaCtaCount ||
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.relay_roles_complete.value) !=
                persistent_v3::kDefaultRelayCtaCount))
      __nanosleep(64U);
    const auto cancelled_epoch =
        persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.generation.control.cancel_epoch.value);
    const auto error_code =
        persistent_v3::load_acquire<::cuda::thread_scope_device>(
            control.generation.control.error_code.value);
    if (cancelled_epoch >= a.epoch) {
      status.error = 2U;
      status.error_field = 900U;
      status.error_origin_rank = a.rank;
      status.error_code = 2U;
      status.error_expected = 0ULL;
      status.error_actual = cancelled_epoch;
    } else if (error_code != 0ULL) {
      status.error = 1U;
      status.error_field = 901U;
      status.error_origin_rank = a.rank;
      status.error_code = static_cast<unsigned int>(error_code);
      status.error_expected = 0ULL;
      status.error_actual = error_code;
    }
    auto expected = 0ULL;
    if (persistent_v3::compare_exchange_acq_rel<::cuda::thread_scope_device>(
            control.generation.control.terminal_epoch.value, expected,
            a.epoch)) {
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.generation.control.terminal_publish_count.value, 1ULL);
      status.complete_clock = device_globaltimer_ns();
      status.dx_reduce_complete_clock = status.complete_clock;
    }
  }
  if (threadIdx.x == 0U)
    persistent_v3::wait_for_terminal(control.generation, a.epoch);
  __syncthreads();
}
#endif

#if MEGAMOE_BUILD_TESTING
// Diagnostic visibility cut: wait for one canonical dY M-block through the
// exact system-scope rail counters, then copy it into a same-GPU snapshot.
// Launching the production compute grid behind this kernel removes all
// peer-write/TMA visibility from the dV input while retaining the unchanged
// persistent dV/dX/dW implementation. It is intentionally testing-only and
// is never part of the production overlap path.
__global__ void backward_v3_snapshot_dy_kernel(BackwardKernelArgs a) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  backward_v3_wait_role_start(a, *control);
  const auto mblock = static_cast<std::uint64_t>(blockIdx.x);
  if (mblock >= control->mblocks)
    return;
  if (a.progress != nullptr && mblock == 0ULL && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
        a.progress[kBackwardEntered]);
    word.store(control->mblocks, ::cuda::memory_order_release);
  }
  const auto *mblock_mapping = peer_at<const unsigned long long>(
      a.local_base, a.v3_mblock_mapping_offset);
  const auto mapping = mblock_mapping[mblock];
  const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
  const auto local_mblock = static_cast<std::uint32_t>(mapping);
  const auto expected =
      backward_v3_mblock_expected_rows(a, expert, local_mblock);
  // This diagnostic intentionally waits for the complete dY transport phase,
  // not just the M-block counter.  That lets it snapshot the final canonical
  // bytes even when the per-M-block counter protocol itself is under test.
  // Every copying thread then performs all system acquires itself instead of
  // relying on a CTA barrier to transfer a remote-memory acquire.
  while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control->local_route_roles_complete.value) !=
             persistent_v3::kDefaultRelayCtaCount ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control->inbound_dy_roles_complete.value) !=
             persistent_v3::kDefaultRelayCtaCount)
    __nanosleep(64U);
  if (a.progress != nullptr && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
        a.progress[kBackwardSetupComplete]);
    word.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
  }
  const auto arrived =
      backward_v3_load_dy_arrived_rows(a, mblock, expert, local_mblock);
  if (a.progress != nullptr && threadIdx.x == 0U && arrived != expected) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
        a.progress[kBackwardDyComplete]);
    word.store((mblock << 32U) | min(arrived, 0xffffffffULL),
               ::cuda::memory_order_release);
  }
  asm volatile("fence.proxy.async.global;" : : : "memory");

  const auto row = a.dispatch.expert_row_prefix[expert] +
                   static_cast<std::uint64_t>(local_mblock) * 128ULL;
  const auto valid_rows = static_cast<std::uint32_t>(
      min(128ULL, a.dispatch.expert_row_count[expert] -
                      static_cast<std::uint64_t>(local_mblock) * 128ULL));
  const auto *source =
      peer_at<const uint4>(a.local_base, a.dy_canonical_offset);
  auto *destination = peer_at<uint4>(a.local_base, a.debug_dv_offset);
  const auto vectors_per_row = a.hidden / 8ULL;
  const auto elements =
      static_cast<std::uint64_t>(valid_rows) * vectors_per_row;
  for (std::uint64_t item = threadIdx.x; item < elements; item += blockDim.x) {
    const auto local_row = item / vectors_per_row;
    const auto vector = item % vectors_per_row;
    destination[(row + local_row) * vectors_per_row + vector] =
        source[(row + local_row) * vectors_per_row + vector];
  }
  __threadfence();
  __syncthreads();
  if (a.progress != nullptr && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> word(
        a.progress[kBackwardDvStarted]);
    word.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
  }
#else
  (void)a;
#endif
}
#endif

__global__
__launch_bounds__(384, 1) void backward_v3_dv_dw2_pipeline_probe_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    const __grid_constant__ CUtensorMap dab_dx_map,
    const __grid_constant__ CUtensorMap w1_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dw1_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  constexpr std::uint32_t kRoleCtas =
      persistent_v3::kRdmaCtaCount + persistent_v3::kDefaultRelayCtaCount;
  const auto token = a.epoch | (1ULL << 63U);
  if (blockIdx.x == kRoleCtas)
    backward_v3_initialize_pipeline(a, *control);
  // One CTA per GPU performs the same-node setup agreement.  In particular,
  // no dX producer may increment a proxy's freshly reset ReturnChunk counts
  // until that proxy has published its setup epoch.
  if (blockIdx.x == kRoleCtas && threadIdx.x == 0U) {
    const auto first_local = a.node * a.local_world_size;
    for (std::uint32_t local = 0; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *peer_control = peer_at<persistent_v3::BackwardPipelineControl>(
          a.peer_bases[peer], a.v3_control_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          peer_control->setup_epoch.value);
      while (ready.load(::cuda::memory_order_acquire) != token)
        __nanosleep(64U);
    }
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control->roles_start_epoch.value, token);
  }
  if (threadIdx.x == 0U) {
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               control->roles_start_epoch.value) != token)
      __nanosleep(64U);
  }
  __syncthreads();

  const auto compute_ctas = static_cast<std::uint32_t>(gridDim.x) - kRoleCtas;
  if (blockIdx.x < persistent_v3::kRdmaCtaCount) {
#if __has_include(<nccl_device.h>)
    backward_v3_rdma_role(a, *control, static_cast<std::uint32_t>(blockIdx.x));
#else
    if (threadIdx.x == 0U)
      persistent_v3::publish_error_once(control->generation, 501ULL);
#endif
  } else if (blockIdx.x < kRoleCtas) {
#if __has_include(<nccl_device.h>)
    backward_v3_relay_role(a, *control,
                           static_cast<std::uint32_t>(blockIdx.x) -
                               persistent_v3::kRdmaCtaCount,
                           storage);
#else
    if (threadIdx.x == 0U)
      persistent_v3::publish_error_once(control->generation, 502ULL);
#endif
  } else {
    const auto ordinal = static_cast<std::uint32_t>(blockIdx.x) - kRoleCtas;
    backward_v3_dv_dw2_compute_role_inline(
        a, dy_map, w2_transpose_map, dab_dx_map, w1_transpose_map, x_mn_map,
        dab_transpose_map, dy_mn_map, u_transpose_map, dw1_store_map,
        dw2_store_map, storage, ordinal, compute_ctas, compute_ctas);
    backward_v3_finish_compute_role(a, *control, *status, ordinal,
                                    compute_ctas);
  }
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dy_map;
  (void)w2_transpose_map;
  (void)dab_dx_map;
  (void)w1_transpose_map;
  (void)x_mn_map;
  (void)dab_transpose_map;
  (void)dy_mn_map;
  (void)u_transpose_map;
  (void)dw1_store_map;
  (void)dw2_store_map;
#endif
}

// Production split-role entry points.  Supplying the same near-full dynamic
// shared-memory reservation to both launches enforces one CTA per SM, so the
// two grids occupy exactly 8 RDMA + 8 Relay + 116 Compute SMs on H200.
__global__
__launch_bounds__(384,
                  1) void backward_v3_transport_kernel(BackwardKernelArgs a) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  if (blockIdx.x == 0U)
    backward_v3_initialize_pipeline(a, *control);
  if (blockIdx.x == 0U)
    backward_v3_release_role_start(a, *control);
  backward_v3_wait_role_start(a, *control);
  if (blockIdx.x < persistent_v3::kRdmaCtaCount) {
#if __has_include(<nccl_device.h>)
    backward_v3_rdma_role(a, *control, static_cast<std::uint32_t>(blockIdx.x));
#else
    if (threadIdx.x == 0U)
      persistent_v3::publish_error_once(control->generation, 501ULL);
#endif
  } else {
#if __has_include(<nccl_device.h>)
    backward_v3_relay_role(a, *control,
                           static_cast<std::uint32_t>(blockIdx.x) -
                               persistent_v3::kRdmaCtaCount,
                           storage);
#else
    if (threadIdx.x == 0U)
      persistent_v3::publish_error_once(control->generation, 502ULL);
#endif
  }
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
#endif
}

__global__ __launch_bounds__(384, 1) void backward_v3_compute_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    const __grid_constant__ CUtensorMap dab_dx_map,
    const __grid_constant__ CUtensorMap w1_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dw1_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    std::uint32_t reduce_ctas, std::uint32_t total_compute_ctas,
    bool terminal_owner) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  backward_v3_wait_role_start(a, *control);
  const auto ordinal = static_cast<std::uint32_t>(blockIdx.x);
  const auto compute_ctas = static_cast<std::uint32_t>(gridDim.x);
  backward_v3_dv_dw2_compute_role_inline<true>(
      a, dy_map, w2_transpose_map, dab_dx_map, w1_transpose_map, x_mn_map,
      dab_transpose_map, dy_mn_map, u_transpose_map, dw1_store_map,
      dw2_store_map, storage, ordinal, compute_ctas, reduce_ctas);
  backward_v3_finish_compute_role(a, *control, *status, ordinal,
                                  total_compute_ctas, terminal_owner);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dy_map;
  (void)w2_transpose_map;
  (void)dab_dx_map;
  (void)w1_transpose_map;
  (void)x_mn_map;
  (void)dab_transpose_map;
  (void)dy_mn_map;
  (void)u_transpose_map;
  (void)dw1_store_map;
  (void)dw2_store_map;
  (void)reduce_ctas;
  (void)total_compute_ctas;
  (void)terminal_owner;
#endif
}

// Return reduction is deliberately its own phase.  dV/dX can retire as soon
// as their Tensor Core work is complete, allowing the full-SM dW phases to
// overlap the still-running put/relay return path.  This final CUDA-core grid
// then waits only for individual ready tokens and owns terminal publication.
__global__ __launch_bounds__(384, 1) void backward_v3_reduce_compute_kernel(
    BackwardKernelArgs a, std::uint32_t total_compute_ctas) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  (void)storage;
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *token_state =
      peer_at<unsigned long long>(a.local_base, a.v3_token_state_offset);
  auto *dx_return =
      peer_at<transport::BFloat16>(a.local_base, a.dx_return_offset);
  auto *dx_combined =
      peer_at<transport::BFloat16>(a.local_base, a.dx_combined_offset);
  __shared__ unsigned long long tokens[12];
  __shared__ unsigned int token_count;
  __shared__ unsigned int phase_done;

  backward_v3_wait_role_start(a, *control);
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                          control->reduce_tasks, a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                          control->reduce_tasks, a.epoch);
  while (true) {
    if (threadIdx.x == 0U) {
      token_count = 0U;
      phase_done =
          backward_v3_generation_cancelled(a, *control) ||
                  persistent_v3::load_acquire<::cuda::thread_scope_device>(
                      control->reduce_tasks_complete.value) ==
                      control->reduce_tasks
              ? 1U
              : 0U;
      while (phase_done == 0U && token_count < 12U) {
        const auto task =
            backward_v3_try_claim_reduce_task(a, *control, token_state);
        if (task == kBackwardV3IdleTask)
          break;
        tokens[token_count++] = task & kBackwardV3TaskPayloadMask;
      }
    }
    __syncthreads();
    if (phase_done != 0U)
      break;
    if (token_count == 0U) {
      if (threadIdx.x == 0U)
        __nanosleep(64U);
      __syncthreads();
      continue;
    }

    for (std::uint32_t item = 0U; item < token_count; ++item) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          token_state[tokens[item]]);
      (void)ready.load(::cuda::memory_order_acquire);
    }
    // One consumer-side proxy transition covers the whole ready batch.  GIN
    // writes all remote routes before Relay extends each token's system
    // release sequence, so no per-token fence is needed after these acquires.
    if (a.v3_direct_return_consume)
      asm volatile("fence.proxy.async.global;" : : : "memory");
    for (std::uint32_t item = 0U; item < token_count; ++item)
      backward_v3_reduce_token<true>(a, tokens[item], dx_return, dx_combined);
    __syncthreads();
    if (threadIdx.x == 0U) {
      for (std::uint32_t item = 0U; item < token_count; ++item) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            state(token_state[tokens[item]]);
        state.store(
            kBackwardV3TokenCompleteBit |
                static_cast<unsigned long long>(a.dispatch.identity.top_k),
            ::cuda::memory_order_release);
      }
      const auto completed =
          persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
              control->reduce_tasks_complete.value, token_count) +
          token_count;
      if (completed == control->reduce_tasks)
        status->dx_reduce_complete_clock = device_globaltimer_ns();
    }
    __syncthreads();
  }

  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                          control->reduce_tasks, a.epoch);
  backward_v3_finish_compute_role(a, *control, *status,
                                  static_cast<std::uint32_t>(blockIdx.x),
                                  total_compute_ctas, true);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)total_compute_ctas;
#endif
}

// Dynamic production compute grid.  Transport remains compile-isolated in
// its fixed 8-RDMA/8-Relay grid, while every remaining resident CTA draws
// activation, reduction, dW1, and dW2 work from the shared ready queues.  The
// mutually exclusive task bodies let ptxas reuse the same register budget
// without relying on over-subscribed persistent grids or a grid barrier.
__global__ __launch_bounds__(384, 1) void backward_v3_dynamic_compute_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    const __grid_constant__ CUtensorMap dab_dx_map,
    const __grid_constant__ CUtensorMap w1_transpose_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dw1_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    std::uint32_t total_compute_ctas) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  extern __shared__ __align__(1024) unsigned char storage[];
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL, a.epoch);
  auto *control = peer_at<persistent_v3::BackwardPipelineControl>(
      a.local_base, a.v3_control_offset);
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  backward_v3_wait_role_start(a, *control);
  const auto ordinal = static_cast<std::uint32_t>(blockIdx.x);
  const auto compute_ctas = static_cast<std::uint32_t>(gridDim.x);
  backward_v3_dv_dw2_compute_role_inline<false>(
      a, dy_map, w2_transpose_map, dab_dx_map, w1_transpose_map, x_mn_map,
      dab_transpose_map, dy_mn_map, u_transpose_map, dw1_store_map,
      dw2_store_map, storage, ordinal, compute_ctas, compute_ctas);
  backward_v3_finish_compute_role(a, *control, *status, ordinal,
                                  total_compute_ctas);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
#else
  (void)a;
  (void)dy_map;
  (void)w2_transpose_map;
  (void)dab_dx_map;
  (void)w1_transpose_map;
  (void)x_mn_map;
  (void)dab_transpose_map;
  (void)dy_mn_map;
  (void)u_transpose_map;
  (void)dw1_store_map;
  (void)dw2_store_map;
  (void)total_compute_ctas;
#endif
}
