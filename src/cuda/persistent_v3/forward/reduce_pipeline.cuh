#pragma once

// Token combine runs after the CTA's final FC2 tile. WGMMA/TMA live ranges have
// ended at this point, so the caller redistributes their fixed register budget
// uniformly across all three warpgroups. The former TMA producer contributes
// four warps instead of leaving one third of the SM idle during the tail.

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

namespace forward_compute_detail {

template <bool StaticLocal, bool SingleRoute, bool BulkReady = false>
__device__ __forceinline__ void reduce_one_token_impl(
    const ForwardKernelArgs &a, std::uint64_t token,
    TransportDeviceStatus *status, transport::BFloat16 *contribution,
    unsigned long long *route_ready, transport::BFloat16 *combined,
    transport_state::GenerationCompletionVector *completion) {
  const auto lane = threadIdx.x & 31U;
  const auto top_k = SingleRoute ? 1U : a.dispatch.identity.top_k;
#if WEFTMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal && !BulkReady) {
    if (a.use_gin && lane == 0U)
      claim_completion_tasks(completion, transport_state::TaskClass::Reduce);
  }
#endif
  if (lane == 0U && status->reduced_tokens == ~0ULL)
    atomicMin(&status->reduced_tokens, device_globaltimer_ns());

  unsigned long long lane_route = 0ULL;
  unsigned long long lane_row = 0ULL;
  unsigned long long lane_pointer = 0ULL;
  if (lane < top_k) {
    const auto slot =
        SingleRoute ? 0U : a.dispatch.combine_order[token * top_k + lane];
    lane_route = token * top_k + slot;
    lane_row = forward_contribution_row(a, lane_route);
    if constexpr (!StaticLocal && BulkReady)
      lane_pointer = reinterpret_cast<unsigned long long>(
          static_v4::network_contribution(a, lane_route));
    // Queue/scan consumers have already acquired every contribution. The
    // static scheduler instead waits for each route's local arrival marker.
    if constexpr (!BulkReady) {
      if (!a.forward_dynamic_reduce_queue && !a.forward_dynamic_reduce_scan) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
          route_ready[lane_route]);
      while (ready.load(::cuda::memory_order_acquire) < execution_epoch(a) &&
             !forward_cancelled(a))
        __nanosleep(forward_tuning::kTaskIdleNanosleep);
      }
    }
  }
  __syncwarp();

  auto *output = reinterpret_cast<uint4 *>(combined + token * a.hidden);
  for (std::uint64_t vector = lane; vector < a.hidden / 8ULL; vector += 32ULL) {
    float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    for (std::uint32_t order = 0; order < top_k; ++order) {
      const auto row =
          __shfl_sync(0xffffffffU, lane_row, order);
      const auto *source = contribution + row * a.hidden;
      if constexpr (!StaticLocal && BulkReady)
        source = reinterpret_cast<const transport::BFloat16 *>(
            __shfl_sync(0xffffffffU, lane_pointer, order));
      const auto packed = reinterpret_cast<const uint4 *>(source)[vector];
      const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (std::uint32_t word = 0; word < 4U; ++word) {
        sums[word * 2U] = __fadd_rn(
            sums[word * 2U],
            bf16_bits_to_float(static_cast<std::uint16_t>(words[word])));
        sums[word * 2U + 1U] = __fadd_rn(
            sums[word * 2U + 1U],
            bf16_bits_to_float(static_cast<std::uint16_t>(words[word] >> 16U)));
      }
    }
    unsigned int words[4];
#pragma unroll
    for (std::uint32_t word = 0; word < 4U; ++word)
      words[word] =
          static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U])) |
          (static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U + 1U]))
           << 16U);
    output[vector] = make_uint4(words[0], words[1], words[2], words[3]);
  }
  __syncwarp();
  if (a.forward_dynamic_reduce_queue && lane == 0U) {
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control->reduce_tokens_complete.value, 1ULL);
  }
#if WEFTMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal && !BulkReady) {
    if (a.use_gin && lane == 0U)
      complete_completion_tasks(completion, transport_state::TaskClass::Reduce);
  }
#endif
}

template <bool StaticLocal, bool BulkReady = false>
__device__ __forceinline__ void
reduce_one_token(const ForwardKernelArgs &a, std::uint64_t token,
                 TransportDeviceStatus *status,
                 transport::BFloat16 *contribution,
                 unsigned long long *route_ready, transport::BFloat16 *combined,
                 transport_state::GenerationCompletionVector *completion) {
  if (a.dispatch.identity.top_k == 1U)
    reduce_one_token_impl<StaticLocal, true, BulkReady>(
        a, token, status, contribution, route_ready, combined, completion);
  else
    reduce_one_token_impl<StaticLocal, false, BulkReady>(
        a, token, status, contribution, route_ready, combined, completion);
}

template <bool StaticLocal, unsigned Threads = kThreads,
          bool BulkReady = false>
__device__ __noinline__ void reduce_role(const ForwardKernelArgs &a,
                                         std::uint32_t role_index,
                                         std::uint32_t role_count) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *contribution =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  auto *route_ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_epoch_offset);
  auto *combined = a.output;
  transport_state::GenerationCompletionVector *completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin)
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
  }

  constexpr std::uint32_t kPhysicalWarpsPerCta = Threads / 32U;
  constexpr std::uint32_t kReduceWarps =
      forward_tuning::kReduceWarps < kPhysicalWarpsPerCta
          ? forward_tuning::kReduceWarps : kPhysicalWarpsPerCta;
  constexpr std::uint32_t kTokensPerClaim =
      forward_tuning::kReduceTokensPerClaim < kReduceWarps
          ? forward_tuning::kReduceTokensPerClaim : kReduceWarps;
  static_assert(kReduceWarps <= kPhysicalWarpsPerCta);
  static_assert(kTokensPerClaim <= kReduceWarps);
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                          a.dispatch.identity.num_tokens, execution_epoch(a));
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                          a.dispatch.identity.num_tokens, execution_epoch(a));

  if (a.forward_dynamic_reduce_queue) {
    __shared__ unsigned long long batch_begin;
    __shared__ unsigned int batch_count;
    __shared__ unsigned int queue_done;
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    auto *ready_token = peer_at<unsigned long long>(
        a.local_base, a.forward_reduce_ready_token_offset);

    while (true) {
      if (threadIdx.x == 0U) {
        batch_count = 0U;
        queue_done = 0U;
        if (forward_cancelled(a)) {
          queue_done = 1U;
        } else {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_device>
              head(control->reduce_ready_head.value);
          // Compute has finished this CTA's GEMM work. Reserve a finite
          // batch once, then acquire each queue entry before reading payload.
          // Transport helpers still only claim already-published entries.
          const auto candidate = head.fetch_add(
              kTokensPerClaim, ::cuda::memory_order_relaxed);
          const auto token_count = static_cast<unsigned long long>(
              a.dispatch.identity.num_tokens);
          if (candidate >= token_count) {
            // Ownership is not completion: finalization also waits for every
            // claimed token's completed counter, including helper consumers.
            queue_done = 1U;
          } else {
            batch_begin = candidate;
            batch_count = static_cast<unsigned int>(min(
                static_cast<unsigned long long>(kTokensPerClaim),
                token_count - candidate));
          }

        }
      }
      cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
      if (queue_done != 0U)
        break;

      if (warp < batch_count) {
        const auto slot = batch_begin + warp;
        unsigned long long token = 0ULL;
        unsigned int token_ready = 0U;
        if (lane == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published(ready_token[slot]);
          auto encoded = published.load(::cuda::memory_order_acquire);
          while (encoded == 0ULL && !forward_cancelled(a)) {
            __nanosleep(forward_tuning::kTaskIdleNanosleep);
            encoded = published.load(::cuda::memory_order_acquire);
          }
          if (encoded != 0ULL) {
#if WEFTMOE_DEVICE_TIMELINE
            constexpr unsigned long long kTokenMask =
                (1ULL << 56U) - 1ULL;
            const auto publisher_kind =
                static_cast<std::uint32_t>(encoded >> 56U);
            token = (encoded & kTokenMask) - 1ULL;
            WEFTMOE_TIMELINE_RECORD_IF(
                a.timeline, true, ReduceTokenReadyObserved, Reduce,
                publisher_kind, token, execution_epoch(a));
#else
            token = encoded - 1ULL;
#endif
            token_ready = 1U;
          }
        }
        token = __shfl_sync(0xffffffffU, token, 0U);
        token_ready = __shfl_sync(0xffffffffU, token_ready, 0U);
        if (token_ready != 0U)
          reduce_one_token<StaticLocal, BulkReady>(
              a, token, status, contribution, route_ready, combined,
              completion);
      }
      cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
    }
  } else if (a.forward_dynamic_reduce_scan) {
    __shared__ unsigned long long batch_begin;
    __shared__ unsigned int scan_done;
    __shared__ unsigned int batch_progress;
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    auto *token_state = peer_at<unsigned int>(
        a.local_base, a.forward_reduce_route_count_offset);

    while (true) {
      if (threadIdx.x == 0U) {
        scan_done = 0U;
        batch_progress = 0U;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(control->reduce_tokens_complete.value);
        if (forward_cancelled(a) ||
            completed.load(::cuda::memory_order_acquire) >=
                a.dispatch.identity.num_tokens) {
          scan_done = 1U;
        } else {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              cursor(control->reduce_ready_head.value);
          batch_begin = cursor.fetch_add(kTokensPerClaim,
                                         ::cuda::memory_order_relaxed);
        }
      }
      cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
      if (scan_done != 0U)
        break;

      if (warp < kTokensPerClaim) {
        const auto token =
            (batch_begin + warp) % a.dispatch.identity.num_tokens;
        unsigned int open = 0U;
        if (lane == 0U) {
          ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device> state(
              token_state[token]);
          open = state.load(::cuda::memory_order_relaxed) == 0U ? 1U : 0U;
        }
        open = __shfl_sync(0xffffffffU, open, 0U);

        bool route_is_ready = open != 0U;
        if (route_is_ready && lane < a.dispatch.identity.top_k) {
          const auto route = token * a.dispatch.identity.top_k + lane;
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published(route_ready[route]);
          route_is_ready = published.load(::cuda::memory_order_acquire) >=
                           execution_epoch(a);
        }
        const auto ready_lanes = __ballot_sync(0xffffffffU, route_is_ready);
        const auto route_mask = (1U << a.dispatch.identity.top_k) - 1U;
        unsigned int owner = 0U;
        if (lane == 0U && open != 0U &&
            (ready_lanes & route_mask) == route_mask) {
          owner = atomicCAS(&token_state[token], 0U, 1U) == 0U ? 1U : 0U;
          if (owner != 0U)
            atomicAdd(&batch_progress, 1U);
        }
        owner = __shfl_sync(0xffffffffU, owner, 0U);
        if (owner != 0U) {
          reduce_one_token<StaticLocal, BulkReady>(
              a, token, status, contribution, route_ready, combined,
              completion);
          if (lane == 0U) {
            ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device>
                state(token_state[token]);
            state.store(2U, ::cuda::memory_order_release);
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_device>
                completed(control->reduce_tokens_complete.value);
            completed.fetch_add(1ULL, ::cuda::memory_order_release);
          }
        }
      }
      cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
      if (threadIdx.x == 0U && batch_progress == 0U)
        __nanosleep(forward_tuning::kSetupNanosleep);
      cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
    }
  } else {
    if (warp < kReduceWarps) {
      for (std::uint64_t token =
               static_cast<std::uint64_t>(role_index) * kReduceWarps + warp;
           token < a.dispatch.identity.num_tokens;
           token += static_cast<std::uint64_t>(role_count) * kReduceWarps) {
        if constexpr (!BulkReady)
          if (forward_cancelled(a))
            break;
        reduce_one_token<StaticLocal, BulkReady>(
            a, token, status, contribution, route_ready, combined,
            completion);
      }
    }
  }

  if (lane == 0U)
    atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
  WEFTMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                          a.dispatch.identity.num_tokens, execution_epoch(a));
  __threadfence();
  cutlass::arch::NamedBarrier::sync(
          Threads, static_cast<cutlass::arch::ReservedNamedBarriers>(7U));
  if (threadIdx.x == 0U) {
    auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
        a.local_base, a.forward_v3_control_offset);
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control->compute_reduce_roles_complete.value, 1ULL);
  }
}

} // namespace forward_compute_detail

#endif // CUTE_ARCH_MMA_SM90A_ENABLED
