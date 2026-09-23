#pragma once

namespace forward_reduce_detail {

// Zero denotes no published work. The CAS never reserves an unpublished entry.
__device__ __forceinline__ unsigned long long
try_claim_reduce_token(const ForwardKernelArgs &a) {
  if (!a.forward_dynamic_reduce_queue || forward_cancelled(a))
    return 0ULL;
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  auto *queue = peer_at<unsigned long long>(
      a.local_base, a.forward_reduce_ready_token_offset);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> head(
      control->reduce_ready_head.value);
  auto slot = head.load(::cuda::memory_order_relaxed);
  if (slot >= a.dispatch.identity.num_tokens)
    return 0ULL;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> entry(
      queue[slot]);
  const auto ready = entry.load(::cuda::memory_order_acquire);
  if (ready == 0ULL || !head.compare_exchange_strong(
                           slot, slot + 1ULL, ::cuda::memory_order_acq_rel,
                           ::cuda::memory_order_relaxed))
    return 0ULL;
  return ready;
}

// Shared payload view for the Compute and Relay Reduce helpers.
struct ReduceTokenView {
  persistent_v3::ForwardPipelineControl *control;
  TransportDeviceStatus *status;
  const transport::BFloat16 *local;
  transport::BFloat16 *output;
  const std::uint32_t *order;
  const std::uint64_t *return_base;
  const std::uint32_t *route_expert;
  const std::uint64_t *local_offset;
  std::uint64_t hidden;
  std::uint32_t top_k;
#if WEFTMOE_DEBUG_VALIDATE
  transport_state::GenerationCompletionVector *completion;
#endif
};

__device__ __forceinline__ ReduceTokenView
reduce_token_view(const ForwardKernelArgs &a) {
  ReduceTokenView v{};
  v.control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  v.status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  v.local =
      peer_at<const transport::BFloat16>(a.local_base, a.contribution_offset);
  v.output = a.output;
  v.order = a.dispatch.combine_order;
  v.return_base = a.forward_compact_return ? a.dispatch.return_global_expert_base : nullptr;
  v.route_expert = a.dispatch.expert_id;
  v.local_offset = a.dispatch.local_offset;
  v.hidden = a.hidden;
  v.top_k = a.dispatch.identity.top_k;
#if WEFTMOE_DEBUG_VALIDATE
  v.completion =
      a.use_gin ? static_cast<transport_state::GenerationCompletionVector *>(
                      a.peer_bases[a.world])
                : nullptr;
#endif
  return v;
}

// Caller owns this token. First-warp execution needs no CTA-wide barrier.
__device__ __forceinline__ bool
reduce_token_payload(const ReduceTokenView &v,
                     unsigned long long encoded,
                     std::uint64_t dense_row_base = ~std::uint64_t{0}) {
  if (encoded == 0ULL)
    return false;
  const auto lane = threadIdx.x & 31U;
  // The queue acquire may have run on lane 0 only. A shuffle broadcasts the
  // token index but is not a memory fence; hand payload visibility to all lanes.
  __syncwarp();
#if WEFTMOE_DEVICE_TIMELINE
  encoded &= (1ULL << 56U) - 1ULL;
#endif
  const auto token = encoded - 1ULL;
#if WEFTMOE_DEBUG_VALIDATE
  if (v.completion && lane == 0U)
    claim_completion_tasks(v.completion, transport_state::TaskClass::Reduce);
#endif
  if (lane == 0U)
    atomicMin(&v.status->reduced_tokens, device_globaltimer_ns());
  for (std::uint64_t column = lane * 8ULL; column < v.hidden;
       column += 256ULL) {
    float sum[8] = {};
    for (std::uint32_t order = 0U; order < v.top_k; ++order) {
      const auto slot = v.order[token * v.top_k + order];
      const auto route = token * v.top_k + slot;
      const auto row = v.return_base
          ? v.return_base[v.route_expert[route]] + v.local_offset[route]
          : (dense_row_base == ~std::uint64_t{0} ? route : dense_row_base + slot);
      const auto *source = v.local + row * v.hidden + column;
      const auto packed = *reinterpret_cast<const uint4 *>(source);
      const unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (std::uint32_t i = 0U; i < 4U; ++i) {
        sum[2U * i] =
            __fadd_rn(sum[2U * i],
                      bf16_bits_to_float(static_cast<std::uint16_t>(words[i])));
        sum[2U * i + 1U] = __fadd_rn(
            sum[2U * i + 1U],
            bf16_bits_to_float(static_cast<std::uint16_t>(words[i] >> 16U)));
      }
    }
    unsigned int words[4];
#pragma unroll
    for (std::uint32_t i = 0U; i < 4U; ++i)
      words[i] =
          static_cast<unsigned int>(float_to_bf16_bits(sum[2U * i])) |
          (static_cast<unsigned int>(float_to_bf16_bits(sum[2U * i + 1U]))
           << 16U);
    *reinterpret_cast<uint4 *>(v.output + token * v.hidden + column) =
        make_uint4(words[0], words[1], words[2], words[3]);
  }
  __syncwarp();
  if (lane == 0U) {
#if WEFTMOE_DEBUG_VALIDATE
    if (v.completion)
      complete_completion_tasks(v.completion,
                                transport_state::TaskClass::Reduce);
#endif
    atomicMax(&v.status->reduce_blocks_done, device_globaltimer_ns());
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        v.control->reduce_tokens_complete.value, 1ULL);
  }
  return true;
}

__device__ __forceinline__ bool
reduce_claimed_token(const ForwardKernelArgs &a, unsigned long long encoded) {
  return reduce_token_payload(reduce_token_view(a), encoded);
}

__device__ __forceinline__ bool
try_reduce_one_token(const ForwardKernelArgs &a) {
  const auto lane = threadIdx.x & 31U;
  auto encoded = lane == 0U ? try_claim_reduce_token(a) : 0ULL;
  encoded = __shfl_sync(0xffffffffU, encoded, 0);
  return reduce_claimed_token(a, encoded);
}

// Relay CTAs with no remaining primary work can consume ready tokens too.
// Queue exhaustion is distinct from completion; finalization waits for both.
__device__ __forceinline__ void drain_reduce_queue(const ForwardKernelArgs &a) {
  if (!a.forward_dynamic_reduce_queue || threadIdx.x >= 32U)
    return;
  auto *control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  while (true) {
    unsigned int done = 0U;
    if (threadIdx.x == 0U)
      done = forward_cancelled(a) ||
             persistent_v3::load_acquire<::cuda::thread_scope_device>(
                 control->reduce_ready_head.value) >=
                 a.dispatch.identity.num_tokens;
    done = __shfl_sync(0xffffffffU, done, 0);
    if (done)
      break;
    if (!try_reduce_one_token(a))
      __nanosleep(forward_tuning::kReduceQueueNanosleep);
  }
}

} // namespace forward_reduce_detail
