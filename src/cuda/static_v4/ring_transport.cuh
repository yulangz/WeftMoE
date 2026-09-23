#pragma once

// Included in forward_unified_transport. One warp per CTA owns each data
// direction. The third warp reduces, except CTA 0 which owns admission.
template <class Storage>
__device__ __noinline__ void ring_pull(const ForwardKernelArgs &a, Storage &s) {
  namespace ring = static_v4::ring;
  const auto lane = threadIdx.x & 31U;
  const auto &r = a.ring;
  const auto &p = r.source;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  auto *x = peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  for (auto index = std::uint64_t(blockIdx.x);
       index < r.state->blocks * p.shape.tile_m; index += gridDim.x) {
    const auto block = index / p.shape.tile_m;
    const auto m = static_cast<unsigned>(index % p.shape.tile_m);
    const auto &desc = r.blocks[block];
    if (m >= desc.tile.rows)
      continue;
    if (lane == 0U)
      while (!ring::admitted(a, block)) {
      }
    __syncwarp();
    const auto first_source = p.shape.source_begin(desc.window);
    const auto source_rank = first_source + lane;
    const auto local_window = desc.window / p.shape.groups();
    const auto expert = a.rank * owned + desc.tile.expert;
    const auto count = source_rank < p.shape.source_end(desc.window)
                           ? static_cast<unsigned>(ring::source_count(
                                 a, source_rank, local_window, expert))
                           : 0U;
    auto ordinal = desc.tile.expert_row + m;
    unsigned peeled = 0U, source = 0U, source_slot = 0U;
    while (true) {
      const auto remaining = count > peeled ? count - peeled : 0U;
      const auto active = __reduce_add_sync(0xffffffffU, remaining != 0U);
      const auto layer =
          __reduce_min_sync(0xffffffffU, remaining ? remaining : ~0U);
      if (ordinal < active * layer) {
        const auto mask = __ballot_sync(0xffffffffU, remaining != 0U);
        source = first_source + __fns(mask, 0U, ordinal % active + 1U);
        source_slot = peeled + ordinal / active;
        break;
      }
      ordinal -= active * layer;
      peeled += layer;
    }
    if (lane == 0U) {
      const auto *prefix = ring::peer(a, source, p.expert_prefix) +
                           local_window * (p.experts + 1U);
      const auto route = ring::peer(
          a, source, p.route_order)[local_window * p.window_capacity() +
                                    prefix[expert] + source_slot];
      const auto token = route / p.shape.top_k;
      const auto seq = ring::source_sequence(a, source, token);
      const auto *contribution =
          ring::peer(a, source, r.contributions) + seq % r.contribution_depth;
      while (ring::acquire<::cuda::thread_scope_system>(
                 contribution->generation) != seq / r.contribution_depth + 1U) {
      }
      const auto target_row = ring::row(a, block) + m;
      gates[target_row] = peer_at<const float>(
          a.peer_bases[source], a.fused_source_gates_offset)[route];
      r.rows[index] = {route, source};
      const auto *input = peer_at<const transport::BFloat16>(
                              a.peer_bases[source], a.source_input_offset) +
                          token * a.hidden;
      asm volatile("fence.proxy.async.global;" ::: "memory");
      static_v4_copy_row(
          s, 0U, input, x + target_row * a.hidden,
          static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16)));
      finish_copies<false>();
      __threadfence();
      const auto credit =
          m + 1U == desc.tile.rows ? p.shape.tile_m - desc.tile.rows + 1U : 1U;
      static_v4::reduce_add_release_device(
          reinterpret_cast<unsigned long long *>(&ring::slot(a, block).x_full),
          credit);
    }
  }
  if (lane == 0U)
    atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->dispatch_complete_clock,
              device_globaltimer_ns());
}

template <class Storage>
__device__ __noinline__ void ring_return(const ForwardKernelArgs &a,
                                         Storage &s) {
  namespace ring = static_v4::ring;
  if ((threadIdx.x & 31U) != 0U)
    return;
  const auto &r = a.ring;
  const auto *c = peer_at<const transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  for (auto index = std::uint64_t(blockIdx.x);
       index < r.state->blocks * r.source.shape.tile_m; index += gridDim.x) {
    const auto block = index / r.source.shape.tile_m;
    const auto m = index % r.source.shape.tile_m;
    if (m >= r.blocks[block].tile.rows)
      continue;
    while (!ring::admitted(a, block)) {
    }
    auto &slot = ring::slot(a, block);
    const auto target = slot.fc2_target;
    while (ring::acquire(slot.fc2_done) < target) {
    }
    const auto identity = r.rows[index];
    const auto token = identity.route / r.source.shape.top_k;
    const auto seq = ring::source_sequence(a, identity.source, token);
    const auto dest_row =
        ring::contribution_row(a, identity.source, identity.route);
    auto *destination =
        peer_at<transport::BFloat16>(a.peer_bases[identity.source],
                                     a.contribution_offset) +
        dest_row * a.hidden;
    asm volatile("fence.proxy.async.global;" ::: "memory");
    static_v4_copy_row(
        s, 1U, c + (ring::row(a, block) + m) * a.hidden, destination,
        static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16)));
    finish_copies<true>();
    ring::publish<::cuda::thread_scope_system>(
        ring::peer(a, identity.source, r.contribution_ready)[dest_row],
        seq / r.contribution_depth + 1U);
    static_v4::reduce_add_release_device(
        reinterpret_cast<unsigned long long *>(&slot.returned), 1U);
  }
  atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                 ->return_complete_clock,
            device_globaltimer_ns());
}

__device__ __noinline__ void ring_reduce(const ForwardKernelArgs &a) {
  namespace ring = static_v4::ring;
  const auto &r = a.ring;
  const auto lane = threadIdx.x & 31U;
  auto view = forward_reduce_detail::reduce_token_view(a);
  for (auto token = std::uint64_t(blockIdx.x - 1U); token < r.source.tokens;
       token += gridDim.x - 1U) {
    const auto seq = ring::source_sequence(a, a.rank, token);
    const auto first_row =
        ring::contribution_row(a, a.rank, token * r.source.shape.top_k);
    if (lane == 0U)
      for (unsigned k = 0; k < r.source.shape.top_k; ++k)
        while (ring::acquire<::cuda::thread_scope_system>(
                   r.contribution_ready[first_row + k]) !=
               seq / r.contribution_depth + 1U) {
        }
    __syncwarp();
    forward_reduce_detail::reduce_token_payload(view, token + 1U, first_row);
    if (lane == 0U)
      ring::arrive(r.contributions[seq % r.contribution_depth].consumed, 1U);
  }
}

template <unsigned Fc1N, unsigned Fc2N, class Storage>
__device__ __forceinline__ void ring_run(const ForwardKernelArgs &a,
                                         Storage &storage) {
  using RingStorage = StaticV4Shared<96U, forward_tuning::kUnifiedSegmentBytes>;
  auto &s = *reinterpret_cast<RingStorage *>(&storage);
  static_assert(sizeof(s) <= sizeof(storage));
  const auto warp = (threadIdx.x - 288U) / 32U;
  const auto lane = threadIdx.x & 31U;
  if (lane == 0U)
    init(&s.row_prefetch_full[warp], 1U);
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  domain_sync<96U>();
  if (a.dispatch.runtime->error)
    return;
  if (warp == 0U)
    ring_pull(a, s);
  else if (warp == 1U)
    ring_return(a, s);
  else if (blockIdx.x == 0U) {
    if (lane == 0U)
      static_v4::ring::admit(a, 2U * a.intermediate / Fc1N, a.hidden / Fc2N);
  } else
    ring_reduce(a);
}
