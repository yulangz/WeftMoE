namespace forward_dispatch_detail {

inline constexpr std::uint32_t kLocalRouteBatch =
    forward_tuning::kLocalRouteBatchRows;
inline constexpr std::uint16_t kDispatchDirectoryTask = 1U;
// Warp 0 owns the GIN context and warp 1 contributes the scalar FC1-ready
// publisher.  The remaining ten complete warps used to sit idle while the
// paired Relay CTA both packed outbound packets and scattered inbound ones.
// Keep the two control warps out of every packing barrier so GIN progress and
// FC1 publication remain independent from the bulk HBM copy.
inline constexpr std::uint32_t kRdmaControlWarps =
    forward_tuning::kRdmaControlWarps;
inline constexpr std::uint32_t kRdmaPackWarps =
    persistent_v3::kWarpsPerCta - kRdmaControlWarps;
inline constexpr std::uint32_t kRdmaPackThreads = kRdmaPackWarps * 32U;

static_assert(kRdmaPackWarps > 0U);

__device__ __forceinline__ void rdma_pack_group_sync() {
  // Named barrier 1 belongs exclusively to threads [64, 384).  Warp 0 and
  // warp 1 execute long-lived scalar state machines and must never be forced
  // through a CTA-wide barrier while Dispatch is live.
  asm volatile("bar.sync 1, %0;" : : "n"(kRdmaPackThreads) : "memory");
}

__device__ __forceinline__ bool cancelled(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control) {
  return forward_cancelled(a) ||
         persistent_v3::load_acquire<::cuda::thread_scope_device>(
             control.generation.control.error_code.value) != 0ULL;
}

__device__ __forceinline__ std::uint64_t
local_route_prefix(const ForwardKernelArgs &a) {
  std::uint64_t begin = 0ULL;
  auto end = a.dispatch.sizes.route_count;
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2ULL;
    const auto route = a.dispatch_waves.local_route_order[middle];
    if (a.dispatch.route_dst_node[route] == a.node)
      begin = middle + 1ULL;
    else
      end = middle;
  }
  return begin;
}

// Incremental reverse-map stores target peer allocations during Dispatch.
// Before allowing any such store, prove that every GPU on this node has
// completed its generation-local map clear.  Block 0 publishes one epoch per
// GPU; each fixed transport CTA independently acquires the eight epochs, so
// this adds no grid-wide rendezvous and no compute CTA participates.
__device__ __forceinline__ void wait_for_local_map_clear(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    __threadfence_system();
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        control.generation.contexts[0].rx_published.value);
    ready.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  if (threadIdx.x == 0U) {
    const auto first_local = a.node * a.local_world_size;
    for (std::uint32_t local = 0U; local < a.local_world_size; ++local) {
      const auto peer = first_local + local;
      auto *peer_control = peer_at<persistent_v3::ForwardPipelineControl>(
          a.peer_bases[peer], a.forward_v3_control_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(peer_control->generation.contexts[0].rx_published.value);
      while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
             !cancelled(a, control))
        __nanosleep(forward_tuning::kReduceQueueNanosleep);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ bool context_wave_span(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t remote_node, std::uint64_t &node_wave_begin,
    std::uint64_t &context_wave_begin, std::uint64_t &context_wave_end) {
  node_wave_begin = 0ULL;
  while (node_wave_begin < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
    ++node_wave_begin;
  auto node_wave_end = node_wave_begin;
  while (node_wave_end < a.dispatch_waves.wave_count &&
         a.dispatch_waves.destination_node[node_wave_end] == remote_node)
    ++node_wave_end;
  if (node_wave_begin == node_wave_end) {
    context_wave_begin = node_wave_begin;
    context_wave_end = node_wave_begin;
    return false;
  }
  const auto node_waves = node_wave_end - node_wave_begin;
  context_wave_begin =
      node_wave_begin + node_waves * context / persistent_v3::kRdmaCtaCount;
  context_wave_end = node_wave_begin +
                     node_waves * (context + 1ULL) /
                         persistent_v3::kRdmaCtaCount;
  return true;
}

#if defined(MEGAMOE_FORWARD_TRANSPORT_TU)

// Transport-only Hopper path. Each warp leader stages one 14-KiB row through
// the SMEM slot and mbarrier already reserved for local Return. Dispatch and
// Return progress serially in a Relay CTA, so this adds no shared memory and
// keeps twelve independent bulk-async streams in flight across the CTA.
__device__ __forceinline__ void publish_local_batch(
    const ForwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end,
    transport_state::GenerationCompletionVector *completion,
    std::byte *tma_storage,
    forward_return_detail::DirectLocalReturnBarrier *tma_full) {
  const auto routes = route_end - route_begin;
  constexpr std::uint32_t kWarps =
      forward_tuning::kLocalDispatchTmaWarps;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto copy_bytes = static_cast<std::uint32_t>(
      a.hidden * sizeof(transport::BFloat16));
  bool copied = false;
  for (std::uint64_t local = warp; warp < kWarps && local < routes;
       local += kWarps) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + local];
    if (lane == 0U) {
      const auto destination = a.dispatch.route_dst_rank[route];
      const auto row = a.dispatch.route_dst_row[route];
      const auto token = route / a.dispatch.identity.top_k;
      const auto expert = a.dispatch.expert_id[route];
      const auto topk_slot = static_cast<std::uint32_t>(
          route % a.dispatch.identity.top_k);
      const auto gate = a.gates[route];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
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
          expert,
          topk_slot};
      auto *destination_gates = peer_at<float>(
          a.peer_bases[destination], a.canonical_gates_offset);
      destination_gates[row] = gate;
      auto *destination_rows = peer_at<transport::BFloat16>(
          a.peer_bases[destination], a.canonical_rows_offset);
      auto *warp_storage = tma_storage + warp * copy_bytes;
      auto &warp_full = tma_full[warp];
      ::cuda::device::experimental::cp_async_bulk_global_to_shared(
          warp_storage, a.input + token * a.hidden, copy_bytes, warp_full);
      auto token_barrier =
          ::cuda::device::barrier_arrive_tx(warp_full, 1U, copy_bytes);
      warp_full.wait(static_cast<decltype(token_barrier) &&>(token_barrier));
      ::cuda::device::experimental::cp_async_bulk_shared_to_global(
          destination_rows + row * a.hidden, warp_storage, copy_bytes);
      ::cuda::device::experimental::cp_async_bulk_commit_group();
      ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
      copied = true;
    }
  }
  if (copied) {
    ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x < routes) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + threadIdx.x];
    const auto destination = a.dispatch.route_dst_rank[route];
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[destination], a.x_arrived_rows_offset);
    const auto rail = a.rank % a.local_world_size;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        ready(arrived[a.dispatch_waves.route_dst_mtile[route] *
                          a.local_world_size +
                      rail]);
    ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if MEGAMOE_DEBUG_VALIDATE
    atomicAdd(&completion->local_rows_published, 1ULL);
    const auto tile = a.dispatch_waves.route_dst_mtile[route];
    if (tile < 64U)
      atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
  }
  __syncthreads();

  if (threadIdx.x == 0U) {
    unsigned long long peer_rows = 0ULL;
    for (auto position = route_begin; position < route_end; ++position) {
      const auto route = a.dispatch_waves.local_route_order[position];
      peer_rows += a.dispatch.route_dst_rank[route] != a.rank;
    }
    if (peer_rows != 0ULL) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(&status->actual_dispatch_peer_bytes,
                peer_rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
}

#else

// Keep the established vector path in the compute translation unit.  The
// separately compiled transport object owns the evolving TMA implementation,
// so transport changes cannot perturb WGMMA register/stack allocation.
// Copy one node-local batch and release exact 128-row M-block counters only
// after every payload/header store made by the CTA is system-visible.  A warp
// owns a row, so the 14 KiB main-shape copy uses broad contiguous transactions
// without a per-row CTA barrier.
__device__ __forceinline__ void publish_local_batch(
    const ForwardKernelArgs &a, std::uint64_t route_begin,
    std::uint64_t route_end,
    transport_state::GenerationCompletionVector *completion) {
  const auto routes = route_end - route_begin;
  const auto vector_count = a.hidden / 8ULL;
  constexpr std::uint32_t kWarps = forward_tuning::kRelayScatterWarps;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  for (std::uint64_t local = warp; warp < kWarps && local < routes;
       local += kWarps) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + local];
    std::uint32_t destination = 0U;
    std::uint64_t row = 0ULL;
    std::uint64_t token = 0ULL;
    std::uint32_t expert = 0U;
    std::uint32_t topk_slot = 0U;
    if (lane == 0U) {
      destination = a.dispatch.route_dst_rank[route];
      row = a.dispatch.route_dst_row[route];
      token = route / a.dispatch.identity.top_k;
      expert = a.dispatch.expert_id[route];
      topk_slot = static_cast<std::uint32_t>(
          route % a.dispatch.identity.top_k);
      const auto gate = a.gates[route];
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          a.peer_bases[destination], a.direct_headers_offset);
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
          expert,
          topk_slot};
      auto *destination_gates = peer_at<float>(
          a.peer_bases[destination], a.canonical_gates_offset);
      destination_gates[row] = gate;
    }
    destination = __shfl_sync(0xffffffffU, destination, 0U);
    row = __shfl_sync(0xffffffffU, row, 0U);
    token = __shfl_sync(0xffffffffU, token, 0U);
    auto *destination_rows = peer_at<transport::BFloat16>(
        a.peer_bases[destination], a.canonical_rows_offset);
    auto *target = reinterpret_cast<uint4 *>(destination_rows) +
                   row * vector_count;
    const auto *source = reinterpret_cast<const uint4 *>(a.input) +
                         token * vector_count;
    for (std::uint64_t vector = lane; vector < vector_count; vector += 32ULL)
      target[vector] = source[vector];
  }

  __threadfence_system();
  __syncthreads();
  if (threadIdx.x < routes) {
    const auto route =
        a.dispatch_waves.local_route_order[route_begin + threadIdx.x];
    const auto destination = a.dispatch.route_dst_rank[route];
    auto *arrived = peer_at<unsigned long long>(
        a.peer_bases[destination], a.x_arrived_rows_offset);
    const auto rail = a.rank % a.local_world_size;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        ready(arrived[a.dispatch_waves.route_dst_mtile[route] *
                          a.local_world_size +
                      rail]);
    // Remote and local rows owned by the same rail can be released by
    // different Relay CTAs.  Chain their stores through the counter's
    // modification order so the final acquire sees every contributing row.
    ready.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
#if MEGAMOE_DEBUG_VALIDATE
    atomicAdd(&completion->local_rows_published, 1ULL);
    const auto tile = a.dispatch_waves.route_dst_mtile[route];
    if (tile < 64U)
      atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
  }
  __syncthreads();

  if (threadIdx.x == 0U) {
    unsigned long long peer_rows = 0ULL;
    for (auto position = route_begin; position < route_end; ++position) {
      const auto route = a.dispatch_waves.local_route_order[position];
      peer_rows += a.dispatch.route_dst_rank[route] != a.rank;
    }
    if (peer_rows != 0ULL) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      atomicAdd(&status->actual_dispatch_peer_bytes,
                peer_rows * a.hidden * sizeof(transport::BFloat16));
    }
  }
}

#endif // MEGAMOE_FORWARD_TRANSPORT_TU
