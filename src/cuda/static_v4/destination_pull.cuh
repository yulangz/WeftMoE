#pragma once

// Included inside forward_unified_transport. The three control warps of every
// homogeneous CTA form one static destination-pull team. Their global-stride
// row enumeration needs no work queue or special transport CTA. Network mode
// pulls from node-local registered replicas after the GIN dispatch boundary.
template <unsigned Threads, unsigned SegmentBytes>
struct alignas(128) StaticV4Shared {
  static_assert(Threads % 32U == 0U);
  static constexpr unsigned kThreads = Threads;
  static constexpr unsigned kWarps = Threads / 32U;
  static constexpr unsigned kSegmentBytes = SegmentBytes;

  alignas(128) unsigned char staging[kWarps * SegmentBytes];
  alignas(16) CopyBarrier row_prefetch_full[kWarps];
};

template <class Storage>
__device__ __forceinline__ void
static_v4_copy_row(Storage &s, unsigned warp, const void *source, void *target,
                   unsigned bytes) {
  auto *buffer = s.staging + warp * Storage::kSegmentBytes;
  auto &full = s.row_prefetch_full[warp];
  for (unsigned offset = 0U; offset < bytes; offset += Storage::kSegmentBytes) {
    const auto extent = min(Storage::kSegmentBytes, bytes - offset);
    ::cuda::device::experimental::cp_async_bulk_global_to_shared(
        buffer, static_cast<const char *>(source) + offset, extent, full);
    auto ticket = ::cuda::device::barrier_arrive_tx(full, 1U, extent);
    full.wait(static_cast<decltype(ticket) &&>(ticket));
    ::cuda::device::experimental::cp_async_bulk_shared_to_global(
        static_cast<char *>(target) + offset, buffer, extent);
    ::cuda::device::experimental::cp_async_bulk_commit_group();
    ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
  }
}

template <unsigned Threads, unsigned BaseThread, bool Network, class Storage>
__device__ __forceinline__ void
static_v4_run_pull(const ForwardKernelArgs &a,
                   persistent_v3::ForwardPipelineControl &control,
                   unsigned long long *scheduler, Storage &s) {
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U;
  const unsigned lane = tid % 32U;
  if (lane == 0U)
    init(&s.row_prefetch_full[warp], 1U);
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  domain_sync<Storage::kThreads>();

  unsigned peer_rows = 0U;
  const auto owned = dispatch_sizes(a.dispatch).owned_expert_count;
  const auto rows = dispatch_sizes(a.dispatch).incoming_row_count;
  const auto *row_prefix = a.dispatch.expert_row_prefix;
  const auto *row_count = a.dispatch.expert_row_count;
  const auto *source_count = a.fused_local_expert_count;
  const auto *route_table = a.fused_source_routes;
  auto *canonical =
      peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *canonical_gates =
      peer_at<float>(a.local_base, a.canonical_gates_offset);
  auto *headers =
      peer_at<static_v4::RouteHeader>(a.local_base, a.direct_headers_offset);
  const auto global_warp =
      static_cast<std::uint64_t>(blockIdx.x) * Storage::kWarps + warp;
  const auto global_warps =
      static_cast<std::uint64_t>(gridDim.x) * Storage::kWarps;

  for (auto row = global_warp; row < rows; row += global_warps) {
    std::uint32_t local_expert = 0U;
    if (lane == 0U) {
      while (local_expert + 1U < owned &&
             row >= row_prefix[local_expert] + row_count[local_expert])
        ++local_expert;
    }
    local_expert = __shfl_sync(0xffffffffU, local_expert, 0U);
    const auto expert_row = row - row_prefix[local_expert];

    // Interleave source ranks within each expert.  Besides matching MegaMOE's
    // deterministic pull order, this keeps neighboring pull warps from
    // converging on one NVLink peer.  All lanes participate in the small
    // rank reduction; lane zero remains the sole TMA issuer.
    std::uint32_t source = 0U;
    std::uint32_t source_slot = 0U;
    if constexpr (Network) {
      static_v4::decode_source_row(a, local_expert, expert_row, source,
                                   source_slot);
    } else {
      const auto source_rows =
          lane < a.world
              ? static_cast<std::uint32_t>(
                    source_count[static_cast<std::uint64_t>(lane) * owned +
                                 local_expert])
              : 0U;
      auto slot = static_cast<std::uint32_t>(expert_row);
      std::uint32_t peeled = 0U;
      while (true) {
        const auto remaining = source_rows > peeled ? source_rows - peeled : 0U;
        const auto active = __reduce_add_sync(0xffffffffU, remaining != 0U);
        const auto layer =
            __reduce_min_sync(0xffffffffU, remaining ? remaining : 0xffffffffU);
        const auto rectangle = active * layer;
        if (slot < rectangle) {
          const auto active_mask = __ballot_sync(0xffffffffU, remaining != 0U);
          source = __fns(active_mask, 0U, slot % active + 1U);
          source_slot = peeled + slot / active;
          break;
        }
        slot -= rectangle;
        peeled += layer;
      }
    }

    if (lane == 0U) {
      std::uint32_t route, return_row;
      const transport::BFloat16 *source_input;
      const float *source_gates;
      if constexpr (Network) {
        const auto expert = a.rank * owned + local_expert;
        const auto *prefix = static_v4::source_prefix(a, source);
        const auto index = prefix[expert] + source_slot;
        route = static_v4::source_pointer<const std::uint32_t>(
            a, source, a.static_network.routes)[index];
        const auto experts_per_node =
            a.dispatch.identity.num_experts / a.dispatch.identity.num_nodes;
        return_row = source / a.local_world_size == a.node
                         ? route
                         : static_cast<std::uint32_t>(
                               index - prefix[a.node * experts_per_node]);
        source_input = static_v4::source_pointer<const transport::BFloat16>(
            a, source, a.static_network.input);
        source_gates = static_v4::source_pointer<const float>(
            a, source, a.static_network.gates);
      } else {
        route = route_table[(std::uint64_t(local_expert) * a.world + source) *
                                a.max_routes_per_rank +
                            source_slot];
        return_row = route;
        source_input = peer_at<const transport::BFloat16>(
            a.peer_bases[source], a.source_input_offset);
        source_gates = peer_at<const float>(a.peer_bases[source],
                                            a.fused_source_gates_offset);
      }
      const auto token = route / a.dispatch.identity.top_k;
      const auto gate = source_gates[route];
      headers[row] = static_v4::RouteHeader{return_row, source};
      canonical_gates[row] = gate;
      static_v4_copy_row(
          s, warp, source_input + token * a.hidden, canonical + row * a.hidden,
          static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16)));
      // Destination-pull writes the canonical row into this GPU.  Waiting for
      // the bulk store and publishing a device-scope release is sufficient;
      // the legacy push path needed a system fence because its destination
      // counter and payload both lived on a peer GPU.
      finish_copies<false>();
      __threadfence();
      const auto mtile = a.dispatch.mtile_row_offsets[local_expert] +
                         expert_row / forward_tuning::kTileM;
      auto *arrived =
          peer_at<unsigned long long>(a.local_base, a.x_arrived_rows_offset);
      static_v4::reduce_add_release_device(
          arrived + mtile * a.local_world_size + a.rank % a.local_world_size,
          1ULL);
      peer_rows += source != a.rank;
    }
  }

  const auto cta_peer_rows = __reduce_add_sync(0xffffffffU, peer_rows);
  if (lane == 0U && cta_peer_rows)
    atomicAdd(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->actual_dispatch_peer_bytes,
              static_cast<unsigned long long>(cta_peer_rows) * a.hidden *
                  sizeof(transport::BFloat16));
  if (tid == 0U) {
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.relay_dispatch_roles_complete.value, 1ULL);
    atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                   ->dispatch_complete_clock,
              device_globaltimer_ns());
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.rdma_return_roles_complete.value, 1ULL);
  }
  domain_sync<Storage::kThreads>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, TransportRoleDone,
                             ProxyRelay, blockIdx.x, 0ULL, execution_epoch(a));
  (void)scheduler;
}
