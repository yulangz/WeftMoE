#if defined(MEGAMOE_FORWARD_TRANSPORT_TU)

// One-node Forward has no GIN context owner. Each Relay CTA gives up to
// kLocalDispatchTmaWarps warps independent row-sized TMA streams and reserves
// the final warp's leader for destination-local FC1 queue publication. The
// two roles do not rendezvous until both their complete finite workspaces have
// drained, so incoming NVLink rows can release FC1 while this CTA is still
// sending later rows to its peers.
__device__ __forceinline__ void local_relay_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t relay_ordinal, unsigned long long *scheduler,
    unsigned char *storage) {
  MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay,
                          relay_ordinal, 0ULL, a.epoch);
  constexpr std::uint32_t kPhysicalWarps =
      forward_tuning::kComputeThreads / 32U;
  constexpr std::uint32_t kPublisherWarp = kPhysicalWarps - 1U;
  // The source default is the two-node production point and may use every
  // warp for rail-local Dispatch. A one-node launch has no RDMA owner, so it
  // must retain one publisher warp even when run without a world-8 generated
  // tuning header.
  constexpr std::uint32_t kCopyWarps =
      forward_tuning::kLocalDispatchTmaWarps < kPublisherWarp
          ? forward_tuning::kLocalDispatchTmaWarps
          : kPublisherWarp;
  constexpr std::uint32_t kCopyThreads = kCopyWarps * 32U;
  constexpr std::uint64_t kPublishBatchRows =
      forward_tuning::kDispatchPublishBatchRows;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto route_count = a.dispatch.sizes.route_count;
  const auto copy_bytes = static_cast<std::uint32_t>(
      a.hidden * sizeof(transport::BFloat16));

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ forward_return_detail::DirectLocalReturnBarrier
      copy_full[persistent_v3::kWarpsPerCta];
  __shared__ forward_return_detail::DirectLocalReturnState
      local_return_state;
  __shared__ unsigned int local_return_done;
  if (warp < kCopyWarps && lane == 0U)
    init(&copy_full[warp], 1U);
  if (warp < kCopyWarps)
    ::cuda::device::experimental::fence_proxy_async_shared_cta();
  __syncwarp();

  unsigned long long peer_rows = 0ULL;
  if (warp < kCopyWarps) {
    const auto first_batch =
        static_cast<std::uint64_t>(relay_ordinal) * kPublishBatchRows;
    const auto batch_stride =
        static_cast<std::uint64_t>(a.forward_relay_cta_count) *
        kPublishBatchRows;
    for (auto batch_begin = first_batch; batch_begin < route_count;
         batch_begin += batch_stride) {
      const auto batch_end = min(batch_begin + kPublishBatchRows, route_count);

      // The ordering permutation is M-block major.  Give each TMA stream a
      // few rows from one contiguous batch, then publish that whole batch;
      // the former global warp-stride forced one system atomic per row and
      // destroyed the useful M-block runs already prepared by Notify.
      if (lane == 0U) {
        for (auto position = batch_begin + warp; position < batch_end;
             position += kCopyWarps) {
          const auto route = a.dispatch_waves.local_route_order[position];
          const auto destination = a.dispatch.route_dst_rank[route];
          const auto row = a.dispatch.route_dst_row[route];
          const auto token = route / a.dispatch.identity.top_k;
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
              a.dispatch.expert_id[route],
              static_cast<std::uint32_t>(route %
                                         a.dispatch.identity.top_k)};

          auto *destination_gates = peer_at<float>(
              a.peer_bases[destination], a.canonical_gates_offset);
          destination_gates[row] = gate;

          auto *destination_rows = peer_at<transport::BFloat16>(
              a.peer_bases[destination], a.canonical_rows_offset);
          auto *warp_storage = reinterpret_cast<std::byte *>(storage) +
                               static_cast<std::size_t>(warp) * copy_bytes;
          auto &full = copy_full[warp];
          ::cuda::device::experimental::cp_async_bulk_global_to_shared(
              warp_storage, a.input + token * a.hidden, copy_bytes, full);
          auto token_barrier =
              ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
          full.wait(static_cast<decltype(token_barrier) &&>(token_barrier));
          ::cuda::device::experimental::cp_async_bulk_shared_to_global(
              destination_rows + row * a.hidden, warp_storage, copy_bytes);
          ::cuda::device::experimental::cp_async_bulk_commit_group();
          ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
          ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
        }
      }

      // Named barrier 2 contains only the copy warps.  The publisher warp can
      // keep polling ready M-blocks and appending FC1 tasks independently.
      asm volatile("bar.sync 2, %0;" : : "n"(kCopyThreads) : "memory");

      // One release RMW per contiguous (destination,M-block) run publishes
      // every row copied above.  A block-scope barrier followed by this
      // system-scope release forms the transitive publication edge for all
      // issuing warps, while reducing the normal 128-row M-block from roughly
      // 128 system atomics to one per contributing source rank.
      for (auto local = static_cast<std::uint64_t>(threadIdx.x);
           local < batch_end - batch_begin; local += kCopyThreads) {
        const auto position = batch_begin + local;
        const auto route = a.dispatch_waves.local_route_order[position];
        const auto destination = a.dispatch.route_dst_rank[route];
        const auto mtile = a.dispatch_waves.route_dst_mtile[route];
        bool run_leader = local == 0U;
        if (!run_leader) {
          const auto previous_route =
              a.dispatch_waves.local_route_order[position - 1ULL];
          run_leader =
              a.dispatch.route_dst_rank[previous_route] != destination ||
              a.dispatch_waves.route_dst_mtile[previous_route] != mtile;
        }
        if (!run_leader)
          continue;
        auto run_end = position + 1ULL;
        while (run_end < batch_end) {
          const auto next_route = a.dispatch_waves.local_route_order[run_end];
          if (a.dispatch.route_dst_rank[next_route] != destination ||
              a.dispatch_waves.route_dst_mtile[next_route] != mtile)
            break;
          ++run_end;
        }
        const auto rows = run_end - position;
        auto *arrived = peer_at<unsigned long long>(
            a.peer_bases[destination], a.x_arrived_rows_offset);
        const auto rail = a.rank % a.local_world_size;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(arrived[static_cast<std::uint64_t>(mtile) *
                              a.local_world_size +
                          rail]);
        ready.fetch_add(rows, ::cuda::memory_order_release);
        if (destination != a.rank)
          peer_rows += rows;
        if (relay_ordinal == 0U && batch_begin == 0ULL && local == 0ULL)
          MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible,
                                  GinDispatch, 0U, route, a.epoch);
      }
      asm volatile("bar.sync 2, %0;" : : "n"(kCopyThreads) : "memory");
    }
  } else if (warp == kPublisherWarp && lane == 0U) {
    publish_ready_fc1_tasks(a, relay_ordinal, a.forward_relay_cta_count,
                            scheduler);
  }

  if (peer_rows != 0ULL) {
    auto *status =
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    atomicAdd(&status->actual_dispatch_peer_bytes,
              peer_rows * a.hidden * sizeof(transport::BFloat16));
  }
  __syncthreads();
  MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay,
                          relay_ordinal, route_count, a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay,
                          relay_ordinal, route_count, a.epoch);
  if (threadIdx.x == 0U) {
    const auto old =
        persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
            control.relay_dispatch_roles_complete.value, 1ULL);
    if (old + 1ULL == a.forward_relay_cta_count) {
      auto *status =
          peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
      status->dispatch_complete_clock = device_globaltimer_ns();
    }
  }

  // Once this CTA's finite Dispatch/publisher work is drained, reuse the
  // whole CTA and the same row-sized SMEM slots for Return.  Every source
  // local rank has at least one owner; extra Relay CTAs take disjoint stripes
  // of the append-only M-block directory, so no queue claim or grid barrier is
  // required.  Compute can continue producing later descriptors concurrently.
  if (a.forward_relay_local_return) {
    const auto context = relay_ordinal % a.local_world_size;
    const auto helper_lane = relay_ordinal / a.local_world_size;
    const auto helpers_for_context =
        (a.forward_relay_cta_count + a.local_world_size - 1U - context) /
        a.local_world_size;
    if (threadIdx.x == 0U) {
      local_return_state = forward_return_detail::DirectLocalReturnState{};
      local_return_state.cursor = helper_lane;
      local_return_state.stride = helpers_for_context;
      local_return_done = 0U;
      for (std::uint32_t return_warp = 0U;
           return_warp < persistent_v3::kWarpsPerCta; ++return_warp)
        init(&copy_full[return_warp], 1U);
      ::cuda::device::experimental::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    while (local_return_done == 0U && !cancelled(a, control)) {
      const bool progressed =
          forward_return_detail::progress_direct_local_return_once(
              a, control, context, scheduler, local_return_state,
              reinterpret_cast<std::byte *>(storage), copy_full);
      if (threadIdx.x == 0U) {
        if (forward_return_detail::direct_local_return_terminal(
                a, scheduler, local_return_state)) {
          local_return_done = 1U;
          MEGAMOE_TIMELINE_RECORD(a.timeline, LocalReturnTerminal,
                                  LocalReturn, context,
                                  local_return_state.cursor, a.epoch);
        } else if (!progressed) {
          __nanosleep(forward_tuning::kReturnIdleNanosleep);
        }
      }
      __syncthreads();
    }
  }
}

#endif
