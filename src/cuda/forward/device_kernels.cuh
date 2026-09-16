#include "cuda/forward/kernels/primitives.cuh"
template <bool StaticLocal, bool SetupOnly = false>
__global__
__maxnreg__(StaticLocal ? 128 : 160) void local_p2p_forward_transport_kernel(
    ForwardKernelArgs a) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  WEFTMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          execution_epoch(a));
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  forward_publish_progress(a, kForwardEntered, execution_epoch(a));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const auto cluster = cg::this_cluster();
#endif
  auto *canonical =
      peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *headers =
      peer_at<LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  const auto forward_rdma_ctas =
      StaticLocal
          ? 0U
          : (a.gin_context_count < gridDim.x ? a.gin_context_count : gridDim.x);
  const auto configured_relay_ctas = a.forward_relay_cta_count;
  const auto relay_cta_capacity = gridDim.x - forward_rdma_ctas;
  const auto forward_relay_ctas = configured_relay_ctas < relay_cta_capacity
                                      ? configured_relay_ctas
                                      : relay_cta_capacity;
  const auto forward_relay_cta_begin = forward_rdma_ctas;
  const auto forward_compute_cta_begin =
      forward_relay_cta_begin + forward_relay_ctas;
  const bool stream_remote_dispatch =
      remote_backend_enabled<StaticLocal>(a.use_gin) &&
      a.dispatch.identity.num_nodes >= 2U &&
      forward_rdma_ctas == persistent_v3::kRdmaCtaCount &&
      (SetupOnly || (forward_relay_ctas == configured_relay_ctas &&
      forward_relay_ctas >= persistent_v3::kRdmaCtaCount &&
      forward_relay_ctas % persistent_v3::kRdmaCtaCount == 0U));
  const bool stream_local_dispatch = StaticLocal &&
                                     a.forward_stream_local_dispatch &&
                                     gridDim.x == a.forward_relay_cta_count;
  const bool stream_remote_return_pipeline = stream_remote_dispatch;
  auto *forward_v3_control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  transport_state::GenerationCompletionVector *completion = nullptr;
  transport_state::ContextCompletionRecord *context_completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
      context_completion =
          static_cast<transport_state::ContextCompletionRecord *>(
              a.peer_bases[a.world + 1U]);
    }
  }
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *forward_reduce_route_count =
      peer_at<unsigned int>(a.local_base, a.forward_reduce_route_count_offset);
  auto *forward_reduce_ready_token = peer_at<unsigned long long>(
      a.local_base, a.forward_reduce_ready_token_offset);
  unsigned long long total_mblocks = 0;
  if (!blockIdx.x && threadIdx.x < 32U) {
    for (unsigned expert = threadIdx.x;
         expert < dispatch_sizes(a.dispatch).owned_expert_count; expert += 32U)
      total_mblocks += forward_expert_mblock_count(a, expert);
    for (unsigned delta = 16U; delta; delta /= 2U)
      total_mblocks += __shfl_down_sync(0xffffffffU, total_mblocks, delta);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->start_clock = device_globaltimer_ns();
    // Transport-only counters are unused by Forward and serve as first-start
    // clocks without growing the pooled status allocation.
    status->dispatch_blocks_done = ~0ULL; // first FC2 tile
    status->relay_blocks_done = ~0ULL;    // first returned M-block
    status->reduced_tokens = ~0ULL;       // first reduced token
    status->fc1_start_clock = ~0ULL;
    scheduler[0] = scheduler[1] = scheduler[2] = scheduler[3] = 0ULL;
    *forward_mixed_scheduler_word(a, scheduler, kForwardTransportInitEpoch) =
        0ULL;
    *forward_mixed_scheduler_word(a, scheduler,
                                  kForwardTransportClearArrivals) = 0ULL;
    *forward_mixed_scheduler_word(a, scheduler, kForwardTransportMapArrivals) =
        0ULL;
    scheduler[3] = total_mblocks;
    scheduler[4U + a.world] = 0ULL;
    scheduler[5U + 2ULL * a.world] = 0ULL;  // GIN headers ready contexts
    scheduler[6U + 2ULL * a.world] = 0ULL;  // Return directory ready epoch
    scheduler[7U + 2ULL * a.world] = 0ULL;  // Relay copy barrier arrivals
    scheduler[8U + 2ULL * a.world] = 0ULL;  // Compute setup ready epoch
    scheduler[9U + 2ULL * a.world] = 0ULL;  // Return-directory totals barrier
    scheduler[10U + 2ULL * a.world] = 0ULL; // Return-directory map barrier
    scheduler[11U + 2ULL * a.world] = 0ULL; // FC1-complete M-block count
    scheduler[12U + 2ULL * a.world] = 0ULL; // Published FC1 queue tail
    scheduler[13U + 2ULL * a.world] = 0ULL; // Completed M-block queue head
    scheduler[14U + 2ULL * a.world] = 0ULL; // Directory fragments barrier
    scheduler[15U + 2ULL * a.world] = 0ULL; // Reserved directory scratch
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        completion->generation = a.dispatch.identity.generation;
        completion->epoch = execution_epoch(a);
#if WEFTMOE_DEBUG_VALIDATE
        completion->first_waiting_tile = ~0ULL;
#endif
        unsigned long long dispatch_waves = 0ULL;
        for (std::uint64_t wave = 0;
             wave < dispatch_wave_count(a.dispatch, a.dispatch_waves); ++wave)
          if (a.dispatch_waves.destination_node[wave] != a.node)
            ++dispatch_waves;
        const auto fc1_ntiles =
            (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
            forward_tuning::kFc1TileN;
        const auto fc2_ntiles = (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
                                forward_tuning::kFc2TileN;
        completion_task(completion, transport_state::TaskClass::DispatchWave)
            .expected = stream_remote_dispatch ? 0ULL : dispatch_waves;
        unsigned long long relay_packets = 0ULL;
        for (std::uint32_t index = 0;
             index < dispatch_sizes(a.dispatch).incoming_signal_count;
             ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node != a.node)
            relay_packets += signal.expected_packets;
        }
        completion_task(completion, transport_state::TaskClass::RelayWave)
            .expected = relay_packets;
        completion_task(completion, transport_state::TaskClass::Fc1).expected =
            scheduler[3] * fc1_ntiles;
        completion_task(completion, transport_state::TaskClass::Fc2).expected =
            scheduler[3] * fc2_ntiles;
        completion_task(completion, transport_state::TaskClass::SourceArrival)
            .expected = dispatch_sizes(a.dispatch).route_count;
        completion_task(completion, transport_state::TaskClass::Reduce)
            .expected = a.dispatch.identity.num_tokens;
        completion_task(completion, transport_state::TaskClass::Return)
            .expected = 0ULL;
        completion->return_local_completion_expected = 0ULL;
        completion->directory_expected =
            stream_remote_dispatch ? 0ULL
                                   : static_cast<unsigned long long>(
                                         a.dispatch.identity.num_nodes - 1U) *
                                         a.gin_context_count;
        completion->dispatch_local_completion_expected =
            stream_remote_dispatch
                ? 0ULL
                : completion->directory_expected + dispatch_waves;
      }
    }
    // This release replaces the first transport-grid barrier.  No other CTA
    // can consume scheduler bounds or phase counters until all scalar
    // generation initialization above is visible.

    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        init_epoch(*forward_mixed_scheduler_word(a, scheduler,
                                                 kForwardTransportInitEpoch));
    init_epoch.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  if (threadIdx.x == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const auto cluster_size = cluster.dim_blocks().x * cluster.dim_blocks().y *
                              cluster.dim_blocks().z;
    atomicMax(&status->runtime_cluster_size, cluster_size);
    const auto cluster_rank = cluster.block_rank();
    if (cluster_rank < 64U)
      atomicOr(&status->cluster_rank_mask, 1ULL << cluster_rank);
#else
    atomicMax(&status->runtime_cluster_size, 1U);
    atomicOr(&status->cluster_rank_mask, 1ULL);
#endif
  }
  // Block 0 publishes the generation-local scheduler bounds and resets all
  // queue cursors above.  Every transport CTA must observe those values
  // before using scheduler[3] as the extent of the parallel initialization
  // below; otherwise a recycled generation can leave stale ready-queue
  // entries when the routed batch changes.
  // This rendezvous is confined to the fixed transport grid and precedes
  // compute setup publication, so it is not an FC1/FC2 pipeline barrier.
  forward_transport_wait_epoch(
      forward_mixed_scheduler_word(a, scheduler, kForwardTransportInitEpoch),
      execution_epoch(a));
  // One completion word per destination->source rank replaces a system-scope
  // signal for every route.  The high word is immutable expected rows; return
  // CTAs add to the low word after their peer stores have been fenced.
  if (blockIdx.x == 0 && threadIdx.x < a.world) {
    unsigned long long expected = 0;
    for (std::uint32_t local_expert = 0;
         local_expert < dispatch_sizes(a.dispatch).owned_expert_count;
         ++local_expert) {
      const auto expert = a.dispatch.owned_expert_ids[local_expert];
      expected +=
          a.dispatch
              .source_expert_count[static_cast<std::uint64_t>(threadIdx.x) *
                                       a.dispatch.identity.num_experts +
                                   expert];
    }
    scheduler[4U + threadIdx.x] = expected << 32U;
  }
  for (std::uint64_t mtile =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       mtile < scheduler[3];
       mtile += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    fc1_done[mtile] = execution_epoch(a) << 32U;
  }
  if (a.forward_dynamic_reduce_queue || a.forward_dynamic_reduce_scan) {
    for (std::uint64_t token =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         token < a.dispatch.identity.num_tokens;
         token += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
      forward_reduce_route_count[token] = 0U;
      if (a.forward_dynamic_reduce_queue)
        forward_reduce_ready_token[token] = 0ULL;
    }
  }
  if (stream_remote_dispatch || stream_local_dispatch) {
    auto *control_words =
        reinterpret_cast<unsigned long long *>(forward_v3_control);
    constexpr auto control_words_count =
        sizeof(persistent_v3::ForwardPipelineControl) /
        sizeof(unsigned long long);
    for (std::uint64_t word =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         word < control_words_count;
         word += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
      control_words[word] = 0ULL;
  }
  forward_transport_cta_barrier(
      forward_mixed_scheduler_word(a, scheduler,
                                   kForwardTransportClearArrivals),
      gridDim.x);
  if (stream_remote_dispatch || stream_local_dispatch) {
    return_queue_detail::initialize<true>(a);
    forward_transport_cta_barrier(
        forward_mixed_scheduler_word(a, scheduler,
                                     kForwardTransportClearArrivals),
        2ULL * gridDim.x);
  }
  forward_publish_progress(a, kForwardSetupComplete, scheduler[3]);
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  const auto setup_fc1_ntiles = static_cast<std::uint32_t>(
      (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
      forward_tuning::kFc1TileN);
  const auto setup_mblocks = scheduler[3];
  // The first words are an immutable expert/local-M lookup in Notify's M-tile
  // wavefront order.  The following regions are generation-local,
  // readiness-ordered FC1 and FC2 task queues.
  forward_initialize_mblocks(a, [&](unsigned long long mblock, unsigned expert,
                                    unsigned local_mblock) {
    claims[mblock] =
        (static_cast<unsigned long long>(expert) << 32U) | local_mblock;
#if __has_include(<nccl_device.h>)
    if (stream_local_dispatch || stream_remote_dispatch)
      forward_dispatch_detail::initialize_input_group(a, mblock, expert,
                                                      local_mblock);
#endif
  });
  // FC2's shared ticket indexes N-pairs in a published M-block. Only its
  // M-block descriptors follow the FC1-ready queue; no per-M ticket array.
  const auto setup_task_count = setup_mblocks * (setup_fc1_ntiles + 1ULL);
  for (std::uint64_t task =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < setup_task_count;
       task += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    claims[setup_mblocks + task] = 0ULL;
  forward_transport_cta_barrier(
      forward_mixed_scheduler_word(a, scheduler, kForwardTransportMapArrivals),
      gridDim.x);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        setup_ready(scheduler[8ULL + 2ULL * a.world]);
    setup_ready.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  forward_transport_wait_epoch(&scheduler[8ULL + 2ULL * a.world],
                               execution_epoch(a));
#endif

  if constexpr (SetupOnly)
    return;

#if __has_include(<nccl_device.h>)
  if (stream_local_dispatch) {
#if defined(WEFTMOE_FORWARD_TRANSPORT_TU)
    forward_dispatch_detail::local_relay_role(
        a, *forward_v3_control, static_cast<std::uint32_t>(blockIdx.x),
        scheduler, storage);
    // Timestamp Return before helping drain Reduce; publish it before the
    // completion counter so the finalizer also acquires the timestamp.
    if (threadIdx.x == 0U) {
      atomicMax(&status->return_complete_clock, device_globaltimer_ns());
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          forward_v3_control->rdma_return_roles_complete.value, 1ULL);
    }
    __syncthreads();
    forward_reduce_detail::drain_reduce_queue(a);
    forward_publish_progress(a, kForwardDispatchStored,
                             dispatch_sizes(a.dispatch).route_count);
    forward_publish_progress(a, kForwardDispatchObserved,
                             dispatch_sizes(a.dispatch).incoming_row_count);
    forward_finalize_detail::local_split_transport_finalize(
        a, *forward_v3_control, scheduler, status);
#endif
  } else if (stream_remote_dispatch) {
    forward_dispatch_detail::dispatch_role(
        a, *forward_v3_control, forward_relay_cta_begin, completion,
        context_completion, scheduler, storage);
    if (threadIdx.x == 0U)
      atomicMax(&status->return_complete_clock, device_globaltimer_ns());
    __syncthreads();
    if (blockIdx.x >= forward_relay_cta_begin)
      forward_reduce_detail::drain_reduce_queue(a);
    forward_publish_progress(a, kForwardDispatchStored,
                             dispatch_sizes(a.dispatch).route_count);
    forward_publish_progress(a, kForwardDispatchObserved,
                             dispatch_sizes(a.dispatch).incoming_row_count);
  } else
#endif
  {
    // Local rows use the frozen CUDA-IPC fast path. Cross-node rows are
    // packed once per (token,node) by the same-rail proxy path below.
    unsigned long long dispatch_peer_rows = 0ULL;
#if WEFTMOE_DEVICE_TIMELINE
    bool timeline_x_local_recorded = false;
#endif
    for (std::uint64_t route = blockIdx.x;
         route < dispatch_sizes(a.dispatch).route_count; route += gridDim.x) {
      const auto destination = a.dispatch.route_dst_rank[route];
      if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
          destination / a.local_world_size != a.node)
        continue;
      const auto row = a.dispatch.route_dst_row[route];
      const auto token = route / a.dispatch.identity.top_k;
      auto *destination_base = a.peer_bases[destination];
      auto *destination_rows = peer_at<transport::BFloat16>(
          destination_base, a.canonical_rows_offset);
      const auto *source = a.input + token * a.hidden;
      auto *target = destination_rows + row * a.hidden;
      const auto *source128 = reinterpret_cast<const uint4 *>(source);
      auto *target128 = reinterpret_cast<uint4 *>(target);
      for (std::uint64_t vector = threadIdx.x; vector < a.hidden / 8ULL;
           vector += blockDim.x)
        target128[vector] = source128[vector];
      if (threadIdx.x == 0) {
        auto *destination_gates =
            peer_at<float>(destination_base, a.canonical_gates_offset);
        destination_gates[row] = a.gates[route];
      }
      if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
        // Every lane's row stores precede the exact destination M-tile row
        // increment.  The destination compute worker acquires this counter;
        // no packet/wave signal is itself interpreted as tile readiness.
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
          auto *arrived = peer_at<unsigned long long>(destination_base,
                                                      a.x_arrived_rows_offset);
          const auto rail = a.rank % a.local_world_size;
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              counter(arrived[a.dispatch_waves.route_dst_mtile[route] *
                                  a.local_world_size +
                              rail]);
          counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if WEFTMOE_DEBUG_VALIDATE
          atomicAdd(&completion->local_rows_published, 1ULL);
          const auto tile = a.dispatch_waves.route_dst_mtile[route];
          if (tile < 64U)
            atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
        }
        __syncthreads();
      }
#if WEFTMOE_DEVICE_TIMELINE
      __syncthreads();
      if (!timeline_x_local_recorded) {
        WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible, GinDispatch,
                                0U, route, execution_epoch(a));
        timeline_x_local_recorded = true;
      }
#endif
      if (threadIdx.x == 0) {
        auto *destination_headers = peer_at<LocalP2pRouteHeader>(
            destination_base, a.direct_headers_offset);
        destination_headers[row] = LocalP2pRouteHeader{
            transport::kActivationTransportSchemaVersion,
            a.rank,
            destination,
            0U,
            a.dispatch.identity.generation,
            execution_epoch(a),
            route,
            route,
            token,
            row,
            a.dispatch.expert_id[route],
            static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
        if (destination != a.rank)
          ++dispatch_peer_rows;
      }
    }
#if __has_include(<nccl_device.h>)
    if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
      grid.sync();
      auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
      auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
          window_bytes + a.gin_send_header_offset);
      auto *send_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
          window_bytes + a.gin_send_fanout_offset);
      auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
          window_bytes + a.gin_send_payload_offset);
      const auto vector_count = a.hidden / 8ULL;
      const auto global_thread =
          static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
      const auto global_threads =
          static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
#if WEFTMOE_DEVICE_TIMELINE
      unsigned long long dispatch_pack_thread_bytes = 0ULL;
      __shared__ unsigned long long dispatch_pack_cta_bytes;
      if (threadIdx.x == 0U)
        dispatch_pack_cta_bytes = 0ULL;
      __syncthreads();
      WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchPackStart, GinDispatch, 0U,
                              0ULL, execution_epoch(a));
#endif
      for (std::uint64_t item = global_thread;
           item < dispatch_sizes(a.dispatch).packet_count * vector_count;
           item += global_threads) {
        const auto wave_position = item / vector_count;
        const auto vector = item % vector_count;
        const auto packet = a.dispatch_waves.packet_order[wave_position];
        const auto descriptor = a.dispatch.packet_headers[packet];
        if (descriptor.destination_node == a.node)
          continue;
        reinterpret_cast<uint4 *>(
            send_payload)[wave_position * vector_count + vector] =
            reinterpret_cast<const uint4 *>(
                a.input)[descriptor.local_token_id * vector_count + vector];
#if WEFTMOE_DEVICE_TIMELINE
        dispatch_pack_thread_bytes += sizeof(uint4);
#endif
      }
      for (std::uint64_t packed = global_thread;
           packed <
           dispatch_sizes(a.dispatch).packet_count * a.dispatch.identity.top_k;
           packed += global_threads) {
        const auto wave_position = packed / a.dispatch.identity.top_k;
        const auto packet = a.dispatch_waves.packet_order[wave_position];
        const auto descriptor = a.dispatch.packet_headers[packet];
        const auto fanout_slot = packed % a.dispatch.identity.top_k;
        const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
        const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
        if (fanout_begin + fanout_slot < fanout_end) {
          const auto target =
              a.dispatch.packet_fanout[fanout_begin + fanout_slot];
          const auto route =
              descriptor.local_token_id * a.dispatch.identity.top_k +
              target.topk_slot;
          send_fanout[packed] = ForwardDispatchFanoutRecord{
              target.dst_row,
              target.return_compact_row,
              target.topk_slot,
              static_cast<std::uint16_t>(target.destination_local_rank),
              static_cast<std::uint16_t>(target.expert_id),
              target.reserved,
              a.gates[route]};
        }
#if WEFTMOE_DEVICE_TIMELINE
        if (fanout_begin + fanout_slot < fanout_end)
          dispatch_pack_thread_bytes += sizeof(ForwardDispatchFanoutRecord);
#endif
      }
      for (std::uint64_t wave_position = global_thread;
           wave_position < dispatch_sizes(a.dispatch).packet_count;
           wave_position += global_threads) {
        const auto packet = a.dispatch_waves.packet_order[wave_position];
        const auto descriptor = a.dispatch.packet_headers[packet];
        if (descriptor.destination_node == a.node)
          continue;
        const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
        const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
        send_headers[wave_position] =
            LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                                a.rank,
                                descriptor.proxy_rank,
                                descriptor.channel,
                                a.dispatch.identity.generation,
                                execution_epoch(a),
                                wave_position,
                                fanout_end - fanout_begin,
                                descriptor.local_token_id,
                                wave_position * a.dispatch.identity.top_k,
                                0,
                                0};
#if WEFTMOE_DEVICE_TIMELINE
        dispatch_pack_thread_bytes += sizeof(LocalP2pRouteHeader);
#endif
      }
#if WEFTMOE_DEVICE_TIMELINE
      atomicAdd(&dispatch_pack_cta_bytes, dispatch_pack_thread_bytes);
      __syncthreads();
      WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchPackCtaBytes, GinDispatch, 0U,
                              dispatch_pack_cta_bytes, execution_epoch(a));
      WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchPackEnd, GinDispatch, 0U,
                              dispatch_pack_cta_bytes, execution_epoch(a));
#endif
      // GIN's device DMA reads this window after the grid-wide producers
      // have populated it. A cooperative grid barrier orders CUDA threads
      // but does not by itself publish their stores to a system-scope
      // device backend.
      WEFTMOE_TIMELINE_RECORD(
          a.timeline, DispatchPacketReady, GinDispatch,
          blockIdx.x < a.gin_context_count ? blockIdx.x : 0U,
          dispatch_sizes(a.dispatch).packet_count, execution_epoch(a));
      __threadfence_system();
      grid.sync();
    }
    if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
        blockIdx.x < a.gin_context_count) {
      const auto context = static_cast<std::uint32_t>(blockIdx.x);
      const ncclGin gin(a.gin_dev_comm, context);
      const auto rail = ncclTeamRail(a.gin_dev_comm);
      auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
      auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
          window_bytes + a.gin_send_dispatch_directory_offset);
      auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
          window_bytes + a.gin_send_header_offset);
#if WEFTMOE_DEBUG_VALIDATE
      auto *send_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
          window_bytes + a.gin_send_fanout_offset);
#endif
      // The owner-side initialization kernel only rendezvous with CUDA-IPC
      // peers on this node.  A remote same-rail producer can otherwise
      // enter this kernel first, publish an exact M-tile row increment, and
      // have that increment erased when the slower destination initializes
      // its local x_arrived_rows storage.  Signal 2 is reserved for a
      // fixed-width, per-context generation-ready handshake.  It runs once
      // before any Dispatch payload put and therefore adds no per-wave or
      // per-task work.
      //
      // Reuse each context's immutable directory slot as the eight-byte DMA
      // source/destination: the real directory record is written only after
      // the handshake completes, and Dispatch signal 0 orders that later
      // record.
      if (threadIdx.x == 0U) {
        unsigned long long setup_peers = 0ULL;
        for (std::uint32_t remote_node = 0;
             remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
          if (remote_node == a.node)
            continue;
          const auto remote_partition =
              remote_node < a.node ? remote_node : remote_node - 1U;
          const auto source_partition =
              a.node < remote_node ? a.node : a.node - 1U;
          const auto directory_index =
              remote_partition * a.gin_context_count + context;
          const auto remote_directory_index =
              source_partition * a.gin_context_count + context;
          send_directory[directory_index] = GinDispatchContextRecord{
              a.dispatch.identity.generation, execution_epoch(a), 0ULL, 0U, 0U};
          __threadfence_system();
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_dispatch_directory_offset +
                      remote_directory_index * sizeof(GinDispatchContextRecord),
                  a.gin_window,
                  a.gin_send_dispatch_directory_offset +
                      directory_index * sizeof(GinDispatchContextRecord),
                  sizeof(GinDispatchContextRecord), ncclGin_SignalInc{2},
                  ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          ++setup_peers;
        }
        if (setup_peers != 0ULL) {
          gin.waitCounter(ncclCoopThread{}, 0, setup_peers);
          gin.resetCounter(0);
          gin.waitSignal(ncclCoopThread{}, 2, setup_peers);
        }
      }
      __syncthreads();
      unsigned long long issued_completions = 0ULL;
      unsigned long long issued_packets = 0ULL;
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == a.node)
          continue;
        std::uint64_t node_wave_begin = 0;
        while (node_wave_begin <
                   dispatch_wave_count(a.dispatch, a.dispatch_waves) &&
               a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
          ++node_wave_begin;
        auto node_wave_end = node_wave_begin;
        while (node_wave_end <
                   dispatch_wave_count(a.dispatch, a.dispatch_waves) &&
               a.dispatch_waves.destination_node[node_wave_end] == remote_node)
          ++node_wave_end;
        const auto node_waves = node_wave_end - node_wave_begin;
        const auto context_wave_begin =
            node_wave_begin + node_waves * context / a.gin_context_count;
        const auto context_wave_end =
            node_wave_begin + node_waves * (context + 1U) / a.gin_context_count;
        const auto node_packet_begin =
            a.dispatch_waves.offsets[node_wave_begin];
        const auto packet_begin =
            a.dispatch_waves.offsets[context_wave_begin] - node_packet_begin;
        const auto packet_end =
            a.dispatch_waves.offsets[context_wave_end] - node_packet_begin;
        const auto remote_partition =
            remote_node < a.node ? remote_node : remote_node - 1U;
        const auto source_partition =
            a.node < remote_node ? a.node : a.node - 1U;
        const auto directory_index =
            remote_partition * a.gin_context_count + context;
        if (threadIdx.x == 0U) {
          send_directory[directory_index] = GinDispatchContextRecord{
              a.dispatch.identity.generation, execution_epoch(a), packet_begin,
              static_cast<std::uint32_t>(packet_end - packet_begin),
              static_cast<std::uint32_t>(context_wave_end -
                                         context_wave_begin)};
          __threadfence_system();
          const auto remote_directory_index =
              source_partition * a.gin_context_count + context;
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_dispatch_directory_offset +
                      remote_directory_index * sizeof(GinDispatchContextRecord),
                  a.gin_window,
                  a.gin_send_dispatch_directory_offset +
                      directory_index * sizeof(GinDispatchContextRecord),
                  sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
                  ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          ++issued_completions;
        }
        __syncthreads();
        // Every context publishes exactly one fixed-width directory record
        // for every remote node that participates in this generation,
        // including an empty record.  Receivers therefore never have to
        // infer which sender contexts own waves before waiting for the
        // directory phase.
        for (auto wave = context_wave_begin; wave < context_wave_end; ++wave) {
          const auto wave_begin = a.dispatch_waves.offsets[wave];
          const auto wave_end = a.dispatch_waves.offsets[wave + 1U];
          const auto wire_begin = wave_begin - node_packet_begin;
          const auto wave_packets = wave_end - wave_begin;
          for (auto position = wave_begin + threadIdx.x; position < wave_end;
               position += blockDim.x) {
            send_headers[position].transport_index = position - node_packet_begin;
            send_headers[position].dst_row =
                (position - node_packet_begin) * a.dispatch.identity.top_k;
            send_headers[position].topk_slot = 0U;
          }
          __syncthreads();
          if (threadIdx.x == 0U) {
            send_headers[wave_begin].topk_slot =
                static_cast<std::uint32_t>(wave_packets);
            __threadfence_system();
            const auto receive_packet =
                source_partition * a.max_tokens_per_rank + wire_begin;
            WEFTMOE_TIMELINE_RECORD(
                a.timeline, GinPutIssue, GinDispatch, context,
                wave_packets * a.hidden * sizeof(transport::BFloat16),
                execution_epoch(a));
            gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                    a.gin_receive_header_offset +
                        receive_packet * sizeof(LocalP2pRouteHeader),
                    a.gin_window,
                    a.gin_send_header_offset +
                        wave_begin * sizeof(LocalP2pRouteHeader),
                    wave_packets * sizeof(LocalP2pRouteHeader));
            gin.put(
                rail, static_cast<int>(remote_node), a.gin_window,
                a.gin_receive_fanout_offset +
                    (source_partition * a.max_tokens_per_rank + wire_begin) *
                        a.dispatch.identity.top_k *
                        sizeof(ForwardDispatchFanoutRecord),
                a.gin_window,
                a.gin_send_fanout_offset +
                    wave_begin * a.dispatch.identity.top_k *
                        sizeof(ForwardDispatchFanoutRecord),
                wave_packets * a.dispatch.identity.top_k *
                    sizeof(ForwardDispatchFanoutRecord));
            gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                    a.gin_receive_payload_offset +
                        receive_packet * a.hidden * sizeof(transport::BFloat16),
                    a.gin_window,
                    a.gin_send_payload_offset +
                        wave_begin * a.hidden * sizeof(transport::BFloat16),
                    wave_packets * a.hidden * sizeof(transport::BFloat16),
                    ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                    ncclCoopThread{}, ncclGin_None{},
                    ::cuda::thread_scope_block, ::cuda::thread_scope_device);
            ++issued_completions;
            issued_packets += wave_packets;
            atomicAdd(&status->gin_dispatch_put_count, 3ULL);
            atomicAdd(&status->gin_signal_count, 1ULL);
            atomicAdd(&status->gin_metadata_bytes,
                      wave_packets * (sizeof(LocalP2pRouteHeader) +
                                      a.dispatch.identity.top_k *
                                          sizeof(ForwardDispatchFanoutRecord)));
          }
          __syncthreads();
        }
      }
      if (threadIdx.x == 0U && issued_completions != 0ULL) {
        gin.waitCounter(ncclCoopThread{}, 0, issued_completions);
        WEFTMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion,
                                GinDispatch, context, issued_packets,
                                execution_epoch(a));
        atomicAdd(&status->deduplicated_cross_node_payload_bytes,
                  issued_packets * a.hidden * sizeof(transport::BFloat16));
        atomicAdd(&status->actual_dispatch_peer_bytes,
                  issued_packets * a.hidden * sizeof(transport::BFloat16));
        gin.resetCounter(0);
        for (std::uint32_t remote_node = 0;
             remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
          if (remote_node == a.node)
            continue;
          const auto remote_partition =
              remote_node < a.node ? remote_node : remote_node - 1U;
          const auto record_index =
              remote_partition * a.gin_context_count + context;
          const auto directory = send_directory[record_index];
          auto &record = context_completion[record_index];
          record.directory_ready = 1ULL;
          record.dispatch_expected = directory.wave_count;
          record.dispatch_claimed = directory.wave_count;
          record.dispatch_completed = directory.wave_count;
          record.local_completion = 1ULL + directory.wave_count;
          record.credit = 1ULL;
          record.last_cursor = directory.packet_begin + directory.packet_count;
          record.last_progress_globaltimer = device_globaltimer_ns();
#if WEFTMOE_DEBUG_VALIDATE
          std::uint64_t first_node_wave = 0;
          while (first_node_wave <
                     dispatch_wave_count(a.dispatch, a.dispatch_waves) &&
                 a.dispatch_waves.destination_node[first_node_wave] <
                     remote_node)
            ++first_node_wave;
          const auto node_packet_begin =
              a.dispatch_waves.offsets[first_node_wave];
          for (std::uint64_t position = directory.packet_begin;
               position < directory.packet_begin + directory.packet_count;
               ++position) {
            const auto header = send_headers[node_packet_begin + position];
            record.dispatch_fanout_rows += header.route_index;
            for (std::uint64_t fanout = 0; fanout < header.route_index;
                 ++fanout)
              record.dispatch_tile0_rows +=
                  send_fanout[(node_packet_begin + position) *
                                  a.dispatch.identity.top_k +
                              fanout]
                      .mtile == 0U;
          }
#endif
          atomicAdd(&completion->directory_ready, 1ULL);
          atomicAdd(&completion->dispatch_local_completion_completed,
                    1ULL + directory.wave_count);
          if (directory.wave_count != 0ULL) {
            claim_completion_tasks(completion,
                                   transport_state::TaskClass::DispatchWave,
                                   directory.wave_count);
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::DispatchWave,
                                      directory.wave_count);
          }
        }
      }
      __syncthreads();
      unsigned long long incoming_sources = 0ULL;
      if (threadIdx.x == 0U) {
        for (std::uint32_t index = 0;
             index < dispatch_sizes(a.dispatch).incoming_signal_count;
             ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node != a.node && signal.expected_packets != 0ULL)
            ++incoming_sources;
        }
        if (incoming_sources != 0ULL)
          gin.waitSignal(ncclCoopThread{}, 0, incoming_sources);
        auto *receive_directory = reinterpret_cast<GinDispatchContextRecord *>(
            static_cast<std::byte *>(a.gin_window_base) +
            a.gin_receive_dispatch_directory_offset);
        unsigned long long incoming_waves = 0ULL;
        for (std::uint32_t index = 0;
             index < dispatch_sizes(a.dispatch).incoming_signal_count;
             ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node == a.node || signal.expected_packets == 0ULL)
            continue;
          const auto source_partition = signal.source_node < a.node
                                            ? signal.source_node
                                            : signal.source_node - 1U;
          const auto directory =
              receive_directory[source_partition * a.gin_context_count +
                                context];
          if (directory.generation == a.dispatch.identity.generation &&
              directory.epoch == execution_epoch(a)) {
            incoming_waves += directory.wave_count;
            const auto record_index =
                source_partition * a.gin_context_count + context;
            context_completion[record_index].relay_expected =
                directory.packet_count;
          }
        }
        if (incoming_waves != 0ULL)
          gin.waitSignal(ncclCoopThread{}, 0,
                         incoming_sources + incoming_waves);
        WEFTMOE_TIMELINE_RECORD(a.timeline, RemoteSignalObserved, GinDispatch,
                                context, incoming_sources, execution_epoch(a));
        WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstRemoteSignal, GinDispatch,
                                context, incoming_sources, execution_epoch(a));
        // The GIN signal orders the network operation, but the next
        // consumer is now a different CTA/SM.  Bridge async-proxy writes
        // into the generic memory domain before publishing the CUDA atomic
        // hand-off.
        asm volatile("fence.proxy.async.global;" : : : "memory");
        __threadfence();
        atomicAdd(&scheduler[5U + 2ULL * a.world], 1ULL);
      }
      __syncthreads();

      // All GIN contexts publish an immutable receive directory before
      // Relay. Packet waves are divided between contexts by wave count,
      // which can be very uneven in bytes/fan-out.  Relaying only the
      // directory owned by the matching context left one CTA copying nearly
      // the entire remote payload while the other progress CTAs were idle.
      // Wait for the fixed directory set, then stripe every directory
      // across the fixed context CTA team.  No task space is extended here
      // and the exact destination-row counters stay the sole FC readiness
      // condition.
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            directories_ready(scheduler[5U + 2ULL * a.world]);
        while (directories_ready.load(::cuda::memory_order_acquire) <
                   a.gin_context_count &&
               !forward_cancelled(a)) {
          __nanosleep(256U);
        }
      }
      __syncthreads();
    }
    if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
        blockIdx.x >= forward_relay_cta_begin &&
        blockIdx.x < forward_compute_cta_begin) {
      // Relay work is row-owned by a static transport team. Wait until the
      // fixed context CTAs have observed every immutable directory before
      // dereference.
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            directories_ready(scheduler[5U + 2ULL * a.world]);
        while (directories_ready.load(::cuda::memory_order_acquire) <
                   a.gin_context_count &&
               !forward_cancelled(a)) {
          __nanosleep(256U);
        }
      }
      __syncthreads();
      // Acquire the cross-CTA hand-off before this Relay team dereferences
      // the registered receive window.  Keeping this fence at the consumer
      // as well makes the proxy-domain transition explicit for every Relay
      // SM.
      asm volatile("fence.proxy.async.global;" : : : "memory");
      __syncthreads();
      auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
      auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
          window_bytes + a.gin_receive_header_offset);
      auto *receive_fanout = reinterpret_cast<ForwardDispatchFanoutRecord *>(
          window_bytes + a.gin_receive_fanout_offset);
      auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
          window_bytes + a.gin_receive_payload_offset);
      auto *receive_directory = reinterpret_cast<GinDispatchContextRecord *>(
          window_bytes + a.gin_receive_dispatch_directory_offset);
      WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, 0U,
                              dispatch_sizes(a.dispatch).incoming_signal_count,
                              execution_epoch(a));
      WEFTMOE_TIMELINE_RECORD(a.timeline, XFirstProxyRelay, ProxyRelay, 0U,
                              dispatch_sizes(a.dispatch).incoming_signal_count,
                              execution_epoch(a));
      const auto vector_count = a.hidden / 8ULL;
      const auto relay_cta =
          static_cast<std::uint64_t>(blockIdx.x - forward_relay_cta_begin);
      const auto relay_thread = relay_cta * blockDim.x + threadIdx.x;
      const auto relay_threads =
          static_cast<std::uint64_t>(forward_relay_ctas) * blockDim.x;
      // Stable control path: stripe the complete immutable receive set
      // across the fixed Relay team, join the team once, then publish row
      // readiness. This deliberately gives up chunk-level FC1 overlap while
      // the dedicated Relay queue protocol is being rebuilt, but it
      // preserves broad vector transactions and makes
      // payload-before-counter ordering exact.
      for (std::uint32_t signal_index = 0;
           signal_index < dispatch_sizes(a.dispatch).incoming_signal_count;
           ++signal_index) {
        const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
        if (signal.source_node == a.node)
          continue;
        const auto source_partition = signal.source_node < a.node
                                          ? signal.source_node
                                          : signal.source_node - 1U;
        const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
        for (std::uint32_t directory_context = 0;
             directory_context < a.gin_context_count; ++directory_context) {
          const auto directory =
              receive_directory[source_partition * a.gin_context_count +
                                directory_context];
          for (std::uint64_t item = relay_thread;
               item < static_cast<std::uint64_t>(directory.packet_count) *
                          packet_vectors;
               item += relay_threads) {
            const auto sequence =
                directory.packet_begin + item / packet_vectors;
            const auto within_packet = item % packet_vectors;
            const auto fanout = within_packet / vector_count;
            const auto vector = within_packet % vector_count;
            const auto packet =
                source_partition * a.max_tokens_per_rank + sequence;
            const auto header = receive_headers[packet];
            if (fanout >= header.route_index)
              continue;
            const auto target =
                receive_fanout[source_partition * a.max_tokens_per_rank *
                                   a.dispatch.identity.top_k +
                               header.dst_row + fanout];
            const auto destination =
                a.node * a.local_world_size + target.destination_local_rank;
            auto *destination_rows = peer_at<transport::BFloat16>(
                a.peer_bases[destination], a.canonical_rows_offset);
            reinterpret_cast<uint4 *>(
                destination_rows)[target.dst_row * vector_count + vector] =
                reinterpret_cast<const uint4 *>(
                    receive_payload)[packet * vector_count + vector];
          }
          for (std::uint64_t slot = relay_thread;
               slot < static_cast<std::uint64_t>(directory.packet_count) *
                          a.dispatch.identity.top_k;
               slot += relay_threads) {
            const auto sequence =
                directory.packet_begin + slot / a.dispatch.identity.top_k;
            const auto fanout = slot % a.dispatch.identity.top_k;
            const auto packet =
                source_partition * a.max_tokens_per_rank + sequence;
            const auto header = receive_headers[packet];
            if (fanout >= header.route_index)
              continue;
            const auto target =
                receive_fanout[source_partition * a.max_tokens_per_rank *
                                   a.dispatch.identity.top_k +
                               header.dst_row + fanout];
            const auto destination =
                a.node * a.local_world_size + target.destination_local_rank;
            const auto gate = target.gate;
            auto *destination_headers = peer_at<LocalP2pRouteHeader>(
                a.peer_bases[destination], a.direct_headers_offset);
            const auto route =
                header.local_token_id * a.dispatch.identity.top_k +
                target.topk_slot;
            destination_headers[target.dst_row] = LocalP2pRouteHeader{
                transport::kActivationTransportSchemaVersion,
                header.source_rank,
                destination,
                header.channel,
                header.generation,
                header.source_epoch,
                header.transport_index,
                route,
                header.local_token_id,
                target.dst_row,
                target.expert_id,
                target.topk_slot};
            auto *destination_gates = peer_at<float>(a.peer_bases[destination],
                                                     a.canonical_gates_offset);
            destination_gates[target.dst_row] = gate;
          }
        }
      }
      // This is the only inter-CTA rendezvous in the temporary stable
      // relay. Every Relay CTA has finished all of its vector/header
      // stripes before any CTA is allowed to release a destination-row
      // counter.
      __threadfence_system();
      asm volatile("fence.proxy.async.global;" : : : "memory");
      __syncthreads();
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            copy_complete(scheduler[7ULL + 2ULL * a.world]);
        copy_complete.fetch_add(1ULL, ::cuda::memory_order_release);
        while (copy_complete.load(::cuda::memory_order_acquire) <
                   forward_relay_ctas &&
               !forward_cancelled(a))
          __nanosleep(256U);
      }
      __syncthreads();
      for (std::uint32_t signal_index = 0;
           signal_index < dispatch_sizes(a.dispatch).incoming_signal_count;
           ++signal_index) {
        const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
        if (signal.source_node == a.node)
          continue;
        const auto source_partition = signal.source_node < a.node
                                          ? signal.source_node
                                          : signal.source_node - 1U;
        for (std::uint32_t directory_context = 0;
             directory_context < a.gin_context_count; ++directory_context) {
          const auto directory =
              receive_directory[source_partition * a.gin_context_count +
                                directory_context];
          for (std::uint64_t slot = relay_thread;
               slot < static_cast<std::uint64_t>(directory.packet_count) *
                          a.dispatch.identity.top_k;
               slot += relay_threads) {
            const auto sequence =
                directory.packet_begin + slot / a.dispatch.identity.top_k;
            const auto fanout = slot % a.dispatch.identity.top_k;
            const auto packet =
                source_partition * a.max_tokens_per_rank + sequence;
            const auto header = receive_headers[packet];
            if (fanout >= header.route_index)
              continue;
            const auto target =
                receive_fanout[source_partition * a.max_tokens_per_rank *
                                   a.dispatch.identity.top_k +
                               header.dst_row + fanout];
            const auto destination =
                a.node * a.local_world_size + target.destination_local_rank;
            auto *arrived = peer_at<unsigned long long>(
                a.peer_bases[destination], a.x_arrived_rows_offset);
            const auto rail = a.rank % a.local_world_size;
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
                counter(arrived[target.mtile * a.local_world_size + rail]);
            counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if WEFTMOE_DEBUG_VALIDATE
            const auto record_index =
                source_partition * a.gin_context_count + directory_context;
            atomicAdd(&context_completion[record_index].relay_fanout_rows,
                      1ULL);
            if (target.mtile == 0U)
              atomicAdd(&context_completion[record_index].relay_tile0_rows,
                        1ULL);
            atomicAdd(&completion->remote_rows_published, 1ULL);
            if (target.mtile < 64U)
              atomicOr(&completion->published_tile_mask, 1ULL << target.mtile);
#endif
          }
        }
      }
      __syncthreads();
      if (relay_cta < a.gin_context_count && threadIdx.x == 0U) {
        const auto context = static_cast<std::uint32_t>(relay_cta);
        unsigned long long relayed = 0ULL;
        for (std::uint32_t remote = 0;
             remote < a.dispatch.identity.num_nodes - 1U; ++remote) {
          const auto index = remote * a.gin_context_count + context;
          auto &record = context_completion[index];
          record.relay_claimed = record.relay_expected;
          record.relay_completed = record.relay_expected;
          record.last_progress_globaltimer = device_globaltimer_ns();
          relayed += record.relay_expected;
        }
        if (relayed != 0ULL) {
          claim_completion_tasks(
              completion, transport_state::TaskClass::RelayWave, relayed);
          complete_completion_tasks(
              completion, transport_state::TaskClass::RelayWave, relayed);
        }
      }
      WEFTMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, 0U,
                              dispatch_sizes(a.dispatch).incoming_row_count,
                              execution_epoch(a));
      WEFTMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, 0U,
                              dispatch_sizes(a.dispatch).incoming_row_count,
                              execution_epoch(a));
    }
#endif
    if (threadIdx.x == 0 && dispatch_peer_rows != 0ULL)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                dispatch_peer_rows * a.hidden * sizeof(transport::BFloat16));
    // Every participating lane orders its peer payload stores before the
    // team rendezvous. A fence issued only by the signal lane would not
    // cover stores performed by the other lanes in the CTA.
    __threadfence_system();
    forward_publish_progress(a, kForwardDispatchStored,
                             dispatch_sizes(a.dispatch).route_count);
    if (!remote_backend_enabled<StaticLocal>(a.use_gin)) {
      grid.sync();
      if (blockIdx.x == 0) {
        const auto first_local = remote_backend_enabled<StaticLocal>(a.use_gin)
                                     ? a.node * a.local_world_size
                                     : 0U;
        const auto local_count = remote_backend_enabled<StaticLocal>(a.use_gin)
                                     ? a.local_world_size
                                     : a.world;
        if (threadIdx.x < local_count) {
          const auto peer = first_local + threadIdx.x;
          auto *signal = peer_at<unsigned long long>(
              a.peer_bases[peer], a.ingress_published_offset);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published(signal[a.rank * a.channels]);
          published.store(execution_epoch(a), ::cuda::memory_order_release);
        }
        for (std::uint32_t local_source = threadIdx.x;
             local_source < local_count; local_source += blockDim.x) {
          const auto source = first_local + local_source;
          auto *signals = peer_at<unsigned long long>(
              a.local_base, a.ingress_published_offset);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              published(signals[source * a.channels]);
          while (published.load(::cuda::memory_order_acquire) <
                     execution_epoch(a) &&
                 !forward_cancelled(a)) {
          }
        }
      }
      grid.sync();
    }
    forward_publish_progress(a, kForwardDispatchObserved,
                             remote_backend_enabled<StaticLocal>(a.use_gin)
                                 ? a.local_world_size
                                 : a.world);
  }
  // Streaming split roles publish their real Dispatch completion as soon as
  // the final Relay CTA drains.  Do not overwrite that timestamp (or the
  // independently observed FC1 start) after the same CTAs finish Return.
  if (!stream_remote_dispatch && !stream_local_dispatch && blockIdx.x == 0 &&
      threadIdx.x == 0) {
    status->dispatch_complete_clock = device_globaltimer_ns();
    status->fc1_start_clock = status->dispatch_complete_clock;
  }
  // A streaming role returns here after both Dispatch and Return drain.
  // Its Dispatch timestamp is published inside the relay pipeline instead.
  if (stream_remote_dispatch || stream_local_dispatch) {
    WEFTMOE_TIMELINE_RECORD(a.timeline, TransportRoleDone, Kernel, 0U, 0ULL,
                            execution_epoch(a));
  } else {
    WEFTMOE_TIMELINE_RECORD(a.timeline, XTileReady, Compute, 0U,
                            dispatch_sizes(a.dispatch).mtile_count,
                            execution_epoch(a));
    WEFTMOE_TIMELINE_RECORD(a.timeline, XDispatchComplete, Compute, 0U,
                            dispatch_sizes(a.dispatch).mtile_count,
                            execution_epoch(a));
  }

  if (stream_local_dispatch)
    return;
  if (!stream_remote_return_pipeline) {
    if (blockIdx.x == 0U && threadIdx.x == 0U)
      publish_local_structured_error(status, a.rank, 133U, 1U, 1ULL, 0ULL);
    return;
  }
  forward_finalize_detail::split_transport_finalize<StaticLocal>(
      a, *forward_v3_control, completion, scheduler, status);
  return;
}
