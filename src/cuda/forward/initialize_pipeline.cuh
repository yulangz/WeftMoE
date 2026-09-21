#pragma once

// Shared initialization for the resident unified grid and legacy transport
// entry. Keep this cold control flow out of the compiled worker loops. The host validates full-grid residency before the unified launch.
// No worker starts until the final release/acquire rendezvous has published
// all queues, input groups and task descriptors.
template <bool StaticLocal, bool Unified = false, bool StaticSchedule = false>
__device__ __noinline__ void
forward_initialize_pipeline(const ForwardKernelArgs &a,
                            bool stream_remote_dispatch,
                            bool stream_local_dispatch) {
  // StaticSchedule is intentionally an ABI discriminator. CUDA 12.9 changes
  // the unified entry's pre-setmaxnreg allocation when this noinline helper is
  // instantiated without the third template argument, even though V4 reuses
  // the same initialization body as V3.
  (void)StaticSchedule;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler =
      peer_at<unsigned long long>(a.local_base, a.forward_scheduler_offset);
  auto *claims =
      peer_at<unsigned long long>(a.local_base, a.forward_task_claim_offset);
  auto *fc1_done =
      peer_at<unsigned long long>(a.local_base, a.forward_fc1_done_offset);
  auto *forward_reduce_route_count =
      peer_at<unsigned int>(a.local_base, a.forward_reduce_route_count_offset);
  auto *forward_reduce_ready_token = peer_at<unsigned long long>(
      a.local_base, a.forward_reduce_ready_token_offset);
  auto *forward_v3_control = peer_at<persistent_v3::ForwardPipelineControl>(
      a.local_base, a.forward_v3_control_offset);
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#if __has_include(<nccl_device.h>)
  if constexpr (Unified) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned channel = index;
         channel < a.local_world_size * a.local_notification_ctas;
         channel += stride)
      local_notification_detail::reset_channel(a, channel);
    if constexpr (!StaticLocal) {
      for (unsigned channel = index; channel < a.gin_rings.channels();
           channel += stride)
        activation_ring_detail::prepare_channel(a, channel);
      for (unsigned sender = index; sender < a.gin_sender_count;
           sender += stride)
        static_cast<unsigned *>(a.gin_sender_done)[sender] = 0U;
    }
  }
#endif
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
  // Control words, Return storage, input groups and task descriptors occupy
  // disjoint allocations. Their initialization can overlap; the map-arrival
  // rendezvous below publishes all of them before any worker starts.
  if (stream_remote_dispatch || stream_local_dispatch)
    return_queue_detail::initialize<true>(a);
#if !defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  forward_transport_cta_barrier(
      forward_mixed_scheduler_word(a, scheduler,
                                   kForwardTransportClearArrivals),
      gridDim.x);
#endif
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
}

// A cooperative grid makes the owner initialization a true launch prologue.
// NCCL's device barrier replaces the host-enqueued GIN reset AllReduce.
template <bool StaticLocal>
__device__ __noinline__ void
forward_initialize_generation(const ForwardKernelArgs &a) {
  auto grid = cooperative_groups::this_grid();
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0)
      *peer_at<TransportDeviceStatus>(a.local_base, a.status_offset) = {};
    initialize_forward_generation_signals(
        a.local_base, a.peer_bases, a.ingress_published_offset,
        a.outgoing_sequence_offset, a.generation_setup_offset,
        a.gin_local_compute_ready_offset, a.x_arrived_rows_offset,
        a.dispatch.sizes.mtile_count, a.completion_offset,
        a.context_completion_offset, a.context_completion_capacity,
        a.dispatch.identity.generation, a.rank, a.world, a.channels, a.epoch,
        StaticLocal ? 0U : a.node * a.local_world_size, a.local_world_size,
        a.runtime_epoch, a.dispatch.runtime);
  }
#if __has_include(<nccl_device.h>)
  if constexpr (!StaticLocal) {
    if (threadIdx.x == 0)
      for (unsigned context = blockIdx.x;
           context < a.gin_allocated_context_count; context += gridDim.x) {
        const ncclGin gin(a.gin_dev_comm, context);
        for (unsigned signal = 0; signal < 3; ++signal)
          gin.resetSignal(signal);
        gin.resetCounter(0);
        gin.resetCounter(1);
      }
  }
#endif
  grid.sync();
#if __has_include(<nccl_device.h>)
  if constexpr (!StaticLocal) {
    if (blockIdx.x == 0 && threadIdx.x < 32)
      ncclBarrier(ncclCoopWarp{}, ncclTeamTagWorld{},
                  ncclGin(a.gin_dev_comm, 0), 0, ::cuda::memory_order_acq_rel,
                  ncclGinFenceLevel::None);
    grid.sync();
  }
#endif
}
