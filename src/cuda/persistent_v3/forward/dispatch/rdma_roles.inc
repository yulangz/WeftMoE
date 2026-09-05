__device__ __forceinline__ void rdma_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler) {
  if (threadIdx.x == 0U) {
    const ncclGin gin(a.gin_dev_comm, context);
    const ncclGin return_gin(a.gin_dev_comm,
                             context + a.gin_context_count);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    const auto world = ncclTeamWorld(a.gin_dev_comm);
    // CTA-wide synchronization in the setup helper is unnecessary here: only
    // the permanent GIN thread touches this context or its queue consumer.
    const auto remote_node = a.node == 0U ? 1U : 0U;
    const auto remote_partition =
        remote_node < a.node ? remote_node : remote_node - 1U;
    const auto source_partition = remote_partition;

    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
        bytes + a.gin_send_dispatch_directory_offset);
    const auto send_directory_index =
        remote_partition * persistent_v3::kRdmaCtaCount + context;

    // Inline the one-thread handshake so the remaining 383 threads never
    // acquire GIN state or register pressure.
    const auto source_directory_index =
        (a.node < remote_node ? a.node : a.node - 1U) *
            persistent_v3::kRdmaCtaCount +
        context;
    send_directory[send_directory_index] = GinDispatchContextRecord{
        a.dispatch.identity.generation, a.epoch, 0ULL, 0U, 0U};
    __threadfence_system();
    gin.put(rail, static_cast<int>(remote_node), a.gin_window,
            a.gin_receive_dispatch_directory_offset +
                source_directory_index * sizeof(GinDispatchContextRecord),
            a.gin_window,
            a.gin_send_dispatch_directory_offset +
                send_directory_index * sizeof(GinDispatchContextRecord),
            sizeof(GinDispatchContextRecord), ncclGin_SignalInc{2},
            ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
    gin.waitCounter(ncclCoopThread{}, 0, 1ULL);
    gin.resetCounter(0);
    gin.waitSignal(ncclCoopThread{}, 2, 1ULL);
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.generation.contexts[context].tx_local_complete.value,
        a.epoch);

    std::uint64_t outbound_node_wave_begin = 0ULL;
    std::uint64_t outbound_wave_begin = 0ULL;
    std::uint64_t outbound_wave_end = 0ULL;
    context_wave_span(a, context, remote_node, outbound_node_wave_begin,
                      outbound_wave_begin, outbound_wave_end);
    const auto outbound_node_packet_begin =
        a.dispatch_waves.offsets[outbound_node_wave_begin];
    auto outbound_wave = outbound_wave_begin;
    unsigned long long issued_completions = 0ULL;
    bool outbound_directory_issued = false;
    bool outbound_terminal = false;
    bool inbound_terminal = false;
    bool directory_seen = false;
    GinDispatchContextRecord inbound_directory{};
    unsigned long long packet_cursor = 0ULL;
    unsigned long long waves_published = 0ULL;
    forward_return_detail::DirectFullRdmaState direct_full_state{};
    while ((!outbound_terminal || !inbound_terminal) &&
           !cancelled(a, control)) {
      bool progressed = false;
      if (!outbound_terminal) {
        if (!outbound_directory_issued &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context]
                    .outbound_directory_ready.value) >= a.epoch) {
          persistent_v3::PipelineTask directory_task{};
          directory_task.epoch = a.epoch;
          directory_task.index =
              static_cast<std::uint32_t>(send_directory_index);
          directory_task.expert = static_cast<std::uint16_t>(remote_node);
          directory_task.kind = static_cast<std::uint8_t>(
              persistent_v3::TaskKind::DispatchRelay);
          directory_task.context = static_cast<std::uint8_t>(context);
          directory_task.row_begin =
              static_cast<std::uint32_t>(source_directory_index);
          directory_task.flags = kDispatchDirectoryTask;
          issue_outbound_task(a, gin, rail, directory_task,
                              issued_completions);
          outbound_directory_issued = true;
          progressed = true;
        }

        if (outbound_directory_issued &&
            outbound_wave < outbound_wave_end) {
          const auto packed_prefix =
              persistent_v3::load_acquire<::cuda::thread_scope_device>(
                  control.dispatch[context]
                      .outbound_waves_packed.value);
          const auto wave_ordinal = outbound_wave - outbound_wave_begin;
          if (wave_ordinal < packed_prefix) {
            const auto wave_begin = a.dispatch_waves.offsets[outbound_wave];
            const auto wave_end =
                a.dispatch_waves.offsets[outbound_wave + 1ULL];
            const auto wave_packets = wave_end - wave_begin;
            if (wave_packets == 0ULL || wave_packets > UINT16_MAX) {
              persistent_v3::publish_error_once(control.generation, 206ULL);
            } else {
              persistent_v3::PipelineTask wave_task{};
              wave_task.epoch = a.epoch;
              wave_task.index = static_cast<std::uint32_t>(wave_begin);
              wave_task.expert = static_cast<std::uint16_t>(remote_node);
              wave_task.kind = static_cast<std::uint8_t>(
                  persistent_v3::TaskKind::DispatchRelay);
              wave_task.context = static_cast<std::uint8_t>(context);
              wave_task.row_begin = static_cast<std::uint32_t>(
                  wave_begin - outbound_node_packet_begin);
              wave_task.row_count =
                  static_cast<std::uint16_t>(wave_packets);
              issue_outbound_task(a, gin, rail, wave_task,
                                  issued_completions);
              ++outbound_wave;
              progressed = true;
            }
          }
        }

        if (outbound_directory_issued &&
            outbound_wave == outbound_wave_end &&
            persistent_v3::load_acquire<::cuda::thread_scope_device>(
                control.dispatch[context]
                    .outbound_pack_complete.value) >= a.epoch) {
          outbound_terminal = true;
          progressed = true;
        }
      }

      if (!inbound_terminal) {
        progressed = publish_one_inbound_wave_sequence(
                         a, gin, control, context, source_partition,
                         directory_seen, inbound_directory, packet_cursor,
                         waves_published, context_completion) ||
                     progressed;
        if (directory_seen &&
            waves_published == inbound_directory.wave_count) {
          inbound_terminal = true;
          atomicAdd(&scheduler[5ULL + 2ULL * a.world], 1ULL);
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, RemoteSignalObserved, GinDispatch, context,
              waves_published, a.epoch);
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, XFirstRemoteSignal, GinDispatch, context,
              waves_published, a.epoch);
          progressed = true;
        }
      }

      progressed =
          forward_return_detail::progress_direct_full_rdma_once(
              a, control, return_gin, world, context, completion,
              context_completion, scheduler, direct_full_state) ||
          progressed;

      if (!progressed)
        __nanosleep(forward_tuning::kTransportIdleNanosleep);
    }

    if (issued_completions != 0ULL) {
      while (gin.readCounter(0, 56, ::cuda::memory_order_acquire) <
                 issued_completions &&
             !cancelled(a, control)) {
        if (!forward_return_detail::progress_direct_full_rdma_once(
                a, control, return_gin, world, context, completion,
                context_completion, scheduler, direct_full_state))
          __nanosleep(forward_tuning::kTransportIdleNanosleep);
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion,
                            GinDispatch, context, issued_completions,
                            a.epoch);

    const auto directory = send_directory[send_directory_index];
    auto &record = context_completion[
        remote_partition * a.gin_context_count + context];
    record.directory_ready = 1ULL;
    record.dispatch_expected = directory.wave_count;
    record.dispatch_claimed = directory.wave_count;
    record.dispatch_completed = directory.wave_count;
    record.local_completion = 1ULL + directory.wave_count;
    record.credit = 1ULL;
    record.last_cursor = directory.packet_begin + directory.packet_count;
    record.last_progress_globaltimer = device_globaltimer_ns();
    atomicAdd(&completion->directory_ready, 1ULL);
    atomicAdd(&completion->dispatch_local_completion_completed,
              1ULL + directory.wave_count);
    if (directory.wave_count != 0U) {
      claim_completion_tasks(completion,
                             transport_state::TaskClass::DispatchWave,
                             directory.wave_count);
      complete_completion_tasks(completion,
                                transport_state::TaskClass::DispatchWave,
                                directory.wave_count);
    }
    // The same permanent scalar owner transitions directly from same-rail
    // Dispatch into its independent FULL Return QP. Lane 32 can continue
    // publishing FC1 work without participating in either network loop.
    persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
        control.rdma_dispatch_roles_complete.value, 1ULL);
    forward_return_detail::direct_full_rdma_thread(
        a, control, context, completion, context_completion, scheduler,
        return_gin, world, direct_full_state);
  } else if (threadIdx.x == 32U) {
    publish_ready_fc1_tasks(a, context, persistent_v3::kRdmaCtaCount,
                            scheduler);
  } else if (threadIdx.x >= kRdmaControlWarps * 32U) {
    rdma_pack_outbound_context(a, control, context);
  }
  if (threadIdx.x >= kRdmaControlWarps * 32U) {
    // The ten former packing warps now prefetch reverse-map readiness while
    // lane 0 retains exclusive ownership of this context's Return QP.
    const auto helper_warp =
        threadIdx.x / 32U - kRdmaControlWarps;
    forward_return_detail::direct_full_mapping_helper(
        a, control, context, scheduler, helper_warp, kRdmaPackWarps);
  }
  __syncthreads();
}

__device__ __forceinline__ void dispatch_role(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t forward_relay_cta_begin,
    transport_state::GenerationCompletionVector *completion,
    transport_state::ContextCompletionRecord *context_completion,
    unsigned long long *scheduler, unsigned char *storage) {
  wait_for_local_map_clear(a, control);
  if (blockIdx.x < persistent_v3::kRdmaCtaCount) {
    rdma_role(a, control, static_cast<std::uint32_t>(blockIdx.x),
              completion, context_completion, scheduler);
    return;
  }
  const auto relay =
      static_cast<std::uint32_t>(blockIdx.x - forward_relay_cta_begin);
  relay_role(a, control, relay, completion, context_completion, scheduler,
             storage);
}

} // namespace forward_dispatch_detail
