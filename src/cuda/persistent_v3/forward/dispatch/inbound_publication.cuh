__device__ __forceinline__ void issue_outbound_task(
    const ForwardKernelArgs &a, const ncclGin &gin, ncclTeam rail,
    const persistent_v3::PipelineTask &task,
    unsigned long long &issued_completions) {
  const auto remote_node = static_cast<std::uint32_t>(task.expert);
  if ((task.flags & kDispatchDirectoryTask) != 0U) {
    gin.put(rail, static_cast<int>(remote_node), a.gin_window,
            a.gin_receive_dispatch_directory_offset +
                static_cast<std::size_t>(task.row_begin) *
                    sizeof(GinDispatchContextRecord),
            a.gin_window,
            a.gin_send_dispatch_directory_offset +
                static_cast<std::size_t>(task.index) *
                    sizeof(GinDispatchContextRecord),
            sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
            ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_block, ::cuda::thread_scope_device);
    ++issued_completions;
    return;
  }

  const auto source_partition =
      a.node < remote_node ? a.node : a.node - 1U;
  const auto receive_packet =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank +
      task.row_begin;
  const auto packets = static_cast<std::uint64_t>(task.row_count);
  MEGAMOE_TIMELINE_RECORD(
      a.timeline, GinPutIssue, GinDispatch, task.context,
      packets * a.hidden * sizeof(transport::BFloat16), a.epoch);
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_header_offset +
              receive_packet * sizeof(LocalP2pRouteHeader),
          a.gin_window,
          a.gin_send_header_offset +
              static_cast<std::size_t>(task.index) *
                  sizeof(LocalP2pRouteHeader),
          packets * sizeof(LocalP2pRouteHeader));
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_fanout_offset +
              receive_packet * a.dispatch.identity.top_k *
                  sizeof(ForwardDispatchFanoutRecord),
          a.gin_window,
          a.gin_send_fanout_offset +
              static_cast<std::size_t>(task.index) *
                  a.dispatch.identity.top_k *
                  sizeof(ForwardDispatchFanoutRecord),
          packets * a.dispatch.identity.top_k *
              sizeof(ForwardDispatchFanoutRecord));
  gin.put(rail, static_cast<int>(remote_node), a.gin_window,
          a.gin_receive_payload_offset +
              receive_packet * a.hidden * sizeof(transport::BFloat16),
          a.gin_window,
          a.gin_send_payload_offset +
              static_cast<std::size_t>(task.index) * a.hidden *
                  sizeof(transport::BFloat16),
          packets * a.hidden * sizeof(transport::BFloat16), ncclGin_None{},
          ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
          ::cuda::thread_scope_block, ::cuda::thread_scope_device);
  ++issued_completions;

  // This ordered 8-byte put is both the per-wave identity and its completion
  // action.  Signal N can therefore be consumed incrementally without
  // assuming that independent GIN WQEs complete in issue order.
  gin.putValue<std::uint64_t>(
      rail, static_cast<int>(remote_node), a.gin_window,
      a.gin_receive_header_offset +
          receive_packet * sizeof(LocalP2pRouteHeader) +
          offsetof(LocalP2pRouteHeader, source_epoch),
      a.epoch, ncclGin_SignalInc{0}, ncclCoopThread{}, ncclGin_None{},
      ::cuda::thread_scope_block, ::cuda::thread_scope_device);

  auto *status =
      peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(&status->gin_dispatch_put_count, 4ULL);
  atomicAdd(&status->gin_signal_count, 1ULL);
  atomicAdd(&status->deduplicated_cross_node_payload_bytes,
            packets * a.hidden * sizeof(transport::BFloat16));
  atomicAdd(&status->actual_dispatch_peer_bytes,
            packets * a.hidden * sizeof(transport::BFloat16));
  atomicAdd(&status->gin_metadata_bytes,
            packets * (sizeof(LocalP2pRouteHeader) +
                       a.dispatch.identity.top_k *
                           sizeof(ForwardDispatchFanoutRecord)));
}

// Ring-free Dispatch arrival publication.  The GIN owner remains responsible
// for acquiring signal 0 and crossing the async-proxy boundary; only after
// that fence does it release a contiguous wave prefix to the Relay CTA.  The
// Relay derives packet extents from the immutable directory and the first
// header in each wave, so no per-wave descriptor needs to be copied.
__device__ __forceinline__ bool publish_one_inbound_wave_sequence(
    const ForwardKernelArgs &a, const ncclGin &gin,
    persistent_v3::ForwardPipelineControl &control,
    std::uint32_t context, std::uint32_t source_partition,
    bool &directory_seen, GinDispatchContextRecord &directory,
    unsigned long long &packet_cursor, unsigned long long &waves_published,
    transport_state::ContextCompletionRecord *context_completion) {
  const auto signal_value =
      gin.readSignal(0, 64, ::cuda::memory_order_acquire);
  const auto observed = static_cast<unsigned long long>(directory_seen) +
                        waves_published;
  if (signal_value <= observed)
    return false;

  auto *bytes = static_cast<std::byte *>(a.gin_window_base);
  auto *directories = reinterpret_cast<GinDispatchContextRecord *>(
      bytes + a.gin_receive_dispatch_directory_offset);
  auto *headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.gin_receive_header_offset);
  if (!directory_seen) {
    asm volatile("fence.proxy.async.global;" : : : "memory");
    directory = directories[
        source_partition * persistent_v3::kRdmaCtaCount + context];
    if (directory.generation != a.dispatch.identity.generation ||
        directory.epoch != a.epoch)
      return false;
    packet_cursor = directory.packet_begin;
    directory_seen = true;
    auto &record = context_completion[
        source_partition * a.gin_context_count + context];
    record.directory_ready = 1ULL;
    record.relay_expected = directory.packet_count;
    persistent_v3::store_release<::cuda::thread_scope_device>(
        control.dispatch[context].inbound_directory_ready.value,
        a.epoch);
    return true;
  }
  if (waves_published >= directory.wave_count)
    return false;

  const auto packet =
      static_cast<std::uint64_t>(source_partition) * a.max_tokens_per_rank +
      packet_cursor;
  ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> marker(
      headers[packet].source_epoch);
  if (marker.load(::cuda::memory_order_acquire) != a.epoch)
    return false;
  asm volatile("fence.proxy.async.global;" : : : "memory");
  const auto wave_packets = headers[packet].topk_slot;
  if (wave_packets == 0U || wave_packets > directory.packet_count ||
      packet_cursor + wave_packets >
          directory.packet_begin + directory.packet_count ||
      wave_packets > UINT16_MAX) {
    persistent_v3::publish_error_once(control.generation, 202ULL);
    return true;
  }

  packet_cursor += wave_packets;
  ++waves_published;
  persistent_v3::store_release<::cuda::thread_scope_device>(
      control.dispatch[context].inbound_waves_ready.value,
      waves_published);
  return true;
}

// Each fixed RDMA CTA contributes one otherwise-idle scalar lane to readiness
// publication. Eight lanes total scan the destination-local Forward M-blocks
// without a 116-compute-CTA system-atomic polling storm. A ready M-block is
// expanded once into its FC1 N-tile tasks in a device-local queue.
__device__ __forceinline__ void publish_ready_fc1_locality_groups(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t publisher_count, unsigned long long *scheduler) {
  constexpr std::uint32_t kStartupMblocks =
      forward_tuning::kStartupMblocks;
  const auto locality_group = a.forward_fc1_locality_group;
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
#if MEGAMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif

  while (!forward_cancelled(a)) {
    bool pending = false;
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += publisher_count) {
      if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
        continue;
      pending = true;
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);

      std::uint32_t group_mblocks = 1U;
      if (local_mblock >= kStartupMblocks) {
        // Four-way N-tile-major publication gives the best W1 L2 reuse, but
        // using it for an expert's first or last body blocks makes FC2 and
        // Return arrive in large bursts.  Keep the configured two-block
        // cadence at both ends and widen only the saturated middle window.
        const bool middle_group4 =
            locality_group == 2U &&
            local_mblock >= a.forward_fc1_locality_group4_begin &&
            local_mblock < a.forward_fc1_locality_group4_end;
        const auto effective_group = middle_group4 ? 4U : locality_group;
        const auto group_origin =
            middle_group4 ? a.forward_fc1_locality_group4_begin
                          : kStartupMblocks;
        const auto group_offset =
            (local_mblock - group_origin) % effective_group;
        if (group_offset != 0U) {
          // Body mappings are expert-major and therefore contiguous. The
          // leader marks every follower before marking itself; observing the
          // leader published is consequently sufficient to retire this lane.
          const auto leader = mblock - group_offset;
          const auto leader_mapping = claims[leader];
          const bool valid_leader =
              static_cast<std::uint32_t>(leader_mapping >> 32U) == expert &&
              static_cast<std::uint32_t>(leader_mapping) + group_offset ==
                  local_mblock;
          if (valid_leader &&
              atomicAdd(&published_mblocks[leader], 0ULL) == 0ULL)
            continue;
          if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
            continue;
          // A malformed/tail mapping must make progress rather than leaving a
          // follower permanently owned by a leader that cannot include it.
          group_mblocks = 1U;
        } else {
          const auto expert_mblocks = static_cast<std::uint32_t>(
              forward_expert_mblock_count(a, expert));
          group_mblocks =
              min(effective_group, expert_mblocks - local_mblock);
          bool group_ready = true;
          for (std::uint32_t member = 0; member < group_mblocks; ++member) {
            const auto sibling = mblock + member;
            if (sibling >= forward_mblocks) {
              group_ready = false;
              break;
            }
            const auto sibling_mapping = claims[sibling];
            if (static_cast<std::uint32_t>(sibling_mapping >> 32U) != expert ||
                static_cast<std::uint32_t>(sibling_mapping) !=
                    local_mblock + member ||
                atomicAdd(&published_mblocks[sibling], 0ULL) != 0ULL) {
              group_ready = false;
              break;
            }
            const auto sibling_row =
                forward_mblock_local_row(a, expert, local_mblock + member);
            const auto sibling_rows =
                forward_mblock_valid_rows(a, expert, local_mblock + member);
            if (!forward_mblock_rows_ready<false>(
                    a, expert, sibling_row, sibling_rows)) {
              group_ready = false;
              break;
            }
          }
          if (!group_ready)
            continue;
        }
      } else {
        const auto local_row =
            forward_mblock_local_row(a, expert, local_mblock);
        const auto valid_rows =
            forward_mblock_valid_rows(a, expert, local_mblock);
        if (!forward_mblock_rows_ready<false>(a, expert, local_row,
                                               valid_rows))
          continue;
      }

      // Followers become visible before their leader. A follower lane that
      // acquires the leader flag can therefore never race a partial group.
      for (std::uint32_t member = group_mblocks; member != 0U; --member)
        atomicExch(&published_mblocks[mblock + member - 1U], 1ULL);
      const auto queue_base =
          atomicAdd(&scheduler[12ULL + 2ULL * a.world],
                    static_cast<unsigned long long>(group_mblocks) *
                        fc1_ntiles);
      for (std::uint32_t ntile = 0; ntile < fc1_ntiles; ++ntile) {
        for (std::uint32_t member = 0; member < group_mblocks; ++member) {
          const auto queue_slot =
              queue_base + static_cast<unsigned long long>(ntile) *
                               group_mblocks + member;
          const auto task = (mblock + member) * fc1_ntiles + ntile;
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              entry(fc1_ready_queue[queue_slot]);
          entry.store(task + 1ULL, ::cuda::memory_order_release);
        }
      }
      progressed = true;
#if MEGAMOE_DEBUG_VALIDATE
      atomicMax(&completion->bulk_pack_complete,
                queue_base +
                    static_cast<unsigned long long>(group_mblocks) *
                        fc1_ntiles);
#endif
    }
    if (!pending)
      break;
    if (!progressed)
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
#if MEGAMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}

__device__ __forceinline__ void publish_ready_fc1_tasks(
    const ForwardKernelArgs &a, std::uint32_t context,
    std::uint32_t publisher_count, unsigned long long *scheduler) {
  if (a.forward_fc1_locality_group > 1U) {
    publish_ready_fc1_locality_groups(a, context, publisher_count, scheduler);
    return;
  }
  const auto forward_mblocks = scheduler[3];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>(
          (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
          forward_tuning::kFc1TileN);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_ready_queue = claims + forward_mblocks;
  auto *published_mblocks = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_published_offset);
  const auto head_mblocks = a.forward_m64_head_wavefront
                                ? *forward_mixed_scheduler_word(
                                      a, scheduler, kForwardHeadMblocks)
                                : forward_mblocks;
#if MEGAMOE_DEBUG_VALIDATE
  auto *completion = static_cast<transport_state::GenerationCompletionVector *>(
      a.peer_bases[a.world]);
#endif
  auto remaining =
      context < forward_mblocks
          ? (forward_mblocks - 1ULL - context) / publisher_count +
                1ULL
          : 0ULL;
  while (remaining != 0ULL && !forward_cancelled(a)) {
    bool progressed = false;
    for (std::uint64_t mblock = context; mblock < forward_mblocks;
         mblock += publisher_count) {
      if (atomicAdd(&published_mblocks[mblock], 0ULL) != 0ULL)
        continue;
      const auto mapping = claims[mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto local_row =
          forward_mblock_local_row(a, expert, local_mblock);
      const auto valid_rows =
          forward_mblock_valid_rows(a, expert, local_mblock);
      if (!forward_mblock_rows_ready<false>(a, expert, local_row, valid_rows))
        continue;
      atomicExch(&published_mblocks[mblock], 1ULL);
      const bool body_task =
          a.forward_m64_head_wavefront && mblock >= head_mblocks;
      auto *publish_tail =
          body_task
              ? forward_mixed_scheduler_word(a, scheduler,
                                             kForwardBodyFc1Publish)
              : &scheduler[12ULL + 2ULL * a.world];
      const auto partition_base =
          body_task ? head_mblocks * fc1_ntiles : 0ULL;
      const auto queue_base =
          partition_base + atomicAdd(
                               publish_tail,
                               static_cast<unsigned long long>(fc1_ntiles));
      for (std::uint32_t ntile = 0; ntile < fc1_ntiles; ++ntile) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            entry(fc1_ready_queue[queue_base + ntile]);
        const auto task = mblock * fc1_ntiles + ntile;
        entry.store(task + 1ULL, ::cuda::memory_order_release);
      }
      --remaining;
      progressed = true;
#if MEGAMOE_DEBUG_VALIDATE
      completion->bulk_pack_complete = queue_base + fc1_ntiles;
#endif
    }
    if (!progressed)
      __nanosleep(forward_tuning::kSetupNanosleep);
  }
#if MEGAMOE_DEBUG_VALIDATE
  atomicOr(&completion->bulk_contexts_issuing, 1ULL << context);
#endif
}

