struct DeviceDispatchHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  DeviceDispatchMetadata device;
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  std::uint64_t dispatch_wave_count = 0;
  DeviceMaterializationAudit audit;
  CudaRouteInput source_routes;
  CudaGateInput source_gates;
  std::shared_ptr<WorkspaceState> workspace;
  BackendCompletion expert_completion;
  BackendCompletion summary_completion;
};

struct DeviceDispatchConsumerLease::Impl {
  DeviceDispatchHandle owner;
  std::shared_ptr<WorkspaceState> workspace;
  DeviceDispatchMetadata device;
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  RankCollectiveContext context;
  std::uint64_t epoch = 0;
  CudaExecution execution;
  std::size_t event_index = SIZE_MAX;
  DeviceDispatchConsumerState state = DeviceDispatchConsumerState::Acquired;

  DeviceDispatchKernelView kernel_view() const {
    if (!owner || state == DeviceDispatchConsumerState::Released ||
        !workspace || workspace->epoch != epoch ||
        identity.generation != epoch ||
        workspace->allocation->dispatch_slot_quarantined.load(
            std::memory_order_acquire))
      invalid("phase=dispatch-consumer-view, field=state, rank=n/a, "
              "expected=acquired/submitted, actual=empty/released");
    const auto &workspace_ref = *workspace;
    const auto &dl = workspace_ref.device_layout;
    auto *base = workspace_ref.allocation->device_ptr;
    auto exec = internal::checked_execution_state(execution);
    auto count = internal::CudaDeviceLeaseAccess::checked(
        device.source_expert_count, exec->device);
    DeviceDispatchKernelView view{};
    view.identity = identity;
    view.sizes = sizes;
    view.expert_id = at<const std::uint32_t>(base, dl.expert_id);
    view.local_offset = at<const std::uint64_t>(base, dl.local_offset);
    view.route_dst_row = at<const std::uint64_t>(base, dl.route_dst_row);
    view.route_dst_rank = at<const std::uint32_t>(base, dl.route_dst_rank);
    view.route_dst_node = at<const std::uint32_t>(base, dl.route_dst_node);
    view.route_dst_local_rank =
        at<const std::uint32_t>(base, dl.route_dst_local_rank);
    view.combine_order = at<const std::uint32_t>(base, dl.combine_order);
    view.destination_node_mask = at<const std::uint64_t>(base, dl.masks);
    view.source_expert_count = static_cast<const std::uint64_t *>(count->pointer);
    view.owned_expert_ids = at<const std::uint32_t>(base, dl.owned_expert_ids);
    view.expert_row_count = at<const std::uint64_t>(base, dl.owned_expert_count);
    view.expert_row_prefix = at<const std::uint64_t>(base, dl.owned_expert_prefix);
    view.mtile_row_offsets = at<const std::uint64_t>(base, dl.mtile_offsets);
    view.mtile_expected_rows = at<const std::uint32_t>(base, dl.mtile_rows);
    view.incoming_rows = at<const DeviceIncomingRowRecord>(base, dl.incoming_rows);
    view.reverse_routes = at<const DeviceReverseRouteRecord>(base, dl.reverse_routes);
    view.packet_headers = at<const DevicePacketHeaderRecord>(base, dl.packet_headers);
    view.packet_fanout_offsets =
        at<const std::uint64_t>(base, dl.packet_fanout_offsets);
    view.packet_fanout = at<const DeviceFanoutRecord>(base, dl.packet_fanout);
    view.outgoing_signals = at<const DeviceSignalRecord>(base, dl.outgoing_signals);
    view.incoming_proxy_signals =
        at<const DeviceSignalRecord>(base, dl.incoming_signals);
    return view;
  }

  void release_owner_noexcept(bool event_usable) noexcept {
    if (!owner || state == DeviceDispatchConsumerState::Released)
      return;
    auto workspace_owner = workspace;
    {
      std::lock_guard<std::mutex> lock(
          workspace_owner->allocation->consumer_mutex);
      if (event_index <
          workspace_owner->allocation->consumer_event_states.size())
        workspace_owner->allocation->consumer_event_states[event_index] =
            event_usable ? internal::ConsumerEventSlotState::Available
                         : internal::ConsumerEventSlotState::Quarantined;
    }
    state = DeviceDispatchConsumerState::Released;
    owner = {};
    device = {};
    workspace.reset();
  }

  bool drain_submitted_noexcept() noexcept {
    auto workspace_owner = workspace;
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    const bool injected_both = take_consumer_fault(
        testing::ConsumerFaultPoint::EventAndStreamSynchronize);
    const bool injected_event = injected_both || take_consumer_fault(
        testing::ConsumerFaultPoint::EventSynchronize);
    const bool event_ok = !injected_event &&
        cudaEventSynchronize(
            workspace_owner->allocation->consumer_events[event_index]) ==
        cudaSuccess;
    bool stream_ok = true;
    if (!event_ok) {
      stream_ok = false;
      try {
        auto exec = internal::checked_execution_state(execution);
        stream_ok = !injected_both &&
                    cudaStreamSynchronize(exec->stream) == cudaSuccess;
      } catch (...) {
        stream_ok = false;
      }
    }
    if (!event_ok && !stream_ok) {
      std::lock_guard<std::mutex> lock(
          workspace_owner->allocation->consumer_mutex);
      workspace_owner->allocation->dispatch_slot_quarantined.store(
          true, std::memory_order_release);
    }
    state = DeviceDispatchConsumerState::Completed;
    release_owner_noexcept(event_ok);
    return event_ok;
  }

  void record_completion() {
    if (state != DeviceDispatchConsumerState::Acquired)
      invalid("phase=dispatch-consumer-submit, field=state, rank=" +
              std::to_string(identity.calling_rank) +
              ", expected=Acquired, actual=invalid");
    try {
      if (take_consumer_fault(testing::ConsumerFaultPoint::EventRecord))
        throw NotifyDispatchError(
            NotifyErrorCode::CudaFailure,
            "phase=dispatch-consumer-submit, field=event_record, rank=" +
                std::to_string(identity.calling_rank) +
                ", expected=success, actual=injected failure");
      internal::throw_cuda(
          "dispatch-consumer-submit",
          cudaEventRecord(workspace->allocation->consumer_events[event_index],
                          internal::checked_execution_state(execution)->stream),
          identity.calling_rank, identity.world_size,
          context.session().identity(), identity.generation,
          ExchangeKind::ExpertCounts, "cudaEventRecord(consumer completion)");
      state = DeviceDispatchConsumerState::Submitted;
    } catch (...) {
      bool stream_ok = false;
      try {
        stream_ok = cudaStreamSynchronize(
                        internal::checked_execution_state(execution)->stream) ==
                    cudaSuccess;
      } catch (...) {
        stream_ok = false;
      }
      if (!stream_ok) {
        std::lock_guard<std::mutex> lock(
            workspace->allocation->consumer_mutex);
        workspace->allocation->dispatch_slot_quarantined.store(
            true, std::memory_order_release);
      }
      release_owner_noexcept(false);
      throw;
    }
  }

  void release_noexcept() noexcept {
    if (!owner || state == DeviceDispatchConsumerState::Released)
      return;
    if (state == DeviceDispatchConsumerState::Submitted) {
      (void)drain_submitted_noexcept();
      return;
    }
    release_owner_noexcept(true);
  }
};

namespace {

void validate_device_completion(const BackendCompletion &completion,
                                const RankCollectiveContext &context,
                                ExchangeKind kind, std::uint64_t width) {
  if (!completion.has_device_result() || completion.has_host_result() ||
      completion.kind() != kind ||
      !completion.context().same_instance(context))
    invalid("phase=device-materialize, field=completion.provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=device-only exact context/kind, actual=mismatch");
  const auto &lease = completion.device_result();
  if (lease.rows() != context.session().world_size() ||
      lease.row_width() != width)
    invalid("phase=device-materialize, field=completion.shape, rank=" +
            std::to_string(context.calling_rank()) + ", expected=[" +
            std::to_string(context.session().world_size()) + "," +
            std::to_string(width) + "], actual=[" +
            std::to_string(lease.rows()) + "," +
            std::to_string(lease.row_width()) + "]");
}

template <typename T>
std::vector<T> download_vector(const std::shared_ptr<WorkspaceState> &state,
                               const CudaExecutionState &exec,
                               const void *source, std::uint64_t count,
                               const char *field) {
  std::vector<T> result(checked_size(count, field));
  if (count == 0)
    return result;
  internal::throw_cuda(
      "device-validation-download",
      cudaMemcpy(result.data(), source,
                 checked_size(checked_mul(count, sizeof(T), field), field),
                 cudaMemcpyDeviceToHost),
      state->context.calling_rank(), state->context.session().world_size(),
      state->context.session().identity(),
      state->context.session().generation(), ExchangeKind::ExpertCounts, field);
  (void)exec;
  return result;
}

} // namespace

DeviceDispatchHandle DeviceMetadataBridge::materialize(
    const CudaLocalNotifyHandle &local, BackendCompletion expert_counts,
    BackendCompletion packet_summary, const CudaExecution &execution,
    Timings *timings) {
  if (!local || !execution)
    invalid("phase=device-materialize, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &context = local.impl_->context;
  const auto &config = local.impl_->config;
  if (!context.execution().same_instance(execution.token()))
    invalid("phase=device-materialize, field=execution, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=local execution, actual=different");
  validate_device_completion(expert_counts, context,
                             ExchangeKind::ExpertCounts,
                             config.num_experts);
  validate_device_completion(packet_summary, context,
                             ExchangeKind::NotifyPacketSummary,
                             config.topology.num_nodes);
  auto exec = internal::checked_execution_state(execution);
  auto count_state = internal::CudaDeviceLeaseAccess::checked(
      expert_counts.device_result(), exec->device);
  auto summary_state = internal::CudaDeviceLeaseAccess::checked(
      packet_summary.device_result(), exec->device);
  auto state = local.impl_->workspace;
  const auto &plan = state->plan;
  const auto &dl = state->device_layout;
  auto *base = state->allocation->device_ptr;
  const auto rank = context.calling_rank();
#if !MEGAMOE_DEBUG_VALIDATE
  auto *host_ready = at<std::uint32_t>(
      state->allocation->host_ptr,
      state->host_layout.materialization_ready);
  __atomic_store_n(host_ready, 0U, __ATOMIC_RELEASE);
#endif
  internal::throw_cuda("device-materialize",
                       cudaSetDevice(static_cast<int>(exec->device)), rank,
                       context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");

  const auto h2d = [&](std::size_t offset, const void *source,
                       std::uint64_t count, std::size_t element_bytes,
                       const char *field) {
    if (count == 0)
      return;
    const auto bytes =
        checked_size(checked_mul(count, element_bytes, field), field);
    internal::throw_cuda(
        "device-materialize",
        cudaMemcpyAsync(at<void>(base, offset), source, bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts, field);
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              bytes);
  };
  h2d(dl.expert_capacity, config.capacity.expert_rows.data(),
      config.capacity.expert_rows.size(), sizeof(std::uint64_t),
      "expert capacity");
  h2d(dl.workspace_capacity,
      config.capacity.workspace_rows_per_rank.data(),
      config.capacity.workspace_rows_per_rank.size(), sizeof(std::uint64_t),
      "workspace capacity");

  const auto zero = [&](std::size_t offset, std::uint64_t count,
                        std::size_t element_bytes, const char *field) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "device-materialize",
        cudaMemsetAsync(at<void>(base, offset), 0,
                        checked_size(checked_mul(count, element_bytes, field),
                                     field),
                        exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts, field);
  };
  zero(dl.materialization_status, 1, sizeof(DeviceMaterializationStatus),
       "materialization status");
  zero(dl.expert_total, plan.num_experts, sizeof(std::uint64_t),
       "expert total");
  zero(dl.expert_base, plan.num_experts, sizeof(std::uint64_t), "expert base");
  zero(dl.expert_tile_base, plan.num_experts, sizeof(std::uint64_t),
       "expert tile base");
  zero(dl.route_dst_row, plan.route_count, sizeof(std::uint64_t),
       "route dst row");
  zero(dl.route_dst_mtile, plan.route_count, sizeof(std::uint32_t),
       "route dst mtile");
  zero(dl.owned_expert_ids, plan.num_experts, sizeof(std::uint32_t),
       "owned experts");
  zero(dl.owned_expert_count, plan.num_experts, sizeof(std::uint64_t),
       "owned counts");
  zero(dl.owned_expert_prefix,
       static_cast<std::uint64_t>(plan.num_experts) + 1,
       sizeof(std::uint64_t), "owned prefix");
  zero(dl.mtile_offsets, static_cast<std::uint64_t>(plan.num_experts) + 1,
       sizeof(std::uint64_t), "mtile offsets");
  zero(dl.mtile_rows, plan.max_incoming_rows, sizeof(std::uint32_t),
       "mtile rows");
  zero(dl.incoming_rows, plan.max_incoming_rows,
       sizeof(DeviceIncomingRowRecord), "incoming rows");
  zero(dl.reverse_routes, plan.route_count,
       sizeof(DeviceReverseRouteRecord), "reverse routes");
  zero(dl.packet_headers, plan.route_count,
       sizeof(DevicePacketHeaderRecord), "packet headers");
  zero(dl.packet_fanout_offsets, plan.route_count + 1,
       sizeof(std::uint64_t), "fanout offsets");
  zero(dl.packet_fanout, plan.route_count, sizeof(DeviceFanoutRecord),
       "fanout");
  zero(dl.packet_primary_tile, plan.route_count, sizeof(std::uint64_t),
       "packet primary tile");
  zero(dl.packet_primary_expert, plan.route_count, sizeof(std::uint32_t),
       "packet primary expert");
  zero(dl.packet_primary_rank, plan.route_count, sizeof(std::uint32_t),
       "packet primary rank");
  zero(dl.dispatch_wave_offsets, plan.route_count + 1,
       sizeof(std::uint64_t), "dispatch wave offsets");
  zero(dl.dispatch_wave_destination_node, plan.route_count,
       sizeof(std::uint32_t), "dispatch wave destination node");
  zero(dl.dispatch_wave_channel, plan.route_count, sizeof(std::uint32_t),
       "dispatch wave channel");
  zero(dl.dispatch_wave_first_primary_tile, plan.route_count,
       sizeof(std::uint64_t), "dispatch wave first primary tile");
  zero(dl.dispatch_wave_last_primary_tile, plan.route_count,
       sizeof(std::uint64_t), "dispatch wave last primary tile");
  zero(dl.dispatch_wave_packet_count, plan.route_count, sizeof(std::uint64_t),
       "dispatch wave packet count");
  zero(dl.dispatch_wave_payload_bytes, plan.route_count,
       sizeof(std::uint64_t), "dispatch wave payload bytes");
  zero(dl.outgoing_signals, plan.num_nodes, sizeof(DeviceSignalRecord),
       "outgoing signals");
  zero(dl.incoming_signals, plan.num_nodes, sizeof(DeviceSignalRecord),
       "incoming signals");

  DeviceMaterializeArgs args{};
  args.rank = rank;
  args.world = plan.world_size;
  args.experts = plan.num_experts;
  args.nodes = plan.num_nodes;
  args.top_k = plan.top_k;
  args.mask_words = plan.node_mask_words;
  args.m_tile = config.m_tile_rows;
  args.tokens = plan.num_tokens;
  args.routes = plan.route_count;
  args.max_tokens = plan.max_tokens_per_rank;
  args.max_incoming_rows = plan.max_incoming_rows;
  args.has_expert_capacity = !config.capacity.expert_rows.empty();
  args.has_workspace_capacity =
      !config.capacity.workspace_rows_per_rank.empty();
  args.count_matrix = static_cast<const std::uint64_t *>(count_state->pointer);
  args.summary_matrix =
      static_cast<const std::uint64_t *>(summary_state->pointer);
  args.local_histogram = at<const std::uint64_t>(base, dl.expert_count);
  args.local_summary = at<const std::uint64_t>(base, dl.packet_count);
  args.expert_to_rank = at<const std::uint32_t>(base, dl.expert_to_rank);
  args.rank_to_node = at<const std::uint32_t>(base, dl.rank_to_node);
  args.rank_to_local = at<const std::uint32_t>(base, dl.rank_to_local);
  args.expert_capacity = at<const std::uint64_t>(base, dl.expert_capacity);
  args.workspace_capacity =
      at<const std::uint64_t>(base, dl.workspace_capacity);
  args.expert_id = at<const std::uint32_t>(base, dl.expert_id);
  args.local_offset = at<const std::uint64_t>(base, dl.local_offset);
  args.route_dst_rank = at<const std::uint32_t>(base, dl.route_dst_rank);
  args.route_dst_node = at<const std::uint32_t>(base, dl.route_dst_node);
  args.route_dst_local =
      at<const std::uint32_t>(base, dl.route_dst_local_rank);
  args.combine_order = at<const std::uint32_t>(base, dl.combine_order);
  args.masks = at<const std::uint64_t>(base, dl.masks);
  args.expert_total = at<std::uint64_t>(base, dl.expert_total);
  args.expert_base = at<std::uint64_t>(base, dl.expert_base);
  args.expert_tile_base = at<std::uint64_t>(base, dl.expert_tile_base);
  args.route_dst_row = at<std::uint64_t>(base, dl.route_dst_row);
  args.route_dst_mtile = at<std::uint32_t>(base, dl.route_dst_mtile);
  args.owned_expert_ids = at<std::uint32_t>(base, dl.owned_expert_ids);
  args.owned_expert_count = at<std::uint64_t>(base, dl.owned_expert_count);
  args.owned_expert_prefix = at<std::uint64_t>(base, dl.owned_expert_prefix);
  args.mtile_offsets = at<std::uint64_t>(base, dl.mtile_offsets);
  args.mtile_rows = at<std::uint32_t>(base, dl.mtile_rows);
  args.incoming_rows =
      at<DeviceIncomingRowRecord>(base, dl.incoming_rows);
  args.reverse_routes =
      at<DeviceReverseRouteRecord>(base, dl.reverse_routes);
  args.packet_headers =
      at<DevicePacketHeaderRecord>(base, dl.packet_headers);
  args.packet_fanout_offsets =
      at<std::uint64_t>(base, dl.packet_fanout_offsets);
  args.packet_fanout = at<DeviceFanoutRecord>(base, dl.packet_fanout);
  args.packet_primary_tile =
      at<std::uint64_t>(base, dl.packet_primary_tile);
  args.packet_primary_expert =
      at<std::uint32_t>(base, dl.packet_primary_expert);
  args.packet_primary_rank =
      at<std::uint32_t>(base, dl.packet_primary_rank);
  args.local_route_order =
      at<std::uint64_t>(base, dl.local_route_order);
  args.sort_keys_in = at<std::uint64_t>(base, dl.keys_in);
  args.sort_keys_out = at<std::uint64_t>(base, dl.keys_out);
  args.sort_indices_in = at<std::uint64_t>(base, dl.indices_in);
  args.sort_indices_out = at<std::uint64_t>(base, dl.indices_out);
  args.wave_flags = at<std::uint64_t>(base, dl.ones);
  args.wave_ids = at<std::uint64_t>(base, dl.inclusive);
  args.dispatch_wave_offsets =
      at<std::uint64_t>(base, dl.dispatch_wave_offsets);
  args.dispatch_wave_destination_node =
      at<std::uint32_t>(base, dl.dispatch_wave_destination_node);
  args.dispatch_wave_channel =
      at<std::uint32_t>(base, dl.dispatch_wave_channel);
  args.dispatch_wave_first_primary_tile =
      at<std::uint64_t>(base, dl.dispatch_wave_first_primary_tile);
  args.dispatch_wave_last_primary_tile =
      at<std::uint64_t>(base, dl.dispatch_wave_last_primary_tile);
  args.dispatch_wave_packet_count =
      at<std::uint64_t>(base, dl.dispatch_wave_packet_count);
  args.dispatch_wave_payload_bytes =
      at<std::uint64_t>(base, dl.dispatch_wave_payload_bytes);
  args.outgoing_signals =
      at<DeviceSignalRecord>(base, dl.outgoing_signals);
  args.incoming_signals =
      at<DeviceSignalRecord>(base, dl.incoming_signals);
  args.status =
      at<DeviceMaterializationStatus>(base, dl.materialization_status);

  constexpr std::uint32_t threads = 256;
  const auto blocks_for = [&](std::uint64_t count, const char *field) {
    const auto blocks = (count + threads - 1U) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max())
      invalid(std::string("phase=device-materialize, field=") + field +
              ", rank=n/a, expected=CUDA grid.x capacity, actual=" +
              std::to_string(blocks));
    return static_cast<unsigned int>(std::max<std::uint64_t>(blocks, 1));
  };
  const auto incoming_pairs = checked_mul(plan.num_experts, plan.world_size,
                                          "incoming expert/source pairs");
  if (incoming_pairs > std::numeric_limits<unsigned int>::max())
    invalid("phase=device-materialize, field=incoming_grid, rank=" +
            std::to_string(rank) + ", expected<=UINT_MAX, actual=" +
            std::to_string(incoming_pairs));

  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_start, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(prefix planning start)");
#if MEGAMOE_DEBUG_VALIDATE
  materialize_prefix_plan_kernel<<<1, 1, 0, exec->stream>>>(args);
#else
  materialize_plan_init_kernel<<<1, 1, 0, exec->stream>>>(args);
  materialize_expert_totals_kernel<<<
      blocks_for(plan.num_experts, "expert_total_grid"), threads, 0,
      exec->stream>>>(args);
  materialize_expert_prefix_kernel<<<
      plan.num_experts, threads, 0,
      exec->stream>>>(args);
  materialize_destination_capacity_kernel<<<
      plan.world_size, threads, 0,
      exec->stream>>>(args);
  materialize_plan_finalize_kernel<<<1, threads, 0, exec->stream>>>(args);
#endif
  materialize_packet_plan_kernel<<<plan.num_nodes, threads, 0, exec->stream>>>(
      args);
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_complete, exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(prefix planning complete)");
  materialize_routes_kernel<<<blocks_for(plan.route_count, "route_grid"),
                              threads, 0, exec->stream>>>(args);
  materialize_incoming_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(incoming_pairs, 1)),
      threads, 0, exec->stream>>>(args);
  materialize_mtiles_kernel<<<plan.num_experts, threads, 0, exec->stream>>>(
      args);
  materialize_fanout_kernel<false>
      <<<blocks_for(plan.route_count, "packet_grid"), threads, 0,
         exec->stream>>>(args);
  if (plan.route_count != 0) {
    materialize_packet_primary_kernel<<<
        blocks_for(plan.route_count, "wave_primary_grid"), threads, 0,
        exec->stream>>>(args);
    const auto packed_one_node_wave_key_fits = [&] {
      if (plan.num_nodes != 1U || plan.world_size == 0U ||
          plan.max_tokens_per_rank == 0U)
        return false;
      const auto max_value = std::numeric_limits<std::uint64_t>::max();
      auto key_space = plan.max_incoming_rows;
      auto extend = [&](std::uint64_t radix) {
        if (radix == 0U || key_space > (max_value - (radix - 1U)) / radix)
          return false;
        key_space = key_space * radix + radix - 1U;
        return true;
      };
      const auto expert_radix =
          static_cast<std::uint64_t>(plan.num_experts) *
          (plan.world_size == 8U ? 8ULL : 1ULL);
      return extend(expert_radix) && extend(plan.world_size) &&
             extend(plan.max_tokens_per_rank);
    }();
    const auto sort_pass = [&](WaveSortField field,
                               const std::uint64_t *indices_input,
                               std::uint64_t *indices_output) {
      const auto blocks = blocks_for(plan.route_count, "wave_sort_grid");
      switch (field) {
      case WaveSortField::Token:
        materialize_wave_sort_keys_kernel<WaveSortField::Token>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case WaveSortField::DestinationRank:
        materialize_wave_sort_keys_kernel<WaveSortField::DestinationRank>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case WaveSortField::Expert:
        materialize_wave_sort_keys_kernel<WaveSortField::Expert>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case WaveSortField::MTile:
        materialize_wave_sort_keys_kernel<WaveSortField::MTile>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case WaveSortField::DestinationNode:
        materialize_wave_sort_keys_kernel<WaveSortField::DestinationNode>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      }
      auto sort_bytes = checked_size(state->plan.sort_temp_bytes,
                                     "wave_sort_temp_bytes");
      internal::throw_cuda(
          "device-materialize",
          cub::DeviceRadixSort::SortPairs(
              at<void>(base, dl.temp), sort_bytes, args.sort_keys_in,
              args.sort_keys_out, indices_input, indices_output,
              static_cast<int>(plan.route_count), 0, 64, exec->stream),
          rank, context.session().world_size(), context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "wave cub::DeviceRadixSort::SortPairs");
    };
    if (packed_one_node_wave_key_fits) {
      materialize_one_node_wave_sort_keys_kernel<<<
          blocks_for(plan.route_count, "wave_sort_grid"), threads, 0,
          exec->stream>>>(args, args.sort_indices_in);
      auto sort_bytes = checked_size(state->plan.sort_temp_bytes,
                                     "wave_sort_temp_bytes");
      internal::throw_cuda(
          "device-materialize",
          cub::DeviceRadixSort::SortPairs(
              at<void>(base, dl.temp), sort_bytes, args.sort_keys_in,
              args.sort_keys_out, args.sort_indices_in, args.sort_indices_out,
              static_cast<int>(plan.route_count), 0, 64, exec->stream),
          rank, context.session().world_size(), context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "one-node wave cub::DeviceRadixSort::SortPairs");
    } else {
      sort_pass(WaveSortField::Token, args.sort_indices_in,
                args.sort_indices_out);
      sort_pass(WaveSortField::DestinationRank, args.sort_indices_out,
                args.sort_indices_in);
      sort_pass(WaveSortField::Expert, args.sort_indices_in,
                args.sort_indices_out);
      sort_pass(WaveSortField::MTile, args.sort_indices_out,
                args.sort_indices_in);
      sort_pass(WaveSortField::DestinationNode, args.sort_indices_in,
                args.sort_indices_out);
    }
    materialize_wave_flags_kernel<<<
        blocks_for(plan.route_count, "wave_flag_grid"), threads, 0,
        exec->stream>>>(args);
    auto wave_scan_bytes = checked_size(state->plan.scan_temp_bytes,
                                        "wave_scan_temp_bytes");
    internal::throw_cuda(
        "device-materialize",
        cub::DeviceScan::InclusiveSum(
            at<void>(base, dl.temp), wave_scan_bytes, args.wave_flags,
            args.wave_ids, static_cast<int>(plan.route_count), exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "wave cub::DeviceScan::InclusiveSum");
    materialize_waves_kernel<<<blocks_for(plan.route_count, "wave_grid"),
                               threads, 0, exec->stream>>>(args);

    // Build a separate local delivery permutation after the packet wave
    // catalog is final.  Three stable passes yield
    // (locality, local M tile, expert, original route).  Keeping the packet
    // permutation in indices_out and using a dedicated final buffer avoids
    // coupling GIN wire identity to local NVLink scheduling.
    materialize_local_route_order_init_kernel<<<
        blocks_for(plan.route_count, "local_route_init_grid"), threads, 0,
        exec->stream>>>(args);
    const auto local_route_sort_pass = [&] (
        LocalRouteSortField field, const std::uint64_t *indices_input,
        std::uint64_t *indices_output) {
      const auto blocks =
          blocks_for(plan.route_count, "local_route_sort_grid");
      switch (field) {
      case LocalRouteSortField::Expert:
        materialize_local_route_sort_keys_kernel<LocalRouteSortField::Expert>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case LocalRouteSortField::MTile:
        materialize_local_route_sort_keys_kernel<LocalRouteSortField::MTile>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      case LocalRouteSortField::Locality:
        materialize_local_route_sort_keys_kernel<LocalRouteSortField::Locality>
            <<<blocks, threads, 0, exec->stream>>>(args, indices_input);
        break;
      }
      auto sort_bytes = checked_size(state->plan.sort_temp_bytes,
                                     "local_route_sort_temp_bytes");
      internal::throw_cuda(
          "device-materialize",
          cub::DeviceRadixSort::SortPairs(
              at<void>(base, dl.temp), sort_bytes, args.sort_keys_in,
              args.sort_keys_out, indices_input, indices_output,
              static_cast<int>(plan.route_count), 0, 64, exec->stream),
          rank, context.session().world_size(), context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "local route cub::DeviceRadixSort::SortPairs");
    };
    if (plan.num_nodes == 1U) {
      materialize_one_node_local_route_sort_keys_kernel<<<
          blocks_for(plan.route_count, "local_route_sort_grid"), threads, 0,
          exec->stream>>>(args, args.sort_indices_in);
      auto sort_bytes = checked_size(state->plan.sort_temp_bytes,
                                     "local_route_sort_temp_bytes");
      internal::throw_cuda(
          "device-materialize",
          cub::DeviceRadixSort::SortPairs(
              at<void>(base, dl.temp), sort_bytes, args.sort_keys_in,
              args.sort_keys_out, args.sort_indices_in,
              args.local_route_order, static_cast<int>(plan.route_count), 0,
              64, exec->stream),
          rank, context.session().world_size(), context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "one-node local route cub::DeviceRadixSort::SortPairs");
    } else {
      local_route_sort_pass(LocalRouteSortField::Expert,
                            args.sort_indices_in, args.local_route_order);
      local_route_sort_pass(LocalRouteSortField::MTile,
                            args.local_route_order, args.sort_indices_in);
      local_route_sort_pass(LocalRouteSortField::Locality,
                            args.sort_indices_in, args.local_route_order);
    }

    // Build the immutable Return wire order only after the Dispatch and local
    // delivery permutations have been materialized into their final buffers.
    // One packed, unique key needs a single radix pass and is outside the
    // steady Forward path.
    materialize_return_order_init_kernel<<<
        blocks_for(plan.route_count, "return_order_init_grid"), threads, 0,
        exec->stream>>>(args);
    auto return_sort_bytes = checked_size(state->plan.sort_temp_bytes,
                                          "return_order_sort_temp_bytes");
    internal::throw_cuda(
        "device-materialize",
        cub::DeviceRadixSort::SortPairs(
            at<void>(base, dl.temp), return_sort_bytes, args.sort_keys_in,
            args.sort_keys_out, args.sort_indices_in, args.wave_flags,
            static_cast<int>(plan.route_count), 0, 64, exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "return order cub::DeviceRadixSort::SortPairs");
    materialize_return_node_offsets_kernel<<<1, threads, 0, exec->stream>>>(
        args);
    materialize_return_compact_rows_kernel<<<
        blocks_for(plan.route_count, "return_compact_grid"), threads, 0,
        exec->stream>>>(args);
    materialize_fanout_kernel<true>
        <<<blocks_for(plan.route_count, "return_fanout_grid"), threads, 0,
           exec->stream>>>(args);
  }
  materialize_signals_kernel<<<blocks_for(plan.num_nodes, "signal_grid"),
                               threads, 0, exec->stream>>>(args);
  internal::throw_cuda("device-materialize", cudaGetLastError(), rank,
                       context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "parallel materialization kernel launch");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->mapping_complete, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(materialization complete)");
#if MEGAMOE_DEBUG_VALIDATE
  internal::throw_cuda(
      "device-materialize",
      cudaMemcpyAsync(at<void>(state->allocation->host_ptr,
                               state->host_layout.materialization_status),
                      args.status, sizeof(DeviceMaterializationStatus),
                      cudaMemcpyDeviceToHost, exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "D2H bounded materialization status");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::DeviceToHost,
      sizeof(DeviceMaterializationStatus));
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->completion, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord");
  internal::throw_cuda(
      "device-materialize",
      cudaEventSynchronize(state->allocation->completion), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventSynchronize");
#else
  auto *mapped_status = at<DeviceMaterializationStatus>(
      state->allocation->host_device_ptr,
      state->host_layout.materialization_status);
  auto *mapped_ready = at<std::uint32_t>(
      state->allocation->host_device_ptr,
      state->host_layout.materialization_ready);
  publish_materialization_status_kernel<<<1, 1, 0, exec->stream>>>(
      args.status, mapped_status, mapped_ready);
  internal::throw_cuda(
      "device-materialize", cudaGetLastError(), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "publish_materialization_status_kernel");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->completion, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapped materialization status)");
  std::uint64_t polls = 0;
  while (__atomic_load_n(host_ready, __ATOMIC_ACQUIRE) != 1U) {
    if ((++polls & 0x3ffU) == 0U) {
      const auto event = cudaEventQuery(state->allocation->completion);
      if (event != cudaSuccess && event != cudaErrorNotReady)
        internal::throw_cuda(
            "device-materialize", event, rank,
            context.session().world_size(), context.session().identity(),
            context.session().generation(), ExchangeKind::ExpertCounts,
            "cudaEventQuery(mapped materialization status)");
      if (event == cudaSuccess &&
          __atomic_load_n(host_ready, __ATOMIC_ACQUIRE) != 1U)
        invalid("phase=device-materialize, field=mapped_status_ready, rank=" +
                std::to_string(rank) +
                ", expected=1 before completion, actual=0");
    }
  }
#endif
  if (timings) {
    float planning_ms = 0;
    float remaining_ms = 0;
    internal::throw_cuda(
        "device-materialize",
        cudaEventElapsedTime(&planning_ms, state->allocation->planning_start,
                             state->allocation->planning_complete),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(prefix planning)");
    internal::throw_cuda(
        "device-materialize",
        cudaEventElapsedTime(&remaining_ms,
                             state->allocation->planning_complete,
                             state->allocation->mapping_complete),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(remaining materialization)");
    timings->prefix_planning_us = static_cast<double>(planning_ms) * 1000.0;
    timings->remaining_parallel_us =
        static_cast<double>(remaining_ms) * 1000.0;
  }
  const auto *status = at<const DeviceMaterializationStatus>(
      state->allocation->host_ptr,
      state->host_layout.materialization_status);
  if (status->error != 0) {
    auto code = NotifyErrorCode::CountExchangeFailure;
    if (status->error ==
        static_cast<unsigned long long>(MaterializationError::ExpertCapacity))
      code = NotifyErrorCode::ExpertCapacityExceeded;
    if (status->error == static_cast<unsigned long long>(
                             MaterializationError::WorkspaceCapacity))
      code = NotifyErrorCode::WorkspaceCapacityExceeded;
    if (status->error == static_cast<unsigned long long>(
                             MaterializationError::ArithmeticOverflow))
      code = NotifyErrorCode::ArithmeticOverflow;
    throw NotifyDispatchError(
        code, "phase=device-materialize, field=" +
                  std::to_string(status->field) + ", rank=" +
                  std::to_string(rank) + ", expected=" +
                  std::to_string(status->expected) + ", actual=" +
                  std::to_string(status->actual));
  }

  auto impl = std::make_shared<DeviceDispatchHandle::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->workspace = state;
  impl->expert_completion = std::move(expert_counts);
  impl->summary_completion = std::move(packet_summary);
  impl->device = local.impl_->device;
  impl->source_routes = local.impl_->routes;
  impl->source_gates = local.impl_->gates;
  impl->device.source_expert_count =
      impl->expert_completion.device_result();
  impl->device.packet_summary_count =
      impl->summary_completion.device_result();
  impl->device.owned_expert_ids =
      make_lease(state, dl.owned_expert_ids, status->owned_experts);
  impl->device.expert_row_count =
      make_lease(state, dl.owned_expert_count, status->owned_experts);
  impl->device.expert_row_prefix =
      make_lease(state, dl.owned_expert_prefix, status->owned_experts + 1);
  impl->device.mtile_row_offsets =
      make_lease(state, dl.mtile_offsets, status->owned_experts + 1);
  impl->device.mtile_expected_rows =
      make_lease(state, dl.mtile_rows, status->mtile_count);
  impl->device.incoming_rows =
      make_lease(state, dl.incoming_rows, status->incoming_rows);
  impl->device.reverse_routes =
      make_lease(state, dl.reverse_routes, plan.route_count);
  impl->device.packet_headers =
      make_lease(state, dl.packet_headers, status->packet_count);
  impl->device.packet_fanout_offsets =
      make_lease(state, dl.packet_fanout_offsets, status->packet_count + 1);
  impl->device.packet_fanout =
      make_lease(state, dl.packet_fanout, status->fanout_count);
  impl->device.local_route_order =
      make_lease(state, dl.local_route_order, plan.route_count);
  impl->device.dispatch_wave_packet_order =
      make_lease(state, dl.indices_out, status->packet_count);
  impl->device.dispatch_wave_offsets =
      make_lease(state, dl.dispatch_wave_offsets, status->wave_count + 1);
  impl->device.dispatch_wave_destination_node = make_lease(
      state, dl.dispatch_wave_destination_node, status->wave_count);
  impl->device.dispatch_wave_channel =
      make_lease(state, dl.dispatch_wave_channel, status->wave_count);
  impl->device.dispatch_wave_first_primary_tile = make_lease(
      state, dl.dispatch_wave_first_primary_tile, status->wave_count);
  impl->device.dispatch_wave_last_primary_tile = make_lease(
      state, dl.dispatch_wave_last_primary_tile, status->wave_count);
  impl->device.dispatch_wave_packet_count =
      make_lease(state, dl.dispatch_wave_packet_count, status->wave_count);
  impl->device.dispatch_wave_payload_bytes =
      make_lease(state, dl.dispatch_wave_payload_bytes, status->wave_count);
  impl->device.outgoing_signals =
      make_lease(state, dl.outgoing_signals, plan.num_nodes);
  impl->device.incoming_proxy_signals =
      make_lease(state, dl.incoming_signals, plan.num_nodes);
  impl->identity.generation = context.session().generation();
  impl->identity.calling_rank = rank;
  impl->identity.world_size = plan.world_size;
  impl->identity.num_tokens = plan.num_tokens;
  impl->identity.top_k = plan.top_k;
  impl->identity.num_experts = plan.num_experts;
  impl->identity.num_nodes = plan.num_nodes;
  const auto wire = encode_canonical_config(
      config, plan.max_tokens_per_rank,
      context.session().stable_identity_bytes());
  const auto config_identity = stable_session_id(wire);
  impl->identity.config_digest_word0 = config_identity.word0;
  impl->identity.config_digest_word1 = config_identity.word1;
  impl->sizes.route_count = plan.route_count;
  impl->sizes.node_mask_words = plan.node_mask_words;
  impl->sizes.owned_expert_count =
      static_cast<std::uint32_t>(status->owned_experts);
  impl->sizes.incoming_row_count = status->incoming_rows;
  impl->sizes.mtile_count = status->mtile_count;
  impl->sizes.packet_count = status->packet_count;
  impl->sizes.fanout_count = status->fanout_count;
  impl->dispatch_wave_count = status->wave_count;
  impl->sizes.outgoing_signal_count = plan.num_nodes;
  impl->sizes.incoming_signal_count = plan.num_nodes;
  return DeviceDispatchHandle(std::move(impl));
}

DeviceDispatchHandle testing::materialize_fake_device_collectives(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const RankRoutingInput &local_routes,
    std::uint64_t max_tokens_per_rank,
    const std::vector<std::uint64_t> &expert_count_matrix,
    const std::vector<std::uint64_t> &packet_summary_matrix) {
  if (!context || !execution ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=fake-device-materializer, field=context, rank=n/a, "
            "expected=valid exact execution, actual=mismatch");
  const auto world = config.topology.world_size;
  const auto count_elements = checked_mul(world, config.num_experts,
                                          "fake count matrix elements");
  const auto summary_elements = checked_mul(
      world, config.topology.num_nodes, "fake summary matrix elements");
  if (expert_count_matrix.size() != checked_size(count_elements, "counts") ||
      packet_summary_matrix.size() !=
          checked_size(summary_elements, "summaries"))
    invalid("phase=fake-device-materializer, field=matrix_shape, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact initialized matrices, actual=mismatch");

  auto routes = upload_cuda_routes(local_routes, config.top_k, context,
                                   execution);
  std::vector<float> unit_gates(
      checked_size(routes.route_count(), "fake materializer gate count"),
      1.0F);
  auto gates = testing::upload_fp32_gates_for_testing(
      unit_gates, routes, context, execution);
  const auto plan = query_cuda_notify_workspace(
      config, context.calling_rank(), local_routes.num_tokens,
      max_tokens_per_rank);
  auto workspace = allocate_cuda_notify_workspace(plan, context, execution);
  auto operation = start_cuda_rank_local_notify(
      config, context, execution, routes, gates, std::move(workspace));
  auto preflight = operation.wait_preflight();
  if (!preflight.valid())
    invalid("phase=fake-device-materializer, field=preflight, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=valid, actual=failed");
  const auto exec = internal::checked_execution_state(execution);

  const auto make_matrix = [&](const std::vector<std::uint64_t> &host,
                               std::uint64_t rows, std::uint64_t width,
                               const char *field) {
    auto allocation = std::make_shared<internal::CudaAllocation>();
    allocation->device = exec->device;
    allocation->device_bytes = checked_size(
        checked_mul(host.size(), sizeof(std::uint64_t), field), field);
    internal::throw_cuda(
        "fake-device-materializer",
        cudaMalloc(&allocation->device_ptr,
                   std::max<std::size_t>(allocation->device_bytes, 1)),
        context.calling_rank(), world, context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(fake initialized matrix)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    if (!host.empty()) {
      internal::throw_cuda(
          "fake-device-materializer",
          cudaMemcpyAsync(allocation->device_ptr, host.data(),
                          allocation->device_bytes, cudaMemcpyHostToDevice,
                          exec->stream),
          context.calling_rank(), world, context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(fake initialized matrix)");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                allocation->device_bytes);
    }
    auto buffer = std::make_shared<internal::CudaBufferState>();
    buffer->owner = allocation;
    buffer->pointer = allocation->device_ptr;
    buffer->device = exec->device;
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = allocation;
    return std::make_pair(
        make_device_buffer_lease(rows, width, host.size(), std::move(buffer)),
        std::static_pointer_cast<const BackendCompletionLeaseState>(lease));
  };

  auto count = make_matrix(expert_count_matrix, world, config.num_experts,
                           "fake count matrix bytes");
  auto summary = make_matrix(packet_summary_matrix, world,
                             config.topology.num_nodes,
                             "fake summary matrix bytes");
  auto count_completion = make_device_backend_completion(
      context, ExchangeKind::ExpertCounts, std::move(count.first),
      std::move(count.second));
  auto summary_completion = make_device_backend_completion(
      context, ExchangeKind::NotifyPacketSummary, std::move(summary.first),
      std::move(summary.second));
  return DeviceMetadataBridge::materialize(
      preflight.local(), std::move(count_completion),
      std::move(summary_completion), execution);
}

const DeviceDispatchIdentity &DeviceDispatchHandle::identity() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle identity");
  return impl_->identity;
}

const DeviceDispatchSizes &DeviceDispatchHandle::sizes() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle sizes");
  return impl_->sizes;
}

const DeviceMaterializationAudit &DeviceDispatchHandle::audit() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle audit");
  return impl_->audit;
}

DeviceDispatchConsumerLease DeviceDispatchHandle::acquire_consumer(
    const CudaExecution &execution,
    const DeviceDispatchViewExpectation &expected) const {
  if (!impl_ || !execution)
    invalid("phase=device-view, field=handle/execution, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &id = impl_->identity;
  if (id.generation != expected.generation ||
      id.calling_rank != expected.calling_rank ||
      id.world_size != expected.world_size ||
      id.num_tokens != expected.num_tokens || id.top_k != expected.top_k)
    invalid("phase=device-view, field=identity/shape/epoch, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=exact handle provenance, actual=mismatch");
  if (id.config_digest_word0 != expected.config_digest_word0 ||
      id.config_digest_word1 != expected.config_digest_word1)
    invalid("phase=device-view, field=config_identity, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=exact handle config digest, actual=mismatch");
  auto exec = internal::checked_execution_state(execution);
  if (exec->device != impl_->execution.device_ordinal())
    invalid("phase=dispatch-consumer-acquire, field=device, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=handle device, actual=foreign device");
  auto workspace = impl_->workspace;
  std::size_t event_index = SIZE_MAX;
  {
    std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
    if (workspace->allocation->dispatch_slot_quarantined.load(
            std::memory_order_acquire))
      invalid("phase=dispatch-consumer-acquire, field=slot_state, rank=" +
              std::to_string(id.calling_rank) +
              ", expected=usable, actual=Quarantined");
    for (std::size_t i = 0;
         i < workspace->allocation->consumer_event_states.size(); ++i) {
      if (workspace->allocation->consumer_event_states[i] ==
          internal::ConsumerEventSlotState::Available) {
        workspace->allocation->consumer_event_states[i] =
            internal::ConsumerEventSlotState::InUse;
        event_index = i;
        break;
      }
    }
  }
  if (event_index == SIZE_MAX)
    throw NotifyDispatchError(
        NotifyErrorCode::ResourceExhausted,
        "phase=dispatch-consumer-acquire, field=consumer_events, rank=" +
            std::to_string(id.calling_rank) +
            ", expected>=1 available pooled event, actual=0");
  try {
    internal::throw_cuda(
        "dispatch-consumer-acquire",
        cudaStreamWaitEvent(exec->stream, workspace->allocation->completion),
        id.calling_rank, id.world_size, impl_->context.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(notify completion)");
  } catch (...) {
    std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
    workspace->allocation->consumer_event_states[event_index] =
        internal::ConsumerEventSlotState::Available;
    throw;
  }
  auto consumer = std::make_shared<DeviceDispatchConsumerLease::Impl>();
  consumer->owner = *this;
  consumer->workspace = impl_->workspace;
  consumer->device = impl_->device;
  consumer->identity = impl_->identity;
  consumer->sizes = impl_->sizes;
  consumer->context = impl_->context;
  consumer->epoch = impl_->workspace->epoch;
  consumer->execution = execution;
  consumer->event_index = event_index;
  return DeviceDispatchConsumerLease(std::move(consumer));
}

DeviceDispatchConsumerLease::~DeviceDispatchConsumerLease() noexcept {
  if (impl_)
    impl_->release_noexcept();
}
DeviceDispatchConsumerLease::DeviceDispatchConsumerLease(
    DeviceDispatchConsumerLease &&other) noexcept = default;
DeviceDispatchConsumerLease &DeviceDispatchConsumerLease::operator=(
    DeviceDispatchConsumerLease &&other) noexcept {
  if (this != &other) {
    if (impl_)
      impl_->release_noexcept();
    impl_ = std::move(other.impl_);
  }
  return *this;
}
DeviceDispatchConsumerState DeviceDispatchConsumerLease::state() const {
  if (!impl_)
    invalid("phase=dispatch-consumer-state, field=lease, rank=n/a, "
            "expected=valid, actual=empty/moved-from");
  return impl_->state;
}
