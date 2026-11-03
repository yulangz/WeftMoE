#pragma once

// Implementation fragment included only by device_notify.cu, inside
// weftmoe::cuda, after the workspace, input, operation, and kernel definitions.
// Materialization and consumer ownership stay together through completion.

struct DeviceDispatchHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  DeviceDispatchMetadata device;
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  DeviceDispatchCapacity capacity;
  CudaLocalNotifyHandle local_notify;
  std::uint64_t dispatch_wave_count = 0;
  std::uint64_t padded_compute_rows = 0;
  DeviceMaterializationAudit audit;
  CudaRouteInput source_routes;
  CudaGateInput source_gates;
  std::shared_ptr<WorkspaceState> workspace;
  bool fused_forward_notify = false;
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
    if (workspace_ref.replay_counts || workspace_ref.training_counts)
      view.runtime = at<const DeviceDispatchRuntime>(base, dl.runtime);
    view.expert_id = at<const std::uint32_t>(base, dl.expert_id);
    view.expert_to_rank = at<const std::uint32_t>(base, dl.expert_to_rank);
    view.local_offset = at<const std::uint64_t>(base, dl.local_offset);
    view.return_global_expert_base =
        at<const std::uint64_t>(base, dl.return_global_expert_base);
    view.route_dst_row = at<const std::uint64_t>(base, dl.route_dst_row);
    view.route_dst_rank = at<const std::uint32_t>(base, dl.route_dst_rank);
    view.route_dst_node = at<const std::uint32_t>(base, dl.route_dst_node);
    view.route_dst_local_rank =
        at<const std::uint32_t>(base, dl.route_dst_local_rank);
    view.combine_order = at<const std::uint32_t>(base, dl.combine_order);
    view.destination_node_mask = at<const std::uint64_t>(base, dl.masks);
    view.source_expert_count =
        static_cast<const std::uint64_t *>(count->pointer);
    view.owned_expert_ids = at<const std::uint32_t>(base, dl.owned_expert_ids);
    view.expert_row_count =
        at<const std::uint64_t>(base, dl.owned_expert_count);
    view.expert_row_prefix =
        at<const std::uint64_t>(base, dl.owned_expert_prefix);
    view.mtile_row_offsets = at<const std::uint64_t>(base, dl.mtile_offsets);
    view.mtile_expected_rows = workspace_ref.receive->tiles();
    view.reverse_routes =
        at<const DeviceReverseRouteRecord>(base, dl.reverse_routes);
    view.packet_headers =
        at<const DevicePacketHeaderRecord>(base, dl.packet_headers);
    view.packet_fanout_offsets =
        at<const std::uint64_t>(base, dl.packet_fanout_offsets);
    view.packet_fanout = at<const DeviceFanoutRecord>(base, dl.packet_fanout);
    view.outgoing_signals =
        at<const DeviceSignalRecord>(base, dl.outgoing_signals);
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
    const bool injected_event =
        injected_both ||
        take_consumer_fault(testing::ConsumerFaultPoint::EventSynchronize);
    const bool event_ok =
        !injected_event &&
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
          internal::record_observable_event(
              workspace->allocation->consumer_events[event_index],
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
        std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
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
      completion.kind() != kind || !completion.context().same_instance(context))
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
    Timings *timings, bool enqueue_only) {
  if (!local || !execution)
    invalid("phase=device-materialize, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &context = local.impl_->context;
  const auto &config = local.impl_->config;
  if (!context.execution().same_instance(execution.token()))
    invalid("phase=device-materialize, field=execution, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=local execution, actual=different");
  validate_device_completion(expert_counts, context, ExchangeKind::ExpertCounts,
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
  internal::throw_cuda(
      "device-materialize", cudaSetDevice(static_cast<int>(exec->device)), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaSetDevice");

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
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  };
  h2d(dl.expert_capacity, config.capacity.expert_rows.data(),
      config.capacity.expert_rows.size(), sizeof(std::uint64_t),
      "expert capacity");
  h2d(dl.workspace_capacity, config.capacity.workspace_rows_per_rank.data(),
      config.capacity.workspace_rows_per_rank.size(), sizeof(std::uint64_t),
      "workspace capacity");

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
  args.route_dst_local = at<const std::uint32_t>(base, dl.route_dst_local_rank);
  args.combine_order = at<const std::uint32_t>(base, dl.combine_order);
  args.masks = at<const std::uint64_t>(base, dl.masks);
  args.expert_total = at<std::uint64_t>(base, dl.expert_total);
  args.expert_base = at<std::uint64_t>(base, dl.expert_base);
  args.return_expert_base = at<std::uint64_t>(base, dl.return_expert_base);
  args.return_global_expert_base =
      at<std::uint64_t>(base, dl.return_global_expert_base);
  args.expert_tile_base = at<std::uint64_t>(base, dl.expert_tile_base);
  args.route_dst_row = at<std::uint64_t>(base, dl.route_dst_row);
  args.route_dst_mtile = at<std::uint32_t>(base, dl.route_dst_mtile);
  args.owned_expert_ids = at<std::uint32_t>(base, dl.owned_expert_ids);
  args.owned_expert_count = at<std::uint64_t>(base, dl.owned_expert_count);
  args.owned_expert_prefix = at<std::uint64_t>(base, dl.owned_expert_prefix);
  args.mtile_offsets = at<std::uint64_t>(base, dl.mtile_offsets);
  args.mtile_rows = state->receive->tiles();
  args.reverse_routes = at<DeviceReverseRouteRecord>(base, dl.reverse_routes);
  args.packet_headers = at<DevicePacketHeaderRecord>(base, dl.packet_headers);
  args.packet_block_counts = at<ulonglong2>(base, dl.packet_block_counts);
  args.packet_chunks = static_cast<unsigned>(
      std::max<std::uint64_t>(1, (plan.num_tokens + 255) / 256));
  args.packet_fanout_offsets =
      at<std::uint64_t>(base, dl.packet_fanout_offsets);
  args.packet_fanout = at<DeviceFanoutRecord>(base, dl.packet_fanout);
  args.packet_primary_tile = at<std::uint64_t>(base, dl.packet_primary_tile);
  args.packet_primary_expert =
      at<std::uint32_t>(base, dl.packet_primary_expert);
  args.packet_primary_rank = at<std::uint32_t>(base, dl.packet_primary_rank);
  args.local_route_order = at<std::uint64_t>(base, dl.local_route_order);
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
  args.outgoing_signals = at<DeviceSignalRecord>(base, dl.outgoing_signals);
  args.incoming_signals = at<DeviceSignalRecord>(base, dl.incoming_signals);
  args.status =
      at<DeviceMaterializationStatus>(base, dl.materialization_status);

  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_start, exec->stream), rank,
      plan.world_size, context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "record notify materialization start");
  const auto packed_wave_key_fits = [&] {
    if (plan.num_nodes == 0U || plan.world_size == 0U ||
        plan.max_tokens_per_rank == 0U)
      return false;
    const auto max_value = std::numeric_limits<std::uint64_t>::max();
    auto key_space = std::uint64_t(plan.num_nodes - 1U);
    auto extend = [&](std::uint64_t radix) {
      if (radix == 0U || key_space > (max_value - (radix - 1U)) / radix)
        return false;
      key_space = key_space * radix + radix - 1U;
      return true;
    };
    const auto expert_radix =
        static_cast<std::uint64_t>(plan.num_experts) *
        (plan.world_size == plan.num_nodes * 8U ? 8ULL : 1ULL);
    return extend(plan.max_incoming_rows) && extend(expert_radix) &&
           extend(plan.world_size) && extend(plan.max_tokens_per_rank);
  }();
  const auto max_key = std::numeric_limits<std::uint64_t>::max();
  const bool packed_local_key_fits =
      plan.max_incoming_rows <= (max_key / 2U - plan.num_experts) /
                                    plan.num_experts /
                                    ((config.m_tile_rows + 31U) / 32U);

  const auto notify_wire =
      encode_canonical_config(config, plan.max_tokens_per_rank,
                              context.session().stable_identity_bytes());
  const auto notify_digest = stable_session_id(notify_wire);
  const auto comm_view = state->communication ? state->communication->view
                                              : NotifyCommunicationView{};
  const bool single_node_inference = state->single_kernel &&
                                     state->unstable_inference_rows &&
                                     config.topology.num_nodes == 1U;
  int sms = 0, resident = 0;
  internal::throw_cuda("device-materialize",
                       cudaDeviceGetAttribute(
                           &sms, cudaDevAttrMultiProcessorCount, exec->device),
                       rank, plan.world_size, context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "notify SM count");
  internal::throw_cuda(
      "device-materialize",
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &resident,
          single_node_inference
              ? single_node_inference_notify_kernel
              : (state->training_counts ? training_count_notify_kernel
                 : state->single_kernel ? single_notify_kernel
                                        : materialize_notify_kernel),
          256, 0),
      rank, plan.world_size, context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "notify residency");
  if (resident < 1)
    invalid("materialization kernel has no resident CTA");
  auto *mapped_ready = at<std::uint32_t>(
      state->allocation->host_ptr, state->host_layout.materialization_ready);
  if (!enqueue_only)
    __atomic_store_n(mapped_ready, 0U, __ATOMIC_RELEASE);
  cudaLaunchAttribute cooperative{};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  cudaLaunchConfig_t launch{};
  const auto work_blocks = std::max<std::uint64_t>(
      (plan.route_count + 1023) / 1024,
      (plan.num_experts + WEFTMOE_NOTIFY_EXPERTS_PER_CTA - 1U) /
          WEFTMOE_NOTIFY_EXPERTS_PER_CTA);
  // Cooperative synchronization requires the entire grid to remain resident.
  // Use the occupancy limit to hide memory latency on larger route tables.
  launch.gridDim = dim3(static_cast<unsigned>(std::max<std::uint64_t>(
      1, std::min<std::uint64_t>(std::uint64_t(sms) * resident, work_blocks))));
  launch.blockDim = dim3(256);
  launch.stream = exec->stream;
  cudaLaunchAttribute attrs[] = {cooperative, programmatic_launch::attribute()};
  launch.attrs = attrs;
  launch.numAttrs = programmatic_launch::enabled() ? 2U : 1U;
  const auto enqueue = [&](auto kernel) {
    return cudaLaunchKernelEx(
        &launch, kernel, args, at<std::uint64_t>(base, dl.sort_histogram),
        at<DeviceDispatchRuntime>(base, dl.runtime),
        at<DeviceMaterializationStatus>(
            state->allocation->host_device_ptr,
            state->host_layout.materialization_status),
        enqueue_only
            ? nullptr
            : at<std::uint32_t>(state->allocation->host_device_ptr,
                                state->host_layout.materialization_ready),
        packed_wave_key_fits, packed_local_key_fits, state->local_args,
        comm_view, notify_digest.word0 ^ context.session().generation(),
        notify_digest.word1);
  };
  internal::throw_cuda(
      "device-materialize",
      single_node_inference
          ? enqueue(single_node_inference_notify_kernel)
          : (state->training_counts ? enqueue(training_count_notify_kernel)
             : state->single_kernel ? enqueue(single_notify_kernel)
                                    : enqueue(materialize_notify_kernel)),
      rank, plan.world_size, context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "fused notify materialization");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_complete, exec->stream), rank,
      plan.world_size, context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "record notify planning complete");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->mapping_complete, exec->stream), rank,
      plan.world_size, context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "record notify materialization complete");
  if (enqueue_only) {
    internal::throw_cuda("device-materialize",
                         internal::record_input_ready(
                             state->allocation->completion, exec->stream,
                             state->allocation->completion_capture_id),
                         rank, plan.world_size, context.session().identity(),
                         context.session().generation(),
                         ExchangeKind::ExpertCounts, "record enqueued Notify");
    return {};
  }
  read_materialization_status(state, exec->stream);
  if (timings) {
    internal::throw_cuda("device-materialize",
                         cudaEventSynchronize(state->allocation->completion),
                         rank, plan.world_size, context.session().identity(),
                         context.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "finish materialization timing events");
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
    timings->remaining_parallel_us = static_cast<double>(remaining_ms) * 1000.0;
  }
  const auto *status = at<const DeviceMaterializationStatus>(
      state->allocation->host_ptr, state->host_layout.materialization_status);
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
        code,
        "phase=device-materialize, field=" + std::to_string(status->field) +
            ", rank=" + std::to_string(rank) +
            ", expected=" + std::to_string(status->expected) +
            ", actual=" + std::to_string(status->actual));
  }

  auto impl = std::make_shared<DeviceDispatchHandle::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->local_notify = local;
  impl->workspace = state;
  impl->expert_completion = std::move(expert_counts);
  impl->summary_completion = std::move(packet_summary);
  impl->device = local.impl_->device;
  impl->source_routes = local.impl_->routes;
  impl->source_gates = local.impl_->gates;
  impl->device.source_expert_count = impl->expert_completion.device_result();
  impl->device.packet_summary_count = impl->summary_completion.device_result();
  impl->device.owned_expert_ids =
      make_lease(state, dl.owned_expert_ids, status->owned_experts);
  impl->device.expert_row_count =
      make_lease(state, dl.owned_expert_count, status->owned_experts);
  impl->device.expert_row_prefix =
      make_lease(state, dl.owned_expert_prefix, status->owned_experts + 1);
  impl->device.mtile_row_offsets =
      make_lease(state, dl.mtile_offsets, status->owned_experts + 1);
  const auto receive_lease = [&](void *pointer, std::uint64_t count) {
    if (count == 0)
      return DeviceBufferLease{};
    auto buffer = std::make_shared<CudaBufferState>();
    buffer->owner = state;
    buffer->pointer = pointer;
    buffer->device = exec->device;
    return make_device_buffer_lease(1, count, count, std::move(buffer));
  };
  impl->device.mtile_expected_rows =
      receive_lease(state->receive->tiles(), status->mtile_count);
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
  impl->device.dispatch_wave_destination_node =
      make_lease(state, dl.dispatch_wave_destination_node, status->wave_count);
  impl->device.dispatch_wave_channel =
      make_lease(state, dl.dispatch_wave_channel, status->wave_count);
  impl->device.dispatch_wave_first_primary_tile = make_lease(
      state, dl.dispatch_wave_first_primary_tile, status->wave_count);
  impl->device.dispatch_wave_last_primary_tile =
      make_lease(state, dl.dispatch_wave_last_primary_tile, status->wave_count);
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
  const auto wire =
      encode_canonical_config(config, plan.max_tokens_per_rank,
                              context.session().stable_identity_bytes());
  const auto config_identity = stable_session_id(wire);
  impl->identity.config_digest_word0 = config_identity.word0;
  impl->identity.config_digest_word1 = config_identity.word1;
  impl->capacity = {status->max_source_tokens, status->max_destination_rows};
  impl->sizes.route_count = plan.route_count;
  impl->sizes.node_mask_words = plan.node_mask_words;
  impl->sizes.owned_expert_count =
      static_cast<std::uint32_t>(status->owned_experts);
  impl->sizes.incoming_row_count = status->incoming_rows;
  impl->padded_compute_rows = status->padded_compute_rows;
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
    const CudaExecution &execution, const RankRoutingInput &local_routes,
    std::uint64_t max_tokens_per_rank,
    const std::vector<std::uint64_t> &expert_count_matrix,
    const std::vector<std::uint64_t> &packet_summary_matrix) {
  if (!context || !execution ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=fake-device-materializer, field=context, rank=n/a, "
            "expected=valid exact execution, actual=mismatch");
  const auto world = config.topology.world_size;
  const auto count_elements =
      checked_mul(world, config.num_experts, "fake count matrix elements");
  const auto summary_elements = checked_mul(world, config.topology.num_nodes,
                                            "fake summary matrix elements");
  if (expert_count_matrix.size() != checked_size(count_elements, "counts") ||
      packet_summary_matrix.size() !=
          checked_size(summary_elements, "summaries"))
    invalid("phase=fake-device-materializer, field=matrix_shape, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact initialized matrices, actual=mismatch");

  auto routes =
      upload_cuda_routes(local_routes, config.top_k, context, execution);
  std::vector<float> unit_gates(
      checked_size(routes.route_count(), "fake materializer gate count"), 1.0F);
  auto gates = testing::upload_fp32_gates_for_testing(unit_gates, routes,
                                                      context, execution);
  const auto plan =
      query_cuda_notify_workspace(config, context.calling_rank(),
                                  local_routes.num_tokens, max_tokens_per_rank);
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
  auto summary =
      make_matrix(packet_summary_matrix, world, config.topology.num_nodes,
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

void DeviceNotifyPoolBridge::reject_single(
    const RankCollectiveContext &context, const CudaExecution &execution,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) {
  const auto exec = internal::checked_execution_state(execution);
  const auto &plan = allocation->max_plan;
  const auto dl = make_device_layout(plan);
  auto *base = allocation->workspace->device_ptr;
  LocalNotifyArgs local{};
  local.histogram = at<std::uint64_t>(base, dl.expert_count);
  local.packet_count = at<std::uint64_t>(base, dl.packet_count);
  local.status = at<DeviceStatus>(base, dl.status);
  DeviceMaterializeArgs args{};
  args.rank = context.calling_rank();
  args.world = plan.world_size;
  args.experts = plan.num_experts;
  args.nodes = plan.num_nodes;
  args.count_matrix = at<const std::uint64_t>(base, dl.count_matrix);
  args.summary_matrix = at<const std::uint64_t>(base, dl.summary_matrix);
  cudaLaunchAttribute cooperative{};
  cooperative.id = cudaLaunchAttributeCooperative;
  cooperative.val.cooperative = 1;
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(1);
  launch.blockDim = dim3(256);
  launch.stream = exec->stream;
  launch.attrs = &cooperative;
  launch.numAttrs = 1;
  internal::throw_cuda("notify-reject", cudaSetDevice(exec->device), args.rank,
                       args.world, context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "set rejection device");
  internal::throw_cuda("notify-reject",
                       cudaLaunchKernelEx(&launch, reject_notify_kernel, local,
                                          args,
                                          allocation->communication->view),
                       args.rank, args.world, context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "join notify rejection");
  internal::throw_cuda("notify-reject", cudaStreamSynchronize(exec->stream),
                       args.rank, args.world, context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "complete notify rejection");
}

DeviceDispatchHandle DeviceNotifyPoolBridge::run_single(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    const CudaGateInput &gates, std::uint64_t max_tokens,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    std::shared_ptr<const void> slot_lease) {
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  DeviceGateProvenanceBridge::validate(gates, routes, context, execution);
  const auto plan = query_cuda_notify_workspace(
      config, context.calling_rank(), routes.num_tokens(), max_tokens);
  auto workspace = DeviceNotifyPoolBridge::workspace(
      plan, context, execution, allocation, std::move(slot_lease));
  auto state = workspace.impl_->state;
  state->single_kernel = true;
  const auto owned = config.num_experts / config.topology.world_size;
  bool fused_forward_notify =
      (state->unstable_inference_rows ||
       (state->training_counts && state->defer_training_notify)) &&
      owned * config.topology.world_size == config.num_experts;
  for (std::uint32_t expert = 0;
       fused_forward_notify && expert < config.num_experts; ++expert)
    fused_forward_notify =
        config.topology.expert_to_rank[expert] == expert / owned;
  if (!fused_forward_notify)
    enqueue_local_notify(config, context, execution, routes, state);
  auto result = std::make_shared<CudaLocalNotifyHandle::Impl>();
  result->config = config;
  result->context = context;
  result->num_tokens = state->plan.num_tokens;
  result->routes = routes;
  result->gates = gates;
  result->workspace = state;
  const auto route_width = state->plan.route_count;
  result->device.expert_id =
      make_lease(state, state->device_layout.expert_id, route_width);
  result->device.local_offset =
      make_lease(state, state->device_layout.local_offset, route_width);
  result->device.route_dst_row =
      make_lease(state, state->device_layout.route_dst_row, route_width);
  result->device.route_dst_mtile =
      make_lease(state, state->device_layout.route_dst_mtile, route_width);
  result->device.route_dst_rank =
      make_lease(state, state->device_layout.route_dst_rank, route_width);
  result->device.route_dst_node =
      make_lease(state, state->device_layout.route_dst_node, route_width);
  result->device.route_dst_local_rank =
      make_lease(state, state->device_layout.route_dst_local_rank, route_width);
  result->device.combine_order =
      make_lease(state, state->device_layout.combine_order, route_width);
  result->device.destination_node_mask =
      make_lease(state, state->device_layout.masks,
                 checked_mul(state->plan.num_tokens,
                             state->plan.node_mask_words, "mask lease"));
  result->expert_count = make_lease(state, state->device_layout.expert_count,
                                    state->plan.num_experts);
  result->packet_count = make_lease(state, state->device_layout.packet_count,
                                    state->plan.num_nodes);

  auto local = CudaLocalNotifyHandle(std::move(result));
  const auto make_matrix = [&](std::size_t offset, std::uint64_t width,
                               ExchangeKind kind) {
    auto buffer = std::make_shared<CudaBufferState>();
    buffer->owner = state;
    buffer->pointer = at<void>(state->allocation->device_ptr, offset);
    buffer->device = execution.device_ordinal();
    auto matrix = make_device_buffer_lease(plan.world_size, width,
                                           plan.world_size * width, buffer);
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = state;
    return make_device_backend_completion(context, kind, std::move(matrix),
                                          std::move(lease));
  };
  auto expert_completion =
      make_matrix(state->device_layout.count_matrix, plan.num_experts,
                  ExchangeKind::ExpertCounts);
  auto summary_completion =
      make_matrix(state->device_layout.summary_matrix, plan.num_nodes,
                  ExchangeKind::NotifyPacketSummary);
  if (!fused_forward_notify)
    return DeviceMetadataBridge::materialize(
        local, std::move(expert_completion), std::move(summary_completion),
        execution);

  const auto exec = internal::checked_execution_state(execution);
  internal::throw_cuda(
      "prepare-fused-forward-notify",
      internal::record_input_ready(state->allocation->completion, exec->stream,
                                   state->allocation->completion_capture_id),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "record fused Forward metadata readiness");
  auto impl = std::make_shared<DeviceDispatchHandle::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->local_notify = local;
  impl->workspace = state;
  impl->expert_completion = std::move(expert_completion);
  impl->summary_completion = std::move(summary_completion);
  impl->device = local.impl_->device;
  impl->device.source_expert_count = impl->expert_completion.device_result();
  impl->device.packet_summary_count = impl->summary_completion.device_result();
  impl->source_routes = routes;
  impl->source_gates = gates;
  impl->fused_forward_notify = true;
  impl->identity.generation = context.session().generation();
  impl->identity.calling_rank = context.calling_rank();
  impl->identity.world_size = plan.world_size;
  impl->identity.num_tokens = plan.num_tokens;
  impl->identity.top_k = plan.top_k;
  impl->identity.num_experts = plan.num_experts;
  impl->identity.num_nodes = plan.num_nodes;
  const auto wire =
      encode_canonical_config(config, plan.max_tokens_per_rank,
                              context.session().stable_identity_bytes());
  const auto config_identity = stable_session_id(wire);
  impl->identity.config_digest_word0 = config_identity.word0;
  impl->identity.config_digest_word1 = config_identity.word1;
  const auto row_capacity =
      config.capacity.workspace_rows_per_rank.empty()
          ? plan.max_incoming_rows
          : std::min(plan.max_incoming_rows,
                     config.capacity
                         .workspace_rows_per_rank[context.calling_rank()]);
  impl->capacity = {plan.max_tokens_per_rank, row_capacity};
  impl->sizes.route_count = plan.route_count;
  impl->sizes.node_mask_words = plan.node_mask_words;
  impl->sizes.owned_expert_count = owned;
  impl->sizes.incoming_row_count = row_capacity;
  impl->sizes.mtile_count = row_capacity;
  impl->sizes.packet_count = plan.num_tokens;
  impl->sizes.fanout_count = plan.route_count;
  impl->sizes.outgoing_signal_count = 1U;
  impl->sizes.incoming_signal_count = 1U;
  return DeviceDispatchHandle(std::move(impl));
}

void DeviceDispatchHandle::enqueue_notify(const NcclSession &session) const {
  if (!impl_)
    invalid("enqueue_notify requires prepared notify storage");
  if (session.session().stable_identity_bytes() !=
          impl_->context.session().stable_identity_bytes() ||
      session.rank() != impl_->identity.calling_rank)
    invalid("enqueue_notify EP session does not match the prepared storage");
  const auto &state = impl_->workspace;
  if (impl_->fused_forward_notify)
    return;
  state->replay_counts = true;
  const auto exec = internal::checked_execution_state(impl_->execution);
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (state->communication) {
    state->single_kernel = true;
    enqueue_local_notify(impl_->config, impl_->context, impl_->execution,
                         impl_->source_routes, state);
    DeviceMetadataBridge::materialize(
        impl_->local_notify, impl_->expert_completion,
        impl_->summary_completion, impl_->execution, nullptr, true);
    return;
  }
  enqueue_local_notify(impl_->config, impl_->context, impl_->execution,
                       impl_->source_routes, state);
  auto counts = internal::CudaDeviceLeaseAccess::checked(
      impl_->expert_completion.device_result(), exec->device);
  auto summary = internal::CudaDeviceLeaseAccess::checked(
      impl_->summary_completion.device_result(), exec->device);
  const auto gather = [&](std::size_t offset, void *target, std::size_t width) {
    internal::throw_nccl(
        "enqueue-notify",
        ncclAllGather(at<void>(state->allocation->device_ptr, offset), target,
                      width, ncclUint64, nccl->comm, exec->stream),
        session.rank(), session.session().world_size(),
        session.session().identity(), impl_->identity.generation,
        ExchangeKind::ExpertCounts, "ncclAllGather(Notify)");
  };
  gather(state->device_layout.expert_count, const_cast<void *>(counts->pointer),
         impl_->config.num_experts);
  gather(state->device_layout.packet_count,
         const_cast<void *>(summary->pointer),
         impl_->config.topology.num_nodes);
  DeviceMetadataBridge::materialize(
      impl_->local_notify, impl_->expert_completion, impl_->summary_completion,
      impl_->execution, nullptr, true);
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

const DeviceDispatchCapacity &DeviceDispatchHandle::capacity() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle capacity");
  return impl_->capacity;
}

std::uint64_t DeviceDispatchHandle::received_row_capacity() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle capacity");
  return impl_->sizes.incoming_row_count;
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
        workspace->allocation->consumer_work_completed.store(
            false, std::memory_order_release);
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
  if (!impl_->execution.token().same_instance(execution.token())) {
    try {
      internal::throw_cuda("dispatch-consumer-acquire",
                           internal::wait_input_ready(
                               exec->stream, workspace->allocation->completion,
                               workspace->allocation->completion_capture_id),
                           id.calling_rank, id.world_size,
                           impl_->context.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaStreamWaitEvent(notify completion)");
    } catch (...) {
      std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
      workspace->allocation->consumer_event_states[event_index] =
          internal::ConsumerEventSlotState::Available;
      throw;
    }
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
void DeviceDispatchConsumerLease::finish() {
  if (!impl_ || impl_->state == DeviceDispatchConsumerState::Released)
    invalid("phase=dispatch-consumer-finish, field=state, rank=n/a, "
            "expected=acquired/submitted, actual=empty/released");
  if (impl_->state == DeviceDispatchConsumerState::Submitted) {
    const auto &id = impl_->identity;
    if (!impl_->drain_submitted_noexcept())
      throw NotifyDispatchError(
          NotifyErrorCode::CudaFailure,
          "phase=dispatch-consumer-finish, field=consumer_event, rank=" +
              std::to_string(id.calling_rank) +
              ", expected=synchronized, actual=CUDA event failure; event "
              "quarantined");
    return;
  }
  impl_->release_noexcept();
}

DeviceDispatchKernelView
ActivationTransportBridge::view(DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_ ||
      consumer.impl_->state != DeviceDispatchConsumerState::Acquired)
    invalid("phase=activation-transport-view, field=consumer, rank=n/a, "
            "expected=Acquired, actual=invalid");
  return consumer.impl_->kernel_view();
}

void ActivationTransportBridge::finish_completed(
    DeviceDispatchConsumerLease &consumer) {
  // Keep the workspace (and its pool lease) alive until completion is marked.
  auto workspace = consumer.impl_->workspace;
  consumer.finish();
  auto &allocation = *workspace->allocation;
  std::lock_guard<std::mutex> lock(allocation.consumer_mutex);
  if (std::none_of(allocation.consumer_event_states.begin(),
                   allocation.consumer_event_states.end(), [](auto state) {
                     return state == internal::ConsumerEventSlotState::InUse;
                   }))
    allocation.consumer_work_completed.store(true, std::memory_order_release);
}

DispatchWaveKernelView
ActivationTransportBridge::wave_view(DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_ ||
      consumer.impl_->state != DeviceDispatchConsumerState::Acquired)
    invalid("phase=activation-wave-view, field=consumer, rank=n/a, "
            "expected=Acquired, actual=invalid");
  const auto &owner = consumer.impl_->owner;
  if (!owner.impl_ || !consumer.impl_->workspace ||
      consumer.impl_->workspace->epoch != consumer.impl_->epoch)
    invalid("phase=activation-wave-view, field=owner/epoch, rank=n/a, "
            "expected=live matching dispatch owner, actual=invalid");
  const auto &workspace = *consumer.impl_->workspace;
  const auto &dl = workspace.device_layout;
  const auto *base = workspace.allocation->device_ptr;
  DispatchWaveKernelView view{};
  view.wave_count = owner.impl_->dispatch_wave_count;
  view.local_route_order = at<const std::uint64_t>(base, dl.local_route_order);
  view.packet_order = at<const std::uint64_t>(base, dl.indices_out);
  view.offsets = at<const std::uint64_t>(base, dl.dispatch_wave_offsets);
  view.destination_node =
      at<const std::uint32_t>(base, dl.dispatch_wave_destination_node);
  view.channel = at<const std::uint32_t>(base, dl.dispatch_wave_channel);
  view.first_primary_tile =
      at<const std::uint64_t>(base, dl.dispatch_wave_first_primary_tile);
  view.last_primary_tile =
      at<const std::uint64_t>(base, dl.dispatch_wave_last_primary_tile);
  view.packet_count =
      at<const std::uint64_t>(base, dl.dispatch_wave_packet_count);
  view.descriptor_bytes =
      at<const std::uint64_t>(base, dl.dispatch_wave_payload_bytes);
  view.route_dst_mtile = at<const std::uint32_t>(base, dl.route_dst_mtile);
  return view;
}

bool ActivationTransportBridge::fused_inference_routes(
    const DeviceDispatchHandle &dispatch) {
  return dispatch.impl_ && dispatch.impl_->fused_forward_notify &&
         !dispatch.impl_->workspace->training_counts;
}

bool ActivationTransportBridge::deferred_training_notify(
    const DeviceDispatchHandle &dispatch) {
  return dispatch.impl_->fused_forward_notify &&
         dispatch.impl_->workspace->training_counts;
}

void ActivationTransportBridge::finish_training_notify(
    const DeviceDispatchHandle &dispatch,
    DeviceDispatchConsumerLease &consumer) {
  // A deferred handle has not published immutable training metadata yet.
  auto &impl = const_cast<DeviceDispatchHandle::Impl &>(*dispatch.impl_);
  const auto &state = *impl.workspace;
  const auto exec = internal::checked_execution_state(impl.execution);
  DeviceDispatchRuntime runtime{};
  internal::throw_cuda(
      "split-pf-notify",
      cudaMemcpyAsync(
          &runtime,
          at<void>(state.allocation->device_ptr, state.device_layout.runtime),
          sizeof(runtime), cudaMemcpyDeviceToHost, exec->stream),
      impl.identity.calling_rank, impl.identity.world_size,
      impl.context.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts, "download planned tape extent");
  internal::throw_cuda("split-pf-notify", cudaStreamSynchronize(exec->stream),
                       impl.identity.calling_rank, impl.identity.world_size,
                       impl.context.session().identity(),
                       impl.identity.generation, ExchangeKind::ExpertCounts,
                       "finish standalone PF planning");
  if (runtime.error)
    throw std::runtime_error("standalone PF notify rejected routes");
  impl.sizes = runtime.sizes;
  impl.padded_compute_rows = runtime.sizes.mtile_count * 128U;
  impl.fused_forward_notify = false;
  consumer.impl_->sizes = impl.sizes;
}

TrainingCountView ActivationTransportBridge::training_counts(
    const DeviceDispatchHandle &dispatch) {
  const auto &state = *dispatch.impl_->workspace;
  if (!state.training_counts)
    return {};
  auto *base = state.allocation->device_ptr;
  return {at<const std::uint64_t>(base, state.device_layout.chunk_counts),
          at<const std::uint64_t>(base, state.device_layout.expert_base),
          at<DeviceDispatchRuntime>(base, state.device_layout.runtime)};
}

std::uint64_t ActivationTransportBridge::padded_received_rows(
    const DeviceDispatchHandle &dispatch) {
  if (!dispatch.impl_ || dispatch.impl_->fused_forward_notify ||
      dispatch.impl_->workspace->replay_counts)
    invalid("training row allocation requires immutable materialized Notify");
  return dispatch.impl_->padded_compute_rows;
}

const std::int32_t *
ActivationTransportBridge::route_pointer(const DeviceDispatchHandle &dispatch) {
  if (!dispatch.impl_ || !dispatch.impl_->source_routes)
    invalid("phase=activation-route-view, field=dispatch/routes, rank=n/a, "
            "expected=valid owning handle, actual=empty");
  DeviceRouteProvenanceBridge::validate(dispatch.impl_->source_routes,
                                        dispatch.impl_->context,
                                        dispatch.impl_->execution);
  return DeviceRouteProvenanceBridge::device_pointer(
      dispatch.impl_->source_routes);
}

const float *
ActivationTransportBridge::gate_pointer(const DeviceDispatchHandle &dispatch) {
  if (!dispatch.impl_ || !dispatch.impl_->source_gates)
    invalid("phase=activation-gate-view, field=dispatch/gates, rank=n/a, "
            "expected=valid owning handle, actual=empty");
  DeviceGateProvenanceBridge::validate(
      dispatch.impl_->source_gates, dispatch.impl_->source_routes,
      dispatch.impl_->context, dispatch.impl_->execution);
  return DeviceGateProvenanceBridge::device_pointer(
      dispatch.impl_->source_gates);
}

void ActivationTransportBridge::wait_for_gates(
    const DeviceDispatchHandle &dispatch, const CudaExecution &execution) {
  if (!dispatch.impl_ || !dispatch.impl_->source_gates.impl_)
    invalid("phase=activation-gate-wait, field=dispatch/gates, rank=n/a, "
            "expected=valid owning handle, actual=empty");
  const auto &gate = *dispatch.impl_->source_gates.impl_;
  const auto exec = internal::checked_execution_state(execution);
  if (gate.device != exec->device || !gate.ready_recorded || !gate.allocation ||
      !gate.bound_routes.impl_ || !gate.bound_routes.impl_->allocation ||
      !gate.bound_routes.impl_->allocation->producer_ready)
    invalid("phase=activation-gate-wait, field=device/ready, rank=" +
            std::to_string(dispatch.impl_->identity.calling_rank) +
            ", expected=matching recorded producer event, actual=mismatch");
  if (gate.producer_execution.same_instance(execution.token()) ||
      produced_on_stream(gate.bound_routes.impl_->allocation, exec->stream))
    return;
  internal::throw_cuda(
      "activation-gate-wait",
      internal::wait_input_ready(
          exec->stream, gate.bound_routes.impl_->allocation->producer_ready,
          gate.bound_routes.impl_->allocation->producer_capture_id),
      dispatch.impl_->identity.calling_rank,
      dispatch.impl_->identity.world_size,
      dispatch.impl_->context.session().identity(),
      dispatch.impl_->identity.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(gate producer ready)");
}

void ActivationTransportBridge::submit(DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_)
    invalid("phase=activation-transport-submit, field=consumer, rank=n/a, "
            "expected=valid, actual=empty");
  consumer.impl_->record_completion();
}

const RankCollectiveContext &
ActivationTransportBridge::context(const DeviceDispatchHandle &dispatch) {
  if (!dispatch.impl_)
    invalid("activation context requires a valid Notify handle");
  return dispatch.impl_->context;
}

const RankCollectiveContext &ActivationTransportBridge::context(
    const DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_)
    invalid("phase=activation-transport-context, field=consumer, rank=n/a, "
            "expected=valid, actual=empty");
  return consumer.impl_->context;
}

class DeviceDispatchTestBridge {
public:
  static void submit(DeviceDispatchConsumerLease &consumer,
                     std::uint64_t delay_cycles) {
    if (!consumer.impl_ ||
        consumer.impl_->state != DeviceDispatchConsumerState::Acquired)
      invalid("phase=dispatch-test-submit, field=consumer_state, rank=n/a, "
              "expected=Acquired, actual=invalid");
    auto &impl = *consumer.impl_;
    const auto view = impl.kernel_view();
    if (view.sizes.route_count == 0)
      invalid("phase=dispatch-test-submit, field=route_count, rank=" +
              std::to_string(view.identity.calling_rank) +
              ", expected>=1, actual=0");
    auto exec = internal::checked_execution_state(impl.execution);
    auto workspace = impl.workspace;
    auto *destination = at<void>(workspace->allocation->host_ptr,
                                 workspace->host_layout.materialization_status +
                                     impl.event_index * sizeof(std::uint64_t));
    if (workspace->host_layout.materialization_status +
            (impl.event_index + 1U) * sizeof(std::uint64_t) >
        workspace->host_layout.total)
      invalid("phase=dispatch-test-submit, field=probe_staging, rank=" +
              std::to_string(view.identity.calling_rank) +
              ", expected=pooled consumer staging, actual=short");
    try {
      dispatch_consumer_probe_delay_kernel<<<1, 1, 0, exec->stream>>>(
          view.route_dst_row, delay_cycles);
      internal::throw_cuda("dispatch-test-submit", cudaGetLastError(),
                           view.identity.calling_rank, view.identity.world_size,
                           impl.context.session().identity(),
                           view.identity.generation, ExchangeKind::ExpertCounts,
                           "consumer probe kernel launch");
      internal::throw_cuda(
          "dispatch-test-submit",
          cudaMemcpyAsync(destination, view.route_dst_row,
                          sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                          exec->stream),
          view.identity.calling_rank, view.identity.world_size,
          impl.context.session().identity(), view.identity.generation,
          ExchangeKind::ExpertCounts, "D2H consumer probe");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(std::uint64_t));
      impl.record_completion();
    } catch (...) {
      if (impl.owner) {
        const bool stream_ok =
            cudaStreamSynchronize(exec->stream) == cudaSuccess;
        if (!stream_ok) {
          std::lock_guard<std::mutex> lock(
              workspace->allocation->consumer_mutex);
          workspace->allocation->dispatch_slot_quarantined.store(
              true, std::memory_order_release);
        }
        impl.release_owner_noexcept(false);
      }
      throw;
    }
  }

  static std::uint64_t finish(DeviceDispatchConsumerLease &consumer) {
    if (!consumer.impl_ ||
        consumer.impl_->state != DeviceDispatchConsumerState::Submitted)
      invalid("phase=dispatch-test-finish, field=consumer_state, rank=n/a, "
              "expected=Submitted, actual=invalid");
    auto workspace = consumer.impl_->workspace;
    consumer.finish();
    return *at<const std::uint64_t>(
        workspace->allocation->host_ptr,
        workspace->host_layout.materialization_status +
            consumer.impl_->event_index * sizeof(std::uint64_t));
  }
};

namespace testing {
void submit_dispatch_probe(DeviceDispatchConsumerLease &consumer,
                           std::uint64_t delay_cycles) {
  DeviceDispatchTestBridge::submit(consumer, delay_cycles);
}
std::uint64_t finish_dispatch_probe(DeviceDispatchConsumerLease &consumer) {
  return DeviceDispatchTestBridge::finish(consumer);
}
} // namespace testing

DispatchHandle
download_device_metadata_for_validation(const DeviceDispatchHandle &handle,
                                        const CudaExecution &execution) {
  if (!handle || !execution)
    invalid("phase=device-validation-download, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &impl = *handle.impl_;
  const auto expectation =
      DeviceDispatchViewExpectation{impl.identity.generation,
                                    impl.identity.calling_rank,
                                    impl.identity.world_size,
                                    impl.identity.num_tokens,
                                    impl.identity.top_k,
                                    impl.identity.config_digest_word0,
                                    impl.identity.config_digest_word1};
  auto consumer = handle.acquire_consumer(execution, expectation);
  const auto view = consumer.impl_->kernel_view();
  auto exec = internal::checked_execution_state(execution);
  const auto state = impl.workspace;
  auto summary_state = internal::CudaDeviceLeaseAccess::checked(
      impl.device.packet_summary_count, exec->device);
  const auto routes = impl.sizes.route_count;
  SourceRouteMetadata outgoing;
  outgoing.num_tokens = impl.identity.num_tokens;
  outgoing.top_k = impl.identity.top_k;
  outgoing.node_mask_words = impl.sizes.node_mask_words;
  outgoing.expert_id = download_vector<std::uint32_t>(
      state, *exec, view.expert_id, routes, "expert_id");
  outgoing.local_offset = download_vector<std::uint64_t>(
      state, *exec, view.local_offset, routes, "local_offset");
  outgoing.route_dst_row = download_vector<std::uint64_t>(
      state, *exec, view.route_dst_row, routes, "route_dst_row");
  outgoing.route_dst_rank = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_rank, routes, "route_dst_rank");
  outgoing.route_dst_node = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_node, routes, "route_dst_node");
  outgoing.route_dst_local_rank = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_local_rank, routes, "route_dst_local_rank");
  outgoing.combine_order = download_vector<std::uint32_t>(
      state, *exec, view.combine_order, routes, "combine_order");
  outgoing.destination_node_mask = download_vector<std::uint64_t>(
      state, *exec, view.destination_node_mask,
      impl.identity.num_tokens * impl.sizes.node_mask_words,
      "destination_node_mask");
  const auto reverse = download_vector<DeviceReverseRouteRecord>(
      state, *exec, view.reverse_routes, routes, "reverse_routes");
  outgoing.reverse_routes.reserve(reverse.size());
  for (const auto &record : reverse)
    outgoing.reverse_routes.push_back(ReverseRouteMetadata{
        record.source_rank, record.local_token_id, record.topk_slot,
        record.expert_id, record.destination_rank, record.dst_row});

  const auto flat_counts = download_vector<std::uint64_t>(
      state, *exec, view.source_expert_count,
      static_cast<std::uint64_t>(impl.identity.world_size) *
          impl.identity.num_experts,
      "source_expert_count");
  const auto flat_summary = download_vector<std::uint64_t>(
      state, *exec, summary_state->pointer,
      static_cast<std::uint64_t>(impl.identity.world_size) *
          impl.identity.num_nodes,
      "packet_summary_count");
  CountMatrix counts(impl.identity.world_size,
                     std::vector<std::uint64_t>(impl.identity.num_experts));
  CountMatrix summaries(impl.identity.world_size,
                        std::vector<std::uint64_t>(impl.identity.num_nodes));
  for (std::uint32_t rank = 0; rank < impl.identity.world_size; ++rank) {
    std::copy_n(flat_counts.begin() +
                    static_cast<std::size_t>(rank) * impl.identity.num_experts,
                impl.identity.num_experts, counts[rank].begin());
    std::copy_n(flat_summary.begin() +
                    static_cast<std::size_t>(rank) * impl.identity.num_nodes,
                impl.identity.num_nodes, summaries[rank].begin());
  }

  DispatchMetadata metadata;
  metadata.config = impl.config;
  metadata.calling_rank = impl.identity.calling_rank;
  metadata.communicator_id = impl.context.session().identity();
  metadata.identity.generation = impl.identity.generation;
  metadata.source_expert_count = counts;
  metadata.outgoing = std::move(outgoing);
  auto &incoming = metadata.incoming;
  incoming.rank = impl.identity.calling_rank;
  incoming.expert_ids = download_vector<std::uint32_t>(
      state, *exec, view.owned_expert_ids, impl.sizes.owned_expert_count,
      "owned_expert_ids");
  incoming.expert_row_count = download_vector<std::uint64_t>(
      state, *exec, view.expert_row_count, impl.sizes.owned_expert_count,
      "expert_row_count");
  incoming.expert_row_prefix = download_vector<std::uint64_t>(
      state, *exec, view.expert_row_prefix,
      static_cast<std::uint64_t>(impl.sizes.owned_expert_count) + 1,
      "expert_row_prefix");
  const auto tile_offsets = download_vector<std::uint64_t>(
      state, *exec, view.mtile_row_offsets,
      static_cast<std::uint64_t>(impl.sizes.owned_expert_count) + 1,
      "mtile_row_offsets");
  const auto tiles = download_vector<std::uint32_t>(
      state, *exec, view.mtile_expected_rows, impl.sizes.mtile_count,
      "mtile_expected_rows");
  incoming.mtile_expected_rows.resize(impl.sizes.owned_expert_count);
  for (std::uint32_t expert = 0; expert < impl.sizes.owned_expert_count;
       ++expert) {
    incoming.mtile_expected_rows[expert].assign(
        tiles.begin() + static_cast<std::ptrdiff_t>(tile_offsets[expert]),
        tiles.begin() + static_cast<std::ptrdiff_t>(tile_offsets[expert + 1]));
  }
  incoming.incoming_rows.reserve(impl.sizes.incoming_row_count);
  for (std::size_t owned = 0; owned < incoming.expert_ids.size(); ++owned) {
    const auto expert = incoming.expert_ids[owned];
    auto row = incoming.expert_row_prefix[owned];
    for (unsigned source = 0; source < impl.identity.world_size; ++source)
      for (std::uint64_t offset = 0; offset < counts[source][expert]; ++offset)
        incoming.incoming_rows.push_back(
            IncomingRowMetadata{source, expert, offset, row++});
  }

  const auto headers = download_vector<DevicePacketHeaderRecord>(
      state, *exec, view.packet_headers, impl.sizes.packet_count,
      "packet_headers");
  const auto fanout_offsets = download_vector<std::uint64_t>(
      state, *exec, view.packet_fanout_offsets, impl.sizes.packet_count + 1,
      "packet_fanout_offsets");
  const auto fanout = download_vector<DeviceFanoutRecord>(
      state, *exec, view.packet_fanout, impl.sizes.fanout_count,
      "packet_fanout");
  for (std::uint64_t packet = 0; packet < impl.sizes.packet_count; ++packet) {
    const auto &header = headers[static_cast<std::size_t>(packet)];
    OutgoingPacketDescriptor descriptor{PacketIdentity{header.source_rank,
                                                       header.destination_node,
                                                       header.sequence_on_link},
                                        header.local_token_id,
                                        header.proxy_rank,
                                        header.channel,
                                        {}};
    for (auto index = fanout_offsets[static_cast<std::size_t>(packet)];
         index < fanout_offsets[static_cast<std::size_t>(packet + 1)];
         ++index) {
      const auto &target = fanout[static_cast<std::size_t>(index)];
      descriptor.fanout.push_back(
          FanoutTarget{target.topk_slot, target.destination_local_rank,
                       target.expert_id, target.dst_row});
    }
    metadata.transport.outgoing_packets.push_back(std::move(descriptor));
  }
  metadata.transport.dispatch_wave_packet_order =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(state->allocation->device_ptr,
                                  state->device_layout.indices_out),
          impl.sizes.packet_count, "dispatch_wave_packet_order");
  metadata.transport.dispatch_wave_offsets = download_vector<std::uint64_t>(
      state, *exec,
      at<const std::uint64_t>(state->allocation->device_ptr,
                              state->device_layout.dispatch_wave_offsets),
      impl.dispatch_wave_count + 1, "dispatch_wave_offsets");
  metadata.transport.dispatch_wave_destination_node =
      download_vector<std::uint32_t>(
          state, *exec,
          at<const std::uint32_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_destination_node),
          impl.dispatch_wave_count, "dispatch_wave_destination_node");
  metadata.transport.dispatch_wave_channel = download_vector<std::uint32_t>(
      state, *exec,
      at<const std::uint32_t>(state->allocation->device_ptr,
                              state->device_layout.dispatch_wave_channel),
      impl.dispatch_wave_count, "dispatch_wave_channel");
  metadata.transport.dispatch_wave_first_primary_tile =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_first_primary_tile),
          impl.dispatch_wave_count, "dispatch_wave_first_primary_tile");
  metadata.transport.dispatch_wave_last_primary_tile =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_last_primary_tile),
          impl.dispatch_wave_count, "dispatch_wave_last_primary_tile");
  metadata.transport.dispatch_wave_packet_count =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_packet_count),
          impl.dispatch_wave_count, "dispatch_wave_packet_count");
  metadata.transport.dispatch_wave_payload_bytes =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_payload_bytes),
          impl.dispatch_wave_count, "dispatch_wave_payload_bytes");
  const auto outgoing_signals = download_vector<DeviceSignalRecord>(
      state, *exec, view.outgoing_signals, impl.sizes.outgoing_signal_count,
      "outgoing_signals");
  const auto incoming_signals = download_vector<DeviceSignalRecord>(
      state, *exec, view.incoming_proxy_signals,
      impl.sizes.incoming_signal_count, "incoming_signals");
  const auto append_signals = [](const auto &source, auto &destination) {
    for (const auto &signal : source)
      destination.push_back(SignalCardinality{
          signal.source_node, signal.destination_node, signal.source_rank,
          signal.destination_proxy_rank, signal.rail_local_rank, signal.channel,
          signal.expected_packets, signal.expected_signals});
  };
  append_signals(outgoing_signals, metadata.transport.outgoing_signals);
  append_signals(incoming_signals, metadata.transport.incoming_proxy_signals);
  auto result =
      DeviceNotifyBridge::make_validation_dispatch(std::move(metadata));
  validate_dispatch_metadata(result);
  consumer.impl_->record_completion();
  consumer.finish();
  return result;
}

void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &handle,
                                     const CudaExecution &execution) {
  (void)download_device_metadata_for_validation(handle, execution);
}
