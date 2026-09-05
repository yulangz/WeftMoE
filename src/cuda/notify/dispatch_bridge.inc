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
              ", expected=synchronized, actual=CUDA event failure; event quarantined");
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
  auto exec = internal::checked_execution_state(consumer.impl_->execution);
  const auto checked = [&](const DeviceBufferLease &lease,
                           const char *field) -> const void * {
    const auto storage = internal::CudaDeviceLeaseAccess::checked(
        lease, exec->device);
    if (!storage || !storage->pointer)
      invalid(std::string("phase=activation-wave-view, field=") + field +
              ", rank=" +
              std::to_string(consumer.impl_->identity.calling_rank) +
              ", expected=live device storage, actual=empty");
    return storage->pointer;
  };
  const auto checked_optional = [&](const DeviceBufferLease &lease,
                                    std::uint64_t count,
                                    const char *field) -> const void * {
    return count == 0U ? nullptr : checked(lease, field);
  };
  const auto &device = owner.impl_->device;
  const auto packet_count = owner.impl_->sizes.packet_count;
  const auto route_count = owner.impl_->sizes.route_count;
  const auto wave_count = owner.impl_->dispatch_wave_count;
  DispatchWaveKernelView view{};
  view.wave_count = wave_count;
  view.local_route_order = static_cast<const std::uint64_t *>(
      checked_optional(device.local_route_order, route_count,
                       "local_route_order"));
  view.packet_order = static_cast<const std::uint64_t *>(
      checked_optional(device.dispatch_wave_packet_order, packet_count,
                       "packet_order"));
  view.offsets = static_cast<const std::uint64_t *>(
      checked(device.dispatch_wave_offsets, "offsets"));
  view.destination_node = static_cast<const std::uint32_t *>(
      checked_optional(device.dispatch_wave_destination_node, wave_count,
                       "destination_node"));
  view.channel = static_cast<const std::uint32_t *>(
      checked_optional(device.dispatch_wave_channel, wave_count, "channel"));
  view.first_primary_tile = static_cast<const std::uint64_t *>(
      checked_optional(device.dispatch_wave_first_primary_tile, wave_count,
                       "first_primary_tile"));
  view.last_primary_tile = static_cast<const std::uint64_t *>(
      checked_optional(device.dispatch_wave_last_primary_tile, wave_count,
                       "last_primary_tile"));
  view.packet_count = static_cast<const std::uint64_t *>(
      checked_optional(device.dispatch_wave_packet_count, wave_count,
                       "packet_count"));
  view.descriptor_bytes = static_cast<const std::uint64_t *>(
      checked_optional(device.dispatch_wave_payload_bytes, wave_count,
                       "descriptor_bytes"));
  view.route_dst_mtile = static_cast<const std::uint32_t *>(
      checked_optional(device.route_dst_mtile, route_count,
                       "route_dst_mtile"));
  return view;
}

const float *ActivationTransportBridge::gate_pointer(
    const DeviceDispatchHandle &dispatch) {
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
  if (gate.device != exec->device || !gate.ready_recorded ||
      !gate.allocation || !gate.bound_routes.impl_ ||
      !gate.bound_routes.impl_->allocation ||
      !gate.bound_routes.impl_->allocation->producer_ready)
    invalid("phase=activation-gate-wait, field=device/ready, rank=" +
            std::to_string(dispatch.impl_->identity.calling_rank) +
            ", expected=matching recorded producer event, actual=mismatch");
  internal::throw_cuda(
      "activation-gate-wait",
      cudaStreamWaitEvent(exec->stream,
                          gate.bound_routes.impl_->allocation->producer_ready),
      dispatch.impl_->identity.calling_rank,
      dispatch.impl_->identity.world_size,
      dispatch.impl_->context.session().identity(),
      dispatch.impl_->identity.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(gate producer ready)");
}

void ActivationTransportBridge::submit(
    DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_)
    invalid("phase=activation-transport-submit, field=consumer, rank=n/a, "
            "expected=valid, actual=empty");
  consumer.impl_->record_completion();
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
      internal::throw_cuda(
          "dispatch-test-submit", cudaGetLastError(),
          view.identity.calling_rank, view.identity.world_size,
          impl.context.session().identity(), view.identity.generation,
          ExchangeKind::ExpertCounts, "consumer probe kernel launch");
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
        const bool stream_ok = cudaStreamSynchronize(exec->stream) == cudaSuccess;
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

DispatchHandle download_device_metadata_for_validation(
    const DeviceDispatchHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    invalid("phase=device-validation-download, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &impl = *handle.impl_;
  const auto expectation = DeviceDispatchViewExpectation{
      impl.identity.generation, impl.identity.calling_rank,
      impl.identity.world_size, impl.identity.num_tokens, impl.identity.top_k,
      impl.identity.config_digest_word0, impl.identity.config_digest_word1};
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
      state, *exec, view.route_dst_local_rank, routes,
      "route_dst_local_rank");
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
  const auto rows = download_vector<DeviceIncomingRowRecord>(
      state, *exec, view.incoming_rows, impl.sizes.incoming_row_count,
      "incoming_rows");
  incoming.incoming_rows.reserve(rows.size());
  for (const auto &row : rows)
    incoming.incoming_rows.push_back(IncomingRowMetadata{
        row.source_rank, row.expert_id, row.source_expert_offset, row.dst_row});

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
    OutgoingPacketDescriptor descriptor{
        PacketIdentity{header.source_rank, header.destination_node,
                       header.sequence_on_link},
        header.local_token_id, header.proxy_rank, header.channel, {}};
    for (auto index = fanout_offsets[static_cast<std::size_t>(packet)];
         index < fanout_offsets[static_cast<std::size_t>(packet + 1)]; ++index) {
      const auto &target = fanout[static_cast<std::size_t>(index)];
      descriptor.fanout.push_back(FanoutTarget{
          target.topk_slot, target.destination_local_rank, target.expert_id,
          target.dst_row});
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
          impl.dispatch_wave_count,
          "dispatch_wave_destination_node");
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
          impl.dispatch_wave_count,
          "dispatch_wave_first_primary_tile");
  metadata.transport.dispatch_wave_last_primary_tile =
      download_vector<std::uint64_t>(
          state, *exec,
          at<const std::uint64_t>(
              state->allocation->device_ptr,
              state->device_layout.dispatch_wave_last_primary_tile),
          impl.dispatch_wave_count,
          "dispatch_wave_last_primary_tile");
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
          impl.dispatch_wave_count,
          "dispatch_wave_payload_bytes");
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
          signal.destination_proxy_rank, signal.rail_local_rank,
          signal.channel, signal.expected_packets,
          signal.expected_signals});
  };
  append_signals(outgoing_signals, metadata.transport.outgoing_signals);
  append_signals(incoming_signals,
                 metadata.transport.incoming_proxy_signals);
  auto result = DeviceNotifyBridge::make_validation_dispatch(
      std::move(metadata));
  validate_dispatch_metadata(result);
  consumer.impl_->record_completion();
  consumer.finish();
  return result;
}

void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &handle,
                                      const CudaExecution &execution) {
  (void)download_device_metadata_for_validation(handle, execution);
}

