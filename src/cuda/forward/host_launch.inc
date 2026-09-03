DeviceActivationTransport::run_device_simulation_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  const auto &id = dispatch.identity();
  auto state = impl_->pool.impl_->state;
  const auto &activation = *activations.impl_;
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens || activation.hidden != config.hidden_size ||
      activation.device != impl_->execution.device_ordinal() ||
      activation.session_identity != impl_->session.stable_identity_bytes() ||
      !activation.producer_execution.same_instance(impl_->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size ||
      id.num_nodes > state->config.max_nodes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "transport-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "transport-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");
  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(impl_->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(impl_->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  const auto links = checked_mul(id.num_nodes, config.channels, "actual links");
  const auto ring_slots = checked_mul(
      checked_mul(id.num_nodes, config.channels, "actual ring links"),
      config.ring_slots_per_source_channel, "actual ring slots");
  const auto route_elements = checked_mul(view.sizes.route_count,
                                          config.hidden_size, "route elements");
  const auto combined_elements = checked_mul(id.num_tokens, config.hidden_size,
                                             "combined elements");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
  };
  internal::throw_cuda(
      "transport-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, ring_slots,
             sizeof(transport::TransportSlotHeader), "ring headers");
  zero_async(layout.ring_ready_sequence, ring_slots, sizeof(std::uint64_t),
             "ring sequences");
  zero_async(layout.link_next_sequence, links, sizeof(std::uint64_t),
             "link sequences");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits), links,
                      config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "transport-run", cudaGetLastError(), id.calling_rank, id.world_size,
      impl_->session.identity(), id.generation, ExchangeKind::ExpertCounts,
      "initialize transport credits kernel");
  zero_async(layout.packet_claim, view.sizes.packet_count,
             sizeof(std::uint32_t), "packet claims");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "canonical elements"),
             sizeof(transport::BFloat16), "canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "status");
  internal::throw_cuda("transport-run", cudaEventRecord(slot.start, exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(start)");
  int sms = 0;
  internal::throw_cuda(
      "transport-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  TransportKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, impl_->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
  args.ring_headers = at<transport::TransportSlotHeader>(slot.device,
                                                          layout.ring_headers);
  args.ring_payload = at<transport::BFloat16>(slot.device, layout.ring_payload);
  args.ring_ready = at<unsigned long long>(slot.device,
                                            layout.ring_ready_sequence);
  args.link_next = at<unsigned long long>(slot.device,
                                           layout.link_next_sequence);
  args.link_credits = at<unsigned int>(slot.device, layout.link_credits);
  args.packet_claim = at<unsigned int>(slot.device, layout.packet_claim);
  args.canonical_rows = at<transport::BFloat16>(slot.device,
                                                 layout.canonical_rows);
  args.contribution = at<transport::BFloat16>(slot.device, layout.contribution);
  args.combine_return = at<transport::BFloat16>(slot.device,
                                                 layout.combine_return);
  args.combined = at<transport::BFloat16>(slot.device, layout.combined);
  args.status = at<TransportDeviceStatus>(slot.device, layout.status);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  activation_transport_persistent_kernel<<<std::max(sms, 1), 128, 0,
                                           exec->stream>>>(args);
  internal::throw_cuda("transport-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "activation transport persistent kernel launch");
  internal::throw_cuda(
      "transport-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(dispatch complete)");
  internal::throw_cuda(
      "transport-run",
      cudaMemcpyAsync(slot.host_status, args.status,
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda("transport-run", cudaEventRecord(slot.complete,
                                                         exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda("transport-run", cudaEventSynchronize(slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventSynchronize(transport complete)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "transport-kernel", "descriptor", id.calling_rank,
                    "valid packet/fanout", "device validation error packet=" +
                        std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda("transport-timing",
                       cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventElapsedTime(total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  // The concrete backend object, never a caller-provided enum, owns this
  // label. This path is deliberately and permanently simulation-only.
  result->audit.backend =
      transport::ActivationTransportBackend::DeviceSimulation;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    result->timings.combine_reduce_us =
        static_cast<double>(complete_clock - dispatch_clock) / 1000.0;
  }
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle LocalCudaP2pActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node and exact local NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "local-p2p-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "local-p2p-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  [[maybe_unused]] const auto route_elements =
      checked_mul(view.sizes.route_count, config.hidden_size,
                  "P2P route elements");
  [[maybe_unused]] const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size,
                  "P2P combined elements");
  [[maybe_unused]] const auto rank_links =
      checked_mul(id.world_size, config.channels, "P2P rank links");
  [[maybe_unused]] const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "P2P ingress slots");
  [[maybe_unused]] const auto node_links =
      static_cast<std::uint64_t>(config.channels);
  [[maybe_unused]] const auto node_ring_slots = checked_mul(
      node_links, config.ring_slots_per_source_channel, "P2P proxy slots");
  std::uint64_t generation_memset_calls = 0;
  std::uint64_t generation_memset_bytes = 0;
  std::uint64_t kernel_launches = 0;
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    const auto bytes = checked_size(checked_mul(count, width, name), name);
    internal::throw_cuda(
        "local-p2p-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0, bytes, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
    ++generation_memset_calls;
    generation_memset_bytes += bytes;
  };
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "local-p2p-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  // The integrated timing starts after the router-owned input is ready and
  // includes every generation-specific transport initialization below.
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P start)");
#if MEGAMOE_DEBUG_VALIDATE
  zero_async(layout.ring_headers, node_ring_slots,
             sizeof(transport::TransportSlotHeader), "P2P proxy headers");
  zero_async(layout.ring_payload,
             checked_mul(node_ring_slots, config.hidden_size,
                         "P2P proxy payload"),
             sizeof(transport::BFloat16), "P2P proxy payload");
  zero_async(layout.ring_ready_sequence, node_ring_slots,
             sizeof(std::uint64_t), "P2P proxy ready");
  zero_async(layout.link_next_sequence, node_links, sizeof(std::uint64_t),
             "P2P proxy sequence");
  zero_async(layout.ingress_headers, ingress_slots,
             sizeof(LocalP2pRouteHeader), "P2P ingress headers");
  zero_async(layout.ingress_ready_sequence, ingress_slots,
             sizeof(std::uint64_t), "P2P ingress ready");
  zero_async(layout.ingress_next_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress next");
  zero_async(layout.ingress_published_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress published");
  zero_async(layout.outgoing_route_sequence, rank_links,
             sizeof(std::uint64_t), "P2P outgoing sequence");
  zero_async(layout.direct_route_headers,
             state->config.max_destination_rows_per_rank,
             sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  zero_async(layout.return_ready_epoch, view.sizes.route_count,
             sizeof(std::uint64_t), "P2P return ready");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "P2P canonical rows"),
             sizeof(transport::BFloat16), "P2P canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "P2P contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "P2P combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "P2P combined");
#else
  // The trusted single-node bulk path overwrites every canonical row,
  // contribution and output element that it later reads. Epoch-valued peer
  // signals make all arena and signal clears unnecessary. Clearing a peer
  // signal here would race a faster rank's release store.
#endif
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "P2P status");
#if MEGAMOE_DEBUG_VALIDATE
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((node_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits),
                      node_links, config.ring_slots_per_source_channel);
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((rank_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.ingress_credits),
                      rank_links, config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P credits");

  // A fixed-width device AllGather is a setup barrier only. It exchanges no
  // activation and ensures every peer ring/credit/sequence region is fully
  // initialized before any persistent worker publishes to it.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, activation.slot->epoch);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
  validate_p2p_barrier_kernel<<<1, 256, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive),
      id.world_size, activation.slot->index, activation.slot->epoch,
      at<TransportDeviceStatus>(slot.device, layout.status));
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "validate P2P setup barrier");
#else
  // This small collective is a real per-generation dependency: peer writes
  // must not begin until every process has reached the transport phase. It
  // carries only 16 bytes/rank and replaces the old multi-gigabyte clears.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, id.generation);
  ++kernel_launches;
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
#endif
  int sms = 0;
  internal::throw_cuda(
      "local-p2p-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  if (sms <= static_cast<int>(id.world_size))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=resident_workers, rank=" +
            std::to_string(id.calling_rank) + ", expected>world=" +
            std::to_string(id.world_size) + ", actual_sms=" +
            std::to_string(sms));
  LocalP2pKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
#if MEGAMOE_DEBUG_VALIDATE
  args.epoch = activation.slot->epoch;
#else
  // Every rank invokes the trusted team in the same generation order. A
  // monotonically increasing per-slot epoch supports both new generations and
  // repeated same-generation correctness runs without clearing peer signals.
  if (slot.transport_epoch == std::numeric_limits<std::uint64_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "local-p2p-run",
                    "transport_epoch", id.calling_rank, "<UINT64_MAX",
                    "UINT64_MAX");
  args.epoch = ++slot.transport_epoch;
#endif
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.local_base = slot.device;
  args.ring_headers_offset = layout.ring_headers;
  args.ring_payload_offset = layout.ring_payload;
  args.ring_ready_offset = layout.ring_ready_sequence;
  args.link_next_offset = layout.link_next_sequence;
  args.link_credits_offset = layout.link_credits;
  args.ingress_headers_offset = layout.ingress_headers;
  args.ingress_ready_offset = layout.ingress_ready_sequence;
  args.ingress_next_offset = layout.ingress_next_sequence;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.ingress_credits_offset = layout.ingress_credits;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.direct_headers_offset = layout.direct_route_headers;
  args.return_ready_offset = layout.return_ready_epoch;
  args.canonical_rows_offset = layout.canonical_rows;
  args.contribution_offset = layout.contribution;
  args.combine_return_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  internal::throw_cuda(
      "local-p2p-run",
      cudaEventRecord(slot.dispatch_complete, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P generation setup complete)");
#if MEGAMOE_DEBUG_VALIDATE
  local_p2p_persistent_kernel<<<sms, 128, 0, exec->stream>>>(args);
  ++kernel_launches;
#else
  local_p2p_bulk_dispatch_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.route_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_return_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.incoming_row_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_reduce_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(id.num_tokens, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_finish_kernel<<<1, 1, 0, exec->stream>>>(args);
  ++kernel_launches;
#endif
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "local P2P persistent kernel launch");
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H P2P transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "local-p2p-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(P2P complete)");
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >=
            options.deadline_after;
    if (!cancelled && (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, id.generation,
                       __ATOMIC_RELEASE);
      internal::throw_cuda(
          "local-p2p-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish cancel epoch");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                sizeof(std::uint64_t));
      internal::throw_cuda(
          "local-p2p-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaStreamSynchronize(cancel)");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  // Debug/cancellation paths retain the explicit team drain. In the trusted
  // release path every source waits for a generation-valued return signal
  // from every destination before producing output. The next Notify control
  // collective is the natural inter-generation barrier.
#if MEGAMOE_DEBUG_VALIDATE
  internal::throw_nccl(
      "local-p2p-drain",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P terminal drain)");
  internal::throw_cuda(
      "local-p2p-drain", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(P2P terminal drain)");
#else
  if (cancelled) {
    internal::throw_nccl(
        "local-p2p-drain",
        ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                      at<void>(slot.device, layout.p2p_barrier_receive), 2,
                      ncclUint64, backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "ncclAllGather(P2P cancellation drain)");
    internal::throw_cuda(
        "local-p2p-drain", cudaStreamSynchronize(exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(P2P cancellation drain)");
  }
#endif
  consumer.finish();
  if (cancelled || slot.host_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=local-p2p-run, field=cancel_epoch, rank=" +
            std::to_string(id.calling_rank) + ", expected=completion, actual=" +
            (options.cancellation.cancellation_requested() ? "cancelled"
                                                           : "deadline") +
            ", packets=" + std::to_string(slot.host_status->packets_done) +
            ", incoming=" + std::to_string(slot.host_status->incoming_done) +
            ", combine=" + std::to_string(slot.host_status->combine_done));
  if (slot.host_status->error != 0)
    transport_error(
        NotifyErrorCode::TransportProtocolFailure, "local-p2p-kernel",
        "ring/sequence/descriptor", id.calling_rank, "valid",
        "device error field=" + std::to_string(slot.host_status->error_field) +
            ", packet=" + std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(P2P total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->local_p2p_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows =
      state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::LocalCudaP2p;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.peak_ring_slots = slot.host_status->peak_ring_slots;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = generation_memset_calls;
  result->audit.generation_memset_bytes = generation_memset_bytes;
  result->audit.kernel_launches = kernel_launches;
  result->audit.dispatch_kernel_blocks = view.sizes.route_count;
  result->audit.return_kernel_blocks = view.sizes.incoming_row_count;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  float setup_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&setup_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(P2P generation setup)");
  result->timings.generation_setup_us =
      static_cast<double>(setup_ms) * 1000.0;
  result->timings.dispatch_put_us =
      static_cast<double>(slot.host_status->dispatch_put_ticks) / 1000.0;
  result->timings.relay_us =
      static_cast<double>(slot.host_status->relay_ticks) / 1000.0;
  result->timings.synthetic_return_us =
      static_cast<double>(slot.host_status->synthetic_ticks) / 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    const auto post_dispatch = complete_clock - dispatch_clock;
    const auto relay_return = slot.host_status->synthetic_ticks;
    result->timings.combine_reduce_us = static_cast<double>(
        post_dispatch >= relay_return ? post_dispatch - relay_return : 0ULL) /
        1000.0;
  }
  const auto kernel_us = complete_clock >= start_clock
                             ? static_cast<double>(complete_clock - start_clock) /
                                   1000.0
                             : 0.0;
  const auto accounted = result->timings.generation_setup_us + kernel_us;
  result->timings.terminal_completion_us =
      std::max(0.0, result->timings.total_us - accounted);
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle
NcclSocketFallbackActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "socket-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (id.num_nodes < 2 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "topology", id.calling_rank,
                    "multi-node exact NCCL world/rank", "mismatch");
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "socket-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_socket_fallback ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "pool geometry", id.calling_rank, "within capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=socket-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  auto *local_counts = at<unsigned long long>(slot.device,
                                               layout.terminal_epoch);
  auto *count_matrix = at<unsigned long long>(slot.device,
                                               layout.socket_count_matrix);
  auto *prefix = at<unsigned long long>(slot.device,
                                        layout.p2p_barrier_receive);
  auto *route_ids = at<unsigned int>(slot.device, layout.packet_claim);
  auto *send_headers = at<SocketRouteHeader>(
      slot.device, layout.socket_header_send);
  auto *receive_headers = at<SocketRouteHeader>(
      slot.device, layout.socket_header_receive);
  auto *send_payload = at<transport::BFloat16>(slot.device,
                                                layout.contribution);
  auto *receive_payload = at<transport::BFloat16>(
      slot.device, layout.socket_payload_receive);
  auto *canonical = at<transport::BFloat16>(slot.device,
                                             layout.canonical_rows);
  auto *by_route = at<transport::BFloat16>(slot.device,
                                            layout.combine_return);
  auto *combined = at<transport::BFloat16>(slot.device, layout.combined);
  auto *status = at<TransportDeviceStatus>(slot.device, layout.status);
  const auto world = id.world_size;
  const auto route_count = view.sizes.route_count;

  internal::throw_cuda(
      "socket-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(Socket start)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(local_counts, 0,
                                     world * sizeof(std::uint64_t),
                                     exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket counts)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(status, 0, sizeof(*status), exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket status)");
  if (route_count != 0) {
    socket_route_count_kernel<<<
        static_cast<unsigned int>((route_count + 255ULL) / 256ULL), 256, 0,
        exec->stream>>>(view, local_counts);
  }
  socket_route_prefix_kernel<<<1, 1, 0, exec->stream>>>(local_counts, prefix,
                                                        world);
  if (route_count != 0) {
    socket_stable_route_ids_kernel<<<world, 256, 0, exec->stream>>>(
        view, prefix, route_ids);
    socket_pack_routes_kernel<<<static_cast<unsigned int>(route_count), 256, 0,
                                exec->stream>>>(
        view, activation.pointer,
        ActivationTransportBridge::gate_pointer(dispatch), config.hidden_size,
        activation.slot->epoch, route_ids, send_headers, send_payload);
  }
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket pack kernels");
  internal::throw_nccl(
      "socket-run",
      ncclAllGather(local_counts, count_matrix, world, ncclUint64,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(Socket route counts)");
  const auto count_bytes = checked_size(
      checked_mul(checked_mul(world, world, "Socket actual count matrix"),
                  sizeof(std::uint64_t), "Socket actual count bytes"),
      "Socket actual count bytes");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_socket_counts, count_matrix, count_bytes,
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "D2H Socket route counts");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            count_bytes);
  internal::throw_cuda(
      "socket-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(Socket counts)");

  std::vector<std::uint64_t> send_prefix(static_cast<std::size_t>(world) + 1U,
                                         0);
  std::vector<std::uint64_t> receive_prefix(
      static_cast<std::size_t>(world) + 1U, 0);
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    send_prefix[peer + 1U] =
        send_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                    world +
                                peer];
    receive_prefix[peer + 1U] =
        receive_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(peer) * world +
                                id.calling_rank];
  }
  const auto incoming = receive_prefix[world];
  if (send_prefix[world] != route_count ||
      incoming > state->config.max_destination_rows_per_rank)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "route totals", id.calling_rank,
                    "send=route_count and incoming<=capacity",
                    "send=" + std::to_string(send_prefix[world]) +
                        ", incoming=" + std::to_string(incoming));

  const auto enqueue_dispatch = [&] {
    internal::throw_nccl("socket-dispatch", ncclGroupStart(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupStart(Socket dispatch)");
    for (std::uint32_t peer = 0; peer < world; ++peer) {
      const auto send_rows = send_prefix[peer + 1U] - send_prefix[peer];
      const auto receive_rows =
          receive_prefix[peer + 1U] - receive_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-dispatch", "self rows", id.calling_rank,
                          std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0) {
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(receive_headers + receive_prefix[peer],
                              send_headers + send_prefix[peer],
                              checked_size(send_rows * sizeof(*send_headers),
                                           "Socket self header bytes"),
                              cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self headers");
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(
                  receive_payload + receive_prefix[peer] * config.hidden_size,
                  send_payload + send_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket self payload bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self payload");
        }
        continue;
      }
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_headers + send_prefix[peer],
                     send_rows * sizeof(*send_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_payload + send_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket payload)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_headers + receive_prefix[peer],
                     receive_rows * sizeof(*receive_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket payload)");
      }
    }
    internal::throw_nccl("socket-dispatch", ncclGroupEnd(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupEnd(Socket dispatch)");
  };
  enqueue_dispatch();
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket dispatch complete)");
  if (incoming != 0)
    socket_relay_transport_kernel<<<static_cast<unsigned int>(incoming), 256,
                                     0, exec->stream>>>(
        receive_headers, receive_payload, canonical,
        at<float>(slot.device, layout.canonical_gates), incoming,
        config.hidden_size, id.generation, id.calling_rank,
        state->config.max_destination_rows_per_rank,
        config.expert_transform ==
            transport::ActivationExpertTransform::Identity,
        status);

  internal::throw_nccl("socket-return", ncclGroupStart(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupStart(Socket return)");
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
    const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
    if (peer == id.calling_rank) {
      if (send_rows != 0)
        internal::throw_cuda(
            "socket-return",
            cudaMemcpyAsync(
                send_payload + send_prefix[peer] * config.hidden_size,
                receive_payload + receive_prefix[peer] * config.hidden_size,
                checked_size(send_rows * config.hidden_size *
                                 sizeof(transport::BFloat16),
                             "Socket self return bytes"),
                cudaMemcpyDeviceToDevice, exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "D2D Socket self return");
      continue;
    }
    if (send_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclSend(receive_payload +
                       receive_prefix[peer] * config.hidden_size,
                   send_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclSend(Socket return)");
    if (receive_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                   receive_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclRecv(Socket return)");
  }
  internal::throw_nccl("socket-return", ncclGroupEnd(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupEnd(Socket return)");
  if (route_count != 0)
    socket_scatter_returns_kernel<<<static_cast<unsigned int>(route_count),
                                    256, 0, exec->stream>>>(
        send_headers, send_payload, by_route, route_count,
        config.hidden_size);
  if (id.num_tokens != 0)
    socket_gate_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens), 256,
                                0, exec->stream>>>(
        view, ActivationTransportBridge::gate_pointer(dispatch), by_route,
        combined, config.hidden_size);
  socket_finalize_status_kernel<<<1, 1, 0, exec->stream>>>(
      status, route_count, incoming, id.num_tokens);
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket relay/return/reduce kernels");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_status, status, sizeof(*status),
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H Socket status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(*status));
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "socket-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(Socket complete)");
  consumer.finish();
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size, *slot.host_status,
      slot.complete);
  if (production_agreement.has_error) {
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  float dispatch_ms = 0.0F;
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket total)");
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&dispatch_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket dispatch)");
  const auto gpus_per_node = world / id.num_nodes;
  const auto source_node = id.calling_rank / gpus_per_node;
  std::uint64_t remote_rows = 0;
  for (std::uint32_t peer = 0; peer < world; ++peer)
    if (peer / gpus_per_node != source_node)
      remote_rows += send_prefix[peer + 1U] - send_prefix[peer];
  const auto remote_bytes = checked_mul(
      checked_mul(remote_rows, config.hidden_size, "Socket remote elements"),
      sizeof(transport::BFloat16), "Socket remote bytes");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->socket_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::NcclSocketFallback;
  result->audit.packet_count = route_count;
  result->audit.fanout_count = incoming;
  result->audit.raw_cross_node_payload_bytes = remote_bytes;
  result->audit.deduplicated_cross_node_payload_bytes = remote_bytes;
  result->audit.actual_dispatch_peer_bytes = remote_bytes;
  result->audit.actual_return_peer_bytes = remote_bytes;
  result->audit.derived_route_input_bytes =
      route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes = route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.kernel_launches = 7;
  result->audit.dispatch_kernel_blocks = route_count;
  result->audit.return_kernel_blocks = incoming;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  // Only fixed-width control/count/status metadata is staged. No activation
  // payload crosses the host boundary.
  result->audit.activation_d2h_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.dispatch_relay_us =
      static_cast<double>(dispatch_ms) * 1000.0;
  result->timings.combine_reduce_us =
      result->timings.total_us - result->timings.dispatch_relay_us;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle GinRdmaActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
#if !__has_include(<nccl_device.h>)
  (void)dispatch;
  (void)activations;
  (void)config;
  (void)options;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-run, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  if (id.num_nodes != backend->num_nodes ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=runtime GIN team topology and exact rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_gin_rdma ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k ||
      activation.slot->index >= backend->windows.size())
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "pool geometry/window", id.calling_rank,
                    "within registered GIN pool capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=gin-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (view.sizes.incoming_row_count >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count >
          checked_mul(state->config.max_tokens_per_rank,
                      state->config.max_top_k, "GIN route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");
  SlotFailureGuard failure_guard{activation.slot, false};
  const auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  internal::throw_cuda(
      "gin-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN start)");
  internal::throw_cuda(
      "gin-run", cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                                  sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN status)");
  internal::throw_cuda(
      "gin-run",
      cudaMemsetAsync(at<void>(slot.gin_window,
                               state->gin_layout.barrier), 0,
                      sizeof(std::uint64_t), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN reset barrier)");
  gin_signal_reset_kernel<<<backend->gin_allocated_context_count, 1, 0,
                            exec->stream>>>(
      backend->dev_comm, backend->gin_allocated_context_count);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "GIN reset kernel launch");
  auto *barrier = at<std::uint64_t>(slot.gin_window,
                                    state->gin_layout.barrier);
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(after GIN reset)");

  GinTransportSliceArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.epoch = activation.slot->epoch;
  args.dev_comm = backend->dev_comm;
  args.window = backend->windows[activation.slot->index];
  args.compute_base = slot.device;
  args.window_base = slot.gin_window;
  args.send_header_offset = state->gin_layout.send_packet_headers;
  args.receive_header_offset = state->gin_layout.receive_packet_headers;
  args.send_fanout_offset = state->gin_layout.send_fanout;
  args.receive_fanout_offset = state->gin_layout.receive_fanout;
  args.send_payload_offset = state->gin_layout.send_payload;
  args.receive_payload_offset = state->gin_layout.receive_payload;
  args.canonical_offset = layout.canonical_rows;
  args.return_staging_offset = state->gin_layout.return_staging;
  args.return_receive_offset = state->gin_layout.return_receive;
  args.combine_compute_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.peer_bases_offset = layout.peer_bases;
  args.local_world_size = backend->local_world_size;
  args.node = backend->node;
  args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  gin_transport_slice_kernel<<<1, 256, 0, exec->stream>>>(args);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "GIN activation persistent kernel launch");
  internal::throw_cuda(
      "gin-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H GIN status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "gin-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(GIN complete)");
  // A rank may destroy its registered window immediately after its result and
  // transport object leave scope. Rendezvous only after every device worker
  // has completed all remote puts so no peer can deregister early.
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(GIN terminal window lease)");
  internal::throw_cuda(
      "gin-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaStreamSynchronize(GIN terminal window lease)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure, "gin-run",
                    "device data plane", id.calling_rank, "valid",
                    "device error field=" +
                        std::to_string(slot.host_status->error_field));

  float total_ms = 0.0F;
  internal::throw_cuda(
      "gin-timing", cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(GIN total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->gin_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend = transport::ActivationTransportBackend::Gin;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = 2;
  result->audit.generation_memset_bytes =
      sizeof(TransportDeviceStatus) + sizeof(std::uint64_t);
  result->audit.kernel_launches = 2;
  result->audit.dispatch_kernel_blocks = 1;
  result->audit.return_kernel_blocks = 1;
  result->audit.reduce_kernel_blocks = 1;
  result->audit.activation_d2h_bytes = 0;
  result->audit.activation_h2d_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
#endif
}

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights,
    const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations || !weights)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-run",
                    "backend/dispatch/activation/weights", 0, "valid",
                    "empty");
  const auto backend = impl_->backend;
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(backend);
  const auto socket_backend =
      std::dynamic_pointer_cast<SocketBackendState>(backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const bool use_socket = static_cast<bool>(socket_backend);
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  const auto &weight = *weights.impl_;
  if ((!use_gin && !use_socket && id.num_nodes != 1) ||
      (use_gin && id.num_nodes != gin_backend->num_nodes) ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=" +
            (use_gin ? "runtime GIN topology"
                     : use_socket ? "runtime NCCL Socket topology"
                                  : "one local node") +
            " and exact NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  // The hot GIN fanout record keeps its established 32-byte width by storing
  // the node-local rank and global expert ID in 16 bits.  Reject wider
  // topologies before any device-side narrowing instead of silently wrapping.
  if (use_gin &&
      (id.world_size > std::numeric_limits<std::uint16_t>::max() ||
       id.num_experts > std::numeric_limits<std::uint16_t>::max()))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "GIN fanout identity width", id.calling_rank,
                    "world_size and num_experts <= 65535",
                    std::to_string(id.world_size) + "/" +
                        std::to_string(id.num_experts));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (weight.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      weight.device != backend->execution.device_ordinal() ||
      weight.hidden != config.hidden_size ||
      weight.intermediate != config.intermediate_size || !weight.allocation ||
      !weight.allocation->ready)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "weight provenance", id.calling_rank,
                    "exact session/execution/device/shape/ready", "mismatch");
  if (state->config.max_hidden_size != config.hidden_size ||
      state->config.max_intermediate_size != config.intermediate_size ||
      !state->slots[activation.slot->index].forward_maps_valid ||
      (config.retain_backward_state &&
       (!state->config.enable_backward_state ||
        state->layout.forward_ab_cache == 0 ||
        state->layout.forward_training_output == 0)))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "forward pool geometry/backward state", id.calling_rank,
                    "Hopper 7168x3072 TMA maps and requested retained state",
                    "mismatch/missing");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  auto view = ActivationTransportBridge::view(consumer);
  // Notify materializes the route permutation and destination M-tile map for
  // every topology.  The split one-node transport consumes the same metadata
  // as GIN Dispatch even though it has no RDMA waves of its own.
  const bool split_local_metadata =
      !use_gin && !use_socket && id.num_nodes == 1U;
  const auto dispatch_waves =
      use_gin || split_local_metadata
          ? ActivationTransportBridge::wave_view(consumer)
          : DispatchWaveKernelView{};
  if (weight.owned_experts != view.sizes.owned_expert_count)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "owned_experts", id.calling_rank,
                    std::to_string(view.sizes.owned_expert_count),
                    std::to_string(weight.owned_experts));
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
#if MEGAMOE_DEBUG_VALIDATE
  const auto debug_forward_host_phase = [&](const char *phase) {
    if (std::getenv("MEGAMOE_FORWARD_PROGRESS") != nullptr)
      std::cerr << "FORWARD_RUN_HOST rank=" << id.calling_rank
                << " generation=" << id.generation << " phase=" << phase
                << std::endl;
  };
  debug_forward_host_phase("consumer_acquired");
#endif
  if (view.sizes.incoming_row_count >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "forward route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");

  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested() ? id.generation
                                                                 : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "forward-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(activation ready)");
  internal::throw_cuda(
      "forward-run",
      cudaStreamWaitEvent(exec->stream, weight.allocation->ready),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(weights ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("stream_dependencies_enqueued");
#endif

  if (slot.transport_epoch >= (1ULL << 63U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "forward-run",
                    "transport_epoch", id.calling_rank, "<2^63-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  std::uint64_t socket_incoming = 0;
  std::uint64_t socket_dispatch_peer_bytes = 0;
  std::uint64_t socket_return_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_route_bytes = 0;
  if (use_socket) {
    // The fallback transport and the fused Forward are sequential consumers
    // of the same immutable Dispatch handle. Release the first token while
    // the host-enqueued Socket slice owns it, then reacquire before launching
    // the production Forward kernel; retaining both would exhaust the bounded
    // consumer-event pool and stall before the first payload collective.
    consumer = {};
    transport::ActivationTransportConfig socket_config{};
    socket_config.hidden_size = config.hidden_size;
    socket_config.channels = state->config.channels;
    socket_config.ring_slots_per_source_channel =
        state->config.ring_slots_per_source_channel;
    socket_config.max_inflight_generations =
        state->config.max_inflight_generations;
    socket_config.expert_transform =
        transport::ActivationExpertTransform::Identity;
    socket_config.gate_placement =
        transport::ActivationGatePlacement::SourceAfter;
    const auto socket_preload = impl_->socket_transport->run_transport_slice(
        dispatch, activations, socket_config, options);
    socket_incoming = socket_preload.audit().fanout_count;
    // Count the actual rank-to-rank NCCL payloads. The transport-only audit is
    // intentionally cross-node centric, whereas this full fallback also sends
    // aggregates between ranks on the same host. The preload performs an
    // identity return which is real lower-bound traffic and must not be hidden.
    const auto local_world_size = id.world_size / id.num_nodes;
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      if (peer == id.calling_rank)
        continue;
      const auto outgoing_rows =
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      const auto incoming_rows =
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
      const auto outgoing_bytes = checked_mul(
          checked_mul(outgoing_rows, config.hidden_size,
                      "Socket Forward preload outgoing elements"),
          sizeof(transport::BFloat16),
          "Socket Forward preload outgoing bytes");
      socket_dispatch_peer_bytes = checked_add(
          socket_dispatch_peer_bytes, outgoing_bytes,
          "Socket Forward total dispatch peer bytes");
      socket_return_peer_bytes = checked_add(
          socket_return_peer_bytes,
          checked_mul(checked_mul(incoming_rows, config.hidden_size,
                                  "Socket Forward preload return elements"),
                      sizeof(transport::BFloat16),
                      "Socket Forward preload return bytes"),
          "Socket Forward total preload return bytes");
      if (peer / local_world_size != id.calling_rank / local_world_size)
        socket_raw_cross_node_route_bytes = checked_add(
            socket_raw_cross_node_route_bytes, outgoing_bytes,
            "Socket Forward cross-node route bytes");
    }
    consumer = dispatch.acquire_consumer(backend->execution, expectation);
    view = ActivationTransportBridge::view(consumer);
  }
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.forward_start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward start)");
  internal::throw_cuda(
      "forward-run",
      cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                      sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "clear forward status");
#if MEGAMOE_DEVICE_TIMELINE
  reset_timeline("forward-run", slot.forward_timeline,
                 state->timeline_layout, state->timeline_max_ctas,
                 exec->stream, id.calling_rank, id.world_size,
                 id.generation, backend->session);
#endif

  // Notify has already completed the rank-wide generation agreement on this
  // communicator.  The IPC team and per-slot peer mappings are session-static,
  // while payload visibility is ordered by the device release/acquire signals
  // below, so a second host-enqueued AllGather here only serialized E2E.

  if (use_gin) {
#if __has_include(<nccl_device.h>)
    auto *gin_barrier =
        at<std::uint64_t>(slot.gin_window, state->gin_layout.barrier);
    internal::throw_cuda(
        "forward-run",
        cudaMemsetAsync(gin_barrier, 0, sizeof(std::uint64_t), exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts, "reset Forward GIN barrier");
    internal::throw_cuda(
        "forward-run",
        cudaMemsetAsync(
            at<void>(slot.gin_window,
                     state->gin_layout.return_fragment_ready),
            0,
            checked_size(checked_mul(state->config.max_tokens_per_rank,
                                     state->config.max_top_k,
                                     "Forward direct Return counters"),
                         "Forward direct Return counter count") *
                sizeof(std::uint64_t),
            exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "reset Forward direct Return fragment counters");
    gin_signal_reset_kernel<<<gin_backend->gin_allocated_context_count, 1, 0,
                              exec->stream>>>(
        gin_backend->dev_comm, gin_backend->gin_allocated_context_count);
    internal::throw_cuda("forward-run", cudaGetLastError(), id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "reset Forward GIN signal/counter");
    internal::throw_nccl(
        "forward-run",
        ncclAllReduce(gin_barrier, gin_barrier, 1, ncclUint64, ncclSum,
                      backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "ncclAllReduce(Forward GIN reset agreement)");
#if MEGAMOE_DEBUG_VALIDATE
    debug_forward_host_phase("gin_reset_enqueued");
#endif
#endif
  }
  if (!use_socket) {
    initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
        slot.device, at<void *>(slot.device, layout.peer_bases),
        layout.ingress_published_sequence, layout.outgoing_route_sequence,
        layout.p2p_barrier_receive, layout.gin_local_compute_ready,
        layout.forward_x_arrived_rows, view.sizes.mtile_count,
        layout.forward_completion, layout.forward_context_completion,
        static_cast<std::uint32_t>((state->config.max_nodes - 1U) * 8U),
        id.generation, id.calling_rank, id.world_size, state->config.channels,
        generation_epoch,
        use_gin ? gin_backend->node * gin_backend->local_world_size : 0U,
        use_gin ? gin_backend->local_world_size : id.world_size);
  } else if (socket_incoming != 0) {
    socket_install_forward_headers_kernel<<<
        static_cast<unsigned int>((socket_incoming + 255ULL) / 256ULL), 256, 0,
        exec->stream>>>(
        at<SocketRouteHeader>(slot.device, layout.socket_header_receive),
        at<LocalP2pRouteHeader>(slot.device, layout.direct_route_headers),
        socket_incoming, id.generation, generation_epoch, id.calling_rank,
        state->config.max_destination_rows_per_rank,
        at<TransportDeviceStatus>(slot.device, layout.status));
  }
  internal::throw_cuda(
      "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize owner completion signals and wait peer setup epoch");
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("generation_init_enqueued");
#endif
  ForwardKernelArgs args{};
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.input = activation.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.w1 = weight.w1;
  args.w2 = weight.w2;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = state->config.channels;
  args.epoch = generation_epoch;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.use_gin = use_gin;
  args.use_socket = use_socket;
  args.local_world_size = id.world_size / id.num_nodes;
  args.node = id.calling_rank / args.local_world_size;
#if MEGAMOE_DEVICE_TIMELINE
  args.timeline = make_timeline_context(
      slot.forward_timeline, state->timeline_layout,
      state->timeline_max_ctas, id.calling_rank,
      profiling::TimelinePhase::Forward, id.generation);
#endif
#if MEGAMOE_BUILD_TESTING
  if (const auto *fault_rank =
          std::getenv("MEGAMOE_INTERNAL_FORWARD_STATUS_FAULT_RANK")) {
    const auto *fault_generation =
        std::getenv("MEGAMOE_INTERNAL_FORWARD_STATUS_FAULT_GENERATION");
    if (fault_generation == nullptr ||
        std::stoull(fault_generation) == id.generation)
      args.injected_status_error_rank = std::stoi(fault_rank);
  }
#endif
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[activation.slot->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.gin_send_header_offset = state->gin_layout.send_packet_headers;
    args.gin_receive_header_offset = state->gin_layout.receive_packet_headers;
    args.gin_send_dispatch_directory_offset =
        state->gin_layout.send_dispatch_directory;
    args.gin_receive_dispatch_directory_offset =
        state->gin_layout.receive_dispatch_directory;
    args.gin_send_fanout_offset = state->gin_layout.send_fanout;
    args.gin_receive_fanout_offset = state->gin_layout.receive_fanout;
    args.gin_send_payload_offset = state->gin_layout.send_payload;
    args.gin_receive_payload_offset = state->gin_layout.receive_payload;
    args.gin_return_staging_offset = state->gin_layout.return_staging;
    args.gin_return_receive_offset = state->gin_layout.return_receive;
    args.gin_return_fragment_ready_offset =
        state->gin_layout.return_fragment_ready;
    args.gin_production_error_records_offset =
        state->gin_layout.production_error_records;
    args.node = gin_backend->node;
    args.local_world_size = gin_backend->local_world_size;
    args.gin_context_count = gin_backend->gin_context_count;
    args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  }
  args.canonical_rows_offset = layout.canonical_rows;
  args.canonical_gates_offset = layout.canonical_gates;
  args.forward_output_rows_offset = config.retain_backward_state
                                        ? layout.forward_training_output
                                        : layout.canonical_rows;
  args.direct_headers_offset = layout.direct_route_headers;
  args.contribution_offset = layout.contribution;
  args.return_ready_epoch_offset = layout.return_ready_epoch;
  args.combined_offset = layout.combined;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.status_offset = layout.status;
  args.forward_s_offset = layout.forward_s;
  args.forward_ab_cache_offset = layout.forward_ab_cache;
  args.retain_backward_state = config.retain_backward_state;
  args.forward_task_claim_offset = layout.forward_task_claim;
  args.forward_fc1_done_offset = layout.forward_fc1_done;
  args.forward_fc1_published_offset = layout.forward_fc1_published;
  args.forward_compute_pair_mailboxes_offset =
      layout.forward_compute_pair_mailboxes;
  args.x_arrived_rows_offset = layout.forward_x_arrived_rows;
  args.forward_output_ready_epoch_offset = layout.forward_output_ready_epoch;
  args.forward_return_chunk_directory_offset =
      layout.forward_return_chunk_directory;
  args.forward_return_chunk_state_offset = layout.forward_return_chunk_state;
  args.forward_return_chunk_ready_count_offset =
      layout.forward_return_chunk_ready_count;
  args.forward_return_fragment_index_offset =
      layout.forward_return_fragment_index;
  args.forward_return_source_compact_index_offset =
      layout.forward_return_source_compact_index;
  args.forward_return_source_route_index_offset =
      layout.forward_return_source_route_index;
  args.forward_direct_source_compact_index_offset =
      layout.forward_direct_source_compact_index;
  args.forward_reduce_route_count_offset =
      layout.forward_reduce_route_count;
  args.forward_reduce_ready_token_offset =
      layout.forward_reduce_ready_token;
  args.forward_direct_return_fragment_offset =
      layout.forward_direct_return_fragment;
  args.forward_return_chunk_capacity = layout.forward_return_chunk_capacity;
  args.forward_direct_return_fragment_capacity =
      layout.forward_direct_return_fragment_capacity;
  // Cross-node Forward has a single Return transport: FULL GIN. Dispatch
  // intentionally remains same-rail, while the second physical context set
  // gives each permanent RDMA CTA an independent FULL Return QP.
  if (use_gin && id.num_nodes != 2U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "FULL Return topology", id.calling_rank,
                    "exactly two nodes", std::to_string(id.num_nodes));
  if (use_gin && !gin_backend->full_connections)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "FULL Return connections", id.calling_rank,
                    "NCCL_GIN_CONNECTION_FULL", "unavailable");
  args.forward_return_chunk_rows = forward_tuning::kReturnChunkRows;
  if constexpr (forward_tuning::kEnableLegacyEnvironment) {
    if (const auto *chunk_rows =
            std::getenv("MEGAMOE_FORWARD_RETURN_CHUNK_ROWS")) {
      const auto parsed = std::stoull(chunk_rows);
      if (parsed != 0ULL)
        args.forward_return_chunk_rows = parsed;
    }
  }
  bool reduce_queue_requested = forward_tuning::kReduceScheduler == 0U;
  bool reduce_scan_requested = forward_tuning::kReduceScheduler == 1U;
  bool legacy_reduce_requested = forward_tuning::kReduceScheduler == 2U;
  if constexpr (forward_tuning::kEnableLegacyEnvironment) {
    reduce_queue_requested = false;
    reduce_scan_requested = false;
    legacy_reduce_requested = false;
    reduce_queue_requested =
        std::getenv("MEGAMOE_FORWARD_DYNAMIC_REDUCE_QUEUE") != nullptr;
    reduce_scan_requested =
        std::getenv("MEGAMOE_FORWARD_DYNAMIC_REDUCE_SCAN") != nullptr;
    legacy_reduce_requested =
        std::getenv("MEGAMOE_FORWARD_LEGACY_REDUCE") != nullptr;
  }
  if (static_cast<unsigned int>(reduce_queue_requested) +
          static_cast<unsigned int>(reduce_scan_requested) +
          static_cast<unsigned int>(legacy_reduce_requested) >
      1U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "MEGAMOE_FORWARD_LEGACY_REDUCE", id.calling_rank,
                    "only one reduce scheduler override",
                    "queue, scan, or legacy");
  // Route-ready queueing is the production scheduler.  The explicit queue
  // variable remains accepted for reproducible A/B commands; scan and fixed
  // token stripes are diagnostic fallbacks.
  args.forward_dynamic_reduce_scan = reduce_scan_requested;
  args.forward_dynamic_reduce_queue = !reduce_scan_requested &&
                                      !legacy_reduce_requested;
  args.forward_scheduler_offset = layout.forward_scheduler;
  args.forward_v3_control_offset = layout.forward_v3_control;
  args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
  args.terminal_epoch_offset = layout.terminal_epoch;
  args.production_error_records_offset = layout.production_error_records;
  args.cancel_epoch =
      reinterpret_cast<const unsigned long long *>(slot.device_cancel_epoch);
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
  constexpr std::size_t forward_progress_base = 0;
  if (std::getenv("MEGAMOE_FORWARD_PROGRESS_DUMP") != nullptr) {
    std::fill_n(slot.host_forward_progress + forward_progress_base,
                kForwardProgressWords + 2ULL * slot.generation_debug_words,
                0ULL);
    args.progress = reinterpret_cast<unsigned long long *>(
        slot.device_forward_progress + forward_progress_base);
    args.progress_contexts = slot.generation_debug_contexts;
  }
#endif

  int cluster_supported = 0;
  int max_active_clusters = 0;
  int sms = 0;
    internal::throw_cuda(
        "forward-run",
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                               static_cast<int>(exec->device)),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaDeviceGetAttribute(SM count)");
    const auto grid_blocks = static_cast<unsigned int>(sms);
    if (grid_blocks == 0)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          "phase=forward-run, field=SM count, rank=" +
              std::to_string(id.calling_rank) +
              ", expected>=2, actual=" + std::to_string(sms));
    const std::size_t forward_compute_smem =
        forward_tuning::compute_smem_bytes(id.world_size);
    const std::size_t forward_transport_smem =
        forward_tuning::transport_smem_bytes(
            static_cast<std::uint32_t>(args.hidden), id.world_size);
    const std::size_t forward_smem =
        std::max(forward_compute_smem, forward_transport_smem);
    const bool remote_transport = use_gin || use_socket;
    const bool split_gin_roles =
        use_gin && view.identity.num_nodes == 2U &&
        args.gin_context_count == persistent_v3::kRdmaCtaCount &&
        forward_tuning::kRdmaCtas == persistent_v3::kRdmaCtaCount;
    const bool split_local_roles =
        !remote_transport && view.identity.num_nodes == 1U;
    const bool split_forward_roles = split_gin_roles || split_local_roles;
    cudaError_t launch_forward = cudaSuccess;
    if (split_forward_roles) {
      const auto &ordinary_output_store_map =
          config.retain_backward_state
              ? slot.forward_training_output_store_map
              : slot.forward_output_store_map;
      args.forward_relay_cta_count = forward_tuning::kRelayCtas;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *relay_ctas =
                std::getenv("MEGAMOE_FORWARD_RELAY_CTAS"))
          args.forward_relay_cta_count =
              static_cast<std::uint32_t>(std::stoul(relay_ctas));
      }
      const bool valid_relay_count =
          split_local_roles
              ? args.forward_relay_cta_count >=
                        std::max(4U, args.local_world_size) &&
                    args.forward_relay_cta_count <= 32U &&
                    (args.forward_relay_cta_count % 4U) == 0U
              : args.forward_relay_cta_count >=
                        persistent_v3::kRdmaCtaCount &&
                    args.forward_relay_cta_count %
                            persistent_v3::kRdmaCtaCount ==
                        0U &&
                    args.forward_relay_cta_count <= 24U;
      if (!valid_relay_count)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "MEGAMOE_FORWARD_RELAY_CTAS", id.calling_rank,
                        split_local_roles ? "one of 4, 8, ..., 32"
                                          : "one of 8, 16, or 24",
                        std::to_string(args.forward_relay_cta_count));
      const unsigned int kTransportBlocks =
          (split_gin_roles ? persistent_v3::kRdmaCtaCount : 0U) +
          args.forward_relay_cta_count;
      if (grid_blocks <= kTransportBlocks)
        throw NotifyDispatchError(
            NotifyErrorCode::BackendUnavailable,
            "phase=forward-run, field=split role residency, rank=" +
                std::to_string(id.calling_rank) + ", expected SMs>" +
                std::to_string(kTransportBlocks) +
                ", actual=" + std::to_string(grid_blocks));
      args.forward_compute_cta_count = grid_blocks - kTransportBlocks;
      args.forward_stream_local_dispatch = split_local_roles;
      args.forward_relay_local_return = forward_tuning::kRelayLocalReturn;
      if constexpr (forward_tuning::kEnableLegacyEnvironment)
        args.forward_relay_local_return =
            std::getenv("MEGAMOE_FORWARD_COMPUTE_LOCAL_RETURN") == nullptr;
      // Eight Relay CTAs with eight independent row streams are the Hopper
      // production point.  Twelve streams slightly improve isolated NVLink
      // copy throughput, but contend more heavily with the paired FC1/FC2
      // pipeline and produce a longer end-to-end Return tail.  A value of
      // zero retains the exact vector implementation for autotuner/regression
      // A/B runs.
      args.forward_relay_tma_warps = forward_tuning::kRelayTmaWarps;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *tma_warps =
                std::getenv("MEGAMOE_FORWARD_RELAY_TMA_WARPS"))
          args.forward_relay_tma_warps =
              static_cast<std::uint32_t>(std::stoul(tma_warps));
      }
      if (args.forward_relay_tma_warps != 0U &&
          args.forward_relay_tma_warps != 4U &&
          args.forward_relay_tma_warps != 8U &&
          args.forward_relay_tma_warps != persistent_v3::kWarpsPerCta)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "MEGAMOE_FORWARD_RELAY_TMA_WARPS", id.calling_rank,
                        "one of 0, 4, 8, or 12",
                        std::to_string(args.forward_relay_tma_warps));
      args.forward_local_return_batch_mblocks =
          forward_tuning::kLocalReturnBatchMblocks;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *batch_mblocks =
                std::getenv("MEGAMOE_FORWARD_LOCAL_RETURN_BATCH_MBLOCKS"))
          args.forward_local_return_batch_mblocks =
              static_cast<std::uint32_t>(std::stoul(batch_mblocks));
      }
      if (args.forward_local_return_batch_mblocks != 1U &&
          args.forward_local_return_batch_mblocks != 2U &&
          args.forward_local_return_batch_mblocks != 4U &&
          args.forward_local_return_batch_mblocks != 8U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_LOCAL_RETURN_BATCH_MBLOCKS", id.calling_rank,
            "one of 1, 2, 4, or 8",
            std::to_string(args.forward_local_return_batch_mblocks));
      args.forward_dispatch_publish_batch_rows =
          forward_tuning::kDispatchPublishBatchRows;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *batch_rows =
                std::getenv("MEGAMOE_FORWARD_DISPATCH_PUBLISH_BATCH_ROWS"))
          args.forward_dispatch_publish_batch_rows =
              static_cast<std::uint32_t>(std::stoul(batch_rows));
      }
      if (args.forward_dispatch_publish_batch_rows != 4U &&
          args.forward_dispatch_publish_batch_rows != 8U &&
          args.forward_dispatch_publish_batch_rows != 16U &&
          args.forward_dispatch_publish_batch_rows != 32U &&
          args.forward_dispatch_publish_batch_rows != 64U &&
          args.forward_dispatch_publish_batch_rows != 128U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_DISPATCH_PUBLISH_BATCH_ROWS", id.calling_rank,
            "one of 4, 8, 16, 32, 64, or 128",
            std::to_string(args.forward_dispatch_publish_batch_rows));
      // A two-block base cadence keeps FC2/Return readiness fine-grained. The
      // saturated middle window widens to four for stronger W1 L2 reuse;
      // keeping its bounds tunable is important because widening near the
      // tail improves isolated GEMM time but lengthens Return/Reduce sharply.
      args.forward_fc1_locality_group = forward_tuning::kFc1LocalityGroup;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *locality_group =
                std::getenv("MEGAMOE_FORWARD_FC1_LOCALITY_GROUP"))
          args.forward_fc1_locality_group =
              static_cast<std::uint32_t>(std::stoul(locality_group));
      }
      if (args.forward_fc1_locality_group != 1U &&
          args.forward_fc1_locality_group != 2U &&
          args.forward_fc1_locality_group != 4U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP", id.calling_rank,
            "one of 1, 2, or 4",
            std::to_string(args.forward_fc1_locality_group));
      std::uint32_t locality_group4_begin =
          forward_tuning::kFc1LocalityGroup4Begin;
      std::uint32_t locality_group4_end =
          forward_tuning::kFc1LocalityGroup4End;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *begin =
                std::getenv("MEGAMOE_FORWARD_FC1_LOCALITY_GROUP4_BEGIN"))
          locality_group4_begin =
              static_cast<std::uint32_t>(std::stoul(begin));
        if (const auto *end =
                std::getenv("MEGAMOE_FORWARD_FC1_LOCALITY_GROUP4_END"))
          locality_group4_end = static_cast<std::uint32_t>(std::stoul(end));
      }
      const auto locality_group4_span =
          locality_group4_end - locality_group4_begin;
      if (locality_group4_begin < 2U || locality_group4_end > UINT8_MAX ||
          (locality_group4_begin & 1U) != 0U ||
          locality_group4_end < locality_group4_begin ||
          (locality_group4_span & 3U) != 0U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP4_BEGIN/END",
            id.calling_rank,
            "begin>=2 and even; end<=255; end>=begin; span divisible by 4",
            std::to_string(locality_group4_begin) + ":" +
                std::to_string(locality_group4_end));
      args.forward_fc1_locality_group4_begin =
          static_cast<std::uint8_t>(locality_group4_begin);
      args.forward_fc1_locality_group4_end =
          static_cast<std::uint8_t>(locality_group4_end);
      args.forward_return_max_inflight = forward_tuning::kReturnMaxInflight;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *max_inflight =
                std::getenv("MEGAMOE_FORWARD_RETURN_MAX_INFLIGHT"))
          args.forward_return_max_inflight =
              static_cast<std::uint32_t>(std::stoul(max_inflight));
      }
      if (args.forward_return_max_inflight != 0U &&
          args.forward_return_max_inflight != 4U &&
          args.forward_return_max_inflight != 8U &&
          args.forward_return_max_inflight != 16U &&
          args.forward_return_max_inflight != 32U &&
          args.forward_return_max_inflight != 64U)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_RETURN_MAX_INFLIGHT", id.calling_rank,
            "one of 0, 4, 8, 16, 32, or 64",
            std::to_string(args.forward_return_max_inflight));
      // FC2 writes its result exactly once into the registered Return plane.
      // Relay TMA and remote GIN then consume that same immutable storage,
      // avoiding a full output-sized duplicate GMEM write.  Keep an explicit
      // 0/1 override for regression A/B and future autotuning.
      args.forward_registered_output = forward_tuning::kRegisteredOutput;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *registered_output =
                std::getenv("MEGAMOE_FORWARD_REGISTERED_OUTPUT")) {
          const std::string value(registered_output);
          if (value != "0" && value != "1")
            transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                            "forward-run",
                            "MEGAMOE_FORWARD_REGISTERED_OUTPUT",
                            id.calling_rank, "0 or 1", value);
          args.forward_registered_output = value == "1";
        }
      }
      bool forward_m64 = forward_tuning::kTileM == 64U;
      bool forward_m64_head = false;
      bool forward_specialize_stages = false;
      bool forward_fc2_tma_store = forward_tuning::kFc2TmaStore;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        forward_m64 = std::getenv("MEGAMOE_FORWARD_M64") != nullptr;
        forward_m64_head =
            std::getenv("MEGAMOE_FORWARD_M64_HEAD") != nullptr;
        forward_specialize_stages =
            std::getenv("MEGAMOE_FORWARD_SPECIALIZE_STAGES") != nullptr;
        if (const auto *tma_store =
                std::getenv("MEGAMOE_FORWARD_FC2_TMA_STORE")) {
          const std::string value(tma_store);
          if (value != "0" && value != "1")
            transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                            "forward-run", "MEGAMOE_FORWARD_FC2_TMA_STORE",
                            id.calling_rank, "0 or 1", value);
          forward_fc2_tma_store = value == "1";
        }
      }
      bool forward_cluster2 = forward_tuning::kClusterSize == 2U;
      bool forward_cluster2_explicit = false;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *cluster2 =
                std::getenv("MEGAMOE_FORWARD_CLUSTER2")) {
          forward_cluster2_explicit = true;
          const std::string value(cluster2);
          if (value != "0" && value != "1")
            transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                            "forward-run", "MEGAMOE_FORWARD_CLUSTER2",
                            id.calling_rank, "0 or 1", value);
          forward_cluster2 = value == "1";
        }
      }
      // The retained checkpoint entry is deliberately compiled separately
      // from inference. Keep its first implementation on the production M128
      // mixed-scheduler/TMA path instead of silently falling back to a runtime
      // branch in one of the obsolete experimental launch shapes.
      if (config.retain_backward_state &&
          (!forward_fc2_tma_store || forward_m64 || forward_m64_head ||
           forward_specialize_stages))
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "retained Forward specialization", id.calling_rank,
            "M128 FC2-priority mixed scheduler with TMA FC2 store",
            "incompatible experimental launch shape");
      // The H200 main-shape autotune selects four async TMA stages for both
      // GEMMs.  Environment overrides retain the other compiled candidates
      // for topology/shape-specific retuning without an inner-loop branch.
      std::uint32_t forward_fc1_stages = forward_tuning::kFc1Stages;
      std::uint32_t forward_fc2_stages = forward_tuning::kFc2Stages;
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (const auto *stages = std::getenv("MEGAMOE_FORWARD_FC1_STAGES"))
          forward_fc1_stages =
              static_cast<std::uint32_t>(std::stoul(stages));
        if (const auto *stages = std::getenv("MEGAMOE_FORWARD_FC2_STAGES"))
          forward_fc2_stages =
              static_cast<std::uint32_t>(std::stoul(stages));
      }
      const bool supported_stage_pair =
          (forward_fc1_stages == 2U && forward_fc2_stages == 4U) ||
          (forward_fc1_stages == 3U && forward_fc2_stages == 3U) ||
          (forward_fc1_stages == 3U && forward_fc2_stages == 4U) ||
          (forward_fc1_stages == 4U && forward_fc2_stages == 4U);
      if (!supported_stage_pair)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "Forward FC1/FC2 TMA stages", id.calling_rank,
            "one of 2x4, 3x3, 3x4, or 4x4",
            std::to_string(forward_fc1_stages) + "x" +
                std::to_string(forward_fc2_stages));
      if ((forward_fc1_stages != 3U || forward_fc2_stages != 4U) &&
          (forward_m64 || forward_m64_head || forward_specialize_stages))
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "Forward FC1/FC2 TMA stages", id.calling_rank,
            "the production mixed M128 scheduler",
            "combined with an obsolete M64/stage-split experiment");
      if (forward_m64 && forward_m64_head)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "Forward M tile mode", id.calling_rank,
                        "at most one of MEGAMOE_FORWARD_M64 and "
                        "MEGAMOE_FORWARD_M64_HEAD",
                        "both enabled");
      if (args.forward_fc1_locality_group > 1U &&
          (forward_m64 || forward_m64_head))
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_FC1_LOCALITY_GROUP", id.calling_rank,
            "M128 without an M64 experiment",
            std::to_string(args.forward_fc1_locality_group));
      if (forward_specialize_stages && (forward_m64 || forward_m64_head))
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "Forward stage specialization", id.calling_rank,
                        "M128 without an M64 experiment",
                        "MEGAMOE_FORWARD_SPECIALIZE_STAGES with M64");
      if (forward_fc2_tma_store &&
          (forward_m64 || forward_m64_head ||
           forward_fc1_stages != 4U || forward_fc2_stages != 4U))
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_FC2_TMA_STORE", id.calling_rank,
            "the production M128 4x4-stage scheduler",
            "an incompatible M tile or TMA stage configuration");
      if (forward_fc2_tma_store && args.use_gin &&
          !args.forward_registered_output)
        transport_error(
            NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
            "MEGAMOE_FORWARD_FC2_TMA_STORE", id.calling_rank,
            "registered output enabled for GIN Return",
            "MEGAMOE_FORWARD_REGISTERED_OUTPUT=0");
      if (forward_m64)
        args.forward_mblock_rows = 64U;
      args.forward_m64_head_wavefront = forward_m64_head;
      if (forward_m64_head) {
        args.forward_m64_head_expert_count = 1U;
        if (const auto *head_experts =
                std::getenv("MEGAMOE_FORWARD_M64_HEAD_EXPERTS"))
          args.forward_m64_head_expert_count =
              static_cast<std::uint32_t>(std::stoul(head_experts));
        if (args.forward_m64_head_expert_count == 0U ||
            args.forward_m64_head_expert_count >
                view.sizes.owned_expert_count)
          transport_error(
              NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
              "MEGAMOE_FORWARD_M64_HEAD_EXPERTS", id.calling_rank,
              "an integer in [1, owned_expert_count]",
              std::to_string(args.forward_m64_head_expert_count));
        args.forward_m64_head_cta_count = persistent_v3::kRdmaCtaCount;
        if (const auto *head_ctas =
                std::getenv("MEGAMOE_FORWARD_M64_HEAD_CTAS"))
          args.forward_m64_head_cta_count =
              static_cast<std::uint32_t>(std::stoul(head_ctas));
        if (args.forward_m64_head_cta_count == 0U ||
            args.forward_m64_head_cta_count >=
                args.forward_compute_cta_count)
          transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                          "MEGAMOE_FORWARD_M64_HEAD_CTAS", id.calling_rank,
                          "an integer in [1, compute_ctas - 1]",
                          std::to_string(args.forward_m64_head_cta_count));
      }
      const auto forward_main_compute_ctas =
          args.forward_compute_cta_count -
          (forward_m64_head ? args.forward_m64_head_cta_count : 0U);
      const bool forward_cluster2_pipeline_compatible =
          forward_fc2_tma_store && !forward_m64 && !forward_m64_head &&
          !forward_specialize_stages && forward_fc1_stages == 4U &&
          forward_fc2_stages == 4U;
      if (forward_cluster2 && !forward_cluster2_pipeline_compatible) {
        if (forward_cluster2_explicit)
          transport_error(
              NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
              "MEGAMOE_FORWARD_CLUSTER2", id.calling_rank,
              "the production M128 4x4-stage TMA-store scheduler",
              "an incompatible Forward experiment");
        forward_cluster2 = false;
      }
      const auto forward_fc1_ntiles =
          static_cast<std::uint32_t>(
              (args.intermediate + forward_tuning::kFc1TileN - 1ULL) /
              forward_tuning::kFc1TileN);
      const auto forward_fc2_ntiles =
          static_cast<std::uint32_t>(
              (args.hidden + forward_tuning::kFc2TileN - 1ULL) /
              forward_tuning::kFc2TileN);
      const bool forward_cluster2_extent_compatible =
          (forward_main_compute_ctas & 1U) == 0U &&
          (forward_fc1_ntiles & 1U) == 0U &&
          (forward_fc2_ntiles & 1U) == 0U &&
          forward_main_compute_ctas / 2U <=
              persistent_v3::kForwardComputePairCapacity;
      if (forward_cluster2 && !forward_cluster2_extent_compatible) {
        if (forward_cluster2_explicit)
          transport_error(
              NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
              "MEGAMOE_FORWARD_CLUSTER2", id.calling_rank,
              "even compute/FC1/FC2 task extents within mailbox capacity",
              std::to_string(forward_main_compute_ctas) + ":" +
                  std::to_string(forward_fc1_ntiles) + ":" +
                  std::to_string(forward_fc2_ntiles));
        forward_cluster2 = false;
      }
      // Expert storage remains padded to M=128. Repeated 32..512-row FULL
      // sweeps selected 256-row completion chunks: fewer marker scans and WQEs
      // with no measurable loss of FC2/Return overlap.
      if constexpr (forward_tuning::kEnableLegacyEnvironment) {
        if (std::getenv("MEGAMOE_FORWARD_RETURN_CHUNK_ROWS") == nullptr)
          args.forward_return_chunk_rows = forward_tuning::kReturnChunkRows;
      }

      // Production launches every compute CTA into one FC2-priority mixed
      // scheduler, so there is no fixed FC1/FC2 ownership split. Keep the
      // split count scoped to the obsolete compiler-specialization A/B path.
      args.forward_fc1_cta_count = 0U;
#if !MEGAMOE_FWD_COMPILED_TUNING
      if (forward_specialize_stages) {
        constexpr std::uint32_t kDefaultEarlyFc2Ctas = 48U;
        if (forward_main_compute_ctas <= kDefaultEarlyFc2Ctas)
          transport_error(
              NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
              "Forward specialized compute CTAs", id.calling_rank,
              "more than 48 main compute CTAs",
              std::to_string(forward_main_compute_ctas));
        args.forward_fc1_cta_count =
            forward_main_compute_ctas - kDefaultEarlyFc2Ctas;
        if (const auto *fc1_ctas =
                std::getenv("MEGAMOE_FORWARD_FC1_CTAS"))
          args.forward_fc1_cta_count =
              static_cast<std::uint32_t>(std::stoul(fc1_ctas));
        if (args.forward_fc1_cta_count == 0U ||
            args.forward_fc1_cta_count >= forward_main_compute_ctas)
          transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                          "forward-run", "MEGAMOE_FORWARD_FC1_CTAS",
                          id.calling_rank,
                          "an integer in [1, initial body compute_ctas - 1]",
                          std::to_string(args.forward_fc1_cta_count));
      } else if (forward_tuning::kEnableLegacyEnvironment &&
                 std::getenv("MEGAMOE_FORWARD_FC1_CTAS") != nullptr) {
        transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                        "MEGAMOE_FORWARD_FC1_CTAS", id.calling_rank,
                        "MEGAMOE_FORWARD_SPECIALIZE_STAGES enabled",
                        "set for the production mixed scheduler");
      }
#endif
      // In head-wavefront mode the initial resident set is exactly one CTA per
      // SM: 16 transport + head + main body.  A same-stream FC2-only tail grid
      // replaces the head CTAs after they exit, restoring all 116 compute
      // owners without ever starving the cooperative transport launch.
      args.forward_reduce_cta_count =
          (forward_main_compute_ctas + forward_tuning::kReduceCtaDivisor -
           1U) /
          forward_tuning::kReduceCtaDivisor;
      internal::throw_cuda(
          "forward-run",
          split_local_roles
              ? forward_transport_launch::set_local_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem))
              : forward_transport_launch::set_max_dynamic_shared_memory(
                    static_cast<int>(forward_transport_smem)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(split transport dynamic smem)");
      const auto set_tma_compute_attribute =
          [&](auto inference_kernel, auto training_kernel) {
            return cudaFuncSetAttribute(
                config.retain_backward_state ? training_kernel
                                             : inference_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(forward_compute_smem));
          };
      cudaError_t compute_attribute = cudaSuccess;
#if MEGAMOE_FWD_COMPILED_TUNING
#if MEGAMOE_FWD_FC2_TMA_STORE
#if MEGAMOE_FWD_CLUSTER_SIZE == 2
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
      else
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
      else
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>);
#endif
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                true, true, MEGAMOE_FWD_TILE_M == 64,
                ForwardTaskPartition::All, true, false, false,
                forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      else
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                false, true, MEGAMOE_FWD_TILE_M == 64,
                ForwardTaskPartition::All, true, false, false,
                forward_tuning::kFc1Stages, forward_tuning::kFc2Stages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
#endif
#else
      if (forward_specialize_stages)
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                false, true, false, ForwardTaskPartition::All, false, false,
                true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      else if (forward_cluster2)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<false, true, false, true>,
            local_p2p_forward_training_tma_store_kernel<false, false, true>);
      else if (forward_fc2_tma_store)
        compute_attribute = set_tma_compute_attribute(
            local_p2p_forward_tma_store_kernel<false, true>,
            local_p2p_forward_training_tma_store_kernel<false>);
      else if (forward_m64)
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<false, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      else if (forward_m64_head)
        compute_attribute = cudaFuncSetAttribute(
            local_p2p_forward_compute_kernel<
                false, true, false,
                ForwardTaskPartition::Body>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(forward_compute_smem));
      else {
        const auto set_mixed_compute_attribute = [&](auto kernel) {
          return cudaFuncSetAttribute(
              kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(forward_compute_smem));
        };
        if (forward_fc1_stages == 2U)
          compute_attribute = set_mixed_compute_attribute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 2U, 4U>);
        else if (forward_fc2_stages == 3U)
          compute_attribute = set_mixed_compute_attribute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 3U, 3U>);
        else if (forward_fc1_stages == 4U)
          compute_attribute = set_mixed_compute_attribute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 4U, 4U>);
        else
          compute_attribute = set_mixed_compute_attribute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 3U, 4U>);
      }
#endif
      internal::throw_cuda(
          "forward-run", compute_attribute,
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(queued FC1/FC2 dynamic smem)");
#if !MEGAMOE_FWD_COMPILED_TUNING
      if (forward_specialize_stages) {
        cudaError_t fc2_attribute = cudaSuccess;
        if (!forward_fc2_tma_store)
          fc2_attribute = cudaFuncSetAttribute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, true>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(forward_compute_smem));
        else
          fc2_attribute = cudaFuncSetAttribute(
              local_p2p_forward_tma_store_kernel<false, false, true>,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              static_cast<int>(forward_compute_smem));
        internal::throw_cuda(
            "forward-run", fc2_attribute, id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaFuncSetAttribute(FC2-only dynamic smem)");
      }
      if (forward_m64_head || forward_specialize_stages)
        internal::throw_cuda(
            "forward-run",
            cudaFuncSetAttribute(local_p2p_forward_m64_head_kernel<false>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(forward_compute_smem)),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaFuncSetAttribute(M64 head dynamic smem)");
      if (forward_m64_head)
        internal::throw_cuda(
            "forward-run",
            cudaFuncSetAttribute(
                local_p2p_forward_compute_kernel<
                    false, true, false, ForwardTaskPartition::Body, false,
                    true>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(forward_compute_smem)),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaFuncSetAttribute(M128 body FC2 tail dynamic smem)");
#endif

      internal::throw_cuda(
          "forward-run",
          cudaEventRecord(slot.forward_roles_ready, exec->stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaEventRecord(split Forward roles ready)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(slot.forward_transport_stream,
                              slot.forward_roles_ready),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport ready)");
#if !MEGAMOE_FWD_COMPILED_TUNING
      if (forward_m64_head || forward_specialize_stages)
        internal::throw_cuda(
            "forward-run",
            cudaStreamWaitEvent(slot.forward_m64_head_stream,
                                slot.forward_roles_ready),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaStreamWaitEvent(split Forward auxiliary compute ready)");
#endif
      internal::throw_cuda("forward-run",
                           cudaStreamWaitEvent(slot.forward_compute_stream,
                                               slot.forward_roles_ready),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaStreamWaitEvent(split Forward compute ready)");

      cudaLaunchConfig_t transport_launch{};
      transport_launch.gridDim = dim3(kTransportBlocks, 1, 1);
      transport_launch.blockDim =
          dim3(forward_tuning::kComputeThreads, 1, 1);
      transport_launch.dynamicSmemBytes = forward_transport_smem;
      transport_launch.stream = slot.forward_transport_stream;
      launch_forward =
          split_local_roles
              ? forward_transport_launch::launch_local(
                    &transport_launch, &args, sizeof(args), slot.forward_x_map,
                    slot.forward_s_map, weight.w1_map, weight.w2_map)
              : forward_transport_launch::launch(
                    &transport_launch, &args, sizeof(args), slot.forward_x_map,
                    slot.forward_s_map, weight.w1_map, weight.w2_map);
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           split_local_roles
                               ? "persistent 0-RDMA/tunable-Relay Forward launch"
                               : "persistent 8-RDMA/tunable-Relay Forward launch");
      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_transport_done,
                                           slot.forward_transport_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward transport done)");

#if !MEGAMOE_FWD_COMPILED_TUNING
      if (forward_m64_head) {
        auto head_args = args;
#if MEGAMOE_DEVICE_TIMELINE
        head_args.timeline.cta_base = kTransportBlocks;
#endif
        cudaLaunchConfig_t head_launch{};
        head_launch.gridDim = dim3(args.forward_m64_head_cta_count, 1, 1);
        head_launch.blockDim = dim3(384, 1, 1);
        head_launch.dynamicSmemBytes = forward_compute_smem;
        head_launch.stream = slot.forward_m64_head_stream;
        launch_forward = cudaLaunchKernelEx(
            &head_launch, local_p2p_forward_m64_head_kernel<false>, head_args,
            slot.forward_x_m64_map, slot.forward_s_m64_map, weight.w1_map,
            weight.w2_map);
        internal::throw_cuda(
            "forward-run", launch_forward, id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "concurrent M64 head FC1/FC2 WGMMA/TMA Forward launch");

        auto tail_args = args;
        tail_args.forward_fc1_cta_count = 0U;
#if MEGAMOE_DEVICE_TIMELINE
        // The tail runs after the head in this stream, so it can reuse the
        // head CTA timeline lanes without overlapping their records.
        tail_args.timeline.cta_base = kTransportBlocks;
#endif
        cudaLaunchConfig_t tail_launch{};
        tail_launch.gridDim = dim3(args.forward_m64_head_cta_count, 1, 1);
        tail_launch.blockDim = dim3(384, 1, 1);
        tail_launch.dynamicSmemBytes = forward_compute_smem;
        tail_launch.stream = slot.forward_m64_head_stream;
        launch_forward = cudaLaunchKernelEx(
            &tail_launch,
            local_p2p_forward_compute_kernel<
                false, true, false, ForwardTaskPartition::Body, false, true>,
            tail_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
        internal::throw_cuda(
            "forward-run", launch_forward, id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "post-head M128 body FC2 queue-consumer Forward launch");
        internal::throw_cuda(
            "forward-run",
            cudaEventRecord(slot.forward_m64_head_done,
                            slot.forward_m64_head_stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaEventRecord(split Forward M64 head done)");
      } else if (forward_specialize_stages) {
        const auto early_fc2_ctas =
            forward_main_compute_ctas - args.forward_fc1_cta_count;

        auto fc1_args = args;
        fc1_args.forward_reduce_cta_offset = 0U;
#if MEGAMOE_DEVICE_TIMELINE
        fc1_args.timeline.cta_base = kTransportBlocks;
#endif
        cudaLaunchConfig_t fc1_launch{};
        fc1_launch.gridDim = dim3(args.forward_fc1_cta_count, 1, 1);
        fc1_launch.blockDim = dim3(384, 1, 1);
        fc1_launch.dynamicSmemBytes = forward_compute_smem;
        fc1_launch.stream = slot.forward_m64_head_stream;
        launch_forward = cudaLaunchKernelEx(
            &fc1_launch,
            local_p2p_forward_compute_kernel<
                false, true, false, ForwardTaskPartition::All, false, false,
                true>,
            fc1_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
        internal::throw_cuda(
            "forward-run", launch_forward, id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "compiler-specialized FC1 WGMMA/TMA Forward launch");

        // This launch is stream-ordered after the FC1 grid.  The 36 early FC2
        // CTAs below keep FC1/FC2 overlapped; these CTAs refill each FC1 SM
        // once the balanced FC1 grid has drained, without carrying FC1 live
        // ranges into the FC2 kernel symbol.
        auto tail_args = args;
        tail_args.forward_reduce_cta_offset = early_fc2_ctas;
#if MEGAMOE_DEVICE_TIMELINE
        tail_args.timeline.cta_base = kTransportBlocks;
#endif
        cudaLaunchConfig_t tail_launch{};
        tail_launch.gridDim = dim3(args.forward_fc1_cta_count, 1, 1);
        tail_launch.blockDim = dim3(384, 1, 1);
        tail_launch.dynamicSmemBytes = forward_compute_smem;
        tail_launch.stream = slot.forward_m64_head_stream;
        if (forward_fc2_tma_store)
          launch_forward = cudaLaunchKernelEx(
              &tail_launch,
              local_p2p_forward_tma_store_kernel<false, false, true>,
              tail_args, slot.forward_s_map, slot.forward_s_map,
              weight.w2_map, weight.w2_map,
              slot.forward_staging_store_map);
        else
          launch_forward = cudaLaunchKernelEx(
              &tail_launch,
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, true>,
              tail_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
              weight.w2_map);
        internal::throw_cuda(
            "forward-run", launch_forward, id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "post-FC1 compiler-specialized FC2 Forward launch");
        internal::throw_cuda(
            "forward-run",
            cudaEventRecord(slot.forward_m64_head_done,
                            slot.forward_m64_head_stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaEventRecord(specialized FC1/tail-FC2 done)");
      }
#endif

      auto compute_args = args;
#if MEGAMOE_DEVICE_TIMELINE
      compute_args.timeline.cta_base =
          kTransportBlocks +
          (forward_m64_head ? args.forward_m64_head_cta_count : 0U);
#endif
      cudaLaunchConfig_t compute_launch{};
      compute_launch.gridDim = dim3(
#if MEGAMOE_FWD_COMPILED_TUNING
          forward_main_compute_ctas,
#else
          forward_specialize_stages
              ? forward_main_compute_ctas - args.forward_fc1_cta_count
              : forward_main_compute_ctas,
#endif
          1, 1);
      compute_launch.blockDim =
          dim3(forward_tuning::kComputeThreads, 1, 1);
      compute_launch.dynamicSmemBytes = forward_compute_smem;
      compute_launch.stream = slot.forward_compute_stream;
      cudaLaunchAttribute compute_cluster_attribute{};
      if (forward_cluster2) {
        compute_cluster_attribute.id = cudaLaunchAttributeClusterDimension;
        compute_cluster_attribute.val.clusterDim = {
            forward_tuning::kClusterSize, 1U, 1U};
        compute_launch.attrs = &compute_cluster_attribute;
        compute_launch.numAttrs = 1U;
      }
      const auto launch_tma_compute =
          [&](auto inference_kernel, auto training_kernel,
              const CUtensorMap &output_store_map) {
            const auto kernel = config.retain_backward_state
                                    ? training_kernel
                                    : inference_kernel;
            return cudaLaunchKernelEx(
                &compute_launch, kernel, compute_args, slot.forward_x_map,
                slot.forward_s_map, weight.w1_map, weight.w2_map,
                output_store_map);
          };
#if MEGAMOE_FWD_COMPILED_TUNING
#if MEGAMOE_FWD_FC2_TMA_STORE
      const auto &output_store_map =
          args.use_gin && args.forward_registered_output
              ? slot.forward_staging_store_map
              : ordinary_output_store_map;
#if MEGAMOE_FWD_CLUSTER_SIZE == 2
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
      else
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, true, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                true, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
      else
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<
                false, true, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            local_p2p_forward_training_tma_store_kernel<
                false, false, false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            output_store_map);
#endif
#elif MEGAMOE_FWD_TILE_M == 64
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                true, true, true, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_m64_map, slot.forward_s_m64_map,
            weight.w1_map, weight.w2_map);
      else
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                false, true, true, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_m64_map, slot.forward_s_m64_map,
            weight.w1_map, weight.w2_map);
#else
      if constexpr (forward_tuning::kRdmaCtas == 0U)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                true, true, false, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
      else
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                false, true, false, ForwardTaskPartition::All, true, false,
                false, forward_tuning::kFc1Stages,
                forward_tuning::kFc2Stages>,
            compute_args, slot.forward_x_map, slot.forward_s_map, weight.w1_map,
            weight.w2_map);
#endif
#else
      if (forward_specialize_stages) {
        compute_args.forward_reduce_cta_offset = 0U;
#if MEGAMOE_DEVICE_TIMELINE
        compute_args.timeline.cta_base =
            kTransportBlocks + args.forward_fc1_cta_count;
#endif
        if (forward_fc2_tma_store)
          launch_forward = cudaLaunchKernelEx(
              &compute_launch,
              local_p2p_forward_tma_store_kernel<false, false, true>,
              compute_args, slot.forward_s_map, slot.forward_s_map,
              weight.w2_map, weight.w2_map,
              slot.forward_staging_store_map);
        else
          launch_forward = cudaLaunchKernelEx(
              &compute_launch,
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, true>,
              compute_args, slot.forward_x_map, slot.forward_s_map,
              weight.w1_map, weight.w2_map);
      } else if (forward_cluster2) {
        const auto &output_store_map =
            args.use_gin && args.forward_registered_output
                ? slot.forward_staging_store_map
                : ordinary_output_store_map;
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<false, true, false, true>,
            local_p2p_forward_training_tma_store_kernel<false, false, true>,
            output_store_map);
      } else if (forward_fc2_tma_store) {
        const auto &output_store_map =
            args.use_gin && args.forward_registered_output
                ? slot.forward_staging_store_map
                : ordinary_output_store_map;
        launch_forward = launch_tma_compute(
            local_p2p_forward_tma_store_kernel<false, true>,
            local_p2p_forward_training_tma_store_kernel<false>,
            output_store_map);
      } else if (forward_m64)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<false, true, true>, compute_args,
            slot.forward_x_m64_map, slot.forward_s_m64_map, weight.w1_map,
            weight.w2_map);
      else if (forward_m64_head)
        launch_forward = cudaLaunchKernelEx(
            &compute_launch,
            local_p2p_forward_compute_kernel<
                false, true, false,
                ForwardTaskPartition::Body>,
            compute_args, slot.forward_x_map, slot.forward_s_map,
            weight.w1_map, weight.w2_map);
      else {
        const auto launch_mixed_compute = [&](auto kernel) {
          return cudaLaunchKernelEx(
              &compute_launch, kernel, compute_args, slot.forward_x_map,
              slot.forward_s_map, weight.w1_map, weight.w2_map);
        };
        if (forward_fc1_stages == 2U)
          launch_forward = launch_mixed_compute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 2U, 4U>);
        else if (forward_fc2_stages == 3U)
          launch_forward = launch_mixed_compute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 3U, 3U>);
        else if (forward_fc1_stages == 4U)
          launch_forward = launch_mixed_compute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 4U, 4U>);
        else
          launch_forward = launch_mixed_compute(
              local_p2p_forward_compute_kernel<
                  false, true, false, ForwardTaskPartition::All, true, false,
                  false, 3U, 4U>);
      }
#endif
      internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           "persistent queued FC1/FC2 WGMMA/TMA Forward launch");

      internal::throw_cuda("forward-run",
                           cudaEventRecord(slot.forward_compute_done,
                                           slot.forward_compute_stream),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventRecord(split Forward compute done)");
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_transport_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward transport done)");
#if !MEGAMOE_FWD_COMPILED_TUNING
      if (forward_m64_head || forward_specialize_stages)
        internal::throw_cuda(
            "forward-run",
            cudaStreamWaitEvent(exec->stream, slot.forward_m64_head_done),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "cudaStreamWaitEvent(split Forward auxiliary compute done)");
#endif
      internal::throw_cuda(
          "forward-run",
          cudaStreamWaitEvent(exec->stream, slot.forward_compute_done),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamWaitEvent(split Forward compute done)");
    } else {
      if (!remote_transport)
        transport_error(NotifyErrorCode::BackendUnavailable, "forward-run",
                        "single-node kernel", id.calling_rank,
                        "split transport/compute architecture", "disabled");
      internal::throw_cuda(
          "forward-run",
          cudaFuncSetAttribute(local_p2p_forward_megakernel<false>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(forward_smem)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaFuncSetAttribute(dynamic smem)");
      cudaLaunchAttribute attributes[1]{};
      attributes[0].id = cudaLaunchAttributeCooperative;
      attributes[0].val.cooperative = 1;
      cudaLaunchConfig_t launch{};
      launch.gridDim = dim3(grid_blocks, 1, 1);
      launch.blockDim = dim3(384, 1, 1);
      launch.dynamicSmemBytes = forward_smem;
      launch.stream = exec->stream;
      launch.attrs = attributes;
      launch.numAttrs = 1;
      internal::throw_cuda(
          "forward-run",
          cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                                 static_cast<int>(exec->device)),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaDevAttrClusterLaunch");
      const auto *forward_kernel = reinterpret_cast<const void *>(
          local_p2p_forward_megakernel<false>);
      internal::throw_cuda("forward-run",
                           cudaOccupancyMaxActiveClusters(
                               &max_active_clusters, forward_kernel, &launch),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaOccupancyMaxActiveClusters");
      launch_forward = cudaLaunchKernelEx(
          &launch, local_p2p_forward_megakernel<false>, args,
          slot.forward_x_map, slot.forward_s_map, weight.w1_map,
          weight.w2_map);
    }
#if MEGAMOE_DEBUG_VALIDATE
    debug_forward_host_phase("kernel_launch_returned");
#endif
    internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "single persistent WGMMA/TMA forward launch");
  if (use_socket) {
    std::vector<std::uint64_t> send_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    std::vector<std::uint64_t> receive_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      send_prefix[peer + 1U] =
          send_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      receive_prefix[peer + 1U] =
          receive_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
    }
    if (send_prefix[id.world_size] != view.sizes.route_count ||
        receive_prefix[id.world_size] != socket_incoming)
      transport_error(NotifyErrorCode::TransportProtocolFailure,
                      "socket-forward-return", "route counts",
                      id.calling_rank, "exact preload counts", "mismatch");
    auto *send_headers = at<SocketRouteHeader>(
        slot.device, layout.socket_header_send);
    auto *receive_headers = at<SocketRouteHeader>(
        slot.device, layout.socket_header_receive);
    auto *send_payload = at<transport::BFloat16>(
        slot.device, layout.contribution);
    auto *receive_payload = at<transport::BFloat16>(
        slot.device, layout.socket_payload_receive);
    auto *forward_output = at<transport::BFloat16>(
        slot.device, args.forward_output_rows_offset);
    if (socket_incoming != 0)
      socket_gather_forward_returns_kernel<<<
          static_cast<unsigned int>(socket_incoming), 256, 0,
          exec->stream>>>(receive_headers, forward_output, receive_payload,
                          socket_incoming, config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "gather canonical FC2 rows");
    internal::throw_nccl(
        "socket-forward-return", ncclGroupStart(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupStart(Socket FC return)");
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
      const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-forward-return", "self rows",
                          id.calling_rank, std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0)
          internal::throw_cuda(
              "socket-forward-return",
              cudaMemcpyAsync(
                  send_payload + send_prefix[peer] * config.hidden_size,
                  receive_payload +
                      receive_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket Forward self return bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket Forward self return");
        continue;
      }
      socket_return_peer_bytes = checked_add(
          socket_return_peer_bytes,
          checked_mul(checked_mul(send_rows, config.hidden_size,
                                  "Socket Forward remote return elements"),
                      sizeof(transport::BFloat16),
                      "Socket Forward remote return bytes"),
          "Socket Forward total remote return bytes");
      if (send_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclSend(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket Forward return)");
      if (receive_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket Forward return)");
    }
    internal::throw_nccl(
        "socket-forward-return", ncclGroupEnd(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupEnd(Socket Forward return)");
    if (view.sizes.route_count != 0)
      socket_scatter_returns_kernel<<<
          static_cast<unsigned int>(view.sizes.route_count), 256, 0,
          exec->stream>>>(send_headers, send_payload,
                          at<transport::BFloat16>(slot.device,
                                                  layout.combine_return),
                          view.sizes.route_count, config.hidden_size);
    if (id.num_tokens != 0)
      socket_sum_forward_routes_kernel<<<
          static_cast<unsigned int>(id.num_tokens), 256, 0, exec->stream>>>(
          view,
          at<transport::BFloat16>(slot.device, layout.combine_return),
          at<transport::BFloat16>(slot.device, layout.combined),
          config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "Socket scatter/unweighted reduce");
  }
  internal::throw_cuda(
      "forward-run",
      cudaEventRecord(slot.forward_output_ready, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(Forward output ready)");
  // Local P2P relies on the peer-visible terminal epoch. GIN payload and
  // return consumers have already acquired their per-generation signals;
  // the next Notify/reset agreement is the registered-window reuse boundary.
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.host_forward_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H forward status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.forward_complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward complete)");
  slot.completion_recorded = true;

#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_enter");
#endif
  std::atomic<bool> forward_stream_done{false};
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "forward-run",
      cudaLaunchHostFunc(exec->stream, publish_stream_completion,
                         &forward_stream_done),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaLaunchHostFunc(Forward completion notification)");
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_done");
#endif

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
  unsigned int progress_dump_count = 0U;
#endif
  while (true) {
    if (forward_stream_done.load(std::memory_order_acquire))
      break;
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >= options.deadline_after;
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
    const auto stalled_for = std::chrono::steady_clock::now() - wait_start;
    const auto progress_dump_after = progress_dump_count == 0U
                                         ? std::chrono::seconds(3)
                                         : std::chrono::seconds(30);
    if (progress_dump_count < 2U && use_gin && args.progress != nullptr &&
        stalled_for >= progress_dump_after) {
      constexpr std::size_t forward_progress_base = 0;
      const auto progress = [&](std::uint32_t word) {
        return __atomic_load_n(slot.host_forward_progress +
                                   forward_progress_base + word,
                               __ATOMIC_ACQUIRE);
      };
      // Emit the mapped coarse milestones before attempting a device-memory
      // snapshot. If a platform cannot schedule the diagnostic D2H while the
      // persistent grid is resident, this line still identifies the stalled
      // phase instead of losing the entire diagnostic.
      std::cerr << "FORWARD_PROGRESS_COARSE rank=" << id.calling_rank
                << " generation=" << id.generation
                << " sample=" << progress_dump_count
                << " entered=" << progress(kForwardEntered)
                << " setup=" << progress(kForwardSetupComplete)
                << " dispatch_stored=" << progress(kForwardDispatchStored)
                << " dispatch_observed="
                << progress(kForwardDispatchObserved)
                << " fc1=" << progress(kForwardFc1Complete)
                << " fc2=" << progress(kForwardFc2Complete)
                << " return=" << progress(kForwardReturnObserved)
                << " reduce=" << progress(kForwardReduceComplete)
                << " terminal=" << progress(kForwardTerminal) << std::endl;
      // Keep production completion atomics in device memory even in debug
      // builds. Redirecting the per-task counters to mapped host memory
      // perturbs the Forward schedule by hundreds of milliseconds and can
      // hide the race being diagnosed. A pair of copy-engine snapshots on the
      // independent cancel stream remains observable while the persistent
      // kernel occupies every SM.
      auto *completion_snapshot = slot.host_forward_progress +
                                  forward_progress_base + kForwardProgressWords;
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaMemcpyAsync(completion_snapshot,
                          at<void>(slot.device, layout.forward_completion),
                          sizeof(transport_state::GenerationCompletionVector),
                          cudaMemcpyDeviceToHost, state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "D2H Forward generation completion snapshot");
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaMemcpyAsync(
              completion_snapshot + transport_state::kGenerationCompletionWords,
              at<void>(slot.device, layout.forward_context_completion),
              static_cast<std::size_t>(slot.generation_debug_contexts) *
                  sizeof(transport_state::ContextCompletionRecord),
              cudaMemcpyDeviceToHost, state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "D2H Forward context completion snapshot");
      internal::throw_cuda(
          "forward-progress-snapshot",
          cudaStreamSynchronize(state->cancel_stream), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "drain Forward completion snapshot");
      std::cerr << "FORWARD_PROGRESS rank=" << id.calling_rank
                << " generation=" << id.generation
                << " sample=" << progress_dump_count;
      dump_generation_completion(std::cerr, completion_snapshot,
                                 slot.generation_debug_contexts,
                                 "forward_completion");
      std::cerr << std::endl;
      ++progress_dump_count;
    }
#endif
    if (!cancelled &&
        (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, args.epoch, __ATOMIC_RELEASE);
      internal::throw_cuda(
          "forward-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "H2D publish forward cancel");
      internal::throw_cuda(
          "forward-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "drain forward cancel stream");
      cancelled = true;
    }
    // Do not call CUDA event-query APIs while the resident GIN kernel is
    // active. On the remote host that can hold a driver-side progress lock and
    // form a liveness cycle with device GIN. The stream callback above is the
    // completion notification; this sleep only bounds host polling overhead.
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("stream_completion_observed");
#endif
  internal::throw_cuda(
      "forward-run", cudaEventQuery(slot.forward_complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventQuery(completed Forward event)");
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("event_verified");
#endif
  consumer.finish();
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_finished");
  debug_forward_host_phase("terminal_agreement_enter");
#endif
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size, *slot.host_forward_status,
      slot.forward_complete);
#if MEGAMOE_DEBUG_VALIDATE
  debug_forward_host_phase("terminal_agreement_done");
#endif
  if (cancelled || slot.host_forward_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=forward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed forward, actual=cancelled");
  if (production_agreement.has_error) {
    // A logical device-status failure reached the fixed terminal agreement;
    // all peer work is drained and this slot is recoverable. Backend/CUDA
    // failures still leave the guard uncommitted and therefore quarantine.
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  internal::throw_cuda(
      "forward-timing",
      cudaEventElapsedTime(&total_ms, slot.forward_start,
                           slot.forward_output_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(forward total)");
  auto result = std::make_shared<DeviceForwardHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->backend = backend;
  result->dispatch = dispatch;
  result->activations = activations;
  result->weights = weights;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->retained_backward_state = config.retain_backward_state;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->audit.specialization =
      ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions =
      slot.host_forward_status->wgmma_instructions;
  result->audit.tma_loads = slot.host_forward_status->tma_loads;
  result->audit.tma_multicast_loads =
      slot.host_forward_status->tma_multicast_loads;
  result->audit.runtime_cluster_supported = cluster_supported != 0;
  result->audit.requested_cluster_size = 2;
  result->audit.runtime_cluster_size =
      slot.host_forward_status->runtime_cluster_size;
  result->audit.observed_cluster_rank_mask =
      slot.host_forward_status->cluster_rank_mask;
  result->audit.max_active_clusters =
      static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
  result->audit.fc1_tasks = slot.host_forward_status->fc1_tasks_done;
  result->audit.fc2_tasks = slot.host_forward_status->fc2_tasks_done;
  result->audit.actual_dispatch_peer_bytes =
      use_socket ? socket_dispatch_peer_bytes
                 : slot.host_forward_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      use_socket ? socket_return_peer_bytes
                 : slot.host_forward_status->actual_return_peer_bytes;
  result->audit.gin_dispatch_put_count =
      slot.host_forward_status->gin_dispatch_put_count;
  result->audit.raw_cross_node_route_bytes =
      use_socket ? socket_raw_cross_node_route_bytes
                 : slot.host_forward_status->raw_cross_node_payload_bytes;
  result->audit.gin_dispatch_payload_bytes =
      slot.host_forward_status->deduplicated_cross_node_payload_bytes;
  result->audit.gin_return_payload_bytes =
      slot.host_forward_status->gin_return_payload_bytes;
  result->audit.gin_return_put_count =
      slot.host_forward_status->gin_return_put_count;
  result->audit.gin_signal_count =
      slot.host_forward_status->gin_signal_count;
  result->audit.gin_metadata_bytes =
      slot.host_forward_status->gin_metadata_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.terminal_status_agreement_us =
      production_agreement.elapsed_us;
  const auto &status = *slot.host_forward_status;
  const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
    return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
  };
  result->timings.dispatch_us =
      elapsed(status.start_clock, status.dispatch_complete_clock);
  result->timings.fc1_swiglu_us =
      elapsed(status.fc1_start_clock, status.fc1_complete_clock);
  result->timings.fc2_us =
      elapsed(status.dispatch_blocks_done, status.fc2_complete_clock);
  result->timings.combine_return_us =
      elapsed(status.relay_blocks_done, status.return_complete_clock);
  result->timings.reduce_us =
      elapsed(status.reduced_tokens, status.reduce_blocks_done);
  result->timings.fc1_fc2_overlap_us =
      elapsed(status.dispatch_blocks_done, status.fc1_complete_clock);
  result->timings.fc2_return_overlap_us =
      elapsed(status.relay_blocks_done, status.fc2_complete_clock);
  result->timings.return_reduce_overlap_us =
      elapsed(status.reduced_tokens, status.return_complete_clock);
  failure_guard.committed = true;
  return DeviceForwardHandle(std::move(result));
}

const ActivationTransportExpectation &DeviceForwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->identity;
}

const ForwardMegaKernelAudit &DeviceForwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->timings;
}

std::vector<transport::BFloat16> download_forward_output_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "forward download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward download elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16),
                    "forward download bytes"),
        "forward download bytes");
    internal::throw_cuda(
        "forward-download",
        cudaMemcpy(output.data(),
                   at<const void>(impl.pool->slots[impl.slot->index].device,
                                  impl.pool->layout.combined),
                   bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(forward output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

ForwardBackwardStateValidationDownload
download_forward_backward_state_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-backward-state-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (!impl.retained_backward_state ||
      impl.pool->layout.forward_ab_cache == 0 ||
      impl.pool->layout.forward_training_output == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-backward-state-download", "retained state",
                    impl.identity.rank, "enabled", "disabled");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-backward-state-download", "device",
                    impl.identity.rank, "handle device", "foreign");

  ForwardBackwardStateValidationDownload result;
  const auto x_elements = checked_mul(
      impl.incoming_rows, impl.identity.hidden,
      "forward retained canonical X elements");
  const auto plane_elements = checked_mul(
      impl.incoming_rows, impl.intermediate,
      "forward retained A/B plane elements");
  result.canonical_x.resize(
      checked_size(x_elements, "forward retained canonical X elements"));
  result.fc1_ab.resize(checked_size(
      checked_mul(plane_elements, 2ULL, "forward retained A/B elements"),
      "forward retained A/B elements"));
  if (impl.incoming_rows == 0)
    return result;

  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto copy = [&](void *destination, const void *source,
                        std::uint64_t elements, const char *name) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16), name), name);
    internal::throw_cuda(
        "forward-backward-state-download",
        cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.canonical_x.data(),
       at<const void>(slot.device, impl.pool->layout.canonical_rows),
       x_elements, "cudaMemcpy(retained canonical X)");
  const auto *ab = at<const transport::BFloat16>(
      slot.device, impl.pool->layout.forward_ab_cache);
  copy(result.fc1_ab.data(), ab, plane_elements,
       "cudaMemcpy(retained FC1 A)");
  const auto b_plane_stride = checked_mul(
      impl.pool->config.max_destination_rows_per_rank, impl.intermediate,
      "forward retained B plane stride");
  copy(result.fc1_ab.data() + plane_elements, ab + b_plane_stride,
       plane_elements, "cudaMemcpy(retained FC1 B)");
  return result;
}
const ActivationTransportExpectation &
DeviceActivationTransportHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const ActivationTransportDeviceAudit &
DeviceActivationTransportHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const ActivationTransportTimings &
DeviceActivationTransportHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ActivationTransportValidationDownload ActivationTransportBridge::download_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto hidden = impl.identity.hidden;
  const auto route_elements = checked_mul(impl.route_count, hidden,
                                          "download route elements");
  const auto combined_elements = checked_mul(impl.identity.tokens, hidden,
                                             "download combined elements");
  const auto canonical_elements =
      checked_mul(impl.destination_rows, hidden, "download canonical elements");
  ActivationTransportValidationDownload result;
  result.contribution_by_route.resize(
      checked_size(route_elements, "download route elements"));
  result.combined_tokens.resize(
      checked_size(combined_elements, "download combined elements"));
  result.canonical_expert_rows.resize(
      checked_size(canonical_elements, "download canonical elements"));
  auto copy = [&](void *destination, std::size_t offset, std::uint64_t count,
                  const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-download",
        cudaMemcpy(destination, at<const void>(slot.device, offset),
                   checked_size(checked_mul(count,
                                            sizeof(transport::BFloat16), name),
                                name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  const auto contribution_offset =
      impl.audit.backend ==
                  transport::ActivationTransportBackend::NcclSocketFallback ||
              impl.audit.backend ==
                  transport::ActivationTransportBackend::Gin
          ? impl.pool->layout.combine_return
          : impl.pool->layout.contribution;
  copy(result.contribution_by_route.data(), contribution_offset,
       route_elements, "download contributions");
  copy(result.combined_tokens.data(), impl.pool->layout.combined,
       combined_elements, "download combined");
  copy(result.canonical_expert_rows.data(), impl.pool->layout.canonical_rows,
       canonical_elements, "download canonical rows");
  result.protocol.dispatch_packets = slot.host_status->packets_done;
  result.protocol.relayed_fanout = slot.host_status->fanout_done;
  result.protocol.synthetic_contributions = slot.host_status->synthetic_done;
  result.protocol.combine_packets = slot.host_status->combine_done;
  result.protocol.reduced_tokens = slot.host_status->reduced_tokens;
  result.protocol.credit_stalls = slot.host_status->credit_stalls;
  result.protocol.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result.protocol.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  return result;
}

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
}

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "transport-output-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "transport-output-download", "device",
                    impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "output download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "output download elements"));
  if (!output.empty()) {
    const auto bytes = checked_mul(elements,
                                   sizeof(transport::BFloat16),
                                   "output download bytes");
    internal::throw_cuda(
        "transport-output-download",
        cudaMemcpy(output.data(),
                   at<const void>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.combined),
                   checked_size(bytes, "output download bytes"),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(final output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}
