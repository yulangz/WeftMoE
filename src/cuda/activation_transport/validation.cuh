#pragma once

// Implementation fragment included only by activation_transport.cu, inside
// weftmoe::cuda, after its shared state and kernel definitions. Keeping the
// same translation unit preserves private types and device compilation.

// Reference transport execution, result downloads, and link diagnostics.

DeviceActivationTransportHandle
DeviceActivationTransport::run_device_simulation_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  const auto &id = dispatch.identity();
  auto state = impl_->pool.impl_->state;
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  const auto &activation = *activations.impl_;
  if (activation.prequantized)
    throw std::invalid_argument("Prequantized inputs require the FP8 PF consumer");
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
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
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "transport-run",
                    "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement != transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "transport-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");
  const auto expectation = DeviceDispatchViewExpectation{
      id.generation,         id.calling_rank, id.world_size,
      id.num_tokens,         id.top_k,        id.config_digest_word0,
      id.config_digest_word1};
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
  const auto route_elements =
      checked_mul(view.sizes.route_count, config.hidden_size, "route elements");
  const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size, "combined elements");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size, impl_->session.identity(),
        id.generation, ExchangeKind::ExpertCounts, name);
  };
  internal::throw_cuda(
      "transport-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, ring_slots,
             sizeof(transport::TransportSlotHeader), "ring headers");
  zero_async(layout.ring_ready_sequence, ring_slots, sizeof(std::uint64_t),
             "ring sequences");
  zero_async(layout.link_next_sequence, links, sizeof(std::uint64_t),
             "link sequences");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((links + 255U) / 256U), 256, 0, exec->stream>>>(
      at<unsigned int>(slot.device, layout.link_credits), links,
      config.ring_slots_per_source_channel);
  internal::throw_cuda("transport-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "initialize transport credits kernel");
  zero_async(layout.packet_claim, view.sizes.packet_count,
             sizeof(std::uint32_t), "packet claims");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "canonical elements"),
             sizeof(transport::BFloat16), "canonical rows");
  zero_async(layout.contribution, route_elements, sizeof(transport::BFloat16),
             "contributions");
  zero_async(layout.combine_return, route_elements, sizeof(transport::BFloat16),
             "combine return");
  zero_async(layout.combined, combined_elements, sizeof(transport::BFloat16),
             "combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "status");
  internal::throw_cuda(
      "transport-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(start)");
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
  args.ring_headers =
      at<transport::TransportSlotHeader>(slot.device, layout.ring_headers);
  args.ring_payload = at<transport::BFloat16>(slot.device, layout.ring_payload);
  args.ring_ready =
      at<unsigned long long>(slot.device, layout.ring_ready_sequence);
  args.link_next =
      at<unsigned long long>(slot.device, layout.link_next_sequence);
  args.link_credits = at<unsigned int>(slot.device, layout.link_credits);
  args.packet_claim = at<unsigned int>(slot.device, layout.packet_claim);
  args.canonical_rows =
      at<transport::BFloat16>(slot.device, layout.canonical_rows);
  args.contribution = at<transport::BFloat16>(slot.device, layout.contribution);
  args.combine_return =
      at<transport::BFloat16>(slot.device, layout.combine_return);
  args.combined = at<transport::BFloat16>(slot.device, layout.combined);
  args.status = at<TransportDeviceStatus>(slot.device, layout.status);
  args.identity_expert =
      config.expert_transform == transport::ActivationExpertTransform::Identity;
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
  internal::throw_cuda("transport-run",
                       cudaMemcpyAsync(slot.host_status, args.status,
                                       sizeof(TransportDeviceStatus),
                                       cudaMemcpyDeviceToHost, exec->stream),
                       id.calling_rank, id.world_size,
                       impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts, "D2H transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "transport-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "transport-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(transport complete)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "transport-kernel", "descriptor", id.calling_rank,
                    "valid packet/fanout",
                    "device validation error packet=" +
                        std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda(
      "transport-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(total)");
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

DeviceActivationTransportHandle
LocalCudaP2pActivationTransport::run_transport_slice(
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
  if (activation.prequantized)
    throw std::invalid_argument("Prequantized inputs require the FP8 PF consumer");
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node and exact local NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) +
            " nodes, world=" + std::to_string(id.world_size));
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
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "local-p2p-run",
                    "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement != transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "local-p2p-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation,         id.calling_rank, id.world_size,
      id.num_tokens,         id.top_k,        id.config_digest_word0,
      id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  [[maybe_unused]] const auto route_elements = checked_mul(
      view.sizes.route_count, config.hidden_size, "P2P route elements");
  [[maybe_unused]] const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size, "P2P combined elements");
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
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts, name);
    ++generation_memset_calls;
    generation_memset_bytes += bytes;
  };
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested() ? id.generation
                                                                 : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "local-p2p-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  // The integrated timing starts after the router-owned input is ready and
  // includes every generation-specific transport initialization below.
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "cudaEventRecord(P2P start)");
#if WEFTMOE_DEBUG_VALIDATE
  zero_async(layout.ring_headers, node_ring_slots,
             sizeof(transport::TransportSlotHeader), "P2P proxy headers");
  zero_async(
      layout.ring_payload,
      checked_mul(node_ring_slots, config.hidden_size, "P2P proxy payload"),
      sizeof(transport::BFloat16), "P2P proxy payload");
  zero_async(layout.ring_ready_sequence, node_ring_slots, sizeof(std::uint64_t),
             "P2P proxy ready");
  zero_async(layout.link_next_sequence, node_links, sizeof(std::uint64_t),
             "P2P proxy sequence");
  zero_async(layout.ingress_headers, ingress_slots, sizeof(LocalP2pRouteHeader),
             "P2P ingress headers");
  zero_async(layout.ingress_ready_sequence, ingress_slots,
             sizeof(std::uint64_t), "P2P ingress ready");
  zero_async(layout.ingress_next_sequence, rank_links, sizeof(std::uint64_t),
             "P2P ingress next");
  zero_async(layout.ingress_published_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress published");
  zero_async(layout.outgoing_route_sequence, rank_links, sizeof(std::uint64_t),
             "P2P outgoing sequence");
  zero_async(layout.direct_route_headers,
             state->config.max_destination_rows_per_rank,
             sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  zero_async(layout.return_ready_epoch, view.sizes.route_count,
             sizeof(std::uint64_t), "P2P return ready");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "P2P canonical rows"),
             sizeof(transport::BFloat16), "P2P canonical rows");
  zero_async(layout.contribution, route_elements, sizeof(transport::BFloat16),
             "P2P contributions");
  zero_async(layout.combine_return, route_elements, sizeof(transport::BFloat16),
             "P2P combine return");
  zero_async(layout.combined, combined_elements, sizeof(transport::BFloat16),
             "P2P combined");
#else
  // The trusted single-node bulk path overwrites every canonical row,
  // contribution and output element that it later reads. Epoch-valued peer
  // signals make all arena and signal clears unnecessary. Clearing a peer
  // signal here would race a faster rank's release store.
#endif
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "P2P status");
#if WEFTMOE_DEBUG_VALIDATE
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((node_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits),
                      node_links, config.ring_slots_per_source_channel);
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((rank_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.ingress_credits),
                      rank_links, config.ring_slots_per_source_channel);
  internal::throw_cuda("local-p2p-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "initialize P2P credits");

  // A fixed-width device AllGather is a setup barrier only. It exchanges no
  // activation and ensures every peer ring/credit/sequence region is fully
  // initialized before any persistent worker publishes to it.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, activation.slot->epoch);
  internal::throw_cuda("local-p2p-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(P2P setup barrier)");
  validate_p2p_barrier_kernel<<<1, 256, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive),
      id.world_size, activation.slot->index, activation.slot->epoch,
      at<TransportDeviceStatus>(slot.device, layout.status));
  internal::throw_cuda("local-p2p-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "validate P2P setup barrier");
#else
  // This small collective is a real per-generation dependency: peer writes
  // must not begin until every process has reached the transport phase. It
  // carries only 16 bytes/rank and replaces the old multi-gigabyte clears.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, id.generation);
  ++kernel_launches;
  internal::throw_cuda("local-p2p-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(P2P setup barrier)");
#endif
  int sms = 0;
  internal::throw_cuda(
      "local-p2p-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaDeviceGetAttribute(SM count)");
  if (sms <= static_cast<int>(id.world_size))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=resident_workers, rank=" +
            std::to_string(id.calling_rank) +
            ", expected>world=" + std::to_string(id.world_size) +
            ", actual_sms=" + std::to_string(sms));
  LocalP2pKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
#if WEFTMOE_DEBUG_VALIDATE
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
  args.cancel_epoch =
      reinterpret_cast<const unsigned long long *>(slot.device_cancel_epoch);
  args.identity_expert =
      config.expert_transform == transport::ActivationExpertTransform::Identity;
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P generation setup complete)");
#if WEFTMOE_DEBUG_VALIDATE
  local_p2p_persistent_kernel<<<sms, 128, 0, exec->stream>>>(args);
  ++kernel_launches;
#else
  local_p2p_bulk_dispatch_kernel<<<static_cast<unsigned int>(
                                       std::max<std::uint64_t>(
                                           view.sizes.route_count, 1ULL)),
                                   256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_return_kernel<<<static_cast<unsigned int>(
                                     std::max<std::uint64_t>(
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
  internal::throw_cuda("local-p2p-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "local P2P persistent kernel launch");
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H P2P transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "local-p2p-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda("local-p2p-run", query, id.calling_rank,
                           id.world_size, backend->session.session().identity(),
                           id.generation, ExchangeKind::ExpertCounts,
                           "cudaEventQuery(P2P complete)");
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >= options.deadline_after;
    if (!cancelled &&
        (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, id.generation, __ATOMIC_RELEASE);
      internal::throw_cuda(
          "local-p2p-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "H2D publish cancel epoch");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                sizeof(std::uint64_t));
      internal::throw_cuda(
          "local-p2p-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "cudaStreamSynchronize(cancel)");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  // Debug/cancellation paths retain the explicit team drain. In the trusted
  // release path every source waits for a generation-valued return signal
  // from every destination before producing output. The next Notify control
  // collective is the natural inter-generation barrier.
#if WEFTMOE_DEBUG_VALIDATE
  internal::throw_nccl(
      "local-p2p-drain",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(P2P terminal drain)");
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
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "ncclAllGather(P2P cancellation drain)");
    internal::throw_cuda("local-p2p-drain", cudaStreamSynchronize(exec->stream),
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
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(P2P total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->local_p2p_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend = transport::ActivationTransportBackend::LocalCudaP2p;
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
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(P2P generation setup)");
  result->timings.generation_setup_us = static_cast<double>(setup_ms) * 1000.0;
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
    result->timings.combine_reduce_us =
        static_cast<double>(post_dispatch >= relay_return
                                ? post_dispatch - relay_return
                                : 0ULL) /
        1000.0;
  }
  const auto kernel_us =
      complete_clock >= start_clock
          ? static_cast<double>(complete_clock - start_clock) / 1000.0
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
  if (activation.prequantized)
    throw std::invalid_argument("Prequantized inputs require the FP8 PF consumer");
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
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
  if (config.gate_placement != transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(NotifyErrorCode::TransportTimeout,
                              "phase=socket-run, field=cancellation, rank=" +
                                  std::to_string(id.calling_rank) +
                                  ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation,         id.calling_rank, id.world_size,
      id.num_tokens,         id.top_k,        id.config_digest_word0,
      id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  auto *local_counts =
      at<unsigned long long>(slot.device, layout.terminal_epoch);
  auto *count_matrix =
      at<unsigned long long>(slot.device, layout.socket_count_matrix);
  auto *prefix =
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive);
  auto *route_ids = at<unsigned int>(slot.device, layout.packet_claim);
  auto *send_headers =
      at<SocketRouteHeader>(slot.device, layout.socket_header_send);
  auto *receive_headers =
      at<SocketRouteHeader>(slot.device, layout.socket_header_receive);
  auto *send_payload =
      at<transport::BFloat16>(slot.device, layout.contribution);
  auto *receive_payload =
      at<transport::BFloat16>(slot.device, layout.socket_payload_receive);
  auto *canonical = at<transport::BFloat16>(slot.device, layout.canonical_rows);
  auto *by_route = at<transport::BFloat16>(slot.device, layout.combine_return);
  auto *combined = at<transport::BFloat16>(slot.device, layout.combined);
  auto *status = at<TransportDeviceStatus>(slot.device, layout.status);
  const auto world = id.world_size;
  const auto route_count = view.sizes.route_count;

  internal::throw_cuda(
      "socket-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(Socket start)");
  internal::throw_cuda(
      "socket-run",
      cudaMemsetAsync(local_counts, 0, world * sizeof(std::uint64_t),
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
    socket_route_count_kernel<<<static_cast<unsigned int>(
                                    (route_count + 255ULL) / 256ULL),
                                256, 0, exec->stream>>>(view, local_counts);
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
  internal::throw_cuda("socket-run", cudaGetLastError(), id.calling_rank, world,
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
      id.generation, ExchangeKind::ExpertCounts, "D2H Socket route counts");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            count_bytes);
  internal::throw_cuda(
      "socket-run", cudaStreamSynchronize(exec->stream), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
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
              id.calling_rank, world, backend->session.session().identity(),
              id.generation, ExchangeKind::ExpertCounts,
              "D2D Socket self headers");
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(
                  receive_payload + receive_prefix[peer] * config.hidden_size,
                  send_payload + send_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket self payload bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world, backend->session.session().identity(),
              id.generation, ExchangeKind::ExpertCounts,
              "D2D Socket self payload");
        }
        continue;
      }
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_headers + send_prefix[peer],
                     send_rows * sizeof(*send_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm, exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "ncclSend(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_payload + send_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "ncclSend(Socket payload)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_headers + receive_prefix[peer],
                     receive_rows * sizeof(*receive_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm, exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "ncclRecv(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(
                receive_payload + receive_prefix[peer] * config.hidden_size,
                receive_rows * config.hidden_size * sizeof(transport::BFloat16),
                ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "ncclRecv(Socket payload)");
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
    socket_relay_transport_kernel<<<static_cast<unsigned int>(incoming), 256, 0,
                                    exec->stream>>>(
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
          ncclSend(receive_payload + receive_prefix[peer] * config.hidden_size,
                   send_rows * config.hidden_size * sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts, "ncclSend(Socket return)");
    if (receive_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                   receive_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts, "ncclRecv(Socket return)");
  }
  internal::throw_nccl("socket-return", ncclGroupEnd(), id.calling_rank, world,
                       backend->session.session().identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "ncclGroupEnd(Socket return)");
  if (route_count != 0)
    socket_scatter_returns_kernel<<<static_cast<unsigned int>(route_count), 256,
                                    0, exec->stream>>>(
        send_headers, send_payload, by_route, route_count, config.hidden_size);
  if (id.num_tokens != 0)
    socket_gate_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens), 256,
                                0, exec->stream>>>(
        view, ActivationTransportBridge::gate_pointer(dispatch), by_route,
        combined, config.hidden_size);
  socket_finalize_status_kernel<<<1, 1, 0, exec->stream>>>(
      status, route_count, incoming, id.num_tokens);
  internal::throw_cuda("socket-run", cudaGetLastError(), id.calling_rank, world,
                       backend->session.session().identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "Socket relay/return/reduce kernels");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_status, status, sizeof(*status),
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H Socket status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(*status));
  internal::throw_cuda(
      "socket-run",
      internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "socket-run", cudaEventSynchronize(slot.complete), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
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
  result->audit.derived_route_input_bytes = route_count * sizeof(std::int32_t);
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
  result->timings.dispatch_relay_us = static_cast<double>(dispatch_ms) * 1000.0;
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
  if (activation.prequantized)
    throw std::invalid_argument("Prequantized inputs require the FP8 PF consumer");
  if (state->config.static_forward_window_tokens)
    throw std::invalid_argument("static ring workspace supports Forward only");
  if (id.num_nodes != backend->num_nodes ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=runtime GIN team topology and exact rank, actual=" +
            std::to_string(id.num_nodes) +
            " nodes, world=" + std::to_string(id.world_size));
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
                    "within registered GIN pool capacity", "exceeded/disabled");
  if (config.gate_placement != transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(NotifyErrorCode::TransportTimeout,
                              "phase=gin-run, field=cancellation, rank=" +
                                  std::to_string(id.calling_rank) +
                                  ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation,         id.calling_rank, id.world_size,
      id.num_tokens,         id.top_k,        id.config_digest_word0,
      id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (dispatch.received_row_capacity() >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "GIN route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");
  SlotFailureGuard failure_guard{activation.slot, false};
  const auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  internal::throw_cuda(
      "gin-run",
      internal::wait_input_ready(exec->stream, slot.producer_ready,
                                 slot.producer_capture_id),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN start)");
  internal::throw_cuda(
      "gin-run",
      cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                      sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN status)");
  internal::throw_cuda(
      "gin-run",
      cudaMemsetAsync(at<void>(slot.gin_window, state->gin_layout.barrier), 0,
                      sizeof(std::uint64_t), exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(GIN reset barrier)");
  gin_signal_reset_kernel<<<backend->gin_allocated_context_count, 1, 0,
                            exec->stream>>>(
      backend->dev_comm, backend->gin_allocated_context_count);
  internal::throw_cuda("gin-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "GIN reset kernel launch");
  auto *barrier = at<std::uint64_t>(slot.gin_window, state->gin_layout.barrier);
  internal::throw_nccl("gin-run",
                       ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                                     backend->nccl->comm, exec->stream),
                       id.calling_rank, id.world_size,
                       backend->session.session().identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "ncclAllReduce(after GIN reset)");

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
      config.expert_transform == transport::ActivationExpertTransform::Identity;
  gin_transport_slice_kernel<<<1, 256, 0, exec->stream>>>(args);
  internal::throw_cuda("gin-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "GIN activation persistent kernel launch");
  internal::throw_cuda(
      "gin-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H GIN status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "gin-run", internal::record_observable_event(slot.complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(GIN complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "gin-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(GIN complete)");
  // A rank may destroy its registered window immediately after its result and
  // transport object leave scope. Rendezvous only after every device worker
  // has completed all remote puts so no peer can deregister early.
  internal::throw_nccl("gin-run",
                       ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                                     backend->nccl->comm, exec->stream),
                       id.calling_rank, id.world_size,
                       backend->session.session().identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "ncclAllReduce(GIN terminal window lease)");
  internal::throw_cuda("gin-run", cudaStreamSynchronize(exec->stream),
                       id.calling_rank, id.world_size,
                       backend->session.session().identity(), id.generation,
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
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(GIN total)");
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

const transport::BFloat16 *
forward_output_device_pointer_for_testing(const DeviceForwardHandle &handle) {
  return static_cast<const transport::BFloat16 *>(
      handle.impl_->tensors.impl_->output->data);
}

std::vector<transport::BFloat16>
download_forward_output_for_validation(const DeviceForwardHandle &handle,
                                       const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "forward-download",
      cudaEventSynchronize(impl.tensors.impl_->completion.ready),
      impl.identity.rank, impl.identity.world, "forward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Forward output");
  const auto elements = checked_mul(impl.identity.tokens, impl.identity.hidden,
                                    "forward download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward download elements"));
  if (!output.empty()) {
    const auto bytes =
        checked_size(checked_mul(elements, sizeof(transport::BFloat16),
                                 "forward download bytes"),
                     "forward download bytes");
    internal::throw_cuda(
        "forward-download",
        cudaMemcpy(output.data(), impl.tensors.impl_->output->data, bytes,
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(forward output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

std::vector<transport::BFloat16>
download_forward_routes_for_validation(const DeviceForwardHandle &handle,
                                       const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-routes-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (impl.pool->layout.static_ring.depth)
    throw std::invalid_argument("static ring Forward does not retain a full "
                                "route contribution snapshot");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-routes-download", "device", impl.identity.rank,
                    "handle device", "foreign");
  const auto check = [&](cudaError_t status) {
    internal::throw_cuda("forward-routes-download", status, impl.identity.rank,
                         impl.identity.world, "forward-validation",
                         impl.identity.generation, ExchangeKind::ExpertCounts,
                         "snapshot route contributions");
  };
  check(cudaEventSynchronize(impl.tensors.impl_->completion.ready));
  const auto elements = checked_mul(
      checked_mul(impl.identity.tokens, impl.dispatch.identity().top_k,
                  "forward route count"),
      impl.identity.hidden, "forward route elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "route elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(output.front()), "forward route bytes"),
        "route bytes");
    check(cudaMemcpy(output.data(),
                     at<void>(impl.pool->slots[impl.workspace_index].device,
                              impl.pool->layout.contribution),
                     bytes, cudaMemcpyDeviceToHost));
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  if (impl.compact_return && !output.empty()) {
    // Preserve the route-major validation contract independently of the
    // production payload layout. Metadata and CPU gathering are debug-only.
    const auto &id = impl.dispatch.identity();
    const DeviceDispatchViewExpectation expected{
        id.generation,         id.calling_rank, id.world_size,
        id.num_tokens,         id.top_k,        id.config_digest_word0,
        id.config_digest_word1};
    auto consumer = impl.dispatch.acquire_consumer(execution, expected);
    const auto view = ActivationTransportBridge::view(consumer);
    const auto routes = impl.identity.tokens * id.top_k;
    std::vector<std::uint32_t> experts(routes);
    std::vector<std::uint64_t> offsets(routes), prefixes(id.num_experts);
    const auto copy_metadata = [&](auto &destination, const auto *source) {
      const auto bytes = destination.size() * sizeof(destination.front());
      check(cudaMemcpy(destination.data(), source, bytes,
                       cudaMemcpyDeviceToHost));
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                bytes);
    };
    copy_metadata(experts, view.expert_id);
    copy_metadata(offsets, view.local_offset);
    copy_metadata(prefixes, view.return_global_expert_base);
    consumer.finish();
    std::vector<transport::BFloat16> ordered(output.size());
    for (std::uint64_t route = 0; route < routes; ++route) {
      if (experts[route] >= prefixes.size())
        throw std::runtime_error(
            "invalid compact Return expert in validation snapshot");
      const auto row = prefixes[experts[route]] + offsets[route];
      if (row >= routes)
        throw std::runtime_error(
            "invalid compact Return row in validation snapshot");
      std::copy_n(output.data() + row * impl.identity.hidden,
                  impl.identity.hidden,
                  ordered.data() + route * impl.identity.hidden);
    }
    return ordered;
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
  if (!impl.retained_backward_state || !impl.tensors)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-backward-state-download", "retained state",
                    impl.identity.rank, "enabled", "disabled");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-backward-state-download", "device",
                    impl.identity.rank, "handle device", "foreign");

  ForwardBackwardStateValidationDownload result;
  const auto x_elements = checked_mul(impl.incoming_rows, impl.identity.hidden,
                                      "forward retained canonical X elements");
  const auto plane_elements =
      checked_mul(impl.incoming_rows, impl.intermediate,
                  "forward retained A/B plane elements");
  result.canonical_x.resize(
      checked_size(x_elements, "forward retained canonical X elements"));
  result.fc1_ab.resize(checked_size(
      checked_mul(plane_elements, 2ULL, "forward retained A/B elements"),
      "forward retained A/B elements"));
  if (impl.incoming_rows == 0)
    return result;

  const auto &saved = *impl.tensors.impl_;
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
  if (saved.x)
    copy(result.canonical_x.data(), saved.x->data, x_elements,
         "cudaMemcpy(retained canonical X)");
  else
    result.canonical_x.clear(); // FP8-only tape has no exact BF16 X to download.
  if (saved.fp8_ab_checkpoint) {
    result.fc1_ab.clear(); // This API downloads BF16 storage only.
    return result;
  }
  const auto *ab = static_cast<const transport::BFloat16 *>(saved.ab->data);
  copy(result.fc1_ab.data(), ab, plane_elements, "cudaMemcpy(retained FC1 A)");
  const auto b_plane_stride = checked_mul(saved.rows, impl.intermediate,
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

ActivationTransportValidationDownload
ActivationTransportBridge::download_for_validation(
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
  const auto route_elements =
      checked_mul(impl.route_count, hidden, "download route elements");
  const auto combined_elements =
      checked_mul(impl.identity.tokens, hidden, "download combined elements");
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
        cudaMemcpy(
            destination, at<const void>(slot.device, offset),
            checked_size(checked_mul(count, sizeof(transport::BFloat16), name),
                         name),
            cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  const auto contribution_offset =
      impl.audit.backend ==
                  transport::ActivationTransportBackend::NcclSocketFallback ||
              impl.audit.backend == transport::ActivationTransportBackend::Gin
          ? impl.pool->layout.combine_return
          : impl.pool->layout.contribution;
  copy(result.contribution_by_route.data(), contribution_offset, route_elements,
       "download contributions");
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

ActivationTransportValidationDownload
download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
}

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-output-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "transport-output-download", "device", impl.identity.rank,
                    "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens, impl.identity.hidden,
                                    "output download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "output download elements"));
  if (!output.empty()) {
    const auto bytes = checked_mul(elements, sizeof(transport::BFloat16),
                                   "output download bytes");
    internal::throw_cuda(
        "transport-output-download",
        cudaMemcpy(output.data(),
                   at<const void>(impl.pool->slots[impl.slot->index].device,
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
BackwardValidationDownload
BackwardValidationBridge::download(const DeviceBackwardHandle &handle,
                                   const CudaExecution &execution) {
  if (!handle.impl_ || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.tensors.impl_->completion.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "backward-download",
      cudaEventSynchronize(impl.tensors.impl_->completion.ready),
      impl.identity.rank, impl.identity.world, "backward",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Backward output");
  const auto &data = *impl.tensors.impl_;
  BackwardValidationDownload result;
  result.dx.resize(
      checked_size(checked_mul(impl.identity.tokens, impl.identity.hidden,
                               "Backward dX download elements"),
                   "Backward dX download elements"));
  result.dgate.resize(
      checked_size(checked_mul(impl.identity.tokens, data.top_k,
                               "Backward dGate download elements"),
                   "Backward dGate download elements"));
  result.dw1.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                              "Backward dW1 download rows"),
                  impl.identity.hidden, "Backward dW1 download elements"),
      "Backward dW1 download elements"));
  result.dw2.resize(checked_size(
      checked_mul(checked_mul(impl.owned_experts, impl.identity.hidden,
                              "Backward dW2 download rows"),
                  impl.intermediate, "Backward dW2 download elements"),
      "Backward dW2 download elements"));
  const auto copy = [&](void *host, const void *device, std::uint64_t bytes,
                        const char *name) {
    if (bytes == 0ULL)
      return;
    internal::throw_cuda("backward-download",
                         cudaMemcpy(host, device, checked_size(bytes, name),
                                    cudaMemcpyDeviceToHost),
                         impl.identity.rank, impl.identity.world, "backward",
                         impl.identity.generation, ExchangeKind::ExpertCounts,
                         name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.dx.data(), data.dx->data,
       result.dx.size() * sizeof(transport::BFloat16), "download dX");
  copy(result.dgate.data(), data.dgate->data,
       result.dgate.size() * sizeof(float), "download dGate");
  const auto copy_weight = [&](auto &host, const void *device, const char *name) {
    if (data.fp32_weight_gradients) {
      // The legacy validation format is BF16. Convert explicitly rather than
      // interpreting the first half of an FP32 allocation as BF16 elements.
      std::vector<float> values(host.size());
      copy(values.data(), device, values.size() * sizeof(float), name);
      for (std::size_t i = 0; i < values.size(); ++i)
        host[i] = transport::float_to_bfloat16(values[i]);
    } else {
      copy(host.data(), device, host.size() * sizeof(transport::BFloat16), name);
    }
  };
  copy_weight(result.dw1, data.dw1->data, "download dW1");
  copy_weight(result.dw2, data.dw2->data, "download dW2");
  return result;
}
testing::GinPutSignalProbeResult
testing::run_distributed_gin_put_signal_probe(const NcclSession &session,
                                              const CudaExecution &execution,
                                              std::uint64_t payload_bytes) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)payload_bytes;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-put-probe, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-put-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(exec->device));
  constexpr std::size_t kAllocationBytes = 4096;
  constexpr std::size_t kReceiveOffset = 2048;
  constexpr std::size_t kBarrierOffset = 3072;
  constexpr std::size_t kErrorOffset = 4088;
  if (payload_bytes == 0 || payload_bytes > kReceiveOffset)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-put-probe",
                    "payload_bytes", nccl->rank, "1..2048",
                    std::to_string(payload_bytes));

  using QueryPropertiesFn =
      ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t, const ncclDevComm_t *);
  using WindowRegisterFn =
      ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-put-probe, field=symbol, rank=0, expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties =
      reinterpret_cast<QueryPropertiesFn>(load("ncclCommQueryProperties"));
  const auto dev_comm_create =
      reinterpret_cast<DevCommCreateFn>(load("ncclDevCommCreate"));
  const auto dev_comm_destroy =
      reinterpret_cast<DevCommDestroyFn>(load("ncclDevCommDestroy"));
  const auto window_register =
      reinterpret_cast<WindowRegisterFn>(load("ncclCommWindowRegister"));
  const auto window_deregister =
      reinterpret_cast<WindowDeregisterFn>(load("ncclCommWindowDeregister"));
  const auto mem_alloc = reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto result = query_properties(nccl->comm, &properties);
  if (result != ncclSuccess || properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-put-probe, field=railed_gin, rank=" +
            std::to_string(nccl->rank) + ", expected=available, actual=" +
            (result == ncclSuccess ? std::string("NCCL_GIN_TYPE_NONE")
                                   : std::string(ncclGetErrorString(result))));
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0)
    transport_error(NotifyErrorCode::BackendUnavailable, "gin-put-probe",
                    "rail topology", nccl->rank, "rail.nRanks>=2, lsa>0",
                    "rail=" + std::to_string(rail.nRanks) +
                        ", lsa=" + std::to_string(lsa.nRanks));

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 32;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  requirements.ginType = NCCL_GIN_TYPE_GDAKI;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };
  try {
    internal::throw_nccl(
        "gin-put-probe", dev_comm_create(nccl->comm, &requirements, &dev_comm),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclDevCommCreate(rail GIN)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-put-probe", mem_alloc(&allocation, kAllocationBytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN probe window)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemsetAsync(allocation, 0, kAllocationBytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN probe window)");
    std::vector<std::uint8_t> send(
        checked_size(payload_bytes, "GIN probe payload"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(nccl->rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda("gin-put-probe",
                         cudaMemcpyAsync(allocation, send.data(), send.size(),
                                         cudaMemcpyHostToDevice, exec->stream),
                         nccl->rank, nccl->world, session.session().identity(),
                         session.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaMemcpyAsync(GIN probe source)");
    internal::throw_cuda("gin-put-probe", cudaStreamSynchronize(exec->stream),
                         nccl->rank, nccl->world, session.session().identity(),
                         session.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaStreamSynchronize(before GIN window register)");
    internal::throw_nccl(
        "gin-put-probe",
        window_register(nccl->comm, allocation, kAllocationBytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN probe)");
    gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
    auto *barrier = static_cast<std::uint64_t *>(allocation) +
                    kBarrierOffset / sizeof(std::uint64_t);
    internal::throw_nccl("gin-put-probe",
                         ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                                       nccl->comm, exec->stream),
                         nccl->rank, nccl->world, session.session().identity(),
                         session.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "ncclAllReduce(after GIN reset)");
    gin_probe_put_signal_kernel<<<1, 1, 0, exec->stream>>>(
        dev_comm, window, checked_size(payload_bytes, "GIN probe payload"),
        kReceiveOffset,
        reinterpret_cast<unsigned int *>(static_cast<std::byte *>(allocation) +
                                         kErrorOffset));
    internal::throw_cuda(
        "gin-put-probe", cudaGetLastError(), nccl->rank, nccl->world,
        session.session().identity(), session.session().generation(),
        ExchangeKind::ExpertCounts, "GIN put/signal kernel launch");
    internal::throw_cuda("gin-put-probe", cudaStreamSynchronize(exec->stream),
                         nccl->rank, nccl->world, session.session().identity(),
                         session.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaStreamSynchronize(GIN put/signal)");
    unsigned int device_error = 0;
    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + kReceiveOffset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN received payload)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(&device_error,
                   static_cast<std::byte *>(allocation) + kErrorOffset,
                   sizeof(device_error), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN device error)");
    if (device_error != 0)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "device_error", nccl->rank, "0",
                      std::to_string(device_error));
    const auto source_rail = (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_rank = source_rail * lsa.nRanks + lsa.rank;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(source_rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    testing::GinPutSignalProbeResult probe;
    probe.payload_bytes = payload_bytes;
    probe.rail_rank = static_cast<std::uint32_t>(rail.rank);
    probe.rail_world = static_cast<std::uint32_t>(rail.nRanks);
    probe.local_rank = static_cast<std::uint32_t>(lsa.rank);
    probe.received_hash = hash(received);
    probe.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "received payload hash", nccl->rank,
                      std::to_string(probe.expected_hash),
                      std::to_string(probe.received_hash));
    cleanup();
    return probe;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

testing::GinLinkBenchmarkResult testing::run_distributed_gin_link_benchmark(
    const NcclSession &session, const CudaExecution &execution,
    GinConnectionMode mode, std::uint32_t target_shift,
    const std::vector<std::uint64_t> &payload_bytes, std::uint32_t warmup,
    std::uint32_t measured, std::uint32_t throughput_puts_per_sample,
    std::uint32_t contexts) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)mode;
  (void)target_shift;
  (void)payload_bytes;
  (void)warmup;
  (void)measured;
  (void)throughput_puts_per_sample;
  (void)contexts;
  throw NotifyDispatchError(NotifyErrorCode::BackendUnavailable,
                            "phase=gin-link-benchmark, field=headers, rank=0, "
                            "expected=nccl_device.h, actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-link-benchmark",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(exec->device));
  if (payload_bytes.empty() || measured == 0U ||
      throughput_puts_per_sample == 0U || contexts == 0U ||
      contexts > static_cast<unsigned>(std::numeric_limits<int>::max()))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-link-benchmark",
                    "benchmark shape", nccl->rank,
                    "non-empty payloads and positive samples/puts/contexts",
                    "invalid");
  const auto maximum_payload =
      *std::max_element(payload_bytes.begin(), payload_bytes.end());
  if (maximum_payload == 0ULL)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-link-benchmark",
                    "payload_bytes", nccl->rank, "positive", "0");
  for (const auto bytes : payload_bytes)
    if (bytes == 0ULL || bytes > maximum_payload)
      transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                      "gin-link-benchmark", "payload_bytes", nccl->rank,
                      "1..max_payload", std::to_string(bytes));

  using QueryPropertiesFn =
      ncclResult_t (*)(ncclComm_t, ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t, const ncclDevComm_t *);
  using WindowRegisterFn =
      ncclResult_t (*)(ncclComm_t, void *, std::size_t, ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-link-benchmark, field=symbol, rank=0, "
                      "expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties =
      reinterpret_cast<QueryPropertiesFn>(load("ncclCommQueryProperties"));
  const auto dev_comm_create =
      reinterpret_cast<DevCommCreateFn>(load("ncclDevCommCreate"));
  const auto dev_comm_destroy =
      reinterpret_cast<DevCommDestroyFn>(load("ncclDevCommDestroy"));
  const auto window_register =
      reinterpret_cast<WindowRegisterFn>(load("ncclCommWindowRegister"));
  const auto window_deregister =
      reinterpret_cast<WindowDeregisterFn>(load("ncclCommWindowDeregister"));
  const auto mem_alloc = reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto nccl_result = query_properties(nccl->comm, &properties);
  if (nccl_result != ncclSuccess)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=properties, rank=" +
            std::to_string(nccl->rank) +
            ", actual=" + ncclGetErrorString(nccl_result));
  const auto selected_type = mode == GinConnectionMode::Rail
                                 ? properties.railedGinType
                                 : properties.ginType;
  if (selected_type == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-link-benchmark, field=" +
            std::string(mode == GinConnectionMode::Rail ? "railedGinType"
                                                        : "ginType") +
            ", rank=" + std::to_string(nccl->rank) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");

  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0 ||
      nccl->world % static_cast<std::uint32_t>(lsa.nRanks) != 0U)
    transport_error(NotifyErrorCode::BackendUnavailable, "gin-link-benchmark",
                    "homogeneous topology", nccl->rank,
                    "at least two nodes and positive LSA",
                    "rail=" + std::to_string(rail.nRanks) +
                        ",lsa=" + std::to_string(lsa.nRanks));
  target_shift %= static_cast<std::uint32_t>(lsa.nRanks);
  const auto destination_node = (rail.rank + 1) % rail.nRanks;
  const auto destination_local =
      (lsa.rank + static_cast<int>(target_shift)) % lsa.nRanks;
  const auto destination =
      mode == GinConnectionMode::Rail
          ? destination_node
          : destination_node * lsa.nRanks + destination_local;
  const bool fanout_full = mode == GinConnectionMode::Full &&
                           std::getenv("WEFTMOE_GIN_BENCH_FANOUT") != nullptr;

  const auto align_4k = [](std::uint64_t value, const char *field) {
    return checked_add(value, 4095ULL, field) / 4096ULL * 4096ULL;
  };
  const auto receive_offset = align_4k(maximum_payload, "GIN receive offset");
  const auto barrier_offset =
      align_4k(checked_mul(receive_offset, std::uint64_t(contexts) + 1U,
                           "GIN benchmark context extents"),
               "GIN benchmark barrier offset");
  const auto allocation_bytes = checked_size(
      checked_add(barrier_offset, 4096ULL, "GIN benchmark window bytes"),
      "GIN benchmark window bytes");
  const auto duration_bytes =
      checked_size(checked_mul(measured, sizeof(unsigned long long),
                               "GIN benchmark duration bytes"),
                   "GIN benchmark duration bytes");

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = static_cast<int>(contexts);
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  requirements.ginSignalCount = contexts;
  requirements.ginCounterCount = contexts;
  requirements.ginConnectionType = mode == GinConnectionMode::Rail
                                       ? NCCL_GIN_CONNECTION_RAIL
                                       : NCCL_GIN_CONNECTION_FULL;
  requirements.ginType = NCCL_GIN_TYPE_GDAKI;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  unsigned long long *device_duration = nullptr;
  unsigned int *device_error = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (device_error != nullptr)
      (void)cudaFree(device_error);
    if (device_duration != nullptr)
      (void)cudaFree(device_duration);
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };

  try {
    internal::throw_nccl(
        "gin-link-benchmark",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        mode == GinConnectionMode::Rail ? "ncclDevCommCreate(RAIL)"
                                        : "ncclDevCommCreate(FULL)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-link-benchmark", mem_alloc(&allocation, allocation_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN link benchmark)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_duration, duration_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN duration samples)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaMalloc(&device_error, sizeof(*device_error)),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN device error)");
    internal::throw_cuda(
        "gin-link-benchmark",
        cudaMemsetAsync(allocation, 0, allocation_bytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN benchmark window)");
    std::vector<std::uint8_t> send(
        checked_size(maximum_payload, "GIN benchmark source"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(nccl->rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda("gin-link-benchmark",
                         cudaMemcpyAsync(allocation, send.data(), send.size(),
                                         cudaMemcpyHostToDevice, exec->stream),
                         nccl->rank, nccl->world, session.session().identity(),
                         session.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaMemcpyAsync(GIN benchmark source)");
    internal::throw_cuda(
        "gin-link-benchmark", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN benchmark registration)");
    internal::throw_nccl(
        "gin-link-benchmark",
        window_register(nccl->comm, allocation, allocation_bytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN link benchmark)");

    GinLinkBenchmarkResult benchmark;
    benchmark.mode = mode;
    benchmark.target_shift = target_shift;
    benchmark.rank = nccl->rank;
    benchmark.world = nccl->world;
    benchmark.local_rank = static_cast<std::uint32_t>(lsa.rank);
    benchmark.local_world = static_cast<std::uint32_t>(lsa.nRanks);
    benchmark.contexts = contexts;
    benchmark.gin_type = static_cast<std::uint32_t>(properties.ginType);
    benchmark.railed_gin_type =
        static_cast<std::uint32_t>(properties.railedGinType);

    std::vector<std::uint32_t> batches{1U};
    if (throughput_puts_per_sample != 1U)
      batches.push_back(throughput_puts_per_sample);
    auto *barrier = reinterpret_cast<std::uint64_t *>(
        static_cast<std::byte *>(allocation) + barrier_offset);
    for (const auto bytes : payload_bytes) {
      for (const auto puts : batches) {
        gin_probe_reset_kernel<<<contexts, 1, 0, exec->stream>>>(dev_comm);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemsetAsync(device_error, 0, sizeof(*device_error),
                            exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemsetAsync(GIN benchmark error)");
        internal::throw_nccl(
            "gin-link-benchmark",
            ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum, nccl->comm,
                          exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "ncclAllReduce(after GIN benchmark reset)");
        cudaLaunchAttribute cooperative{};
        cooperative.id = cudaLaunchAttributeCooperative;
        cooperative.val.cooperative = 1;
        cudaLaunchConfig_t launch{};
        launch.gridDim = dim3(contexts);
        launch.blockDim = dim3(1U);
        launch.stream = exec->stream;
        launch.attrs = &cooperative;
        launch.numAttrs = 1U;
        const auto launch_error = cudaLaunchKernelEx(
            &launch, gin_link_benchmark_kernel, dev_comm, window,
            mode == GinConnectionMode::Full, fanout_full, destination,
            lsa.nRanks, checked_size(bytes, "GIN benchmark payload"),
            checked_size(receive_offset, "GIN benchmark receive offset"),
            warmup, measured, puts, device_duration, device_error);
        internal::throw_cuda(
            "gin-link-benchmark", launch_error, nccl->rank, nccl->world,
            session.session().identity(), session.session().generation(),
            ExchangeKind::ExpertCounts, "GIN benchmark kernel launch");
        internal::throw_cuda(
            "gin-link-benchmark", cudaStreamSynchronize(exec->stream),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaStreamSynchronize(GIN link benchmark)");
        unsigned int host_error = 0U;
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(&host_error, device_error, sizeof(host_error),
                       cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark error)");
        if (host_error != 0U)
          transport_error(NotifyErrorCode::NcclFailure, "gin-link-benchmark",
                          "device_error", nccl->rank, "0",
                          std::to_string(host_error));
        GinLinkBenchmarkCaseResult sample;
        sample.payload_bytes = bytes;
        sample.puts_per_sample = puts;
        sample.duration_ns.resize(measured);
        internal::throw_cuda(
            "gin-link-benchmark",
            cudaMemcpy(sample.duration_ns.data(), device_duration,
                       duration_bytes, cudaMemcpyDeviceToHost),
            nccl->rank, nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaMemcpy(GIN benchmark durations)");
        benchmark.cases.push_back(std::move(sample));
      }
    }

    std::vector<std::uint8_t> received(send.size());
    const auto source_node = (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_local =
        mode == GinConnectionMode::Rail
            ? lsa.rank
            : (lsa.rank + lsa.nRanks - static_cast<int>(target_shift)) %
                  lsa.nRanks;
    const auto source_rank = source_node * lsa.nRanks + source_local;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          ((fanout_full ? 0ULL : static_cast<std::uint64_t>(source_rank)) *
               17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    benchmark.expected_hash = hash(expected);
    for (unsigned context = 0U; context < contexts; ++context) {
      internal::throw_cuda(
          "gin-link-benchmark",
          cudaMemcpy(received.data(),
                     static_cast<std::byte *>(allocation) +
                         (context + 1ULL) * receive_offset,
                     received.size(), cudaMemcpyDeviceToHost),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpy(GIN benchmark received payload)");
      benchmark.received_hash = hash(received);
      if (received != expected)
        transport_error(NotifyErrorCode::NcclFailure, "gin-link-benchmark",
                        "received context payload", nccl->rank,
                        std::to_string(benchmark.expected_hash),
                        "context=" + std::to_string(context) +
                            ",hash=" + std::to_string(benchmark.received_hash));
    }
    cleanup();
    return benchmark;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

bool testing::run_local_p2p_put_signal_probe(std::uint32_t source_device,
                                             std::uint32_t destination_device,
                                             std::uint64_t elements) {
  if (elements == 0)
    return false;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return false;
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
  bool ok = false;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination,
                 checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                          "peer destination bytes"),
                              "peer destination bytes")) != cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemset(signal, 0, sizeof(*signal));
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable =
        cudaDeviceEnablePeerAccess(static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source,
                 checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                          "peer source bytes"),
                              "peer source bytes")) != cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  if (cudaMemsetAsync(source, 0x5a,
                      checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                               "peer copy bytes"),
                                   "peer copy bytes"),
                      source_stream) != cudaSuccess ||
      cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
    const auto blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
    const auto run = [&](bool early_signal, unsigned int &observed_errors) {
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaMemsetAsync(
              destination, 0,
              checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                       "peer reset bytes"),
                           "peer reset bytes"),
              destination_stream) != cudaSuccess ||
          cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
              cudaSuccess ||
          cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(destination_stream) != cudaSuccess)
        return false;
      peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
          destination, signal, errors, elements, expected);
      if (cudaGetLastError() != cudaSuccess)
        return false;
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                          source_stream) != cudaSuccess ||
          cudaStreamSynchronize(source_stream) != cudaSuccess)
        return false;
      peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
          source, destination, signal, completed_blocks, elements,
          early_signal);
      if (cudaGetLastError() != cudaSuccess)
        return false;
      // The destination independently observes the signal and validates every
      // word before the source stream is synchronized.
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaStreamSynchronize(destination_stream) != cudaSuccess ||
          cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                     cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
      (void)cudaSetDevice(static_cast<int>(source_device));
      return cudaStreamSynchronize(source_stream) == cudaSuccess;
    };
    {
      unsigned int positive_errors = 0;
      unsigned int early_signal_errors = 0;
      ok = run(false, positive_errors) && positive_errors == 0U &&
           run(true, early_signal_errors) && early_signal_errors != 0U;
    }
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return ok;
}

double testing::measure_local_p2p_put_bandwidth_gbps(
    std::uint32_t source_device, std::uint32_t destination_device,
    std::uint64_t bytes, std::uint32_t warmup, std::uint32_t measured) {
  if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0 || measured == 0)
    return 0.0;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return 0.0;
  const auto elements = bytes / sizeof(std::uint64_t);
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  double result = 0.0;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable =
        cudaDeviceEnablePeerAccess(static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemsetAsync(
      source, 0x5a, checked_size(bytes, "P2P bandwidth bytes"), source_stream);
  if (cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
    constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
    const auto blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
    double measured_seconds = 0.0;
    for (std::uint32_t iteration = 0; iteration < warmup + measured;
         ++iteration) {
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
              cudaSuccess ||
          cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
          destination, signal, errors, elements, expected);
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                          source_stream) != cudaSuccess ||
          cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      const auto start = std::chrono::steady_clock::now();
      peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
          source, destination, signal, completed_blocks, elements, false);
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      const auto finish = std::chrono::steady_clock::now();
      unsigned int observed_errors = 0;
      if (cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          observed_errors != 0U)
        goto cleanup;
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      if (iteration >= warmup)
        measured_seconds +=
            std::chrono::duration<double>(finish - start).count();
    }
    if (measured_seconds > 0.0)
      result = static_cast<double>(bytes) * measured / measured_seconds / 1.0e9;
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return result;
}
