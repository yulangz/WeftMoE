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

