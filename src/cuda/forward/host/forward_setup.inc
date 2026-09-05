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
