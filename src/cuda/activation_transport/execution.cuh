#pragma once

// Implementation fragment included only by activation_transport.cu, inside
// weftmoe::cuda, after its shared state and kernel definitions. Keeping the
// same translation unit preserves private types and device compilation.

// FWD/BWD launch, completion, and compute-handle lifecycle.

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::enqueue(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations || !weights)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-run",
                    "backend/dispatch/activation/weights", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto gin_backend = std::dynamic_pointer_cast<GinBackendState>(backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  if (id.top_k > 32U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "top_k", id.calling_rank,
                    "at most one route per warp lane (32)",
                    std::to_string(id.top_k));
  const auto &activation = *activations.impl_;
  const auto &weight = *weights.impl_;
  if (weight.fp8.w1 && config.retain_backward_state && !weight.fp8.w1_t)
    throw std::invalid_argument("FP8 training requires pretransposed weights");
  if ((!use_gin && id.num_nodes != 1) ||
      (use_gin && id.num_nodes != gin_backend->num_nodes) ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) + ", expected=" +
            (use_gin ? "runtime GIN topology" : "one local node") +
            " and exact NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) +
            " nodes, world=" + std::to_string(id.world_size));
  // The hot GIN fanout record keeps its established 32-byte width by storing
  // the node-local rank and global expert ID in 16 bits.  Reject wider
  // topologies before any device-side narrowing instead of silently wrapping.
  if (use_gin && (id.world_size > std::numeric_limits<std::uint16_t>::max() ||
                  id.num_experts > std::numeric_limits<std::uint16_t>::max()))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "GIN fanout identity width", id.calling_rank,
                    "world_size and num_experts <= 65535",
                    std::to_string(id.world_size) + "/" +
                        std::to_string(id.num_experts));
  if ((activation.pool && activation.pool.get() != state.get()) ||
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
  auto lease = activation.slot
                   ? activation.slot
                   : acquire_slot(state, id.generation, id.calling_rank);
  if (state->config.max_hidden_size != config.hidden_size ||
      state->config.max_intermediate_size != config.intermediate_size ||
      !state->slots[lease->index].forward_maps_valid ||
      (config.retain_backward_state &&
       (!state->config.enable_backward_state ||
        state->layout.forward_training_output == 0)))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "forward pool geometry/backward state", id.calling_rank,
                    "Hopper 7168x3072 TMA maps and requested retained state",
                    "mismatch/missing");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation,         id.calling_rank, id.world_size,
      id.num_tokens,         id.top_k,        id.config_digest_word0,
      id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  auto view = ActivationTransportBridge::view(consumer);
  // Notify materializes the route permutation and destination M-tile map for
  // every topology.  The split one-node transport consumes the same metadata
  // as GIN Dispatch even though it has no RDMA waves of its own.
  const bool split_local_metadata = !use_gin && id.num_nodes == 1U;
  auto dispatch_waves = use_gin || split_local_metadata
                            ? ActivationTransportBridge::wave_view(consumer)
                            : DispatchWaveKernelView{};
  const bool fused_inference_routes =
      ActivationTransportBridge::fused_inference_routes(dispatch);
  // A checkpoint's no-grad pass reuses the training ring's Notify and
  // stream-ordered workspace, but leaves the saved X/AB pointers null.
  const bool training_ring =
      state->layout.static_ring.depth && !fused_inference_routes &&
      state->config.enable_backward_state;
  const auto training_counts =
      ActivationTransportBridge::training_counts(dispatch);
  const bool split_notify =
      ActivationTransportBridge::deferred_training_notify(dispatch);
  const bool static_v4_candidate =
      forward_tuning::kUnifiedWorkers &&
      (!use_gin || fused_inference_routes || training_ring) &&
      (!config.retain_backward_state || training_ring);
  const bool local_prefill =
      static_v4_candidate && local_prefill_streaming(state->config);
  const bool network_prefill =
      static_v4_candidate && network_prefill_streaming(state->config);
  const bool standalone_notify = local_prefill || network_prefill;
  if (!standalone_notify &&
      !weftmoe_dt::eligible(state->config, state->layout.static_ring))
    throw std::invalid_argument("Forward requires a current PF or DT "
                                "specialization and ring workspace");
  if (state->layout.static_ring.depth &&
      (!static_v4_candidate || (!fused_inference_routes && !training_ring)))
    throw std::invalid_argument("ring workspace requires static Forward");
  if (training_ring) {
    if (fused_inference_routes || (view.runtime && !training_counts.runtime))
      throw std::invalid_argument(
          "training ring requires immutable materialized Notify");
    cudaStreamCaptureStatus capture;
    const auto error = cudaStreamIsCapturing(
        internal::checked_execution_state(backend->execution)->stream,
        &capture);
    if (error != cudaSuccess || capture != cudaStreamCaptureStatusNone ||
        config.serialized_graph_replay)
      throw std::invalid_argument(
          "training ring plan/allocation requires eager execution");
  }
  const bool serialized_graph_replay =
      static_v4_candidate && config.serialized_graph_replay;
  const bool direct_inference_routes =
      config.inference_routes.data != nullptr ||
      config.inference_gates.data != nullptr;
  if (direct_inference_routes) {
    if (!static_v4_candidate || !serialized_graph_replay ||
        !config.inference_routes.data || !config.inference_gates.data)
      throw std::invalid_argument(
          "direct inference routes require both buffers and serialized "
          "static inference");
    validate_external_cuda_range(
        config.inference_routes.data, view.sizes.route_count,
        config.inference_routes.bytes / sizeof(std::int32_t),
        sizeof(std::int32_t), backend->execution.device_ordinal(),
        "inference_routes");
    validate_external_cuda_range(
        config.inference_gates.data, view.sizes.route_count,
        config.inference_gates.bytes / sizeof(float), sizeof(float),
        backend->execution.device_ordinal(), "inference_gates");
  }
  if (weight.owned_experts != view.sizes.owned_expert_count)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "owned_experts", id.calling_rank,
                    std::to_string(view.sizes.owned_expert_count),
                    std::to_string(weight.owned_experts));
  ForwardTensorStorage tensors;
  const bool dt = fused_inference_routes && weftmoe_dt::eligible(state->config, state->layout.static_ring);
  const bool source_fp8 = WEFTMOE_FP8_SOURCE_X && WEFTMOE_FP8_PACKED_RING &&
      weight.fp8.w1 && (dt || (standalone_notify && forward_tuning::kFc1TileN == 256U)) &&
      (!config.retain_backward_state || WEFTMOE_FP8_STREAMED_WGRAD);
  if (activation.prequantized && !source_fp8)
    throw std::invalid_argument("Prequantized input requires FP8 weights and source FP8 transport");
  auto storage_config = config;
  if (weight.fp8.w1 && !source_fp8)
    throw std::invalid_argument("FP8 requires source transport and either DT or PF N256");
  if (state->config.fp8_workspace && !source_fp8)
    throw std::invalid_argument("Compact FP8 workspace requires source FP8 transport with DT or PF N256");
  storage_config.fp8_x_checkpoint = source_fp8 && config.retain_backward_state;
  storage_config.fp8_ab_checkpoint = source_fp8 && config.retain_backward_state;
  const bool allocate_training_tape = training_ring && !config.tensors;
  const auto prepare_storage = [&] {
    if (config.tensors)
      tensors = config.tensors;
    else if (config.tensor_factory)
      tensors = config.tensor_factory(storage_config);
    else
      tensors = prepare_forward_tensors(dispatch, backend->execution, storage_config);
    if (!tensors)
      throw std::invalid_argument(
          "Forward tensor preparation returned empty storage");
    auto &data = *tensors.impl_;
    if (data.fp8_ab_checkpoint != storage_config.fp8_ab_checkpoint)
      throw std::invalid_argument("AB checkpoint format does not match Forward compute path");
    if (storage_config.fp8_x_checkpoint && data.x)
      throw std::invalid_argument("source FP8 requires tensor storage prepared without a BF16 X checkpoint");
    if (data.training && !storage_config.fp8_x_checkpoint && !data.x)
      throw std::invalid_argument("this Forward path requires a BF16 X checkpoint");
    if (data.tokens != id.num_tokens || data.hidden != config.hidden_size ||
        data.intermediate != config.intermediate_size ||
        data.rows < dispatch.received_row_capacity() ||
        data.training != config.retain_backward_state ||
        data.completion.device != backend->execution.device_ordinal())
      throw std::runtime_error(
          "Forward tensor storage does not match this invocation");
    // This cache belongs to the retained tape, never to a recycled ring slot.
    if (WEFTMOE_FP8_STREAMED_WGRAD && weight.fp8.w1 && data.training &&
        standalone_notify && forward_tuning::kFc1TileN == 256U) {
      const auto rows = std::max<std::uint64_t>(1U,
          ActivationTransportBridge::padded_received_rows(dispatch));
      const auto elements = checked_mul(rows, data.hidden, "FP8 saved X");
      std::size_t cursor = 0;
      append(cursor, elements, 1U, "FP8 saved X data");
      const auto sf_offset = append(cursor, elements / 128U, 4U, "FP8 saved X scales");
      if (!data.fp8_x || data.fp8_x->bytes < cursor) {
        const auto binding = data.allocator ? data.allocator(cursor) : CudaTensorBuffer{};
        if (data.allocator && !binding.owner)
          throw std::runtime_error("tensor allocator returned no owner");
        data.fp8_x = tensor_storage_detail::allocate(cursor, data.completion.device, binding);
      }
      data.fp8_sf_offset = sf_offset;
    }
    const auto writes = {data.output, data.x, data.ab, data.fp8_x};
    tensor_storage_detail::validate_disjoint(
        {data.output, data.x, data.ab, data.gates, data.headers, data.fp8_x});
    if (activation.prequantized) {
      for (const auto *buffer : {&activation.fp8.data, &activation.fp8.scales})
        tensor_storage_detail::validate_input_disjoint(writes, buffer->data, buffer->bytes);
    } else {
      tensor_storage_detail::validate_input_disjoint(
          writes, activation.pointer,
          id.num_tokens * config.hidden_size * sizeof(transport::BFloat16));
    }
    const auto w2_bytes = static_cast<std::uint64_t>(weight.owned_experts) *
                          config.hidden_size * config.intermediate_size *
                          sizeof(transport::BFloat16);
    if (weight.fp8.w1) {
      const auto &b = weight.fp8_buffers;
      for (const auto *buffer : {&b.w1, &b.s1, &b.w2, &b.s2, &b.w1_t, &b.s1_t, &b.w2_t, &b.s2_t})
        tensor_storage_detail::validate_input_disjoint(writes, buffer->data, buffer->bytes);
    } else {
      tensor_storage_detail::validate_input_disjoint(writes, weight.w1, 2 * w2_bytes);
      tensor_storage_detail::validate_input_disjoint(writes, weight.w2, w2_bytes);
    }
  };
  if (!allocate_training_tape)
    prepare_storage();

  auto record = state->config.stream_ordered_training && training_ring
                    ? state->training_completions->acquire()
                    : nullptr;
  auto flight = std::make_shared<OperatorFlight>(lease, false);
  SlotFailureGuard failure_guard{lease, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[lease->index];
  GinExecution communication(state->gin_resources, exec->stream,
                             serialized_graph_replay);
  // On a host-side preparation failure, do not release an input owner while
  // source preparation is still reading it. Successful calls transfer that
  // lifetime to the Forward handle and its normal completion protocol.
  struct SourcePreparationGuard {
    cudaStream_t stream = nullptr;
    cudaStream_t planner_stream = nullptr;
    ~SourcePreparationGuard() {
      if (stream) {
        (void)cudaStreamSynchronize(stream);
        if (planner_stream && planner_stream != stream)
          (void)cudaStreamSynchronize(planner_stream);
      }
    }
  } source_preparation_guard;
  const auto &layout = state->layout;
#if WEFTMOE_DEBUG_VALIDATE
  const auto debug_forward_host_phase = [&](const char *phase) {
    if (std::getenv("WEFTMOE_FORWARD_PROGRESS") != nullptr)
      std::cerr << "FORWARD_RUN_HOST rank=" << id.calling_rank
                << " generation=" << id.generation << " phase=" << phase
                << std::endl;
  };
  debug_forward_host_phase("consumer_acquired");
#endif
  if (dispatch.received_row_capacity() >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "forward route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");

  if (slot.transport_epoch >= (1ULL << 32U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "forward-run",
                    "transport_epoch", id.calling_rank, "<2^32-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  const bool cancellation_requested =
      options.cancellation.cancellation_requested();
  __atomic_store_n(slot.host_cancel_epoch,
                   cancellation_requested ? generation_epoch : 0ULL,
                   __ATOMIC_RELEASE);
  if (cancellation_requested || slot.cancel_epoch_dirty) {
    internal::throw_cuda(
        "forward-run",
        cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                        sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                        exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "H2D initialize cancel epoch");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              sizeof(std::uint64_t));
    slot.cancel_epoch_dirty = cancellation_requested;
  }
  const bool activation_on_execution_stream =
      activation.producer_execution.same_instance(backend->execution.token()) ||
      (activation.allocation && activation.allocation->external_producer &&
       activation.allocation->external_producer->stream == exec->stream);
  if (!activation_on_execution_stream)
    internal::throw_cuda(
        "forward-run",
        internal::wait_input_ready(
            exec->stream,
            activation.allocation ? activation.allocation->ready
                                  : slot.producer_ready,
            activation.allocation ? activation.allocation->ready_capture_id
                                  : slot.producer_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(activation ready)");
  const bool weights_on_execution_stream =
      weight.execution.same_instance(backend->execution.token()) ||
      (weight.allocation && weight.allocation->external_producer &&
       weight.allocation->external_producer->stream == exec->stream);
  if (!weights_on_execution_stream)
    internal::throw_cuda(
        "forward-run",
        internal::wait_input_ready(exec->stream, weight.allocation->ready,
                                   weight.allocation->ready_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(weights ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("stream_dependencies_enqueued");
#endif

  if (!static_v4_candidate)
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.forward_start, exec->stream,
                                          &slot.forward_capture_id),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaEventRecord(forward start)");
#if WEFTMOE_DEVICE_TIMELINE
  reset_timeline("forward-run", slot.forward_timeline, state->timeline_layout,
                 state->timeline_max_ctas, exec->stream, id.calling_rank,
                 id.world_size, id.generation, backend->session);
#endif
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("generation_init_enqueued");
#endif
  ForwardKernelArgs args{};
  if (fused_inference_routes) {
    view.runtime = at<const DeviceDispatchRuntime>(
        slot.device, layout.fused_dispatch_runtime);
    view.expert_id =
        at<const std::uint32_t>(slot.device, layout.fused_route_expert);
    view.local_offset =
        at<const std::uint64_t>(slot.device, layout.fused_route_local_offset);
    view.return_global_expert_base = nullptr;
    view.route_dst_row =
        at<const std::uint64_t>(slot.device, layout.fused_route_dst_row);
    view.route_dst_rank =
        at<const std::uint32_t>(slot.device, layout.fused_route_dst_rank);
    view.route_dst_node = nullptr;
    view.route_dst_local_rank = nullptr;
    view.combine_order =
        at<const std::uint32_t>(slot.device, layout.fused_combine_order);
    view.source_expert_count =
        at<const std::uint64_t>(slot.device, layout.fused_source_expert_count);
    view.owned_expert_ids =
        at<const std::uint32_t>(slot.device, layout.fused_owned_expert_ids);
    view.expert_row_count =
        at<const std::uint64_t>(slot.device, layout.fused_expert_row_count);
    view.expert_row_prefix =
        at<const std::uint64_t>(slot.device, layout.fused_expert_row_prefix);
    view.mtile_row_offsets =
        at<const std::uint64_t>(slot.device, layout.fused_mtile_offsets);
    view.mtile_expected_rows =
        at<const std::uint32_t>(slot.device, layout.fused_mtile_rows);
    dispatch_waves.local_route_order = nullptr;
    dispatch_waves.route_dst_mtile =
        at<const std::uint32_t>(slot.device, layout.fused_route_dst_mtile);
  }
  args.dispatch = view;
  args.checkpoint.counts = training_counts;
  if (training_ring)
    args.dispatch.runtime =
        at<DeviceDispatchRuntime>(slot.device, layout.fused_dispatch_runtime);
  args.dispatch_waves = dispatch_waves;
  args.fused_inference_routes = fused_inference_routes;
  args.selected_experts =
      fused_inference_routes || training_ring
          ? (direct_inference_routes
                 ? static_cast<const std::int32_t *>(
                       config.inference_routes.data)
                 : ActivationTransportBridge::route_pointer(dispatch))
          : nullptr;
  args.fused_local_expert_count =
      fused_inference_routes || local_prefill
          ? at<std::uint64_t>(slot.device, layout.fused_local_expert_count)
          : nullptr;
  args.fused_source_routes =
      fused_inference_routes || local_prefill
          ? at<std::uint32_t>(slot.device, layout.fused_source_routes)
          : nullptr;
  args.max_routes_per_rank =
      checked_mul(state->config.max_tokens_per_rank, state->config.max_top_k,
                  "single-node source route capacity");
  args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  args.source_input_offset = layout.input;
  args.fp8_source_x = source_fp8;
  args.prequantized_input = activation.prequantized ? static_cast<const std::uint8_t *>(activation.fp8.data.data) : nullptr;
  args.prequantized_scales = activation.prequantized ? static_cast<const float *>(activation.fp8.scales.data) : nullptr;
  args.fp8_x_pitch = state->config.fp8_workspace ? fp8::source_pitch(config.hidden_size) : config.hidden_size * 2U;
  args.fp8_s_pitch = state->config.fp8_workspace ? fp8::source_pitch(config.intermediate_size) : config.intermediate_size * 2U;
  args.network_pull = network_prefill;
  args.fused_source_gates_offset = layout.fused_source_gates;
  args.input = activation.pointer;
  args.gates = direct_inference_routes
                   ? static_cast<const float *>(config.inference_gates.data)
                   : ActivationTransportBridge::gate_pointer(dispatch);
  args.w1 = weight.w1;
  args.w2 = weight.w2;
  args.fp8 = weight.fp8;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = state->config.channels;
  args.epoch = generation_epoch;
  args.runtime_epoch =
      at<unsigned long long>(slot.device, layout.execution_epoch);
  args.generation_setup_offset = layout.p2p_barrier_receive;
  args.completion_offset = layout.forward_completion;
  args.context_completion_offset = layout.forward_context_completion;
  args.context_completion_capacity = (state->config.max_nodes - 1U) * 8U;
  args.gin_allocated_context_count =
      use_gin ? gin_backend->gin_allocated_context_count : 0;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.use_gin = use_gin;
  args.local_world_size = id.world_size / id.num_nodes;
  args.node = id.calling_rank / args.local_world_size;
#if WEFTMOE_DEVICE_TIMELINE
  args.timeline = make_timeline_context(
      slot.forward_timeline, state->timeline_layout, state->timeline_max_ctas,
      id.calling_rank, profiling::TimelinePhase::Forward, id.generation);
#endif
#if WEFTMOE_BUILD_TESTING
  if (const auto *fault_rank =
          std::getenv("WEFTMOE_INTERNAL_FORWARD_STATUS_FAULT_RANK")) {
    const auto *fault_generation =
        std::getenv("WEFTMOE_INTERNAL_FORWARD_STATUS_FAULT_GENERATION");
    if (fault_generation == nullptr ||
        std::stoull(fault_generation) == id.generation)
      args.injected_status_error_rank = std::stoi(fault_rank);
  }
#endif
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[lease->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.static_network = state->gin_layout.static_forward;
    args.gin_rings = state->gin_layout.rings;
    args.gin_rings.counter_base +=
        lease->index * args.gin_rings.channels() * communication::kRingDepth;
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
    args.gin_return_send_rows_offset = state->gin_layout.return_send_rows;
    args.gin_return_receive_rows_offset = state->gin_layout.return_receive_rows;
    args.gin_production_error_records_offset =
        state->gin_layout.production_error_records;
    args.node = gin_backend->node;
    args.local_world_size = gin_backend->local_world_size;
    args.gin_context_count = gin_backend->gin_context_count;
  }
  args.canonical_rows_offset = layout.canonical_rows;
  if (layout.static_ring.depth) {
    const auto &l = layout.static_ring;
    auto &r = args.ring;
    r.depth = l.depth;
    r.contribution_depth = l.contribution_depth;
    r.contribution_stride =
        std::uint64_t(l.shape.tokens) * state->config.max_top_k;
    r.state = at<static_forward::RingState>(slot.device, l.state);
    r.source_epochs =
        at<static_forward::SourceEpoch>(slot.device, l.source_epochs);
    r.source_reserved = at<std::uint64_t>(slot.device, l.source_reserved);
    r.slots = at<static_forward::ComputeSlot>(slot.device, l.slots);
    r.contributions =
        at<static_forward::ContributionSlot>(slot.device, l.contributions);
    r.contribution_ready = at<std::uint64_t>(slot.device, l.contribution_ready);
    r.window_prefix = at<std::uint64_t>(slot.device, l.window_prefix);
    r.window_sources =
        at<static_forward::WindowSource>(slot.device, l.window_sources);
    r.expert_rows = at<std::uint64_t>(slot.device, l.expert_rows);
    r.expert_blocks = at<std::uint64_t>(slot.device, l.expert_blocks);
    r.blocks = at<static_forward::RingBlock>(slot.device, l.blocks);
    r.rows = at<static_forward::RingRow>(slot.device, l.rows);
    if (training_ring)
      args.checkpoint.rows = at<std::uint64_t>(slot.device, l.tape_rows);
    if (use_gin
#if defined(WEFTMOE_PF_LOCAL_COMPACT_RETURN) && WEFTMOE_PF_LOCAL_COMPACT_RETURN
        || (training_ring &&
            forward_tuning::local_compact_return_capacity_supported(
                state->config.max_tokens_per_rank))
#endif
    )
      r.contribution_rows = at<std::uint32_t>(slot.device, l.contribution_rows);
    if (use_gin) {
      r.pull_packets =
          at<static_forward::PullPacket>(slot.device, l.pull_packets);
      r.window_compute =
          at<static_forward::WindowCompute>(slot.device, l.window_compute);
      r.start_prefix = at<std::uint64_t>(slot.device, l.start_prefix);
      r.pull_prefix = at<std::uint64_t>(slot.device, l.pull_prefix);
      r.start_blocks = at<std::uint32_t>(slot.device, l.start_blocks);
      r.canonical_blocks = at<std::uint32_t>(slot.device, l.canonical_blocks);
      r.pull_order = at<std::uint32_t>(slot.device, l.pull_order);
      r.return_tasks = at<static_forward::ReturnTask>(slot.device, l.return_tasks);
      r.return_capacity = l.return_capacity;
      r.return_slots = at<static_forward::ReturnSlot>(slot.device, l.return_slots);
      r.return_experts = at<static_forward::ReturnExpert>(slot.device, l.return_experts);
    }
    const auto &m = l.source;
    auto *metadata =
        use_gin ? at<char>(slot.gin_window,
                           args.static_network.source +
                               args.node * args.static_network.source_stride)
                : at<char>(slot.device, l.metadata);
    r.source_header = at<static_forward::SourceHeader>(metadata, m.header);
    r.source_gates = at<float>(metadata, m.gates);
    if (training_ring)
      args.checkpoint.source_rows = at<std::uint64_t>(metadata, m.tape_rows);
    auto &p = r.source;
    p.shape = l.shape;
    p.shape.world = id.world_size;
    p.shape.sources = std::min(id.world_size, l.shape.sources);
    p.shape.top_k = id.top_k;
    if (network_prefill)
      p.shape.first_group = args.node * args.local_world_size / p.shape.sources;
    p.packets = l.packets;
    p.source_rank = id.calling_rank;
    p.experts = id.num_experts;
    p.nodes = id.num_nodes;
    p.tokens = id.num_tokens;
    p.selected_experts = args.selected_experts;
    p.valid = nullptr;
    p.expert_prefix = at<std::uint64_t>(metadata, m.expert_prefix);
    p.node_prefix = at<std::uint64_t>(metadata, m.node_prefix);
    p.route_order = at<std::uint64_t>(metadata, m.route_order);
    p.node_tokens = at<static_forward::NodeToken>(metadata, m.node_tokens);
    p.packet_prefix = at<std::uint64_t>(metadata, m.packet_prefix);
    p.error = &r.source_header->error;
    if (id.num_nodes > 1U) {
      p.selected_experts = at<std::int32_t>(metadata, m.selected);
      r.locations = at<static_forward::RouteLocation>(metadata, m.locations);
      r.dispatch = {
          at<std::uint64_t>(metadata, m.dispatch_offsets),
          at<static_forward::DispatchPacketPlan>(metadata, m.dispatch_packets),
          at<std::uint32_t>(metadata, m.dispatch_consumers)};
      r.returning = {
          at<std::uint64_t>(metadata, m.return_prefix),
          at<std::uint64_t>(metadata, m.return_offsets),
          at<static_forward::ReturnPacketPlan>(metadata, m.return_packets),
          static_forward::return_partition(p.shape, p.tokens)};
    }
  }
  args.canonical_gates_offset = layout.canonical_gates;
  args.forward_output_rows_offset = config.retain_backward_state
                                        ? layout.forward_training_output
                                        : layout.canonical_rows;
  args.direct_headers_offset = layout.direct_route_headers;
  args.contribution_offset = layout.contribution;
  args.return_ready_epoch_offset = layout.return_ready_epoch;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.status_offset = layout.status;
  args.forward_s_offset = layout.forward_s;
  args.fp8_ab_quantized = source_fp8;
  const auto bind_storage = [&] {
    const auto &data = *tensors.impl_;
    args.output = static_cast<transport::BFloat16 *>(data.output->data);
    if (!data.training)
      return;
    args.saved_x = data.x ? static_cast<transport::BFloat16 *>(data.x->data) : nullptr;
#if !defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
    if ((!training_ring || !use_gin) && data.rows && data.x)
      args.saved_x_map = make_bf16_tma_store_map(
          data.x->data, config.hidden_size, data.rows, "saved FC1 X", 0U,
          training_ring ? 32U : 128U);
#endif
    args.saved_ab = data.fp8_ab_checkpoint ? nullptr : static_cast<transport::BFloat16 *>(data.ab->data);
    if (data.fp8_ab_checkpoint) {
      args.fp8_ab_data = at<std::uint8_t>(data.ab->data, 0);
      args.fp8_ab_scales = at<float>(data.ab->data, data.fp8_ab_sf_offset);
    }
    args.checkpoint.gates = static_cast<float *>(data.gates->data);
    args.checkpoint.headers =
        static_cast<LocalP2pRouteHeader *>(data.headers->data);
    args.saved_rows = data.rows;
    if (weight.fp8.w1 && data.fp8_x) {
      args.fp8_x_data = at<std::uint8_t>(data.fp8_x->data, 0);
      args.fp8_x_scales = at<float>(data.fp8_x->data, data.fp8_sf_offset);
    }
  };
  if (!allocate_training_tape)
    bind_storage();
  args.retain_backward_state = config.retain_backward_state;
  args.forward_task_claim_offset = layout.forward_task_claim;
  args.forward_fc1_done_offset = layout.forward_fc1_done;
  args.x_arrived_rows_offset = layout.forward_x_arrived_rows;
  args.forward_reduce_route_count_offset = layout.forward_reduce_route_count;
  args.forward_reduce_ready_token_offset = layout.forward_reduce_ready_token;
  args.forward_return_rows_offset = layout.forward_return_rows;
  args.forward_return_queues_offset = layout.forward_return_queues;
  args.local_notification_offset = layout.local_notifications;
  args.local_notification_ctas = layout.local_notification_ctas;
  // Cross-node Forward has a single Return transport: FULL GIN. Dispatch
  // intentionally remains same-rail, while the second physical context set
  // gives each permanent RDMA CTA an independent FULL Return QP.
  if (use_gin && !gin_backend->full_connections)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "FULL Return connections", id.calling_rank,
                    "NCCL_GIN_CONNECTION_FULL", "unavailable");
  // Scheduler selection belongs to the generated tuning header. Runtime
  // environment overrides made one binary silently represent several kernels
  // and defeated the content-addressed JIT cache.
  args.forward_dynamic_reduce_scan = forward_tuning::kReduceScheduler == 1U;
  args.forward_dynamic_reduce_queue = forward_tuning::kReduceScheduler == 0U;
  args.forward_scheduler_offset = layout.forward_scheduler;
  if constexpr (forward_tuning::kUnifiedWorkers)
    args.gin_sender_done = at<void>(slot.device, layout.gin_sender_done);
  args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
  args.terminal_epoch_offset = layout.terminal_epoch;
  args.production_error_records_offset = layout.production_error_records;
  args.cancel_epoch =
      reinterpret_cast<const unsigned long long *>(slot.device_cancel_epoch);
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
  constexpr std::size_t forward_progress_base = 0;
  if (std::getenv("WEFTMOE_FORWARD_PROGRESS_DUMP") != nullptr) {
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
  internal::throw_cuda(
      "forward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "cudaDevAttrClusterLaunch");
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
  cudaError_t launch_forward = cudaSuccess;
  const auto &ordinary_output_store_map = slot.forward_output_store_map;
  args.forward_dynamic_reduce_queue = false;
  args.forward_dynamic_reduce_scan = false;
  args.forward_compact_return = false;
  auto forward_main_compute_ctas = forward_tuning::kStaticV4CtaCount
                                       ? forward_tuning::kStaticV4CtaCount
                                       : (grid_blocks & ~1U);
  args.forward_compute_cta_count = forward_main_compute_ctas;
  args.forward_reduce_cta_count = forward_main_compute_ctas;
  if (network_prefill && forward_main_compute_ctas < weftmoe_pf::kNetworkOwners)
    throw std::invalid_argument("PF network requires a CTA per network owner");
  auto kStaticUnified = weight.fp8.w1 ? weftmoe_pf::local::kernel<true>
                                      : weftmoe_pf::local::kernel<false>;
  const bool decode = dt;
  if (network_prefill)
    kStaticUnified = weight.fp8.w1 ? weftmoe_pf::network::kernel<true>
                                   : weftmoe_pf::network::kernel<false>;
  const auto decode_kernel = weight.fp8.w1 ? weftmoe_dt::kernel<true>
                                            : weftmoe_dt::kernel<false>;
  auto threads = decode ? (weight.fp8.w1 ? weftmoe_dt::kKernelThreads<true>
                                        : weftmoe_dt::kKernelThreads<false>)
                        : forward_tuning::kUnifiedThreads;
  if (network_prefill)
    threads = weftmoe_pf::network::kThreads;
  if (decode && (id.num_tokens >= 4U || weftmoe_dt::kCtaMultiplier)) {
    // More residency helps many-expert decode, but a small expert
    // set should not pay the grid collective cost of mostly idle CTAs.
    const auto expert_tiles = view.sizes.owned_expert_count *
                              (weftmoe_dt::kFc1Tiles + weftmoe_dt::kFc2Tiles);
    forward_main_compute_ctas *=
        weftmoe_dt::kCtaMultiplier
            ? weftmoe_dt::kCtaMultiplier
            : std::clamp(expert_tiles / forward_main_compute_ctas, 1U, 2U);
    args.forward_compute_cta_count = forward_main_compute_ctas;
    args.forward_reduce_cta_count = forward_main_compute_ctas;
  }
  if (decode && weight.fp8.w1 && weftmoe_dt::kFP8Ctas) {
    // The resident-cluster clamp below remains mandatory for rank barriers.
    forward_main_compute_ctas = weftmoe_dt::kFP8Ctas;
    args.forward_compute_cta_count = forward_main_compute_ctas;
    args.forward_reduce_cta_count = forward_main_compute_ctas;
  }
  auto smem = decode ? (weight.fp8.w1
                            ? weftmoe_dt::shared_bytes<true>(id.num_experts, view.sizes.owned_expert_count)
                            : weftmoe_dt::shared_bytes<false>(id.num_experts, view.sizes.owned_expert_count))
                     : sizeof(weftmoe_pf::local::Shared);
  const bool pdl = programmatic_launch::enabled();
  if (network_prefill)
    smem = weight.fp8.w1 ? weftmoe_pf::local::kFP8SharedBytes : weftmoe_pf::network::kSharedBytes;
  else if (!decode && weight.fp8.w1 && weftmoe_pf::local::kN == 256U)
    smem = weftmoe_pf::local::kFP8SharedBytes;
  auto error =
      network_prefill
          ? backend->prepare_unified_launch<weftmoe_pf::network::Resources>(
                reinterpret_cast<const void *>(kStaticUnified), smem,
                forward_main_compute_ctas, threads,
                forward_tuning::kClusterSize, pdl, false, &max_active_clusters)
      : decode
          ? backend->prepare_unified_launch<>(
                reinterpret_cast<const void *>(decode_kernel), smem,
                forward_main_compute_ctas, threads,
                forward_tuning::kClusterSize, pdl, false, &max_active_clusters)
          : backend->prepare_unified_launch<weftmoe_pf::local::Resources>(
                reinterpret_cast<const void *>(kStaticUnified), smem,
                forward_main_compute_ctas, threads,
                forward_tuning::kClusterSize, pdl, false, &max_active_clusters);
  internal::throw_cuda("forward-run", error, id.calling_rank, id.world_size,
                       backend->session.session().identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "prepare cached V4 unified launch");
  if (decode) {
    forward_main_compute_ctas = std::min(
        forward_main_compute_ctas, static_cast<unsigned>(max_active_clusters) *
                                       forward_tuning::kClusterSize);
    args.forward_compute_cta_count = forward_main_compute_ctas;
    args.forward_reduce_cta_count = forward_main_compute_ctas;
  }
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(forward_main_compute_ctas, 1, 1);
  launch.blockDim = dim3(threads, 1, 1);
  launch.dynamicSmemBytes = smem;
  launch.stream = exec->stream;
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim = {forward_tuning::kClusterSize, 1U, 1U};
  cudaLaunchAttribute attributes[2]{};
  unsigned attribute_count = 0U;
  attributes[attribute_count++] = cluster;
  if (pdl)
    attributes[attribute_count++] = programmatic_launch::attribute();
  launch.attrs = attributes;
  launch.numAttrs = attribute_count;
  if (static_cast<unsigned>(max_active_clusters) *
          forward_tuning::kClusterSize <
      forward_main_compute_ctas)
    transport_error(NotifyErrorCode::BackendUnavailable, "forward-run",
                    "V4 unified resident grid", id.calling_rank,
                    std::to_string(forward_main_compute_ctas) +
                        " resident CTAs",
                    std::to_string(max_active_clusters) + " resident clusters");
  if (standalone_notify) {
    const bool overlap_source =
        source_fp8 && !network_prefill && id.num_nodes == 1U;
    const auto source_stream =
        overlap_source ? slot.backward_transport_stream : exec->stream;
    const auto source_check = [&](cudaError_t error, const char *what) {
      internal::throw_cuda(
          "fp8-source", error, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, what);
    };
    const auto quantize_source = [&] {
      if (id.num_tokens) {
        const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
            (id.num_tokens + 3U) / 4U, 4U * sms));
        auto *destination = network_prefill
            ? at<std::uint8_t>(slot.gin_window, args.static_network.input)
            : at<std::uint8_t>(slot.device, layout.input);
        if (activation.prequantized) {
          fp8::copy_source_x<forward_tuning::kMainHidden>
              <<<blocks, 128, 0, source_stream>>>(
                  static_cast<const std::uint8_t *>(activation.fp8.data.data),
                  static_cast<const float *>(activation.fp8.scales.data),
                  destination, id.num_tokens);
        } else {
          fp8::prepare_source_x<forward_tuning::kMainHidden>
              <<<blocks, 128, 0, source_stream>>>(
                  reinterpret_cast<const __nv_bfloat16 *>(activation.pointer),
                  destination, id.num_tokens);
        }
        source_check(cudaGetLastError(), "launch source preparation");
      }
      // Empty sources also join and publish the same rank phase.
      if (overlap_source)
        source_check(cudaEventRecord(slot.backward_join_transport, source_stream),
                     "record source quantization complete");
    };
    if (source_fp8) {
      source_preparation_guard.stream = source_stream;
      source_preparation_guard.planner_stream = exec->stream;
      if (overlap_source) {
        source_check(cudaEventRecord(slot.backward_roles_ready, exec->stream),
                     "record source prerequisites");
        source_check(cudaStreamWaitEvent(source_stream, slot.backward_roles_ready),
                     "fork source quantization");
      } else {
        quantize_source();
      }
    }
    args.training_counts_ready = !split_notify;
    const auto notify = network_prefill ? weftmoe_pf::notify_kernel<true>
                                        : weftmoe_pf::notify_kernel<false>;
    constexpr std::size_t notify_smem = 64U * 1024U;
    int notify_clusters = 0;
    auto error = backend->prepare_unified_launch<>(
        reinterpret_cast<const void *>(notify), notify_smem,
        forward_main_compute_ctas, threads, forward_tuning::kClusterSize, pdl,
        false, &notify_clusters);
    if (error == cudaSuccess &&
        unsigned(notify_clusters) * forward_tuning::kClusterSize <
            forward_main_compute_ctas)
      error = cudaErrorLaunchOutOfResources;
    internal::throw_cuda("split-pf-notify", error, id.calling_rank,
                         id.world_size, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "prepare standalone resident planner");
    auto notify_launch = launch;
    notify_launch.blockDim = dim3(threads);
    notify_launch.dynamicSmemBytes = notify_smem;
    internal::throw_cuda(
        "split-pf-notify", cudaLaunchKernelEx(&notify_launch, notify, args),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "launch standalone PF planner");
    if (overlap_source) {
      // Submit the planner first so its independent work can overlap quantization.
      quantize_source();
      source_check(cudaStreamWaitEvent(exec->stream, slot.backward_join_transport),
                   "join source quantization");
      // A local event is insufficient: every peer packet must be published
      // before Main can pull it. Queue this before the host count download.
      weftmoe_pf::source_ready_kernel<<<1, 128, 0, exec->stream>>>(args);
      source_check(cudaGetLastError(), "publish source packets");
    }
    if (split_notify) {
      ActivationTransportBridge::finish_training_notify(dispatch, consumer);
      view.sizes = args.dispatch.sizes = dispatch.sizes();
    }
  }
  if (allocate_training_tape) {
    prepare_storage();
    bind_storage();
  }
  if (decode)
    launch_forward = cudaLaunchKernelEx(
        &launch, decode_kernel, args,
        weight.fp8.w1 ? slot.decode_fp8_x_map : slot.decode_x_map,
        weight.fp8.w1 ? slot.decode_fp8_s_map : slot.decode_s_map,
        weight.decode_w1_map, weight.decode_w2_map);
  else
    launch_forward = cudaLaunchKernelEx(
        &launch, kStaticUnified, args,
        weight.fp8.w1 ? slot.forward_fp8_x_map : slot.forward_x_map,
        weight.fp8.w1 ? slot.forward_fp8_s_map : slot.forward_s_map,
        weight.fc1_w1_map, weight.w2_map, ordinary_output_store_map);
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("kernel_launch_returned");
#endif
  internal::throw_cuda("forward-run", launch_forward, id.calling_rank,
                       id.world_size, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "single persistent WGMMA/TMA forward launch");
  if (args.fp8_x_data)
    tensors.impl_->fp8_x_columns_queued = false;

  const auto &data = *tensors.impl_;
  if (config.retain_backward_state && !training_ring) {
    const auto save =
        [&](const std::shared_ptr<tensor_storage_detail::Allocation> &target,
            std::size_t offset, std::uint64_t row_bytes) {
          const auto bytes = dispatch.received_row_capacity() * row_bytes;
          if (!bytes)
            return;
          internal::throw_cuda(
              "forward-save",
              cudaMemcpyAsync(target->data, at<void>(slot.device, offset),
                              bytes, cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "save caller-owned Forward state");
        };
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
    // The communication probe skips FC1 and cannot checkpoint X in compute.
    save(data.x, layout.canonical_rows,
         config.hidden_size * sizeof(transport::BFloat16));
#endif
    save(data.gates, layout.canonical_gates, sizeof(float));
    save(data.headers, layout.direct_route_headers,
         sizeof(LocalP2pRouteHeader));
  }
  if (!serialized_graph_replay)
    tensors.impl_->completion.record(exec->stream);
  if (!static_v4_candidate)
    internal::throw_cuda("forward-run",
                         internal::record_observable_event(
                             slot.forward_output_ready, exec->stream),
                         id.calling_rank, id.world_size,
                         backend->session.session().identity(), id.generation,
                         ExchangeKind::ExpertCounts,
                         "cudaEventRecord(Forward output ready)");
  // Local P2P relies on the peer-visible terminal epoch. GIN payload and
  // return consumers have already acquired their per-generation signals;
  // the next Notify/reset agreement is the registered-window reuse boundary.
  if (!static_v4_candidate)
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.forward_complete, exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaEventRecord(forward complete)");
  if (record)
    record->record(at<TransportDeviceStatus>(slot.device, layout.status),
                   exec->stream);
  if (!serialized_graph_replay) {
    internal::throw_cuda(
        "forward-run",
        internal::record_observable_event(slot.complete, exec->stream),
        id.calling_rank, id.world_size, backend->session.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "record workspace completion");
    slot.completion_recorded = true;
  }

  if (record)
    record->publish(exec->stream);
#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_enter");
#endif
  auto forward_stream_done = std::make_shared<std::atomic<bool>>(false);
  if (!static_v4_candidate)
    ActivationTransportBridge::submit(consumer);

#if WEFTMOE_DEBUG_VALIDATE
  debug_forward_host_phase("consumer_submit_done");
#endif

  auto result = std::make_shared<DeviceForwardHandle::Impl>();
  result->slot = lease;
  result->workspace_index = lease->index;
  result->tensors = tensors;
  result->pool = state;
  result->backend = backend;
  result->dispatch = dispatch;
  result->activations = activations;
  result->weights = weights;
  result->inference_routes = config.inference_routes;
  result->inference_gates = config.inference_gates;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->retained_backward_state = config.retain_backward_state;
  if (training_ring)
    result->pf_args = args;
  result->serialized_graph_replay = serialized_graph_replay;
  result->compact_return = args.forward_compact_return;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->completion = std::make_unique<OperatorCompletion>();
  result->completion->record = record;
  result->completion
      ->finish = [backend, state, flight, lease, id, options, args, use_gin,
                  cluster_supported, max_active_clusters, forward_stream_done,
                  static_v4_candidate, serialized_graph_replay, record,
#if WEFTMOE_DEBUG_VALIDATE
                  layout,
#endif
                  result = result.get(),
                  consumer_owner =
                      std::make_shared<DeviceDispatchConsumerLease>(
                          std::move(consumer))]() {
    auto &slot = state->slots[lease->index];
    auto exec = internal::checked_execution_state(backend->execution);
    auto &consumer = *consumer_owner;
    SlotFailureGuard failure_guard{lease, false};
#if WEFTMOE_DEBUG_VALIDATE
    const auto debug_forward_host_phase = [](const char *) {};
#endif
    bool cancelled = false;
    if (record) {
      record->wait();
    } else {
      // The slot lease keeps this mapped word unique until finish completes.
      // Publish only after the status copies and CUDA completion events above.
      // The default stream write includes a system-scope fence; it does not
      // depend on scheduling a driver callback thread to wake the consumer.
      __atomic_store_n(slot.host_forward_done, 0U, __ATOMIC_RELEASE);
      const auto write_status = cuStreamWriteValue32(
          exec->stream, reinterpret_cast<CUdeviceptr>(slot.device_forward_done),
          1U, CU_STREAM_WRITE_VALUE_DEFAULT);
      const bool mapped_completion = write_status == CUDA_SUCCESS;
      if (!mapped_completion && write_status != CUDA_ERROR_NOT_SUPPORTED) {
        const char *text = nullptr;
        (void)cuGetErrorString(write_status, &text);
        transport_error(NotifyErrorCode::CudaFailure, "forward-run",
                        "stream completion write", id.calling_rank,
                        "cuStreamWriteValue32 success",
                        text ? text : "unknown CUDA driver error");
      }
      if (!mapped_completion)
        internal::throw_cuda("forward-run",
                             cudaLaunchHostFunc(exec->stream,
                                                publish_stream_completion,
                                                forward_stream_done.get()),
                             id.calling_rank, id.world_size,
                             backend->session.session().identity(),
                             id.generation, ExchangeKind::ExpertCounts,
                             "cudaLaunchHostFunc(Forward completion fallback)");
      const auto wait_start = std::chrono::steady_clock::now();
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
      unsigned int progress_dump_count = 0U;
#endif
      while (true) {
        if (mapped_completion
                ? __atomic_load_n(slot.host_forward_done, __ATOMIC_ACQUIRE) !=
                      0U
                : forward_stream_done->load(std::memory_order_acquire))
          break;
        const auto expired = options.deadline_after.count() > 0 &&
                             std::chrono::steady_clock::now() - wait_start >=
                                 options.deadline_after;
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
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
          // snapshot. If a platform cannot schedule the diagnostic D2H while
          // the persistent grid is resident, this line still identifies the
          // stalled phase instead of losing the entire diagnostic.
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
          // hide the race being diagnosed. A pair of copy-engine snapshots on
          // the independent cancel stream remains observable while the
          // persistent kernel occupies every SM.
          auto *completion_snapshot = slot.host_forward_progress +
                                      forward_progress_base +
                                      kForwardProgressWords;
          internal::throw_cuda(
              "forward-progress-snapshot",
              cudaMemcpyAsync(
                  completion_snapshot,
                  at<void>(slot.device, layout.forward_completion),
                  sizeof(transport_state::GenerationCompletionVector),
                  cudaMemcpyDeviceToHost, state->cancel_stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts,
              "D2H Forward generation completion snapshot");
          internal::throw_cuda(
              "forward-progress-snapshot",
              cudaMemcpyAsync(
                  completion_snapshot +
                      transport_state::kGenerationCompletionWords,
                  at<void>(slot.device, layout.forward_context_completion),
                  static_cast<std::size_t>(slot.generation_debug_contexts) *
                      sizeof(transport_state::ContextCompletionRecord),
                  cudaMemcpyDeviceToHost, state->cancel_stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts,
              "D2H Forward context completion snapshot");
          internal::throw_cuda("forward-progress-snapshot",
                               cudaStreamSynchronize(state->cancel_stream),
                               id.calling_rank, id.world_size,
                               backend->session.session().identity(),
                               id.generation, ExchangeKind::ExpertCounts,
                               "drain Forward completion snapshot");
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
          __atomic_store_n(slot.host_cancel_epoch, args.epoch,
                           __ATOMIC_RELEASE);
          internal::throw_cuda(
              "forward-cancel",
              cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                              sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                              state->cancel_stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "H2D publish forward cancel");
          internal::throw_cuda(
              "forward-cancel", cudaStreamSynchronize(state->cancel_stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "drain forward cancel stream");
          slot.cancel_epoch_dirty = true;
          cancelled = true;
        }
        // Do not call CUDA event-query APIs while the resident GIN kernel is
        // active. On the remote host that can hold a driver-side progress lock
        // and form a liveness cycle with device GIN. The stream-ordered mapped
        // word (or callback fallback) signals completion without querying that
        // lock. Synchronous wait stays runnable: a nominal 50us nanosleep can
        // delay completion by tens of milliseconds under host scheduling
        // jitter. enqueue() remains asynchronous; only a caller waiting for its
        // result occupies a host thread here. Cancellation/deadline checks stay
        // above.
#if defined(__x86_64__) || defined(__i386__)
        asm volatile("pause");
#elif defined(__aarch64__)
        asm volatile("yield");
#else
        std::atomic_signal_fence(std::memory_order_acq_rel);
#endif
      }
#if WEFTMOE_DEBUG_VALIDATE
      debug_forward_host_phase("stream_completion_observed");
#endif
      internal::throw_cuda(
          "forward-run",
          cudaMemcpyAsync(slot.host_forward_status,
                          at<void>(slot.device, state->layout.status),
                          sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                          exec->stream),
          id.calling_rank, id.world_size, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts, "D2H forward status");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(TransportDeviceStatus));
      internal::throw_cuda(
          "forward-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "complete D2H forward status");
      if (serialized_graph_replay) {
        result->tensors.impl_->completion.record(exec->stream);
        internal::throw_cuda(
            "forward-run",
            internal::record_observable_event(slot.complete, exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts,
            "record serialized graph workspace completion");
        slot.completion_recorded = true;
        internal::throw_cuda("forward-run", cudaEventSynchronize(slot.complete),
                             id.calling_rank, id.world_size,
                             backend->session.session().identity(),
                             id.generation, ExchangeKind::ExpertCounts,
                             "complete serialized graph workspace event");
      }
      const auto completion_event =
          static_v4_candidate ? slot.complete : slot.forward_complete;
      internal::throw_cuda("forward-run", cudaEventQuery(completion_event),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventQuery(completed Forward event)");
    }
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("event_verified");
#endif
    ActivationTransportBridge::finish_completed(consumer);
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("consumer_finished");
    debug_forward_host_phase("terminal_agreement_enter");
#endif
    const auto production_agreement = agree_production_terminal_status(
        backend, slot, ProductionStatusPhase::Forward, id.generation,
        id.generation, id.calling_rank, id.world_size,
        record ? record->host->status : *slot.host_forward_status,
        static_v4_candidate ? slot.complete : slot.forward_complete);
    const auto &status =
        record ? record->host->status : *slot.host_forward_status;
#if WEFTMOE_DEBUG_VALIDATE
    debug_forward_host_phase("terminal_agreement_done");
#endif
    if (cancelled || status.error == 2U)
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
    if (!static_v4_candidate)
      internal::throw_cuda("forward-timing",
                           cudaEventElapsedTime(&total_ms, slot.forward_start,
                                                slot.forward_output_ready),
                           id.calling_rank, id.world_size,
                           backend->session.session().identity(), id.generation,
                           ExchangeKind::ExpertCounts,
                           "cudaEventElapsedTime(forward total)");
    result->audit.specialization =
        ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
    result->audit.persistent_kernel_launches = 1;
    result->audit.wgmma_instructions =
        status.wgmma_instructions;
    result->audit.tma_loads = status.tma_loads;
    result->audit.tma_multicast_loads =
        status.tma_multicast_loads;
    result->audit.runtime_cluster_supported = cluster_supported != 0;
    result->audit.requested_cluster_size = forward_tuning::kClusterSize;
    result->audit.runtime_cluster_size =
        status.runtime_cluster_size;
    result->audit.observed_cluster_rank_mask =
        status.cluster_rank_mask;
    result->audit.max_active_clusters =
        static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
    result->audit.fc1_tasks = status.fc1_tasks_done;
    result->audit.fc2_tasks = status.fc2_tasks_done;
    result->audit.actual_dispatch_peer_bytes =
        status.actual_dispatch_peer_bytes;
    result->audit.actual_return_peer_bytes =
        status.actual_return_peer_bytes;
    result->audit.gin_dispatch_put_count =
        status.gin_dispatch_put_count;
    result->audit.raw_cross_node_route_bytes =
        status.raw_cross_node_payload_bytes;
    result->audit.gin_dispatch_payload_bytes =
        status.deduplicated_cross_node_payload_bytes;
    result->audit.gin_return_payload_bytes =
        status.gin_return_payload_bytes;
    result->audit.gin_return_put_count =
        status.gin_return_put_count;
    result->audit.gin_signal_count = status.gin_signal_count;
    result->audit.gin_metadata_bytes =
        status.gin_metadata_bytes;
    result->audit.workspace_bytes = state->layout.total;
    result->timings.terminal_status_agreement_us =
        production_agreement.elapsed_us;
    const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
      return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
    };
    result->timings.device_kernel_us =
        elapsed(status.start_clock, status.complete_clock);
    result->timings.total_us = static_v4_candidate
                                   ? result->timings.device_kernel_us
                                   : static_cast<double>(total_ms) * 1000.0;
    result->timings.route_exchange_us =
        elapsed(status.route_exchange_start_clock,
                status.route_exchange_complete_clock);
    result->timings.dispatch_us =
        elapsed(status.route_exchange_complete_clock
                    ? status.route_exchange_complete_clock
                    : status.start_clock,
                status.dispatch_complete_clock);
    result->timings.fc1_swiglu_us =
        elapsed(status.fc1_start_clock, status.fc1_complete_clock);
    result->timings.fc2_us =
        elapsed(status.dispatch_blocks_done, status.fc2_complete_clock);
    result->timings.before_combine_us =
        elapsed(status.before_combine_start_clock,
                status.before_combine_complete_clock);
    result->timings.combine_return_us =
        elapsed(status.relay_blocks_done, status.return_complete_clock);
    result->timings.reduce_us =
        elapsed(status.reduced_tokens, status.reduce_blocks_done);
    result->timings.after_combine_us = elapsed(
        status.after_combine_start_clock, status.after_combine_complete_clock);
    result->timings.fc1_fc2_overlap_us =
        elapsed(status.dispatch_blocks_done, status.fc1_complete_clock);
    result->timings.fc2_return_overlap_us =
        elapsed(status.relay_blocks_done, status.fc2_complete_clock);
    result->timings.return_reduce_overlap_us =
        elapsed(status.reduced_tokens, status.return_complete_clock);
    failure_guard.committed = true;
    result->activations = {};
    if (!result->tensors.impl_->completion.capture)
      result->slot.reset();
  };
  failure_guard.committed = true;
  communication.record();
  if (record) {
    lease->release_on(exec->stream);
    result->slot.reset();
  }
  source_preparation_guard.stream = nullptr;
  return DeviceForwardHandle(std::move(result));
}

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  auto handle = enqueue(dispatch, activations, weights, config, options);
  handle.wait();
  return handle;
}

const ForwardTensorStorage &DeviceForwardHandle::tensors() const {
  (void)identity();
  return impl_->tensors;
}

void DeviceForwardHandle::wait() const {
  (void)identity();
  impl_->completion->wait();
}

bool DeviceForwardHandle::ready() const {
  (void)identity();
  return impl_->completion->ready();
}

void DeviceForwardHandle::wait_on(CUstream_st *stream) const {
  (void)identity();
  impl_->tensors.impl_->completion.wait_on(stream);
}

void DeviceForwardHandle::wait_workspace_on(CUstream_st *stream) const {
  (void)identity();
  const auto &slot = impl_->pool->slots[impl_->workspace_index];
  internal::throw_cuda(
      "forward-workspace-wait", cudaStreamWaitEvent(stream, slot.complete),
      impl_->identity.rank, impl_->identity.world,
      impl_->backend->session.session().identity(), impl_->identity.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(workspace complete)");
}

void DeviceForwardHandle::release_serialized_graph_workspace() const {
  (void)identity();
  if (impl_->retained_backward_state || !impl_->serialized_graph_replay)
    throw std::runtime_error(
        "serialized graph workspace release requires captured inference");
  wait();
  impl_->slot.reset();
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
  wait();
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  wait();
  return impl_->timings;
}

namespace {

double backward_elapsed_us(cudaEvent_t begin, cudaEvent_t end,
                           const ActivationTransportExpectation &identity,
                           const char *name) {
  float milliseconds = 0.0F;
  internal::throw_cuda("backward-timing",
                       cudaEventElapsedTime(&milliseconds, begin, end),
                       identity.rank, identity.world, "backward",
                       identity.generation, ExchangeKind::ExpertCounts, name);
  return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace

BackwardTensorStorage BackwardMegaKernel::prepare_tensors(
    const DeviceForwardHandle &forward, const BackwardMegaKernelConfig &config,
    const BackwardTensorBuffers &bindings,
    const CudaTensorAllocator &allocator) const {
  if (!forward.impl_ || !forward.impl_->retained_backward_state)
    throw std::invalid_argument(
        "Backward tensors require a retained Forward tape");
  const auto &retained = *forward.impl_;
  if (config.fp32_weight_gradients && !retained.pf_args.fp8.w1)
    throw std::invalid_argument("FP32 weight gradients currently require FP8 compute");
  const auto depth = retained.pf_args.ring.depth;
  if (!weftmoe_pf::backward_stream::kEnabled || !depth ||
      (retained.pf_args.use_gin && !retained.pf_args.network_pull))
    throw std::invalid_argument(
        "Backward requires a current PF tape and cluster-N2 stream configuration");
  BackwardTensorStorage result;
  result.impl_ = BackwardTensorStorage::Impl::make(
      retained.dispatch, retained.backend->execution, config, bindings,
      allocator, depth,
      retained.pf_args.use_gin &&
          (weftmoe_pf::kNetworkPrivateDr || retained.pf_args.fp8.w1),
      retained.pf_args.fp8.w1 != nullptr,
      retained.pf_args.fp8_x_data != nullptr);
  return result;
}

DeviceBackwardHandle BackwardMegaKernel::enqueue(
    const DeviceForwardHandle &forward,
    const BFloat16OutputGradientLease &output_gradients,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) const {
  if (!forward.impl_ || !output_gradients.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "forward/output gradients", 0, "valid", "empty");
  const auto &retained = *forward.impl_;
  const auto &gradient = *output_gradients.impl_;
  const auto &identity = retained.identity;
  if (!retained.retained_backward_state || !retained.tensors ||
      !retained.pool || !retained.backend || !retained.weights.impl_ ||
      !gradient.allocation || !gradient.allocation->ready)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "retained Forward state", identity.rank,
                    "complete cached X/A/B state and ready dY", "missing");
  if (config.hidden_size != backward_tuning::kMainHidden ||
      config.intermediate_size != backward_tuning::kMainIntermediate ||
      config.hidden_size != identity.hidden ||
      config.intermediate_size != retained.intermediate)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "specialization shape", identity.rank,
                    "compiled Backward dimensions",
                    std::to_string(config.hidden_size) + "x" +
                        std::to_string(config.intermediate_size));
  if (gradient.generation != identity.generation ||
      gradient.tokens != identity.tokens ||
      gradient.hidden != identity.hidden ||
      gradient.device != retained.pool->device ||
      gradient.session_identity != retained.pool->session_identity)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "dY provenance", identity.rank,
                    "same generation/session/device/[tokens,H] as Forward",
                    "mismatch");
  if (options.cancellation.cancellation_requested() ||
      options.deadline_after.count() != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "cancellation/deadline", identity.rank,
                    "disabled for local correctness milestone", "requested");
  const auto local_backend =
      std::dynamic_pointer_cast<LocalP2pBackendState>(retained.backend);
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(retained.backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  // FP8 dispatch writes padded canonical packets, never the optional BF16
  // peer cache (which streamed Wgrad deliberately does not allocate).
  const bool private_dr = use_gin &&
      (weftmoe_pf::kNetworkPrivateDr || retained.pf_args.fp8.w1);
  const bool valid_local =
      local_backend && identity.world == local_backend->peer_slot_bases.size();
  const bool valid_gin =
      gin_backend && gin_backend->num_nodes >= 2U &&
      gin_backend->local_world_size == backward_tuning::kRdmaCtas &&
      gin_backend->gin_context_count == backward_tuning::kRdmaCtas &&
      gin_backend->full_connections;
  if (!valid_local && !valid_gin)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "transport backend", identity.rank,
                    "one-node LocalCudaP2p or multi-node 8-rail GIN retained "
                    "Forward",
                    "unsupported");
  const auto &pool = retained.pool;
  const auto &layout = pool->layout;
  auto lease = retained.slot
                   ? retained.slot
                   : acquire_slot(pool, identity.generation, identity.rank);
  if (!pool->config.enable_backward_state ||
      !pool->slots[lease->index].backward_peer_workspace)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "backward-run",
                    "Backward workspace", identity.rank,
                    "training-enabled pool with owned expert capacity",
                    "missing");
  const auto &dispatch_id = retained.dispatch.identity();
  if (dispatch_id.top_k > 32U)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "top_k", identity.rank,
                    "at most one route per warp lane (32)",
                    std::to_string(dispatch_id.top_k));
  const auto expectation =
      DeviceDispatchViewExpectation{dispatch_id.generation,
                                    dispatch_id.calling_rank,
                                    dispatch_id.world_size,
                                    dispatch_id.num_tokens,
                                    dispatch_id.top_k,
                                    dispatch_id.config_digest_word0,
                                    dispatch_id.config_digest_word1};
  auto consumer = retained.dispatch.acquire_consumer(
      retained.backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (view.sizes.incoming_row_count != retained.incoming_rows ||
      view.sizes.owned_expert_count != retained.weights.impl_->owned_experts)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "retained Dispatch geometry", identity.rank,
                    "same rows/experts as Forward", "mismatch");

  auto exec = internal::checked_execution_state(retained.backend->execution);

  auto tensors =
      config.tensors ? config.tensors : prepare_tensors(forward, config);
  const auto &data = *tensors.impl_;
  const auto &saved = *retained.tensors.impl_;
  if (data.tokens != identity.tokens || data.hidden != config.hidden_size ||
      data.intermediate != config.intermediate_size ||
      data.top_k != dispatch_id.top_k ||
      data.experts != view.sizes.owned_expert_count ||
      data.rows < view.sizes.incoming_row_count ||
      data.padded_rows <
          ActivationTransportBridge::padded_received_rows(retained.dispatch) ||
      data.pf_depth != retained.pf_args.ring.depth ||
      data.streamed_fp8_wgrad != (retained.pf_args.fp8_x_data != nullptr) ||
      data.private_dr != private_dr || data.completion.device != pool->device)
    throw std::runtime_error(
        "Backward tensor storage does not match this invocation");
  const auto writes = {data.dx, data.dgate, data.dw1, data.dw2, data.workspace};
  for (const auto &input :
       {saved.output, saved.x, saved.ab, saved.gates, saved.headers})
    if (input)
    tensor_storage_detail::validate_input_disjoint(writes, input->data,
                                                   input->bytes);
  if (saved.fp8_x)
    tensor_storage_detail::validate_input_disjoint(writes, saved.fp8_x->data,
                                                  saved.fp8_x->bytes);
  if (gradient.prequantized) {
    if (!retained.pf_args.fp8_source_x)
      throw std::invalid_argument("Prequantized dY requires an FP8 PF tape");
    for (const auto *buffer : {&gradient.fp8.data, &gradient.fp8.scales})
      tensor_storage_detail::validate_input_disjoint(writes, buffer->data, buffer->bytes);
  } else {
    tensor_storage_detail::validate_input_disjoint(writes, gradient.pointer,
                                                   data.dx->bytes);
  }
  if (retained.weights.impl_->fp8.w1) {
    const auto &b = retained.weights.impl_->fp8_buffers;
    for (const auto *buffer : {&b.w1, &b.s1, &b.w2, &b.s2, &b.w1_t, &b.s1_t, &b.w2_t, &b.s2_t})
      tensor_storage_detail::validate_input_disjoint(writes, buffer->data, buffer->bytes);
  } else {
    tensor_storage_detail::validate_input_disjoint(writes, retained.weights.impl_->w1, data.dw1->bytes);
    tensor_storage_detail::validate_input_disjoint(writes, retained.weights.impl_->w2, data.dw2->bytes);
  }

  auto &slot = pool->slots[lease->index];
  GinExecution communication(pool->gin_resources, exec->stream);
  auto record = pool->config.stream_ordered_training
                    ? pool->training_completions->acquire()
                    : nullptr;
  auto flight = std::make_shared<OperatorFlight>(lease, true);
  auto &peer_workspace = *slot.backward_peer_workspace;
  if (!private_dr)
    peer_workspace.prepare(identity.generation, view.sizes.incoming_row_count,
                           checked_mul(identity.tokens, dispatch_id.top_k,
                                       "Backward source routes"),
                           config.hidden_size, !data.streamed_fp8_wgrad);
  SlotFailureGuard failure_guard{lease, false};
  const auto scheduler_mblock_capacity =
      (layout.backward_padded_capacity_rows + backward_tuning::kTileM - 1ULL) /
      backward_tuning::kTileM;
  if (scheduler_mblock_capacity > UINT32_MAX)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "backward-run",
                    "scheduler M-block capacity", identity.rank, "uint32",
                    std::to_string(scheduler_mblock_capacity));
  const auto scheduler_layout =
      pipeline::backward::make_backward_scheduler_layout(
          scheduler_mblock_capacity,
          static_cast<std::uint32_t>(config.hidden_size /
                                     backward_tuning::kDxTileN /
                                     backward_tuning::kActivationClusterN),
          pool->config.max_owned_experts);
  const auto scheduler_capacity_words =
      (layout.forward_fc1_done - layout.forward_task_claim) /
      sizeof(unsigned long long);
  if (scheduler_layout.total_words > scheduler_capacity_words)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "backward-run",
                    "aliased Backward scheduler words", identity.rank,
                    std::to_string(scheduler_capacity_words),
                    std::to_string(scheduler_layout.total_words));
  BackwardKernelArgs args{};
  args.dispatch = view;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  // Actual padded rows and active experts are loaded from the device directory.
  args.scheduler_mblock_capacity =
      static_cast<std::uint32_t>(scheduler_mblock_capacity);
  args.scheduler_expert_capacity = pool->config.max_owned_experts;
  args.rank = identity.rank;
  args.world = identity.world;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.saved_gates = static_cast<const float *>(saved.gates->data);
  args.saved_ab = saved.fp8_ab_checkpoint ? nullptr : static_cast<const transport::BFloat16 *>(saved.ab->data);
  if (saved.fp8_ab_checkpoint) {
    args.fp8_ab_data = at<const std::uint8_t>(saved.ab->data, 0);
    args.fp8_ab_scales = at<const float>(saved.ab->data, saved.fp8_ab_sf_offset);
  }
  args.fp8_du_ring = data.streamed_fp8_wgrad;
  args.saved_x = saved.x ? static_cast<const transport::BFloat16 *>(saved.x->data) : nullptr;
  args.fp8 = retained.weights.impl_->fp8.w1 != nullptr;
  if (data.fp8_compute != args.fp8)
    throw std::invalid_argument("Backward compute dtype does not match prepared storage");
  if (args.fp8) {
    args.fp8_wgrad_data = at<std::uint8_t>(data.workspace->data, data.fp8_wgrad_offset);
    args.fp8_wgrad_scales = at<float>(data.workspace->data, data.fp8_scales_offset);
    args.fp8_wgrad_ready = at<unsigned long long>(data.workspace->data, data.fp8_ready_offset);
    args.fp8_wgrad_rows = data.padded_rows;
    if (data.streamed_fp8_wgrad) {
      args.fp8_x_data = retained.pf_args.fp8_x_data;
      args.fp8_x_scales = retained.pf_args.fp8_x_scales;
      if (!saved.fp8_x_columns_queued) {
        args.fp8_x_row_data = args.fp8_x_data;
        args.fp8_x_row_scales = args.fp8_x_scales;
      }
    }
  }
  args.fp32_weight_gradients = config.fp32_weight_gradients;
  if (data.fp32_weight_gradients != config.fp32_weight_gradients ||
      (config.fp32_weight_gradients && !args.fp8))
    throw std::invalid_argument("Backward weight-gradient dtype does not match prepared storage");
  args.saved_rows = saved.rows;
  args.peer_payloads = private_dr ? nullptr : peer_workspace.device_payloads();
  args.dr = private_dr
                ? at<transport::BFloat16>(data.workspace->data, data.dr_offset)
                : peer_workspace.payloads[identity.rank].dr;
  args.recomputed_u =
      at<transport::BFloat16>(data.workspace->data, data.u_offset);
  args.dab = at<transport::BFloat16>(data.workspace->data, data.dab_offset);
  args.dgate_rows = at<float>(data.workspace->data, data.dgate_rows_offset);
  args.dgate = static_cast<float *>(data.dgate->data);
  auto dab_map = data.dab_map;
  auto pf_args = retained.pf_args;
#if WEFTMOE_DEVICE_TIMELINE
  pf_args.timeline = {};
#endif
  pf_args.source_input_offset = layout.input;
  // Reuse the source packet protocol for dY. The streamed column cache is
  // required because this Pull path does not materialize canonical BF16 dR.
  pf_args.fp8_source_x = WEFTMOE_FP8_SOURCE_DY && WEFTMOE_FP8_PACKED_RING &&
      args.fp8 && args.fp8_x_data &&
      forward_tuning::kFc1TileN == 256U;
  if (gradient.prequantized && !pf_args.fp8_source_x)
    throw std::invalid_argument("Prequantized dY requires source-FP8 Backward");
  pf_args.input = gradient.pointer;
  pf_args.training_counts_ready = true;
  pf_args.saved_x = args.dr;
  pf_args.output = static_cast<transport::BFloat16 *>(data.dx->data);
  args.activation_stride = std::max<std::uint64_t>(
      config.hidden_size, 2ULL * config.intermediate_size);
  args.fp8_activation_pitch = pool->config.fp8_workspace
      ? fp8::source_pitch(args.activation_stride) : args.activation_stride * 2U;
  if (use_gin && !args.fp8) {
    dab_map = make_bf16_tma_map(
        args.dab, 2ULL * config.intermediate_size, data.padded_rows, 128U,
        "PF Backward canonical dAB", 64U);
  } else if (2U * config.intermediate_size <= config.hidden_size) {
    args.dx_rows =
        at<transport::BFloat16>(slot.device, layout.canonical_rows);
    dab_map = make_bf16_tma_map(args.dx_rows, config.hidden_size,
                                std::uint64_t(data.pf_depth) * 128U, 128U,
                                "PF Backward shared dY/dAB ring", 64U);
  } else {
    args.dx_rows =
        at<transport::BFloat16>(data.workspace->data, data.pf_dab_offset);
  }

  if (args.fp8)
    dab_map = make_fp8_tma_map(args.dx_rows, 2U * config.intermediate_size,
        std::uint64_t(data.pf_depth) * 128U, 128U, "FP8 dAB ring", args.fp8_activation_pitch,
        WEFTMOE_FP8_PACKED_RING && forward_tuning::kFc1TileN == 256U);

  args.dw1 = static_cast<transport::BFloat16 *>(data.dw1->data);
  args.dw2 = static_cast<transport::BFloat16 *>(data.dw2->data);
  args.direct_headers_offset = layout.direct_route_headers;
  args.compact_to_padded_offset = layout.backward_compact_to_padded;
  args.padded_expert_rows_offset = layout.backward_padded_expert_rows;
  args.padded_expert_offsets_offset = layout.backward_padded_expert_offsets;
  args.setup_epoch_offset = layout.backward_setup_epoch;
  args.scheduler_offset = layout.forward_task_claim;
  args.status_offset = layout.status;
  // Compute queues use an execution epoch. Wire-slot sequences are separate
  // and remain monotonic across Forward, Backward and graph replays.
  if (slot.transport_epoch >= (1ULL << 32U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "backward-run",
                    "transport_epoch", identity.rank, "<2^32-1", "exhausted");
  const auto backward_epoch = ++slot.transport_epoch;
  args.epoch = backward_epoch;
  args.runtime_epoch =
      at<unsigned long long>(slot.device, layout.execution_epoch);
  if (std::getenv("WEFTMOE_BACKWARD_PROGRESS") != nullptr) {
    std::cerr << "BACKWARD_ADDR rank=" << identity.rank
              << " epoch=" << backward_epoch
              << " status=" << at<void>(slot.device, layout.status)
              << " host_status=" << static_cast<void *>(slot.host_status)
              << " scheduler=" << at<void>(slot.device, args.scheduler_offset)
              << " mblock_capacity=" << args.scheduler_mblock_capacity
              << " expert_capacity=" << args.scheduler_expert_capacity;
    std::cerr << std::endl;
  }

  auto transport_stream = slot.backward_transport_stream;
  int sm_count = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "query Backward SM count");
  const auto x_map = args.fp8_x_data ? CUtensorMap{} : make_bf16_tma_map_mn(
      saved.rows && saved.x ? saved.x->data : at<void>(slot.device, layout.canonical_rows),
      config.hidden_size, std::max<std::uint64_t>(saved.rows, 1),
      "saved X for dW1");
  const auto dw1_map = config.fp32_weight_gradients ? CUtensorMap{} :
      make_bf16_tma_store_map(data.dw1->data, config.hidden_size,
                              static_cast<std::uint64_t>(data.experts) * 2ULL *
                                  config.intermediate_size,
                              "caller dW1");
  const auto dw2_map = config.fp32_weight_gradients ? CUtensorMap{} : make_bf16_tma_store_map(
      data.dw2->data, config.intermediate_size,
      static_cast<std::uint64_t>(data.experts) * config.hidden_size,
      "caller dW2");
  const auto compute_ctas = static_cast<std::uint32_t>(sm_count) & ~1U;

  int cluster_supported = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "query Backward cluster support");
  if (!cluster_supported || compute_ctas < 2U)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "Backward cluster extent", identity.rank,
                    "cluster launch support and at least two compute SMs",
                    std::to_string(compute_ctas));
  saved.completion.wait_on(exec->stream);
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.backward_roles_ready, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward caller-stream dependency");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(transport_stream, slot.backward_roles_ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait Backward caller-stream dependency");
  const bool overlap_source = pf_args.fp8_source_x && !use_gin &&
                              pf_args.dispatch.identity.num_nodes == 1U;
  const auto source_stream = overlap_source ? exec->stream : transport_stream;
  internal::throw_cuda(
      "backward-run",
      internal::wait_input_ready(source_stream, gradient.allocation->ready,
                                 gradient.allocation->ready_capture_id),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait dY producer");
  // Keep dY alive on host-side launch failures, as for source X preparation.
  struct SourcePreparationGuard {
    cudaStream_t stream = nullptr;
    cudaStream_t planner_stream = nullptr;
    ~SourcePreparationGuard() {
      if (stream) {
        (void)cudaStreamSynchronize(stream);
        if (planner_stream && planner_stream != stream)
          (void)cudaStreamSynchronize(planner_stream);
      }
    }
  } source_preparation_guard;
  const auto source_check = [&](cudaError_t error, const char *what) {
    internal::throw_cuda("fp8-source-dy", error, identity.rank, identity.world,
                         "backward", identity.generation,
                         ExchangeKind::ExpertCounts, what);
  };
  const auto quantize_source = [&] {
    if (pf_args.fp8_source_x && identity.tokens) {
      source_preparation_guard.stream = source_stream;
      source_preparation_guard.planner_stream = transport_stream;
      const auto blocks = static_cast<unsigned>(std::min<std::uint64_t>(
          (identity.tokens + 3U) / 4U, 4U * sm_count));
      auto *destination = use_gin
                  ? at<std::uint8_t>(slot.gin_window, pf_args.static_network.input)
                  : at<std::uint8_t>(slot.device, layout.input);
      if (gradient.prequantized) {
        fp8::copy_source_x<backward_tuning::kMainHidden>
            <<<blocks, 128, 0, source_stream>>>(
                static_cast<const std::uint8_t *>(gradient.fp8.data.data),
                static_cast<const float *>(gradient.fp8.scales.data),
                destination, identity.tokens);
      } else {
        fp8::prepare_source_x<backward_tuning::kMainHidden>
            <<<blocks, 128, 0, source_stream>>>(
                reinterpret_cast<const __nv_bfloat16 *>(gradient.pointer),
                destination, identity.tokens);
      }
      source_check(cudaGetLastError(), "launch source dY preparation");
    }
    if (overlap_source)
      source_check(cudaEventRecord(slot.backward_join_compute, source_stream),
                   "record dY quantization complete");
  };
  if (!overlap_source)
    quantize_source();
  // Only count/route metadata belongs to the tape. Rebuild the physical plan
  // from retained counts; local FP8 dY is published separately after the join.
  const auto planner = use_gin ? weftmoe_pf::notify_kernel<true, true>
                              : weftmoe_pf::notify_kernel<false>;
  constexpr std::size_t planner_smem = 64U * 1024U;
  int planner_clusters = 0;
  auto planner_error = retained.backend->prepare_unified_launch<>(
      reinterpret_cast<const void *>(planner), planner_smem, compute_ctas,
      384U, 2U, false, false, &planner_clusters);
  if (planner_error == cudaSuccess &&
      unsigned(planner_clusters) * 2U < compute_ctas)
    planner_error = cudaErrorLaunchOutOfResources;
  internal::throw_cuda("backward-plan", planner_error, identity.rank,
                       identity.world, "backward", identity.generation,
                       ExchangeKind::ExpertCounts, "prepare PF planner");
  cudaLaunchAttribute planner_cluster{};
  planner_cluster.id = cudaLaunchAttributeClusterDimension;
  planner_cluster.val.clusterDim = {2U, 1U, 1U};
  cudaLaunchConfig_t planner_launch{};
  planner_launch.gridDim = dim3(compute_ctas);
  planner_launch.blockDim = dim3(384U);
  planner_launch.dynamicSmemBytes = planner_smem;
  planner_launch.stream = transport_stream;
  planner_launch.attrs = &planner_cluster;
  planner_launch.numAttrs = 1U;
  internal::throw_cuda(
      "backward-plan", cudaLaunchKernelEx(&planner_launch, planner, pf_args),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "restore current tape PF plan and stage dY");
  if (overlap_source) {
    quantize_source();
    source_check(cudaStreamWaitEvent(transport_stream, slot.backward_join_compute),
                 "join dY quantization");
    weftmoe_pf::source_ready_kernel<<<1, 128, 0, transport_stream>>>(pf_args);
    source_check(cudaGetLastError(), "publish source dY packets");
  }
  if (use_gin) {
    backward_build_row_directory_kernel<<<1, 256, 0, transport_stream>>>(
        args);
    weftmoe_pf::backward_stream::prepare<true>
        <<<dim3(128U, view.sizes.owned_expert_count), 256, 0,
           transport_stream>>>(pf_args, args);
    weftmoe_pf::backward_stream::
        prepare_network<<<compute_ctas, 256, 0, transport_stream>>>(pf_args,
                                                                    args);
    const auto blocks =
        ActivationTransportBridge::padded_received_rows(retained.dispatch) /
        backward_tuning::kTileM;
    if (blocks)
      weftmoe_pf::backward_stream::
          order_windows<<<(blocks + 255U) / 256U, 256, 0, transport_stream>>>(
              args);
  } else {
    weftmoe_pf::backward_stream::prepare<false>
        <<<dim3(128U, view.sizes.owned_expert_count), 256, 0,
           transport_stream>>>(pf_args, args);
    backward_setup_local_kernel<<<1, 256, 0, transport_stream>>>(args);
  }
  internal::throw_cuda("backward-plan", cudaGetLastError(), identity.rank,
                       identity.world, "backward", identity.generation,
                       ExchangeKind::ExpertCounts,
                       "prepare PF Backward directory and input epoch");

  internal::throw_cuda("backward-run",
                       internal::record_observable_event(
                           record ? record->start : slot.backward_start,
                           transport_stream, &slot.backward_capture_id),
                       identity.rank, identity.world, "backward",
                       identity.generation, ExchangeKind::ExpertCounts,
                       "record Backward start");

  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_roles_ready, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward launch prerequisites");
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(record ? record->compute
                                               : slot.backward_compute_start,
                                        transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward compute start");

  const auto kernel = args.fp8
      ? (use_gin ? weftmoe_pf::backward_stream::kernel<true, true>
                 : weftmoe_pf::backward_stream::kernel<false, true>)
      : (use_gin ? weftmoe_pf::backward_stream::kernel<true, false>
                 : weftmoe_pf::backward_stream::kernel<false, false>);
  const auto smem = use_gin ? (args.fp8 ? weftmoe_pf::backward_stream::kFP8NetworkSharedBytes
                                             : weftmoe_pf::backward_stream::kNetworkSharedBytes)
      : args.fp8 ? weftmoe_pf::backward_stream::kFP8SharedBytes
                 : weftmoe_pf::backward_stream::kSharedBytes;
  const auto entry = reinterpret_cast<const void *>(kernel);
  int clusters = 0;
  internal::throw_cuda(
      "backward-run",
      retained.backend
          ->prepare_unified_launch<backward_tuning::UnifiedResources>(
              entry, smem, compute_ctas, backward_tuning::kUnifiedThreads, 2U,
              false, false, &clusters),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "prepare unified Backward launch");
  cudaLaunchAttribute cluster{};
  cluster.id = cudaLaunchAttributeClusterDimension;
  cluster.val.clusterDim = {2U, 1U, 1U};
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(compute_ctas);
  launch.blockDim = dim3(backward_tuning::kUnifiedThreads);
  launch.dynamicSmemBytes = smem;
  launch.stream = transport_stream;
  launch.attrs = &cluster;
  launch.numAttrs = 1U;
  if (2U * static_cast<unsigned>(clusters) < compute_ctas)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "unified Backward residency", identity.rank,
                    "all worker CTAs resident", std::to_string(clusters));
  const auto dr_rows =
      std::max(view.sizes.incoming_row_count, std::uint64_t{1});
  const auto dr_map =
      use_gin && args.fp8
          ? make_fp8_tma_map(args.dr, config.hidden_size, std::max<std::uint64_t>(128U, data.padded_rows),
                            128U, "FP8 canonical dR", fp8::source_pitch(config.hidden_size), true)
      : private_dr
          ? make_bf16_tma_map(args.dr, config.hidden_size, dr_rows,
                              backward_tuning::kActivationTileM,
                              "private Backward dR", backward_tuning::kTileK)
          : peer_workspace.buffer->dr_map;
  const auto dr_mn_map = args.fp8 ? CUtensorMap{} :
      private_dr ? make_bf16_tma_map_mn(args.dr, config.hidden_size, dr_rows,
                                        "private Backward dR MN")
                 : peer_workspace.buffer->dr_mn_map;
  const auto launch_result = cudaLaunchKernelEx(
      &launch, kernel, pf_args, args,
      use_gin ? dr_map : args.fp8 ? slot.forward_fp8_x_map : slot.forward_x_map,
      dab_map, retained.weights.impl_->backward_w2_map,
      retained.weights.impl_->backward_w1_map, dr_mn_map, data.u_mn_map,
      data.dab_mn_map, x_map, dw2_map, dw1_map);
  internal::throw_cuda("backward-run", launch_result, identity.rank,
                       identity.world, "backward", identity.generation,
                       ExchangeKind::ExpertCounts,
                       "launch unified Backward workers");
  // First use converts each complete block in place. Later uses of this tape
  // reuse its column layout; stream joins below order them after this launch.
  if (args.fp8_x_row_data)
    saved.fp8_x_columns_queued = true;
  for (auto event : {slot.backward_transport_done,
                     record ? record->computed : slot.backward_compute_done,
                     slot.backward_join_compute})
    internal::throw_cuda(
        "backward-run",
        internal::record_observable_event(event, transport_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "record unified Backward completion");

  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_join_transport, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward transport dependency");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.backward_join_compute),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward compute roles");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.backward_join_transport),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "join late Backward compute workers and reduction");
  if (data.dgate->bytes && !use_gin)
    internal::throw_cuda(
        "backward-output",
        cudaMemcpyAsync(
            data.dgate->data, peer_workspace.payloads[identity.rank].dgate,
            data.dgate->bytes, cudaMemcpyDeviceToDevice, exec->stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "publish caller-owned dGate");
  tensors.impl_->completion.record(exec->stream);
  if (record) {
    record->record(at<TransportDeviceStatus>(slot.device, layout.status),
                   exec->stream);
  } else {
    internal::throw_cuda(
        "backward-run",
        cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                        sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                        exec->stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts, "download Backward pipeline clocks");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              sizeof(TransportDeviceStatus));
  }
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(
          record ? record->complete : slot.backward_complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward complete");
  internal::throw_cuda(
      "backward-run",
      internal::record_observable_event(slot.complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record pool slot complete");
  slot.completion_recorded = true;
  if (record)
    record->publish(exec->stream);
  if (!record)
    ActivationTransportBridge::submit(consumer);
  auto result = std::make_shared<DeviceBackwardHandle::Impl>();
  result->slot = lease;
  result->workspace_index = lease->index;
  result->tensors = tensors;
  result->pool = pool;
  result->forward = forward;
  result->output_gradients = output_gradients;
  result->identity = identity;
  result->owned_experts = view.sizes.owned_expert_count;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->completion = std::make_unique<OperatorCompletion>();
  result->completion->record = record;
  result->completion
      ->finish = [forward, pool, flight, lease, identity, config, args, use_gin,
                  backward_epoch, record,
                  scheduler_words = scheduler_layout.total_words,
                  result = result.get(),
                  consumer_owner =
                      std::make_shared<DeviceDispatchConsumerLease>(
                          std::move(consumer))]() {
    forward.wait();
    const auto &retained = *forward.impl_;
    auto &slot = pool->slots[lease->index];
    const auto &layout = pool->layout;
    auto &consumer = *consumer_owner;
    SlotFailureGuard failure_guard{lease, false};
    const bool diagnostic_progress =
        !record && std::getenv("WEFTMOE_BACKWARD_PROGRESS") != nullptr;
    const auto diagnostic_status = [&](const TransportDeviceStatus &s,
                                       bool pending) {
      std::ostringstream record;
      record << "BACKWARD_STATUS rank=" << identity.rank
             << " epoch=" << backward_epoch << " pending=" << pending
             << " return_puts=" << s.gin_return_put_count
             << " return_bytes=" << s.gin_return_payload_bytes
             << " start=" << s.start_clock << " du_start=" << s.fc1_start_clock
             << " du_complete=" << s.fc1_complete_clock
             << " dx_start=" << s.relay_blocks_done
             << " dispatch_complete=" << s.dispatch_complete_clock
             << " dx_complete=" << s.fc2_complete_clock
             << " return_complete=" << s.return_complete_clock
             << " dw2_start=" << s.backward_dw2_start_clock
             << " dw2_complete=" << s.backward_dw2_complete_clock
             << " dw1_start=" << s.backward_dw1_start_clock
             << " dw1_complete=" << s.backward_dw1_complete_clock
             << " reduce_start=" << s.backward_reduce_start_clock
             << " reduce_complete=" << s.backward_reduce_complete_clock
             << " sender_complete=" << s.backward_sender_complete_clock
             << " transport_complete=" << s.backward_transport_complete_clock
             << " return_send_complete="
             << s.backward_return_send_complete_clock
             << " return_receive_complete="
             << s.backward_return_receive_complete_clock
#if defined(WEFTMOE_PF_BWD_TASK_TIMING)
             << " du_gemm_complete=" << s.before_combine_complete_clock
             << " du_service_ns=" << s.fc1_tasks_done
             << " dx_service_ns=" << s.fc2_tasks_done
             << " swiglu_service_ns=" << s.synthetic_ticks
             << " du_ready_wait_ns=" << s.dispatch_put_ticks
             << " dx_ready_wait_ns=" << s.relay_ticks
#endif
#if __has_include(<nccl_device.h>)
             << " gin_contexts="
             << (use_gin ? retained.pf_args.gin_dev_comm.ginContextCount : 0)
#endif
             << std::endl;
      std::cerr << record.str();
    };
    if (diagnostic_progress) {
      const auto started = std::chrono::steady_clock::now();
      bool sampled = false;
      while (cudaEventQuery(slot.backward_complete) == cudaErrorNotReady) {
        if (!sampled &&
            std::chrono::steady_clock::now() - started >
                std::chrono::seconds(3) &&
            slot.forward_progress_words * sizeof(std::uint64_t) >=
                sizeof(TransportDeviceStatus)) {
          internal::throw_cuda(
              "backward-diagnostic",
              cudaMemcpyAsync(slot.host_forward_progress,
                              at<void>(slot.device, layout.status),
                              sizeof(TransportDeviceStatus),
                              cudaMemcpyDeviceToHost, pool->cancel_stream),
              identity.rank, identity.world, "backward", identity.generation,
              ExchangeKind::ExpertCounts, "snapshot pending Backward status");
          internal::throw_cuda(
              "backward-diagnostic", cudaStreamSynchronize(pool->cancel_stream),
              identity.rank, identity.world, "backward", identity.generation,
              ExchangeKind::ExpertCounts, "drain Backward status snapshot");
          diagnostic_status(*reinterpret_cast<const TransportDeviceStatus *>(
                                slot.host_forward_progress),
                            true);
          for (std::uint64_t offset = 0; offset < scheduler_words;) {
            const auto count = std::min<std::uint64_t>(
                slot.forward_progress_words, scheduler_words - offset);
            internal::throw_cuda(
                "backward-diagnostic",
                cudaMemcpyAsync(
                    slot.host_forward_progress,
                    at<std::uint64_t>(slot.device, args.scheduler_offset) +
                        offset,
                    count * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                    pool->cancel_stream),
                identity.rank, identity.world, "backward", identity.generation,
                ExchangeKind::ExpertCounts, "snapshot Backward scheduler");
            internal::throw_cuda("backward-diagnostic",
                                 cudaStreamSynchronize(pool->cancel_stream),
                                 identity.rank, identity.world, "backward",
                                 identity.generation,
                                 ExchangeKind::ExpertCounts,
                                 "drain Backward scheduler snapshot");
            std::cerr << "BACKWARD_SCHED rank=" << identity.rank
                      << " epoch=" << backward_epoch << " offset=" << offset
                      << " values=";
            for (std::uint64_t i = 0; i < count; ++i)
              std::cerr << (i ? "," : "") << slot.host_forward_progress[i];
            std::cerr << std::endl;
            offset += count;
          }
          sampled = true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(100));
      }
    }
    if (record)
      record->wait();
    else
      internal::throw_cuda(
          "backward-run", cudaEventSynchronize(slot.backward_complete),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "wait Backward completion");
    const auto &status = record ? record->host->status : *slot.host_status;
    if (diagnostic_progress || (record && std::getenv("WEFTMOE_BACKWARD_PROGRESS")))
      diagnostic_status(status, false);
    ActivationTransportBridge::finish_completed(consumer);

    result->audit.specialization =
        BackwardKernelSpecialization::Sm90CachedXAb7168x3072;
    result->audit.gemm_launches = 2ULL;
    result->audit.actual_dispatch_peer_bytes =
        retained.audit.actual_dispatch_peer_bytes;
    const auto remote_return_rows =
        retained.audit.actual_return_peer_bytes /
        (config.hidden_size * sizeof(transport::BFloat16));
    result->audit.actual_return_peer_bytes =
        use_gin ? remote_return_rows * pool->gin_layout.rings.return_row_bytes
                : retained.audit.actual_return_peer_bytes +
                      remote_return_rows * sizeof(float);
    result->audit.gin_return_put_count = status.gin_return_put_count;
    result->audit.gin_return_payload_bytes = status.gin_return_payload_bytes;
    result->audit.workspace_bytes = pool->layout.total;
    result->audit.hot_path_allocations = 0;
    const auto device_span_us = [](unsigned long long begin,
                                   unsigned long long end) {
      return begin == ~0ULL || end == 0ULL || end < begin
                 ? 0.0
                 : static_cast<double>(end - begin) / 1000.0;
    };
    result->timings.prepare_us = backward_elapsed_us(
        record ? record->start : slot.backward_start,
        record ? record->compute : slot.backward_compute_start, identity,
        "Backward cache preparation elapsed");
    const auto primary_compute_us = backward_elapsed_us(
        record ? record->compute : slot.backward_compute_start,
        record ? record->computed : slot.backward_compute_done, identity,
        "Backward primary persistent compute elapsed");

    result->timings.compute_us = primary_compute_us;
    // The embedded reduction has device clocks; an adjacent host event
    // would measure only the empty stream interval after the kernel.
    result->timings.reduce_us =
        device_span_us(status.backward_reduce_start_clock,
                       status.backward_reduce_complete_clock);

    result->timings.transport_us = backward_elapsed_us(
        record ? record->compute : slot.backward_compute_start,
        record ? record->computed : slot.backward_transport_done, identity,
        "Backward persistent transport elapsed");
    result->timings.dispatch_us =
        device_span_us(status.start_clock, status.dispatch_complete_clock);
    result->timings.dab_us =
        device_span_us(status.fc1_start_clock, status.fc1_complete_clock);
    result->timings.dx_us =
        device_span_us(status.relay_blocks_done, status.fc2_complete_clock);
    result->timings.dw2_us = device_span_us(status.backward_dw2_start_clock,
                                            status.backward_dw2_complete_clock);
    result->timings.dw1_us = device_span_us(status.backward_dw1_start_clock,
                                            status.backward_dw1_complete_clock);
    result->timings.combine_us =
        device_span_us(status.fc2_complete_clock, status.return_complete_clock);
    const auto dw_after_dx = device_span_us(status.fc2_complete_clock,
                                            status.backward_dw1_complete_clock);
    result->timings.dw_combine_overlap_us =
        std::min(dw_after_dx, result->timings.combine_us);
    result->timings.total_us =
        backward_elapsed_us(record ? record->start : slot.backward_start,
                            record ? record->complete : slot.backward_complete,
                            identity, "Backward total elapsed");
    failure_guard.committed = true;
    if (!result->tensors.impl_->completion.capture) {
      result->slot.reset();
      result->forward = {};
      result->output_gradients = {};
#if !WEFTMOE_DEVICE_TIMELINE
      result->pool.reset();
#endif
    }
  };
  failure_guard.committed = true;
  communication.record();
  if (record) {
    lease->release_on(exec->stream);
    result->slot.reset();
  }
  source_preparation_guard.stream = nullptr;
  return DeviceBackwardHandle(std::move(result));
}

DeviceBackwardHandle
BackwardMegaKernel::run(const DeviceForwardHandle &forward,
                        const BFloat16OutputGradientLease &gradients,
                        const BackwardMegaKernelConfig &config,
                        const ActivationTransportRunOptions &options) const {
  auto handle = enqueue(forward, gradients, config, options);
  handle.wait();
  return handle;
}

const BackwardTensorStorage &DeviceBackwardHandle::tensors() const {
  (void)identity();
  return impl_->tensors;
}

void DeviceBackwardHandle::wait() const {
  (void)identity();
  impl_->completion->wait();
}

bool DeviceBackwardHandle::ready() const {
  (void)identity();
  return impl_->completion->ready();
}

void DeviceBackwardHandle::wait_on(CUstream_st *stream) const {
  (void)identity();
  impl_->tensors.impl_->completion.wait_on(stream);
}

const ActivationTransportExpectation &DeviceBackwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle", "handle",
                    0, "valid", "empty");
  return impl_->identity;
}

const BackwardMegaKernelAudit &DeviceBackwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle", "handle",
                    0, "valid", "empty");
  wait();
  return impl_->audit;
}

const BackwardMegaKernelTimings &DeviceBackwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle", "handle",
                    0, "valid", "empty");
  wait();
  return impl_->timings;
}
