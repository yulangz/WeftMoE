namespace {

double backward_elapsed_us(cudaEvent_t begin, cudaEvent_t end,
                           const ActivationTransportExpectation &identity,
                           const char *name) {
  float milliseconds = 0.0F;
  internal::throw_cuda(
      "backward-timing", cudaEventElapsedTime(&milliseconds, begin, end),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, name);
  return static_cast<double>(milliseconds) * 1000.0;
}

} // namespace

DeviceBackwardHandle BackwardMegaKernel::run(
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
  if (!retained.retained_backward_state || !retained.slot || !retained.pool ||
      !retained.backend || !retained.weights.impl_ ||
      !gradient.allocation || !gradient.allocation->ready)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "retained Forward state", identity.rank,
                    "complete cached X/A/B state and ready dY", "missing");
  if (config.hidden_size != 7168ULL ||
      config.intermediate_size != 3072ULL ||
      config.hidden_size != identity.hidden ||
      config.intermediate_size != retained.intermediate)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "specialization shape", identity.rank, "7168x3072",
                    std::to_string(config.hidden_size) + "x" +
                        std::to_string(config.intermediate_size));
  if (gradient.generation != identity.generation ||
      gradient.tokens != identity.tokens || gradient.hidden != identity.hidden ||
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
  const bool valid_local =
      local_backend && identity.world == local_backend->peer_slot_bases.size();
  const bool valid_gin =
      gin_backend && gin_backend->num_nodes == 2U &&
      gin_backend->local_world_size == backward_tuning::kRdmaCtas &&
      gin_backend->gin_context_count == backward_tuning::kRdmaCtas &&
      gin_backend->full_connections;
  if (!valid_local && !valid_gin)
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "transport backend", identity.rank,
                    "one-node LocalCudaP2p or two-node 8-rail GIN retained "
                    "Forward",
                    "unsupported");
  const auto &pool = retained.pool;
  const auto &layout = pool->layout;
  if (!pool->config.enable_backward_state ||
      layout.backward_dab_rows == 0 || layout.backward_dx_rows == 0 ||
      layout.backward_dw1 == 0 || layout.backward_dw2 == 0 ||
      !pool->slots[retained.slot->index].backward_maps_valid)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "Backward workspace", identity.rank,
                    "training-enabled pool with owned expert capacity",
                    "missing");
  const auto &dispatch_id = retained.dispatch.identity();
  const auto expectation = DeviceDispatchViewExpectation{
      dispatch_id.generation, dispatch_id.calling_rank,
      dispatch_id.world_size, dispatch_id.num_tokens, dispatch_id.top_k,
      dispatch_id.config_digest_word0, dispatch_id.config_digest_word1};
  auto consumer = retained.dispatch.acquire_consumer(retained.backend->execution,
                                                      expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  const auto dispatch_waves = ActivationTransportBridge::wave_view(consumer);
  if (view.sizes.incoming_row_count != retained.incoming_rows ||
      view.sizes.owned_expert_count != retained.weights.impl_->owned_experts)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "retained Dispatch geometry", identity.rank,
                    "same rows/experts as Forward", "mismatch");

  auto exec = internal::checked_execution_state(retained.backend->execution);
  {
    std::lock_guard<std::mutex> plan_lock(retained.backward_plan_mutex);
    if (retained.backward_expert_rows.empty()) {
      retained.backward_expert_rows.resize(view.sizes.owned_expert_count);
      retained.backward_expert_offsets.resize(
          static_cast<std::size_t>(view.sizes.owned_expert_count) + 1ULL);
      const auto rows_bytes = checked_size(
          checked_mul(view.sizes.owned_expert_count, sizeof(std::uint64_t),
                      "Backward expert row plan bytes"),
          "Backward expert row plan bytes");
      const auto offsets_bytes = checked_size(
          checked_mul(static_cast<std::uint64_t>(view.sizes.owned_expert_count) +
                          1ULL,
                      sizeof(std::uint64_t),
                      "Backward expert offset plan bytes"),
          "Backward expert offset plan bytes");
      internal::throw_cuda(
          "backward-plan",
          cudaMemcpyAsync(retained.backward_expert_rows.data(),
                          view.expert_row_count, rows_bytes,
                          cudaMemcpyDeviceToHost, exec->stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "D2H expert row counts");
      internal::throw_cuda(
          "backward-plan",
          cudaMemcpyAsync(retained.backward_expert_offsets.data(),
                          view.expert_row_prefix, offsets_bytes,
                          cudaMemcpyDeviceToHost, exec->stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "D2H expert row offsets");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                rows_bytes + offsets_bytes);
      internal::throw_cuda(
          "backward-plan", cudaStreamSynchronize(exec->stream), identity.rank,
          identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "cache expert row plan");
      retained.backward_padded_expert_rows.resize(
          view.sizes.owned_expert_count);
      retained.backward_padded_expert_offsets.resize(
          static_cast<std::size_t>(view.sizes.owned_expert_count) + 1ULL);
      retained.backward_compact_to_padded.resize(
          checked_size(view.sizes.incoming_row_count,
                       "Backward compact-to-padded entries"));
      std::uint64_t padded_cursor = 0ULL;
      for (std::uint32_t expert = 0U;
           expert < view.sizes.owned_expert_count; ++expert) {
        const auto rows = retained.backward_expert_rows[expert];
        const auto padded = checked_mul((rows + 127ULL) / 128ULL, 128ULL,
                                        "Backward padded expert rows");
        retained.backward_padded_expert_offsets[expert] = padded_cursor;
        retained.backward_padded_expert_rows[expert] = padded;
        const auto compact_begin = retained.backward_expert_offsets[expert];
        for (std::uint64_t local_row = 0ULL; local_row < rows; ++local_row) {
          retained.backward_compact_to_padded[checked_size(
              compact_begin + local_row,
              "Backward compact-to-padded index")] =
              static_cast<std::uint32_t>(padded_cursor + local_row);
        }
        padded_cursor = checked_add(padded_cursor, padded,
                                    "Backward padded row count");
      }
      retained.backward_padded_expert_offsets[view.sizes.owned_expert_count] =
          padded_cursor;
      retained.backward_padded_rows = padded_cursor;
      if (padded_cursor > layout.backward_padded_capacity_rows ||
          padded_cursor > UINT32_MAX)
        transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                        "backward-plan", "padded expert rows", identity.rank,
                        std::to_string(layout.backward_padded_capacity_rows),
                        std::to_string(padded_cursor));
    }
    if (!retained.backward_plan_uploaded) {
      const auto copy_plan = [&](std::size_t offset, const auto &values,
                                 const char *name) {
        if (values.empty())
          return;
        const auto bytes = checked_mul(values.size(), sizeof(values.front()),
                                       name);
        internal::throw_cuda(
            "backward-plan",
            cudaMemcpyAsync(
                at<void>(pool->slots[retained.slot->index].device, offset),
                            values.data(), checked_size(bytes, name),
                            cudaMemcpyHostToDevice, exec->stream),
            identity.rank, identity.world, "backward", identity.generation,
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                  bytes);
      };
      copy_plan(layout.backward_compact_to_padded,
                retained.backward_compact_to_padded,
                "upload Backward compact-to-padded directory");
      copy_plan(layout.backward_padded_expert_rows,
                retained.backward_padded_expert_rows,
                "upload Backward padded expert rows");
      copy_plan(layout.backward_padded_expert_offsets,
                retained.backward_padded_expert_offsets,
                "upload Backward padded expert offsets");
      internal::throw_cuda(
          "backward-plan", cudaStreamSynchronize(exec->stream), identity.rank,
          identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "publish padded Backward directory");
      retained.backward_plan_uploaded = true;
    }
  }

  auto &slot = pool->slots[retained.slot->index];
  SlotFailureGuard failure_guard{retained.slot, false};
  const auto scheduler_mblock_capacity =
      (layout.backward_padded_capacity_rows + backward_tuning::kTileM - 1ULL) /
      backward_tuning::kTileM;
  if (scheduler_mblock_capacity > UINT32_MAX)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "scheduler M-block capacity",
                    identity.rank, "uint32",
                    std::to_string(scheduler_mblock_capacity));
  const auto scheduler_layout =
      persistent_v3::backward::make_backward_scheduler_layout(
          scheduler_mblock_capacity,
          static_cast<std::uint32_t>(config.hidden_size /
                                     backward_tuning::kDxTileN),
          pool->config.max_owned_experts);
  const auto scheduler_capacity_words =
      (layout.forward_fc1_done - layout.forward_task_claim) /
      sizeof(unsigned long long);
  if (scheduler_layout.total_words > scheduler_capacity_words)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "aliased Backward scheduler words",
                    identity.rank, std::to_string(scheduler_capacity_words),
                    std::to_string(scheduler_layout.total_words));
  BackwardKernelArgs args{};
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.dy = gradient.pointer;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows = pool->config.max_destination_rows_per_rank;
  args.padded_capacity_rows = layout.backward_padded_capacity_rows;
  args.padded_rows = retained.backward_padded_rows;
  args.scheduler_words = scheduler_layout.total_words;
  args.scheduler_mblock_capacity =
      static_cast<std::uint32_t>(scheduler_mblock_capacity);
  args.scheduler_expert_capacity = pool->config.max_owned_experts;
  args.mblocks = static_cast<std::uint32_t>(
      retained.backward_padded_rows / backward_tuning::kTileM);
  args.active_experts = static_cast<std::uint32_t>(std::count_if(
      retained.backward_padded_expert_rows.begin(),
      retained.backward_padded_expert_rows.end(),
      [](std::uint64_t rows) { return rows != 0ULL; }));
  args.rank = identity.rank;
  args.world = identity.world;
  args.node = use_gin ? gin_backend->node : 0U;
  args.local_world_size =
      use_gin ? gin_backend->local_world_size : identity.world;
  args.gin_context_count =
      use_gin ? gin_backend->gin_context_count : 0U;
  args.max_tokens_per_rank = pool->config.max_tokens_per_rank;
  args.forward_return_chunk_rows = retained.forward_return_chunk_rows;
  args.use_gin = use_gin;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[retained.slot->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.gin_send_dispatch_directory_offset =
        pool->gin_layout.send_dispatch_directory;
    args.gin_receive_dispatch_directory_offset =
        pool->gin_layout.receive_dispatch_directory;
    args.gin_send_header_offset = pool->gin_layout.send_packet_headers;
    args.gin_receive_header_offset = pool->gin_layout.receive_packet_headers;
    args.gin_send_fanout_offset = pool->gin_layout.send_fanout;
    args.gin_receive_fanout_offset = pool->gin_layout.receive_fanout;
    args.gin_send_payload_offset = pool->gin_layout.send_payload;
    args.gin_receive_payload_offset = pool->gin_layout.receive_payload;
    args.gin_return_staging_offset = pool->gin_layout.return_staging;
    args.gin_return_receive_offset = pool->gin_layout.return_receive;
    args.gin_return_fragment_ready_offset =
        pool->gin_layout.return_fragment_ready;
    args.gin_return_row_bytes = pool->gin_layout.backward_return_row_bytes;
  }
  args.canonical_x_offset = layout.canonical_rows;
  args.canonical_gate_offset = layout.canonical_gates;
  args.cached_ab_offset = layout.forward_ab_cache;
  args.padded_x_offset = layout.backward_padded_x;
  args.padded_ab_offset = layout.backward_padded_ab;
  args.padded_dr_offset = layout.backward_padded_dr;
  args.recomputed_u_offset = layout.backward_u;
  args.dab_offset = layout.backward_dab_rows;
  args.dx_rows_offset = layout.backward_dx_rows;
  args.dgate_rows_offset = layout.backward_dgate_rows;
  args.dgate_output_offset = layout.backward_dgate_output;
  args.dw1_offset = layout.backward_dw1;
  args.dw2_offset = layout.backward_dw2;
  args.contribution_offset = layout.contribution;
  args.combined_dx_offset = layout.combined;
  args.direct_headers_offset = layout.direct_route_headers;
  args.compact_to_padded_offset = layout.backward_compact_to_padded;
  args.padded_expert_rows_offset = layout.backward_padded_expert_rows;
  args.padded_expert_offsets_offset = layout.backward_padded_expert_offsets;
  args.setup_epoch_offset = layout.backward_setup_epoch;
  args.dispatch_ready_offset = layout.backward_dispatch_ready;
  args.return_ready_offset = layout.backward_return_ready;
  args.forward_output_ready_epoch_offset =
      layout.forward_output_ready_epoch;
  args.forward_return_chunk_state_offset =
      layout.forward_return_chunk_state;
  args.forward_return_source_route_index_offset =
      layout.forward_return_source_route_index;
  args.forward_direct_source_compact_index_offset =
      layout.forward_direct_source_compact_index;
  args.scheduler_offset = layout.forward_task_claim;
  args.status_offset = layout.status;
  auto backward_epoch = ++slot.backward_epoch;
  if (backward_epoch == 0ULL)
    backward_epoch = ++slot.backward_epoch;
  // FWD's retained packet headers use the low epoch namespace.  The high bit
  // prevents a first BWD launch from accepting an old FWD wave marker with the
  // same numeric slot epoch.
  args.epoch = backward_epoch | (1ULL << 63U);

  auto transport_stream = slot.forward_transport_stream;
  auto compute_stream = slot.forward_compute_stream;
  int sm_count = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "query Backward SM count");
  const auto transport_ctas =
      backward_tuning::kRelayCtas +
      (use_gin ? backward_tuning::kRdmaCtas : 0U);
  if (sm_count <= static_cast<int>(transport_ctas))
    transport_error(NotifyErrorCode::BackendUnavailable, "backward-run",
                    "split Backward role residency", identity.rank,
                    "more SMs than dedicated transport CTAs",
                    std::to_string(sm_count));
  constexpr std::uint32_t kActivationStorageTileN =
      backward_tuning::kDabTileN > backward_tuning::kDxTileN
          ? backward_tuning::kDabTileN
          : backward_tuning::kDxTileN;
  constexpr std::uint32_t kStorageTileN =
      kActivationStorageTileN > backward_tuning::kDwTileN
          ? kActivationStorageTileN
          : backward_tuning::kDwTileN;
  constexpr auto kPersistentSmem =
      backward_compute_detail::dynamic_shared_bytes<kStorageTileN>();
  internal::throw_cuda(
      "backward-run",
      cudaFuncSetAttribute(
          backward_local_transport_kernel,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kPersistentSmem)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "reserve one SM per Backward Relay CTA");
  internal::throw_cuda(
      "backward-run",
      cudaFuncSetAttribute(
          backward_local_compute_kernel<backward_tuning::kDabTileN,
                                        backward_tuning::kDxTileN,
                                        backward_tuning::kDwTileN>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(kPersistentSmem)),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts,
      "set persistent Backward compute shared memory");
  internal::throw_cuda(
      "backward-run", cudaStreamWaitEvent(transport_stream,
                                           slot.forward_complete),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait retained Forward completion");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(transport_stream, gradient.allocation->ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait dY producer");
  if (use_gin) {
    const auto marker_bytes = checked_size(
        checked_mul(pool->config.max_tokens_per_rank,
                    pool->config.max_top_k,
                    "Backward FULL Return marker rows") *
            sizeof(std::uint64_t),
        "Backward FULL Return marker bytes");
    internal::throw_cuda(
        "backward-run",
        cudaMemsetAsync(
            at<void>(slot.gin_window,
                     pool->gin_layout.return_fragment_ready),
            0, marker_bytes, transport_stream),
        identity.rank, identity.world, "backward", identity.generation,
        ExchangeKind::ExpertCounts,
        "reset Backward FULL Return fragment counters");
  }
  backward_setup_pipeline_kernel<<<1, backward_tuning::kSetupThreads, 0,
                                   transport_stream>>>(args);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "initialize fixed Backward pipeline state");
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.backward_start, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward start");
  if (view.sizes.owned_expert_count != 0U) {
    if (!retained.backward_cache_packed) {
      backward_pack_cached_state_kernel<<<
          view.sizes.owned_expert_count * backward_tuning::kCachePackChunks,
          backward_tuning::kCacheThreads, 0, transport_stream>>>(
          args, backward_tuning::kCachePackChunks);
      retained.backward_cache_packed = true;
    } else {
      backward_pack_gate_kernel<<<view.sizes.owned_expert_count,
                                  backward_tuning::kCacheThreads, 0,
                                  transport_stream>>>(args);
    }
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "materialize immutable Backward cache and per-run gates");
    // Every active expert writes every dW tile.  The dense training shape has
    // no inactive expert, so clearing several GiB immediately before those
    // stores is pure HBM traffic.  Sparse fixtures retain the full clear to
    // preserve the public zero-gradient contract for inactive experts.
    if (args.active_experts != view.sizes.owned_expert_count) {
      internal::throw_cuda(
          "backward-run",
          cudaMemsetAsync(at<void>(slot.device, layout.backward_dw1), 0,
                          checked_size(
                              static_cast<std::uint64_t>(
                                  view.sizes.owned_expert_count) *
                                  2ULL * config.intermediate_size *
                                  config.hidden_size *
                                  sizeof(transport::BFloat16),
                              "Backward dW1 clear bytes"),
                          transport_stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "clear sparse Backward dW1 output");
      internal::throw_cuda(
          "backward-run",
          cudaMemsetAsync(at<void>(slot.device, layout.backward_dw2), 0,
                          checked_size(
                              static_cast<std::uint64_t>(
                                  view.sizes.owned_expert_count) *
                                  config.hidden_size *
                                  config.intermediate_size *
                                  sizeof(transport::BFloat16),
                              "Backward dW2 clear bytes"),
                          transport_stream),
          identity.rank, identity.world, "backward", identity.generation,
          ExchangeKind::ExpertCounts, "clear sparse Backward dW2 output");
    }
  }
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.forward_roles_ready,
                                       transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "publish Backward cached rows");
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_compute_start, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward compute start");
  backward_local_transport_kernel<<<transport_ctas,
                                    backward_tuning::kComputeThreads,
                                    kPersistentSmem,
                                    transport_stream>>>(args);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "persistent local Backward Relay launch");
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_transport_done, transport_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward transport completion");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(compute_stream, slot.forward_roles_ready),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "start Backward compute roles");
  backward_local_compute_kernel<backward_tuning::kDabTileN,
                                backward_tuning::kDxTileN,
                                backward_tuning::kDwTileN>
      <<<static_cast<unsigned int>(sm_count) - transport_ctas,
         backward_tuning::kComputeThreads, kPersistentSmem,
         compute_stream>>>(args, slot.backward_dr_map, slot.backward_dab_map,
                           retained.weights.impl_->backward_w2_map,
                           retained.weights.impl_->backward_w1_map,
                           slot.backward_dr_mn_map,
                           slot.backward_u_mn_map,
                           slot.backward_dab_mn_map,
                           slot.backward_x_mn_map,
                           slot.backward_u_store_map,
                           slot.backward_dx_store_map,
                           slot.backward_dw2_store_map,
                           slot.backward_dw1_store_map);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), identity.rank, identity.world,
      "backward", identity.generation, ExchangeKind::ExpertCounts,
      "persistent priority DAB/DX/dW TMA-WGMMA launch");
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_compute_done, compute_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward compute completion");

  // Queue route reduction behind the persistent compute grid, not behind the
  // transport grid.  This guarantees all compute CTAs were resident (avoiding
  // waiting reducers stealing their SMs), then overlaps reduction with the
  // remaining FULL-return tail on the compute SMs that just became free.
  if (identity.tokens != 0ULL) {
    backward_reduce_dx_kernel<<<static_cast<unsigned int>(identity.tokens),
                                backward_tuning::kReduceThreads, 0,
                                compute_stream>>>(args);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), identity.rank, identity.world,
        "backward", identity.generation, ExchangeKind::ExpertCounts,
        "streaming Backward dX reduce launch");
  }
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.backward_reduce_done, compute_stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward reduction completion");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.backward_reduce_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward reduction");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, slot.backward_transport_done),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "join Backward transport roles");
  internal::throw_cuda(
      "backward-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "download Backward pipeline clocks");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.backward_complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record Backward complete");
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.complete, exec->stream),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "record pool slot complete");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "backward-run", cudaEventSynchronize(slot.backward_complete),
      identity.rank, identity.world, "backward", identity.generation,
      ExchangeKind::ExpertCounts, "wait Backward completion");
  consumer.finish();

  auto result = std::make_shared<DeviceBackwardHandle::Impl>();
  result->slot = retained.slot;
  result->pool = pool;
  result->forward = forward;
  result->output_gradients = output_gradients;
  result->identity = identity;
  result->owned_experts = view.sizes.owned_expert_count;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->audit.specialization =
      BackwardKernelSpecialization::Sm90CachedXAb7168x3072;
  result->audit.gemm_launches = 1ULL;
  result->audit.actual_dispatch_peer_bytes =
      retained.audit.actual_dispatch_peer_bytes;
  const auto remote_return_rows =
      retained.audit.actual_return_peer_bytes /
      (config.hidden_size * sizeof(transport::BFloat16));
  result->audit.actual_return_peer_bytes =
      use_gin ? remote_return_rows * args.gin_return_row_bytes
              : retained.audit.actual_return_peer_bytes +
                    remote_return_rows * sizeof(float);
  result->audit.gin_return_put_count = slot.host_status->gin_return_put_count;
  result->audit.gin_return_payload_bytes =
      slot.host_status->gin_return_payload_bytes;
  result->audit.workspace_bytes = pool->layout.total;
  result->audit.hot_path_allocations = 0;
  const auto device_span_us = [](unsigned long long begin,
                                 unsigned long long end) {
    return begin == ~0ULL || end == 0ULL || end < begin
               ? 0.0
               : static_cast<double>(end - begin) / 1000.0;
  };
  result->timings.prepare_us = backward_elapsed_us(
      slot.backward_start, slot.backward_compute_start, identity,
      "Backward cache preparation elapsed");
  result->timings.compute_us = backward_elapsed_us(
      slot.backward_compute_start, slot.backward_compute_done, identity,
      "Backward persistent compute elapsed");
  result->timings.reduce_us = backward_elapsed_us(
      slot.backward_compute_done, slot.backward_reduce_done, identity,
      "Backward dX reduction elapsed");
  result->timings.transport_us = backward_elapsed_us(
      slot.backward_compute_start, slot.backward_transport_done, identity,
      "Backward persistent transport elapsed");
  result->timings.dispatch_us = device_span_us(
      slot.host_status->start_clock,
      slot.host_status->dispatch_complete_clock);
  result->timings.dab_us = device_span_us(
      slot.host_status->fc1_start_clock,
      slot.host_status->fc1_complete_clock);
  result->timings.dx_us = device_span_us(
      slot.host_status->relay_blocks_done,
      slot.host_status->fc2_complete_clock);
  result->timings.dw2_us = device_span_us(
      slot.host_status->backward_dw2_start_clock,
      slot.host_status->backward_dw2_complete_clock);
  result->timings.dw1_us = device_span_us(
      slot.host_status->backward_dw1_start_clock,
      slot.host_status->backward_dw1_complete_clock);
  result->timings.combine_us = device_span_us(
      slot.host_status->fc2_complete_clock,
      slot.host_status->return_complete_clock);
  const auto dw_after_dx = device_span_us(
      slot.host_status->fc2_complete_clock,
      slot.host_status->backward_dw1_complete_clock);
  result->timings.dw_combine_overlap_us =
      std::min(dw_after_dx, result->timings.combine_us);
  result->timings.total_us = backward_elapsed_us(
      slot.backward_start, slot.backward_complete, identity,
      "Backward total elapsed");
  failure_guard.committed = true;
  return DeviceBackwardHandle(std::move(result));
}

const ActivationTransportExpectation &DeviceBackwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}

const BackwardMegaKernelAudit &DeviceBackwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}

const BackwardMegaKernelTimings &DeviceBackwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

BackwardValidationDownload BackwardValidationBridge::download(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  if (!handle.impl_ || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  internal::throw_cuda(
      "backward-download", cudaEventSynchronize(
                               impl.pool->slots[impl.slot->index].backward_complete),
      impl.identity.rank, impl.identity.world, "backward",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait Backward output");
  const auto &layout = impl.pool->layout;
  const auto &slot = impl.pool->slots[impl.slot->index];
  BackwardValidationDownload result;
  result.dx.resize(checked_size(
      checked_mul(impl.identity.tokens, impl.identity.hidden,
                  "Backward dX download elements"),
      "Backward dX download elements"));
  result.dgate.resize(checked_size(
      checked_mul(impl.identity.tokens,
                  impl.forward.impl_->dispatch.identity().top_k,
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
    internal::throw_cuda(
        "backward-download",
        cudaMemcpy(host, device, checked_size(bytes, name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "backward",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.dx.data(), at<const void>(slot.device, layout.combined),
       result.dx.size() * sizeof(transport::BFloat16), "download dX");
  copy(result.dgate.data(),
       at<const void>(slot.device, layout.backward_dgate_output),
       result.dgate.size() * sizeof(float), "download dGate");
  copy(result.dw1.data(), at<const void>(slot.device, layout.backward_dw1),
       result.dw1.size() * sizeof(transport::BFloat16), "download dW1");
  copy(result.dw2.data(), at<const void>(slot.device, layout.backward_dw2),
       result.dw2.size() * sizeof(transport::BFloat16), "download dW2");
  return result;
}
