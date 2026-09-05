ActivationTransportResourcePool make_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config) {
  if (!session || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-create",
                    "session/execution", 0, "valid", "empty");
  if (config.max_world_size < session.world_size())
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "max_world_size", 0,
                    ">= session world", std::to_string(config.max_world_size));
  auto state = std::make_shared<ActivationPoolState>();
  state->config = config;
  state->layout = make_layout(config);
  state->gin_layout = make_gin_window_layout(config);
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda("transport-pool-create",
                       cudaSetDevice(static_cast<int>(state->device)), 0,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  if (config.enable_backward_state) {
    internal::throw_cuda(
        "transport-pool-create",
        cudaFuncSetAttribute(
            backward_du_wgmma_kernel<backward_tuning::kDabTileN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(backward_compute_detail::dynamic_shared_bytes<
                             backward_tuning::kDabTileN>())),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts,
        "cudaFuncSetAttribute(Backward dU shared memory)");
    internal::throw_cuda(
        "transport-pool-create",
        cudaFuncSetAttribute(
            backward_dx_wgmma_kernel<backward_tuning::kDxTileN>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(backward_compute_detail::dynamic_shared_bytes<
                             backward_tuning::kDxTileN>())),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts,
        "cudaFuncSetAttribute(Backward dX shared memory)");
    throw_cublas("transport-pool-create",
                 cublasCreate(&state->backward_blas), 0,
                 "cublasCreate(Backward)");
    throw_cublas("transport-pool-create",
                 cublasSetAtomicsMode(state->backward_blas,
                                      CUBLAS_ATOMICS_NOT_ALLOWED),
                 0, "cublasSetAtomicsMode(Backward deterministic)");
  }
#if MEGAMOE_DEVICE_TIMELINE
  cudaDeviceProp timeline_device_properties{};
  internal::throw_cuda(
      "transport-pool-create",
      cudaGetDeviceProperties(&timeline_device_properties,
                              static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts,
      "cudaGetDeviceProperties(device timeline)");
  if (timeline_device_properties.multiProcessorCount <= 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "timeline CTA capacity", 0,
                    "positive SM count",
                    std::to_string(timeline_device_properties.multiProcessorCount));
  // Forward's transport, FC1 and FC2 grids execute concurrently.  Give their
  // lock-free trace writers disjoint logical CTA ranges; one-SM capacity was
  // sufficient only while FC1 and FC2 were accidentally stream-serialized.
  state->timeline_max_ctas =
      3U * static_cast<std::uint32_t>(
               timeline_device_properties.multiProcessorCount);
  state->timeline_layout =
      make_timeline_allocation_layout(state->timeline_max_ctas);
#endif
  state->slots.resize(config.max_inflight_generations);
  try {
    internal::throw_cuda(
        "transport-pool-create",
        cudaStreamCreateWithFlags(&state->cancel_stream, cudaStreamNonBlocking),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaStreamCreate(cancel)");
    for (auto &slot : state->slots) {
      internal::throw_cuda(
          "transport-pool-create", cudaMalloc(&slot.device,
                                                state->layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(transport slot)");
      if (config.enable_gin_rdma) {
        using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
        const auto mem_alloc = reinterpret_cast<MemAllocFn>(
            dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
        if (mem_alloc == nullptr)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=transport-pool-create, field=ncclMemAlloc, rank=0, "
              "expected=runtime symbol, actual=missing");
        internal::throw_nccl(
            "transport-pool-create",
            mem_alloc(&slot.gin_window, state->gin_layout.total), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, "ncclMemAlloc(GIN window slot)");
      }
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
#if MEGAMOE_DEVICE_TIMELINE
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(&slot.forward_timeline, state->timeline_layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(Forward timeline)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.forward_timeline, 0,
                          state->timeline_layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize Forward timeline");
#endif
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device, 0, state->layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize complete transport slot storage");
      if (slot.gin_window)
        internal::throw_cuda(
            "transport-pool-create",
            cudaMemsetAsync(slot.gin_window, 0, state->gin_layout.total,
                            internal::checked_execution_state(execution)->stream),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "one-time initialize GIN window slot storage");
      if (config.max_intermediate_size != 0 &&
          config.max_hidden_size == 7168 &&
          config.max_intermediate_size == 3072) {
        slot.forward_x_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank,
            forward_tuning::kTileM, "canonical X", forward_tuning::kTileK);
        slot.forward_s_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, config.max_destination_rows_per_rank,
            forward_tuning::kTileM, "gated SwiGLU U",
            forward_tuning::kTileK);
        slot.forward_x_m64_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank, 64,
            "canonical X M64");
        slot.forward_s_m64_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size, config.max_destination_rows_per_rank,
            64, "gated SwiGLU U M64");
        slot.forward_output_store_map = make_bf16_tma_store_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank,
            "Forward canonical output");
        if (config.enable_backward_state)
          slot.forward_training_output_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward training output");
        if (config.enable_backward_state) {
          slot.backward_dr_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              backward_tuning::kTileM, "Backward compact dR",
              backward_tuning::kTileK);
          slot.backward_dab_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_dab_rows),
              2ULL * config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              backward_tuning::kTileM, "Backward padded dAB",
              backward_tuning::kTileK);
          slot.backward_dr_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.forward_training_output),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Backward compact dW2 dR MN-major");
          slot.backward_u_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.backward_u),
              config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              "Backward dW2 U MN-major");
          slot.backward_dab_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.backward_dab_rows),
              2ULL * config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              "Backward dW1 dAB MN-major");
          slot.backward_x_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.canonical_rows),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Backward compact dW1 X MN-major");
          slot.backward_u_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_u),
              config.max_intermediate_size,
              state->layout.backward_padded_capacity_rows,
              "Backward dU output");
          if (slot.gin_window != nullptr) {
            slot.backward_dx_store_map = make_bf16_tma_store_map(
                at<void>(slot.gin_window,
                         state->gin_layout.return_staging),
                config.max_hidden_size,
                state->layout.backward_padded_capacity_rows,
                "Backward registered dX output",
                state->gin_layout.backward_return_row_bytes);
          } else {
            slot.backward_dx_store_map = make_bf16_tma_store_map(
                at<void>(slot.device, state->layout.backward_dx_rows),
                config.max_hidden_size,
                state->layout.backward_padded_capacity_rows,
                "Backward dX output");
          }
          slot.backward_dw2_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_dw2),
              config.max_intermediate_size,
              checked_mul(config.max_owned_experts, config.max_hidden_size,
                          "Backward dW2 output rows"),
              "Backward dW2 output");
          slot.backward_dw1_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_dw1),
              config.max_hidden_size,
              checked_mul(config.max_owned_experts,
                          2ULL * config.max_intermediate_size,
                          "Backward dW1 output rows"),
              "Backward dW1 output");
          slot.backward_maps_valid = true;
        }
        if (slot.gin_window != nullptr)
          slot.forward_staging_store_map = make_bf16_tma_store_map(
              at<void>(slot.gin_window, state->gin_layout.return_staging),
              config.max_hidden_size, config.max_destination_rows_per_rank,
              "Forward GIN return staging");
        slot.forward_maps_valid = true;
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(transport status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_forward_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(forward status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_production_status),
              checked_size(checked_mul(config.max_world_size,
                                       sizeof(ProductionStatusRecord),
                                       "production status host matrix"),
                           "production status host matrix"),
              cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(production terminal status matrix)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      std::fill_n(slot.host_production_status, config.max_world_size,
                  ProductionStatusRecord{});
      slot.generation_debug_contexts = static_cast<std::uint32_t>(
          (config.max_nodes > 0 ? config.max_nodes - 1ULL : 0ULL) * 8ULL);
      slot.generation_debug_words = transport_state::completion_debug_words(
          slot.generation_debug_contexts);
      slot.forward_progress_words =
          kForwardProgressWords + 2ULL * slot.generation_debug_words;
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_forward_progress),
              slot.forward_progress_words * sizeof(std::uint64_t),
              cudaHostAllocPortable | cudaHostAllocMapped),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(mapped Forward progress)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostGetDevicePointer(
              reinterpret_cast<void **>(&slot.device_forward_progress),
              slot.host_forward_progress, 0),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostGetDevicePointer(Forward progress)");
      std::fill_n(slot.host_forward_progress, slot.forward_progress_words,
                  0ULL);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_cancel_epoch),
                        sizeof(std::uint64_t),
                        cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      *slot.host_cancel_epoch = 0;
      if (config.enable_socket_fallback) {
        const auto socket_count_bytes = checked_size(
            checked_mul(checked_mul(config.max_world_size,
                                    config.max_world_size,
                                    "Socket host count matrix"),
                        sizeof(std::uint64_t),
                        "Socket host count matrix bytes"),
            "Socket host count matrix bytes");
        internal::throw_cuda(
            "transport-pool-create",
            cudaHostAlloc(reinterpret_cast<void **>(&slot.host_socket_counts),
                          socket_count_bytes, cudaHostAllocPortable),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "cudaHostAlloc(Socket count matrix)");
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
        std::memset(slot.host_socket_counts, 0, socket_count_bytes);
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(reinterpret_cast<void **>(&slot.device_cancel_epoch),
                     sizeof(std::uint64_t)),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device_cancel_epoch, 0, sizeof(std::uint64_t),
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize device cancel epoch");
      auto create_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create",
            cudaEventCreateWithFlags(event, cudaEventDisableTiming), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      auto create_timing_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create", cudaEventCreate(event), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      int least_stream_priority = 0;
      int greatest_stream_priority = 0;
      internal::throw_cuda(
          "transport-pool-create",
          cudaDeviceGetStreamPriorityRange(&least_stream_priority,
                                           &greatest_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaDeviceGetStreamPriorityRange(Forward roles)");
      internal::throw_cuda("transport-pool-create",
                           cudaStreamCreateWithPriority(
                               &slot.forward_transport_stream,
                               cudaStreamNonBlocking, greatest_stream_priority),
                           0, session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaStreamCreateWithPriority(Forward transport)");
      internal::throw_cuda(
          "transport-pool-create",
          cudaStreamCreateWithPriority(&slot.forward_m64_head_stream,
                                       cudaStreamNonBlocking,
                                       greatest_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaStreamCreateWithPriority(Forward M64 head)");
      internal::throw_cuda(
          "transport-pool-create",
          cudaStreamCreateWithPriority(&slot.forward_compute_stream,
                                       cudaStreamNonBlocking,
                                       least_stream_priority),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaStreamCreateWithPriority(Forward compute)");
      create_event(&slot.producer_ready, "cudaEventCreate(producer-ready)");
      create_event(&slot.forward_roles_ready,
                   "cudaEventCreate(Forward roles ready)");
      create_event(&slot.forward_transport_done,
                   "cudaEventCreate(Forward transport done)");
      create_event(&slot.forward_m64_head_done,
                   "cudaEventCreate(Forward M64 head done)");
      create_event(&slot.forward_compute_done,
                   "cudaEventCreate(Forward compute done)");
      if (config.enable_backward_state) {
        create_timing_event(&slot.backward_start,
                            "cudaEventCreate(Backward start)");
        create_timing_event(&slot.backward_compute_start,
                            "cudaEventCreate(Backward compute start)");
        create_timing_event(&slot.backward_compute_done,
                            "cudaEventCreate(Backward compute done)");
        create_timing_event(&slot.backward_transport_done,
                            "cudaEventCreate(Backward transport done)");
        create_timing_event(&slot.backward_reduce_done,
                            "cudaEventCreate(Backward reduce done)");
        create_timing_event(&slot.backward_late_compute_done,
                            "cudaEventCreate(Backward late compute done)");
        create_timing_event(&slot.backward_complete,
                            "cudaEventCreate(Backward complete)");
      }
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaEventCreate(&slot.forward_output_ready), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaEventCreate(forward output ready)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.dispatch_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(dispatch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    }
  } catch (...) {
    throw;
  }
  auto impl = std::make_shared<ActivationTransportResourcePool::Impl>();
  impl->state = std::move(state);
  return ActivationTransportResourcePool(std::move(impl));
}
