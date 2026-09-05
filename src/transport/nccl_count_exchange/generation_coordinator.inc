namespace {

LocalPreflightStatus host_preflight_status(
    const RuntimeConfig &config, const RankRoutingInput &routes,
    std::uint64_t max_tokens_per_rank,
    const CudaRouteInput *device_routes = nullptr) noexcept {
  LocalPreflightStatus status;
  const auto fail = [&](ControlLocalError error, ControlField field,
                        std::uint64_t token, std::uint64_t slot,
                        std::int64_t expert, std::uint64_t expected,
                        std::uint64_t actual) {
    status = {error, field, token, slot, expert, expected, actual};
  };
  try {
    validate_runtime_config(config);
  } catch (...) {
    fail(ControlLocalError::InvalidRuntimeConfig,
         ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0, 1, 0);
    return status;
  }
  if (routes.num_tokens > max_tokens_per_rank) {
    fail(ControlLocalError::CapacityExceeded, ControlField::NumTokens,
         UINT64_MAX, UINT64_MAX, 0, max_tokens_per_rank, routes.num_tokens);
    return status;
  }
  if (config.top_k != 0 && routes.num_tokens > UINT64_MAX / config.top_k) {
    fail(ControlLocalError::ArithmeticOverflow, ControlField::RouteCount,
         UINT64_MAX, UINT64_MAX, 0, UINT64_MAX, UINT64_MAX);
    return status;
  }
  const auto expected = routes.num_tokens * config.top_k;
  const auto actual_route_count = device_routes
                                      ? device_routes->route_count()
                                      : routes.selected_experts.size();
  if (actual_route_count != expected ||
      (device_routes &&
       (device_routes->num_tokens() != routes.num_tokens ||
        device_routes->top_k() != config.top_k))) {
    fail(ControlLocalError::InvalidRouteShape, ControlField::RouteCount,
         UINT64_MAX, UINT64_MAX, 0, expected,
         actual_route_count);
    return status;
  }
  if (device_routes)
    return status;
  for (std::uint64_t token = 0; token < routes.num_tokens; ++token) {
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
      const auto index = token * config.top_k + slot;
      const auto expert = routes.selected_experts[static_cast<std::size_t>(index)];
      if (expert < 0 || static_cast<std::uint64_t>(expert) >= config.num_experts) {
        fail(ControlLocalError::ExpertOutOfRange,
             ControlField::LocalValidation, token, slot, expert,
             config.num_experts, static_cast<std::uint64_t>(expert));
        return status;
      }
      for (std::uint32_t earlier = 0; earlier < slot; ++earlier) {
        const auto earlier_index = token * config.top_k + earlier;
        if (routes.selected_experts[static_cast<std::size_t>(earlier_index)] ==
            expert) {
          fail(ControlLocalError::DuplicateExpert,
               ControlField::LocalValidation, token, slot, expert, 0,
               static_cast<std::uint64_t>(expert));
          return status;
        }
      }
    }
  }
  return status;
}

} // namespace

CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    const NcclSession &session, const CudaExecution &execution) {
  if (!session || !execution)
    invalid("phase=generation-coordinator-create, field=input, rank=n/a, expected=valid session/execution, actual=empty");
  const auto start = std::chrono::steady_clock::now();
  auto impl = std::make_shared<Impl>();
  impl->session = session;
  impl->execution = execution;
  impl->context = make_rank_collective_context(
      session.session(), session.rank(), execution.token());
  const auto backend_start = std::chrono::steady_clock::now();
  impl->backend =
      std::make_shared<NcclAllGatherCountExchange>(session, execution);
  const auto backend_end = std::chrono::steady_clock::now();
  impl->setup = impl->backend->impl_->setup;
  impl->setup.coordinator_backend_create_us =
      std::chrono::duration<double, std::micro>(backend_end - backend_start)
          .count();
  impl->setup.per_generation_setup_total_us =
      std::chrono::duration<double, std::micro>(backend_end - start).count();
  impl_ = std::move(impl);
}

CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    const NcclSession &session, const CudaExecution &execution,
    const DeviceNotifyResourcePool &pool) {
  if (!session || !execution || !pool)
    invalid("phase=generation-coordinator-create, field=input, rank=n/a, "
            "expected=valid session/execution/pool, actual=empty");
  if (!session.shares_communicator_with(pool.impl_->base_session) ||
      !execution.token().same_instance(pool.impl_->execution.token()))
    invalid("phase=generation-coordinator-create, field=pool_provenance, rank=" +
            std::to_string(session.rank()) +
            ", expected=same communicator/execution, actual=mismatch");
  auto impl = std::make_shared<Impl>();
  impl->session = session;
  impl->execution = execution;
  impl->context = make_rank_collective_context(
      session.session(), session.rank(), execution.token());
  impl->pool = pool.impl_;
  impl->setup.one_time_pool_setup_us = pool.impl_->setup_us;
  impl_ = std::move(impl);
}

CudaNotifyGenerationCoordinator::~CudaNotifyGenerationCoordinator() noexcept =
    default;
CudaNotifyGenerationCoordinator::CudaNotifyGenerationCoordinator(
    CudaNotifyGenerationCoordinator &&) noexcept = default;
CudaNotifyGenerationCoordinator &
CudaNotifyGenerationCoordinator::operator=(
    CudaNotifyGenerationCoordinator &&) noexcept = default;

namespace testing {
CudaNotifyGenerationCoordinator make_unpooled_coordinator_for_testing(
    const NcclSession &session, const CudaExecution &execution) {
  return CudaNotifyGenerationCoordinator(session, execution);
}
} // namespace testing

ValidatedCudaGeneration CudaNotifyGenerationCoordinator::run(
    NotifyGenerationRequest request, GenerationRunOptions options) {
  if (!impl_ || impl_->used)
    invalid("phase=generation-coordinator-run, field=state, rank=n/a, expected=fresh coordinator, actual=empty/already-used");
  impl_->used = true;
  const auto run_start = std::chrono::steady_clock::now();
  const auto offset_us = [&](std::chrono::steady_clock::time_point point) {
    return std::chrono::duration<double, std::micro>(point - run_start)
        .count();
  };
  const auto pool_acquire_begin = std::chrono::steady_clock::now();
  const auto deadline = options.deadline_after.count() > 0
                            ? run_start + options.deadline_after
                            : std::chrono::steady_clock::time_point::max();

  std::shared_ptr<DeviceNotifyResourcePool::Impl::Lease> pool_lease;
  std::shared_ptr<CudaNotifyPoolAllocation> pool_allocation;
  if (impl_->pool) {
    const auto acquire_start = std::chrono::steady_clock::now();
    pool_lease = impl_->pool->acquire(impl_->session);
    auto &slot = impl_->pool->slots[pool_lease->index];
    pool_allocation = slot.notify;
    impl_->backend = slot.backend;
    impl_->setup.pool_acquire_us =
        std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - acquire_start)
            .count();
  }
  const auto pool_acquire_end = std::chrono::steady_clock::now();
  if (!impl_->backend)
    invalid("phase=generation-coordinator-run, field=backend, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=available, actual=empty");

  ConfigWire config_wire;
  LocalPreflightStatus config_status;
  auto protocol_identity = impl_->context.session().stable_identity_bytes();
#if MEGAMOE_BUILD_TESTING
  std::unique_ptr<testing::IdentityOverride> identity_override;
  if (testing::g_identity_override) {
    identity_override = std::move(testing::g_identity_override);
    protocol_identity = identity_override->raw;
  }
#endif
  (void)try_encode_canonical_config(
      request.config, request.max_tokens_per_rank, config_wire, config_status,
      protocol_identity);
  const auto *direct_routes = impl_->direct_routes
                                  ? &impl_->direct_routes
                                  : static_cast<const CudaRouteInput *>(nullptr);
  const auto *direct_gates = impl_->direct_gates
                                 ? &impl_->direct_gates
                                 : static_cast<const CudaGateInput *>(nullptr);
  auto status = host_preflight_status(request.config, request.local_routes,
                                      request.max_tokens_per_rank,
                                      direct_routes);
  if (status.error == ControlLocalError::None)
    status = config_status;
  if (status.error == ControlLocalError::None && direct_routes) {
    if (direct_routes->route_generation() !=
        impl_->context.session().generation()) {
      status = {ControlLocalError::InvalidRouteShape,
                ControlField::Generation, UINT64_MAX, UINT64_MAX, 0,
                impl_->context.session().generation(),
                direct_routes->route_generation()};
    } else {
      try {
        DeviceRouteProvenanceBridge::validate(
            *direct_routes, impl_->context, impl_->execution);
        if (!direct_gates)
          throw NotifyDispatchError(
              NotifyErrorCode::InvalidRuntimeConfig,
              "phase=generation-preflight, field=gates, expected=valid, actual=empty");
        DeviceGateProvenanceBridge::validate(
            *direct_gates, *direct_routes, impl_->context, impl_->execution);
      } catch (const NotifyDispatchError &) {
        status = {ControlLocalError::InvalidRouteShape,
                  ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0,
                  1, 0};
      }
    }
  }
  if (status.error == ControlLocalError::None && impl_->pool) {
    const auto &capacity = impl_->pool->config;
    if (request.config.num_experts > capacity.capacity_config.num_experts ||
        request.config.top_k > capacity.capacity_config.top_k ||
        request.config.topology.world_size !=
            capacity.capacity_config.topology.world_size ||
        request.config.topology.num_nodes >
            capacity.capacity_config.topology.num_nodes ||
        request.max_tokens_per_rank > capacity.max_tokens_per_rank ||
        config_wire.size() > capacity.max_config_wire_bytes) {
      status = {ControlLocalError::CapacityExceeded,
                ControlField::LocalValidation, UINT64_MAX, UINT64_MAX, 0,
                capacity.max_tokens_per_rank,
                request.max_tokens_per_rank};
    }
  }

  bool use_fast_notify = false;
#if !MEGAMOE_DEBUG_VALIDATE
  use_fast_notify = request.config.topology.num_nodes == 1 && impl_->pool &&
                    impl_->backend->fast_path_ready()
#if MEGAMOE_BUILD_TESTING
                    && !identity_override &&
                    testing::g_config_corruption_offset == SIZE_MAX &&
                    testing::g_next_fault.load(std::memory_order_acquire) == 0
#endif
                    ;
  // A changed config cannot use the cached fixed-layout record. Fall back to
  // the complete agreement protocol so a uniformly changed, valid generation
  // can recover on the same communicator, while mismatched ranks still fail
  // at the exact config boundary.
  if (use_fast_notify && !impl_->backend->fast_config_matches(config_wire))
    use_fast_notify = false;
#endif

  CudaLocalNotifyHandle local;
  double local_mapping_us = 0;
  std::uint64_t route_input_bytes = 0;
  std::uint64_t notify_arena_bytes = 0;
  std::uint64_t notify_pinned_bytes = 0;
  const auto local_histogram_begin = std::chrono::steady_clock::now();
  if (status.error == ControlLocalError::None) {
    try {
      auto routes = direct_routes
                        ? *direct_routes
                        : pool_allocation
                        ? DeviceNotifyPoolBridge::upload(
                              request.local_routes, request.config.top_k,
                              impl_->context,
                              impl_->execution, pool_allocation, pool_lease)
                        : upload_cuda_routes(request.local_routes,
                                             request.config.top_k,
                                             impl_->context,
                                             impl_->execution);
      CudaGateInput gates;
      if (direct_gates) {
        gates = *direct_gates;
      } else {
        std::vector<float> unit_gates(
            static_cast<std::size_t>(routes.route_count()), 1.0F);
        gates = DeviceGateProvenanceBridge::upload_internal(
            unit_gates, routes, impl_->context, impl_->execution);
      }
      route_input_bytes = routes.device_bytes();
      auto plan = query_cuda_notify_workspace(
          request.config, impl_->session.rank(),
          request.local_routes.num_tokens, request.max_tokens_per_rank);
      notify_arena_bytes = plan.device_bytes;
      notify_pinned_bytes = plan.host_staging_bytes;
      auto workspace = pool_allocation
                           ? DeviceNotifyPoolBridge::workspace(
                                 plan, impl_->context, impl_->execution,
                                 pool_allocation, pool_lease)
                           : allocate_cuda_notify_workspace(
                                 plan, impl_->context, impl_->execution);
      auto operation = start_cuda_rank_local_notify(
          request.config, impl_->context, impl_->execution, routes,
          gates, std::move(workspace));
      auto preflight = operation.wait_preflight();
      local_mapping_us = preflight.device_mapping_us();
      status = preflight.status();
      if (preflight.valid())
        local = preflight.local();
    } catch (const NotifyDispatchError &) {
      // Predictable input/config failures must still join fixed control. A
      // CUDA runtime/context failure is not predictable and remains fatal.
      status = host_preflight_status(request.config, request.local_routes,
                                     request.max_tokens_per_rank,
                                     direct_routes);
      if (status.error == ControlLocalError::None)
        throw;
    }
  }
  const auto local_histogram_end = std::chrono::steady_clock::now();

  auto header = make_control_header(
      request.config, impl_->context, request.local_routes.num_tokens,
      direct_routes ? direct_routes->route_count()
                    : request.local_routes.selected_experts.size(),
      request.max_tokens_per_rank, config_wire.size(),
      config_wire_digest(config_wire), status);
#if MEGAMOE_BUILD_TESTING
  if (identity_override) {
    header.session_id_word0 = identity_override->summary.word0;
    header.session_id_word1 = identity_override->summary.word1;
  }
#endif
  const auto collective_enqueue_begin = std::chrono::steady_clock::now();
  auto control_config_start = collective_enqueue_begin;
  auto control_config_end = collective_enqueue_begin;
  if (!use_fast_notify) {
    control_config_start = std::chrono::steady_clock::now();
    auto control = impl_->backend->start_control(impl_->context, header);
    auto control_done = control.wait();

#if MEGAMOE_BUILD_TESTING
    if (testing::g_config_corruption_offset != SIZE_MAX) {
      const auto offset = testing::g_config_corruption_offset;
      testing::g_config_corruption_offset = SIZE_MAX;
      if (offset < config_wire.size())
        config_wire[offset] ^= std::byte{1};
    }
#endif

    auto config = impl_->backend->start_config(impl_->context, config_wire,
                                               control_done);
    (void)config.wait();
    control_config_end = std::chrono::steady_clock::now();
  }
  const auto coordinator_create = impl_->setup.coordinator_backend_create_us;
  const auto initial_total = impl_->setup.per_generation_setup_total_us;
  const auto pool_setup = impl_->setup.one_time_pool_setup_us;
  const auto pool_acquire = impl_->setup.pool_acquire_us;
  if (impl_->pool) {
    impl_->setup = {};
    impl_->setup.one_time_pool_setup_us = pool_setup;
    impl_->setup.pool_acquire_us = pool_acquire;
    impl_->setup.per_generation_setup_total_us = pool_acquire;
  } else {
    impl_->setup = impl_->backend->impl_->setup;
    impl_->setup.coordinator_backend_create_us = coordinator_create;
    impl_->setup.per_generation_setup_total_us =
        initial_total + impl_->setup.config_slot_allocation_us +
        impl_->setup.count_slot_allocation_us +
        impl_->setup.summary_slot_allocation_us;
  }

  BackendCompletion count_done;
  BackendCompletion summary_done;
  auto count_start = std::chrono::steady_clock::now();
  auto count_end = count_start;
  auto summary_start = count_start;
  auto summary_end = count_start;
  auto collective_enqueue_end = count_start;
  auto collective_complete = count_start;
  double fused_enqueue_us = 0.0;
  double fused_wait_us = 0.0;
  if (use_fast_notify) {
    const DeviceBufferLease *count_input = nullptr;
    const DeviceBufferLease *summary_input = nullptr;
    DeviceBufferLease count_lease;
    DeviceBufferLease summary_lease;
    // A failed preflight still participates in the fixed-width fused control
    // collective, but its variable rows are deliberately represented by the
    // already-zeroed record payload.  In particular, a config mismatch may
    // have produced a local row whose width differs from the session-verified
    // layout; never copy that row using the canonical record width.
    if (local && status.error == ControlLocalError::None) {
      count_lease = local.device_expert_count_for_backend();
      summary_lease = local.device_packet_count_for_backend();
      count_input = &count_lease;
      summary_input = &summary_lease;
    }
    auto fused = impl_->backend->run_fast_notify(
        impl_->context, header, count_input, summary_input);
    count_done = std::move(fused.expert_counts);
    summary_done = std::move(fused.packet_summary);
    fused_enqueue_us = fused.enqueue_us;
    fused_wait_us = fused.wait_us;
    collective_enqueue_end =
        collective_enqueue_begin + std::chrono::duration_cast<
                                       std::chrono::steady_clock::duration>(
                                       std::chrono::duration<double, std::micro>(
                                           fused_enqueue_us));
  } else {
    if (!local)
      invalid("phase=generation-coordinator, field=local_handle, rank=" +
              std::to_string(impl_->session.rank()) +
              ", expected=valid after successful agreement, actual=empty");
    count_start = std::chrono::steady_clock::now();
    auto count = impl_->backend->start(CountExchangeRequest{
        impl_->context, ExchangeKind::ExpertCounts, request.config.num_experts,
        {}, local.device_expert_count_for_backend()});
    if (options.cancellation.cancellation_requested() ||
        std::chrono::steady_clock::now() >= deadline) {
      count->cancel();
      throw NotifyDispatchError(
          NotifyErrorCode::CountExchangeFailure,
          "phase=generation-cancel, field=cancel_or_deadline, rank=" +
              std::to_string(impl_->session.rank()) + ", world=" +
              std::to_string(impl_->session.session().world_size()) +
              ", session='" + impl_->session.session().identity() +
              "', generation=" +
              std::to_string(impl_->session.session().generation()) +
              ", expected=active generation, actual=cancel requested/deadline expired");
    }
    count_done = count->wait();
    maybe_fault(testing::NcclFaultPoint::CountKernelCompletion,
                "count-kernel-completion", impl_->context);
    count_end = std::chrono::steady_clock::now();
    summary_start = std::chrono::steady_clock::now();
    auto summary = impl_->backend->start(CountExchangeRequest{
        impl_->context, ExchangeKind::NotifyPacketSummary,
        request.config.topology.num_nodes, {},
        local.device_packet_count_for_backend()});
    summary_done = summary->wait();
    maybe_fault(testing::NcclFaultPoint::SummaryKernelCompletion,
                "summary-kernel-completion", impl_->context);
    summary_end = std::chrono::steady_clock::now();
    collective_enqueue_end = summary_start;
    collective_complete = summary_end;
  }
  if (!local && use_fast_notify) {
    // Predictable invalid input has no local device metadata to materialize,
    // but every rank has already joined the fixed collective. Complete and
    // validate that collective so all ranks select the same structured error.
    (void)impl_->backend->finish_fast_notify();
    collective_complete = std::chrono::steady_clock::now();
  }
  if (!local)
    invalid("phase=generation-coordinator, field=local_handle, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=valid after successful agreement, actual=empty");
  const auto materialize_start = std::chrono::steady_clock::now();
  DeviceMetadataBridge::Timings materialize_timings;
  auto materialize_end = materialize_start;
  maybe_fault(testing::NcclFaultPoint::MaterializationKernel,
              "materialization-kernel", impl_->context);
  DeviceDispatchHandle device;
  try {
    device = DeviceMetadataBridge::materialize(
        local, std::move(count_done), std::move(summary_done), impl_->execution,
        &materialize_timings);
    materialize_end = std::chrono::steady_clock::now();
  } catch (...) {
    if (use_fast_notify) {
      try {
        (void)impl_->backend->finish_fast_notify();
      } catch (...) {
      }
    }
    throw;
  }
  if (use_fast_notify) {
    (void)impl_->backend->finish_fast_notify();
    const auto combined_wait_us =
        std::chrono::duration<double, std::micro>(materialize_end -
                                                  materialize_start)
            .count();
    fused_wait_us = std::max(
        0.0, combined_wait_us - materialize_timings.prefix_planning_us -
                 materialize_timings.remaining_parallel_us);
    collective_complete = std::chrono::steady_clock::now();
  }
  const auto audit_value = impl_->backend->audit();
  const bool valid_fused_audit =
      use_fast_notify && audit_value.fused_notify_starts == 1 &&
      audit_value.control_starts == 0 && audit_value.config_starts == 0 &&
      audit_value.expert_count_starts == 0 &&
      audit_value.packet_summary_starts == 0 &&
      audit_value.config_cache_hits == 1;
  const bool valid_full_audit =
      !use_fast_notify && audit_value.fused_notify_starts == 0 &&
      audit_value.control_starts == 1 &&
      audit_value.config_starts == 1 && audit_value.config_cache_hits == 0 &&
      audit_value.expert_count_starts == 1 &&
      audit_value.packet_summary_starts == 1;
  if (!valid_fused_audit && !valid_full_audit)
    invalid("phase=generation-publish, field=audit, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=full 1/1/1/1 or fused 1, actual=" +
            std::to_string(audit_value.control_starts) + "/" +
            std::to_string(audit_value.config_starts) + "+cache=" +
            std::to_string(audit_value.config_cache_hits) + "/" +
            std::to_string(audit_value.expert_count_starts) + "/" +
            std::to_string(audit_value.packet_summary_starts) + "/fused=" +
            std::to_string(audit_value.fused_notify_starts));
  auto result = std::make_shared<ValidatedCudaGeneration::Impl>();
  result->dispatch = std::move(device);
  result->audit = audit_value;
  result->workspace = impl_->backend->workspace_usage();
  result->workspace.route_input_device_bytes = route_input_bytes;
  result->workspace.device_notify_arena_bytes = notify_arena_bytes;
  result->workspace.device_notify_pinned_staging_bytes = notify_pinned_bytes;
  result->workspace.final_device_metadata_arena_view_capacity_bytes =
      notify_arena_bytes;
  result->workspace.device_bytes += route_input_bytes + notify_arena_bytes;
  result->workspace.pinned_host_bytes += notify_pinned_bytes;
  // Notify workspace/route/consumer events plus preallocated collective events.
  result->workspace.event_count = 15;
  result->workspace.quantified_event_bytes = 0;
  if (impl_->pool) {
    result->workspace = {};
    for (const auto &slot : impl_->pool->slots) {
      const auto backend_usage = slot.backend->workspace_usage();
      result->workspace.device_bytes +=
          DeviceNotifyPoolBridge::device_bytes(slot.notify) +
          backend_usage.device_bytes;
      result->workspace.pinned_host_bytes +=
          DeviceNotifyPoolBridge::pinned_bytes(slot.notify) +
          backend_usage.pinned_host_bytes;
      result->workspace.route_input_device_bytes +=
          static_cast<std::uint64_t>(impl_->pool->max_plan.route_count) *
          sizeof(std::int32_t);
      result->workspace.device_notify_arena_bytes +=
          impl_->pool->max_plan.device_bytes;
      result->workspace.device_notify_pinned_staging_bytes +=
          impl_->pool->max_plan.host_staging_bytes;
      result->workspace.control_slot_bytes += backend_usage.control_slot_bytes;
      result->workspace.config_wire_bytes += backend_usage.config_wire_bytes;
      result->workspace.config_slot_bytes += backend_usage.config_slot_bytes;
      result->workspace.expert_slot_bytes += backend_usage.expert_slot_bytes;
      result->workspace.summary_slot_bytes += backend_usage.summary_slot_bytes;
      result->workspace.fused_notify_slot_bytes +=
          backend_usage.fused_notify_slot_bytes;
      result->workspace.event_count +=
          5U + DeviceNotifyPoolBridge::event_count(slot.notify);
    }
    result->workspace.final_device_metadata_arena_view_capacity_bytes =
        impl_->pool->max_plan.device_bytes;
    // A production router-owned input is outside the pool arenas and is a
    // unique live allocation. Account it once in addition to all pool slots;
    // the host-convenience path uses a pool route slot and adds nothing here.
    if (direct_routes) {
      result->workspace.route_input_device_bytes += route_input_bytes;
      result->workspace.device_bytes += route_input_bytes;
    }
  }
  result->setup = impl_->setup;
  const auto run_end = std::chrono::steady_clock::now();
  result->execution.coordinator_run_wall_us =
      std::chrono::duration<double, std::micro>(run_end - run_start).count();
  const auto dynamic_setup = result->setup.config_slot_allocation_us +
                             result->setup.count_slot_allocation_us +
                             result->setup.summary_slot_allocation_us +
                             result->setup.pool_acquire_us;
  result->execution.steady_execution_us =
      std::max(0.0, result->execution.coordinator_run_wall_us - dynamic_setup);
  result->execution.local_mapping_us = local_mapping_us;
  result->execution.control_config_us =
      std::chrono::duration<double, std::micro>(control_config_end -
                                                control_config_start)
          .count();
  result->execution.expert_count_us =
      std::chrono::duration<double, std::micro>(count_end - count_start).count();
  result->execution.packet_summary_us =
      std::chrono::duration<double, std::micro>(summary_end - summary_start)
          .count();
  result->execution.device_materialization_us =
      use_fast_notify
          ? materialize_timings.prefix_planning_us +
                materialize_timings.remaining_parallel_us
          : std::chrono::duration<double, std::micro>(materialize_end -
                                                      materialize_start)
                .count();
  result->execution.prefix_planning_us =
      materialize_timings.prefix_planning_us;
  result->execution.remaining_parallel_materialization_us =
      materialize_timings.remaining_parallel_us;
  result->execution.pool_acquire_release_us = result->setup.pool_acquire_us;
  result->execution.route_gate_ready_offset_us = 0.0;
  result->execution.pool_acquire_begin_offset_us =
      offset_us(pool_acquire_begin);
  result->execution.pool_acquire_end_offset_us = offset_us(pool_acquire_end);
  result->execution.local_histogram_begin_offset_us =
      offset_us(local_histogram_begin);
  result->execution.local_histogram_end_offset_us =
      offset_us(local_histogram_end);
  result->execution.collective_enqueue_begin_offset_us =
      offset_us(collective_enqueue_begin);
  result->execution.collective_enqueue_end_offset_us =
      offset_us(collective_enqueue_end);
  result->execution.collective_complete_offset_us =
      offset_us(collective_complete);
  result->execution.device_materialization_end_offset_us =
      offset_us(materialize_end);
  result->execution.notify_handle_ready_offset_us = offset_us(run_end);
  result->execution.fused_collective_enqueue_us = fused_enqueue_us;
  result->execution.fused_collective_wait_us = fused_wait_us;
  result->resource_lease = pool_lease
                               ? std::static_pointer_cast<const void>(pool_lease)
                               : std::static_pointer_cast<const void>(impl_);
  testing::g_validated_publishes.fetch_add(1);
  return ValidatedCudaGeneration(std::move(result));
}

ValidatedCudaGeneration CudaNotifyGenerationCoordinator::run_device(
    DeviceNotifyGenerationRequest request, GenerationRunOptions options) {
  if (!impl_ || !impl_->pool)
    invalid("phase=generation-coordinator-device, field=resource_pool, "
            "rank=n/a, expected=explicit session pool, actual=missing");
  if (!request.local_routes || !request.local_gates)
    invalid("phase=generation-coordinator-device, field=routes, rank=" +
            std::to_string(impl_->session.rank()) +
            ", expected=valid route+gate device leases, actual=empty");
  impl_->direct_routes = request.local_routes;
  impl_->direct_gates = request.local_gates;
  NotifyGenerationRequest adapted;
  adapted.config = std::move(request.config);
  adapted.local_routes.num_tokens = request.local_routes.num_tokens();
  adapted.max_tokens_per_rank = request.max_tokens_per_rank;
  return run(std::move(adapted), std::move(options));
}

CollectiveProtocolAudit CudaNotifyGenerationCoordinator::audit() const {
  if (!impl_)
    invalid("phase=generation-audit, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->backend->audit();
}

NcclWorkspaceUsage CudaNotifyGenerationCoordinator::workspace_usage() const {
  if (!impl_)
    invalid("phase=generation-workspace, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->backend->workspace_usage();
}

GenerationSetupTimings CudaNotifyGenerationCoordinator::setup_timings() const {
  if (!impl_)
    invalid("phase=generation-setup, field=coordinator, rank=n/a, expected=valid, actual=empty");
  return impl_->setup;
}

