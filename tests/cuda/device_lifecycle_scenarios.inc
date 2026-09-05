void test_device_only_sanitizer_paths() {
  const auto config = make_config(1, 1, 8, 3, 4);
  auto execution = megamoe::cuda::make_cuda_execution("device-only", 0);
  auto session = megamoe::make_collective_session("device-only", 1, 5001);
  auto context = megamoe::make_rank_collective_context(
      session, 0, execution.token());
  const auto run = [&](megamoe::RankRoutingInput input, bool valid) {
    auto routes = megamoe::cuda::upload_cuda_routes(
        input, config.top_k, context, execution);
    auto plan = megamoe::cuda::query_cuda_notify_workspace(
        config, 0, input.num_tokens, 4);
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        plan, context, execution);
    auto gates = make_unit_gates(routes, context, execution);
    auto operation = megamoe::cuda::start_cuda_rank_local_notify(
        config, context, execution, routes, gates, std::move(workspace));
    const auto result = operation.wait_preflight();
    require(result.valid() == valid,
            "device-only validation status mismatch");
    if (!valid)
      require(result.status().error ==
                  megamoe::cuda::ControlLocalError::ExpertOutOfRange,
              "device error path lost coordinates");
  };
  run({3, {5, 1, 7, 1, 2, 3, 0, 7, 4}}, true);
  run({1, {0, 1, 8}}, false);
  const megamoe::RankRoutingInput handoff_input{1, {0, 2, 4}};
  auto producer_routes = megamoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto consumer_execution =
      megamoe::cuda::make_cuda_execution("device-only-consumer", 0);
  auto consumer_context = megamoe::make_rank_collective_context(
      session, 0, consumer_execution.token());
  auto consumer_plan = megamoe::cuda::query_cuda_notify_workspace(
      config, 0, handoff_input.num_tokens, 4);
  bool no_wait_failed = false;
  try {
    auto workspace = megamoe::cuda::allocate_cuda_notify_workspace(
        consumer_plan, consumer_context, consumer_execution);
    auto gates = make_unit_gates(producer_routes, context, execution);
    (void)megamoe::cuda::start_cuda_rank_local_notify(
        config, consumer_context, consumer_execution, producer_routes,
        gates, std::move(workspace));
  } catch (const megamoe::NotifyDispatchError &) {
    no_wait_failed = true;
  }
  require(no_wait_failed,
          "cross-stream route consumption without event handoff succeeded");
  auto imported = megamoe::cuda::import_selected_expert_routes(
      producer_routes, consumer_context, consumer_execution);
  auto imported_workspace = megamoe::cuda::allocate_cuda_notify_workspace(
      consumer_plan, consumer_context, consumer_execution);
  auto imported_gates = make_unit_gates(
      imported, consumer_context, consumer_execution);
  auto imported_operation = megamoe::cuda::start_cuda_rank_local_notify(
      config, consumer_context, consumer_execution, imported,
      imported_gates, std::move(imported_workspace));
  require(imported_operation.wait_preflight().valid(),
          "legal cross-stream event handoff failed");
  auto foreign_session =
      megamoe::make_collective_session("foreign-device-only", 1, 5001);
  auto foreign_context = megamoe::make_rank_collective_context(
      foreign_session, 0, consumer_execution.token());
  bool foreign_failed = false;
  try {
    (void)megamoe::cuda::import_selected_expert_routes(
        producer_routes, foreign_context, consumer_execution);
  } catch (const megamoe::NotifyDispatchError &) {
    foreign_failed = true;
  }
  require(foreign_failed,
          "foreign route pipeline session handoff did not fail closed");
  auto alternate_routes = megamoe::cuda::upload_cuda_routes(
      handoff_input, config.top_k, context, execution);
  auto original_gates = make_unit_gates(producer_routes, context, execution);
  bool route_gate_binding_failed = false;
  try {
    megamoe::cuda::DeviceGateProvenanceBridge::validate(
        original_gates, alternate_routes, context, execution);
  } catch (const megamoe::NotifyDispatchError &) {
    route_gate_binding_failed = true;
  }
  require(route_gate_binding_failed,
          "gate lease bound to another route allocation was accepted");
  const auto wide_config = make_config(73, 1, 146, 6, 4);
  auto wide_session =
      megamoe::make_collective_session("device-wide-mask", 73, 5002);
  auto wide_context = megamoe::make_rank_collective_context(
      wide_session, 0, execution.token());
  std::vector<megamoe::RankRoutingInput> wide_inputs(73);
  for (std::uint32_t rank = 0; rank < 73; ++rank) {
    wide_inputs[rank].num_tokens = rank % 5U == 0 ? 0 : 1;
    if (wide_inputs[rank].num_tokens == 0)
      continue;
    for (std::uint32_t slot = 0; slot < wide_config.top_k; ++slot)
      wide_inputs[rank].selected_experts.push_back(
          static_cast<std::int32_t>((rank * 2U + slot * 23U) % 146U));
  }
  wide_inputs[0] = {1, {0, 2, 64, 128, 144, 145}};
  std::vector<std::uint64_t> count_matrix(73U * 146U, 0);
  std::vector<std::uint64_t> summary_matrix(73U * 73U, 0);
  for (std::uint32_t source = 0; source < 73; ++source) {
    std::vector<bool> seen_node(73, false);
    for (const auto selected : wide_inputs[source].selected_experts) {
      const auto expert = static_cast<std::uint32_t>(selected);
      ++count_matrix[static_cast<std::size_t>(source) * 146U + expert];
      const auto destination_rank =
          wide_config.topology.expert_to_rank[expert];
      seen_node[wide_config.topology.rank_to_node[destination_rank]] = true;
    }
    for (std::uint32_t node = 0; node < 73; ++node)
      summary_matrix[static_cast<std::size_t>(source) * 73U + node] =
          seen_node[node] ? wide_inputs[source].num_tokens : 0;
  }
  const auto expected_wide = megamoe::build_notify_dispatch_for_testing(
      wide_config, wide_inputs, "device-wide-mask", 5002);
  std::string wide_baseline;
  megamoe::cuda::DeviceDispatchHandle wide_handle;
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    wide_handle = megamoe::cuda::testing::materialize_fake_device_collectives(
        wide_config, wide_context, execution, wide_inputs[0], 4,
        count_matrix, summary_matrix);
    const auto downloaded =
        megamoe::cuda::download_device_metadata_for_validation(
            wide_handle, execution);
    const auto serialized = megamoe::serialize_dispatch_metadata(downloaded);
    require(serialized == megamoe::serialize_dispatch_metadata(expected_wide[0]),
            "73-node full device materializer/oracle mismatch");
    if (repeat == 0)
      wide_baseline = serialized;
    else
      require(serialized == wide_baseline,
              "73-node materializer is not byte deterministic");
  }
  auto wide_plan = megamoe::cuda::query_cuda_notify_workspace(
      wide_config, 0, wide_inputs[0].num_tokens, 4);
  require(wide_plan.node_mask_words == 2,
          "73-node workspace did not allocate multiword mask");
  require(wide_handle.sizes().node_mask_words == 2 &&
              wide_handle.sizes().packet_count == 6 &&
              wide_handle.sizes().fanout_count == 6,
          "73-node packet dedup/fan-out shape mismatch");

  // The transport-only persistent worker consumes the immutable device
  // dispatch handle directly. No activation bulk download occurs until the
  // explicit validation call below.
  const megamoe::RankRoutingInput transport_routes{
      3, {5, 1, 7, 1, 2, 3, 0, 7, 4}};
  std::vector<std::uint64_t> transport_counts(8, 0);
  for (const auto selected : transport_routes.selected_experts)
    ++transport_counts[static_cast<std::size_t>(selected)];
  const std::vector<std::uint64_t> transport_summary{3};
  auto transport_dispatch =
      megamoe::cuda::testing::materialize_fake_device_collectives(
          config, context, execution, transport_routes, 4, transport_counts,
          transport_summary);
  const auto host_dispatch = megamoe::build_notify_dispatch_for_testing(
      config, {transport_routes}, "device-only", 5001);
  megamoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = 33;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 2;
  std::vector<megamoe::transport::BFloat16> activations(3U * 33U);
  for (std::size_t i = 0; i < activations.size(); ++i)
    activations[i] = megamoe::transport::float_to_bfloat16(
        static_cast<float>((i * 13U) % 97U) / 16.0F);
  const auto oracle = megamoe::transport::simulate_activation_transport(
      host_dispatch[0], activations,
      std::vector<float>(transport_routes.selected_experts.size(), 1.0F),
      transport_config, 1);
  megamoe::cuda::ActivationTransportPoolConfig transport_pool_config;
  transport_pool_config.max_tokens_per_rank = 4;
  transport_pool_config.max_hidden_size = 33;
  transport_pool_config.max_top_k = 3;
  transport_pool_config.max_world_size = 1;
  transport_pool_config.max_nodes = 1;
  transport_pool_config.max_destination_rows_per_rank = 12;
  transport_pool_config.channels = 1;
  transport_pool_config.ring_slots_per_source_channel = 2;
  transport_pool_config.max_inflight_generations = 2;
  auto transport_pool = megamoe::cuda::make_activation_transport_pool(
      session, execution, transport_pool_config);
  auto activation_lease =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, transport_pool);
  megamoe::cuda::DeviceActivationTransport transport(
      session, execution, transport_pool);
  const auto api_before = megamoe::cuda::testing::cuda_api_audit_snapshot();
  auto transport_result = transport.run_device_simulation_synthetic(
      transport_dispatch, activation_lease, transport_config);
  const auto api_after = megamoe::cuda::testing::cuda_api_audit_snapshot();
  require(api_after.device_allocations == api_before.device_allocations &&
              api_after.device_frees == api_before.device_frees &&
              api_after.pinned_allocations == api_before.pinned_allocations &&
              api_after.pinned_frees == api_before.pinned_frees &&
              api_after.event_creates == api_before.event_creates &&
              api_after.event_destroys == api_before.event_destroys,
          "steady activation transport allocated CUDA resources");
  const auto downloaded_transport =
      megamoe::cuda::download_activation_transport_for_validation(
          transport_result, execution);
  require(downloaded_transport.contribution_by_route ==
              oracle.contribution_by_route &&
              downloaded_transport.combined_tokens == oracle.combined_tokens,
          "device transport synthetic contribution/combine oracle mismatch");
  require(std::equal(oracle.canonical_expert_rows.begin(),
                     oracle.canonical_expert_rows.end(),
                     downloaded_transport.canonical_expert_rows.begin()),
          "device transport canonical expert rows mismatch");
  require(downloaded_transport.protocol.dispatch_packets == 3 &&
              downloaded_transport.protocol.relayed_fanout == 9 &&
              downloaded_transport.protocol.reduced_tokens == 3,
          "device transport phase counters mismatch");
  require(transport_result.audit().backend ==
              megamoe::transport::ActivationTransportBackend::DeviceSimulation,
          "simulation backend was mislabeled as P2P");
  const auto transport_baseline = downloaded_transport.combined_tokens;
  transport_result = {};
  activation_lease = {};
  for (std::uint32_t repeat = 0; repeat < 20; ++repeat) {
    auto repeated_activation =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, 3, 33, context, execution, transport_pool);
    auto repeated_result = transport.run_device_simulation_synthetic(
        transport_dispatch, repeated_activation, transport_config);
    const auto repeated =
        megamoe::cuda::download_activation_transport_for_validation(
            repeated_result, execution);
    require(repeated.combined_tokens == transport_baseline,
            "ring epoch reuse changed device transport bytes");
  }
  require(transport_pool.usage().slots_available == 2 &&
              transport_pool.usage().slots_in_use == 0,
          "transport slots were not returned after final lease destruction");

  auto next_transport_session =
      megamoe::make_collective_session("device-only", 1, 5002);
  auto next_transport_context = megamoe::make_rank_collective_context(
      next_transport_session, 0, execution.token());
  auto wrong_generation_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, next_transport_context, execution,
          transport_pool);
  bool generation_failed = false;
  try {
    (void)transport.run_device_simulation_synthetic(transport_dispatch,
                                  wrong_generation_activation,
                                  transport_config);
  } catch (const megamoe::NotifyDispatchError &error) {
    generation_failed = error.code() == megamoe::NotifyErrorCode::WorkspaceMismatch;
  }
  require(generation_failed,
          "cross-generation activation lease was accepted");
  wrong_generation_activation = {};

  auto exhausted_config = transport_pool_config;
  exhausted_config.max_inflight_generations = 1;
  auto exhausted_pool = megamoe::cuda::make_activation_transport_pool(
      session, execution, exhausted_config);
  auto held_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activations, 3, 33, context, execution, exhausted_pool);
  bool exhausted = false;
  try {
    (void)megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
        activations, 3, 33, context, execution, exhausted_pool);
  } catch (const megamoe::NotifyDispatchError &error) {
    exhausted = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted && exhausted_pool.usage().slots_in_use == 1,
          "transport pool exhaustion did not fail closed");
  held_activation = {};
  require(exhausted_pool.usage().slots_available == 1,
          "transport pool did not recover after final input lease");

  megamoe::transport::ActivationTransportConfig wide_transport_config;
  wide_transport_config.hidden_size = 7;
  wide_transport_config.channels = 1;
  wide_transport_config.ring_slots_per_source_channel = 1;
  megamoe::cuda::ActivationTransportPoolConfig wide_transport_pool_config;
  wide_transport_pool_config.max_tokens_per_rank = 4;
  wide_transport_pool_config.max_hidden_size = 7;
  wide_transport_pool_config.max_top_k = 6;
  wide_transport_pool_config.max_world_size = 73;
  wide_transport_pool_config.max_nodes = 73;
  wide_transport_pool_config.max_destination_rows_per_rank = 32;
  wide_transport_pool_config.channels = 1;
  wide_transport_pool_config.ring_slots_per_source_channel = 1;
  wide_transport_pool_config.max_inflight_generations = 1;
  auto wide_transport_pool = megamoe::cuda::make_activation_transport_pool(
      wide_session, execution, wide_transport_pool_config);
  std::vector<megamoe::transport::BFloat16> wide_activations(7);
  for (std::size_t i = 0; i < wide_activations.size(); ++i)
    wide_activations[i] = megamoe::transport::float_to_bfloat16(
        static_cast<float>(i + 1U) / 8.0F);
  auto wide_activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          wide_activations, 1, 7, wide_context, execution,
          wide_transport_pool);
  megamoe::cuda::DeviceActivationTransport wide_transport(
      wide_session, execution, wide_transport_pool);
  auto wide_transport_result = wide_transport.run_device_simulation_synthetic(
      wide_handle, wide_activation, wide_transport_config);
  auto gate_before = wide_transport_config;
  gate_before.gate_placement =
      megamoe::transport::ActivationGatePlacement::DestinationBeforeUnsupported;
  bool gate_before_failed = false;
  try {
    (void)wide_transport.run_device_simulation_synthetic(
        wide_handle, wide_activation, gate_before);
  } catch (const megamoe::NotifyDispatchError &) {
    gate_before_failed = true;
  }
  require(gate_before_failed,
          "unsupported gate-before placement was silently accepted");
  const auto wide_transport_download =
      megamoe::cuda::download_activation_transport_for_validation(
          wide_transport_result, execution);
  const auto wide_transport_oracle =
      megamoe::transport::simulate_activation_transport(
          expected_wide[0], wide_activations,
          std::vector<float>(
              expected_wide[0]->outgoing.expert_id.size(), 1.0F),
          wide_transport_config, 1);
  require(wide_transport_download.contribution_by_route ==
              wide_transport_oracle.contribution_by_route &&
              wide_transport_download.combined_tokens ==
                  wide_transport_oracle.combined_tokens &&
              wide_transport_download.protocol.dispatch_packets == 6,
          "73-node full device transport/oracle mismatch");
  const auto capability =
      megamoe::cuda::probe_activation_transport_capability(2);
  require(capability.nccl_device_headers &&
              capability.gin_headers_available &&
              capability.gin_backend_implemented && !capability.gin_usable,
          "GIN header/runtime capability boundary was not detected");
  require(megamoe::cuda::testing::run_local_p2p_put_signal_probe(0, 1, 4096),
          "real CUDA P2P put+system-signal probe failed");
  std::cout << "[PASS] transport-only persistent device worker, deterministic "
               "synthetic combine, pooled steady path, P2P put/signal, GIN "
               "fail-closed\n";
  auto short_plan = wide_plan;
  --short_plan.device_bytes;
  bool short_failed = false;
  try {
    (void)megamoe::cuda::allocate_cuda_notify_workspace(
        short_plan, wide_context, execution);
  } catch (const megamoe::NotifyDispatchError &) {
    short_failed = true;
  }
  require(short_failed, "workspace capacity minus one byte did not fail");
  std::cout << "[PASS] full fake device materializer including 73-node "
               "multiword descriptors, 20 byte-identical repeats\n";
}
void test_delayed_route_producer_readiness() {
  const auto config = make_config(1, 1, 8, 1, 4);
  auto producer = megamoe::cuda::make_cuda_execution("delay-producer", 0);
  auto consumer = megamoe::cuda::make_cuda_execution("delay-consumer", 0);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto base = megamoe::cuda::make_nccl_session(
      "delay-session", 1, 0, 6900, 0, bootstrap);
  const auto producer_context = megamoe::make_rank_collective_context(
      base.session(), 0, producer.token());
  auto storage = megamoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, producer_context, producer);
  auto delayed = megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      storage, 5, 5000000, producer_context, producer);
  const auto consumer_context = megamoe::make_rank_collective_context(
      base.session(), 0, consumer.token());
  auto imported = megamoe::cuda::import_selected_expert_routes(
      delayed, consumer_context, consumer);
  auto pool = megamoe::cuda::make_device_notify_resource_pool(
      base, consumer, {config, 2, 4096, 1, 2});
  megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
      base, consumer, pool);
  auto imported_gates = make_unit_gates(imported, consumer_context, consumer);
  auto result = coordinator.run_device(
      {config, imported, imported_gates, 2});
  const auto downloaded =
      megamoe::cuda::download_device_metadata_for_validation(
          result.dispatch(), consumer);
  require(downloaded->outgoing.expert_id == std::vector<std::uint32_t>{5},
          "cross-stream Notify observed poison instead of ready-event result");

  auto next = megamoe::cuda::make_nccl_generation(
      base, "delay-session", 6901);
  const auto next_producer_context = megamoe::make_rank_collective_context(
      next.session(), 0, producer.token());
  auto delayed_without_ready =
      megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer, false);
  const auto next_consumer_context = megamoe::make_rank_collective_context(
      next.session(), 0, consumer.token());
  bool missing_ready_failed = false;
  try {
    (void)megamoe::cuda::import_selected_expert_routes(
        delayed_without_ready, next_consumer_context, consumer);
  } catch (const megamoe::NotifyDispatchError &) {
    missing_ready_failed = true;
  }
  require(missing_ready_failed,
          "unrecorded producer-ready event was accepted");

  auto unsafe_delayed =
      megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
          storage, 4, 100000000, next_producer_context, producer);
  auto unsafe =
      megamoe::cuda::DeviceRouteProvenanceBridge::unsafe_import_without_wait_for_testing(
          unsafe_delayed, next_consumer_context, consumer);
  bool unsafe_failed = false;
  try {
    megamoe::cuda::CudaNotifyGenerationCoordinator unsafe_coordinator(
        next, consumer, pool);
    auto unsafe_gates = make_unit_gates(unsafe, next_consumer_context,
                                        consumer);
    (void)unsafe_coordinator.run_device(
        {config, unsafe, unsafe_gates, 2});
  } catch (const megamoe::NotifyDispatchError &) {
    unsafe_failed = true;
  }
  require(unsafe_failed,
          "negative no-wait seam did not demonstrate producer race");

  auto same_session = megamoe::cuda::make_nccl_session(
      "delay-same-stream", 1, 0, 6920, 0,
      megamoe::cuda::make_nccl_bootstrap_id());
  const auto same_context = megamoe::make_rank_collective_context(
      same_session.session(), 0, producer.token());
  auto same_storage = megamoe::cuda::upload_cuda_routes(
      {1, {0}}, 1, same_context, producer);
  auto same_delayed = megamoe::cuda::DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
      same_storage, 6, 5000000, same_context, producer);
  auto same_pool = megamoe::cuda::make_device_notify_resource_pool(
      same_session, producer, {config, 2, 4096, 1, 2});
  megamoe::cuda::CudaNotifyGenerationCoordinator same_coordinator(
      same_session, producer, same_pool);
  auto same_gates = make_unit_gates(same_delayed, same_context, producer);
  auto same_result = same_coordinator.run_device(
      {config, same_delayed, same_gates, 2});
  const auto same_download =
      megamoe::cuda::download_device_metadata_for_validation(
          same_result.dispatch(), producer);
  require(same_download->outgoing.expert_id ==
              std::vector<std::uint32_t>{6},
          "same-stream producer order was not preserved");
  std::cout << "[PASS] delayed producer ready-event cross/same-stream + "
               "no-wait negative seam\n";
}

void test_device_dispatch_handle_lifetime() {
  const auto config = make_config(1, 1, 8, 3, 4);
  const megamoe::RankRoutingInput routes{2, {7, 1, 4, 0, 6, 2}};
  auto execution = megamoe::cuda::make_cuda_execution("handle-exec", 0);
  megamoe::cuda::DeviceDispatchHandle retained;
  {
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto session = megamoe::cuda::make_nccl_session(
        "handle-lifetime", 1, 0, 6100, 0, bootstrap);
    auto coordinator =
        megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
            session, execution);
    auto result = coordinator.run({config, routes, 10});
    auto first_copy = result.dispatch();
    retained = first_copy;
    auto moved = std::move(first_copy);
    require(!first_copy && moved, "device handle move state is unsafe");
    const auto &identity = moved.identity();
    const auto expectation = megamoe::cuda::DeviceDispatchViewExpectation{
        6100, 0, 1, routes.num_tokens, config.top_k,
        identity.config_digest_word0, identity.config_digest_word1};
    auto consumer = moved.acquire_consumer(execution, expectation);
    require(consumer && consumer.state() ==
                            megamoe::cuda::DeviceDispatchConsumerState::Acquired,
            "owning device consumer was not acquired");
    consumer.finish();
  }
  require(static_cast<bool>(retained),
          "device handle did not outlive coordinator/result");
  const auto actual = megamoe::cuda::download_device_metadata_for_validation(
      retained, execution);
  const auto expected = megamoe::build_notify_dispatch_for_testing(
      config, {routes}, "handle-lifetime", 6100);
  require(megamoe::serialize_dispatch_metadata(actual) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "retained device handle metadata changed after owner destruction");
  bool mismatch_failed = false;
  try {
    (void)retained.acquire_consumer(
        execution,
        {6101, 0, 1, routes.num_tokens, config.top_k,
         retained.identity().config_digest_word0,
         retained.identity().config_digest_word1});
  } catch (const megamoe::NotifyDispatchError &) {
    mismatch_failed = true;
  }
  require(mismatch_failed,
          "cross-generation device view mismatch did not fail closed");
  auto second_execution =
      megamoe::cuda::make_cuda_execution("handle-consumer-stream-2", 0);
  const auto retained_id = retained.identity();
  const megamoe::cuda::DeviceDispatchViewExpectation retained_expectation{
      retained_id.generation, retained_id.calling_rank, retained_id.world_size,
      retained_id.num_tokens, retained_id.top_k,
      retained_id.config_digest_word0, retained_id.config_digest_word1};
  auto first_stream_consumer =
      retained.acquire_consumer(execution, retained_expectation);
  auto second_stream_consumer =
      retained.acquire_consumer(second_execution, retained_expectation);
  megamoe::cuda::testing::submit_dispatch_probe(first_stream_consumer, 10000);
  megamoe::cuda::testing::submit_dispatch_probe(second_stream_consumer, 10000);
  retained = {};
  (void)megamoe::cuda::testing::finish_dispatch_probe(first_stream_consumer);
  (void)megamoe::cuda::testing::finish_dispatch_probe(second_stream_consumer);
  megamoe::cuda::DeviceDispatchHandle empty;
  bool empty_failed = false;
  try {
    (void)empty.sizes();
  } catch (const megamoe::NotifyDispatchError &) {
    empty_failed = true;
  }
  require(empty_failed, "empty device handle accessor did not fail closed");
  std::cout << "[PASS] immutable device handle copy/move/lifetime/consumer checks\n";
}

void test_resource_pool_and_cuda_api_audit() {
  const auto config = make_config(1, 1, 16, 4, 7);
  const auto routes = seeded_inputs(1, 16, 4, 0xabcddcbaULL).front();
  auto execution = megamoe::cuda::make_cuda_execution("pool-exec", 0);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto base = megamoe::cuda::make_nccl_session(
      "pool-base", 1, 0, 6200, 0, bootstrap);
  auto pool = megamoe::cuda::make_device_notify_resource_pool(
      base, execution,
      {config, 10, 4096, 2});
  const auto initial_usage = pool.usage();
  require(initial_usage.slot_count == 2 && initial_usage.slots_in_use == 0 &&
              initial_usage.device_bytes != 0 &&
              initial_usage.pinned_host_bytes != 0 &&
              initial_usage.event_count == 30,
          "session resource pool capacity/accounting mismatch");

  auto device_routes = megamoe::cuda::upload_cuda_routes(
      routes, config.top_k,
      megamoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  auto device_gates = make_unit_gates(
      device_routes,
      megamoe::make_rank_collective_context(base.session(), 0,
                                            execution.token()),
      execution);
  const auto exact_routes = [&](const megamoe::cuda::NcclSession &session) {
    return megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
        device_routes,
        megamoe::make_rank_collective_context(session.session(), 0,
                                              execution.token()),
        execution);
  };
  const auto exact_request = [&](const megamoe::cuda::NcclSession &session) {
    auto routes_for_generation = exact_routes(session);
    auto gates_for_generation =
        megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
            device_gates, routes_for_generation,
            megamoe::make_rank_collective_context(
                session.session(), 0, execution.token()),
            execution);
    return megamoe::cuda::DeviceNotifyGenerationRequest{
        config, std::move(routes_for_generation),
        std::move(gates_for_generation), 10};
  };
  {
    auto next = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6201);
    auto next_routes = exact_routes(next);
    bool stale_gate_failed = false;
    try {
      megamoe::cuda::DeviceGateProvenanceBridge::validate(
          device_gates, next_routes,
          megamoe::make_rank_collective_context(
              next.session(), 0, execution.token()),
          execution);
    } catch (const megamoe::NotifyDispatchError &) {
      stale_gate_failed = true;
    }
    require(stale_gate_failed,
            "generation-6200 gate lease was accepted at generation 6201");
  }
  for (std::uint64_t generation = 6201; generation <= 6210; ++generation) {
    bool rejected = false;
    try {
      auto mismatched = megamoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          mismatched, execution, pool);
      (void)coordinator.run_device(
          {config, device_routes, device_gates, 10});
    } catch (const megamoe::NotifyDispatchError &error) {
      const std::string message = error.what();
      rejected = message.find("field=generation") != std::string::npos &&
                 message.find("expected=" + std::to_string(generation)) !=
                     std::string::npos &&
                 message.find("actual=6200") != std::string::npos;
    }
    require(rejected,
            "generation-6200 route lease was accepted by a later generation");
    const auto rebound = exact_routes(megamoe::cuda::make_nccl_generation(
        base, "pool-base", generation));
    require(rebound.route_generation() == generation &&
                device_routes.route_generation() == 6200,
            "private exact-generation reissue mutated or forged provenance");
  }
  const auto before = megamoe::cuda::testing::cuda_api_audit_snapshot();
  megamoe::cuda::DeviceDispatchHandle retained;
  for (std::uint64_t iteration = 0; iteration < 10; ++iteration) {
    auto session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6201 + iteration);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    auto result = coordinator.run_device(exact_request(session));
    require(result.dispatch().audit().bulk_route_d2h == 0 &&
                result.dispatch().audit().bulk_collective_d2h == 0 &&
                result.dispatch().audit().host_materializations == 0 &&
                result.dispatch().audit().bulk_metadata_h2d == 0,
            "pooled device generation used bulk host staging");
    if (iteration >= 2U) {
      const auto audit = result.audit();
#if MEGAMOE_DEBUG_VALIDATE
      require(audit.fused_notify_starts == 0U &&
                  audit.control_starts == 1U && audit.config_starts == 1U &&
                  audit.expert_count_starts == 1U &&
                  audit.packet_summary_starts == 1U,
              "debug generation did not retain the full Notify protocol");
#else
      require(audit.fused_notify_starts == 1U &&
                  audit.control_starts == 0U && audit.config_starts == 0U &&
                  audit.expert_count_starts == 0U &&
                  audit.packet_summary_starts == 0U,
              "steady single-node release generation did not use one fused "
              "Notify collective");
#endif
    }
    if (iteration == 0)
      retained = result.dispatch();
  }
  const auto after = megamoe::cuda::testing::cuda_api_audit_snapshot();
  require(after.device_allocations == before.device_allocations &&
              after.device_frees == before.device_frees &&
              after.pinned_allocations == before.pinned_allocations &&
              after.pinned_frees == before.pinned_frees &&
              after.event_creates == before.event_creates &&
              after.event_destroys == before.event_destroys &&
              after.device_synchronizes == before.device_synchronizes,
          "steady generation performed forbidden CUDA resource API call");
  require(after.h2d_bytes - before.h2d_bytes < 1024U * 1024U &&
              after.d2h_bytes - before.d2h_bytes < 1024U * 1024U &&
              after.d2d_bytes - before.d2d_bytes < 1024U * 1024U,
          "steady generation performed an unexpected bulk memcpy");
  require(pool.usage().slots_in_use == 1,
          "retained handle did not keep exactly one pool slot leased");

#if !MEGAMOE_DEBUG_VALIDATE
  const auto require_base_recovery = [&](std::uint64_t generation,
                                         const char *failure) {
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", generation);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    auto recovered = recovery.run_device(exact_request(recovery_session));
    const auto audit = recovery.audit();
    require(recovered && audit.fused_notify_starts == 1U &&
                audit.control_starts == 0U && audit.config_starts == 0U &&
                audit.expert_count_starts == 0U &&
                audit.packet_summary_starts == 0U,
            failure);
    const auto actual = megamoe::serialize_dispatch_metadata(
        megamoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, {routes}, "pool-base", generation);
    require(actual == megamoe::serialize_dispatch_metadata(expected.front()),
            "release protocol recovery metadata/hash mismatch");
  };

  // A predictable route error does not change the cached static layout.  It
  // must therefore be dataized in the fixed-width fused record instead of
  // returning locally before the collective.
  auto invalid_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6299);
  megamoe::cuda::CudaNotifyGenerationCoordinator invalid_coordinator(
      invalid_session, execution, pool);
  auto invalid_routes = routes;
  invalid_routes.selected_experts.front() =
      static_cast<std::int32_t>(config.num_experts);
  bool fused_status_failed = false;
  try {
    (void)invalid_coordinator.run({config, invalid_routes, 10});
  } catch (const megamoe::NotifyDispatchError &error) {
    fused_status_failed =
        error.code() == megamoe::NotifyErrorCode::ControlProtocolFailure;
  }
  const auto invalid_audit = invalid_coordinator.audit();
  require(fused_status_failed && invalid_audit.fused_notify_starts == 1U &&
              invalid_audit.control_starts == 0U &&
              invalid_audit.config_starts == 0U &&
              invalid_audit.expert_count_starts == 0U &&
              invalid_audit.packet_summary_starts == 0U,
          "invalid local status did not fail after exactly one fused Notify "
          "collective");
  require(pool.usage().slots_in_use == 1,
          "failed fused Notify generation did not return its pool slot");
  require_base_recovery(
      6300, "same-communicator recovery after fused route failure did not "
            "use the cached release protocol");

  // A request that changes a layout-relevant field cannot use a record laid
  // out from the cached config.  It must enter the complete protocol and fail
  // at control validation; forcing it through fused would interpret the old
  // layout.  Returning to the cached config on the same communicator must
  // recover through fused and reproduce the oracle bytes.
  auto changed_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6301);
  megamoe::cuda::CudaNotifyGenerationCoordinator changed_coordinator(
      changed_session, execution, pool);
  auto changed_config = config;
  changed_config.top_k = changed_config.num_experts + 1U;
  bool changed_config_failed = false;
  try {
    (void)changed_coordinator.run({changed_config, routes, 10});
  } catch (const megamoe::NotifyDispatchError &error) {
    changed_config_failed =
        error.code() == megamoe::NotifyErrorCode::ControlProtocolFailure;
  }
  const auto changed_audit = changed_coordinator.audit();
  require(changed_config_failed && changed_audit.fused_notify_starts == 0U &&
              changed_audit.control_starts == 1U &&
              changed_audit.config_starts == 0U &&
              changed_audit.expert_count_starts == 0U &&
              changed_audit.packet_summary_starts == 0U,
          "layout-changing invalid config did not fail at the complete "
          "protocol control boundary");
  require(pool.usage().slots_in_use == 1,
          "failed full-protocol config generation did not return its slot");
  require_base_recovery(
      6302, "same-communicator recovery after config fallback did not return "
            "to the cached release protocol");
#endif

  auto second_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6310);
  megamoe::cuda::CudaNotifyGenerationCoordinator second_coordinator(
      second_session, execution, pool);
  auto second = second_coordinator.run_device(exact_request(second_session));
  require(pool.usage().slots_in_use == 2,
          "two live handles did not occupy two exclusive pool slots");
  bool exhausted = false;
  try {
    auto third_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6311);
    megamoe::cuda::CudaNotifyGenerationCoordinator third_coordinator(
        third_session, execution, pool);
    (void)third_coordinator.run_device(exact_request(third_session));
  } catch (const megamoe::NotifyDispatchError &error) {
    exhausted = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
  }
  require(exhausted, "pool exhaustion did not fail with ResourceExhausted");
  second = {};
  require(pool.usage().slots_in_use == 1,
          "last handle destruction did not return its slot");
  megamoe::cuda::testing::inject_next_nccl_fault(
      megamoe::cuda::testing::NcclFaultPoint::ControlStart);
  bool injected_failed = false;
  try {
    auto failed_session = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6312);
    megamoe::cuda::CudaNotifyGenerationCoordinator failed_coordinator(
        failed_session, execution, pool);
    (void)failed_coordinator.run_device(exact_request(failed_session));
  } catch (const megamoe::NotifyDispatchError &) {
    injected_failed = true;
  }
  require(injected_failed && pool.usage().slots_in_use == 1,
          "failed generation leaked or published a pool slot");
  auto recovery_session = megamoe::cuda::make_nccl_generation(
      base, "pool-base", 6313);
  megamoe::cuda::CudaNotifyGenerationCoordinator recovery_coordinator(
      recovery_session, execution, pool);
  auto recovered = recovery_coordinator.run_device(
      exact_request(recovery_session));
  require(recovered && pool.usage().slots_in_use == 2,
          "released pool slot was not reusable");

  const auto actual = megamoe::cuda::download_device_metadata_for_validation(
      recovered.dispatch(), execution);
  const auto expected = megamoe::build_notify_dispatch_for_testing(
      config, {routes}, "pool-base", 6313);
  require(megamoe::serialize_dispatch_metadata(actual) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "pooled device input metadata mismatch");
  auto ownerless_handle = recovered.dispatch();
  recovered = {};
  retained = {};
  pool = {};
  require(megamoe::serialize_dispatch_metadata(
              megamoe::cuda::download_device_metadata_for_validation(
                  ownerless_handle, execution)) ==
              megamoe::serialize_dispatch_metadata(expected.front()),
          "device handle did not safely outlive the public pool owner");
  ownerless_handle = {};

  for (const auto slot_count : {1U, 3U}) {
    auto sized_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, slot_count});
    std::vector<megamoe::cuda::ValidatedCudaGeneration> live;
    for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
      const auto generation = 6400U + slot_count * 10U + slot;
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base", generation);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      live.push_back(coordinator.run_device(exact_request(session)));
    }
    require(sized_pool.usage().slots_in_use == slot_count,
            "pool size 1/N did not retain every live slot");
    bool sized_exhausted = false;
    try {
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base",
          6500U + slot_count);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      (void)coordinator.run_device(exact_request(session));
    } catch (const megamoe::NotifyDispatchError &error) {
      sized_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(sized_exhausted,
            "pool size 1/N did not reject one generation beyond capacity");
    live.clear();
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N did not release all final handle leases");
    {
      auto session = megamoe::cuda::make_nccl_generation(
          base, "pool-base",
          6600U + slot_count);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, sized_pool);
      auto reused = coordinator.run_device(exact_request(session));
      require(static_cast<bool>(reused),
              "pool size 1/N released slot was not reusable");
    }
    require(sized_pool.usage().slots_in_use == 0,
            "pool size 1/N reuse leaked a lease");
  }

  {
    auto stale_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 1});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6700);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, stale_pool);
    auto gen1 = coordinator.run_device(exact_request(generation));
    auto handle = gen1.dispatch();
    const auto &id = handle.identity();
    auto consumer = handle.acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    bool consumer_exhausted = false;
    try {
      (void)handle.acquire_consumer(
          execution, {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, id.top_k, id.config_digest_word0,
                      id.config_digest_word1});
    } catch (const megamoe::NotifyDispatchError &error) {
      consumer_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(consumer_exhausted,
            "consumer event pool exhaustion did not fail closed");
    auto moved_consumer = std::move(consumer);
    require(!consumer && moved_consumer,
            "consumer lease move/moved-from state is unsafe");
    moved_consumer = std::move(moved_consumer);
    require(moved_consumer &&
                moved_consumer.state() ==
                    megamoe::cuda::DeviceDispatchConsumerState::Acquired,
            "consumer self-move corrupted the owning lease");
    megamoe::cuda::testing::submit_dispatch_probe(moved_consumer, 1000000);
    require(moved_consumer.state() ==
                megamoe::cuda::DeviceDispatchConsumerState::Submitted,
            "consumer probe was not submitted");
    handle = {};
    gen1 = {};
    require(stale_pool.usage().slots_in_use == 1,
            "submitted consumer did not retain generation slot");
    bool stale_exhausted = false;
    try {
      auto blocked_generation = megamoe::cuda::make_nccl_generation(
          base, "pool-base", 6701);
      megamoe::cuda::CudaNotifyGenerationCoordinator blocked(
          blocked_generation, execution, stale_pool);
      (void)blocked.run_device(exact_request(blocked_generation));
    } catch (const megamoe::NotifyDispatchError &error) {
      stale_exhausted =
          error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(stale_exhausted,
            "pool reused slot while stale consumer remained live");
    const auto stale_value =
        megamoe::cuda::testing::finish_dispatch_probe(moved_consumer);
    require(stale_value == expected.front()->outgoing.route_dst_row.front(),
            "old consumer did not read generation-1 canonical row");
    require(stale_pool.usage().slots_in_use == 0,
            "finished consumer did not release final slot lease");
    auto recovered_generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6702);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovered_coordinator(
        recovered_generation, execution, stale_pool);
    auto gen2 = recovered_coordinator.run_device(
        exact_request(recovered_generation));
    require(gen2 && stale_pool.usage().slots_in_use == 1,
            "slot was not reusable after final consumer completion");
  }


  {
    auto quarantine_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base", 6800);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, quarantine_pool);
    auto result = coordinator.run_device(exact_request(generation));
    auto handle = result.dispatch();
    const auto &id = handle.identity();
    auto consumer = handle.acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    megamoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
    megamoe::cuda::testing::inject_next_consumer_fault(
        megamoe::cuda::testing::ConsumerFaultPoint::EventAndStreamSynchronize);
    bool finish_failed = false;
    try {
      consumer.finish();
    } catch (const megamoe::NotifyDispatchError &) {
      finish_failed = true;
    }
    require(finish_failed, "consumer event+stream failure was not surfaced");
    const auto quarantined = quarantine_pool.usage();
    require(quarantined.slots_quarantined == 1 &&
                quarantined.consumer_events_quarantined == 1 &&
                quarantined.consumer_events_available == 1 &&
                quarantined.consumer_events_in_use == 0 &&
                quarantined.slots_available == 0 &&
                quarantined.slots_in_use == 0,
            "consumer failure did not independently quarantine event and slot");
    bool handle_failed = false;
    try {
      (void)handle.acquire_consumer(
          execution, {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, id.top_k, id.config_digest_word0,
                      id.config_digest_word1});
    } catch (const megamoe::NotifyDispatchError &) {
      handle_failed = true;
    }
    require(handle_failed,
            "quarantined generation handle accepted another consumer");
    bool next_failed = false;
    try {
      auto next = megamoe::cuda::make_nccl_generation(
          base, "pool-base", 6801);
      megamoe::cuda::CudaNotifyGenerationCoordinator blocked(
          next, execution, quarantine_pool);
      (void)blocked.run_device(exact_request(next));
    } catch (const megamoe::NotifyDispatchError &error) {
      next_failed = error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
    }
    require(next_failed,
            "quarantined slot was allocated to a later generation");
  }

  for (const auto fault : {
           megamoe::cuda::testing::ConsumerFaultPoint::EventRecord,
           megamoe::cuda::testing::ConsumerFaultPoint::EventSynchronize}) {
    auto event_pool = megamoe::cuda::make_device_notify_resource_pool(
        base, execution, {config, 10, 4096, 1, 2});
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "pool-base",
        fault == megamoe::cuda::testing::ConsumerFaultPoint::EventRecord
            ? 6810
            : 6811);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        generation, execution, event_pool);
    auto result = coordinator.run_device(exact_request(generation));
    const auto &id = result.dispatch().identity();
    auto consumer = result.dispatch().acquire_consumer(
        execution, {id.generation, id.calling_rank, id.world_size,
                    id.num_tokens, id.top_k, id.config_digest_word0,
                    id.config_digest_word1});
    megamoe::cuda::testing::inject_next_consumer_fault(fault);
    bool failed = false;
    try {
      megamoe::cuda::testing::submit_dispatch_probe(consumer, 1000);
      consumer.finish();
    } catch (const megamoe::NotifyDispatchError &) {
      failed = true;
    }
    require(failed, "consumer event-only fault was not surfaced");
    const auto usage = event_pool.usage();
    require(usage.slots_quarantined == 0 &&
                usage.slots_in_use == 1 &&
                usage.consumer_events_quarantined == 1 &&
                usage.consumer_events_available == 1 &&
                usage.consumer_events_in_use == 0,
            "event-only failure incorrectly recycled event or quarantined slot");
  }

  std::cout << "[PASS] exact route generation 6200->6201..6210 rejection + "
               "resource pool size=1/2/3 lease/exhaustion/reuse/quarantine + "
               "10-generation "
               "CUDA API audit alloc/free/pinned/event/device-sync=0/0/0/0/0, "
               "H2D_bytes="
            << after.h2d_bytes - before.h2d_bytes << ", D2H_bytes="
            << after.d2h_bytes - before.d2h_bytes << ", D2D_bytes="
            << after.d2d_bytes - before.d2d_bytes << '\n';
}
