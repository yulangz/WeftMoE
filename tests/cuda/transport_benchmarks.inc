struct Statistics {
  double mean = 0, p50 = 0, p95 = 0, minimum = 0, maximum = 0;
};

Statistics summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  double sum = 0;
  for (const auto value : values)
    sum += value;
  const auto percentile = [&](double fraction) {
    const auto ordinal = static_cast<std::size_t>(
        std::ceil(fraction * static_cast<double>(values.size())));
    return values[std::max<std::size_t>(1, ordinal) - 1];
  };
  return {sum / values.size(), percentile(.5), percentile(.95),
          values.front(), values.back()};
}

void transport_benchmark() {
  constexpr std::uint32_t world = 8;
  constexpr std::uint64_t tokens = 8192;
  constexpr std::uint64_t hidden = 7168;
  constexpr std::uint32_t experts = 384;
  constexpr std::uint32_t top_k = 6;
  const auto config = make_config(8, 1, experts, top_k, 64);
  // Use the exact requested shape and deterministic, duplicate-free tuples.
  std::vector<megamoe::RankRoutingInput> exact(world);
  std::vector<std::uint64_t> counts(
      static_cast<std::size_t>(world) * experts, 0);
  std::vector<std::uint64_t> summaries(
      static_cast<std::size_t>(world) * world, 0);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    exact[rank].num_tokens = tokens;
    for (std::uint64_t token = 0; token < tokens; ++token) {
      bool seen_node[world]{};
      const auto base = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        // Distinct experts on one destination node make the node-level
        // payload-dedup traffic reduction directly measurable.
        const auto expert = (base + slot * 8U) % experts;
        exact[rank].selected_experts.push_back(
            static_cast<std::int32_t>(expert));
        ++counts[static_cast<std::size_t>(rank) * experts + expert];
        seen_node[config.topology.expert_to_rank[expert]] = true;
      }
      for (std::uint32_t node = 0; node < world; ++node)
        summaries[static_cast<std::size_t>(rank) * world + node] +=
            seen_node[node] ? 1U : 0U;
    }
  }
  auto execution = megamoe::cuda::make_cuda_execution("transport-bench", 0);
  auto session = megamoe::make_collective_session("transport-bench", world,
                                                   9100);
  auto context = megamoe::make_rank_collective_context(
      session, 0, execution.token());
  auto dispatch = megamoe::cuda::testing::materialize_fake_device_collectives(
      config, context, execution, exact[0], tokens, counts, summaries);
  megamoe::cuda::ActivationTransportPoolConfig pool_config;
  pool_config.max_tokens_per_rank = tokens;
  pool_config.max_hidden_size = hidden;
  pool_config.max_top_k = top_k;
  pool_config.max_world_size = world;
  pool_config.max_nodes = world;
  pool_config.max_destination_rows_per_rank = tokens * top_k;
  pool_config.channels = 1;
  pool_config.ring_slots_per_source_channel = 8;
  pool_config.max_inflight_generations = 1;
  const auto setup_start = Clock::now();
  auto pool = megamoe::cuda::make_activation_transport_pool(
      session, execution, pool_config);
  const auto setup_us = std::chrono::duration<double, std::micro>(
                            Clock::now() - setup_start)
                            .count();
  std::vector<megamoe::transport::BFloat16> host(
      static_cast<std::size_t>(tokens * hidden));
  for (std::size_t i = 0; i < host.size(); ++i)
    host[i].bits = static_cast<std::uint16_t>(0x3f00U + (i % 127U));
  auto activation =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          host, tokens, hidden, context, execution, pool);
  megamoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = hidden;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 8;
  megamoe::cuda::DeviceActivationTransport transport(session, execution, pool);
  std::vector<double> total;
  std::vector<double> dispatch_relay;
  std::vector<double> combine_reduce;
  megamoe::cuda::ActivationTransportDeviceAudit last_audit;
  for (std::uint32_t iteration = 0; iteration < 60; ++iteration) {
    auto result = transport.run_device_simulation_synthetic(
        dispatch, activation, transport_config);
    if (iteration >= 10) {
      total.push_back(result.timings().total_us);
      dispatch_relay.push_back(result.timings().dispatch_relay_us);
      combine_reduce.push_back(result.timings().combine_reduce_us);
    }
    last_audit = result.audit();
  }
  const auto total_stats = summarize(total);
  const auto dispatch_stats = summarize(dispatch_relay);
  const auto reduce_stats = summarize(combine_reduce);
  const auto usage = pool.usage();
  const auto print_stats = [](const char *name, const Statistics &stats) {
    std::cout << "  " << name << " mean=" << stats.mean
              << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  };
  std::cout << "Activation transport benchmark: tokens/rank=" << tokens
            << " hidden=" << hidden << " top_k=" << top_k
            << " logical_world=" << world
            << " backend=device-simulation (no GIN/RDMA)\n"
            << "  one_time_pool_setup_us=" << setup_us << '\n'
            << "  raw_cross_node_payload_bytes="
            << last_audit.raw_cross_node_payload_bytes << '\n'
            << "  deduplicated_cross_node_payload_bytes="
            << last_audit.deduplicated_cross_node_payload_bytes << '\n'
            << "  device_workspace_bytes=" << usage.device_bytes << '\n'
            << "  pinned_workspace_bytes=" << usage.pinned_bytes << '\n';
  print_stats("dispatch_relay_us", dispatch_stats);
  print_stats("combine_reduce_us", reduce_stats);
  print_stats("transport_steady_total_us", total_stats);
  std::cout
      << "TRANSPORT_BENCH_JSON {\"schema\":1,\"warmup\":10,\"measured\":50,"
         "\"backend\":\"device-simulation\",\"one_time_pool_setup_us\":"
      << setup_us << ",\"raw_cross_node_payload_bytes\":"
      << last_audit.raw_cross_node_payload_bytes
      << ",\"deduplicated_cross_node_payload_bytes\":"
      << last_audit.deduplicated_cross_node_payload_bytes
      << ",\"device_workspace_bytes\":" << usage.device_bytes
      << ",\"pinned_workspace_bytes\":" << usage.pinned_bytes
      << ",\"dispatch_relay_us\":{\"mean\":" << dispatch_stats.mean
      << ",\"p50\":" << dispatch_stats.p50 << ",\"p95\":"
      << dispatch_stats.p95 << ",\"min\":" << dispatch_stats.minimum
      << ",\"max\":" << dispatch_stats.maximum
      << "},\"combine_reduce_us\":{\"mean\":" << reduce_stats.mean
      << ",\"p50\":" << reduce_stats.p50 << ",\"p95\":"
      << reduce_stats.p95 << ",\"min\":" << reduce_stats.minimum
      << ",\"max\":" << reduce_stats.maximum
      << "},\"steady_total_us\":{\"mean\":" << total_stats.mean
      << ",\"p50\":" << total_stats.p50 << ",\"p95\":"
      << total_stats.p95 << ",\"min\":" << total_stats.minimum
      << ",\"max\":" << total_stats.maximum << "}}\n";
}

void benchmark(std::uint32_t world) {
  const auto config = make_config(1, world, 384, 6, 64);
  std::vector<megamoe::RankRoutingInput> routes(world);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    routes[rank].num_tokens = 8192;
    for (std::uint64_t token = 0; token < 8192; ++token) {
      const auto base = (token * 17U + rank * 31U) % 384U;
      for (std::uint32_t slot = 0; slot < 6; ++slot)
        routes[rank].selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % 384U));
    }
  }
  constexpr std::size_t warmup = 10;
  constexpr std::size_t measured = 100;
  std::vector<double> steady;
  std::vector<double> setup;
  std::vector<double> generation_end_to_end;
  using SetupMember = double megamoe::cuda::GenerationSetupTimings::*;
  const std::vector<std::pair<const char *, SetupMember>> setup_stages{
      {"pool_acquire_us", &megamoe::cuda::GenerationSetupTimings::pool_acquire_us},
      {"coordinator_backend_create_us", &megamoe::cuda::GenerationSetupTimings::coordinator_backend_create_us},
      {"control_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::control_slot_allocation_us},
      {"config_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::config_slot_allocation_us},
      {"count_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::count_slot_allocation_us},
      {"summary_slot_allocation_us", &megamoe::cuda::GenerationSetupTimings::summary_slot_allocation_us},
      {"event_allocation_us", &megamoe::cuda::GenerationSetupTimings::event_allocation_us},
      {"pinned_allocation_us", &megamoe::cuda::GenerationSetupTimings::pinned_allocation_us},
      {"device_allocation_us", &megamoe::cuda::GenerationSetupTimings::device_allocation_us}};
  std::vector<std::vector<double>> setup_stage_samples(setup_stages.size());
  using ExecutionMember =
      double megamoe::cuda::GenerationExecutionTimings::*;
  const std::vector<std::pair<const char *, ExecutionMember>> execution_stages{
      {"local_mapping_us",
       &megamoe::cuda::GenerationExecutionTimings::local_mapping_us},
      {"control_config_us",
       &megamoe::cuda::GenerationExecutionTimings::control_config_us},
      {"expert_count_us",
       &megamoe::cuda::GenerationExecutionTimings::expert_count_us},
      {"packet_summary_us",
       &megamoe::cuda::GenerationExecutionTimings::packet_summary_us},
      {"device_materialization_us",
       &megamoe::cuda::GenerationExecutionTimings::device_materialization_us},
      {"prefix_planning_us",
       &megamoe::cuda::GenerationExecutionTimings::prefix_planning_us},
      {"remaining_parallel_materialization_us",
       &megamoe::cuda::GenerationExecutionTimings::remaining_parallel_materialization_us},
      {"pool_acquire_release_us",
       &megamoe::cuda::GenerationExecutionTimings::pool_acquire_release_us}};
  std::vector<std::vector<double>> execution_stage_samples(
      execution_stages.size());
  megamoe::cuda::NcclWorkspaceUsage peak;
  // One communicator/session establishment per rank, once for the complete
  // 10+50 run. Samples are the ranks participating in that one establishment.
  auto resources =
      make_reusable_resources(config, "benchmark-session", 2999, 8192);
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    const auto context = megamoe::make_rank_collective_context(
        resources[rank].base_session.session(), rank,
        resources[rank].execution.token());
    resources[rank].device_routes =
        megamoe::cuda::upload_cuda_routes(
            routes[rank], config.top_k,
            context,
            resources[rank].execution);
    resources[rank].device_gates =
        megamoe::cuda::testing::upload_fp32_gates_for_testing(
            fixed_random_selected_gates(routes[rank].num_tokens,
                                        config.top_k, rank,
                                        0x4d4547414d4f4547ULL),
            resources[rank].device_routes, context,
            resources[rank].execution);
  }
  std::vector<double> session_setup;
  std::vector<double> pool_setup;
  for (const auto &resource : resources)
    session_setup.push_back(resource.session_setup_us);
  for (const auto &resource : resources)
    pool_setup.push_back(resource.pool_setup_us);
  const auto session_setup_stats = summarize(session_setup);
  const auto pool_setup_stats = summarize(pool_setup);
  for (std::size_t iteration = 0; iteration < warmup + measured; ++iteration) {
    const auto result = run_reused_fixture(
        config, routes, resources, "benchmark-session",
        3000 + iteration, 8192);
    rethrow_errors(result);
    if (iteration >= warmup) {
      double slow_wall = 0;
      double slow_setup = 0;
      double slow_steady = 0;
      std::vector<double> slow_setup_stages(setup_stages.size(), 0.0);
      std::vector<double> slow_execution_stages(execution_stages.size(), 0.0);
      for (const auto &rank : result) {
        const auto rank_steady = rank.execution.steady_execution_us;
        const auto rank_setup = rank.setup.per_generation_setup_total_us;
        // Compute setup-inclusive E2E per rank first; only then select the
        // iteration's slowest rank. No cross-rank max subtraction is used.
        slow_wall = std::max(slow_wall, rank_setup + rank_steady);
        slow_setup = std::max(slow_setup, rank_setup);
        slow_steady = std::max(slow_steady, rank_steady);
        for (std::size_t stage = 0; stage < setup_stages.size(); ++stage)
          slow_setup_stages[stage] = std::max(
              slow_setup_stages[stage], rank.setup.*setup_stages[stage].second);
        for (std::size_t stage = 0; stage < execution_stages.size(); ++stage)
          slow_execution_stages[stage] = std::max(
              slow_execution_stages[stage],
              rank.execution.*execution_stages[stage].second);
        peak.device_bytes = std::max(peak.device_bytes,
                                     rank.workspace.device_bytes);
        peak.pinned_host_bytes = std::max(
            peak.pinned_host_bytes, rank.workspace.pinned_host_bytes);
        peak.control_slot_bytes = std::max(
            peak.control_slot_bytes, rank.workspace.control_slot_bytes);
        peak.config_wire_bytes = std::max(
            peak.config_wire_bytes, rank.workspace.config_wire_bytes);
        peak.config_slot_bytes = std::max(
            peak.config_slot_bytes, rank.workspace.config_slot_bytes);
        peak.expert_slot_bytes = std::max(
            peak.expert_slot_bytes, rank.workspace.expert_slot_bytes);
        peak.summary_slot_bytes = std::max(
            peak.summary_slot_bytes, rank.workspace.summary_slot_bytes);
        peak.route_input_device_bytes = std::max(
            peak.route_input_device_bytes,
            rank.workspace.route_input_device_bytes);
        peak.device_notify_arena_bytes = std::max(
            peak.device_notify_arena_bytes,
            rank.workspace.device_notify_arena_bytes);
        peak.device_notify_pinned_staging_bytes = std::max(
            peak.device_notify_pinned_staging_bytes,
            rank.workspace.device_notify_pinned_staging_bytes);
        peak.final_device_metadata_arena_view_capacity_bytes = std::max(
            peak.final_device_metadata_arena_view_capacity_bytes,
            rank.workspace.final_device_metadata_arena_view_capacity_bytes);
        peak.event_count = std::max(peak.event_count,
                                    rank.workspace.event_count);
        peak.quantified_event_bytes = std::max(
            peak.quantified_event_bytes,
            rank.workspace.quantified_event_bytes);
      }
      steady.push_back(slow_steady);
      setup.push_back(slow_setup);
      generation_end_to_end.push_back(slow_wall);
      for (std::size_t stage = 0; stage < setup_stages.size(); ++stage)
        setup_stage_samples[stage].push_back(slow_setup_stages[stage]);
      for (std::size_t stage = 0; stage < execution_stages.size(); ++stage)
        execution_stage_samples[stage].push_back(
            slow_execution_stages[stage]);
    }
  }
  const auto steady_stats = summarize(steady);
  const auto setup_stats = summarize(setup);
  const auto generation_stats = summarize(generation_end_to_end);
  std::vector<Statistics> setup_stage_stats;
  for (auto &samples : setup_stage_samples)
    setup_stage_stats.push_back(summarize(std::move(samples)));
  std::vector<Statistics> execution_stage_stats;
  for (auto &samples : execution_stage_samples)
    execution_stage_stats.push_back(summarize(std::move(samples)));
  std::cout << "BENCH world=" << world
            << " warmup=" << warmup << " measured=" << measured
            << " aggregation=slowest-rank"
               " session_reused=true\n"
            << "  one_time_session_setup_us mean="
            << session_setup_stats.mean << " p50=" << session_setup_stats.p50
            << " p95=" << session_setup_stats.p95
            << " min=" << session_setup_stats.minimum
            << " max=" << session_setup_stats.maximum << '\n'
            << "  one_time_pool_setup_us mean=" << pool_setup_stats.mean
            << " p50=" << pool_setup_stats.p50
            << " p95=" << pool_setup_stats.p95
            << " min=" << pool_setup_stats.minimum
            << " max=" << pool_setup_stats.maximum << '\n'
            << "  per_generation_setup_us mean=" << setup_stats.mean
            << " p50=" << setup_stats.p50 << " p95=" << setup_stats.p95
            << " min=" << setup_stats.minimum
            << " max=" << setup_stats.maximum << '\n'
            << "  setup_inclusive_generation_end_to_end_us mean="
            << generation_stats.mean << " p50=" << generation_stats.p50
            << " p95=" << generation_stats.p95
            << " min=" << generation_stats.minimum
            << " max=" << generation_stats.maximum << '\n'
            << "  steady_state_execution_us mean=" << steady_stats.mean
            << " p50=" << steady_stats.p50
            << " p95=" << steady_stats.p95
            << " min=" << steady_stats.minimum
            << " max=" << steady_stats.maximum << '\n';
  for (std::size_t stage = 0; stage < setup_stages.size(); ++stage) {
    const auto &stats = setup_stage_stats[stage];
    std::cout << "  " << setup_stages[stage].first << " mean=" << stats.mean
              << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  }
  for (std::size_t stage = 0; stage < execution_stages.size(); ++stage) {
    const auto &stats = execution_stage_stats[stage];
    std::cout << "  " << execution_stages[stage].first << " mean="
              << stats.mean << " p50=" << stats.p50 << " p95=" << stats.p95
              << " min=" << stats.minimum << " max=" << stats.maximum
              << '\n';
  }
  std::cout << "BENCH_JSON {\"schema\":7,\"warmup\":" << warmup
            << ",\"measured\":" << measured << ','
            << "\"aggregation\":\"slowest-rank\",\"session_reused\":true,"
               "\"session_setup_sample\":\"one establishment; samples are ranks\","
               "\"one_time_session_setup_us\":{\"mean\":"
            << session_setup_stats.mean << ",\"p50\":"
            << session_setup_stats.p50 << ",\"p95\":"
            << session_setup_stats.p95 << ",\"min\":"
            << session_setup_stats.minimum << ",\"max\":"
            << session_setup_stats.maximum
            << "},\"one_time_pool_setup_us\":{\"mean\":"
            << pool_setup_stats.mean << ",\"p50\":"
            << pool_setup_stats.p50 << ",\"p95\":"
            << pool_setup_stats.p95 << ",\"min\":"
            << pool_setup_stats.minimum << ",\"max\":"
            << pool_setup_stats.maximum
            << "},\"per_generation_setup_us\":{\"mean\":"
            << setup_stats.mean << ",\"p50\":" << setup_stats.p50
            << ",\"p95\":" << setup_stats.p95 << ",\"min\":"
            << setup_stats.minimum << ",\"max\":" << setup_stats.maximum
            << "},\"setup_inclusive_generation_end_to_end_us\":{\"mean\":"
            << generation_stats.mean << ",\"p50\":"
            << generation_stats.p50 << ",\"p95\":"
            << generation_stats.p95 << ",\"min\":"
            << generation_stats.minimum << ",\"max\":"
            << generation_stats.maximum
            << "},\"steady_state_execution_us\":{\"mean\":"
            << steady_stats.mean << ",\"p50\":" << steady_stats.p50
            << ",\"p95\":" << steady_stats.p95 << ",\"min\":"
            << steady_stats.minimum << ",\"max\":" << steady_stats.maximum
            << "},\"setup_stages\":{";
  for (std::size_t stage = 0; stage < setup_stages.size(); ++stage) {
    if (stage != 0)
      std::cout << ',';
    const auto &stats = setup_stage_stats[stage];
    std::cout << '\"' << setup_stages[stage].first << "\":{\"mean\":"
              << stats.mean << ",\"p50\":" << stats.p50
              << ",\"p95\":" << stats.p95 << ",\"min\":"
              << stats.minimum << ",\"max\":" << stats.maximum << '}';
  }
  std::cout << "},\"execution_stages\":{";
  for (std::size_t stage = 0; stage < execution_stages.size(); ++stage) {
    if (stage != 0)
      std::cout << ',';
    const auto &stats = execution_stage_stats[stage];
    std::cout << '\"' << execution_stages[stage].first
              << "\":{\"mean\":" << stats.mean << ",\"p50\":" << stats.p50
              << ",\"p95\":" << stats.p95 << ",\"min\":" << stats.minimum
              << ",\"max\":" << stats.maximum << '}';
  }
  std::cout << "},\"workspace\":{\"control_slot\":"
            << peak.control_slot_bytes << ",\"config_wire\":"
            << peak.config_wire_bytes << ",\"config_slot\":"
            << peak.config_slot_bytes << ",\"expert_slot\":"
            << peak.expert_slot_bytes << ",\"summary_slot\":"
            << peak.summary_slot_bytes << ",\"device\":"
            << peak.device_bytes << ",\"pinned\":"
            << peak.pinned_host_bytes
            << ",\"route_input_device\":"
            << peak.route_input_device_bytes
            << ",\"device_notify_arena\":"
            << peak.device_notify_arena_bytes
            << ",\"device_notify_pinned_staging\":"
            << peak.device_notify_pinned_staging_bytes
            << ",\"final_device_metadata_arena_view_capacity\":"
            << peak.final_device_metadata_arena_view_capacity_bytes
            << ",\"final_metadata_is_arena_subview\":true"
            << ",\"event_count\":" << peak.event_count
            << ",\"quantified_event_bytes\":"
            << peak.quantified_event_bytes
            << "},\"hot_path\":{\"per_route_copy\":false,"
               "\"per_route_allocation\":false,\"per_route_sync\":false}}\n";
}
