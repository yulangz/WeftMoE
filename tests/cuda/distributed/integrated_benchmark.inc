  if (benchmark_integrated) {
    require(nodes == 1 && (world == 2 || world == 8),
            "integrated benchmark requires one 2/8-GPU node");
    constexpr std::uint64_t hidden = 7168;
    const auto setup_start = Clock::now();
    auto setup_generation = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode-base", 701);
    auto setup_context = megamoe::make_rank_collective_context(
        setup_generation.session(), rank, execution.token());
    auto route_storage = megamoe::cuda::upload_cuda_routes(
        common_routes, top_k, setup_context, execution);
    const auto gate_mode_text = std::getenv("MEGAMOE_GATE_MODE");
    const std::string gate_mode = gate_mode_text ? gate_mode_text
                                                  : "fixed-random";
    require(gate_mode == "fixed-random" || gate_mode == "unit",
            "gate mode must be fixed-random or unit");
    auto route_gates = gate_mode == "unit"
        ? std::vector<float>(common_routes.selected_experts.size(), 1.0F)
        : fixed_random_selected_gates(common_routes.num_tokens, top_k, rank,
                                      0x4d4547414d4f4547ULL);
    if (gate_mode == "fixed-random")
      for (std::uint64_t token = 0; token < common_routes.num_tokens; ++token) {
        float sum = 0.0F;
        for (std::uint32_t slot = 0; slot < top_k; ++slot) {
          const auto gate = route_gates[
              static_cast<std::size_t>(token * top_k + slot)];
          require(gate > 0.0F && gate <= 1.0F,
                  "fixed-random gate is outside (0,1]");
          sum += gate;
        }
        require(std::abs(sum - 1.0F) <= 2.0e-6F,
                "fixed-random selected gates do not sum to one");
      }
    auto gate_storage = megamoe::cuda::testing::upload_fp32_gates_for_testing(
        route_gates, route_storage, setup_context, execution);
    const auto gate_hash = hash_fp32(route_gates);
    megamoe::cuda::ActivationTransportPoolConfig transport_pool_config;
    transport_pool_config.max_tokens_per_rank = max_tokens;
    transport_pool_config.max_hidden_size = hidden;
    transport_pool_config.max_top_k = top_k;
    transport_pool_config.max_world_size = world;
    transport_pool_config.max_nodes = 1;
    std::vector<std::uint64_t> destination_rows(world, 0);
    for (std::uint32_t source = 0; source < world; ++source)
      for (std::uint64_t token = 0; token < max_tokens; ++token) {
        const auto base_expert = static_cast<std::uint32_t>(
            (token * 17U + source * 31U) % experts);
        for (std::uint32_t route_slot = 0; route_slot < top_k; ++route_slot) {
          const auto expert =
              (base_expert + route_slot * 53U) % experts;
          ++destination_rows[expert / (experts / world)];
        }
      }
    transport_pool_config.max_destination_rows_per_rank =
        *std::max_element(destination_rows.begin(), destination_rows.end());
    transport_pool_config.channels = 1;
    transport_pool_config.ring_slots_per_source_channel =
        static_cast<std::uint32_t>(max_tokens * top_k);
    transport_pool_config.max_inflight_generations = 1;
    auto transport_pool = megamoe::cuda::make_activation_transport_pool(
        setup_generation.session(), execution, transport_pool_config);
    std::vector<megamoe::transport::BFloat16> activations(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    for (std::size_t index = 0; index < activations.size(); ++index) {
      const auto value = static_cast<float>(rank * 101U + index % 97U) / 32.0F;
      activations[index] = megamoe::transport::float_to_bfloat16(value);
    }
    auto activation_storage =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, common_routes.num_tokens, hidden, setup_context,
            execution, transport_pool);
    megamoe::cuda::LocalCudaP2pActivationTransport transport(
        setup_generation, execution, transport_pool);
    const auto one_time_setup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - setup_start)
            .count();

    megamoe::transport::ActivationTransportConfig transport_config;
    transport_config.hidden_size = hidden;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel =
        transport_pool_config.ring_slots_per_source_channel;
    transport_config.expert_transform =
        megamoe::transport::ActivationExpertTransform::Identity;
    transport_config.gate_placement =
        megamoe::transport::ActivationGatePlacement::SourceAfter;
    const auto rounds_text =
        std::getenv("MEGAMOE_INTEGRATED_BENCH_ROUNDS");
    const auto rounds = rounds_text
                            ? static_cast<std::uint32_t>(
                                  std::stoul(rounds_text))
                            : 110U;
    require(rounds > 10,
            "integrated benchmark requires 10 warmups plus measurements");
    std::vector<double> wall_samples;
    std::vector<double> notify_samples;
    std::vector<double> mapping_samples;
    std::vector<double> control_samples;
    std::vector<double> count_summary_samples;
    std::vector<double> materialize_samples;
    std::vector<double> transport_samples;
    std::vector<double> transport_setup_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> relay_return_samples;
    std::vector<double> reduce_samples;
    std::vector<double> terminal_samples;
    megamoe::cuda::ActivationTransportDeviceAudit last_audit;
    megamoe::cuda::CollectiveProtocolAudit last_notify_audit;
    const auto api_before =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      const auto generation = 10000ULL + round;
      auto generation_session = megamoe::cuda::make_nccl_generation(
          base, "cuda-multinode-base", generation);
      auto context = megamoe::make_rank_collective_context(
          generation_session.session(), rank, execution.token());
      auto exact_routes =
          megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              route_storage, context, execution);
      auto exact_gates =
          megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              gate_storage, exact_routes, context, execution);
      auto exact_activation =
          megamoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activation_storage, context, execution);
      megamoe::cuda::GenerationExecutionTimings notify_timing;
      megamoe::cuda::ActivationTransportTimings transport_timing;
      const auto wall_start = Clock::now();
      {
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            generation_session, execution, pool);
        auto notify = coordinator.run_device(
            {common_config, exact_routes, exact_gates, max_tokens});
        notify_timing = notify.execution_timings();
        last_notify_audit = notify.audit();
        megamoe::cuda::ActivationTransportRunOptions options;
        options.deadline_after = std::chrono::seconds(120);
        auto transported = transport.run_transport_slice(
            notify.dispatch(), exact_activation, transport_config, options);
        transport_timing = transported.timings();
        last_audit = transported.audit();
        require(last_audit.backend ==
                    megamoe::transport::ActivationTransportBackend::LocalCudaP2p,
                "integrated benchmark did not run LocalCudaP2p");
        require(last_audit.activation_d2h_bytes == 0 &&
                    last_audit.activation_h2d_bytes == 0 &&
                    last_audit.hot_path_allocations == 0,
                "integrated benchmark staged or allocated activation data");
        transported = {};
        notify = {};
      }
      const auto wall_us =
          std::chrono::duration<double, std::micro>(Clock::now() - wall_start)
              .count();
      if (round >= 10U) {
        wall_samples.push_back(wall_us);
        notify_samples.push_back(notify_timing.coordinator_run_wall_us);
        mapping_samples.push_back(notify_timing.local_mapping_us);
        control_samples.push_back(notify_timing.control_config_us);
        count_summary_samples.push_back(notify_timing.expert_count_us +
                                        notify_timing.packet_summary_us);
        materialize_samples.push_back(
            notify_timing.device_materialization_us);
        transport_samples.push_back(transport_timing.total_us);
        transport_setup_samples.push_back(
            transport_timing.generation_setup_us);
        dispatch_samples.push_back(transport_timing.dispatch_put_us);
        relay_return_samples.push_back(
            transport_timing.synthetic_return_us);
        reduce_samples.push_back(transport_timing.combine_reduce_us);
        terminal_samples.push_back(transport_timing.terminal_completion_us);
      }
    }
    const auto api_after =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
    require(api_after.device_allocations == api_before.device_allocations &&
                api_after.device_frees == api_before.device_frees &&
                api_after.pinned_allocations == api_before.pinned_allocations &&
                api_after.pinned_frees == api_before.pinned_frees &&
                api_after.event_creates == api_before.event_creates &&
                api_after.event_destroys == api_before.event_destroys &&
                api_after.device_synchronizes == api_before.device_synchronizes,
            "integrated steady generations performed dynamic CUDA setup");

    // One explicit validation generation is outside every timed sample.
    auto validation_session = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode-base", 50000);
    auto validation_context = megamoe::make_rank_collective_context(
        validation_session.session(), rank, execution.token());
    auto validation_routes =
        megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
            route_storage, validation_context, execution);
    auto validation_gates =
        megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
            gate_storage, validation_routes, validation_context, execution);
    auto validation_activation =
        megamoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
            activation_storage, validation_context, execution);
    megamoe::cuda::CudaNotifyGenerationCoordinator validation_coordinator(
        validation_session, execution, pool);
    auto validation_notify = validation_coordinator.run_device(
        {common_config, validation_routes, validation_gates, max_tokens});
    auto validation_transport = transport.run_transport_slice(
        validation_notify.dispatch(), validation_activation, transport_config,
        {});
    const auto output =
        megamoe::cuda::download_activation_output_for_validation(
            validation_transport, execution);
    require(output.size() == activations.size(),
            "integrated output shape mismatch");
    std::vector<megamoe::transport::BFloat16> expected_by_token_value(
        static_cast<std::size_t>(common_routes.num_tokens * 97U));
    for (std::uint64_t token = 0; token < common_routes.num_tokens; ++token)
      for (std::uint32_t value_index = 0; value_index < 97U; ++value_index) {
        const auto input = megamoe::transport::float_to_bfloat16(
            static_cast<float>(rank * 101U + value_index) / 32.0F);
        float expected = 0.0F;
        std::vector<std::uint32_t> order(top_k);
        for (std::uint32_t slot = 0; slot < top_k; ++slot)
          order[slot] = slot;
        std::sort(order.begin(), order.end(), [&](std::uint32_t lhs,
                                                  std::uint32_t rhs) {
          return common_routes.selected_experts[
                     static_cast<std::size_t>(token * top_k + lhs)] <
                 common_routes.selected_experts[
                     static_cast<std::size_t>(token * top_k + rhs)];
        });
        for (const auto slot : order) {
          const auto route = static_cast<std::size_t>(token * top_k + slot);
          volatile float product =
              route_gates[route] *
              megamoe::transport::bfloat16_to_float(input);
          expected += product;
        }
        expected_by_token_value[
            static_cast<std::size_t>(token * 97U + value_index)] =
            megamoe::transport::float_to_bfloat16(expected);
      }
    std::uint64_t output_hash = 1469598103934665603ULL;
    bool output_equal = true;
    for (std::size_t index = 0; index < output.size(); ++index) {
      const auto token = index / static_cast<std::size_t>(hidden);
      output_equal = output_equal &&
                     output[index] == expected_by_token_value[
                         token * 97U + index % 97U];
      output_hash = (output_hash ^ output[index].bits) * 1099511628211ULL;
    }
    require(output_equal,
            "integrated identity/gated output differs from common reference");

    const auto emit_values = [](const std::vector<double> &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index];
      }
    };
    const auto usage = transport_pool.usage();
    std::cout << "INTEGRATED_BENCH rank=" << rank
              << " backend=local-cuda-p2p"
              << " setup_us=" << one_time_setup_us
              << " device_bytes=" << usage.device_bytes
              << " pinned_bytes=" << usage.pinned_bytes
              << " route_input_bytes=" << route_storage.device_bytes()
              << " local_gate_bytes=" << gate_storage.device_bytes()
              << " peer_gate_bytes=0"
              << " gate_mode=" << gate_mode
              << " gate_hash=" << gate_hash
              << " output_equal=1 output_hash=" << output_hash
              << " raw_bytes=" << last_audit.raw_cross_node_payload_bytes
              << " dedup_bytes="
              << last_audit.deduplicated_cross_node_payload_bytes
              << " actual_dispatch_peer_bytes="
              << last_audit.actual_dispatch_peer_bytes
              << " actual_return_peer_bytes="
              << last_audit.actual_return_peer_bytes
              << " derived_route_bytes="
              << last_audit.derived_route_input_bytes
              << " derived_local_gate_bytes="
              << last_audit.derived_local_gate_bytes
              << " derived_peer_gate_bytes="
              << last_audit.derived_peer_gate_bytes
              << " generation_memset_calls="
              << last_audit.generation_memset_calls
              << " generation_memset_bytes="
              << last_audit.generation_memset_bytes
              << " kernel_launches=" << last_audit.kernel_launches
              << " control_enqueues=" << last_notify_audit.control_starts
              << " config_enqueues=" << last_notify_audit.config_starts
              << " count_enqueues="
              << last_notify_audit.expert_count_starts
              << " summary_enqueues="
              << last_notify_audit.packet_summary_starts
              << " dispatch_kernel_blocks="
              << last_audit.dispatch_kernel_blocks
              << " return_kernel_blocks="
              << last_audit.return_kernel_blocks
              << " reduce_kernel_blocks="
              << last_audit.reduce_kernel_blocks
              << " steady_allocs="
              << api_after.device_allocations - api_before.device_allocations
              << " steady_h2d_bytes="
              << api_after.h2d_bytes - api_before.h2d_bytes
              << " steady_d2h_bytes="
              << api_after.d2h_bytes - api_before.d2h_bytes
              << " steady_d2d_bytes="
              << api_after.d2d_bytes - api_before.d2d_bytes
              << " wall=";
    emit_values(wall_samples);
    std::cout << " notify=";
    emit_values(notify_samples);
    std::cout << " mapping=";
    emit_values(mapping_samples);
    std::cout << " control=";
    emit_values(control_samples);
    std::cout << " count_summary=";
    emit_values(count_summary_samples);
    std::cout << " materialize=";
    emit_values(materialize_samples);
    std::cout << " transport=";
    emit_values(transport_samples);
    std::cout << " transport_setup=";
    emit_values(transport_setup_samples);
    std::cout << " dispatch=";
    emit_values(dispatch_samples);
    std::cout << " relay_return=";
    emit_values(relay_return_samples);
    std::cout << " reduce=";
    emit_values(reduce_samples);
    std::cout << " terminal=";
    emit_values(terminal_samples);
    std::cout << std::endl;
    return;
  }
