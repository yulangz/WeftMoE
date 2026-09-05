  if (scenario == "activation_p2p" || scenario == "activation_p2p_cancel" ||
      scenario == "activation_socket" || scenario == "activation_gin" ||
      benchmark_p2p) {
    const bool socket_fallback = scenario == "activation_socket";
    const bool gin_rdma = scenario == "activation_gin";
    require(socket_fallback || gin_rdma ? nodes == 2 : nodes == 1,
            socket_fallback || gin_rdma
                ? "cross-host activation scenario requires exactly two nodes"
                : "local CUDA P2P scenario requires exactly one node");
    auto session = megamoe::cuda::make_nccl_generation(
        base, "cuda-local-p2p", 701);
    auto context = megamoe::make_rank_collective_context(
        session.session(), rank, execution.token());
    auto device_routes = megamoe::cuda::upload_cuda_routes(
        common_routes, top_k, context, execution);
    const auto gate_mode_text = std::getenv("MEGAMOE_GATE_MODE");
    const std::string gate_mode = gate_mode_text ? gate_mode_text
                                                  : "fixed-random";
    require(gate_mode == "fixed-random" || gate_mode == "unit",
            "gate mode must be fixed-random or unit");
    auto route_gates = gate_mode == "unit"
        ? std::vector<float>(common_routes.selected_experts.size(), 1.0F)
        : fixed_random_selected_gates(common_routes.num_tokens, top_k, rank,
                                      0x4d4547414d4f4547ULL);
    auto device_gates =
        megamoe::cuda::testing::upload_fp32_gates_for_testing(
            route_gates, device_routes, context, execution);
    const auto gate_hash = hash_fp32(route_gates);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    auto notify = coordinator.run_device(
        {common_config, device_routes, device_gates, max_tokens});
    megamoe::DispatchHandle cross_host_oracle_dispatch;
    if (socket_fallback || gin_rdma)
      cross_host_oracle_dispatch =
          megamoe::cuda::download_device_metadata_for_validation(
              notify.dispatch(), execution);
    const std::uint64_t hidden = benchmark_p2p ? 7168U : 17U;
    megamoe::cuda::ActivationTransportPoolConfig transport_pool_config;
    transport_pool_config.max_tokens_per_rank = max_tokens;
    transport_pool_config.max_hidden_size = hidden;
    transport_pool_config.max_top_k = top_k;
    transport_pool_config.max_world_size = world;
    transport_pool_config.max_nodes = nodes;
    transport_pool_config.max_destination_rows_per_rank = benchmark_p2p
        ? max_tokens * top_k
        : static_cast<std::uint64_t>(world) * max_tokens * top_k;
    transport_pool_config.channels = 1;
    const auto no_wrap_test = std::getenv("MEGAMOE_P2P_NO_WRAP") != nullptr;
    const auto ring_slots_text = std::getenv("MEGAMOE_P2P_RING_SLOTS");
    transport_pool_config.ring_slots_per_source_channel =
        benchmark_p2p
            ? static_cast<std::uint32_t>(max_tokens * top_k)
            : (no_wrap_test
                   ? static_cast<std::uint32_t>(
                         transport_pool_config.max_destination_rows_per_rank)
                   : (ring_slots_text
                          ? static_cast<std::uint32_t>(
                                std::stoul(ring_slots_text))
                          : 2U));
    transport_pool_config.max_inflight_generations = 1;
    transport_pool_config.enable_socket_fallback = socket_fallback;
    transport_pool_config.enable_gin_rdma = gin_rdma;
    const auto setup_start = Clock::now();
    auto transport_pool = megamoe::cuda::make_activation_transport_pool(
        session.session(), execution, transport_pool_config);
    std::vector<megamoe::transport::BFloat16> activations(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    for (std::size_t index = 0; index < activations.size(); ++index) {
      const auto value = static_cast<float>(rank * 101U + index % 97U) / 32.0F;
      activations[index] = megamoe::transport::float_to_bfloat16(value);
    }
    auto activation =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activations, common_routes.num_tokens, hidden, context, execution,
            transport_pool);
    std::unique_ptr<megamoe::cuda::LocalCudaP2pActivationTransport>
        local_transport;
    std::unique_ptr<megamoe::cuda::NcclSocketFallbackActivationTransport>
        socket_transport;
    std::unique_ptr<megamoe::cuda::GinRdmaActivationTransport> gin_transport;
    if (socket_fallback)
      socket_transport = std::make_unique<
          megamoe::cuda::NcclSocketFallbackActivationTransport>(
          session, execution, transport_pool);
    else if (gin_rdma)
      gin_transport =
          std::make_unique<megamoe::cuda::GinRdmaActivationTransport>(
              session, execution, transport_pool);
    else
      local_transport = std::make_unique<
          megamoe::cuda::LocalCudaP2pActivationTransport>(
          session, execution, transport_pool);
    const auto setup_us = std::chrono::duration<double, std::micro>(
                              Clock::now() - setup_start)
                              .count();
    megamoe::transport::ActivationTransportConfig transport_config;
    transport_config.hidden_size = hidden;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel =
        transport_pool_config.ring_slots_per_source_channel;
    transport_config.expert_transform =
        megamoe::transport::ActivationExpertTransform::SyntheticValidation;
    transport_config.gate_placement =
        megamoe::transport::ActivationGatePlacement::SourceAfter;
    if (scenario == "activation_p2p_cancel") {
      megamoe::cuda::ActivationTransportRunOptions cancel_options;
      cancel_options.cancellation.request_cancel();
      bool cancelled = false;
      try {
        (void)local_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, cancel_options);
      } catch (const megamoe::NotifyDispatchError &error) {
        cancelled = error.code() == megamoe::NotifyErrorCode::TransportTimeout;
      }
      require(cancelled, "all-rank local P2P cancellation was not observed");
      activation = {};
      const auto usage = transport_pool.usage();
      require(usage.slots_quarantined == 1 && usage.slots_available == 0,
              "cancelled local P2P slot was not quarantined");
      std::cout << "P2P_CANCEL rank=" << rank
                << " cancelled=1 quarantined=" << usage.slots_quarantined
                << std::endl;
      return;
    }
    const auto rounds_text = std::getenv("MEGAMOE_P2P_ROUNDS");
    const auto benchmark_rounds_text =
        std::getenv("MEGAMOE_P2P_BENCH_ROUNDS");
    const auto rounds = benchmark_p2p
                            ? (benchmark_rounds_text
                                   ? static_cast<std::uint32_t>(
                                         std::stoul(benchmark_rounds_text))
                                   : 110U)
                            : (rounds_text ? static_cast<std::uint32_t>(
                                                 std::stoul(rounds_text))
                                           : (socket_fallback || gin_rdma
                                                  ? 20U
                                                  : 1U));
    require(rounds > 0, "local P2P rounds must be positive");
    require(!benchmark_p2p || rounds > 10,
            "local P2P benchmark requires 10 warmups plus measurements");
    megamoe::cuda::DeviceActivationTransportHandle transported;
    megamoe::cuda::ActivationTransportValidationDownload downloaded;
    std::vector<megamoe::transport::BFloat16> baseline(
        static_cast<std::size_t>(common_routes.num_tokens * hidden));
    std::vector<double> total_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> relay_samples;
    std::vector<double> synthetic_samples;
    std::vector<double> reduce_samples;
    const auto steady_api_before =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      megamoe::cuda::ActivationTransportRunOptions run_options;
      if (benchmark_p2p)
        run_options.deadline_after = std::chrono::seconds(120);
      if (socket_fallback)
        transported = socket_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      else if (gin_rdma)
        transported = gin_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      else
        transported = local_transport->run_transport_slice(
            notify.dispatch(), activation, transport_config, run_options);
      const auto warmup = benchmark_p2p ? 10U : 0U;
      if (benchmark_p2p && round >= warmup) {
        total_samples.push_back(transported.timings().total_us);
        dispatch_samples.push_back(transported.timings().dispatch_put_us);
        relay_samples.push_back(transported.timings().relay_us);
        synthetic_samples.push_back(
            transported.timings().synthetic_return_us);
        reduce_samples.push_back(transported.timings().combine_reduce_us);
      }
      if (benchmark_p2p) {
        if (round + 1U != rounds)
          transported = {};
        continue;
      }
      downloaded = megamoe::cuda::download_activation_transport_for_validation(
          transported, execution);
      if (round == 0)
        std::copy(downloaded.combined_tokens.begin(),
                  downloaded.combined_tokens.end(), baseline.begin());
      else
        require(downloaded.combined_tokens == baseline,
                "local P2P repeated generation changed output bytes");
      if (round + 1U != rounds) {
        downloaded = {};
        transported = {};
      }
    }
    require(transported.audit().backend ==
                (socket_fallback
                     ? megamoe::transport::ActivationTransportBackend::
                           NcclSocketFallback
                     : gin_rdma
                           ? megamoe::transport::ActivationTransportBackend::Gin
                     : megamoe::transport::ActivationTransportBackend::
                           LocalCudaP2p),
            "concrete activation backend audit mismatch");
    require(transported.audit().activation_d2h_bytes == 0 &&
                transported.audit().activation_h2d_bytes == 0 &&
                transported.audit().hot_path_allocations == 0,
            "real local P2P path staged activation or allocated hot resources");
    if (socket_fallback || gin_rdma) {
      const auto oracle = megamoe::transport::simulate_activation_transport(
          cross_host_oracle_dispatch, activations, route_gates,
          transport_config, 1);
      if (downloaded.contribution_by_route != oracle.contribution_by_route) {
        const auto mismatch = std::mismatch(
            downloaded.contribution_by_route.begin(),
            downloaded.contribution_by_route.end(),
            oracle.contribution_by_route.begin());
        const auto index = static_cast<std::size_t>(
            mismatch.first - downloaded.contribution_by_route.begin());
        throw std::runtime_error(
            "cross-host route contributions differ from host oracle: "
            "index=" + std::to_string(index) + ", actual=" +
            std::to_string(mismatch.first->bits) + ", expected=" +
            std::to_string(mismatch.second->bits));
      }
      require(downloaded.combined_tokens == oracle.combined_tokens,
              "cross-host combined output differs from host oracle");
    }
    const auto steady_api_after =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
    require(steady_api_after.device_allocations ==
                steady_api_before.device_allocations &&
                steady_api_after.device_frees == steady_api_before.device_frees &&
                steady_api_after.pinned_allocations ==
                    steady_api_before.pinned_allocations &&
                steady_api_after.pinned_frees == steady_api_before.pinned_frees &&
                steady_api_after.event_creates == steady_api_before.event_creates &&
                steady_api_after.event_destroys ==
                    steady_api_before.event_destroys &&
                steady_api_after.device_synchronizes ==
                    steady_api_before.device_synchronizes,
            "real local P2P steady path allocated or device-synchronized");
    if (benchmark_p2p) {
      const auto emit_values = [](const std::vector<double> &values) {
        for (std::size_t index = 0; index < values.size(); ++index) {
          if (index != 0)
            std::cout << ',';
          std::cout << values[index];
        }
      };
      const auto usage = transport_pool.usage();
      std::cout << "P2P_BENCH rank=" << rank
                << " backend=local-cuda-p2p setup_us=" << setup_us
                << " device_bytes=" << usage.device_bytes
                << " pinned_bytes=" << usage.pinned_bytes
                << " raw_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " actual_dispatch_peer_bytes="
                << transported.audit().actual_dispatch_peer_bytes
                << " actual_return_peer_bytes="
                << transported.audit().actual_return_peer_bytes
                << " stalls=" << transported.audit().credit_stalls
                << " peak_ring=" << transported.audit().peak_ring_slots
                << " total=";
      emit_values(total_samples);
      std::cout << " dispatch=";
      emit_values(dispatch_samples);
      std::cout << " relay=";
      emit_values(relay_samples);
      std::cout << " synthetic=";
      emit_values(synthetic_samples);
      std::cout << " reduce=";
      emit_values(reduce_samples);
      std::cout << std::endl;
      return;
    }
    if (socket_fallback) {
      std::cout << "SOCKET_RESULT rank=" << rank
                << " backend="
                << megamoe::transport::activation_transport_backend_name(
                       transported.audit().backend)
                << " tokens=" << common_routes.num_tokens
                << " top_k=" << top_k << " hidden=" << hidden
                << " packets=" << transported.audit().packet_count
                << " fanout=" << transported.audit().fanout_count
                << " raw_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " gate_mode=" << gate_mode
                << " gate_hash=" << gate_hash
                << " output_hash="
                << hash_bf16_indexed(downloaded.combined_tokens)
                << " contribution_hash="
                << hash_bf16_indexed(downloaded.contribution_by_route)
                << " rounds=" << rounds
                << " oracle=1 total_us=" << transported.timings().total_us
                << std::endl;
      return;
    }
    if (gin_rdma) {
      std::cout << "GIN_RESULT rank=" << rank
                << " backend="
                << megamoe::transport::activation_transport_backend_name(
                       transported.audit().backend)
                << " tokens=" << common_routes.num_tokens
                << " top_k=" << top_k << " hidden=" << hidden
                << " packets=" << transported.audit().packet_count
                << " fanout=" << transported.audit().fanout_count
                << " raw_remote_bytes="
                << transported.audit().raw_cross_node_payload_bytes
                << " dedup_remote_bytes="
                << transported.audit().deduplicated_cross_node_payload_bytes
                << " return_remote_bytes="
                << transported.audit().actual_return_peer_bytes
                << " gate_mode=" << gate_mode
                << " gate_hash=" << gate_hash
                << " output_hash="
                << hash_bf16_indexed(downloaded.combined_tokens)
                << " contribution_hash="
                << hash_bf16_indexed(downloaded.contribution_by_route)
                << " rounds=" << rounds
                << " oracle=1 total_us=" << transported.timings().total_us
                << std::endl;
      return;
    }
    const auto emit_bits = [](const auto &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index].bits;
      }
    };
    std::cout << "P2P_RESULT rank=" << rank
              << " backend="
              << megamoe::transport::activation_transport_backend_name(
                     transported.audit().backend)
              << " tokens=" << common_routes.num_tokens
              << " top_k=" << top_k << " hidden=" << hidden
              << " packets=" << transported.audit().packet_count
              << " fanout=" << transported.audit().fanout_count
              << " stalls=" << transported.audit().credit_stalls
              << " peak_ring=" << transported.audit().peak_ring_slots
              << " raw_bytes="
              << transported.audit().raw_cross_node_payload_bytes
              << " dedup_bytes="
              << transported.audit().deduplicated_cross_node_payload_bytes
              << " gate_mode=" << gate_mode
              << " gate_hash=" << gate_hash
              << " rounds=" << rounds
              << " total_us=" << transported.timings().total_us
              << " combined=";
    emit_bits(downloaded.combined_tokens);
    std::cout << " contribution=";
    emit_bits(downloaded.contribution_by_route);
    std::cout << " canonical=";
    emit_bits(downloaded.canonical_expert_rows);
    std::cout << std::endl;
    return;
  }
