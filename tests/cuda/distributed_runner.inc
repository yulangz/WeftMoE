void run_rank(std::uint32_t rank, std::uint32_t nodes,
              std::uint32_t gpus_per_node, std::uint32_t local_rank,
              std::uint64_t seed, int bootstrap_input_fd,
              int bootstrap_output_fd,
              const std::string &scenario) {
  const auto world = nodes * gpus_per_node;
  constexpr std::array<std::uint32_t, 12> kCrossEpExperts{
      1U, 68U, 135U, 202U, 269U, 336U,
      145U, 289U, 10U, 90U, 230U, 350U};
  const auto benchmark_p2p = scenario == "activation_p2p_bench";
  const auto benchmark_integrated =
      scenario == "activation_integrated_bench";
  const auto gin_forward_smoke = scenario == "gin_forward_smoke";
  const auto gin_forward_benchmark = scenario == "gin_forward_bench";
  const auto socket_forward_benchmark =
      scenario == "socket_forward_bench";
  const auto gin_backward_smoke = scenario == "gin_backward_smoke";
  const auto gin_backward_benchmark = scenario == "gin_backward_bench";
  const auto backward_validation =
      scenario == "backward_validate" || gin_backward_smoke;
  const auto backward_benchmark =
      scenario == "backward_bench" || gin_backward_benchmark;
  const auto forward_smoke = scenario == "forward_smoke" || gin_forward_smoke;
  const auto benchmark_forward =
      scenario == "forward_bench" || gin_forward_benchmark ||
      socket_forward_benchmark;
  const auto benchmark_shape = benchmark_p2p || benchmark_integrated ||
                               benchmark_forward || backward_benchmark;
  auto experts = std::max<std::uint32_t>(16, world * 2);
  auto top_k = std::min<std::uint32_t>(6, experts);
  if (const auto *p2p_top_k = std::getenv("MEGAMOE_P2P_TOP_K"))
    top_k = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(std::stoul(p2p_top_k)), experts);
  auto common_config = make_config(nodes, gpus_per_node, experts, top_k, 7);
  auto common_routes = seeded_inputs(world, experts, top_k, seed)[rank];
  std::uint64_t max_tokens = 10;
  if (benchmark_shape) {
    const auto cross_host_benchmark = gin_forward_benchmark ||
                                      socket_forward_benchmark ||
                                      gin_backward_benchmark;
    require((!cross_host_benchmark && nodes == 1 &&
             ((benchmark_p2p && world == 8) ||
              ((benchmark_forward || backward_benchmark) &&
               (world == 1 || world == 8)) ||
              (benchmark_integrated && (world == 2 || world == 8)))) ||
                (cross_host_benchmark && nodes == 2 &&
                 (world == 2 || world == 16)),
            "CUDA benchmark requires local 1/2/8 GPUs or GIN 2x1/2x8 GPUs");
    experts = 384;
    const auto benchmark_top_k =
        std::getenv("MEGAMOE_INTEGRATED_TOP_K")
            ? std::getenv("MEGAMOE_INTEGRATED_TOP_K")
            : std::getenv("MEGAMOE_P2P_TOP_K");
    top_k = benchmark_top_k
                ? static_cast<std::uint32_t>(std::stoul(benchmark_top_k))
                : 6U;
    require(top_k == 1 || top_k == 6,
            "integrated benchmark top-k must be 1 or 6");
    const auto bench_tokens_text = std::getenv("MEGAMOE_P2P_BENCH_TOKENS");
    max_tokens = bench_tokens_text ? std::stoull(bench_tokens_text) : 8192U;
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % experts));
    }
  }
  if (forward_smoke) {
    const auto cross_host_megakernel_smoke = gin_forward_smoke;
    require((!cross_host_megakernel_smoke && nodes == 1 &&
             (world == 1 || world == 2 || world == 8)) ||
                (cross_host_megakernel_smoke && nodes == 2 &&
                 (world == 2 || world == 16)),
            "forward smoke requires local 1/2/8 GPUs or cross-host "
            "2x1/2x8 GPUs");
    // Cross-topology fixture: all EP sizes retain the same 384 global experts,
    // source activation, selected IDs and gates. Only contiguous ownership
    // changes. The selected experts span the complete ownership range.
    experts = 384U;
    max_tokens = 2;
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    for (std::uint64_t token = 0; token < max_tokens; ++token)
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            kCrossEpExperts[token * 6U + slot]));
  }
  if (scenario == "activation_gin") {
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base_expert = static_cast<std::uint32_t>(
          (token * 17U + rank * 31U) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base_expert + slot * 5U) % experts));
    }
  }
  if (backward_validation) {
    require((!gin_backward_smoke && nodes == 1U &&
             (world == 1U || world == 2U || world == 8U)) ||
                (gin_backward_smoke && nodes == 2U && world == 16U),
            "Backward validation requires local 1/2/8 GPUs or GIN 2x8 "
            "GPUs");
    // Keep at least the small topology already exercised by the Forward
    // scheduler; one-expert fixtures do not provide its split-role terminal
    // accounting with a representative task domain.
    experts = std::max<std::uint32_t>(16U, world * 2U);
    top_k = 1U;
    const auto *tokens_text = std::getenv("MEGAMOE_BACKWARD_TOKENS");
    max_tokens = tokens_text ? std::stoull(tokens_text) : 8ULL;
    // Forward publishes 64-row Dispatch arrival tiles. Backward later groups
    // and pads those immutable rows into its own 128-row dependency blocks.
    common_config = make_config(nodes, gpus_per_node, experts, top_k, 64U);
    for (std::uint32_t expert = 0; expert < experts; ++expert)
      common_config.topology.expert_to_rank[expert] =
          expert / (experts / world);
    common_routes.num_tokens = max_tokens;
    common_routes.selected_experts.clear();
    common_routes.selected_experts.reserve(
        static_cast<std::size_t>(max_tokens));
    for (std::uint64_t token = 0; token < max_tokens; ++token)
      common_routes.selected_experts.push_back(
          static_cast<std::int32_t>((token * world + rank) % experts));
  }
  megamoe::cuda::NcclBootstrapId bootstrap;
  if (rank == 0) {
    require(rank == 0, "only rank 0 may generate bootstrap ID");
    require(bootstrap_output_fd >= 0 && bootstrap_input_fd < 0,
            "rank 0 requires only a bootstrap output descriptor");
    bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    megamoe::cuda::NcclBootstrapOutputChannel channel(bootstrap_output_fd);
    channel.write(bootstrap);
    require(channel.state() == megamoe::cuda::BootstrapChannelState::Consumed,
            "bootstrap output channel did not reach Consumed");
  } else {
    require(bootstrap_input_fd >= 0 && bootstrap_output_fd < 0,
            "nonzero rank requires only a bootstrap input descriptor");
    megamoe::cuda::NcclBootstrapInputChannel channel(bootstrap_input_fd);
    bootstrap = channel.read();
    require(channel.state() == megamoe::cuda::BootstrapChannelState::Consumed,
            "bootstrap input channel did not reach Consumed");
  }
  auto execution = megamoe::cuda::make_cuda_execution(
      "multinode-exec-" + std::to_string(rank), local_rank);
  auto base = megamoe::cuda::make_nccl_session(
      "cuda-multinode-base", world, rank, 700, local_rank, bootstrap);
  if (backward_validation || backward_benchmark) {
    run_cached_backward_rank(rank, nodes, world, experts, top_k, max_tokens,
                             common_config, common_routes, base, execution,
                             backward_validation);
    return;
  }
  if (scenario == "gin_probe") {
    const auto capability =
        megamoe::cuda::probe_activation_transport_capability(base, execution);
    std::cout << "GIN_DISTRIBUTED_PROBE rank=" << rank
              << " world=" << world
              << " headers=" << capability.gin_headers_available
              << " symbols=" << capability.gin_symbols_present
              << " runtime_initializable="
              << capability.gin_runtime_initializable
              << " backend_implemented="
              << capability.gin_backend_implemented
              << " usable=" << capability.gin_usable
              << " reason=" << capability.gin_unavailable_reason
              << std::endl;
    return;
  }
  if (scenario == "gin_put_probe") {
    const auto probe =
        megamoe::cuda::testing::run_distributed_gin_put_signal_probe(
            base, execution, 1024);
    std::cout << "GIN_PUT_SIGNAL rank=" << rank << " world=" << world
              << " rail_rank=" << probe.rail_rank
              << " rail_world=" << probe.rail_world
              << " local_rank=" << probe.local_rank
              << " bytes=" << probe.payload_bytes
              << " received_hash=" << probe.received_hash
              << " expected_hash=" << probe.expected_hash
              << " match=" << (probe.received_hash == probe.expected_hash)
              << std::endl;
    return;
  }
  if (scenario == "gin_link_bench") {
    const auto *mode_text = std::getenv("MEGAMOE_GIN_BENCH_MODE");
    const auto full = mode_text != nullptr && std::string(mode_text) == "full";
    const auto *shift_text = std::getenv("MEGAMOE_GIN_BENCH_SHIFT");
    const auto shift = shift_text == nullptr
                           ? 0U
                           : static_cast<std::uint32_t>(std::stoul(shift_text));
    const auto *warmup_text = std::getenv("MEGAMOE_GIN_BENCH_WARMUP");
    const auto warmup = warmup_text == nullptr
                            ? 4U
                            : static_cast<std::uint32_t>(std::stoul(warmup_text));
    const auto *measured_text = std::getenv("MEGAMOE_GIN_BENCH_MEASURED");
    const auto measured =
        measured_text == nullptr
            ? 20U
            : static_cast<std::uint32_t>(std::stoul(measured_text));
    const auto *batch_text = std::getenv("MEGAMOE_GIN_BENCH_BATCH");
    const auto batch = batch_text == nullptr
                           ? 32U
                           : static_cast<std::uint32_t>(std::stoul(batch_text));
    const std::vector<std::uint64_t> payloads{
        1ULL * 7168ULL * sizeof(megamoe::transport::BFloat16),
        2ULL * 7168ULL * sizeof(megamoe::transport::BFloat16),
        64ULL * 1024ULL,
        64ULL * 7168ULL * sizeof(megamoe::transport::BFloat16),
        128ULL * 7168ULL * sizeof(megamoe::transport::BFloat16),
        4ULL * 1024ULL * 1024ULL};
    const auto benchmark =
        megamoe::cuda::testing::run_distributed_gin_link_benchmark(
            base, execution,
            full ? megamoe::cuda::testing::GinConnectionMode::Full
                 : megamoe::cuda::testing::GinConnectionMode::Rail,
            shift, payloads, warmup, measured, batch);
    for (const auto &sample : benchmark.cases) {
      std::cout << "GIN_LINK_BENCH rank=" << rank
                << " mode=" << (full ? "full" : "rail")
                << " shift=" << benchmark.target_shift
                << " local_rank=" << benchmark.local_rank
                << " local_world=" << benchmark.local_world
                << " gin_type=" << benchmark.gin_type
                << " railed_gin_type=" << benchmark.railed_gin_type
                << " payload=" << sample.payload_bytes
                << " puts=" << sample.puts_per_sample
                << " durations_ns=";
      for (std::size_t index = 0; index < sample.duration_ns.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << sample.duration_ns[index];
      }
      std::cout << " received_hash=" << benchmark.received_hash
                << " expected_hash=" << benchmark.expected_hash
                << " match="
                << (benchmark.received_hash == benchmark.expected_hash)
                << std::endl;
    }
    return;
  }
  if (scenario == "gin_team_pool") {
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "cuda-gin-team-pool", 701);
    megamoe::cuda::ActivationTransportPoolConfig config;
    config.max_tokens_per_rank = 2;
    config.max_hidden_size = 17;
    config.max_top_k = 2;
    config.max_world_size = world;
    config.max_nodes = nodes;
    config.max_destination_rows_per_rank = world * 4ULL;
    config.channels = 2;
    config.ring_slots_per_source_channel = 2;
    config.max_inflight_generations = 2;
    config.enable_gin_rdma = true;
    auto activation_pool = megamoe::cuda::make_activation_transport_pool(
        generation.session(), execution, config);
    megamoe::cuda::GinRdmaActivationTransport transport(
        generation, execution, activation_pool);
    const auto usage = activation_pool.usage();
    require(usage.compute_device_bytes != 0 &&
                usage.gin_registered_bytes != 0 &&
                usage.device_bytes > usage.compute_device_bytes,
            "GIN pool did not separate compute and registered storage");
    std::cout << "GIN_TEAM_POOL rank=" << rank << " world=" << world
              << " nodes=" << nodes
              << " local_world=" << gpus_per_node
              << " local_rank=" << local_rank
              << " compute_bytes=" << usage.compute_device_bytes
              << " registered_bytes=" << usage.gin_registered_bytes
              << " slots=" << config.max_inflight_generations << std::endl;
    return;
  }
  auto pool = megamoe::cuda::make_device_notify_resource_pool(
      base, execution,
      {common_config, max_tokens, 4096,
       forward_smoke ? 3U : 2U});
  if (benchmark_forward) {
    constexpr std::uint64_t hidden = 7168;
    constexpr std::uint64_t intermediate = 3072;
    const bool retain_backward_state =
        std::getenv("MEGAMOE_FORWARD_BENCH_RETAIN_STATE") != nullptr;
    auto setup_generation = megamoe::cuda::make_nccl_generation(
        base, "cuda-forward-bench", 701);
    auto setup_context = megamoe::make_rank_collective_context(
        setup_generation.session(), rank, execution.token());
    auto route_storage = megamoe::cuda::upload_cuda_routes(
        common_routes, top_k, setup_context, execution);
    const auto route_gates = fixed_random_selected_gates(
        common_routes.num_tokens, top_k, rank, 0x4d4547414d4f4547ULL);
    auto gate_storage =
        megamoe::cuda::testing::upload_fp32_gates_for_testing(
            route_gates, route_storage, setup_context, execution);
    megamoe::cuda::ActivationTransportPoolConfig transport_config;
    transport_config.max_tokens_per_rank = max_tokens;
    transport_config.max_hidden_size = hidden;
    transport_config.max_top_k = top_k;
    transport_config.max_world_size = world;
    transport_config.max_nodes = nodes;
    // The deterministic fixture is close to uniform but integer ownership
    // boundaries can put slightly more than route_count/world rows on a rank.
    transport_config.max_destination_rows_per_rank =
        max_tokens * (top_k + 1U) * (world == 2U ? 2U : 1U);
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel = 1;
    transport_config.max_inflight_generations = 1U;
    transport_config.max_intermediate_size = intermediate;
    transport_config.max_owned_experts = experts / world;
    transport_config.enable_backward_state = retain_backward_state;
    transport_config.enable_gin_rdma = gin_forward_benchmark;
    transport_config.enable_socket_fallback = socket_forward_benchmark;
    const auto setup_start = Clock::now();
    auto transport_pool = megamoe::cuda::make_activation_transport_pool(
        setup_generation.session(), execution, transport_config);
    std::vector<megamoe::transport::BFloat16> activation_host(
        static_cast<std::size_t>(max_tokens * hidden));
    for (std::size_t index = 0; index < activation_host.size(); ++index)
      activation_host[index] = megamoe::transport::float_to_bfloat16(
          static_cast<float>(static_cast<int>((index + rank * 13U) % 127U) -
                             63) /
          64.0F);
    auto activation_storage =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            activation_host, max_tokens, hidden, setup_context, execution,
            transport_pool);
    const auto experts_per_rank = experts / world;
    const auto activation_hash = hash_bf16_indexed(activation_host);
    const auto route_hash = hash_i32(common_routes.selected_experts);
    const auto gate_hash = hash_fp32(route_gates);
    const auto weight_hash = forward_weight_sample_hash(
        rank * experts_per_rank, experts_per_rank, hidden, intermediate);
    auto weights =
        megamoe::cuda::testing::make_deterministic_forward_weights_for_testing(
            experts_per_rank, rank * experts_per_rank, hidden, intermediate,
            setup_generation.session(), execution);
    const auto actual_weight_hashes =
        megamoe::cuda::testing::hash_forward_weight_storage_for_validation(
            weights, execution);
    const auto all_distinct = [](const auto &hashes) {
      for (std::size_t left = 0; left < hashes.size(); ++left)
        for (std::size_t right = left + 1; right < hashes.size(); ++right)
          if (hashes[left].low == hashes[right].low &&
              hashes[left].high == hashes[right].high)
            return false;
      return true;
    };
    require(all_distinct(actual_weight_hashes.sampled_w1_experts),
            "actual W1 device storage is not expert-distinct");
    require(all_distinct(actual_weight_hashes.sampled_w2_experts),
            "actual W2 device storage is not expert-distinct");
    require(all_distinct(actual_weight_hashes.sampled_w2_output_rows),
            "actual W2 device output rows are not distinct");
    std::unique_ptr<megamoe::cuda::LocalCudaP2pForwardMegaKernel>
        local_forward;
    std::unique_ptr<megamoe::cuda::GinRdmaForwardMegaKernel> gin_forward;
    std::unique_ptr<megamoe::cuda::NcclSocketFallbackForwardMegaKernel>
        socket_forward;
    if (gin_forward_benchmark)
      gin_forward =
          std::make_unique<megamoe::cuda::GinRdmaForwardMegaKernel>(
              setup_generation, execution, transport_pool);
    else if (socket_forward_benchmark)
      socket_forward = std::make_unique<
          megamoe::cuda::NcclSocketFallbackForwardMegaKernel>(
          setup_generation, execution, transport_pool);
    else
      local_forward =
          std::make_unique<megamoe::cuda::LocalCudaP2pForwardMegaKernel>(
              setup_generation, execution, transport_pool);
    const auto run_forward = [&](const auto &dispatch,
                                 const auto &activations,
                                 const auto &forward_weights,
                                 const megamoe::cuda::ForwardMegaKernelConfig
                                     &forward_config,
                                 const megamoe::cuda::
                                     ActivationTransportRunOptions
                                         &run_options) {
      if (gin_forward)
        return gin_forward->run(dispatch, activations, forward_weights,
                                forward_config, run_options);
      if (socket_forward)
        return socket_forward->run(dispatch, activations, forward_weights,
                                   forward_config, run_options);
      return local_forward->run(dispatch, activations, forward_weights,
                                forward_config, run_options);
    };
    const auto one_time_setup_us =
        std::chrono::duration<double, std::micro>(Clock::now() - setup_start)
            .count();
    const auto rounds_text = std::getenv("MEGAMOE_FORWARD_BENCH_ROUNDS");
    const auto rounds = rounds_text
                            ? static_cast<std::uint32_t>(
                                  std::stoul(rounds_text))
                            : 110U;
    require(rounds > 10, "forward benchmark requires >10 rounds");
    const auto generation_base_text =
        std::getenv("MEGAMOE_FORWARD_BENCH_GENERATION_BASE");
    const auto generation_base =
        generation_base_text
            ? static_cast<std::uint64_t>(std::stoull(generation_base_text))
            : 1000ULL;
    require(generation_base <=
                std::numeric_limits<std::uint64_t>::max() - rounds,
            "forward benchmark generation range overflows");
    std::vector<double> wall_samples;
    std::vector<double> notify_samples;
    std::vector<double> forward_samples;
    std::vector<double> dispatch_samples;
    std::vector<double> fc1_samples;
    std::vector<double> fc2_samples;
    std::vector<double> return_samples;
    std::vector<double> reduce_samples;
    std::vector<double> fc1_fc2_overlap_samples;
    std::vector<double> fc2_return_overlap_samples;
    std::vector<double> return_reduce_overlap_samples;
    std::vector<double> terminal_status_samples;
    megamoe::cuda::ForwardMegaKernelAudit last_audit;
    megamoe::cuda::DeviceForwardHandle last_output;
    const auto forward_api_before =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
    for (std::uint32_t round = 0; round < rounds; ++round) {
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_PROGRESS rank=" << rank << " round=" << round
                  << std::endl;
      auto generation = megamoe::cuda::make_nccl_generation(
          base, "cuda-forward-bench", generation_base + round);
      auto context = megamoe::make_rank_collective_context(
          generation.session(), rank, execution.token());
      auto routes =
          megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              route_storage, context, execution);
      auto gates =
          megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              gate_storage, routes, context, execution);
      auto activations =
          megamoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activation_storage, context, execution);
      const auto wall_start = Clock::now();
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          generation, execution, pool);
      auto notify = coordinator.run_device(
          {common_config, routes, gates, max_tokens});
      const auto notify_end = Clock::now();
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=notify_done"
                  << std::endl;
      megamoe::cuda::ActivationTransportRunOptions run_options;
      auto progress_deadline_seconds = 5U;
      if (const auto *deadline =
              std::getenv("MEGAMOE_FORWARD_PROGRESS_DEADLINE_SECONDS"))
        progress_deadline_seconds =
            static_cast<std::uint32_t>(std::stoul(deadline));
      require(progress_deadline_seconds != 0U,
              "Forward progress deadline must be positive");
      run_options.deadline_after =
          std::getenv("MEGAMOE_FORWARD_PROGRESS")
              ? std::chrono::seconds(progress_deadline_seconds)
              : std::chrono::seconds(300);
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=forward_enter"
                  << std::endl;
      auto output = run_forward(
          notify.dispatch(), activations, weights,
          {hidden, intermediate, retain_backward_state}, run_options);
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_HOST_PROGRESS rank=" << rank
                  << " round=" << round << " phase=forward_done"
                  << std::endl;
      const auto wall_end = Clock::now();
      last_audit = output.audit();
      last_output = output;
      if (round >= 10U) {
        wall_samples.push_back(
            std::chrono::duration<double, std::micro>(wall_end - wall_start)
                .count());
        notify_samples.push_back(
            std::chrono::duration<double, std::micro>(notify_end - wall_start)
                .count());
        forward_samples.push_back(output.timings().total_us);
        dispatch_samples.push_back(output.timings().dispatch_us);
        fc1_samples.push_back(output.timings().fc1_swiglu_us);
        fc2_samples.push_back(output.timings().fc2_us);
        return_samples.push_back(output.timings().combine_return_us);
        reduce_samples.push_back(output.timings().reduce_us);
        fc1_fc2_overlap_samples.push_back(
            output.timings().fc1_fc2_overlap_us);
        fc2_return_overlap_samples.push_back(
            output.timings().fc2_return_overlap_us);
        return_reduce_overlap_samples.push_back(
            output.timings().return_reduce_overlap_us);
        terminal_status_samples.push_back(
            output.timings().terminal_status_agreement_us);
      }
    }
    const auto forward_api_after =
        megamoe::cuda::testing::cuda_api_audit_snapshot();
#if MEGAMOE_DEVICE_TIMELINE
    if (std::getenv("MEGAMOE_DEVICE_TIMELINE_PREFIX") != nullptr) {
      const std::string prefix =
          std::getenv("MEGAMOE_DEVICE_TIMELINE_PREFIX");
      write_device_timeline(
          megamoe::cuda::testing::download_forward_device_timeline(
              last_output),
          prefix);
    }
#endif
    require(forward_api_after.device_allocations ==
                    forward_api_before.device_allocations &&
                forward_api_after.device_frees ==
                    forward_api_before.device_frees &&
                forward_api_after.pinned_allocations ==
                    forward_api_before.pinned_allocations &&
                forward_api_after.pinned_frees ==
                    forward_api_before.pinned_frees &&
                forward_api_after.event_creates ==
                    forward_api_before.event_creates &&
                forward_api_after.event_destroys ==
                    forward_api_before.event_destroys &&
                forward_api_after.device_synchronizes ==
                    forward_api_before.device_synchronizes,
            "Forward steady path allocated resources or device-synchronized");
    const auto emit_values = [](const std::vector<double> &values) {
      for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
          std::cout << ',';
        std::cout << values[index];
      }
    };
    const auto downloaded_output =
        megamoe::cuda::download_forward_output_for_validation(last_output,
                                                               execution);
    std::uint64_t output_hash = 1469598103934665603ULL;
    std::vector<std::uint16_t> first_token_unique;
    for (const auto value : downloaded_output) {
      output_hash = (output_hash ^ static_cast<std::uint8_t>(value.bits)) *
                    1099511628211ULL;
      output_hash =
          (output_hash ^ static_cast<std::uint8_t>(value.bits >> 8U)) *
          1099511628211ULL;
    }
    for (std::uint64_t column = 0;
         column < hidden && first_token_unique.size() < 32U; ++column) {
      const auto bits = downloaded_output[static_cast<std::size_t>(column)].bits;
      if (std::find(first_token_unique.begin(), first_token_unique.end(), bits) ==
          first_token_unique.end())
        first_token_unique.push_back(bits);
    }
    require(first_token_unique.size() > 1U,
            "Forward output fixture collapsed to one repeated value");
    if (const auto *output_dir = std::getenv("MEGAMOE_FORWARD_OUTPUT_DIR")) {
      const auto path = std::string(output_dir) + "/megamoe-rank-" +
                        std::to_string(rank) + ".bf16";
      std::ofstream destination(path, std::ios::binary | std::ios::trunc);
      require(destination.good(), "failed to open forward output artifact");
      destination.write(
          reinterpret_cast<const char *>(downloaded_output.data()),
          static_cast<std::streamsize>(downloaded_output.size() *
                                       sizeof(downloaded_output.front())));
      require(destination.good(), "failed to write forward output artifact");
    }
    const auto usage = transport_pool.usage();
    const char *reduce_scheduler =
        std::getenv("MEGAMOE_FORWARD_DYNAMIC_REDUCE_SCAN") != nullptr
            ? "scan"
            : (std::getenv("MEGAMOE_FORWARD_LEGACY_REDUCE") != nullptr
                   ? "legacy"
                   : "queue");
    std::cout << "FORWARD_BENCH rank=" << rank
              << " world=" << world
              << " tokens=" << common_routes.num_tokens
              << " experts=" << experts << " top_k=" << top_k
              << " hidden=" << hidden << " intermediate=" << intermediate
              << " backend="
              << (gin_forward_benchmark
                      ? "gin-rdma"
                      : socket_forward_benchmark ? "nccl-socket-fallback"
                                                 : "local-cuda-p2p")
              << " specialization=sm90-wgmma-tma-7168x3072"
              << " retained_backward_state=" << retain_backward_state
              << " reduce_scheduler=" << reduce_scheduler
              << " setup_us=" << one_time_setup_us
              << " device_pool_bytes=" << usage.device_bytes
              << " compute_pool_bytes=" << usage.compute_device_bytes
              << " gin_registered_bytes=" << usage.gin_registered_bytes
              << " pinned_pool_bytes=" << usage.pinned_bytes
              << " pool_events=" << usage.event_count
              << " weight_bytes=" << weights.device_bytes()
              << " persistent_launches="
              << last_audit.persistent_kernel_launches
              << " wgmma=" << last_audit.wgmma_instructions
              << " tma=" << last_audit.tma_loads
              << " multicast=" << last_audit.tma_multicast_loads
              << " dispatch_peer_bytes="
              << last_audit.actual_dispatch_peer_bytes
              << " return_peer_bytes="
              << last_audit.actual_return_peer_bytes
              << " gin_dispatch_puts="
              << last_audit.gin_dispatch_put_count
              << " raw_remote_route_bytes="
              << last_audit.raw_cross_node_route_bytes
              << " gin_dispatch_payload_bytes="
              << last_audit.gin_dispatch_payload_bytes
              << " gin_return_payload_bytes="
              << last_audit.gin_return_payload_bytes
              << " gin_return_puts=" << last_audit.gin_return_put_count
              << " gin_signals=" << last_audit.gin_signal_count
              << " gin_metadata_bytes=" << last_audit.gin_metadata_bytes
              << " activation_hash=" << activation_hash
              << " route_hash=" << route_hash
              << " gate_hash=" << gate_hash
              << " weight_generator_version=3"
              << " weight_hash=" << weight_hash
              << " actual_w1_hash_low=" << actual_weight_hashes.w1.low
              << " actual_w1_hash_high=" << actual_weight_hashes.w1.high
              << " actual_w2_hash_low=" << actual_weight_hashes.w2.low
              << " actual_w2_hash_high=" << actual_weight_hashes.w2.high
              << " first_token_output_unique_at_least="
              << first_token_unique.size()
              << " cluster_supported="
              << static_cast<unsigned int>(
                     last_audit.runtime_cluster_supported)
              << " cluster_requested=" << last_audit.requested_cluster_size
              << " cluster_actual=" << last_audit.runtime_cluster_size
              << " cluster_rank_mask="
              << last_audit.observed_cluster_rank_mask
              << " max_active_clusters=" << last_audit.max_active_clusters
              << " steady_forbidden_cuda_api_calls=0"
              << " output_hash=" << output_hash << " wall=";
    emit_values(wall_samples);
    std::cout << " notify=";
    emit_values(notify_samples);
    std::cout << " forward=";
    emit_values(forward_samples);
    std::cout << " dispatch=";
    emit_values(dispatch_samples);
    std::cout << " fc1=";
    emit_values(fc1_samples);
    std::cout << " fc2=";
    emit_values(fc2_samples);
    std::cout << " return=";
    emit_values(return_samples);
    std::cout << " reduce=";
    emit_values(reduce_samples);
    std::cout << " fc1_fc2_overlap=";
    emit_values(fc1_fc2_overlap_samples);
    std::cout << " fc2_return_overlap=";
    emit_values(fc2_return_overlap_samples);
    std::cout << " return_reduce_overlap=";
    emit_values(return_reduce_overlap_samples);
    std::cout << " terminal_status_agreement=";
    emit_values(terminal_status_samples);
    std::cout << std::endl;
    return;
  }
  if (forward_smoke) {
    constexpr std::uint64_t hidden = 7168;
    constexpr std::uint64_t intermediate = 3072;
    auto generation = megamoe::cuda::make_nccl_generation(
        base, "cuda-forward-smoke", 701);
    auto context = megamoe::make_rank_collective_context(
        generation.session(), rank, execution.token());
    auto routes = megamoe::cuda::upload_cuda_routes(
        common_routes, top_k, context, execution);
    const auto gate_mode = std::getenv("MEGAMOE_GATE_MODE")
                               ? std::getenv("MEGAMOE_GATE_MODE")
                               : "fixed-random";
    const auto gates = std::string(gate_mode) == "unit"
                           ? std::vector<float>(max_tokens * top_k, 1.0F)
                           : fixed_random_selected_gates(
                                 max_tokens, top_k, rank,
                                 0x4d4547414d4f4547ULL);
    auto device_gates =
        megamoe::cuda::testing::upload_fp32_gates_for_testing(
            gates, routes, context, execution);
    megamoe::cuda::ActivationTransportPoolConfig transport_config;
    transport_config.max_tokens_per_rank = max_tokens;
    transport_config.max_hidden_size = hidden;
    transport_config.max_top_k = top_k;
    transport_config.max_world_size = world;
    transport_config.max_nodes = nodes;
    transport_config.max_destination_rows_per_rank =
        world * max_tokens * top_k;
    transport_config.channels = 1;
    transport_config.ring_slots_per_source_channel = 1;
    transport_config.max_inflight_generations = world == 1U ? 2U : 1U;
    transport_config.max_intermediate_size = intermediate;
    transport_config.enable_backward_state = true;
    transport_config.enable_gin_rdma = gin_forward_smoke;
    auto transport_pool = megamoe::cuda::make_activation_transport_pool(
        generation.session(), execution, transport_config);
    std::vector<megamoe::transport::BFloat16> input(max_tokens * hidden);
    for (std::uint64_t index = 0; index < max_tokens * hidden; ++index)
      input[static_cast<std::size_t>(index)] =
          megamoe::transport::float_to_bfloat16(
              static_cast<float>(static_cast<int>((index + rank * 13U) %
                                                   127U) -
                                 63) /
              64.0F);
    auto activations =
        megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            input, max_tokens, hidden, context, execution, transport_pool);
    auto weights =
        megamoe::cuda::testing::make_deterministic_forward_weights_for_testing(
            experts / world, rank * (experts / world), hidden, intermediate,
            generation.session(), execution);
    std::unique_ptr<megamoe::cuda::LocalCudaP2pForwardMegaKernel>
        local_forward;
    std::unique_ptr<megamoe::cuda::GinRdmaForwardMegaKernel> gin_forward;
    if (gin_forward_smoke)
      gin_forward =
          std::make_unique<megamoe::cuda::GinRdmaForwardMegaKernel>(
              generation, execution, transport_pool);
    else
      local_forward =
          std::make_unique<megamoe::cuda::LocalCudaP2pForwardMegaKernel>(
              generation, execution, transport_pool);
    const auto run_forward = [&](const auto &dispatch_handle,
                                 const auto &activation_handle,
                                 const auto &weight_handle,
                                 const megamoe::cuda::ForwardMegaKernelConfig
                                     &forward_config,
                                 const auto &run_options_value) {
      return gin_forward_smoke
                 ? gin_forward->run(dispatch_handle, activation_handle,
                                    weight_handle, forward_config,
                                    run_options_value)
                 : local_forward->run(dispatch_handle, activation_handle,
                                      weight_handle, forward_config,
                                      run_options_value);
    };
    megamoe::cuda::ActivationTransportRunOptions run_options;
    run_options.deadline_after = std::chrono::seconds(120);
    // Two physical transport slots and two live Forward handles must remain
    // isolated. Drop the producer leases after submission so the handles are
    // the final owners, then prove pool exhaustion and exact release.
    if (world == 1U) {
      auto second_input = input;
      for (auto &value : second_input)
        value = megamoe::transport::float_to_bfloat16(
            -megamoe::transport::bfloat16_to_float(value));
      auto second_generation = megamoe::cuda::make_nccl_generation(
          base, "cuda-forward-smoke", 702U);
      auto second_context = megamoe::make_rank_collective_context(
          second_generation.session(), rank, execution.token());
      auto second_activations =
          megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
              second_input, max_tokens, hidden, second_context, execution,
              transport_pool);
      const auto run_exact = [&](std::uint64_t exact_generation,
                                 const auto &source) {
        auto current_generation = megamoe::cuda::make_nccl_generation(
            base, "cuda-forward-smoke", exact_generation);
        auto current_context = megamoe::make_rank_collective_context(
            current_generation.session(), rank, execution.token());
        auto current_routes =
            megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
                routes, current_context, execution);
        auto current_gates =
            megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
                device_gates, current_routes, current_context, execution);
        auto current_activation =
            megamoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
                source, current_context, execution);
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            current_generation, execution, pool);
        auto notify = coordinator.run_device(
            {common_config, current_routes, current_gates, max_tokens});
        return run_forward(
            notify.dispatch(), current_activation, weights,
            {hidden, intermediate}, run_options);
      };
      auto live_first = run_exact(780U, activations);
      auto first_before =
          megamoe::cuda::download_forward_output_for_validation(live_first,
                                                                 execution);
      activations = {};
      auto live_second = run_exact(781U, second_activations);
      second_activations = {};
      const auto second_output =
          megamoe::cuda::download_forward_output_for_validation(live_second,
                                                                 execution);
      const auto first_after =
          megamoe::cuda::download_forward_output_for_validation(live_first,
                                                                 execution);
      require(first_before == first_after && first_before != second_output,
              "two live generations aliased output storage");
      auto live_usage = transport_pool.usage();
      require(live_usage.slots_in_use == 2 && live_usage.slots_available == 0,
              "two live Forward handles did not retain both slots");
      bool exhausted = false;
      try {
        (void)megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
            input, max_tokens, hidden, context, execution, transport_pool);
      } catch (const megamoe::NotifyDispatchError &error) {
        exhausted =
            error.code() == megamoe::NotifyErrorCode::ResourceExhausted;
      }
      require(exhausted, "pool=2 accepted a third live generation");
      live_first = {};
      live_second = {};
      live_usage = transport_pool.usage();
      require(live_usage.slots_available == 2 && live_usage.slots_in_use == 0,
              "last Forward handle did not release its slot");
      activations =
          megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
              input, max_tokens, hidden, context, execution, transport_pool);
    }
    std::vector<megamoe::transport::BFloat16> reference_output;
    megamoe::cuda::ForwardBackwardStateValidationDownload
        reference_backward_state;
    bool saw_backward_state = false;
    megamoe::cuda::ForwardMegaKernelAudit reference_audit;
    megamoe::cuda::ForwardMegaKernelTimings reference_timings;
    const auto forward_smoke_repeats =
        std::getenv("MEGAMOE_FORWARD_SMOKE_REPEATS")
            ? static_cast<std::uint32_t>(std::stoul(
                  std::getenv("MEGAMOE_FORWARD_SMOKE_REPEATS")))
            : 20U;
    require(forward_smoke_repeats >= 1U,
            "Forward smoke repeat count must be nonzero");
    for (std::uint32_t repeat = 0; repeat < forward_smoke_repeats; ++repeat) {
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS"))
        std::cerr << "FORWARD_SMOKE_PROGRESS rank=" << rank
                  << " repeat=" << repeat << std::endl;
      auto repeat_generation = megamoe::cuda::make_nccl_generation(
          base, "cuda-forward-smoke", 800U + repeat);
      auto repeat_context = megamoe::make_rank_collective_context(
          repeat_generation.session(), rank, execution.token());
      auto repeat_routes =
          megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
              routes, repeat_context, execution);
      auto repeat_gates =
          megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
              device_gates, repeat_routes, repeat_context, execution);
      auto repeat_activations =
          megamoe::cuda::ActivationTransportBridge::reissue_for_benchmark(
              activations, repeat_context, execution);
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          repeat_generation, execution, pool);
      auto notify = coordinator.run_device(
          {common_config, repeat_routes, repeat_gates, max_tokens});
      const bool retain_backward_state = (repeat & 1U) != 0U;
      auto retained_metadata =
          retain_backward_state && !saw_backward_state
              ? megamoe::cuda::download_device_metadata_for_validation(
                    notify.dispatch(), execution)
              : megamoe::DispatchHandle{};
      auto current = run_forward(notify.dispatch(), repeat_activations, weights,
                                 {hidden, intermediate,
                                  retain_backward_state},
                                 run_options);
      const auto current_output =
          megamoe::cuda::download_forward_output_for_validation(current,
                                                                 execution);
      if (retain_backward_state) {
        const auto state =
            megamoe::cuda::download_forward_backward_state_for_validation(
                current, execution);
        require(state.fc1_ab.size() % (2ULL * intermediate) == 0ULL &&
                    state.canonical_x.size() % hidden == 0ULL &&
                    state.fc1_ab.size() / (2ULL * intermediate) ==
                        state.canonical_x.size() / hidden,
                "retained Forward state has inconsistent compact row shapes");
        if (!state.fc1_ab.empty())
          require(std::any_of(state.fc1_ab.begin(), state.fc1_ab.end(),
                              [](const auto value) {
                                return value.bits != 0U;
                              }),
                  "retained BF16 FC1 A/B checkpoint is all zero");
        if (!saw_backward_state) {
          require(retained_metadata &&
                      retained_metadata->incoming.incoming_rows.size() ==
                          state.canonical_x.size() / hidden,
                  "retained canonical X metadata shape mismatch");
          for (const auto &row :
               retained_metadata->incoming.incoming_rows) {
            const auto route = std::find(kCrossEpExperts.begin(),
                                         kCrossEpExperts.end(),
                                         row.expert_id);
            require(route != kCrossEpExperts.end(),
                    "retained canonical X referenced an unknown smoke expert");
            const auto token =
                static_cast<std::uint64_t>(route - kCrossEpExperts.begin()) /
                6ULL;
            for (std::uint64_t column = 0; column < hidden; ++column) {
              const auto linear = token * hidden + column;
              const auto expected =
                  megamoe::transport::float_to_bfloat16(
                      static_cast<float>(static_cast<int>(
                          (linear + row.source_rank * 13U) % 127U) - 63) /
                      64.0F);
              require(state.canonical_x[static_cast<std::size_t>(
                          row.dst_row * hidden + column)] == expected,
                      "retained canonical X was overwritten by FC2 output");
            }
          }
          reference_backward_state = state;
          saw_backward_state = true;
        } else {
          require(state.canonical_x == reference_backward_state.canonical_x &&
                      state.fc1_ab == reference_backward_state.fc1_ab,
                  "retained Forward state changed across identical runs");
        }
      }
      if (std::getenv("MEGAMOE_FORWARD_PROGRESS")) {
        std::uint64_t repeat_hash = 1469598103934665603ULL;
        for (const auto value : current_output)
          repeat_hash = (repeat_hash ^ value.bits) * 1099511628211ULL;
        std::cerr << "FORWARD_SMOKE_HASH rank=" << rank
                  << " repeat=" << repeat << " hash=" << repeat_hash
                  << std::endl;
      }
      if (repeat == 0U) {
        reference_output = current_output;
        reference_audit = current.audit();
        reference_timings = current.timings();
      } else {
        if (current_output != reference_output &&
            std::getenv("MEGAMOE_FORWARD_PROGRESS")) {
          const auto mismatch = std::mismatch(reference_output.begin(),
                                              reference_output.end(),
                                              current_output.begin());
          const auto index = static_cast<std::size_t>(
              mismatch.first - reference_output.begin());
          std::cerr << "FORWARD_SMOKE_MISMATCH rank=" << rank
                    << " repeat=" << repeat << " index=" << index
                    << " expected_bits=" << mismatch.first->bits
                    << " actual_bits=" << mismatch.second->bits << std::endl;
        }
        require(current_output == reference_output,
                "Forward output changed across identical repetitions");
      }
    }
    require(reference_audit.persistent_kernel_launches == 1 &&
                ((reference_audit.wgmma_instructions != 0 &&
                  reference_audit.tma_loads != 0) ||
                 reference_audit.fc1_tasks == 0),
            "forward smoke did not execute one WGMMA/TMA persistent kernel");
    require(saw_backward_state,
            "forward smoke did not exercise retained Backward state");
    std::uint64_t output_hash = 1469598103934665603ULL;
    for (const auto value : reference_output)
      output_hash = (output_hash ^ value.bits) * 1099511628211ULL;
    if (const auto *output_dir = std::getenv("MEGAMOE_FORWARD_OUTPUT_DIR")) {
      const auto prefix = std::string(output_dir) + "/megamoe-rank-" +
                          std::to_string(rank);
      std::ofstream output_file(prefix + ".bf16",
                                std::ios::binary | std::ios::trunc);
      require(output_file.good(), "failed to open Forward smoke output");
      output_file.write(reinterpret_cast<const char *>(reference_output.data()),
                        static_cast<std::streamsize>(
                            reference_output.size() * sizeof(reference_output[0])));
      std::ofstream cache_file(prefix + ".ab.bf16",
                               std::ios::binary | std::ios::trunc);
      require(cache_file.good(),
              "failed to open Forward A/B checkpoint artifact");
      cache_file.write(
          reinterpret_cast<const char *>(
              reference_backward_state.fc1_ab.data()),
          static_cast<std::streamsize>(
              reference_backward_state.fc1_ab.size() *
              sizeof(reference_backward_state.fc1_ab[0])));
    }
    std::cout << "FORWARD_SMOKE rank=" << rank
              << " specialization=sm90-wgmma-tma-7168x3072"
              << " persistent_launches="
              << reference_audit.persistent_kernel_launches
              << " wgmma=" << reference_audit.wgmma_instructions
              << " tma=" << reference_audit.tma_loads
              << " fc1_tasks=" << reference_audit.fc1_tasks
              << " fc2_tasks=" << reference_audit.fc2_tasks
              << " repeat_equal=1 repeats="
              << forward_smoke_repeats << " gate_mode="
              << gate_mode << " top_k=" << top_k
              << " output_hash=" << output_hash
              << " pool2_isolated=1"
              << " backward_state_equal=1"
              << " total_us=" << reference_timings.total_us
              << " dispatch_us=" << reference_timings.dispatch_us
              << " fc1_us=" << reference_timings.fc1_swiglu_us
              << " fc2_us=" << reference_timings.fc2_us
              << " return_us=" << reference_timings.combine_return_us
              << " reduce_us=" << reference_timings.reduce_us << std::endl;
    return;
  }
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
  if (scenario == "valid") {
    auto session = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode", 701);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution,
                                                                pool);
    auto result = coordinator.run({common_config, common_routes, 10});
    std::cout << megamoe::serialize_dispatch_metadata(
                     megamoe::cuda::download_device_metadata_for_validation(
                         result.dispatch(), execution))
              << '\n';
    return;
  }

  for (std::uint64_t round = 0; round < 20; ++round) {
    auto config = common_config;
    if (scenario == "expert_capacity_mismatch")
      config.capacity.expert_rows.assign(experts, 100);
    if (scenario == "workspace_capacity_mismatch")
      config.capacity.workspace_rows_per_rank.assign(world, 1000);
    auto routes = common_routes;
    std::uint64_t max_tokens = 10;
    std::string identity = "cuda-multinode-invalid-" + scenario;
    std::uint64_t generation = 800 + round * 2;
    mutate_scenario(scenario, rank, config, routes, max_tokens, identity,
                    generation);
    auto session = megamoe::cuda::make_nccl_generation(
        base, identity, generation);
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(session,
                                                                execution,
                                                                pool);
    try {
      (void)coordinator.run({config, routes, max_tokens});
      throw std::runtime_error("invalid cross-host generation passed");
    } catch (const megamoe::NotifyDispatchError &error) {
      const auto audit = coordinator.audit();
      require(audit.control_starts == 1 &&
                  audit.expert_count_starts == 0 &&
                  audit.packet_summary_starts == 0,
              "failed cross-host generation entered data collective");
      std::cout << "CONTROL_ERROR round=" << round << " " << error.what()
                << std::endl;
    }

    auto recovery_session = megamoe::cuda::make_nccl_generation(
        base, "cuda-multinode-recovery-" + scenario, 801 + round * 2);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(recovery_session,
                                                             execution,
                                                             pool);
    std::vector<std::int32_t> recovery_experts;
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      recovery_experts.push_back(static_cast<std::int32_t>(slot));
    auto recovered = recovery.run({common_config,
                                   {1, std::move(recovery_experts)}, 10});
    require(recovered.audit().control_starts == 1 &&
                recovered.audit().config_starts == 1 &&
                recovered.audit().expert_count_starts == 1 &&
                recovered.audit().packet_summary_starts == 1,
            "cross-host recovery failed");
    require(recovered && !megamoe::serialize_dispatch_metadata(
                              megamoe::cuda::download_device_metadata_for_validation(
                                  recovered.dispatch(), execution)).empty(),
            "cross-host recovery metadata missing");
    std::cout << "RECOVERY_OK rank=" << rank << " round=" << round
              << std::endl;
  }
  throw megamoe::NotifyDispatchError(
      megamoe::NotifyErrorCode::ControlProtocolFailure,
      "expected invalid cross-host generation");
}
