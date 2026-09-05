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
        static_cast<std::size_t>(max_tokens * top_k));
    for (std::uint64_t token = 0; token < max_tokens; ++token) {
      const auto base = static_cast<std::uint32_t>(
          (token * world + rank) % experts);
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        common_routes.selected_experts.push_back(static_cast<std::int32_t>(
            (base + slot * 53U) % experts));
    }
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
