void run_cached_backward_rank(
    std::uint32_t rank, std::uint32_t nodes, std::uint32_t world,
    std::uint32_t experts, std::uint32_t top_k, std::uint64_t max_tokens,
    const megamoe::RuntimeConfig &config,
    const megamoe::RankRoutingInput &routes,
    const megamoe::cuda::NcclSession &base,
    const megamoe::cuda::CudaExecution &execution, bool validation) {
  constexpr std::uint64_t hidden = 7168;
  constexpr std::uint64_t intermediate = 3072;
  require(nodes == 1U && (world == 1U || world == 2U || world == 8U),
          "cached Backward currently requires one local node");
  require(experts % world == 0U && routes.num_tokens == max_tokens,
          "cached Backward fixture shape is inconsistent");

  auto generation = megamoe::cuda::make_nccl_generation(
      base, "cuda-cached-backward", 701U);
  auto context = megamoe::make_rank_collective_context(
      generation.session(), rank, execution.token());
  auto notify_pool = megamoe::cuda::make_device_notify_resource_pool(
      base, execution, {config, max_tokens, 4096, 3U});
  auto route_storage = megamoe::cuda::upload_cuda_routes(
      routes, top_k, context, execution);
  const auto route_gates = fixed_random_selected_gates(
      max_tokens, top_k, rank, 0x4d4547414d4f4547ULL);
  auto gate_storage = megamoe::cuda::testing::upload_fp32_gates_for_testing(
      route_gates, route_storage, context, execution);

  megamoe::cuda::ActivationTransportPoolConfig pool_config;
  pool_config.max_tokens_per_rank = max_tokens;
  pool_config.max_hidden_size = hidden;
  pool_config.max_top_k = top_k;
  pool_config.max_world_size = world;
  pool_config.max_nodes = 1U;
  pool_config.max_destination_rows_per_rank =
      max_tokens * (top_k + 1ULL) * (world == 2U ? 2ULL : 1ULL);
  pool_config.channels = 1U;
  pool_config.ring_slots_per_source_channel = 1U;
  pool_config.max_inflight_generations = 1U;
  pool_config.max_intermediate_size = intermediate;
  pool_config.max_owned_experts = experts / world;
  pool_config.enable_backward_state = true;
  auto activation_pool = megamoe::cuda::make_activation_transport_pool(
      generation.session(), execution, pool_config);

  std::vector<megamoe::transport::BFloat16> activation_host(
      static_cast<std::size_t>(max_tokens * hidden));
  std::vector<megamoe::transport::BFloat16> dy_host(activation_host.size());
  for (std::size_t index = 0; index < activation_host.size(); ++index) {
    activation_host[index] = megamoe::transport::float_to_bfloat16(
        static_cast<float>(static_cast<int>((index + rank * 13U) % 127U) -
                           63) /
        64.0F);
    dy_host[index] = megamoe::transport::float_to_bfloat16(
        static_cast<float>(static_cast<int>((index * 29U + rank * 17U) %
                                            113U) -
                           56) /
        64.0F);
  }
  auto activations =
      megamoe::cuda::testing::upload_bfloat16_activations_for_testing(
          activation_host, max_tokens, hidden, context, execution,
          activation_pool);
  auto output_gradients =
      megamoe::cuda::BackwardValidationBridge::upload_for_testing(
          dy_host, max_tokens, hidden, context, execution);
  auto weights =
      megamoe::cuda::testing::make_deterministic_forward_weights_for_testing(
          experts / world, rank * (experts / world), hidden, intermediate,
          generation.session(), execution);
  megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
      generation, execution, notify_pool);
  auto notify = coordinator.run_device(
      {config, route_storage, gate_storage, max_tokens});
  const auto progress = [&](const char *phase) {
    if (std::getenv("MEGAMOE_BACKWARD_PROGRESS") != nullptr)
      std::cerr << "BACKWARD_PROGRESS rank=" << rank << " phase=" << phase
                << std::endl;
  };
  progress("notify_done");
  megamoe::cuda::LocalCudaP2pForwardMegaKernel forward_kernel(
      generation, execution, activation_pool);
  megamoe::cuda::ActivationTransportRunOptions forward_options;
  forward_options.deadline_after = std::chrono::seconds(300);
  auto forward = forward_kernel.run(
      notify.dispatch(), activations, weights,
      {hidden, intermediate, true}, forward_options);
  progress("forward_done");

  const auto rounds_text = std::getenv("MEGAMOE_BACKWARD_ROUNDS");
  const auto rounds = rounds_text
                          ? static_cast<std::uint32_t>(std::stoul(rounds_text))
                          : (validation ? 2U : 30U);
  const auto warmups_text = std::getenv("MEGAMOE_BACKWARD_WARMUPS");
  const auto warmups = warmups_text
                           ? static_cast<std::uint32_t>(
                                 std::stoul(warmups_text))
                           : (validation ? 0U : 10U);
  require(rounds > warmups, "Backward requires at least one measured round");
  std::vector<double> wall_samples;
  std::vector<double> total_samples;
  std::vector<double> dispatch_samples;
  std::vector<double> dab_samples;
  std::vector<double> dx_samples;
  std::vector<double> dw2_samples;
  std::vector<double> dw1_samples;
  std::vector<double> combine_samples;
  std::vector<double> overlap_samples;
  megamoe::cuda::DeviceBackwardHandle last;
  megamoe::cuda::BackwardMegaKernel backward;
  for (std::uint32_t round = 0; round < rounds; ++round) {
    progress("backward_enter");
    const auto begin = Clock::now();
    auto output = backward.run(
        forward, output_gradients, {hidden, intermediate}, {});
    const auto end = Clock::now();
    if (round >= warmups) {
      wall_samples.push_back(
          std::chrono::duration<double, std::micro>(end - begin).count());
      total_samples.push_back(output.timings().total_us);
      dispatch_samples.push_back(output.timings().dispatch_us);
      dab_samples.push_back(output.timings().dab_us);
      dx_samples.push_back(output.timings().dx_us);
      dw2_samples.push_back(output.timings().dw2_us);
      dw1_samples.push_back(output.timings().dw1_us);
      combine_samples.push_back(output.timings().combine_us);
      overlap_samples.push_back(output.timings().dw_combine_overlap_us);
    }
    last = std::move(output);
    progress("backward_done");
  }

  bool repeat_equal = true;
  std::uint64_t dx_hash = 0ULL;
  std::uint64_t dgate_hash = 0ULL;
  if (validation) {
    progress("validation_download_enter");
    auto first = megamoe::cuda::BackwardValidationBridge::download(
        last, execution);
    last = backward.run(forward, output_gradients,
                        {hidden, intermediate}, {});
    progress("validation_repeat_done");
    auto repeated = megamoe::cuda::BackwardValidationBridge::download(
        last, execution);
    const auto equal_bytes = [](const auto &left, const auto &right) {
      return left.size() == right.size() &&
             (left.empty() ||
              std::memcmp(left.data(), right.data(),
                          left.size() * sizeof(left.front())) == 0);
    };
    repeat_equal = equal_bytes(first.dx, repeated.dx) &&
                   equal_bytes(first.dgate, repeated.dgate) &&
                   equal_bytes(first.dw1, repeated.dw1) &&
                   equal_bytes(first.dw2, repeated.dw2);
    dx_hash = hash_bf16_indexed(repeated.dx);
    dgate_hash = hash_fp32(repeated.dgate);
    if (const auto *artifact_dir = std::getenv("MEGAMOE_BACKWARD_OUTPUT_DIR")) {
      const auto prefix = std::string(artifact_dir) + "/rank-" +
                          std::to_string(rank) + "-";
      const auto write = [&](const std::string &name, const auto &values) {
        std::ofstream output(prefix + name,
                             std::ios::binary | std::ios::trunc);
        require(output.good(), "failed opening Backward artifact " + name);
        output.write(reinterpret_cast<const char *>(values.data()),
                     static_cast<std::streamsize>(values.size() *
                                                  sizeof(values.front())));
        require(output.good(), "failed writing Backward artifact " + name);
      };
      const auto retained =
          megamoe::cuda::download_forward_backward_state_for_validation(
              forward, execution);
      write("routes.i32", routes.selected_experts);
      write("gates.f32", route_gates);
      write("source-x.bf16", activation_host);
      write("source-dy.bf16", dy_host);
      write("canonical-x.bf16", retained.canonical_x);
      write("canonical-ab.bf16", retained.fc1_ab);
      write("dx.bf16", repeated.dx);
      write("dgate.f32", repeated.dgate);
      write("dw1.bf16", repeated.dw1);
      write("dw2.bf16", repeated.dw2);
    }
    progress("validation_done");
  }
  require(repeat_equal, "cached Backward outputs changed across executions");

  const auto emit = [](const std::vector<double> &values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0U)
        std::cout << ',';
      std::cout << values[index];
    }
  };
  const auto usage = activation_pool.usage();
  std::cout << "BACKWARD_RESULT rank=" << rank << " world=" << world
            << " mode=" << (validation ? "validation" : "benchmark")
            << " tokens=" << max_tokens << " experts=" << experts
            << " top_k=" << top_k << " hidden=" << hidden
            << " intermediate=" << intermediate
            << " incoming_rows="
            << notify.dispatch().sizes().incoming_row_count
            << " gemm_launches=" << last.audit().gemm_launches
            << " dispatch_peer_bytes="
            << last.audit().actual_dispatch_peer_bytes
            << " return_peer_bytes=" << last.audit().actual_return_peer_bytes
            << " workspace_bytes=" << last.audit().workspace_bytes
            << " pool_bytes=" << usage.device_bytes
            << " repeat_equal=" << repeat_equal << " dx_hash=" << dx_hash
            << " dgate_hash=" << dgate_hash << " wall=";
  emit(wall_samples);
  std::cout << " total=";
  emit(total_samples);
  std::cout << " dispatch=";
  emit(dispatch_samples);
  std::cout << " dab=";
  emit(dab_samples);
  std::cout << " dx=";
  emit(dx_samples);
  std::cout << " dw2=";
  emit(dw2_samples);
  std::cout << " dw1=";
  emit(dw1_samples);
  std::cout << " combine=";
  emit(combine_samples);
  std::cout << " dw_combine_overlap=";
  emit(overlap_samples);
  std::cout << std::endl;
}
