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
