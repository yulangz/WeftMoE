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
    transport_config.max_owned_experts = experts / world;
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
