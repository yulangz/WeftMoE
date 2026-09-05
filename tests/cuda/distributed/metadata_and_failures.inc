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
