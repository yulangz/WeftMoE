void test_local(std::uint32_t world) {
  const auto experts = std::max<std::uint32_t>(16, world * 2);
  const auto top_k = std::min<std::uint32_t>(6, experts);
  const auto config = make_config(1, world, experts, top_k, 7);
  for (const auto seed : fixed_seeds()) {
    const auto routes = seeded_inputs(world, experts, top_k, seed);
    const auto actual = run_fixture(config, routes, "local-fixture", seed + 1,
                                    10, true);
    rethrow_errors(actual);
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, routes, "local-fixture", seed + 1);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      require(actual[rank].metadata ==
                  megamoe::serialize_dispatch_metadata(expected[rank]),
              "CUDA/NCCL full-field mismatch rank=" +
                  std::to_string(rank) + " seed=" + std::to_string(seed));
      const auto audit = actual[rank].audit;
      require(audit.control_starts == 1 && audit.config_starts == 1 &&
                  audit.expert_count_starts == 1 &&
                  audit.packet_summary_starts == 1,
              "successful audit is not 1/1/1/1");
      const auto staging = actual[rank].device_audit;
      require(staging.bulk_route_d2h == 0 &&
                  staging.bulk_collective_d2h == 0 &&
                  staging.host_materializations == 0 &&
                  staging.bulk_metadata_h2d == 0,
              "production generation used a retired host roundtrip");
    }
  }
  const auto repeated = seeded_inputs(world, experts, top_k, 0x1234abcdULL);
  std::vector<std::string> baseline(world);
  for (std::uint64_t iteration = 0; iteration < 20; ++iteration) {
    const auto actual = run_fixture(config, repeated, "repeat", 900,
                                    10, iteration % 2 != 0);
    rethrow_errors(actual);
    for (std::uint32_t rank = 0; rank < world; ++rank) {
      if (iteration == 0)
        baseline[rank] = actual[rank].metadata;
      else
        require(actual[rank].metadata == baseline[rank],
                "20-run byte determinism mismatch");
    }
  }
  std::cout << "[PASS] coordinator local world=" << world
            << " 25 fixed seeds + 20 byte-identical repeats\n";
}

struct FailureResult {
  std::string error;
  megamoe::cuda::CollectiveProtocolAudit audit;
  std::exception_ptr unexpected;
};

void mutate_scenario(const std::string &scenario, std::uint32_t rank,
                     megamoe::RuntimeConfig &config,
                     megamoe::RankRoutingInput &routes,
                     std::uint64_t &max_tokens, std::string &identity,
                     std::uint64_t &generation) {
  if (scenario == "two_failures") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(static_cast<std::int32_t>(slot));
    if (rank == 0)
      routes.selected_experts.back() =
          static_cast<std::int32_t>(config.num_experts);
    else
      routes.selected_experts[1] = routes.selected_experts[0];
    return;
  }
  if (rank != 1)
    return;
  if (scenario == "out_of_range") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(
          slot + 1U == config.top_k
              ? static_cast<std::int32_t>(config.num_experts)
              : static_cast<std::int32_t>(slot));
  } else if (scenario == "duplicate") {
    routes.num_tokens = 1;
    routes.selected_experts.clear();
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
      routes.selected_experts.push_back(
          static_cast<std::int32_t>(slot == 1 ? 0 : slot));
  } else if (scenario == "max_mismatch") {
    max_tokens = 11;
  } else if (scenario == "over_capacity") {
    routes.num_tokens = 11;
    routes.selected_experts.clear();
    for (std::uint64_t token = 0; token < 11; ++token)
      for (std::uint32_t slot = 0; slot < config.top_k; ++slot)
        routes.selected_experts.push_back(static_cast<std::int32_t>(slot));
  } else if (scenario == "generation_mismatch") {
    ++generation;
  } else if (scenario == "session_mismatch") {
    identity += "-different";
  } else if (scenario == "mtile_mismatch") {
    ++config.m_tile_rows;
  } else if (scenario == "multi_field_mismatch") {
    ++config.m_tile_rows;
    max_tokens = 11;
  } else if (scenario == "topology_mismatch") {
    std::swap(config.topology.expert_to_rank[0],
              config.topology.expert_to_rank[1]);
  } else if (scenario == "expert_capacity_mismatch") {
    config.capacity.expert_rows.assign(config.num_experts, 100);
    config.capacity.expert_rows[0] = 101;
  } else if (scenario == "workspace_capacity_mismatch") {
    config.capacity.workspace_rows_per_rank.assign(
        config.topology.world_size, 1000);
    config.capacity.workspace_rows_per_rank[0] = 1001;
  }
}

void test_failure(const std::string &scenario) {
  constexpr std::uint32_t world = 2;
  auto base_config = make_config(1, world, 8, 3, 4);
  if (scenario == "expert_capacity_mismatch")
    base_config.capacity.expert_rows.assign(base_config.num_experts, 100);
  if (scenario == "workspace_capacity_mismatch")
    base_config.capacity.workspace_rows_per_rank.assign(world, 1000);
  const auto base_routes = seeded_inputs(world, 8, 3, 0x55aaULL);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "failure-exec-" + std::to_string(rank), rank);
        auto config = base_config;
        auto routes = base_routes[rank];
        std::uint64_t max_tokens = 10;
        std::string identity = "failure-" + scenario;
        std::uint64_t generation = 1001;
        mutate_scenario(scenario, rank, config, routes, max_tokens, identity,
                        generation);
        auto session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, rank, bootstrap);
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        try {
          (void)coordinator.run({config, routes, max_tokens});
          throw std::runtime_error("invalid generation unexpectedly passed");
        } catch (const megamoe::NotifyDispatchError &error) {
          require(error.code() ==
                      megamoe::NotifyErrorCode::ControlProtocolFailure,
                  "logical mismatch was not a control protocol failure");
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  require(!results[0].error.empty() &&
              results[0].error == results[1].error,
          "ranks reported different structured errors");
  const bool config_failure = scenario == "topology_mismatch" ||
                              scenario == "expert_capacity_mismatch" ||
                              scenario == "workspace_capacity_mismatch";
  for (const auto &result : results) {
    require(result.audit.control_starts == 1,
            "failed generation omitted fixed control");
    require(result.audit.config_starts == (config_failure ? 1U : 0U),
            "failed generation has wrong config audit");
    require(result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "failed generation entered data collective");
  }
  std::cout << "[PASS] coordinator rejection scenario=" << scenario
            << " error='" << results[0].error << "'\n";
}

void test_protocol_regressions() {
  for (const auto *scenario :
       {"out_of_range", "duplicate", "max_mismatch", "over_capacity",
        "generation_mismatch", "session_mismatch", "mtile_mismatch",
        "two_failures", "multi_field_mismatch",
        "topology_mismatch", "expert_capacity_mismatch",
        "workspace_capacity_mismatch"})
    test_failure(scenario);
}

void test_device_capacity_failure_and_recovery() {
  constexpr std::uint32_t world = 2;
  auto bad_config = make_config(1, world, 4, 1, 4);
  bad_config.capacity.expert_rows.assign(4, 100);
  bad_config.capacity.expert_rows[0] = 1;
  auto good_config = bad_config;
  good_config.capacity.expert_rows[0] = 2;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "capacity-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "capacity-base", world, rank, 6200, rank, bootstrap);
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            base, execution, {good_config, 1, 4096, 1, 2});
        auto bad_session = megamoe::cuda::make_nccl_generation(
            base, "capacity-bad", 6201);
        megamoe::cuda::CudaNotifyGenerationCoordinator bad(
            bad_session, execution, pool);
        bool rejected = false;
        try {
          (void)bad.run({bad_config, {1, {0}}, 1});
        } catch (const megamoe::NotifyDispatchError &error) {
          rejected = error.code() ==
                     megamoe::NotifyErrorCode::ExpertCapacityExceeded;
        }
        require(rejected, "device expert capacity did not reject generation");
        const auto bad_audit = bad.audit();
        require(bad_audit.control_starts == 1 &&
                    bad_audit.config_starts == 1 &&
                    bad_audit.expert_count_starts == 1 &&
                    bad_audit.packet_summary_starts == 1,
                "device capacity failure occurred before complete collectives");
        auto good_session = megamoe::cuda::make_nccl_generation(
            base, "capacity-good", 6202);
        megamoe::cuda::CudaNotifyGenerationCoordinator good(
            good_session, execution, pool);
        auto result = good.run({good_config, {1, {0}}, 1});
        require(static_cast<bool>(result),
                "valid exact-capacity recovery did not publish");
      } catch (...) {
        errors[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : errors)
    if (error)
      std::rethrow_exception(error);
  std::cout << "[PASS] device capacity failure and exact-capacity recovery\n";
}

#if !MEGAMOE_DEBUG_VALIDATE
void test_release_fused_status_and_config_fallback() {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<megamoe::RankRoutingInput> good_routes(
      world, megamoe::RankRoutingInput{1, {0, 1, 2}});
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> unexpected(world);
  std::array<std::array<std::string, world>, 2> failures;
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "release-contract-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "release-contract", world, rank, 6600, rank, bootstrap);
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            base, execution, {config, 10, 4096, 1, 2});

        const auto run_good = [&](std::uint64_t generation) {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", generation);
          megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
              session, execution, pool);
          auto result = coordinator.run({config, good_routes[rank], 10});
          const auto audit = coordinator.audit();
          require(result && audit.fused_notify_starts == 1U &&
                      audit.control_starts == 0U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "release recovery did not use one fused collective");
          const auto actual = megamoe::serialize_dispatch_metadata(
              megamoe::cuda::download_device_metadata_for_validation(
                  result.dispatch(), execution));
          const auto expected = megamoe::build_notify_dispatch_for_testing(
              config, good_routes, "release-contract", generation);
          require(actual ==
                      megamoe::serialize_dispatch_metadata(expected[rank]),
                  "release recovery metadata/hash differs from oracle");
        };

        // Establish the immutable session config. The prime itself uses the
        // complete protocol; every following base-config generation is fused.
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6601);
          megamoe::cuda::CudaNotifyGenerationCoordinator prime(
              session, execution, pool);
          auto result = prime.run({config, good_routes[rank], 10});
          const auto audit = prime.audit();
          require(result && audit.fused_notify_starts == 0U &&
                      audit.control_starts == 1U &&
                      audit.config_starts == 1U &&
                      audit.expert_count_starts == 1U &&
                      audit.packet_summary_starts == 1U,
                  "release config prime did not use the complete protocol");
        }

        auto invalid_routes = good_routes[rank];
        if (rank == 1U)
          invalid_routes.selected_experts.back() =
              static_cast<std::int32_t>(config.num_experts);
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6602);
          megamoe::cuda::CudaNotifyGenerationCoordinator invalid(
              session, execution, pool);
          try {
            (void)invalid.run({config, invalid_routes, 10});
            throw std::runtime_error(
                "fused invalid-route generation unexpectedly passed");
          } catch (const megamoe::NotifyDispatchError &error) {
            failures[0][rank] = error.what();
          }
          const auto audit = invalid.audit();
          require(audit.fused_notify_starts == 1U &&
                      audit.control_starts == 0U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "same-config invalid route crossed the fused boundary");
        }
        run_good(6603);

        auto changed_config = config;
        changed_config.top_k = changed_config.num_experts + 1U;
        {
          auto session = megamoe::cuda::make_nccl_generation(
              base, "release-contract", 6604);
          megamoe::cuda::CudaNotifyGenerationCoordinator changed(
              session, execution, pool);
          try {
            (void)changed.run({changed_config, good_routes[rank], 10});
            throw std::runtime_error(
                "layout-changing invalid generation unexpectedly passed");
          } catch (const megamoe::NotifyDispatchError &error) {
            failures[1][rank] = error.what();
          }
          const auto audit = changed.audit();
          require(audit.fused_notify_starts == 0U &&
                      audit.control_starts == 1U &&
                      audit.config_starts == 0U &&
                      audit.expert_count_starts == 0U &&
                      audit.packet_summary_starts == 0U,
                  "layout-changing config entered the cached fused path");
        }
        run_good(6605);
      } catch (...) {
        unexpected[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : unexpected)
    if (error)
      std::rethrow_exception(error);
  for (const auto &failure : failures)
    require(!failure[0].empty() && failure[0] == failure[1],
            "release protocol failure was not identical on every rank");
  std::cout << "[PASS] release fused invalid-status + full config fallback + "
               "same-communicator oracle recoveries\n";
}
#endif

void test_failure_recovery_rounds() {
  constexpr std::uint32_t world = 2;
  constexpr std::size_t recovery_rounds = 4;
  const auto config = make_config(1, world, 8, 3, 4);
  const std::vector<std::string> scenarios{
      "generation_mismatch", "session_mismatch", "topology_mismatch",
      "max_mismatch", "out_of_range"};
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<std::exception_ptr> errors(world);
  std::vector<std::vector<std::vector<std::string>>> structured(
      scenarios.size(),
      std::vector<std::vector<std::string>>(
          recovery_rounds, std::vector<std::string>(world)));
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "recovery-exec-" + std::to_string(rank), rank);
        auto base = megamoe::cuda::make_nccl_session(
            "recovery-base", world, rank, 2000, rank, bootstrap);
        for (std::size_t category = 0; category < scenarios.size();
             ++category) {
          for (std::uint64_t round = 0; round < recovery_rounds; ++round) {
            auto bad_config = config;
            auto routes = megamoe::RankRoutingInput{1, {0, 1, 2}};
            auto max_tokens = std::uint64_t{10};
            auto identity = std::string("recovery-") + scenarios[category];
            auto generation = 3000U +
                              static_cast<std::uint64_t>(category) * 100U +
                              round * 2U;
            mutate_scenario(scenarios[category], rank, bad_config, routes,
                            max_tokens, identity, generation);
            auto bad_session = megamoe::cuda::make_nccl_generation(
                base, identity, generation);
            auto bad =
                megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    bad_session, execution);
            bool rejected = false;
            try {
              auto unexpected = bad.run({bad_config, routes, max_tokens});
              require(!unexpected,
                      "invalid generation published validated result");
            } catch (const megamoe::NotifyDispatchError &error) {
              rejected = true;
              structured[category][round][rank] = error.what();
            }
            require(rejected, "invalid recovery round was not rejected");
            const auto bad_audit = bad.audit();
            const bool config_failure =
                scenarios[category] == "topology_mismatch";
            require(bad_audit.control_starts == 1 &&
                        bad_audit.config_starts ==
                            (config_failure ? 1U : 0U) &&
                        bad_audit.expert_count_starts == 0 &&
                        bad_audit.packet_summary_starts == 0,
                    "invalid recovery audit crossed protocol boundary");

            auto good_session = megamoe::cuda::make_nccl_generation(
                base, "recovery-good-" + scenarios[category],
                3001U + static_cast<std::uint64_t>(category) * 100U +
                    round * 2U);
            auto good =
                megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                    good_session, execution);
            auto result = good.run({config, {1, {0, 1, 2}}, 10});
            require(result.audit().control_starts == 1 &&
                        result.audit().config_starts == 1 &&
                        result.audit().expert_count_starts == 1 &&
                        result.audit().packet_summary_starts == 1,
                    "recovery audit is not 1/1/1/1");
            require(result && !megamoe::serialize_dispatch_metadata(
                                   megamoe::cuda::download_device_metadata_for_validation(
                                       result.dispatch(), execution)).empty(),
                    "recovery did not publish checked metadata");
          }
        }
      } catch (...) {
        errors[rank] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &error : errors)
    if (error)
      std::rethrow_exception(error);
  for (std::size_t category = 0; category < scenarios.size(); ++category)
    for (std::size_t round = 0; round < recovery_rounds; ++round)
      require(!structured[category][round][0].empty() &&
                  structured[category][round][0] ==
                      structured[category][round][1],
              "recovery ranks selected different structured errors");
  std::cout << "[PASS] same ncclComm_t 5 categories x " << recovery_rounds
            << " invalid->valid recovery rounds\n";
}

void test_fault_injection() {
  using Point = megamoe::cuda::testing::NcclFaultPoint;
  const auto require_recovery = [](const auto &recovered, const auto &audit,
                                   const auto &config, const auto &routes,
                                   const auto &execution,
                                   const std::string &identity,
                                   std::uint64_t generation,
                                   const char *failure) {
#if MEGAMOE_DEBUG_VALIDATE
    const bool protocol_ok =
        audit.fused_notify_starts == 0 && audit.control_starts == 1 &&
        audit.config_starts == 1 && audit.expert_count_starts == 1 &&
        audit.packet_summary_starts == 1;
#else
    const bool protocol_ok =
        audit.fused_notify_starts == 1 && audit.control_starts == 0 &&
        audit.config_starts == 0 && audit.expert_count_starts == 0 &&
        audit.packet_summary_starts == 0;
#endif
    require(recovered && protocol_ok, failure);
    const auto actual = megamoe::serialize_dispatch_metadata(
        megamoe::cuda::download_device_metadata_for_validation(
            recovered.dispatch(), execution));
    const auto expected = megamoe::build_notify_dispatch_for_testing(
        config, {routes}, identity, generation);
    require(actual == megamoe::serialize_dispatch_metadata(expected.front()),
            "fault recovery produced incorrect dispatch metadata/hash");
  };
  struct FaultExpectation {
    Point point;
    megamoe::cuda::CollectiveProtocolAudit audit;
    const char *stage;
  };
  const std::vector<FaultExpectation> points{
      {Point::ControlStart, {0, 0, 0, 0}, "control-start"},
      {Point::ControlEnqueue, {0, 0, 0, 0}, "control-enqueue"},
      {Point::D2HStaging, {1, 0, 0, 0}, "control-d2h-staging"},
      {Point::EventRecord, {1, 0, 0, 0}, "control-event-record"},
      {Point::EventWait, {1, 0, 0, 0}, "control-event-wait"},
      {Point::ConfigSlotAllocation, {1, 0, 0, 0},
       "config-slot-allocation"},
      {Point::CountSlotAllocation, {1, 1, 0, 0},
       "count-slot-allocation"},
      {Point::SummarySlotAllocation, {1, 1, 0, 0},
       "summary-slot-allocation"},
      {Point::BackendCompletion, {1, 1, 1, 1},
       "summary-backend-completion"}};
  const auto config = make_config(1, 1, 8, 3, 4);
  for (std::size_t index = 0; index < points.size(); ++index) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution("fault-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        "fault", 1, 0, 4000 + index, 0, bootstrap);
    megamoe::cuda::CollectiveProtocolAudit audit;
    {
      auto pool = megamoe::cuda::make_device_notify_resource_pool(
          session, execution, {config, 10, 4096, 1, 2});
      // Establish the session-static config on this exact pooled backend.
      // Release must then recover through the fused protocol; debug must
      // continue to execute the complete protocol without this release-only
      // setup prime. Lifecycle assertions below distinguish both modes.
#if !MEGAMOE_DEBUG_VALIDATE
      {
        auto prime_session = megamoe::cuda::make_nccl_generation(
            session, "fault", 3900 + index);
        megamoe::cuda::CudaNotifyGenerationCoordinator prime(
            prime_session, execution, pool);
        require(static_cast<bool>(
                    prime.run({config, {1, {0, 1, 2}}, 10})),
                "failed to prime fault-recovery static config");
      }
#endif
      megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
          session, execution, pool);
      megamoe::cuda::testing::inject_next_nccl_fault(points[index].point);
      bool failed = false;
      try {
        auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
        require(!unexpected, "fault generation published validated result");
      } catch (const megamoe::NotifyDispatchError &error) {
        failed = true;
        const std::string message = error.what();
        require(message.find(points[index].stage) != std::string::npos,
                "fault error lost exact stage");
        require(message.find("controlled test fault") != std::string::npos,
                "fault error lost original backend text");
      }
      require(failed, std::string("fault generation published validated result at ") +
                          points[index].stage);
      audit = coordinator.audit();
      require(pool.usage().slots_in_use == 0,
              "pooled fault did not return failed generation slot");
      auto recovery_session = megamoe::cuda::make_nccl_generation(
          session, "fault", 5000 + index);
      megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
          recovery_session, execution, pool);
      const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
      auto recovered = recovery.run({config, routes, 10});
      require_recovery(recovered, recovery.audit(), config, routes, execution,
                       "fault", 5000 + index,
                       "pooled fault did not recover with the exact protocol "
                       "on the same communicator");
    }
    const auto expected = points[index].audit;
    require(audit.control_starts == expected.control_starts &&
                audit.config_starts == expected.config_starts &&
                audit.expert_count_starts == expected.expert_count_starts &&
                audit.packet_summary_starts ==
                    expected.packet_summary_starts,
            "fault audit does not match enqueue boundary");
    require(audit.control_starts <= 1 && audit.config_starts <= 1 &&
                audit.expert_count_starts <= 1 &&
                audit.packet_summary_starts <= 1,
            "fault audit regressed or double-incremented");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.slot_allocations == lifecycle.slot_destructions,
            "fault leaked a collective slot allocation");
    require(lifecycle.completion_lease_creates ==
                lifecycle.completion_lease_destroys,
            "fault leaked a completion lease");
#if MEGAMOE_DEBUG_VALIDATE
    require(lifecycle.validated_publishes == 1,
            "debug fault/recovery published other than recovery");
#else
    require(lifecycle.validated_publishes == 2,
            "release fault/recovery published other than prime + recovery");
#endif
    require(lifecycle.operation_cancels <= 1,
            "coordinator fault double-cancelled");
    require(lifecycle.operation_waits <= 8 &&
                lifecycle.destructor_drains <= 2,
            "fault lifecycle repeated wait/drain");
  }
  const std::vector<FaultExpectation> device_stage_points{
      {Point::CountKernelCompletion, {1, 1, 1, 0},
       "count-kernel-completion"},
      {Point::SummaryKernelCompletion, {1, 1, 1, 1},
       "summary-kernel-completion"},
      {Point::MaterializationKernel, {1, 1, 1, 1},
       "materialization-kernel"}};
  for (std::size_t index = 0; index < device_stage_points.size(); ++index) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution(
        "device-stage-fault-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        "device-stage-fault", 1, 0, 4050 + index, 0, bootstrap);
    auto pool = megamoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
#if !MEGAMOE_DEBUG_VALIDATE
    {
      auto prime_session = megamoe::cuda::make_nccl_generation(
          session, "device-stage-fault", 3950 + index);
      megamoe::cuda::CudaNotifyGenerationCoordinator prime(
          prime_session, execution, pool);
      require(static_cast<bool>(
                  prime.run({config, {1, {0, 1, 2}}, 10})),
              "failed to prime device-stage recovery static config");
    }
#endif
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    megamoe::cuda::testing::inject_next_nccl_fault(
        device_stage_points[index].point);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10});
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = std::string(error.what()).find(
                   device_stage_points[index].stage) != std::string::npos;
    }
    require(failed && pool.usage().slots_in_use == 0,
            "pooled device-stage fault leaked/published its slot");
    const auto actual = coordinator.audit();
    const auto expected = device_stage_points[index].audit;
    require(actual.control_starts == expected.control_starts &&
                actual.config_starts == expected.config_starts &&
                actual.expert_count_starts == expected.expert_count_starts &&
                actual.packet_summary_starts ==
                    expected.packet_summary_starts,
            "device-stage fault audit crossed the wrong boundary");
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        session, "device-stage-fault", 4150 + index);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
    auto recovered = recovery.run({config, routes, 10});
    require_recovery(recovered, recovery.audit(), config, routes, execution,
                     "device-stage-fault", 4150 + index,
                     "device-stage fault did not recover with the exact "
                     "protocol on the same communicator");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
#if MEGAMOE_DEBUG_VALIDATE
    require(lifecycle.validated_publishes == 1 &&
                lifecycle.operation_cancels == 0,
            "debug device-stage fault published/cancelled an invalid "
            "generation");
#else
    require(lifecycle.validated_publishes == 2 &&
                lifecycle.operation_cancels == 0,
            "release device-stage fault published other than prime + "
            "recovery or cancelled an invalid generation");
#endif
  }
  for (const bool deadline : {false, true}) {
    megamoe::cuda::testing::reset_nccl_lifecycle_counters();
    auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
    auto execution = megamoe::cuda::make_cuda_execution(
        deadline ? "deadline-exec" : "cancel-exec", 0);
    auto session = megamoe::cuda::make_nccl_session(
        deadline ? "deadline" : "cancel", 1, 0,
        deadline ? 4091 : 4090, 0, bootstrap);
    auto pool = megamoe::cuda::make_device_notify_resource_pool(
        session, execution, {config, 10, 4096, 1, 2});
    megamoe::cuda::GenerationRunOptions options;
    if (deadline)
      options.deadline_after = std::chrono::microseconds(1);
    else
      options.cancellation.request_cancel();
    megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
        session, execution, pool);
    bool failed = false;
    try {
      (void)coordinator.run({config, {1, {0, 1, 2}}, 10}, options);
    } catch (const megamoe::NotifyDispatchError &error) {
      failed = std::string(error.what()).find("generation-cancel") !=
               std::string::npos;
    }
    require(failed, "production cancellation/deadline path did not fail");
    const auto audit = coordinator.audit();
    require(audit.control_starts == 1 && audit.config_starts == 1 &&
                audit.expert_count_starts == 1 &&
                audit.packet_summary_starts == 0,
            "cancellation audit is not 1/1/1/0");
    const auto lifecycle =
        megamoe::cuda::testing::nccl_lifecycle_snapshot();
    require(lifecycle.operation_cancels == 1 &&
                lifecycle.validated_publishes == 0,
            "cancellation was not executed exactly once or published a handle");
    auto recovery_session = megamoe::cuda::make_nccl_generation(
        session, deadline ? "deadline" : "cancel",
        deadline ? 4191 : 4190);
    megamoe::cuda::CudaNotifyGenerationCoordinator recovery(
        recovery_session, execution, pool);
    const megamoe::RankRoutingInput routes{1, {0, 1, 2}};
    auto recovered = recovery.run({config, routes, 10});
    require_recovery(recovered, recovery.audit(), config, routes, execution,
                     deadline ? "deadline" : "cancel",
                     deadline ? 4191 : 4190,
                     "communicator did not recover with the exact protocol "
                     "after cancellation/deadline");
  }
  std::cout << "[PASS] 9-point NCCL + 3 device-stage pooled fault matrix + "
               "production cancel/deadline exact cancel=1\n";
}

void test_config_wire_corruption(std::uint32_t world,
                                 std::uint32_t corrupt_rank) {
  const auto config = make_config(1, world, 8, 3, 4);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "wire-corruption-exec-" + std::to_string(rank), rank);
        auto session = megamoe::cuda::make_nccl_session(
            "wire-corruption", world, rank, 5100 + corrupt_rank, rank,
            bootstrap);
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        if (rank == corrupt_rank)
          megamoe::cuda::testing::corrupt_next_config_wire(64);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected, "corrupt config published validated result");
        } catch (const megamoe::NotifyDispatchError &error) {
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  for (const auto &result : results) {
    require(!result.error.empty() &&
                result.error.find("checksum") != std::string::npos,
            "config corruption did not report integrity failure");
    require(result.audit.control_starts == 1 &&
                result.audit.config_starts == 1 &&
                result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "config corruption audit is not 1/1/0/0");
  }
  for (std::uint32_t rank = 1; rank < world; ++rank)
    require(results[rank].error == results[0].error,
            "config corruption errors differ by validation rank");
  std::cout << "[PASS] config envelope corruption world=" << world
            << " origin_rank=" << corrupt_rank << " audit=1/1/0/0\n";
}

void test_exact_session_identity_agreement(bool different_length) {
  constexpr std::uint32_t world = 2;
  const auto config = make_config(1, world, 8, 3, 4);
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<FailureResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto execution = megamoe::cuda::make_cuda_execution(
            "exact-session-exec-" + std::to_string(rank), rank);
        // Display name and header summary are deliberately equal. Only one raw
        // byte differs, proving that exact identity bytes are the final gate.
        auto session = megamoe::cuda::make_nccl_session(
            "same-display", world, rank, different_length ? 5201 : 5200,
            rank, bootstrap);
        std::vector<std::byte> raw_identity{
            std::byte{0x41}, std::byte{0x42},
            static_cast<std::byte>(0x43U + rank)};
        if (different_length && rank == 1)
          raw_identity.push_back(std::byte{0x44});
        megamoe::cuda::testing::override_next_protocol_identity(
            std::move(raw_identity),
            {0x1111111111111111ULL, 0x2222222222222222ULL});
        auto coordinator =
            megamoe::cuda::testing::make_unpooled_coordinator_for_testing(
                session, execution);
        try {
          auto unexpected = coordinator.run({config, {1, {0, 1, 2}}, 10});
          require(!unexpected,
                  "different exact session identity published result");
        } catch (const megamoe::NotifyDispatchError &error) {
          results[rank].error = error.what();
          results[rank].audit = coordinator.audit();
        }
      } catch (...) {
        results[rank].unexpected = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &result : results)
    if (result.unexpected)
      std::rethrow_exception(result.unexpected);
  require(!results[0].error.empty() && results[0].error == results[1].error &&
              results[0].error.find("field=session_identity") !=
                  std::string::npos &&
              results[0].error.find("origin_rank=1") != std::string::npos,
          "exact session identity error is not globally deterministic");
  for (const auto &result : results)
    require(result.audit.control_starts == 1 &&
                result.audit.config_starts == 1 &&
                result.audit.expert_count_starts == 0 &&
                result.audit.packet_summary_starts == 0,
            "exact session identity audit is not 1/1/0/0");
  std::cout << "[PASS] equal summary/display but different raw session identity"
            << (different_length ? " length" : " byte")
            << " audit=1/1/0/0\n";
}

void test_protocol_failure_suite() {
  test_protocol_regressions();
  test_failure_recovery_rounds();
  test_config_wire_corruption(2, 0);
  test_config_wire_corruption(2, 1);
  test_exact_session_identity_agreement(false);
  test_exact_session_identity_agreement(true);
}
