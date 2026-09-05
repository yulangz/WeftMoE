using Clock = std::chrono::steady_clock;

#if MEGAMOE_DEVICE_TIMELINE
void write_device_timeline(
    const megamoe::cuda::profiling::DeviceTimelineSnapshot &snapshot,
    const std::string &prefix) {
  const auto phase = "forward";
  const auto path = prefix + ".rank" + std::to_string(snapshot.rank) + "." +
                    phase + ".json";
  std::ofstream output(path, std::ios::trunc);
  if (!output)
    throw std::runtime_error("cannot create device timeline: " + path);
  output << "{\"schema\":" << snapshot.schema_version
         << ",\"rank\":" << snapshot.rank
         << ",\"generation\":" << snapshot.generation
         << ",\"phase\":\"" << phase << "\""
         << ",\"max_ctas\":" << snapshot.max_ctas
         << ",\"capacity_per_cta\":" << snapshot.capacity_per_cta
         << ",\"overflow_events\":" << snapshot.overflow_events
         << ",\"events\":[";
  for (std::size_t index = 0; index < snapshot.records.size(); ++index) {
    const auto &event = snapshot.records[index];
    if (index != 0)
      output << ',';
    output << "{\"timestamp_ns\":" << event.timestamp_ns
           << ",\"generation\":" << event.generation
           << ",\"rank\":" << event.rank << ",\"cta\":" << event.cta
           << ",\"channel\":" << event.channel
           << ",\"phase\":" << ((event.metadata >> 24U) & 0xffU)
           << ",\"role\":" << ((event.metadata >> 16U) & 0xffU)
           << ",\"event\":" << (event.metadata & 0xffffU)
           << ",\"arg0\":" << event.arg0 << ",\"arg1\":" << event.arg1
           << '}';
  }
  output << "]}\n";
  if (!output)
    throw std::runtime_error("failed writing device timeline: " + path);
  std::cout << "DEVICE_TIMELINE rank=" << snapshot.rank
            << " phase=" << phase << " events=" << snapshot.records.size()
            << " overflow=" << snapshot.overflow_events
            << " path=" << path << std::endl;
}
#endif

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}
megamoe::cuda::CudaGateInput make_unit_gates(
    const megamoe::cuda::CudaRouteInput &routes,
    const megamoe::RankCollectiveContext &context,
    const megamoe::cuda::CudaExecution &execution) {
  return megamoe::cuda::testing::upload_fp32_gates_for_testing(
      std::vector<float>(static_cast<std::size_t>(routes.route_count()),
                         1.0F),
      routes, context, execution);
}

megamoe::RuntimeConfig make_config(std::uint32_t nodes, std::uint32_t gpus,
                                   std::uint32_t experts,
                                   std::uint32_t top_k,
                                   std::uint32_t mtile) {
  megamoe::RuntimeConfig config;
  config.num_experts = experts;
  config.top_k = top_k;
  config.m_tile_rows = mtile;
  config.topology.num_nodes = nodes;
  config.topology.world_size = nodes * gpus;
  for (std::uint32_t node = 0; node < nodes; ++node)
    for (std::uint32_t local = 0; local < gpus; ++local) {
      config.topology.rank_to_node.push_back(node);
      config.topology.rank_to_local_rank.push_back(local);
    }
  for (std::uint32_t expert = 0; expert < experts; ++expert)
    config.topology.expert_to_rank.push_back(
        expert % config.topology.world_size);
  return config;
}

std::vector<megamoe::RankRoutingInput>
seeded_inputs(std::uint32_t world, std::uint32_t experts,
              std::uint32_t top_k, std::uint64_t seed,
              std::uint64_t max_tokens = 10) {
  const auto next = [&seed]() mutable {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return seed;
  };
  std::vector<megamoe::RankRoutingInput> result(world);
  auto random = next;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    result[rank].num_tokens = random() % (max_tokens + 1);
    for (std::uint64_t token = 0; token < result[rank].num_tokens; ++token) {
      std::vector<std::uint32_t> choices(experts);
      for (std::uint32_t expert = 0; expert < experts; ++expert)
        choices[expert] = expert;
      for (std::uint32_t slot = 0; slot < top_k; ++slot) {
        const auto selected = slot + static_cast<std::uint32_t>(
                                         random() % (experts - slot));
        std::swap(choices[slot], choices[selected]);
        result[rank].selected_experts.push_back(
            static_cast<std::int32_t>(choices[slot]));
      }
    }
  }
  return result;
}

std::vector<std::uint64_t> fixed_seeds() {
  return {0x5eed1234ULL, 0x10203040ULL, 0xdeadbeefULL, 0xc001d00dULL,
          0x01234567ULL, 0x89abcdefULL, 0x31415926ULL, 0x27182818ULL,
          0x0badf00dULL, 0xfeedfaceULL, 0x13579bdfULL, 0x2468ace0ULL,
          0xa5a5a5a5ULL, 0x5a5a5a5aULL, 0x11223344ULL, 0x55667788ULL,
          0x99aabbccULL, 0xddeeff00ULL, 0x42424242ULL, 0x77777777ULL,
          0xabcdef01ULL, 0x10fedcbaULL, 0x600dbeefULL, 0x7f4a7c15ULL,
          0xf00dcafeULL};
}

std::vector<float> fixed_random_selected_gates(
    std::uint64_t tokens, std::uint32_t top_k, std::uint32_t rank,
    std::uint64_t seed) {
  std::vector<float> gates(
      static_cast<std::size_t>(tokens * static_cast<std::uint64_t>(top_k)));
  for (std::uint64_t token = 0; token < tokens; ++token) {
    std::uint64_t total = 0;
    std::vector<std::uint32_t> scores(top_k);
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      const auto mixed = seed + token * 1315423911ULL +
                         static_cast<std::uint64_t>(slot) * 2654435761ULL +
                         static_cast<std::uint64_t>(rank) * 97531ULL;
      scores[slot] = 1U + static_cast<std::uint32_t>(mixed % 65521ULL);
      total += scores[slot];
    }
    for (std::uint32_t slot = 0; slot < top_k; ++slot)
      gates[static_cast<std::size_t>(token * top_k + slot)] =
          static_cast<float>(static_cast<double>(scores[slot]) /
                             static_cast<double>(total));
  }
  return gates;
}

std::uint64_t hash_fp32(const std::vector<float> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const float value : values) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      hash = (hash ^ ((bits >> (byte * 8U)) & 0xffU)) *
             1099511628211ULL;
  }
  return hash;
}

std::uint64_t hash_i32(const std::vector<std::int32_t> &values) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const auto value : values) {
    const auto bits = static_cast<std::uint32_t>(value);
    for (unsigned byte = 0; byte < sizeof(bits); ++byte)
      hash = (hash ^ ((bits >> (byte * 8U)) & 0xffU)) *
             1099511628211ULL;
  }
  return hash;
}

std::uint64_t hash_bf16_indexed(
    const std::vector<megamoe::transport::BFloat16> &values) {
  std::uint64_t hash = 0;
  for (std::size_t index = 0; index < values.size(); ++index)
    hash += static_cast<std::uint64_t>(values[index].bits) *
            (static_cast<std::uint64_t>(index) + 1ULL);
  return hash;
}

std::uint32_t fixture_mix32_host(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

std::uint16_t fixture_weight_bits(std::uint32_t expert,
                                  std::uint32_t tensor_kind,
                                  std::uint64_t row,
                                  std::uint64_t column) {
  auto key = fixture_mix32_host(expert ^ 0x9e3779b9U);
  key ^= fixture_mix32_host(tensor_kind ^ 0x85ebca6bU);
  key ^= fixture_mix32_host(static_cast<std::uint32_t>(row) ^ 0xc2b2ae35U);
  key ^=
      fixture_mix32_host(static_cast<std::uint32_t>(column) ^ 0x27d4eb2fU);
  const auto signed_value =
      static_cast<int>(fixture_mix32_host(key) & 1U) * 2 - 1;
  return megamoe::transport::float_to_bfloat16(
             static_cast<float>(signed_value) / 256.0F)
      .bits;
}

std::uint64_t forward_weight_sample_hash(std::uint32_t expert_begin,
                                         std::uint32_t expert_count,
                                         std::uint64_t hidden,
                                         std::uint64_t intermediate) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (std::uint32_t local = 0; local < expert_count; ++local) {
    const auto expert = expert_begin + local;
    for (std::uint64_t sample = 0; sample < 32ULL; ++sample) {
      const auto row = (sample * 7919ULL + 5ULL) % (2ULL * intermediate);
      const auto column = (sample * 104729ULL + 17ULL) % hidden;
      const auto bits = fixture_weight_bits(expert, 1U, row, column);
      hash = (hash ^ static_cast<std::uint8_t>(bits)) * 1099511628211ULL;
      hash = (hash ^ static_cast<std::uint8_t>(bits >> 8U)) *
             1099511628211ULL;
    }
    for (std::uint64_t sample = 0; sample < 32ULL; ++sample) {
      const auto row = (sample * 65537ULL + 11ULL) % hidden;
      const auto column = (sample * 8191ULL + 3ULL) % intermediate;
      const auto bits = fixture_weight_bits(expert, 2U, row, column);
      hash = (hash ^ static_cast<std::uint8_t>(bits)) * 1099511628211ULL;
      hash = (hash ^ static_cast<std::uint8_t>(bits >> 8U)) *
             1099511628211ULL;
    }
  }
  return hash;
}

struct RunResult {
  std::string metadata;
  megamoe::cuda::CollectiveProtocolAudit audit;
  megamoe::cuda::DeviceMaterializationAudit device_audit;
  megamoe::cuda::NcclWorkspaceUsage workspace;
  megamoe::cuda::GenerationSetupTimings setup;
  megamoe::cuda::GenerationExecutionTimings execution;
  double session_setup_us = 0;
  double pool_setup_us = 0;
  double wall_us = 0;
  std::exception_ptr error;
};

std::vector<RunResult> run_fixture(
    const megamoe::RuntimeConfig &config,
    const std::vector<megamoe::RankRoutingInput> &routes,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens, bool stagger = false) {
  const auto world = config.topology.world_size;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        auto execution = megamoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto session_start = Clock::now();
        auto session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        const auto session_end = Clock::now();
        results[rank].session_setup_us =
            std::chrono::duration<double, std::micro>(session_end -
                                                      session_start)
                .count();
        if (stagger && rank % 3U != 0)
          std::this_thread::sleep_for(
              std::chrono::microseconds(20U * (rank % 3U)));
        const auto start = Clock::now();
        auto pool = megamoe::cuda::make_device_notify_resource_pool(
            session, execution, {config, max_tokens, 4096, 2});
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, execution, pool);
        auto result = coordinator.run({config, routes[rank], max_tokens});
        const auto finish = Clock::now();
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            megamoe::cuda::download_device_metadata_for_validation(
                result.dispatch(), execution));
        results[rank].audit = result.audit();
        results[rank].device_audit = result.dispatch().audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
        results[rank].wall_us =
            std::chrono::duration<double, std::micro>(finish - start).count();
        megamoe::cuda::validate_cuda_dispatch_metadata(result.dispatch(),
                                                       execution);
      } catch (...) {
        results[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  return results;
}

struct ReusableRankResources {
  megamoe::cuda::CudaExecution execution;
  megamoe::cuda::NcclSession base_session;
  megamoe::cuda::DeviceNotifyResourcePool pool;
  megamoe::cuda::CudaRouteInput device_routes;
  megamoe::cuda::CudaGateInput device_gates;
  double session_setup_us = 0;
  double pool_setup_us = 0;
  std::exception_ptr error;
};

std::vector<ReusableRankResources>
make_reusable_resources(const megamoe::RuntimeConfig &config,
                        const std::string &identity,
                        std::uint64_t generation,
                        std::uint64_t max_tokens) {
  const auto world = config.topology.world_size;
  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  std::vector<ReusableRankResources> resources(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        const auto device = config.topology.rank_to_local_rank[rank];
        resources[rank].execution = megamoe::cuda::make_cuda_execution(
            identity + "-exec-" + std::to_string(rank), device);
        const auto start = Clock::now();
        resources[rank].base_session = megamoe::cuda::make_nccl_session(
            identity, world, rank, generation, device, bootstrap);
        resources[rank].session_setup_us =
            std::chrono::duration<double, std::micro>(Clock::now() - start)
                .count();
        resources[rank].pool = megamoe::cuda::make_device_notify_resource_pool(
            resources[rank].base_session, resources[rank].execution,
            {config, max_tokens, 4096, 2});
        resources[rank].pool_setup_us =
            resources[rank].pool.usage().one_time_setup_us;
      } catch (...) {
        resources[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &resource : resources)
    if (resource.error)
      std::rethrow_exception(resource.error);
  return resources;
}

std::vector<RunResult> run_reused_fixture(
    const megamoe::RuntimeConfig &config,
    const std::vector<megamoe::RankRoutingInput> &routes,
    const std::vector<ReusableRankResources> &resources,
    const std::string &identity, std::uint64_t generation,
    std::uint64_t max_tokens) {
  const auto world = config.topology.world_size;
  std::vector<RunResult> results(world);
  std::vector<std::thread> threads;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    threads.emplace_back([&, rank] {
      try {
        auto session = megamoe::cuda::make_nccl_generation(
            resources[rank].base_session, identity, generation);
        const auto context = megamoe::make_rank_collective_context(
            session.session(), rank, resources[rank].execution.token());
        const auto exact_routes = resources[rank].device_routes
                                      ? megamoe::cuda::DeviceRouteProvenanceBridge::reissue_for_testing(
                                            resources[rank].device_routes,
                                            context,
                                            resources[rank].execution)
                                      : megamoe::cuda::CudaRouteInput{};
        const auto exact_gates = exact_routes
            ? megamoe::cuda::DeviceGateProvenanceBridge::reissue_for_testing(
                  resources[rank].device_gates, exact_routes, context,
                  resources[rank].execution)
            : megamoe::cuda::CudaGateInput{};
        megamoe::cuda::CudaNotifyGenerationCoordinator coordinator(
            session, resources[rank].execution, resources[rank].pool);
        auto result = exact_routes
                          ? coordinator.run_device(
                                {config, exact_routes, exact_gates, max_tokens})
                          : coordinator.run(
                                {config, routes[rank], max_tokens});
        results[rank].metadata = megamoe::serialize_dispatch_metadata(
            megamoe::cuda::download_device_metadata_for_validation(
                result.dispatch(), resources[rank].execution));
        results[rank].audit = result.audit();
        results[rank].device_audit = result.dispatch().audit();
        results[rank].workspace = result.workspace_usage();
        results[rank].setup = result.setup_timings();
        results[rank].execution = result.execution_timings();
        const auto release_start = Clock::now();
        result = {};
        results[rank].execution.pool_acquire_release_us +=
            std::chrono::duration<double, std::micro>(Clock::now() -
                                                      release_start)
                .count();
      } catch (...) {
        results[rank].error = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  return results;
}

void rethrow_errors(const std::vector<RunResult> &results) {
  for (const auto &result : results)
    if (result.error)
      std::rethrow_exception(result.error);
}
