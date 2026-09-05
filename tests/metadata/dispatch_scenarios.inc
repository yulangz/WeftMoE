void test_rank_local_async_and_deterministic_mapping() {
  const auto config = make_config(2, 2, 8, 3, 4);
  const auto inputs = seeded_inputs(4, 8, 3, 0x1234ULL);
  const auto session = megamoe::make_collective_session("async", 4, 9);
  std::vector<LocalHistogramHandle> locals;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    auto context = megamoe::make_rank_collective_context(
        session, rank,
        megamoe::make_execution_token("exec-" + std::to_string(rank)));
    locals.push_back(
        megamoe::build_rank_local_histogram(config, context, inputs[rank]));
  }
  HostFakeCountExchange backend(4, "async");
  std::vector<CollectiveOperation> count_ops;
  count_ops.reserve(4);
  count_ops.push_back(megamoe::start_count_exchange(locals[0], backend));
  REQUIRE(count_ops[0].state() == ExchangeState::Pending);
  auto wait0 =
      std::async(std::launch::async, [&] { return count_ops[0].wait(); });
  REQUIRE(wait0.wait_for(std::chrono::milliseconds(20)) ==
          std::future_status::timeout);
  for (std::uint32_t rank = 1; rank < 4; ++rank) {
    count_ops.push_back(megamoe::start_count_exchange(locals[rank], backend));
  }
  std::vector<ValidatedCollectiveHandle> counts;
  counts.push_back(wait0.get());
  for (std::uint32_t rank = 1; rank < 4; ++rank) {
    counts.push_back(count_ops[rank].wait());
  }
  std::vector<CollectiveOperation> summary_ops;
  summary_ops.reserve(4);
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    summary_ops.push_back(
        megamoe::start_notify_summary_exchange(locals[rank], backend));
  }
  std::vector<DispatchHandle> handles;
  for (std::uint32_t rank = 0; rank < 4; ++rank) {
    const auto summary = summary_ops[rank].wait();
    handles.push_back(megamoe::materialize_rank_metadata(
        locals[rank], counts[rank], summary));
  }
  verify_mapping(handles);
}
void test_existing_shapes_descriptors_and_repeatability() {
  auto config = make_config(2, 2, 8, 2, 4);
  std::vector<RankRoutingInput> inputs(4);
  inputs[0] = {3, {0, 1, 0, 2, 0, 3}};
  inputs[1] = {2, {0, 4, 0, 5}};
  inputs[3] = {1, {0, 6}};
  const auto first =
      megamoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  const auto second =
      megamoe::build_notify_dispatch_for_testing(config, inputs, "repeat", 7);
  for (std::size_t rank = 0; rank < first.size(); ++rank) {
    REQUIRE(megamoe::serialize_dispatch_metadata(first[rank]) ==
            megamoe::serialize_dispatch_metadata(second[rank]));
  }
  REQUIRE(first[0]->incoming.mtile_expected_rows[0] ==
          std::vector<std::uint32_t>({4, 2}));
  REQUIRE(first[2]->outgoing.num_tokens == 0);
  verify_mapping(first);

  auto packet_config = make_config(2, 2, 6, 3, 4);
  packet_config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3};
  std::vector<RankRoutingInput> packet_inputs(4);
  packet_inputs[0] = {2, {2, 4, 3, 0, 2, 5}};
  const auto packet_handles = megamoe::build_notify_dispatch_for_testing(
      packet_config, packet_inputs, "packet", 3);
  const auto &transport = packet_handles[0]->transport;
  REQUIRE(transport.outgoing_packets.size() == 3);
  REQUIRE(transport.outgoing_packets[1].identity.destination_node == 1);
  REQUIRE(transport.outgoing_packets[1].fanout.size() == 3);
  REQUIRE(transport.outgoing_packets[1].proxy_rank == 2);
  REQUIRE(transport.outgoing_signals[1].expected_packets == 2);
  REQUIRE(
      packet_handles[2]->transport.incoming_proxy_signals[0].expected_signals ==
      2);

  auto many = make_config(73, 1, 73, 2, 8);
  std::vector<RankRoutingInput> many_inputs(73);
  many_inputs[0] = {1, {0, 72}};
  const auto many_handles =
      megamoe::build_notify_dispatch_for_testing(many, many_inputs, "many", 11);
  REQUIRE(many_handles[0]->outgoing.node_mask_words == 2);
  REQUIRE(many_handles[0]->outgoing.destination_node_mask ==
          std::vector<std::uint64_t>({1, std::uint64_t{1} << 8U}));
  REQUIRE(many_handles[0]->transport.outgoing_signals.size() == 73);
}

void test_activation_transport_protocol_oracle() {
  auto config = make_config(2, 2, 8, 3, 4);
  config.topology.expert_to_rank = {0, 1, 2, 3, 2, 3, 0, 1};
  std::vector<RankRoutingInput> inputs(4);
  inputs[0] = {3, {4, 1, 5, 0, 3, 6, 2, 7, 4}};
  const auto handles = megamoe::build_notify_dispatch_for_testing(
      config, inputs, "transport-oracle", 77);
  megamoe::transport::ActivationTransportConfig transport_config;
  transport_config.hidden_size = 33;
  transport_config.channels = 1;
  transport_config.ring_slots_per_source_channel = 2;
  std::vector<megamoe::transport::BFloat16> activations(3U * 33U);
  for (std::size_t i = 0; i < activations.size(); ++i)
    activations[i] = megamoe::transport::float_to_bfloat16(
        static_cast<float>((i * 17U) % 101U) / 32.0F);
  const std::vector<float> gates{0.2F, 0.3F, 0.5F, 0.1F, 0.6F,
                                 0.3F, 0.7F, 0.2F, 0.1F};
  const auto canonical = megamoe::transport::simulate_activation_transport(
      handles[0], activations, gates, transport_config, 1);
  std::vector<std::uint64_t> reversed(
      handles[0]->transport.outgoing_packets.size());
  std::iota(reversed.rbegin(), reversed.rend(), std::uint64_t{0});
  const auto reordered = megamoe::transport::simulate_activation_transport(
      handles[0], activations, gates, transport_config, 1, reversed);
  REQUIRE(canonical.combined_tokens.size() == activations.size());
  REQUIRE(canonical.contribution_by_route == reordered.contribution_by_route);
  REQUIRE(canonical.combined_tokens == reordered.combined_tokens);
  REQUIRE(canonical.canonical_expert_rows == reordered.canonical_expert_rows);
  REQUIRE(canonical.audit.dispatch_packets ==
          handles[0]->transport.outgoing_packets.size());
  REQUIRE(canonical.audit.relayed_fanout == 9);
  REQUIRE(canonical.audit.synthetic_contributions == 9);
  const auto bad_sequence =
      megamoe::testing::NotifyCorruptionHook::with_packet_sequence(
          handles[0], 0, 7);
  expect_error(NotifyErrorCode::InvalidHandle,
               [&] {
                 (void)megamoe::transport::simulate_activation_transport(
                     bad_sequence, activations, gates, transport_config, 1,
                     reversed);
               },
               {"phase=dispatch-validation",
                "field=transport.packet.identity"});

  auto many = make_config(73, 1, 73, 2, 8);
  std::vector<RankRoutingInput> many_inputs(73);
  many_inputs[0] = {1, {0, 72}};
  const auto many_handles = megamoe::build_notify_dispatch_for_testing(
      many, many_inputs, "transport-73", 78);
  transport_config.hidden_size = 7;
  std::vector<megamoe::transport::BFloat16> many_activations(7);
  const auto many_result = megamoe::transport::simulate_activation_transport(
      many_handles[0], many_activations, {0.25F, 0.75F}, transport_config, 2);
  REQUIRE(many_handles[0]->outgoing.node_mask_words == 2);
  REQUIRE(many_result.audit.dispatch_packets == 2);
  REQUIRE(many_result.audit.deduplicated_cross_node_payload_bytes == 14);
  REQUIRE(many_result.audit.raw_cross_node_payload_bytes == 14);

  // Descriptor/team logic is runtime-shaped and is not restricted to the
  // physically available two-node, 1/8-GPU acceptance topology.
  const auto verify_runtime_topology = [&](std::uint32_t nodes,
                                            std::uint32_t gpus) {
    const auto world = nodes * gpus;
    const auto top_k = std::min<std::uint32_t>(6U, world);
    auto runtime = make_config(nodes, gpus, world, top_k, 4);
    const auto source_rank = std::min<std::uint32_t>(2U, gpus - 1U);
    std::vector<RankRoutingInput> runtime_inputs(world);
    runtime_inputs[source_rank].num_tokens = 2;
    for (std::uint64_t token = 0; token < 2; ++token)
      for (std::uint32_t slot = 0; slot < top_k; ++slot)
        runtime_inputs[source_rank].selected_experts.push_back(
            static_cast<std::int32_t>((token * top_k + slot) % world));
    const auto runtime_handles = megamoe::build_notify_dispatch_for_testing(
        runtime, runtime_inputs,
        "runtime-topology-" + std::to_string(nodes) + "x" +
            std::to_string(gpus),
        79 + nodes * 10U + gpus);
    const auto &descriptor = runtime_handles[source_rank]->transport;
    REQUIRE(descriptor.outgoing_signals.size() == nodes);
    REQUIRE(descriptor.outgoing_packets.size() <= 2ULL * nodes);
    const auto source_local = source_rank % gpus;
    for (const auto &packet : descriptor.outgoing_packets) {
      REQUIRE(packet.proxy_rank ==
              packet.identity.destination_node * gpus + source_local);
      REQUIRE(packet.channel == 0U);
      for (const auto &fanout : packet.fanout) {
        const auto owner = runtime.topology.expert_to_rank[fanout.expert_id];
        REQUIRE(fanout.destination_local_rank == owner % gpus);
      }
    }
    megamoe::transport::ActivationTransportConfig runtime_transport;
    runtime_transport.hidden_size = 7;
    runtime_transport.channels = 4;
    runtime_transport.ring_slots_per_source_channel = 2;
    std::vector<megamoe::transport::BFloat16> runtime_activations(14);
    std::vector<float> runtime_gates(2ULL * top_k,
                                     1.0F / static_cast<float>(top_k));
    const auto result = megamoe::transport::simulate_activation_transport(
        runtime_handles[source_rank], runtime_activations, runtime_gates,
        runtime_transport, 3);
    REQUIRE(result.audit.dispatch_packets ==
            descriptor.outgoing_packets.size());
    REQUIRE(result.audit.relayed_fanout == 2ULL * top_k);
    REQUIRE(result.audit.peak_ring_slots <=
            runtime_transport.ring_slots_per_source_channel);
    verify_mapping(runtime_handles);
  };
  verify_runtime_topology(3, 1);
  verify_runtime_topology(3, 4);
  verify_runtime_topology(4, 2);

  auto invalid = transport_config;
  invalid.ring_slots_per_source_channel = 0;
  expect_error(NotifyErrorCode::TransportProtocolFailure,
               [&] {
                 (void)megamoe::transport::simulate_activation_transport(
                     many_handles[0], many_activations, {0.25F, 0.75F},
                     invalid, 2);
               },
               {"phase=activation-transport",
                "field=ring_slots_per_source_channel"});
}

void test_topology_route_capacity_and_overflow() {
  auto bad = make_config(1, 6, 8, 2, 4);
  bad.topology.num_nodes = 4;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(bad, make_context(bad, 0),
                                                     {});
               },
               {"heterogeneous"});
  auto sparse = make_config(2, 2, 8, 2, 4);
  sparse.topology.rank_to_local_rank[1] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     sparse, make_context(sparse, 0), {});
               },
               {"local coordinate"});
  auto duplicate = make_config(2, 2, 8, 2, 4);
  duplicate.topology.rank_to_node[2] = 0;
  duplicate.topology.rank_to_local_rank[2] = 0;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     duplicate, make_context(duplicate, 0), {});
               },
               {"duplicate topology coordinate", "node=0", "local_rank=0"});
  auto bad_node = make_config(2, 2, 8, 2, 4);
  bad_node.topology.rank_to_node[0] = 2;
  expect_error(NotifyErrorCode::InvalidTopology,
               [&] {
                 megamoe::build_rank_local_histogram(
                     bad_node, make_context(bad_node, 0), {});
               },
               {"out-of-range node coordinate", "rank=0", "node=2"});
  auto topk = make_config(1, 1, 2, 3, 4);
  expect_error(NotifyErrorCode::InvalidRuntimeConfig,
               [&] {
                 megamoe::build_rank_local_histogram(topk,
                                                     make_context(topk, 0), {});
               },
               {"top_k exceeds"});

  auto config = make_config(1, 2, 4, 2, 4);
  const auto context = make_context(config, 0);
  expect_error(NotifyErrorCode::DuplicateExpert,
               [&] {
                 megamoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 2}});
               },
               {"rank=0", "topk_slot=1"});
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 megamoe::build_rank_local_histogram(config, context,
                                                     {1, {2, 4}});
               },
               {"actual=4"});
  expect_error(NotifyErrorCode::ArithmeticOverflow,
               [&] {
                 megamoe::build_rank_local_histogram(
                     config, context,
                     {std::numeric_limits<std::uint64_t>::max(), {}});
               },
               {"route count"});
  config.capacity.expert_rows = {1, 100, 100, 100};
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {0, 2}}};
  expect_error(NotifyErrorCode::ExpertCapacityExceeded,
               [&] {
                 megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "capacity", 1);
               },
               {"expert=0"});
  config.capacity.expert_rows.clear();
  config.capacity.workspace_rows_per_rank = {2, 100};
  expect_error(NotifyErrorCode::WorkspaceCapacityExceeded,
               [&] {
                 megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                            "workspace-cap", 1);
               },
               {"destination_rank=0", "actual_rows=3", "capacity=2"});
}
