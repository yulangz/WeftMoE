void test_context_kind_and_provenance_mismatch_messages() {
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)megamoe::make_collective_session("", 1, 1); },
               {"identity", "actual=empty"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)megamoe::make_collective_session("bad", 1, 0); },
               {"generation", "actual=0"});

  const auto fake_config = make_config(1, 2, 4, 1, 4);
  const auto fake_session_a =
      megamoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_session_b =
      megamoe::make_collective_session("fake-cross", 2, 1);
  const auto fake_local_a = megamoe::build_rank_local_histogram(
      fake_config,
      megamoe::make_rank_collective_context(
          fake_session_a, 0, megamoe::make_execution_token("fake-a")),
      {});
  const auto fake_local_b = megamoe::build_rank_local_histogram(
      fake_config,
      megamoe::make_rank_collective_context(
          fake_session_b, 1, megamoe::make_execution_token("fake-b")),
      {});
  HostFakeCountExchange cross_backend(2, "fake-cross");
  auto first_fake_operation =
      megamoe::start_count_exchange(fake_local_a, cross_backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::start_count_exchange(fake_local_b, cross_backend); },
      {"phase=start", "cross-session mix", "different session instance"});
  first_fake_operation.cancel();

  const auto config = make_config(1, 2, 4, 2, 4);
  const auto session = megamoe::make_collective_session("base", 2, 4);
  const auto execution0 = megamoe::make_execution_token("exec0");
  const auto context0 =
      megamoe::make_rank_collective_context(session, 0, execution0);
  const auto local0 =
      megamoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto count0 = complete_scripted(local0, ExchangeKind::ExpertCounts,
                                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  const auto summary0 =
      complete_scripted(local0, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  (void)megamoe::materialize_rank_metadata(local0, count0, summary0);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::materialize_rank_metadata(local0, count0, count0); },
      {"field=collective.kind", "expected=notify_packet_summary",
       "actual=expert_counts"});

  const auto local_rank1 = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(
          session, 1, megamoe::make_execution_token("exec1")),
      {});
  const auto count_rank1 = complete_scripted(
      local_rank1, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, count_rank1,
                                                          summary0);
               },
               {"field=calling_rank", "expected=0", "actual=1"});

  const auto other_execution = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(
          session, 0, megamoe::make_execution_token("other-exec")),
      {1, {0, 1}});
  const auto other_exec_count =
      complete_scripted(other_execution, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, other_exec_count, summary0);
               },
               {"field=execution.provenance", "exec0", "other-exec"});

  const auto generation_context = make_context(config, 0, "base", 5, "exec0");
  const auto generation_local = megamoe::build_rank_local_histogram(
      config, generation_context, {1, {0, 1}});
  const auto generation_count =
      complete_scripted(generation_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, generation_count, summary0);
               },
               {"field=session.generation", "expected=4", "actual=5"});

  const auto session_local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "other-session", 4, "exec0"),
      {1, {0, 1}});
  const auto session_count = complete_scripted(
      session_local, ExchangeKind::ExpertCounts, {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, session_count,
                                                          summary0);
               },
               {"field=session.identity", "base", "other-session"});

  const auto same_fields_local =
      megamoe::build_rank_local_histogram(config, context0, {1, {0, 1}});
  const auto same_fields_count =
      complete_scripted(same_fields_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, same_fields_count, summary0);
               },
               {"field=local.provenance"});

  const auto same_values_session =
      megamoe::make_collective_session("base", 2, 4);
  const auto other_session_instance_local = megamoe::build_rank_local_histogram(
      config,
      megamoe::make_rank_collective_context(same_values_session, 0, execution0),
      {1, {0, 1}});
  const auto other_session_instance_count = complete_scripted(
      other_session_instance_local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local0, other_session_instance_count, summary0);
               },
               {"field=session.provenance", "different session instance"});

  const auto world3 = make_config(1, 3, 4, 2, 4);
  const auto world3_local = megamoe::build_rank_local_histogram(
      world3, make_context(world3, 0, "base", 4, "exec0"), {1, {0, 1}});
  const auto world3_count =
      complete_scripted(world3_local, ExchangeKind::ExpertCounts,
                        {{1, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local0, world3_count,
                                                          summary0);
               },
               {"field=world_size", "expected=2", "actual=3"});
}
void test_malicious_completion_context_and_corruption_hooks() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "malicious", 2), {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  behavior->override_context = true;
  behavior->returned_context = megamoe::make_rank_collective_context(
      local.context().session(), 0, megamoe::make_execution_token("other"));
  ScriptedBackend backend(behavior);
  auto operation = megamoe::start_count_exchange(local, backend);
  expect_error(
      NotifyErrorCode::CountExchangeFailure, [&] { (void)operation.wait(); },
      {"execution token mismatch", "expected='exec'", "actual='other'"});

  const auto count =
      complete_scripted(local, ExchangeKind::ExpertCounts, {{1, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}});
  const auto bad_expert =
      megamoe::testing::NotifyCorruptionHook::with_expert_id(local, 0, 99);
  expect_error(NotifyErrorCode::ExpertOutOfRange,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_expert, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.expert_id", "rank=0",
                "expected=[0,2)", "actual=99"});
  const auto truncated =
      megamoe::testing::NotifyCorruptionHook::with_truncated_expert_ids(local);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(truncated, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=expert_id", "rank=0",
                "expected=1", "actual=0"});

  const auto bad_offset =
      megamoe::testing::NotifyCorruptionHook::with_local_offset(local, 0, 7);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_offset, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.local_offset",
                "rank=0", "expected=0", "actual=7"});
  const auto bad_combine =
      megamoe::testing::NotifyCorruptionHook::with_combine_order(local, 0, 9);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_combine, count,
                                                          summary);
               },
               {"phase=materialize-local", "field=outgoing.combine_order",
                "rank=0", "actual=9"});
  const auto bad_mask =
      megamoe::testing::NotifyCorruptionHook::with_node_mask(local, 0, 0);
  expect_error(
      NotifyErrorCode::InvalidRouteShape,
      [&] {
        (void)megamoe::materialize_rank_metadata(bad_mask, count, summary);
      },
      {"phase=materialize-local", "field=outgoing.destination_node_mask",
       "rank=0", "expected=recomputed node mask", "actual=content mismatch"});
  const auto bad_local_count =
      megamoe::testing::NotifyCorruptionHook::with_local_expert_count(local, 0,
                                                                      8);
  expect_error(NotifyErrorCode::InvalidRouteShape,
               [&] {
                 (void)megamoe::materialize_rank_metadata(bad_local_count,
                                                          count, summary);
               },
               {"phase=materialize-local", "field=local_expert_count", "rank=0",
                "actual=content mismatch"});

  const auto bad_matrix_shape =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count, {});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local, bad_matrix_shape, summary);
               },
               {"phase=materialize", "field=matrix.rows", "rank=0",
                "expected=1", "actual=0"});
  const auto bad_matrix_width =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{1}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(
                     local, bad_matrix_width, summary);
               },
               {"phase=materialize", "field=matrix.row_width", "rank=0",
                "expected=2", "actual=1"});
  const auto bad_matrix_row =
      megamoe::testing::NotifyCorruptionHook::with_collective_matrix(count,
                                                                     {{0, 0}});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::materialize_rank_metadata(local, bad_matrix_row,
                                                          summary);
               },
               {"phase=materialize", "field=expert_counts.calling_rank_row",
                "rank=0", "expected=local histogram count row",
                "actual=content mismatch"});

  const auto handle = megamoe::materialize_rank_metadata(local, count, summary);
  const auto validate_corrupt = [&](const DispatchHandle &corrupt,
                                    const std::string &field) {
    expect_error(NotifyErrorCode::InvalidHandle,
                 [&] { (void)megamoe::make_workspace_plan(corrupt); },
                 {"phase=dispatch-validation", "field=" + field, "rank=0",
                  "expected=", "actual="});
  };
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_expert_row_count(handle, 0,
                                                                    9),
      "incoming.expert_row_count");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_expert_row_prefix(handle, 1,
                                                                     9),
      "incoming.expert_row_prefix");
  validate_corrupt(megamoe::testing::NotifyCorruptionHook::
                       with_packet_identity_destination_node(handle, 0, 9),
                   "transport.packet.identity");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_packet_proxy(handle, 0, 9),
      "transport.packet.proxy_rank");
  validate_corrupt(megamoe::testing::NotifyCorruptionHook::with_fanout_expert(
                       handle, 0, 0, 9),
                   "transport.fanout.expert_id");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_local_rank(handle, 0,
                                                                     0, 9),
      "transport.fanout.destination_local_rank");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_row(handle, 0, 0, 9),
      "transport.fanout.dst_row");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_fanout_slot(handle, 0, 0, 9),
      "transport.fanout.topk_slot");
  validate_corrupt(
      megamoe::testing::NotifyCorruptionHook::with_mtile_row(handle, 0, 0, 9),
      "incoming.mtile_expected_rows");
}

void test_handle_copy_move_workspace_and_overflow() {
  const auto config = make_config(1, 2, 4, 2, 4);
  const std::vector<RankRoutingInput> inputs{{1, {0, 1}}, {1, {2, 3}}};
  const auto handles = megamoe::build_notify_dispatch_for_testing(
      config, inputs, "workspace", 5);
  DispatchHandle empty;
  expect_error(NotifyErrorCode::InvalidHandle, [&] { (void)empty.metadata(); });
  const auto plan = megamoe::make_workspace_plan(handles[0]);
  megamoe::validate_workspace_plan(handles[0], plan);
  (void)megamoe::ForwardPersistentInputs::checked(handles[0], plan);
  megamoe::WorkspacePlan empty_plan;
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.requirements(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.shape(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.identity(); }, {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { (void)empty_plan.epoch(); }, {"default/invalid"});
  expect_error(
      NotifyErrorCode::WorkspaceMismatch,
      [&] { megamoe::validate_workspace_plan(handles[0], empty_plan); },
      {"default/invalid"});
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { megamoe::validate_workspace_plan(handles[1], plan); },
               {"another DispatchHandle"});
  const auto other = megamoe::build_notify_dispatch_for_testing(config, inputs,
                                                                "workspace", 5);
  expect_error(NotifyErrorCode::WorkspaceMismatch,
               [&] { megamoe::validate_workspace_plan(other[0], plan); },
               {"another DispatchHandle"});

  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "overflow", 1), {1, {0, 1}});
  const auto counts = complete_scripted(
      local, ExchangeKind::ExpertCounts,
      {{1, 1, 0, 0}, {std::numeric_limits<std::uint64_t>::max(), 0, 0, 0}});
  const auto summary =
      complete_scripted(local, ExchangeKind::NotifyPacketSummary, {{1}, {0}});
  expect_error(
      NotifyErrorCode::ArithmeticOverflow,
      [&] { (void)megamoe::materialize_rank_metadata(local, counts, summary); },
      {"source prefix"});
}
