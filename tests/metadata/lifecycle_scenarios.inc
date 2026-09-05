void test_operation_move_and_terminal_state_machine() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "state", 1), {1, {0}});

  auto success = std::make_shared<BackendBehavior>();
  success->expert_rows = {{1, 0}};
  ScriptedBackend success_backend(success);
  auto original = megamoe::start_count_exchange(local, success_backend);
  CollectiveOperation moved(std::move(original));
  REQUIRE(!original);
  auto *self_alias = &moved;
  moved = std::move(*self_alias);
  REQUIRE(static_cast<bool>(moved));
  const auto validated = moved.wait();
  REQUIRE(validated.rows() == 1);
  REQUIRE(success->wait_calls == 1);
  REQUIRE(success->cancel_calls == 0);
  moved.cancel();
  moved.cancel();
  REQUIRE(success->cancel_calls == 0);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)moved.wait(); },
               {"phase=wait", "expected=pending", "actual=terminal"});
  REQUIRE(success->wait_calls == 1);

  auto cancelled = std::make_shared<BackendBehavior>();
  cancelled->expert_rows = {{1, 0}};
  ScriptedBackend cancelled_backend(cancelled);
  {
    auto operation = megamoe::start_count_exchange(local, cancelled_backend);
    operation.cancel();
    operation.cancel();
    REQUIRE(cancelled->cancel_calls == 1);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)operation.wait(); },
                 {"phase=wait", "actual=terminal"});
    REQUIRE(cancelled->wait_calls == 0);
  }
  REQUIRE(cancelled->cancel_calls == 1);
  REQUIRE(cancelled->operation_destructs == 1);

  auto destination = std::make_shared<BackendBehavior>();
  destination->expert_rows = {{1, 0}};
  ScriptedBackend destination_backend(destination);
  auto source = std::make_shared<BackendBehavior>();
  source->expert_rows = {{1, 0}};
  ScriptedBackend source_backend(source);
  {
    auto destination_operation =
        megamoe::start_count_exchange(local, destination_backend);
    auto source_operation =
        megamoe::start_count_exchange(local, source_backend);
    destination_operation = std::move(source_operation);
    REQUIRE(destination->cancel_calls == 1);
    REQUIRE(!source_operation);
    destination_operation.cancel();
    REQUIRE(source->cancel_calls == 1);
  }
  REQUIRE(destination->cancel_calls == 1);
  REQUIRE(source->cancel_calls == 1);
  REQUIRE(destination->operation_destructs == 1);
  REQUIRE(source->operation_destructs == 1);
}
void test_typed_storage_lifetimes_and_unwaited_destruction() {
  struct SessionState final : megamoe::BackendSessionState {};
  struct ExecutionState final : megamoe::BackendExecutionState {};
  const auto config = make_config(1, 1, 2, 1, 4);
  auto session_state = std::make_shared<SessionState>();
  auto execution_state = std::make_shared<ExecutionState>();
  const std::weak_ptr<SessionState> session_lifetime = session_state;
  const std::weak_ptr<ExecutionState> execution_lifetime = execution_state;
  auto session =
      megamoe::make_collective_session("lifetime", 1, 1, session_state);
  auto execution = megamoe::make_execution_token("exec", execution_state);
  auto context = megamoe::make_rank_collective_context(session, 0, execution);
  auto local = megamoe::build_rank_local_histogram(config, context, {1, {0}});
  auto behavior = std::make_shared<BackendBehavior>();
  behavior->expert_rows = {{1, 0}};
  ScriptedBackend backend(behavior);
  {
    auto operation = megamoe::start_count_exchange(local, backend);
    local = {};
    context = {};
    execution = {};
    session = {};
    session_state.reset();
    execution_state.reset();
    REQUIRE(!behavior->input_lifetime.expired());
    REQUIRE(!session_lifetime.expired());
    REQUIRE(!execution_lifetime.expired());
  }
  REQUIRE(behavior->cancel_calls == 1);
  REQUIRE(behavior->input_lifetime.expired());
  REQUIRE(session_lifetime.expired());
  REQUIRE(execution_lifetime.expired());

  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto throwing = std::make_shared<BackendBehavior>();
    throwing->expert_rows = {{1, 0}};
    throwing->cancel = mode;
    ScriptedBackend throwing_backend(throwing);
    const auto throwing_local = megamoe::build_rank_local_histogram(
        config, make_context(config, 0, "destructor", 2), {1, {0}});
    {
      auto operation =
          megamoe::start_count_exchange(throwing_local, throwing_backend);
      (void)operation;
    }
    REQUIRE(throwing->cancel_calls == 1);
    REQUIRE(throwing->operation_destructs == 1);
  }

  auto retained_session_state = std::make_shared<SessionState>();
  auto retained_execution_state = std::make_shared<ExecutionState>();
  const std::weak_ptr<SessionState> retained_session = retained_session_state;
  const std::weak_ptr<ExecutionState> retained_execution =
      retained_execution_state;
  auto retained_context = megamoe::make_rank_collective_context(
      megamoe::make_collective_session("validated-life", 1, 3,
                                       retained_session_state),
      0,
      megamoe::make_execution_token("validated-exec",
                                    retained_execution_state));
  auto retained_local =
      megamoe::build_rank_local_histogram(config, retained_context, {1, {0}});
  auto retained_behavior = std::make_shared<BackendBehavior>();
  retained_behavior->expert_rows = {{1, 0}};
  ScriptedBackend retained_backend(retained_behavior);
  auto retained_operation =
      megamoe::start_count_exchange(retained_local, retained_backend);
  auto validated = retained_operation.wait();
  ValidatedCollectiveHandle validated_copy = validated;
  ValidatedCollectiveHandle validated_move = std::move(validated_copy);
  retained_session_state.reset();
  retained_execution_state.reset();
  retained_context = {};
  retained_local = {};
  retained_operation = {};
  REQUIRE(!retained_behavior->input_lifetime.expired());
  REQUIRE(!retained_session.expired());
  REQUIRE(!retained_execution.expired());
  REQUIRE(retained_behavior->wait_calls == 1);
  REQUIRE(retained_behavior->cancel_calls == 0);
  REQUIRE(retained_behavior->operation_destructs == 0);
  REQUIRE(retained_behavior->completion_lease_destructs == 0);
  REQUIRE(validated_move.rows() == 1);
  validated = {};
  REQUIRE(retained_behavior->operation_destructs == 0);
  REQUIRE(retained_behavior->completion_lease_destructs == 0);
  validated_move = {};
  REQUIRE(retained_behavior->input_lifetime.expired());
  REQUIRE(retained_session.expired());
  REQUIRE(retained_execution.expired());
  REQUIRE(retained_behavior->wait_calls == 1);
  REQUIRE(retained_behavior->cancel_calls == 0);
  REQUIRE(retained_behavior->operation_destructs == 1);
  REQUIRE(retained_behavior->completion_lease_destructs == 1);
}

void test_backend_exception_boundaries_and_device_rejection() {
  const auto config = make_config(1, 1, 2, 1, 4);
  const auto local = megamoe::build_rank_local_histogram(
      config, make_context(config, 0, "errors", 3), {1, {0}});
  LocalHistogramHandle empty_local;
  auto unused_behavior = std::make_shared<BackendBehavior>();
  ScriptedBackend unused_backend(unused_behavior);
  expect_error(
      NotifyErrorCode::CountExchangeFailure,
      [&] { (void)megamoe::start_count_exchange(empty_local, unused_backend); },
      {"phase=start", "actual=empty/default"});
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->start = mode;
    ScriptedBackend backend(behavior);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)megamoe::start_count_exchange(local, backend); },
                 {"phase=start", "kind=expert_counts", "rank=0", "errors",
                  "generation=3", "backend_error"});
  }
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->wait = mode;
    behavior->expert_rows = {{1, 0}};
    ScriptedBackend backend(behavior);
    auto operation = megamoe::start_count_exchange(local, backend);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { (void)operation.wait(); },
                 {"phase=wait", "backend_error"});
    REQUIRE(behavior->wait_calls == 1);
    REQUIRE(behavior->cancel_calls == 1);
    operation = {};
    REQUIRE(behavior->operation_destructs == 1);
    REQUIRE(behavior->cancel_calls == 1);
  }
  for (const auto mode :
       {ThrowMode::Notify, ThrowMode::Standard, ThrowMode::Unknown}) {
    auto behavior = std::make_shared<BackendBehavior>();
    behavior->cancel = mode;
    behavior->expert_rows = {{1, 0}};
    ScriptedBackend backend(behavior);
    auto operation = megamoe::start_count_exchange(local, backend);
    expect_error(NotifyErrorCode::CountExchangeFailure,
                 [&] { operation.cancel(); },
                 {"phase=cancel", "backend_error"});
    REQUIRE(behavior->cancel_calls == 1);
    operation = {};
    REQUIRE(behavior->operation_destructs == 1);
    REQUIRE(behavior->cancel_calls == 1);
  }
  auto device_behavior = std::make_shared<BackendBehavior>();
  device_behavior->expert_rows = {{1, 0}};
  device_behavior->return_device = true;
  ScriptedBackend device_backend(device_behavior);
  auto device_op = megamoe::start_count_exchange(local, device_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)device_op.wait(); },
               {"typed host matrix", "opaque device lease"});

  auto kind_behavior = std::make_shared<BackendBehavior>();
  kind_behavior->expert_rows = {{1, 0}};
  kind_behavior->override_kind = true;
  kind_behavior->returned_kind = ExchangeKind::NotifyPacketSummary;
  ScriptedBackend kind_backend(kind_behavior);
  auto kind_op = megamoe::start_count_exchange(local, kind_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)kind_op.wait(); },
               {"collective kind mismatch", "expected=expert_counts",
                "actual=notify_packet_summary"});

  auto shape_behavior = std::make_shared<BackendBehavior>();
  shape_behavior->expert_rows = {};
  ScriptedBackend shape_backend(shape_behavior);
  auto shape_op = megamoe::start_count_exchange(local, shape_backend);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] { (void)shape_op.wait(); },
               {"field=matrix.rows", "expected=1", "actual=0"});

  const auto valid_device = megamoe::make_device_buffer_lease(
      2, 3, 8, std::make_shared<TestDeviceLease>());
  REQUIRE(valid_device.rows() == 2);
  REQUIRE(valid_device.row_width() == 3);
  REQUIRE(valid_device.element_count() == 6);
  REQUIRE(valid_device.capacity_elements() == 8);
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     0, 3, 3, std::make_shared<TestDeviceLease>());
               },
               {"phase=device-lease-validation", "field=rows", "expected=>0",
                "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     1, 0, 1, std::make_shared<TestDeviceLease>());
               },
               {"field=row_width", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     2, 4, 7, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>=8", "actual=7"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     1, 1, 0, std::make_shared<TestDeviceLease>());
               },
               {"field=capacity_elements", "expected=>0", "actual=0"});
  expect_error(NotifyErrorCode::CountExchangeFailure,
               [&] {
                 (void)megamoe::make_device_buffer_lease(
                     2, std::numeric_limits<std::uint64_t>::max(),
                     std::numeric_limits<std::uint64_t>::max(),
                     std::make_shared<TestDeviceLease>());
               },
               {"field=element_count.uint64", "actual=overflow"});
}
