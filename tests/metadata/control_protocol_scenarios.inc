void test_fixed_control_protocol_agreement() {
  const auto config = make_config(1, 2, 8, 3, 4);
  const auto context0 = make_context(config, 0, "control", 17, "control-exec-0");
  const auto context1 = make_context(config, 1, "control", 17, "control-exec-1");
  const auto record0 = megamoe::cuda::make_control_record(
      config, context0, 2, 6, 10);
  const auto record1 = megamoe::cuda::make_control_record(
      config, context1, 3, 9, 10);
  megamoe::cuda::validate_control_matrix({record0, record1});

  const auto expect_control_error = [&](const megamoe::cuda::ControlMatrix &rows,
                                        const std::vector<std::string> &fields) {
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] { megamoe::cuda::validate_control_matrix(rows); },
                 fields);
  };

  const auto max_mismatch = megamoe::cuda::make_control_record(
      config, context1, 3, 9, 11);
  expect_control_error({record0, max_mismatch},
                       {"phase=control-agreement", "field=max_tokens_per_rank",
                        "origin_rank=1", "expected=10", "actual=11"});

  auto expert_config = make_config(1, 2, 9, 3, 4);
  const auto expert_mismatch = megamoe::cuda::make_control_record(
      expert_config, context1, 3, 9, 10);
  expect_control_error({record0, expert_mismatch},
                       {"field=num_experts", "origin_rank=1"});

  megamoe::cuda::LocalPreflightStatus invalid_route;
  invalid_route.error = megamoe::cuda::ControlLocalError::ExpertOutOfRange;
  invalid_route.field = megamoe::cuda::ControlField::LocalValidation;
  invalid_route.token = 4;
  invalid_route.slot = 2;
  invalid_route.expert = 99;
  invalid_route.expected = 8;
  invalid_route.actual = 99;
  const auto bad_route = megamoe::cuda::make_control_record(
      config, context1, 5, 15, 10, invalid_route);
  expect_control_error({record0, bad_route},
                       {"field=local_validation", "origin_rank=1",
                        "token=4", "slot=2", "expert=99"});

  const auto over_capacity = megamoe::cuda::make_control_record(
      config, context1, 11, 33, 10);
  expect_control_error({record0, over_capacity},
                       {"field=num_tokens", "origin_rank=1",
                        "expected=10", "actual=11"});

  auto generation_mismatch = record1;
  ++generation_mismatch.generation;
  expect_control_error({record0, generation_mismatch},
                       {"field=generation", "origin_rank=1",
                        "expected=17", "actual=18"});

  auto session_mismatch = record1;
  ++session_mismatch.session_id_word1;
  expect_control_error({record0, session_mismatch},
                       {"field=session_id_word1", "origin_rank=1"});

  // Validation has no calling-rank context: the same gathered matrix always
  // produces an identical structured error.
  std::string first_error;
  std::string second_error;
  try {
    megamoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    first_error = error.what();
  }
  try {
    megamoe::cuda::validate_control_matrix({record0, generation_mismatch});
  } catch (const NotifyDispatchError &error) {
    second_error = error.what();
  }
  REQUIRE(!first_error.empty() && first_error == second_error);

  const auto wire0 = megamoe::cuda::encode_control_header(record0);
  REQUIRE(megamoe::cuda::decode_control_header(wire0).generation == 17);
  REQUIRE(static_cast<unsigned>(wire0[0]) == 0x43U &&
          static_cast<unsigned>(wire0[7]) == 0x4dU);
  auto corrupt_wire = wire0;
  corrupt_wire[80] ^= std::byte{1};
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)megamoe::cuda::decode_control_header(corrupt_wire); },
               {"field=checksum"});
  auto byteswapped = wire0;
  for (std::size_t word = 0; word < megamoe::cuda::kControlSchemaWords;
       ++word)
    std::reverse(byteswapped.begin() + static_cast<std::ptrdiff_t>(word * 8),
                 byteswapped.begin() +
                     static_cast<std::ptrdiff_t>((word + 1) * 8));
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] { (void)megamoe::cuda::decode_control_header(byteswapped); },
               {"phase=control-agreement"});

  const auto config0 = megamoe::cuda::encode_canonical_config(
      config, 10, context0.session().stable_identity_bytes());
  const auto decoded_config =
      megamoe::cuda::decode_canonical_config(config0);
  REQUIRE(decoded_config.rank_to_node == config.topology.rank_to_node);
  REQUIRE(decoded_config.expert_to_rank == config.topology.expert_to_rank);
  REQUIRE(decoded_config.stable_session_identity ==
          context0.session().stable_identity_bytes());
  REQUIRE(config0.size() >= megamoe::cuda::kCanonicalConfigHeaderBytes);
  const std::array<std::byte, 16> config_golden{
      std::byte{0x52}, std::byte{0x49}, std::byte{0x57}, std::byte{0x47},
      std::byte{0x46}, std::byte{0x43}, std::byte{0x4d}, std::byte{0x4d},
      std::byte{0x01}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00},
      std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}};
  REQUIRE(std::equal(config_golden.begin(), config_golden.end(),
                     config0.begin()));
  {
    auto byteswapped_config = config0;
    for (std::size_t word = 0;
         word * 8U < megamoe::cuda::kCanonicalConfigHeaderBytes; ++word)
      std::reverse(
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>(word * 8U),
          byteswapped_config.begin() +
              static_cast<std::ptrdiff_t>((word + 1U) * 8U));
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   (void)megamoe::cuda::decode_canonical_config(
                       byteswapped_config);
                 },
                 {"magic"});
  }

  // The envelope is self-validating even at world=1. Every corruption class
  // must fail before payload values can be used.
  auto world1_header = record0;
  world1_header.calling_rank = 0;
  world1_header.world_size = 1;
  world1_header.config_blob_bytes = config0.size();
  world1_header.config_digest = megamoe::cuda::config_wire_digest(config0);
  megamoe::cuda::validate_config_matrix({config0}, {world1_header});
  const auto corrupt_config = [&](std::size_t offset,
                                  const std::string &needle) {
    auto damaged = config0;
    damaged.at(offset) ^= std::byte{1};
    auto header = world1_header;
    header.config_digest = megamoe::cuda::config_wire_digest(damaged);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({damaged}, {header});
                 },
                 {needle});
  };
  corrupt_config(0, "magic");
  corrupt_config(8, "protocol_version");
  corrupt_config(16, "header_length");
  corrupt_config(24, "config_blob_bytes");
  corrupt_config(32, "reserved");
  corrupt_config(48, "checksum");
  corrupt_config(64, "checksum");
  {
    auto truncated = config0;
    truncated.pop_back();
    auto header = world1_header;
    header.config_blob_bytes = truncated.size();
    header.config_digest = megamoe::cuda::config_wire_digest(truncated);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({truncated}, {header});
                 },
                 {"truncated"});
  }
  {
    auto trailing = config0;
    trailing.push_back(std::byte{0});
    auto header = world1_header;
    header.config_blob_bytes = trailing.size();
    header.config_digest = megamoe::cuda::config_wire_digest(trailing);
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix({trailing}, {header});
                 },
                 {"trailing_bytes"});
  }
  {
    auto wrong_header_digest = world1_header;
    ++wrong_header_digest.config_digest;
    expect_error(NotifyErrorCode::ControlProtocolFailure,
                 [&] {
                   megamoe::cuda::validate_config_matrix(
                       {config0}, {wrong_header_digest});
                 },
                 {"field=config_digest", "control_header_vs_wire"});
  }
  auto config_changed = config;
  std::swap(config_changed.topology.expert_to_rank[0],
            config_changed.topology.expert_to_rank[1]);
  const auto config1 = megamoe::cuda::encode_canonical_config(
      config_changed, 10, context1.session().stable_identity_bytes());
  auto same_digest0 = record0;
  auto same_digest1 = record1;
  same_digest0.config_blob_bytes = config0.size();
  same_digest1.config_blob_bytes = config1.size();
  same_digest0.config_digest = megamoe::cuda::config_wire_digest(config0);
  same_digest1.config_digest = megamoe::cuda::config_wire_digest(config1);
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_config_matrix(
                     {config0, config1}, {same_digest0, same_digest1});
               },
               {"phase=config-agreement", "field=expert_to_rank",
                "origin_rank=1"});

  // A forged equal diagnostic digest cannot bypass per-rank integrity. Once
  // both diagnostic digests are correctly recomputed, exact decoded/byte
  // agreement still rejects the legitimate-but-different payload.
  auto forged_digest1 = same_digest1;
  forged_digest1.config_digest = same_digest0.config_digest;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_config_matrix(
                     {config0, config1}, {same_digest0, forged_digest1});
               },
               {"field=config_digest", "origin_rank=1"});

  // Original stable identity bytes, not their two header summaries or display
  // names, are the final session agreement boundary.
  const std::vector<std::byte> raw_identity0{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x30}};
  const std::vector<std::byte> raw_identity1{
      std::byte{0x10}, std::byte{0x20}, std::byte{0x31}};
  const auto raw_session0 = megamoe::make_collective_session(
      "same-display", raw_identity0, 2, 91);
  const auto raw_session1 = megamoe::make_collective_session(
      "same-display", raw_identity1, 2, 91);
  const auto raw_context0 = megamoe::make_rank_collective_context(
      raw_session0, 0, megamoe::make_execution_token("raw-exec-0"));
  const auto raw_context1 = megamoe::make_rank_collective_context(
      raw_session1, 1, megamoe::make_execution_token("raw-exec-1"));
  const auto raw_wire0 = megamoe::cuda::encode_canonical_config(
      config, 10, raw_identity0);
  const auto raw_wire1 = megamoe::cuda::encode_canonical_config(
      config, 10, raw_identity1);
  auto raw_header0 = megamoe::cuda::make_control_record(
      config, raw_context0, 1, 3, 10);
  auto raw_header1 = megamoe::cuda::make_control_record(
      config, raw_context1, 1, 3, 10);
  // Simulate an adversarial summary collision: fixed control passes, exact raw
  // identity comparison in config agreement still fails.
  raw_header1.session_id_word0 = raw_header0.session_id_word0;
  raw_header1.session_id_word1 = raw_header0.session_id_word1;
  megamoe::cuda::validate_control_matrix({raw_header0, raw_header1});
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_config_matrix(
                     {raw_wire0, raw_wire1}, {raw_header0, raw_header1});
               },
               {"field=session_identity", "origin_rank=1", "coordinate=2"});

  const auto config3 = make_config(1, 3, 9, 3, 4);
  const auto context30 = make_context(config3, 0, "common", 44, "e0");
  const auto context31 = make_context(config3, 1, "common", 44, "e1");
  const auto context32 = make_context(config3, 2, "common", 44, "e2");
  const auto row30 = megamoe::cuda::make_control_record(
      config3, context30, 1, 3, 10);
  auto row31 = megamoe::cuda::make_control_record(
      config3, context31, 1, 3, 10);
  auto row32 = megamoe::cuda::make_control_record(
      config3, context32, 1, 3, 10);
  ++row31.session_id_word1;
  ++row32.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=session_id_word1", "origin_rank=1"});

  // For one rank with several differences, fixed field order selects
  // generation before session and config fields.
  ++row31.generation;
  expect_error(NotifyErrorCode::ControlProtocolFailure,
               [&] {
                 megamoe::cuda::validate_control_matrix(
                     {row30, row31, row32});
               },
               {"field=generation", "origin_rank=1"});
}
