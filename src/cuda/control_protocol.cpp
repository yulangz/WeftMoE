#include "megamoe/cuda/control_protocol.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>

#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
namespace {

class Hash64 {
public:
  void add(std::uint64_t value) noexcept {
    for (unsigned byte = 0; byte < 8; ++byte) {
      value_ ^= static_cast<std::uint8_t>(value >> (byte * 8U));
      value_ *= 1099511628211ULL;
    }
  }
  void add(const std::string &value) noexcept {
    for (const auto byte : value) {
      value_ ^= static_cast<std::uint8_t>(byte);
      value_ *= 1099511628211ULL;
    }
  }
  template <typename T> void add_vector(const std::vector<T> &values) noexcept {
    add(values.size());
    for (const auto value : values)
      add(static_cast<std::uint64_t>(value));
  }
  std::uint64_t value() const noexcept { return value_; }

private:
  std::uint64_t value_ = 1469598103934665603ULL;
};

std::uint64_t topology_fingerprint(const RuntimeConfig &config) noexcept {
  Hash64 hash;
  hash.add(config.topology.world_size);
  hash.add(config.topology.num_nodes);
  hash.add_vector(config.topology.rank_to_node);
  hash.add_vector(config.topology.rank_to_local_rank);
  hash.add_vector(config.topology.expert_to_rank);
  return hash.value();
}

std::uint64_t config_fingerprint(const RuntimeConfig &config,
                                 std::uint64_t max_tokens) noexcept {
  Hash64 hash;
  hash.add(kControlProtocolVersion);
  hash.add(kControlDescriptorSchema);
  hash.add(config.num_experts);
  hash.add(config.top_k);
  hash.add(config.m_tile_rows);
  hash.add(topology_fingerprint(config));
  hash.add(max_tokens);
  hash.add(1);  // control dtype: uint64
  hash.add(1);  // count dtype: uint64
  hash.add(64); // route index bits
  hash.add(32); // expert index bits
  hash.add_vector(config.capacity.expert_rows);
  hash.add_vector(config.capacity.workspace_rows_per_rank);
  return hash.value();
}

std::uint64_t record_checksum(const ControlRecord &record) noexcept {
  Hash64 hash;
  std::array<std::uint64_t, kControlSchemaWords> words{};
  std::memcpy(words.data(), &record, sizeof(record));
  for (std::size_t index = 0; index + 1 < kControlSchemaWords; ++index)
    hash.add(words[index]);
  return hash.value();
}

void set_error(ControlRecord &record, const LocalPreflightStatus &status) {
  if (record.local_error_code != 0 || status.error == ControlLocalError::None)
    return;
  record.local_error_code = static_cast<std::uint64_t>(status.error);
  record.local_error_field = static_cast<std::uint64_t>(status.field);
  record.error_token = status.token;
  record.error_slot = status.slot;
  std::memcpy(&record.error_expert, &status.expert, sizeof(status.expert));
  record.error_expected = status.expected;
  record.error_actual = status.actual;
}

[[noreturn]] void agreement_failure(
    ControlField field, std::uint32_t origin_rank, std::uint64_t expected,
    std::uint64_t actual, const ControlRecord &record,
    const char *detail = "collective mismatch") {
  std::int64_t expert = 0;
  std::memcpy(&expert, &record.error_expert, sizeof(expert));
  std::ostringstream message;
  message << "phase=control-agreement, field=" << control_field_name(field)
          << ", origin_rank=" << origin_rank << ", expected=" << expected
          << ", actual=" << actual << ", error_code="
          << control_error_name(static_cast<ControlLocalError>(
                 record.local_error_code))
          << ", token=";
  if (record.error_token == UINT64_MAX)
    message << "n/a";
  else
    message << record.error_token;
  message << ", slot=";
  if (record.error_slot == UINT64_MAX)
    message << "n/a";
  else
    message << record.error_slot;
  message << ", expert=" << expert << ", detail=" << detail;
  throw NotifyDispatchError(NotifyErrorCode::ControlProtocolFailure,
                            message.str());
}

} // namespace

const char *control_field_name(ControlField field) noexcept {
  switch (field) {
  case ControlField::None: return "none";
  case ControlField::Magic: return "magic";
  case ControlField::ProtocolVersion: return "protocol_version";
  case ControlField::SchemaWords: return "schema_words";
  case ControlField::Generation: return "generation";
  case ControlField::CallingRank: return "calling_rank";
  case ControlField::WorldSize: return "world_size";
  case ControlField::SessionIdentity: return "session_identity";
  case ControlField::TopologyFingerprint: return "topology_fingerprint";
  case ControlField::NumExperts: return "num_experts";
  case ControlField::TopK: return "top_k";
  case ControlField::MTileRows: return "m_tile_rows";
  case ControlField::NumNodes: return "num_nodes";
  case ControlField::MaxTokensPerRank: return "max_tokens_per_rank";
  case ControlField::ControlDtype: return "control_dtype";
  case ControlField::CountDtype: return "count_dtype";
  case ControlField::RouteIndexBits: return "route_index_bits";
  case ControlField::ExpertIndexBits: return "expert_index_bits";
  case ControlField::ExpertRowWidth: return "expert_row_width";
  case ControlField::SummaryRowWidth: return "summary_row_width";
  case ControlField::DescriptorSchema: return "descriptor_schema";
  case ControlField::ConfigFingerprint: return "config_fingerprint";
  case ControlField::NumTokens: return "num_tokens";
  case ControlField::RouteCount: return "route_count";
  case ControlField::LocalValidation: return "local_validation";
  case ControlField::Reserved: return "reserved";
  case ControlField::Checksum: return "checksum";
  }
  return "unknown";
}

const char *control_error_name(ControlLocalError error) noexcept {
  switch (error) {
  case ControlLocalError::None: return "none";
  case ControlLocalError::InvalidRuntimeConfig: return "invalid_runtime_config";
  case ControlLocalError::InvalidRouteShape: return "invalid_route_shape";
  case ControlLocalError::CapacityExceeded: return "capacity_exceeded";
  case ControlLocalError::ArithmeticOverflow: return "arithmetic_overflow";
  case ControlLocalError::ExpertOutOfRange: return "expert_out_of_range";
  case ControlLocalError::DuplicateExpert: return "duplicate_expert";
  }
  return "unknown";
}

ControlRecord make_control_record(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    std::uint64_t num_tokens, std::uint64_t actual_route_elements,
    std::uint64_t max_tokens_per_rank, LocalPreflightStatus device_status) {
  if (!context)
    throw NotifyDispatchError(
        NotifyErrorCode::ControlProtocolFailure,
        "phase=control-record, field=context, origin_rank=n/a, expected=valid, actual=empty");
  ControlRecord record;
  record.generation = context.session().generation();
  record.calling_rank = context.calling_rank();
  record.world_size = context.session().world_size();
  Hash64 session_hash;
  session_hash.add(context.session().identity());
  record.session_fingerprint = session_hash.value();
  record.topology_fingerprint = topology_fingerprint(config);
  record.config_fingerprint = config_fingerprint(config, max_tokens_per_rank);
  record.num_experts = config.num_experts;
  record.top_k = config.top_k;
  record.m_tile_rows = config.m_tile_rows;
  record.num_nodes = config.topology.num_nodes;
  record.max_tokens_per_rank = max_tokens_per_rank;
  record.num_tokens = num_tokens;
  record.expert_row_width = config.num_experts;
  record.summary_row_width = config.topology.num_nodes;

  bool route_overflow = config.top_k != 0 &&
                        num_tokens > std::numeric_limits<std::uint64_t>::max() /
                                         config.top_k;
  record.route_count = route_overflow
                           ? UINT64_MAX
                           : num_tokens * static_cast<std::uint64_t>(config.top_k);
  try {
    validate_runtime_config(config);
  } catch (const NotifyDispatchError &) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::InvalidRuntimeConfig,
                          ControlField::LocalValidation, UINT64_MAX,
                          UINT64_MAX, 0, 1, 0});
  }
  if (config.topology.world_size != context.session().world_size()) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::InvalidRuntimeConfig,
                          ControlField::WorldSize, UINT64_MAX, UINT64_MAX, 0,
                          context.session().world_size(),
                          config.topology.world_size});
  } else if (context.calling_rank() >= config.topology.world_size) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::InvalidRuntimeConfig,
                          ControlField::CallingRank, UINT64_MAX, UINT64_MAX, 0,
                          config.topology.world_size,
                          context.calling_rank()});
  }
  if (route_overflow) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::ArithmeticOverflow,
                          ControlField::RouteCount, UINT64_MAX, UINT64_MAX, 0,
                          std::numeric_limits<std::uint64_t>::max(),
                          UINT64_MAX});
  } else if (actual_route_elements != record.route_count) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::InvalidRouteShape,
                          ControlField::RouteCount, UINT64_MAX, UINT64_MAX, 0,
                          record.route_count, actual_route_elements});
  } else if (num_tokens > max_tokens_per_rank) {
    set_error(record, LocalPreflightStatus{
                          ControlLocalError::CapacityExceeded,
                          ControlField::NumTokens, UINT64_MAX, UINT64_MAX, 0,
                          max_tokens_per_rank, num_tokens});
  }
  set_error(record, device_status);
  record.checksum = record_checksum(record);
  return record;
}

void validate_control_matrix(const ControlMatrix &matrix,
                             const RankCollectiveContext &context) {
  if (!context)
    throw NotifyDispatchError(NotifyErrorCode::ControlProtocolFailure,
                              "phase=control-agreement, field=context, origin_rank=n/a, expected=valid, actual=empty");
  const auto world = context.session().world_size();
  if (matrix.size() != world) {
    ControlRecord empty;
    agreement_failure(ControlField::WorldSize, 0, world, matrix.size(), empty,
                      "control matrix row count");
  }
  Hash64 session_hash;
  session_hash.add(context.session().identity());
  const auto expected_session = session_hash.value();
  const auto &baseline = matrix.front();
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    const auto &record = matrix[rank];
    const auto check = [&](ControlField field, std::uint64_t expected,
                           std::uint64_t actual) {
      if (expected != actual)
        agreement_failure(field, rank, expected, actual, record);
    };
    check(ControlField::Magic, kControlProtocolMagic, record.magic);
    check(ControlField::ProtocolVersion, kControlProtocolVersion,
          record.protocol_version);
    check(ControlField::SchemaWords, kControlSchemaWords, record.schema_words);
    check(ControlField::Generation, context.session().generation(),
          record.generation);
    check(ControlField::CallingRank, rank, record.calling_rank);
    check(ControlField::WorldSize, world, record.world_size);
    check(ControlField::SessionIdentity, expected_session,
          record.session_fingerprint);
    check(ControlField::ControlDtype, 1, record.control_dtype);
    check(ControlField::CountDtype, 1, record.count_dtype);
    check(ControlField::RouteIndexBits, 64, record.route_index_bits);
    check(ControlField::ExpertIndexBits, 32, record.expert_index_bits);
    check(ControlField::DescriptorSchema, kControlDescriptorSchema,
          record.descriptor_schema);
    check(ControlField::Reserved, 0, record.reserved);
    check(ControlField::Checksum, record_checksum(record), record.checksum);
    if (rank != 0) {
      check(ControlField::NumExperts, baseline.num_experts,
            record.num_experts);
      check(ControlField::TopK, baseline.top_k, record.top_k);
      check(ControlField::MTileRows, baseline.m_tile_rows,
            record.m_tile_rows);
      check(ControlField::NumNodes, baseline.num_nodes, record.num_nodes);
      check(ControlField::MaxTokensPerRank,
            baseline.max_tokens_per_rank, record.max_tokens_per_rank);
      check(ControlField::ExpertRowWidth, baseline.expert_row_width,
            record.expert_row_width);
      check(ControlField::SummaryRowWidth, baseline.summary_row_width,
            record.summary_row_width);
      check(ControlField::TopologyFingerprint,
            baseline.topology_fingerprint, record.topology_fingerprint);
      check(ControlField::ConfigFingerprint, baseline.config_fingerprint,
            record.config_fingerprint);
    }
    if (record.local_error_code != 0) {
      const auto field = static_cast<ControlField>(record.local_error_field);
      agreement_failure(field == ControlField::None
                            ? ControlField::LocalValidation
                            : field,
                        rank, record.error_expected, record.error_actual,
                        record, "rank-local preflight failure");
    }
    if (record.num_tokens > record.max_tokens_per_rank)
      agreement_failure(ControlField::NumTokens, rank,
                        record.max_tokens_per_rank, record.num_tokens, record,
                        "tokens exceed collective capacity");
    if (record.top_k != 0 &&
        record.num_tokens <= UINT64_MAX / record.top_k) {
      check(ControlField::RouteCount, record.num_tokens * record.top_k,
            record.route_count);
    }
  }
}

} // namespace megamoe::cuda
