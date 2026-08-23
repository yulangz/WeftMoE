#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "megamoe/config/runtime_config.hpp"
#include "megamoe/transport/count_exchange.hpp"

namespace megamoe::cuda {

inline constexpr std::uint64_t kControlProtocolMagic = 0x4d4547414d4f4543ULL;
inline constexpr std::uint64_t kControlProtocolVersion = 1;
inline constexpr std::uint64_t kControlSchemaWords = 32;
inline constexpr std::uint64_t kControlDescriptorSchema = 1;

enum class ControlLocalError : std::uint64_t {
  None = 0,
  InvalidRuntimeConfig = 1,
  InvalidRouteShape = 2,
  CapacityExceeded = 3,
  ArithmeticOverflow = 4,
  ExpertOutOfRange = 5,
  DuplicateExpert = 6,
};

enum class ControlField : std::uint64_t {
  None = 0,
  Magic = 1,
  ProtocolVersion = 2,
  SchemaWords = 3,
  Generation = 4,
  CallingRank = 5,
  WorldSize = 6,
  SessionIdentity = 7,
  TopologyFingerprint = 8,
  NumExperts = 9,
  TopK = 10,
  MTileRows = 11,
  NumNodes = 12,
  MaxTokensPerRank = 13,
  ControlDtype = 14,
  CountDtype = 15,
  RouteIndexBits = 16,
  ExpertIndexBits = 17,
  ExpertRowWidth = 18,
  SummaryRowWidth = 19,
  DescriptorSchema = 20,
  ConfigFingerprint = 21,
  NumTokens = 22,
  RouteCount = 23,
  LocalValidation = 24,
  Reserved = 25,
  Checksum = 26,
};

struct LocalPreflightStatus {
  ControlLocalError error = ControlLocalError::None;
  ControlField field = ControlField::None;
  std::uint64_t token = UINT64_MAX;
  std::uint64_t slot = UINT64_MAX;
  std::int64_t expert = 0;
  std::uint64_t expected = 0;
  std::uint64_t actual = 0;
};

// Fixed-width wire schema. Every field is uint64_t so an AllGather always uses
// exactly kControlSchemaWords ncclUint64 elements, independent of local config.
struct ControlRecord {
  std::uint64_t magic = kControlProtocolMagic;
  std::uint64_t protocol_version = kControlProtocolVersion;
  std::uint64_t schema_words = kControlSchemaWords;
  std::uint64_t generation = 0;
  std::uint64_t calling_rank = 0;
  std::uint64_t world_size = 0;
  std::uint64_t session_fingerprint = 0;
  std::uint64_t topology_fingerprint = 0;
  std::uint64_t config_fingerprint = 0;
  std::uint64_t num_experts = 0;
  std::uint64_t top_k = 0;
  std::uint64_t m_tile_rows = 0;
  std::uint64_t num_nodes = 0;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t num_tokens = 0;
  std::uint64_t route_count = 0;
  std::uint64_t control_dtype = 1; // uint64
  std::uint64_t count_dtype = 1;   // uint64
  std::uint64_t route_index_bits = 64;
  std::uint64_t expert_index_bits = 32;
  std::uint64_t expert_row_width = 0;
  std::uint64_t summary_row_width = 0;
  std::uint64_t descriptor_schema = kControlDescriptorSchema;
  std::uint64_t local_error_code = 0;
  std::uint64_t local_error_field = 0;
  std::uint64_t error_token = UINT64_MAX;
  std::uint64_t error_slot = UINT64_MAX;
  std::uint64_t error_expert = 0;
  std::uint64_t error_expected = 0;
  std::uint64_t error_actual = 0;
  std::uint64_t reserved = 0;
  std::uint64_t checksum = 0;
};

static_assert(sizeof(ControlRecord) ==
                  kControlSchemaWords * sizeof(std::uint64_t),
              "ControlRecord wire layout must remain fixed");
static_assert(alignof(ControlRecord) == alignof(std::uint64_t),
              "ControlRecord wire alignment must remain uint64_t");
static_assert(std::is_trivially_copyable_v<ControlRecord>,
              "ControlRecord must be safe to copy from the fixed wire image");

using ControlMatrix = std::vector<ControlRecord>;

ControlRecord make_control_record(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    std::uint64_t num_tokens, std::uint64_t actual_route_elements,
    std::uint64_t max_tokens_per_rank,
    LocalPreflightStatus device_status = {});

// Returns normally only when every row is structurally valid, has no local
// error, and agrees on every collective-layout field. Otherwise every rank
// deterministically throws the same ControlProtocolFailure.
void validate_control_matrix(const ControlMatrix &matrix,
                             const RankCollectiveContext &context);

const char *control_field_name(ControlField field) noexcept;
const char *control_error_name(ControlLocalError error) noexcept;

} // namespace megamoe::cuda
