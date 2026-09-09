#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "weftmoe/version.hpp"

WEFTMOE_PUBLIC_API_BEGIN

#include "weftmoe/config/runtime_config.hpp"
#include "weftmoe/transport/count_exchange.hpp"

namespace weftmoe::cuda {

inline constexpr std::uint64_t kControlProtocolMagic = 0x4d4547414d4f4543ULL;
inline constexpr std::uint64_t kControlProtocolVersion = 2;
inline constexpr std::size_t kControlWireBytes = 256;
inline constexpr std::size_t kControlSchemaWords = 32;
inline constexpr std::uint64_t kControlDescriptorSchema = 1;
inline constexpr std::uint64_t kCanonicalConfigMagic = 0x4d4d434647574952ULL;
inline constexpr std::uint64_t kCanonicalConfigVersion = 1;
inline constexpr std::size_t kCanonicalConfigHeaderBytes = 64;
inline constexpr std::size_t kMaxCanonicalConfigBytes = 64U * 1024U * 1024U;

using ControlWire = std::array<std::byte, kControlWireBytes>;
using ConfigWire = std::vector<std::byte>;

struct StableSessionId {
  std::uint64_t word0 = 0;
  std::uint64_t word1 = 0;
};

enum class ControlLocalError : std::uint64_t {
  None = 0,
  InvalidRuntimeConfig = 1,
  InvalidRouteShape = 2,
  CapacityExceeded = 3,
  ArithmeticOverflow = 4,
  ExpertOutOfRange = 5,
  DuplicateExpert = 6,
  ConfigEncodingFailure = 7,
};

enum class ControlField : std::uint64_t {
  None = 0,
  Magic = 1,
  ProtocolVersion = 2,
  SchemaBytes = 3,
  Generation = 4,
  CallingRank = 5,
  WorldSize = 6,
  SessionIdWord0 = 7,
  SessionIdWord1 = 8,
  ConfigDigest = 9,
  ConfigBlobBytes = 10,
  NumExperts = 11,
  TopK = 12,
  MTileRows = 13,
  NumNodes = 14,
  MaxTokensPerRank = 15,
  NumTokens = 16,
  RouteCount = 17,
  ControlDtype = 18,
  CountDtype = 19,
  RouteIndexBits = 20,
  ExpertIndexBits = 21,
  ExpertRowWidth = 22,
  SummaryRowWidth = 23,
  DescriptorSchema = 24,
  LocalValidation = 25,
  Checksum = 26,
  RankToNode = 27,
  RankToLocalRank = 28,
  ExpertToRank = 29,
  ExpertRows = 30,
  WorkspaceRowsPerRank = 31,
  TopologyShape = 32,
  ConfigBlob = 33,
  SessionIdentity = 34,
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

// Logical decoded header. It is never copied directly onto the wire.
struct ControlHeader {
  std::uint64_t magic = kControlProtocolMagic;
  std::uint64_t protocol_version = kControlProtocolVersion;
  std::uint64_t schema_bytes = kControlWireBytes;
  std::uint64_t generation = 0;
  std::uint64_t calling_rank = 0;
  std::uint64_t world_size = 0;
  std::uint64_t session_id_word0 = 0;
  std::uint64_t session_id_word1 = 0;
  std::uint64_t config_digest = 0;
  std::uint64_t config_blob_bytes = 0;
  std::uint64_t num_experts = 0;
  std::uint64_t top_k = 0;
  std::uint64_t m_tile_rows = 0;
  std::uint64_t num_nodes = 0;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t num_tokens = 0;
  std::uint64_t route_count = 0;
  std::uint64_t control_dtype = 1;
  std::uint64_t count_dtype = 1;
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
  std::uint64_t checksum = 0;
};

using ControlMatrix = std::vector<ControlHeader>;
using ControlWireMatrix = std::vector<ControlWire>;

struct CanonicalConfig {
  std::uint64_t num_experts = 0;
  std::uint64_t top_k = 0;
  std::uint64_t m_tile_rows = 0;
  std::uint64_t world_size = 0;
  std::uint64_t num_nodes = 0;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t control_dtype = 1;
  std::uint64_t count_dtype = 1;
  std::uint64_t route_index_bits = 64;
  std::uint64_t expert_index_bits = 32;
  std::uint64_t descriptor_schema = kControlDescriptorSchema;
  std::vector<std::byte> stable_session_identity;
  std::vector<std::uint32_t> rank_to_node;
  std::vector<std::uint32_t> rank_to_local_rank;
  std::vector<std::uint32_t> expert_to_rank;
  std::vector<std::uint64_t> expert_rows;
  std::vector<std::uint64_t> workspace_rows_per_rank;
};

StableSessionId
stable_session_id(const std::vector<std::byte> &identity) noexcept;
std::uint64_t config_wire_digest(const ConfigWire &wire) noexcept;
ControlWire encode_control_header(const ControlHeader &header);
ControlHeader decode_control_header(const ControlWire &wire);
ConfigWire encode_canonical_config(const RuntimeConfig &config,
                                   std::uint64_t max_tokens_per_rank,
                                   const std::vector<std::byte>
                                       &stable_session_identity);
bool try_encode_canonical_config(const RuntimeConfig &config,
                                 std::uint64_t max_tokens_per_rank,
                                 ConfigWire &wire,
                                 LocalPreflightStatus &status,
                                 const std::vector<std::byte>
                                     &stable_session_identity) noexcept;
CanonicalConfig decode_canonical_config(const ConfigWire &wire);

ControlHeader make_control_header(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    std::uint64_t num_tokens, std::uint64_t actual_route_elements,
    std::uint64_t max_tokens_per_rank, std::uint64_t config_blob_bytes,
    std::uint64_t config_digest,
    LocalPreflightStatus device_status = {});

inline ControlHeader make_control_record(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    std::uint64_t num_tokens, std::uint64_t actual_route_elements,
    std::uint64_t max_tokens_per_rank,
    LocalPreflightStatus device_status = {}) {
  ConfigWire wire;
  LocalPreflightStatus encoding_status;
  const std::vector<std::byte> *identity = nullptr;
  if (context)
    identity = &context.session().stable_identity_bytes();
  static const std::vector<std::byte> empty_identity;
  (void)try_encode_canonical_config(
      config, max_tokens_per_rank, wire, encoding_status,
      identity ? *identity : empty_identity);
  if (device_status.error == ControlLocalError::None)
    device_status = encoding_status;
  return make_control_header(config, context, num_tokens,
                             actual_route_elements, max_tokens_per_rank,
                             wire.size(), config_wire_digest(wire),
                             device_status);
}

ControlMatrix validate_control_wires(const ControlWireMatrix &matrix);
void validate_control_matrix(const ControlMatrix &matrix);
void validate_config_matrix(const std::vector<ConfigWire> &matrix,
                            const ControlMatrix &headers);

const char *control_field_name(ControlField field) noexcept;
const char *control_error_name(ControlLocalError error) noexcept;

} // namespace weftmoe::cuda

WEFTMOE_PUBLIC_API_END
