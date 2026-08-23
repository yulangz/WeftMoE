#include "megamoe/cuda/control_protocol.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
namespace {

constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kConfigChecksumSeed = 1469598103934665603ULL;
constexpr std::uint64_t kConfigDigestSeed = 7809847782465536322ULL;
constexpr std::size_t kConfigChecksumOffset = 48U;
constexpr std::size_t kConfigDigestOffset = 56U;

std::uint64_t hash_bytes(const std::byte *bytes, std::size_t size,
                         std::uint64_t seed) noexcept {
  auto value = seed;
  for (std::size_t index = 0; index < size; ++index) {
    value ^= static_cast<std::uint8_t>(bytes[index]);
    value *= kFnvPrime;
  }
  return value;
}

void write_u64(std::byte *destination, std::uint64_t value) noexcept {
  for (unsigned index = 0; index < 8; ++index)
    destination[index] =
        static_cast<std::byte>((value >> (index * 8U)) & 0xffU);
}

std::uint64_t read_u64(const std::byte *source) noexcept {
  std::uint64_t value = 0;
  for (unsigned index = 0; index < 8; ++index)
    value |= static_cast<std::uint64_t>(
                 static_cast<std::uint8_t>(source[index]))
             << (index * 8U);
  return value;
}

std::array<std::uint64_t, kControlSchemaWords>
header_words(const ControlHeader &header) noexcept {
  return {header.magic,
          header.protocol_version,
          header.schema_bytes,
          header.generation,
          header.calling_rank,
          header.world_size,
          header.session_id_word0,
          header.session_id_word1,
          header.config_digest,
          header.config_blob_bytes,
          header.num_experts,
          header.top_k,
          header.m_tile_rows,
          header.num_nodes,
          header.max_tokens_per_rank,
          header.num_tokens,
          header.route_count,
          header.control_dtype,
          header.count_dtype,
          header.route_index_bits,
          header.expert_index_bits,
          header.expert_row_width,
          header.summary_row_width,
          header.descriptor_schema,
          header.local_error_code,
          header.local_error_field,
          header.error_token,
          header.error_slot,
          header.error_expert,
          header.error_expected,
          header.error_actual,
          header.checksum};
}

ControlHeader words_header(
    const std::array<std::uint64_t, kControlSchemaWords> &word) noexcept {
  ControlHeader header;
  header.magic = word[0];
  header.protocol_version = word[1];
  header.schema_bytes = word[2];
  header.generation = word[3];
  header.calling_rank = word[4];
  header.world_size = word[5];
  header.session_id_word0 = word[6];
  header.session_id_word1 = word[7];
  header.config_digest = word[8];
  header.config_blob_bytes = word[9];
  header.num_experts = word[10];
  header.top_k = word[11];
  header.m_tile_rows = word[12];
  header.num_nodes = word[13];
  header.max_tokens_per_rank = word[14];
  header.num_tokens = word[15];
  header.route_count = word[16];
  header.control_dtype = word[17];
  header.count_dtype = word[18];
  header.route_index_bits = word[19];
  header.expert_index_bits = word[20];
  header.expert_row_width = word[21];
  header.summary_row_width = word[22];
  header.descriptor_schema = word[23];
  header.local_error_code = word[24];
  header.local_error_field = word[25];
  header.error_token = word[26];
  header.error_slot = word[27];
  header.error_expert = word[28];
  header.error_expected = word[29];
  header.error_actual = word[30];
  header.checksum = word[31];
  return header;
}

std::uint64_t control_checksum(const ControlHeader &header) noexcept {
  auto words = header_words(header);
  std::array<std::byte, (kControlSchemaWords - 1U) * 8U> bytes{};
  for (std::size_t index = 0; index + 1U < words.size(); ++index)
    write_u64(bytes.data() + index * 8U, words[index]);
  return hash_bytes(bytes.data(), bytes.size(), 1469598103934665603ULL);
}

void set_error(ControlHeader &header, const LocalPreflightStatus &status) {
  if (header.local_error_code != 0 || status.error == ControlLocalError::None)
    return;
  header.local_error_code = static_cast<std::uint64_t>(status.error);
  header.local_error_field = static_cast<std::uint64_t>(status.field);
  header.error_token = status.token;
  header.error_slot = status.slot;
  std::memcpy(&header.error_expert, &status.expert, sizeof(status.expert));
  header.error_expected = status.expected;
  header.error_actual = status.actual;
}

[[noreturn]] void control_failure(ControlField field,
                                  std::uint64_t origin_rank,
                                  std::uint64_t expected,
                                  std::uint64_t actual,
                                  const ControlHeader &header,
                                  const char *detail) {
  std::int64_t expert = 0;
  std::memcpy(&expert, &header.error_expert, sizeof(expert));
  std::ostringstream message;
  message << "phase=control-agreement, field=" << control_field_name(field)
          << ", origin_rank=" << origin_rank << ", expected=" << expected
          << ", actual=" << actual << ", error_code="
          << control_error_name(
                 static_cast<ControlLocalError>(header.local_error_code))
          << ", token=";
  if (header.error_token == UINT64_MAX)
    message << "n/a";
  else
    message << header.error_token;
  message << ", slot=";
  if (header.error_slot == UINT64_MAX)
    message << "n/a";
  else
    message << header.error_slot;
  message << ", expert=" << expert << ", detail=" << detail;
  throw NotifyDispatchError(NotifyErrorCode::ControlProtocolFailure,
                            message.str());
}

[[noreturn]] void config_failure(ControlField field, std::uint64_t origin_rank,
                                 const std::string &expected,
                                 const std::string &actual,
                                 const std::string &coordinate = "n/a") {
  std::ostringstream message;
  message << "phase=config-agreement, field=" << control_field_name(field)
          << ", origin_rank=" << origin_rank << ", coordinate=" << coordinate
          << ", expected=" << expected << ", actual=" << actual;
  throw NotifyDispatchError(NotifyErrorCode::ControlProtocolFailure,
                            message.str());
}

void checked_append_u64(ConfigWire &wire, std::uint64_t value) {
  if (wire.size() > kMaxCanonicalConfigBytes - 8U)
    throw NotifyDispatchError(
        NotifyErrorCode::ControlProtocolFailure,
        "phase=config-encode, field=blob_bytes, origin_rank=n/a, expected<=67108864, actual=overflow");
  const auto old = wire.size();
  wire.resize(old + 8U);
  write_u64(wire.data() + old, value);
}

void checked_append_bytes(ConfigWire &wire, const std::vector<std::byte> &bytes) {
  if (bytes.size() > kMaxCanonicalConfigBytes ||
      wire.size() > kMaxCanonicalConfigBytes - bytes.size())
    throw NotifyDispatchError(
        NotifyErrorCode::ControlProtocolFailure,
        "phase=config-encode, field=blob_bytes, origin_rank=n/a, expected<=67108864, actual=overflow");
  wire.insert(wire.end(), bytes.begin(), bytes.end());
}

std::size_t checked_encoded_bytes(const RuntimeConfig &config,
                                  std::size_t identity_bytes) {
  constexpr std::size_t kPayloadScalarWords = 17U;
  std::size_t words = kPayloadScalarWords;
  const auto add_words = [&](std::size_t count) {
    if (count > (kMaxCanonicalConfigBytes / 8U) - words)
      throw NotifyDispatchError(
          NotifyErrorCode::ControlProtocolFailure,
          "phase=config-encode, field=blob_bytes, origin_rank=n/a, expected<=67108864, actual=overflow");
    words += count;
  };
  add_words(config.topology.rank_to_node.size());
  add_words(config.topology.rank_to_local_rank.size());
  add_words(config.topology.expert_to_rank.size());
  add_words(config.capacity.expert_rows.size());
  add_words(config.capacity.workspace_rows_per_rank.size());
  if (words > (kMaxCanonicalConfigBytes - kCanonicalConfigHeaderBytes) / 8U)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(kMaxCanonicalConfigBytes), "overflow");
  const auto word_bytes = words * 8U;
  if (identity_bytes > kMaxCanonicalConfigBytes - kCanonicalConfigHeaderBytes -
                           word_bytes)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(kMaxCanonicalConfigBytes), "overflow");
  return kCanonicalConfigHeaderBytes + word_bytes + identity_bytes;
}

std::uint64_t normalized_config_digest(const ConfigWire &wire) noexcept {
  auto value = kConfigDigestSeed;
  for (std::size_t index = 0; index < wire.size(); ++index) {
    const auto byte = index >= kConfigDigestOffset &&
                              index < kConfigDigestOffset + 8U
                          ? std::byte{0}
                          : wire[index];
    value ^= static_cast<std::uint8_t>(byte);
    value *= kFnvPrime;
  }
  return value;
}

template <typename T>
void append_vector(ConfigWire &wire, const std::vector<T> &values) {
  checked_append_u64(wire, values.size());
  for (const auto value : values)
    checked_append_u64(wire, static_cast<std::uint64_t>(value));
}

class ConfigReader {
public:
  ConfigReader(const ConfigWire &wire, std::size_t offset, std::size_t end)
      : wire_(wire), offset_(offset), end_(end) {}

  std::uint64_t read(const char *field) {
    if (offset_ > end_ || end_ - offset_ < 8U)
      throw NotifyDispatchError(
          NotifyErrorCode::ControlProtocolFailure,
          std::string("phase=config-decode, field=") + field +
              ", origin_rank=n/a, expected=8 available bytes, actual=truncated");
    const auto value = read_u64(wire_.data() + offset_);
    offset_ += 8U;
    return value;
  }

  template <typename T>
  std::vector<T> vector(const char *field, std::uint64_t maximum) {
    const auto count = read(field);
    if (count > maximum || offset_ > end_ || count > (end_ - offset_) / 8U)
      throw NotifyDispatchError(
          NotifyErrorCode::ControlProtocolFailure,
          std::string("phase=config-decode, field=") + field +
              ", origin_rank=n/a, expected=declared elements within blob, actual=" +
              std::to_string(count));
    std::vector<T> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
      const auto value = read(field);
      if (value > static_cast<std::uint64_t>(std::numeric_limits<T>::max()))
        throw NotifyDispatchError(
            NotifyErrorCode::ControlProtocolFailure,
            std::string("phase=config-decode, field=") + field +
                ", origin_rank=n/a, expected=value in range, actual=" +
                std::to_string(value));
      values.push_back(static_cast<T>(value));
    }
    return values;
  }

  std::vector<std::byte> bytes(const char *field, std::uint64_t maximum) {
    const auto count = read(field);
    if (count > maximum || count > std::numeric_limits<std::size_t>::max() ||
        offset_ > end_ || static_cast<std::size_t>(count) > end_ - offset_)
      throw NotifyDispatchError(
          NotifyErrorCode::ControlProtocolFailure,
          std::string("phase=config-decode, field=") + field +
              ", origin_rank=n/a, expected=bounded bytes within payload, actual=" +
              std::to_string(count));
    const auto begin = wire_.begin() + static_cast<std::ptrdiff_t>(offset_);
    std::vector<std::byte> value(begin,
                                 begin + static_cast<std::ptrdiff_t>(count));
    offset_ += static_cast<std::size_t>(count);
    return value;
  }

  std::size_t offset() const noexcept { return offset_; }

private:
  const ConfigWire &wire_;
  std::size_t offset_ = 0;
  std::size_t end_ = 0;
};

template <typename T>
void compare_vector(const std::vector<T> &expected,
                    const std::vector<T> &actual, ControlField field,
                    std::uint32_t rank) {
  if (expected.size() != actual.size())
    config_failure(field, rank, std::to_string(expected.size()),
                   std::to_string(actual.size()), "length");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index])
      config_failure(field, rank, std::to_string(expected[index]),
                     std::to_string(actual[index]), std::to_string(index));
  }
}

std::string safe_identity_description(const std::vector<std::byte> &value) {
  const auto digest = stable_session_id(value);
  std::ostringstream out;
  out << "length=" << value.size() << ",digest=" << std::hex
      << digest.word0 << ':' << digest.word1;
  return out.str();
}

void compare_identity(const std::vector<std::byte> &expected,
                      const std::vector<std::byte> &actual,
                      std::uint32_t rank) {
  if (expected.size() != actual.size())
    config_failure(ControlField::SessionIdentity, rank,
                   safe_identity_description(expected),
                   safe_identity_description(actual), "length");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (expected[index] != actual[index])
      config_failure(ControlField::SessionIdentity, rank,
                     safe_identity_description(expected),
                     safe_identity_description(actual),
                     std::to_string(index));
  }
}

} // namespace

StableSessionId
stable_session_id(const std::vector<std::byte> &identity) noexcept {
  StableSessionId result;
  result.word0 = hash_bytes(identity.data(), identity.size(),
                            1469598103934665603ULL);
  result.word1 = hash_bytes(identity.data(), identity.size(),
                            7809847782465536322ULL);
  return result;
}

std::uint64_t config_wire_digest(const ConfigWire &wire) noexcept {
  return normalized_config_digest(wire);
}

ControlWire encode_control_header(const ControlHeader &input) {
  auto header = input;
  header.checksum = control_checksum(header);
  const auto words = header_words(header);
  ControlWire wire{};
  for (std::size_t index = 0; index < words.size(); ++index)
    write_u64(wire.data() + index * 8U, words[index]);
  return wire;
}

ControlHeader decode_control_header(const ControlWire &wire) {
  std::array<std::uint64_t, kControlSchemaWords> words{};
  for (std::size_t index = 0; index < words.size(); ++index)
    words[index] = read_u64(wire.data() + index * 8U);
  auto header = words_header(words);
  const auto expected = control_checksum(header);
  if (header.checksum != expected)
    control_failure(ControlField::Checksum, header.calling_rank, expected,
                    header.checksum, header, "wire checksum mismatch");
  return header;
}

ConfigWire encode_canonical_config(const RuntimeConfig &config,
                                   std::uint64_t max_tokens_per_rank,
                                   const std::vector<std::byte>
                                       &stable_session_identity) {
  constexpr std::size_t kMaxStableIdentityBytes = 4096U;
  if (stable_session_identity.empty() ||
      stable_session_identity.size() > kMaxStableIdentityBytes)
    config_failure(ControlField::ConfigBlob, 0, "stable_identity_bytes=(0,4096]",
                   std::to_string(stable_session_identity.size()),
                   "session_identity.length");
  const auto total_bytes =
      checked_encoded_bytes(config, stable_session_identity.size());
  ConfigWire wire;
  wire.reserve(total_bytes);
  // Self-checking envelope: eight explicit little-endian words.
  checked_append_u64(wire, kCanonicalConfigMagic);
  checked_append_u64(wire, kCanonicalConfigVersion);
  checked_append_u64(wire, kCanonicalConfigHeaderBytes);
  checked_append_u64(wire, total_bytes - kCanonicalConfigHeaderBytes);
  checked_append_u64(wire, 0); // reserved
  checked_append_u64(wire, 0); // reserved
  checked_append_u64(wire, 0); // payload checksum, filled below
  checked_append_u64(wire, 0); // normalized wire digest, filled below
  checked_append_u64(wire, config.num_experts);
  checked_append_u64(wire, config.top_k);
  checked_append_u64(wire, config.m_tile_rows);
  checked_append_u64(wire, config.topology.world_size);
  checked_append_u64(wire, config.topology.num_nodes);
  checked_append_u64(wire, max_tokens_per_rank);
  checked_append_u64(wire, 1); // control dtype
  checked_append_u64(wire, 1); // count dtype
  checked_append_u64(wire, 64);
  checked_append_u64(wire, 32);
  checked_append_u64(wire, kControlDescriptorSchema);
  append_vector(wire, config.topology.rank_to_node);
  append_vector(wire, config.topology.rank_to_local_rank);
  append_vector(wire, config.topology.expert_to_rank);
  append_vector(wire, config.capacity.expert_rows);
  append_vector(wire, config.capacity.workspace_rows_per_rank);
  checked_append_u64(wire, stable_session_identity.size());
  checked_append_bytes(wire, stable_session_identity);
  if (wire.size() != total_bytes)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(total_bytes), std::to_string(wire.size()),
                   "encoder_total");
  const auto payload_checksum = hash_bytes(
      wire.data() + kCanonicalConfigHeaderBytes,
      wire.size() - kCanonicalConfigHeaderBytes, kConfigChecksumSeed);
  write_u64(wire.data() + kConfigChecksumOffset, payload_checksum);
  write_u64(wire.data() + kConfigDigestOffset,
            normalized_config_digest(wire));
  return wire;
}

bool try_encode_canonical_config(const RuntimeConfig &config,
                                 std::uint64_t max_tokens_per_rank,
                                 ConfigWire &wire,
                                 LocalPreflightStatus &status,
                                 const std::vector<std::byte>
                                     &stable_session_identity) noexcept {
  wire.clear();
  status = {};
  try {
    wire = encode_canonical_config(config, max_tokens_per_rank,
                                   stable_session_identity);
    return true;
  } catch (...) {
    wire.clear();
    status.error = ControlLocalError::ConfigEncodingFailure;
    status.field = ControlField::ConfigBlobBytes;
    status.expected = kMaxCanonicalConfigBytes;
    status.actual = UINT64_MAX;
    return false;
  }
}

CanonicalConfig decode_canonical_config(const ConfigWire &wire) {
  if (wire.size() > kMaxCanonicalConfigBytes)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(kMaxCanonicalConfigBytes),
                   std::to_string(wire.size()));
  if (wire.size() < kCanonicalConfigHeaderBytes)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(kCanonicalConfigHeaderBytes),
                   std::to_string(wire.size()), "truncated_envelope");
  ConfigReader envelope(wire, 0, kCanonicalConfigHeaderBytes);
  const auto magic = envelope.read("magic");
  if (magic != kCanonicalConfigMagic)
    config_failure(ControlField::Magic, 0,
                   std::to_string(kCanonicalConfigMagic),
                   std::to_string(magic));
  const auto version = envelope.read("version");
  if (version != kCanonicalConfigVersion)
    config_failure(ControlField::ProtocolVersion, 0,
                   std::to_string(kCanonicalConfigVersion),
                   std::to_string(version));
  const auto header_bytes = envelope.read("header_bytes");
  if (header_bytes != kCanonicalConfigHeaderBytes)
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(kCanonicalConfigHeaderBytes),
                   std::to_string(header_bytes), "header_length");
  const auto payload_bytes = envelope.read("payload_bytes");
  const auto reserved0 = envelope.read("reserved0");
  const auto reserved1 = envelope.read("reserved1");
  if (reserved0 != 0 || reserved1 != 0)
    config_failure(ControlField::ConfigBlob, 0, "reserved=0",
                   std::to_string(reserved0) + "/" +
                       std::to_string(reserved1),
                   "reserved_words");
  const auto checksum = envelope.read("payload_checksum");
  const auto digest = envelope.read("wire_digest");
  if (payload_bytes > kMaxCanonicalConfigBytes - kCanonicalConfigHeaderBytes)
    config_failure(ControlField::ConfigBlobBytes, 0, "payload within 64 MiB",
                   std::to_string(payload_bytes), "payload_length");
  const auto total = kCanonicalConfigHeaderBytes +
                     static_cast<std::size_t>(payload_bytes);
  if (total != wire.size())
    config_failure(ControlField::ConfigBlobBytes, 0,
                   std::to_string(total), std::to_string(wire.size()),
                   wire.size() < total ? "truncated" : "trailing_bytes");
  const auto actual_checksum = hash_bytes(
      wire.data() + kCanonicalConfigHeaderBytes,
      static_cast<std::size_t>(payload_bytes), kConfigChecksumSeed);
  if (checksum != actual_checksum)
    config_failure(ControlField::Checksum, 0,
                   std::to_string(actual_checksum), std::to_string(checksum),
                   "payload_checksum");
  const auto actual_digest = normalized_config_digest(wire);
  if (digest != actual_digest)
    config_failure(ControlField::ConfigDigest, 0,
                   std::to_string(actual_digest), std::to_string(digest),
                   "wire_digest");
  ConfigReader reader(wire, kCanonicalConfigHeaderBytes, wire.size());
  CanonicalConfig result;
  result.num_experts = reader.read("num_experts");
  result.top_k = reader.read("top_k");
  result.m_tile_rows = reader.read("m_tile_rows");
  result.world_size = reader.read("world_size");
  result.num_nodes = reader.read("num_nodes");
  result.max_tokens_per_rank = reader.read("max_tokens_per_rank");
  result.control_dtype = reader.read("control_dtype");
  result.count_dtype = reader.read("count_dtype");
  result.route_index_bits = reader.read("route_index_bits");
  result.expert_index_bits = reader.read("expert_index_bits");
  result.descriptor_schema = reader.read("descriptor_schema");
  result.rank_to_node = reader.vector<std::uint32_t>("rank_to_node", UINT32_MAX);
  result.rank_to_local_rank =
      reader.vector<std::uint32_t>("rank_to_local_rank", UINT32_MAX);
  result.expert_to_rank =
      reader.vector<std::uint32_t>("expert_to_rank", UINT32_MAX);
  result.expert_rows = reader.vector<std::uint64_t>("expert_rows", UINT32_MAX);
  result.workspace_rows_per_rank =
      reader.vector<std::uint64_t>("workspace_rows_per_rank", UINT32_MAX);
  result.stable_session_identity = reader.bytes("session_identity", 4096U);
  if (reader.offset() != wire.size())
    config_failure(ControlField::ConfigBlob, 0, "canonical end",
                   std::to_string(wire.size() - reader.offset()),
                   "trailing_bytes");
  if (result.stable_session_identity.empty())
    config_failure(ControlField::SessionIdentity, 0, "non-empty", "0",
                   "length");
  if (result.control_dtype != 1 || result.count_dtype != 1 ||
      result.route_index_bits != 64 || result.expert_index_bits != 32 ||
      result.descriptor_schema != kControlDescriptorSchema)
    config_failure(ControlField::ConfigBlob, 0, "supported protocol layout",
                   "unsupported dtype/index/schema", "layout");
  if (result.max_tokens_per_rank == 0)
    config_failure(ControlField::MaxTokensPerRank, 0, ">0", "0");
  if (result.num_experts > UINT32_MAX || result.top_k > UINT32_MAX ||
      result.m_tile_rows > UINT32_MAX || result.world_size > UINT32_MAX ||
      result.num_nodes > UINT32_MAX)
    config_failure(ControlField::ConfigBlob, 0, "runtime fields within uint32",
                   "overflow", "shape");
  RuntimeConfig runtime;
  runtime.num_experts = static_cast<std::uint32_t>(result.num_experts);
  runtime.top_k = static_cast<std::uint32_t>(result.top_k);
  runtime.m_tile_rows = static_cast<std::uint32_t>(result.m_tile_rows);
  runtime.topology.world_size = static_cast<std::uint32_t>(result.world_size);
  runtime.topology.num_nodes = static_cast<std::uint32_t>(result.num_nodes);
  runtime.topology.rank_to_node = result.rank_to_node;
  runtime.topology.rank_to_local_rank = result.rank_to_local_rank;
  runtime.topology.expert_to_rank = result.expert_to_rank;
  runtime.capacity.expert_rows = result.expert_rows;
  runtime.capacity.workspace_rows_per_rank = result.workspace_rows_per_rank;
  try {
    validate_runtime_config(runtime);
  } catch (const NotifyDispatchError &error) {
    config_failure(ControlField::ConfigBlob, 0,
                   "valid canonical runtime config", error.what(),
                   "runtime_validation");
  }
  return result;
}

ControlHeader make_control_header(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    std::uint64_t num_tokens, std::uint64_t actual_route_elements,
    std::uint64_t max_tokens_per_rank, std::uint64_t config_blob_bytes,
    std::uint64_t config_digest, LocalPreflightStatus device_status) {
  ControlHeader header;
  if (!context) {
    set_error(header, {ControlLocalError::InvalidRuntimeConfig,
                       ControlField::LocalValidation, UINT64_MAX, UINT64_MAX,
                       0, 1, 0});
  } else {
    header.generation = context.session().generation();
    header.calling_rank = context.calling_rank();
    header.world_size = context.session().world_size();
    const auto session =
        stable_session_id(context.session().stable_identity_bytes());
    header.session_id_word0 = session.word0;
    header.session_id_word1 = session.word1;
  }
  header.config_digest = config_digest;
  header.config_blob_bytes = config_blob_bytes;
  header.num_experts = config.num_experts;
  header.top_k = config.top_k;
  header.m_tile_rows = config.m_tile_rows;
  header.num_nodes = config.topology.num_nodes;
  header.max_tokens_per_rank = max_tokens_per_rank;
  header.num_tokens = num_tokens;
  header.expert_row_width = config.num_experts;
  header.summary_row_width = config.topology.num_nodes;

  const bool overflow = config.top_k != 0 &&
                        num_tokens > UINT64_MAX / config.top_k;
  header.route_count = overflow ? UINT64_MAX : num_tokens * config.top_k;
  try {
    validate_runtime_config(config);
  } catch (...) {
    set_error(header, {ControlLocalError::InvalidRuntimeConfig,
                       ControlField::LocalValidation, UINT64_MAX, UINT64_MAX,
                       0, 1, 0});
  }
  if (context && config.topology.world_size != context.session().world_size())
    set_error(header, {ControlLocalError::InvalidRuntimeConfig,
                       ControlField::WorldSize, UINT64_MAX, UINT64_MAX, 0,
                       context.session().world_size(),
                       config.topology.world_size});
  else if (context && context.calling_rank() >= config.topology.world_size)
    set_error(header, {ControlLocalError::InvalidRuntimeConfig,
                       ControlField::CallingRank, UINT64_MAX, UINT64_MAX, 0,
                       config.topology.world_size, context.calling_rank()});
  if (overflow)
    set_error(header, {ControlLocalError::ArithmeticOverflow,
                       ControlField::RouteCount, UINT64_MAX, UINT64_MAX, 0,
                       UINT64_MAX, UINT64_MAX});
  else if (actual_route_elements != header.route_count)
    set_error(header, {ControlLocalError::InvalidRouteShape,
                       ControlField::RouteCount, UINT64_MAX, UINT64_MAX, 0,
                       header.route_count, actual_route_elements});
  else if (num_tokens > max_tokens_per_rank)
    set_error(header, {ControlLocalError::CapacityExceeded,
                       ControlField::NumTokens, UINT64_MAX, UINT64_MAX, 0,
                       max_tokens_per_rank, num_tokens});
  set_error(header, device_status);
  header.checksum = control_checksum(header);
  return header;
}

ControlMatrix validate_control_wires(const ControlWireMatrix &matrix) {
  ControlMatrix decoded;
  decoded.reserve(matrix.size());
  for (const auto &wire : matrix)
    decoded.push_back(decode_control_header(wire));
  validate_control_matrix(decoded);
  return decoded;
}

void validate_control_matrix(const ControlMatrix &matrix) {
  if (matrix.empty()) {
    ControlHeader empty;
    control_failure(ControlField::WorldSize, 0, 1, 0, empty,
                    "empty control matrix");
  }
  const auto &baseline = matrix.front();
  // Rank-coordinate validation is deliberately performed before all other
  // fields, so even corrupt baseline/world declarations cannot change origin.
  for (std::size_t rank = 0; rank < matrix.size(); ++rank)
    if (matrix[rank].calling_rank != rank)
      control_failure(ControlField::CallingRank, rank, rank,
                      matrix[rank].calling_rank, matrix[rank],
                      "AllGather row/rank mismatch");

  const auto check = [&](const ControlHeader &header, std::uint32_t rank,
                         ControlField field, std::uint64_t expected,
                         std::uint64_t actual, const char *detail) {
    if (expected != actual)
      control_failure(field, rank, expected, actual, header, detail);
  };
  if (baseline.world_size != matrix.size())
    control_failure(ControlField::WorldSize, 0, matrix.size(),
                    baseline.world_size, baseline, "matrix row count");

  for (std::uint32_t rank = 0; rank < matrix.size(); ++rank) {
    const auto &header = matrix[rank];
    check(header, rank, ControlField::Magic, kControlProtocolMagic,
          header.magic, "wire constant");
    check(header, rank, ControlField::ProtocolVersion,
          kControlProtocolVersion, header.protocol_version, "wire constant");
    check(header, rank, ControlField::SchemaBytes, kControlWireBytes,
          header.schema_bytes, "wire constant");
    check(header, rank, ControlField::ControlDtype, 1, header.control_dtype,
          "wire constant");
    check(header, rank, ControlField::CountDtype, 1, header.count_dtype,
          "wire constant");
    check(header, rank, ControlField::RouteIndexBits, 64,
          header.route_index_bits, "wire constant");
    check(header, rank, ControlField::ExpertIndexBits, 32,
          header.expert_index_bits, "wire constant");
    check(header, rank, ControlField::DescriptorSchema,
          kControlDescriptorSchema, header.descriptor_schema,
          "wire constant");
    if (rank != 0) {
      // Fixed field order is the protocol's deterministic error priority.
      check(header, rank, ControlField::Generation, baseline.generation,
            header.generation, "rank-0 baseline");
      check(header, rank, ControlField::SessionIdWord0,
            baseline.session_id_word0, header.session_id_word0,
            "rank-0 baseline");
      check(header, rank, ControlField::SessionIdWord1,
            baseline.session_id_word1, header.session_id_word1,
            "rank-0 baseline");
      check(header, rank, ControlField::WorldSize, baseline.world_size,
            header.world_size, "rank-0 baseline");
      // Blob lengths may differ because original stable session identities are
      // length-prefixed. Config AllGather uses the deterministic maximum
      // declared width and reconstructs every exact row length.
      check(header, rank, ControlField::NumExperts, baseline.num_experts,
            header.num_experts, "rank-0 baseline");
      check(header, rank, ControlField::TopK, baseline.top_k, header.top_k,
            "rank-0 baseline");
      check(header, rank, ControlField::MTileRows, baseline.m_tile_rows,
            header.m_tile_rows, "rank-0 baseline");
      check(header, rank, ControlField::NumNodes, baseline.num_nodes,
            header.num_nodes, "rank-0 baseline");
      check(header, rank, ControlField::MaxTokensPerRank,
            baseline.max_tokens_per_rank, header.max_tokens_per_rank,
            "rank-0 baseline");
      check(header, rank, ControlField::ExpertRowWidth,
            baseline.expert_row_width, header.expert_row_width,
            "rank-0 baseline");
      check(header, rank, ControlField::SummaryRowWidth,
            baseline.summary_row_width, header.summary_row_width,
            "rank-0 baseline");
      // config_digest is diagnostic only and intentionally not an agreement
      // gate. Exact config bytes are always compared in the next collective.
    }
    if (header.local_error_code != 0) {
      auto field = static_cast<ControlField>(header.local_error_field);
      if (field == ControlField::None)
        field = ControlField::LocalValidation;
      control_failure(field, rank, header.error_expected, header.error_actual,
                      header, "rank-local preflight failure");
    }
    if (header.config_blob_bytes < kCanonicalConfigHeaderBytes ||
        header.config_blob_bytes > kMaxCanonicalConfigBytes)
      control_failure(ControlField::ConfigBlobBytes, rank,
                      kMaxCanonicalConfigBytes, header.config_blob_bytes,
                      header, "declared canonical blob outside bounds");
    if (header.num_tokens > header.max_tokens_per_rank)
      control_failure(ControlField::NumTokens, rank,
                      header.max_tokens_per_rank, header.num_tokens, header,
                      "tokens exceed common capacity");
    if (header.top_k != 0 && header.num_tokens <= UINT64_MAX / header.top_k)
      check(header, rank, ControlField::RouteCount,
            header.num_tokens * header.top_k, header.route_count,
            "local route shape");
  }
}

void validate_config_matrix(const std::vector<ConfigWire> &matrix,
                            const ControlMatrix &headers) {
  if (matrix.size() != headers.size())
    config_failure(ControlField::WorldSize, 0,
                   std::to_string(headers.size()),
                   std::to_string(matrix.size()));
  std::vector<CanonicalConfig> decoded;
  decoded.reserve(matrix.size());
  for (std::size_t rank = 0; rank < matrix.size(); ++rank) {
    if (matrix[rank].size() != headers[rank].config_blob_bytes)
      config_failure(ControlField::ConfigBlobBytes, rank,
                     std::to_string(headers[rank].config_blob_bytes),
                     std::to_string(matrix[rank].size()));
    try {
      decoded.push_back(decode_canonical_config(matrix[rank]));
    } catch (const NotifyDispatchError &error) {
      config_failure(ControlField::ConfigBlob, rank, "valid canonical blob",
                     error.what());
    }
    const auto digest = config_wire_digest(matrix[rank]);
    if (headers[rank].config_digest != digest)
      config_failure(ControlField::ConfigDigest, rank,
                     std::to_string(digest),
                     std::to_string(headers[rank].config_digest),
                     "control_header_vs_wire");
  }
  const auto &base = decoded.front();
  for (std::uint32_t rank = 1; rank < decoded.size(); ++rank) {
    const auto &value = decoded[rank];
    const auto scalar = [&](ControlField field, std::uint64_t expected,
                            std::uint64_t actual) {
      if (expected != actual)
        config_failure(field, rank, std::to_string(expected),
                       std::to_string(actual));
    };
    scalar(ControlField::NumExperts, base.num_experts, value.num_experts);
    scalar(ControlField::TopK, base.top_k, value.top_k);
    scalar(ControlField::MTileRows, base.m_tile_rows, value.m_tile_rows);
    scalar(ControlField::WorldSize, base.world_size, value.world_size);
    scalar(ControlField::NumNodes, base.num_nodes, value.num_nodes);
    scalar(ControlField::MaxTokensPerRank, base.max_tokens_per_rank,
           value.max_tokens_per_rank);
    scalar(ControlField::ControlDtype, base.control_dtype,
           value.control_dtype);
    scalar(ControlField::CountDtype, base.count_dtype, value.count_dtype);
    scalar(ControlField::RouteIndexBits, base.route_index_bits,
           value.route_index_bits);
    scalar(ControlField::ExpertIndexBits, base.expert_index_bits,
           value.expert_index_bits);
    scalar(ControlField::DescriptorSchema, base.descriptor_schema,
           value.descriptor_schema);
    compare_identity(base.stable_session_identity,
                     value.stable_session_identity, rank);
    compare_vector(base.rank_to_node, value.rank_to_node,
                   ControlField::RankToNode, rank);
    compare_vector(base.rank_to_local_rank, value.rank_to_local_rank,
                   ControlField::RankToLocalRank, rank);
    compare_vector(base.expert_to_rank, value.expert_to_rank,
                   ControlField::ExpertToRank, rank);
    compare_vector(base.expert_rows, value.expert_rows,
                   ControlField::ExpertRows, rank);
    compare_vector(base.workspace_rows_per_rank,
                   value.workspace_rows_per_rank,
                   ControlField::WorkspaceRowsPerRank, rank);
    if (matrix[rank] != matrix.front())
      config_failure(ControlField::ConfigBlob, rank, "byte-identical rank 0",
                     "different bytes");
  }
}

const char *control_field_name(ControlField field) noexcept {
  switch (field) {
  case ControlField::None: return "none";
  case ControlField::Magic: return "magic";
  case ControlField::ProtocolVersion: return "protocol_version";
  case ControlField::SchemaBytes: return "schema_bytes";
  case ControlField::Generation: return "generation";
  case ControlField::CallingRank: return "calling_rank";
  case ControlField::WorldSize: return "world_size";
  case ControlField::SessionIdWord0: return "session_id_word0";
  case ControlField::SessionIdWord1: return "session_id_word1";
  case ControlField::ConfigDigest: return "config_digest";
  case ControlField::ConfigBlobBytes: return "config_blob_bytes";
  case ControlField::NumExperts: return "num_experts";
  case ControlField::TopK: return "top_k";
  case ControlField::MTileRows: return "m_tile_rows";
  case ControlField::NumNodes: return "num_nodes";
  case ControlField::MaxTokensPerRank: return "max_tokens_per_rank";
  case ControlField::NumTokens: return "num_tokens";
  case ControlField::RouteCount: return "route_count";
  case ControlField::ControlDtype: return "control_dtype";
  case ControlField::CountDtype: return "count_dtype";
  case ControlField::RouteIndexBits: return "route_index_bits";
  case ControlField::ExpertIndexBits: return "expert_index_bits";
  case ControlField::ExpertRowWidth: return "expert_row_width";
  case ControlField::SummaryRowWidth: return "summary_row_width";
  case ControlField::DescriptorSchema: return "descriptor_schema";
  case ControlField::LocalValidation: return "local_validation";
  case ControlField::Checksum: return "checksum";
  case ControlField::RankToNode: return "rank_to_node";
  case ControlField::RankToLocalRank: return "rank_to_local_rank";
  case ControlField::ExpertToRank: return "expert_to_rank";
  case ControlField::ExpertRows: return "expert_rows";
  case ControlField::WorkspaceRowsPerRank: return "workspace_rows_per_rank";
  case ControlField::TopologyShape: return "topology_shape";
  case ControlField::ConfigBlob: return "config_blob";
  case ControlField::SessionIdentity: return "session_identity";
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
  case ControlLocalError::ConfigEncodingFailure: return "config_encoding_failure";
  }
  return "unknown";
}

} // namespace megamoe::cuda
