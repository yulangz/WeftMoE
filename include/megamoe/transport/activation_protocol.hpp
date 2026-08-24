#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "megamoe/metadata/dispatch_handle.hpp"

namespace megamoe::transport {

// Storage type used at the public boundary. It deliberately does not expose a
// CUDA-native half type, so host-only clients and public-header checks remain
// independent from the CUDA toolkit.
struct BFloat16 {
  std::uint16_t bits = 0;
};
static_assert(sizeof(BFloat16) == 2);
inline bool operator==(BFloat16 lhs, BFloat16 rhs) noexcept {
  return lhs.bits == rhs.bits;
}
inline bool operator!=(BFloat16 lhs, BFloat16 rhs) noexcept {
  return !(lhs == rhs);
}

float bfloat16_to_float(BFloat16 value) noexcept;
BFloat16 float_to_bfloat16(float value) noexcept;

inline constexpr std::uint32_t kActivationTransportSchemaVersion = 1;

struct ActivationTransportConfig {
  std::uint64_t hidden_size = 0;
  std::uint32_t channels = 1;
  std::uint32_t ring_slots_per_source_channel = 0;
  std::uint32_t max_inflight_generations = 2;
  std::uint32_t max_inflight_consumers = 4;
};

enum class ActivationTransportBackend : std::uint8_t {
  DeviceSimulation,
  LocalCudaP2p,
  NcclSocketFallback,
  Gin,
};

const char *activation_transport_backend_name(
    ActivationTransportBackend backend) noexcept;

struct TransportSlotHeader {
  std::uint32_t schema_version = kActivationTransportSchemaVersion;
  std::uint32_t source_rank = 0;
  std::uint32_t source_node = 0;
  std::uint32_t destination_node = 0;
  std::uint32_t proxy_rank = 0;
  std::uint32_t channel = 0;
  std::uint64_t generation = 0;
  std::uint64_t epoch = 0;
  std::uint64_t sequence = 0;
  std::uint64_t local_token_id = 0;
  std::uint64_t fanout_begin = 0;
  std::uint64_t fanout_end = 0;
};

struct TransportProtocolAudit {
  std::uint64_t dispatch_packets = 0;
  std::uint64_t relayed_fanout = 0;
  std::uint64_t synthetic_contributions = 0;
  std::uint64_t combine_packets = 0;
  std::uint64_t reduced_tokens = 0;
  std::uint64_t raw_cross_node_payload_bytes = 0;
  std::uint64_t deduplicated_cross_node_payload_bytes = 0;
  std::uint64_t credit_stalls = 0;
  std::uint64_t peak_ring_slots = 0;
};

struct SyntheticTransportResult {
  // One contribution per original [token, top-k] route, never per arrival.
  std::vector<BFloat16> contribution_by_route;
  // Fixed expert-id order FP32 reduction followed by one BF16 cast.
  std::vector<BFloat16> combined_tokens;
  // Canonical destination-rank row image indexed by dst_row.
  std::vector<BFloat16> canonical_expert_rows;
  TransportProtocolAudit audit;
};

// Independent deterministic oracle and ring-protocol simulator. arrival_order
// is a permutation of outgoing packet indices; an empty vector means canonical
// packet order. It is intentionally not a production data-plane entry.
SyntheticTransportResult simulate_activation_transport(
    const DispatchHandle &dispatch,
    const std::vector<BFloat16> &token_activations,
    const ActivationTransportConfig &config,
    std::uint64_t epoch,
    const std::vector<std::uint64_t> &arrival_order = {});

} // namespace megamoe::transport
