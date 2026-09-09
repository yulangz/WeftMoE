#pragma once

#include "weftmoe/transport/activation_protocol.hpp"

namespace weftmoe::transport {
inline constexpr std::uint32_t kActivationTransportSchemaVersion = 2;
enum class ActivationExpertTransform : std::uint8_t { SyntheticValidation, Identity };
enum class ActivationGatePlacement : std::uint8_t {
  SourceAfter,
  DestinationBeforeUnsupported,
};
struct ActivationTransportConfig {
  std::uint64_t hidden_size = 0;
  std::uint32_t channels = 1;
  std::uint32_t ring_slots_per_source_channel = 0;
  std::uint32_t max_inflight_generations = 2;
  std::uint32_t max_inflight_consumers = 4;
  ActivationExpertTransform expert_transform =
      ActivationExpertTransform::SyntheticValidation;
  ActivationGatePlacement gate_placement = ActivationGatePlacement::SourceAfter;
};
enum class ActivationTransportBackend : std::uint8_t {
  DeviceSimulation,
  LocalCudaP2p,
  NcclSocketFallback,
  Gin,
};
const char *activation_transport_backend_name(ActivationTransportBackend) noexcept;
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
  std::vector<BFloat16> contribution_by_route;
  std::vector<BFloat16> combined_tokens;
  std::vector<BFloat16> canonical_expert_rows;
  TransportProtocolAudit audit;
};
SyntheticTransportResult simulate_activation_transport(
    const DispatchHandle &, const std::vector<BFloat16> &,
    const std::vector<float> &, const ActivationTransportConfig &,
    std::uint64_t, const std::vector<std::uint64_t> & = {});
} // namespace weftmoe::transport
