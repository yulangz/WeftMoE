#pragma once

namespace weftmoe_pf {

// Each rail holds one source replica per node. Return TX and RX must be
// separate: simultaneous bidirectional puts may otherwise overwrite a TX
// range that the local NIC has not finished reading.
struct NetworkLayout {
  unsigned contexts = 0;
  std::size_t source = 0;
  std::size_t source_stride = 0;
  std::size_t routes = 0;
  std::size_t gates = 0;
  std::size_t input = 0;
  std::size_t return_tx = 0;
  std::size_t return_rx = 0;
  std::size_t return_stride = 0;
  std::size_t contribution = 0; // Source-owned, expert-ordered window ring.
  static_forward::NetworkRings rings{};
  std::size_t ring_region = 0;
  std::size_t source_bytes = 0; // Immutable sender metadata, excluding RX geometry.
  unsigned ring_workers = 0;
};

inline constexpr unsigned kNetworkContexts = 8U;
inline constexpr std::size_t kNetworkChunkBytes = 256U * 1024U;

} // namespace weftmoe_pf
