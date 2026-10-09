#pragma once

#ifndef WEFTMOE_PF_NETWORK_WORKERS
#define WEFTMOE_PF_NETWORK_WORKERS 0
#endif

namespace weftmoe_pf {

// 0: legacy network entry; 1: dedicated warps in the 384-thread entry;
// 2: a separate fourth warpgroup. The two new entries share their data plane.
inline constexpr unsigned kNetworkWorkerMode = WEFTMOE_PF_NETWORK_WORKERS;
inline constexpr unsigned kGetWorkers = 16U, kPutWorkers = 8U;
inline constexpr unsigned kProgressWorkers = 4U;
inline constexpr unsigned kNetworkOwners =
    kGetWorkers + kPutWorkers + kProgressWorkers + 1U;
static_assert(kNetworkWorkerMode <= 2U);

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
