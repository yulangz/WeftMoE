#pragma once

#ifndef WEFTMOE_PF_NETWORK_PENDING
#define WEFTMOE_PF_NETWORK_PENDING 4
#endif

namespace weftmoe_pf {

// Borrow one control warp per CTA: dispatch, return, progress, admission.
inline constexpr unsigned kNetworkPending = WEFTMOE_PF_NETWORK_PENDING;
inline constexpr unsigned kGetWorkers = 64U;
inline constexpr unsigned kPutWorkers = 48U;
inline constexpr unsigned kProgressWorkers = 19U;
inline constexpr unsigned kNetworkOwners =
    kGetWorkers + kPutWorkers + kProgressWorkers + 1U;

// Each rail holds one source replica per node. Return TX and RX must be
// separate: simultaneous bidirectional puts may otherwise overwrite a TX
// range that the local NIC has not finished reading.
struct NetworkLayout {
  unsigned contexts = 0;
  std::size_t source = 0;
  std::size_t source_stride = 0;
  std::size_t input = 0;
  std::size_t contribution = 0; // Source-owned, expert-ordered window ring.
  std::size_t gradient_tx = 0, gradient_contribution = 0;
  static_forward::NetworkRings rings{};
  std::size_t ring_region = 0;
  std::size_t source_bytes =
      0; // Immutable sender metadata, excluding RX geometry.
  unsigned ring_workers = 0;
};

inline constexpr std::size_t kNetworkChunkBytes = 256U * 1024U;

} // namespace weftmoe_pf
