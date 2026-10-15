#pragma once

#ifndef WEFTMOE_PF_NETWORK_PENDING
#define WEFTMOE_PF_NETWORK_PENDING 4
#endif
#ifndef WEFTMOE_PF_NETWORK_GET_WORKERS
#define WEFTMOE_PF_NETWORK_GET_WORKERS 64
#endif
#ifndef WEFTMOE_PF_NETWORK_PRIVATE_DR
#define WEFTMOE_PF_NETWORK_PRIVATE_DR 0
#endif
#ifndef WEFTMOE_PF_CREDIT_WINDOWS
#define WEFTMOE_PF_CREDIT_WINDOWS 1
#endif

namespace weftmoe_pf {

inline constexpr bool kNetworkPrivateDr = WEFTMOE_PF_NETWORK_PRIVATE_DR;
inline constexpr unsigned kNetworkCreditWindows = WEFTMOE_PF_CREDIT_WINDOWS;
static_assert(kNetworkCreditWindows >= 1U && kNetworkCreditWindows <= 8U);

// Borrow one control warp per CTA: dispatch, return, progress, admission.
inline constexpr unsigned kNetworkPending = WEFTMOE_PF_NETWORK_PENDING;
inline constexpr unsigned kGetWorkers = WEFTMOE_PF_NETWORK_GET_WORKERS;
inline constexpr unsigned kPutWorkers = 112U - kGetWorkers;
static_assert(kGetWorkers > 0U && kGetWorkers < 112U);
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
  std::uint64_t return_rows = 0; // Shared bounded FWD/BWD canonical TX arena.
  std::size_t source_bytes =
      0; // Immutable sender metadata, excluding RX geometry.
  std::size_t dispatch_bytes = 0; // FWD/BWD publish only the dispatch directory.
  unsigned ring_workers = 0;
};

inline constexpr std::size_t kNetworkChunkBytes = 256U * 1024U;

} // namespace weftmoe_pf
