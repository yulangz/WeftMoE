#pragma once

#ifndef WEFTMOE_PF_NETWORK_WORKERS
#define WEFTMOE_PF_NETWORK_WORKERS 0
#endif
#ifndef WEFTMOE_PF_WG3_MATH_REGISTERS
#define WEFTMOE_PF_WG3_MATH_REGISTERS 208
#endif
#ifndef WEFTMOE_PF_WG3_LOCAL_REGISTERS
#define WEFTMOE_PF_WG3_LOCAL_REGISTERS 48
#endif
#ifndef WEFTMOE_PF_WG3_NETWORK_REGISTERS
#define WEFTMOE_PF_WG3_NETWORK_REGISTERS 48
#endif
#ifndef WEFTMOE_PF_WG3_WAIT_GROUP
#define WEFTMOE_PF_WG3_WAIT_GROUP 0
#endif
#ifndef WEFTMOE_PF_NETWORK_PENDING
#define WEFTMOE_PF_NETWORK_PENDING 4
#endif
#ifndef WEFTMOE_PF_LOCAL_FIRST
#define WEFTMOE_PF_LOCAL_FIRST 1
#endif
#ifndef WEFTMOE_PF_EARLY_REDUCE
#define WEFTMOE_PF_EARLY_REDUCE 0
#endif
#ifndef WEFTMOE_PF_BORROW_ALL_CTA
#define WEFTMOE_PF_BORROW_ALL_CTA 0
#endif
#ifndef WEFTMOE_PF_BORROW_CONTROL_REGISTERS
#define WEFTMOE_PF_BORROW_CONTROL_REGISTERS 80
#endif

namespace weftmoe_pf {

// 0: legacy network entry; 1: dedicated warps in the 384-thread entry;
// 2: a separate fourth warpgroup. The two new entries share their data plane.
inline constexpr unsigned kNetworkWorkerMode = WEFTMOE_PF_NETWORK_WORKERS;
inline constexpr unsigned kNetworkPending = WEFTMOE_PF_NETWORK_PENDING;
#ifndef WEFTMOE_PF_WG3_ALL_CTA
#define WEFTMOE_PF_WG3_ALL_CTA 0
#endif
inline constexpr unsigned kGetWorkers = WEFTMOE_PF_BORROW_ALL_CTA ? 64U :
    WEFTMOE_PF_WG3_ALL_CTA ? 132U : 16U;
inline constexpr unsigned kPutWorkers = WEFTMOE_PF_BORROW_ALL_CTA ? 48U :
    WEFTMOE_PF_WG3_ALL_CTA ? 132U : 8U;
inline constexpr unsigned kProgressWorkers = WEFTMOE_PF_BORROW_ALL_CTA ? 19U :
    WEFTMOE_PF_WG3_ALL_CTA ? 132U : 4U;
inline constexpr unsigned kNetworkOwners =
    kGetWorkers + kPutWorkers + kProgressWorkers + 1U;
static_assert(kNetworkWorkerMode <= 2U);
static_assert(!WEFTMOE_PF_BORROW_ALL_CTA || kNetworkWorkerMode == 1U);

// Each rail holds one source replica per node. Return TX and RX must be
// separate: simultaneous bidirectional puts may otherwise overwrite a TX
// range that the local NIC has not finished reading.
struct NetworkLayout {
  unsigned contexts = 0;
  bool dispatch_push = false;
  bool phase_order = false;
  bool publication_plan = false;
  bool extra_pull = false;
  bool deduplicate_return_ready = false;
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
