#pragma once

#include "cuda/weftmoe_pf/plan.hpp"
#include <cstddef>

namespace weftmoe::cuda::static_forward {

// Cumulative counters survive invocations and profile changes. The admission
// owner installs new targets only after all three old planes have retired.
struct alignas(64) ComputeSlot {
  std::uint64_t generation;
  std::uint64_t x_full;
  std::uint64_t fc1_done;
  std::uint64_t fc2_done;
  std::uint64_t returned;
  std::uint64_t fc1_target;
  std::uint64_t fc2_target;
  std::uint64_t return_target;
  // Unlike fc2_done (S read retirement), this publishes FC2's remote stores.
  std::uint64_t fc2_published;
  std::uint64_t activation_ready; // BWD SwiGLU halves; independent of Return.
};

struct alignas(64) ContributionSlot {
  std::uint64_t generation;
  std::uint64_t consumed;
  std::uint64_t retire_target;
};

struct RingBlock {
  WindowMblock tile;
  std::uint32_t window;
  std::uint32_t slot;
  std::uint64_t generation;
};
static_assert(sizeof(RingBlock) == 32U);

// Single-window source metadata is pushed together, avoiding a peer gate
// read while the destination constructs its immutable reverse rows.
struct alignas(8) SourceRoute {
  std::uint32_t route;
  float gate;
};

struct RingRow {
  std::uint32_t route;
  std::uint32_t source;
  float gate;
  std::uint32_t token;
  void *output; // Source contribution or this GPU's canonical Return TX row.
  std::uint64_t contribution; // Source-owned compact Return row (network FWD).
};

// Communication windows describe arrivals, not GEMM padding. Complete blocks
// use the ordinary ring; cross-window blocks use a reserved tail ring/expert.
struct WindowCompute {
  std::uint64_t row_begin, total_rows;
  std::uint64_t pull_begin, full_begin, start_begin;
  std::uint64_t tail_index, block_begin;
};

// The source address is constant for all routes in a (window, source) pair.
// Freeze division/modulo here, rather than repeating them in every row copy.
struct WindowSource {
  std::uint64_t sequence;
  std::uint64_t generation;
  std::uint64_t row_base;
  std::uint32_t slot;
};

struct RingState {
  std::uint64_t block_base;
  std::uint64_t source_base;
  std::uint64_t windows;
  std::uint64_t blocks;
  std::uint64_t rows;
  std::uint64_t admitted_windows;
};

struct SourceEpoch {
  std::uint64_t tokens;
  std::uint64_t base;
};

struct SourceHeader {
  SourceEpoch epoch;
  unsigned error;
};

// Written once by the destination GPU after window geometry. A rail's Return
// owner can locate C without retaining X or scanning remote expert prefixes.
struct RouteLocation {
  std::uint64_t generation;
  std::uint64_t block;
  std::uint32_t row; // Physical X/S/C row on the destination GPU.
};

// One fixed (window, packet ordinal, source) entry on the consuming GPU.
// Empty entries have rows == 0 and never inspect reusable RX storage.
struct PullPacket {
  const NodeToken *tokens;
  const std::int32_t *experts;
  const float *gates;
  const RouteLocation *locations;
  const void *payload;
  const std::uint64_t *visible_generation;
  std::uint64_t *consumed_total;
  std::uint64_t generation;
  std::uint32_t token_count;
  std::uint32_t rows;
  std::uint32_t peer_rows;
};

// Offsets from one source's immutable metadata region. The same layout is
// locally allocated on one node, and replicated in registered rail memory
// for network planning. X is deliberately absent from this region.
struct SourceMetadataLayout {
  std::size_t header = 0, selected = 0, gates = 0;
  std::size_t tape_rows = 0;
  std::size_t expert_prefix = 0, node_prefix = 0, route_order = 0;
  std::size_t node_tokens = 0, packet_prefix = 0;
  std::size_t dispatch_offsets = 0, dispatch_packets = 0;
  std::size_t dispatch_consumers = 0;
  std::size_t return_prefix = 0, return_offsets = 0, return_packets = 0;
  std::size_t locations = 0;
  std::size_t published_bytes = 0;
  std::size_t bytes = 0;
};

struct ComputeRingView {
  SourcePlanView source;
  SourceHeader *source_header = nullptr;
  float *source_gates = nullptr;
  DispatchPlanView dispatch{};
  ReturnPlanView returning{};
  RouteLocation *locations = nullptr;
  std::uint32_t depth = 0;
  std::uint32_t tail_slots = 0;
  std::uint32_t contribution_depth = 2;
  std::uint64_t contribution_stride = 0;
  RingState *state = nullptr;
  SourceEpoch *source_epochs = nullptr;
  std::uint64_t *source_reserved = nullptr; // Imported exclusive source head.
  ComputeSlot *slots = nullptr;
  ContributionSlot *contributions = nullptr;
  std::uint64_t *contribution_ready = nullptr;
  std::uint32_t *contribution_rows = nullptr; // [route] -> row within its window.
  std::uint64_t *window_prefix = nullptr;
  WindowSource *window_sources = nullptr; // [windows, bounded source group]
  std::uint64_t *expert_rows = nullptr;   // [windows, owned]
  std::uint64_t *expert_blocks = nullptr; // [windows, owned+1]
  RingBlock *blocks = nullptr;
  RingRow *rows = nullptr; // Immutable reverse metadata, not stored in X.
  PullPacket *pull_packets = nullptr;
  WindowCompute *window_compute = nullptr;
  std::uint64_t *start_prefix = nullptr, *pull_prefix = nullptr;
  std::uint32_t *start_blocks = nullptr, *canonical_blocks = nullptr;
  std::uint32_t *pull_order = nullptr;
};

struct ComputeRingLayout {
  WindowShape shape{};
  PacketPartition packets = kNetworkPackets;
  SourceMetadataLayout source;
  std::size_t metadata = 0;
  std::uint32_t depth = 0;
  std::uint32_t tail_slots = 0;
  std::uint32_t contribution_depth = 2;
  std::uint64_t max_windows = 0;
  std::uint64_t max_blocks = 0;
  std::size_t state = 0, slots = 0, contributions = 0, contribution_ready = 0;
  std::size_t contribution_rows = 0;
  std::size_t source_epochs = 0;
  std::size_t source_reserved = 0;
  std::size_t window_prefix = 0, expert_rows = 0, expert_blocks = 0;
  std::size_t window_sources = 0;
  std::size_t blocks = 0, rows = 0;
  std::size_t tape_rows = 0;
  std::size_t pull_packets = 0;
  std::size_t window_compute = 0, start_prefix = 0, pull_prefix = 0;
  std::size_t start_blocks = 0, canonical_blocks = 0, pull_order = 0;
};

} // namespace weftmoe::cuda::static_forward
