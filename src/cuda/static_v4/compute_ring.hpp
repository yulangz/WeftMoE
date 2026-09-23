#pragma once

#include "cuda/static_v4/plan.hpp"
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

struct RingRow {
  std::uint32_t route;
  std::uint32_t source;
  float gate;
  std::uint32_t token;
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
  std::uint64_t tokens;
  std::uint64_t windows;
  std::uint64_t blocks;
  std::uint64_t rows;
  unsigned error;
};

struct SourceEpoch {
  std::uint64_t tokens;
  std::uint64_t base;
};

struct ComputeRingView {
  SourcePlanView source;
  std::uint32_t depth = 0;
  std::uint32_t contribution_depth = 2;
  std::uint64_t contribution_stride = 0;
  RingState *state = nullptr;
  SourceEpoch *source_epochs = nullptr;
  std::uint64_t *source_reserved = nullptr; // Imported exclusive source head.
  ComputeSlot *slots = nullptr;
  ContributionSlot *contributions = nullptr;
  std::uint64_t *contribution_ready = nullptr;
  std::uint64_t *window_prefix = nullptr;
  WindowSource *window_sources = nullptr; // [windows, bounded source group]
  std::uint64_t *expert_rows = nullptr;   // [windows, owned]
  std::uint64_t *expert_blocks = nullptr; // [windows, owned+1]
  RingBlock *blocks = nullptr;
  RingRow *rows = nullptr; // Immutable reverse metadata, not stored in X.
};

struct ComputeRingLayout {
  WindowShape shape{};
  std::uint32_t depth = 0;
  std::uint32_t contribution_depth = 2;
  std::uint64_t max_windows = 0;
  std::uint64_t max_blocks = 0;
  std::size_t state = 0, slots = 0, contributions = 0, contribution_ready = 0;
  std::size_t source_epochs = 0;
  std::size_t source_reserved = 0;
  std::size_t expert_prefix = 0, node_prefix = 0, route_order = 0;
  std::size_t source_slot = 0, node_tokens = 0, packet_prefix = 0;
  std::size_t window_prefix = 0, expert_rows = 0, expert_blocks = 0;
  std::size_t window_sources = 0;
  std::size_t blocks = 0, rows = 0, output = 0;
};

} // namespace weftmoe::cuda::static_forward
