#pragma once

#include "cuda/weftmoe_pf/compute_ring.hpp"
#include "weftmoe/transport/activation_protocol.hpp"

namespace weftmoe::cuda::backward_detail {

// Window-major dX work retains canonical U/dAB row identities for dW. These
// immutable directories reuse Forward's idle plan storage, not its payloads
// or lifetime counters.
struct ReturnBlock {
  std::uint64_t ready;
  unsigned expert, rows;
  unsigned window, local_sources;
  std::uint64_t input_rows;
};
struct ReturnRow {
  unsigned compact, source, route, queue_position;
};
struct ReturnBucket {
  unsigned rows, cursor;
  std::uint64_t begin;
  std::uint64_t packet_begin;
};
static_assert(sizeof(ReturnBlock) <= sizeof(static_forward::RingBlock));
static_assert(sizeof(ReturnRow) <= sizeof(static_forward::RingRow));
static_assert(sizeof(ReturnBucket) <= sizeof(static_forward::WindowSource));

struct alignas(64) ReturnSlot {
  // Fixed M-row credits include padding. A future block cannot occupy its
  // residue until both local copies and remote TX packing have retired it.
  std::uint64_t consumed;
};

struct alignas(64) SourceReturnState {
  std::uint64_t reserved;
};

struct ReturnRingView {
  static_forward::WindowShape shape{};
  unsigned depth = 0, contribution_depth = 0;
  std::uint64_t windows = 0;
  static_forward::RingState *state = nullptr;
  std::uint64_t *window_prefix = nullptr;
  std::uint64_t *expert_rows = nullptr;
  std::uint64_t *expert_blocks = nullptr;
  ReturnBlock *blocks = nullptr;
  ReturnRow *rows = nullptr;
  unsigned *row_blocks = nullptr;
  unsigned *du_window = nullptr, *du_order = nullptr, *du_offsets = nullptr;
  ReturnBucket *buckets = nullptr;
  ReturnSlot *slots = nullptr;
  transport::BFloat16 *gathered_dab = nullptr;
};

} // namespace weftmoe::cuda::backward_detail
