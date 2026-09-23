#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define WEFTMOE_STATIC_PLAN_HD __host__ __device__
#else
#define WEFTMOE_STATIC_PLAN_HD
#endif

namespace weftmoe::cuda::static_forward {

WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
ceil_div(std::uint64_t value, std::uint64_t divisor) {
  return value / divisor + (value % divisor != 0U);
}

WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t
block_rows(std::uint64_t rows, std::uint64_t block, std::uint32_t tile_m) {
  const auto remaining = rows - block * tile_m;
  return static_cast<std::uint32_t>(remaining < tile_m ? remaining : tile_m);
}

// Validated once at the launch boundary. A window contains complete tokens
// from a bounded group of source ranks, never a capacity-limited expert batch.
struct WindowShape {
  std::uint32_t world;
  std::uint32_t sources;
  std::uint32_t tokens;
  std::uint32_t top_k;
  std::uint32_t tile_m = 128U;

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t groups() const {
    return static_cast<std::uint32_t>(ceil_div(world, sources));
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  window(std::uint32_t source, std::uint64_t token) const {
    return (token / tokens) * groups() + source / sources;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t
  source_begin(std::uint64_t window_id) const {
    return static_cast<std::uint32_t>(window_id % groups()) * sources;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t
  source_end(std::uint64_t window_id) const {
    const auto end = source_begin(window_id) + sources;
    return end < world ? end : world;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  token_begin(std::uint64_t window_id) const {
    return (window_id / groups()) * tokens;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t
  token_count(std::uint64_t window_id, std::uint32_t source,
              std::uint64_t source_tokens) const {
    const auto begin = token_begin(window_id);
    if (source < source_begin(window_id) || source >= source_end(window_id) ||
        source_tokens <= begin)
      return 0U;
    const auto remaining = source_tokens - begin;
    return static_cast<std::uint32_t>(remaining < tokens ? remaining : tokens);
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  windows(std::uint64_t max_source_tokens) const {
    return ceil_div(max_source_tokens, tokens) * groups();
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  max_mblocks(std::uint32_t owned_experts) const {
    const auto active_sources = sources < world ? sources : world;
    return ceil_div(std::uint64_t(active_sources) * tokens * top_k, tile_m) +
           owned_experts;
  }
};

// A route's explicit validity is separate from its expert id. Invalid expert
// ids must be rejected by planning, not silently treated as masked routes.
struct TokenRoute {
  std::uint32_t expert;
  bool valid;
};

// Emit each actually hit node once, in first-topk-occurrence order. Every
// valid topk slot still contributes one fanout, including duplicate experts.
template <class Emit>
WEFTMOE_STATIC_PLAN_HD inline void
for_each_token_node(const TokenRoute *routes, std::uint32_t top_k,
                    std::uint32_t experts_per_node, Emit emit) {
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    if (!routes[slot].valid)
      continue;
    const auto node = routes[slot].expert / experts_per_node;
    bool first = true;
    for (std::uint32_t previous = 0; previous < slot; ++previous)
      if (routes[previous].valid &&
          routes[previous].expert / experts_per_node == node)
        first = false;
    if (!first)
      continue;
    std::uint32_t fanouts = 0;
    for (std::uint32_t other = slot; other < top_k; ++other)
      fanouts += routes[other].valid &&
                 routes[other].expert / experts_per_node == node;
    emit(node, fanouts);
  }
}

// Within one (source, window, destination-node) token list, first divide into
// complete packets, then stripe packets over lanes. Never split a token's
// fanouts between packets or infer terminal packets from the fanout count.
struct PacketPartition {
  std::uint32_t rows;
  std::uint32_t lanes;

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  packets(std::uint64_t token_count) const {
    return ceil_div(token_count, rows);
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  lane_packets(std::uint64_t token_count, std::uint32_t lane) const {
    const auto count = packets(token_count);
    return count > lane ? ceil_div(count - lane, lanes) : 0U;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint64_t
  token_begin(std::uint32_t lane, std::uint64_t lane_sequence) const {
    return (lane_sequence * lanes + lane) * rows;
  }

  WEFTMOE_STATIC_PLAN_HD constexpr std::uint32_t
  token_count(std::uint64_t tokens, std::uint32_t lane,
              std::uint64_t lane_sequence) const {
    return block_rows(tokens, lane_sequence * lanes + lane, rows);
  }
};

struct TokenRecord {
  std::uint64_t token;
  std::uint32_t fanout_begin;
  std::uint32_t fanout_count;
};

struct FanoutRecord {
  std::uint64_t logical_mblock;
  std::uint64_t source_route;
  std::uint32_t token_index;
  std::uint32_t destination_rank;
  std::uint32_t row;
  float gate;
};

struct PacketHeader {
  std::uint64_t sequence;
  std::uint64_t window;
  std::uint64_t retire_target;
  std::uint32_t rows;
  std::uint32_t fanouts; // Dispatch only; Return retires exactly rows units.
};

struct ReturnRecord {
  std::uint64_t source_route;
};

static_assert(sizeof(TokenRecord) == 16U);
static_assert(sizeof(FanoutRecord) == 32U);
static_assert(sizeof(PacketHeader) == 32U);
static_assert(sizeof(ReturnRecord) == 8U);

} // namespace weftmoe::cuda::static_forward

#undef WEFTMOE_STATIC_PLAN_HD
