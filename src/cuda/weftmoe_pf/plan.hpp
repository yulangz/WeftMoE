#pragma once

// Packet geometry is part of the wire contract, including in translation
// units that include the planner before the Forward GEMM tuning header.
#include "cuda/communication/tuning_config.hpp"

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

#ifndef WEFTMOE_FWD_STATIC_V4_NETWORK_LANES
#define WEFTMOE_FWD_STATIC_V4_NETWORK_LANES 2
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP
#define WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP 0
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_NETWORK_MIN_DEPTH
#define WEFTMOE_FWD_STATIC_V4_NETWORK_MIN_DEPTH 8
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS
#define WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS 32
#endif
static_assert(WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP <= 16);
static_assert(WEFTMOE_FWD_STATIC_V4_NETWORK_MIN_DEPTH >= 8);
static_assert(WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS > 0 &&
              WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS <= 32);
#ifndef WEFTMOE_FWD_STATIC_V4_PACKET_ROWS
#define WEFTMOE_FWD_STATIC_V4_PACKET_ROWS 32
#endif
inline constexpr PacketPartition kNetworkPackets{
    WEFTMOE_FWD_STATIC_V4_PACKET_ROWS, WEFTMOE_FWD_STATIC_V4_NETWORK_LANES};
static_assert(kNetworkPackets.rows > 0U && kNetworkPackets.lanes > 0U);
#ifndef WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS
#define WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS WEFTMOE_FWD_STATIC_V4_PACKET_ROWS
#endif
inline constexpr PacketPartition kNetworkReturnPackets{
    WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS, kNetworkPackets.lanes};
static_assert(kNetworkReturnPackets.rows > 0U);

// Short sources need enough packets to use the fixed network lanes. All
// peers derive this partition from the immutable source epoch, not their own
// token count. Base the choice on actual window work even when the pool has
// clipped B to max_tokens; large windows retain the configured throughput Q.
WEFTMOE_STATIC_PLAN_HD constexpr PacketPartition
return_partition(WindowShape shape, std::uint64_t tokens) {
  auto result = kNetworkReturnPackets;
  const auto window_tokens = tokens < shape.tokens ? tokens : shape.tokens;
  const auto needed = ceil_div(window_tokens * shape.top_k, result.lanes);
  std::uint64_t rows = 1U;
  while (rows < needed && rows < result.rows)
    rows *= 2U;
  if (rows < result.rows)
    result.rows = static_cast<std::uint32_t>(rows);
  return result;
}

struct NodeToken {
  std::uint64_t token;
  std::uint32_t fanouts;
};

// Immutable directories, separate from reusable network packet storage.
struct DispatchPacketPlan {
  std::uint64_t window;
  std::uint64_t token_begin; // Index into the source's frozen NodeToken array.
  std::uint32_t tokens;
  std::uint32_t fanouts;
};

struct DispatchPlanView {
  std::uint64_t *channel_offsets; // [nodes * lanes + 1], self channels empty.
  DispatchPacketPlan *packets;    // [sum(channel terminal sequences)]
  std::uint32_t *consumer_rows;   // [packets, local_world]
};

struct ReturnPacketPlan {
  std::uint64_t window;
  std::uint64_t route_begin; // Contiguous interval in source route_order.
  std::uint32_t rows;       // Complete H-dimensional contributions, not fragments.
};

struct ReturnPlanView {
  std::uint64_t *packet_prefix;   // [nodes * lanes, local_windows + 1]
  std::uint64_t *channel_offsets; // [nodes * lanes + 1], self channels empty.
  ReturnPacketPlan *packets;
  PacketPartition partition = kNetworkReturnPackets;
};

// Source-local immutable metadata. Activation payload is stored separately.
struct SourcePlanView {
  WindowShape shape;
  PacketPartition packets;
  std::uint32_t source_rank;
  std::uint32_t experts;
  std::uint32_t nodes;
  std::uint64_t tokens;
  const std::int32_t *selected_experts;
  const std::uint8_t *valid;
  std::uint64_t *expert_prefix; // [local_windows, experts+1]
  std::uint64_t *node_prefix;   // [local_windows, nodes+1]
  std::uint64_t *route_order;   // [local_windows, B*K]
  NodeToken *node_tokens;      // [local_windows, B*K]
  std::uint64_t *packet_prefix; // [nodes, lanes, local_windows+1]
  unsigned *error;

  WEFTMOE_STATIC_PLAN_HD std::uint64_t local_windows() const {
    return ceil_div(tokens, shape.tokens);
  }
  WEFTMOE_STATIC_PLAN_HD std::uint64_t window_capacity() const {
    return std::uint64_t(shape.tokens) * shape.top_k;
  }
  WEFTMOE_STATIC_PLAN_HD unsigned scratch_words(bool canonical = false) const {
    const auto counts = experts + nodes;
    return canonical ? ((counts + 1U) & ~1U) + 2U * experts : counts;
  }
};

struct WindowMblock {
  std::uint32_t expert;
  std::uint32_t expert_row;
  std::uint32_t rows;
};

} // namespace weftmoe::cuda::static_forward

#undef WEFTMOE_STATIC_PLAN_HD
