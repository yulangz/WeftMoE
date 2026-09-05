__global__ void prepare_routes_kernel(
    const std::int32_t *selected, std::uint64_t routes,
    std::uint64_t route_count_key_stride, std::uint32_t num_experts,
    const std::uint32_t *expert_to_rank, const std::uint32_t *rank_to_node,
    const std::uint32_t *rank_to_local, std::uint64_t *keys,
    std::uint64_t *indices, std::uint32_t *expert_id,
    std::uint32_t *dst_rank, std::uint32_t *dst_node,
    std::uint32_t *dst_local, std::uint64_t *ones, DeviceStatus *status) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (route >= routes)
    return;
  const auto raw = selected[route];
  indices[route] = route;
  ones[route] = 1;
  if (raw < 0 || static_cast<std::uint32_t>(raw) >= num_experts) {
    const auto packed =
        (static_cast<unsigned long long>(route) << 32U) |
        static_cast<std::uint32_t>(raw);
    atomicMin(&status->out_of_range_packed, packed);
    keys[route] = ~std::uint64_t{0};
    expert_id[route] = num_experts;
    dst_rank[route] = 0;
    dst_node[route] = 0;
    dst_local[route] = 0;
    return;
  }
  const auto expert = static_cast<std::uint32_t>(raw);
  const auto destination = expert_to_rank[expert];
  keys[route] = static_cast<std::uint64_t>(expert) * route_count_key_stride +
                route;
  expert_id[route] = expert;
  dst_rank[route] = destination;
  dst_node[route] = rank_to_node[destination];
  dst_local[route] = rank_to_local[destination];
}

__global__ void validate_and_describe_tokens_kernel(
    const std::int32_t *selected, std::uint64_t num_tokens,
    std::uint32_t top_k, std::uint32_t num_experts,
    const std::uint32_t *expert_to_rank, const std::uint32_t *rank_to_node,
    std::uint32_t num_nodes, std::uint32_t mask_words,
    std::uint32_t *combine_order, std::uint64_t *masks,
    std::uint64_t *packet_count, DeviceStatus *status) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (token >= num_tokens)
    return;
  const auto base = token * top_k;
  for (std::uint32_t word = 0; word < mask_words; ++word)
    masks[token * mask_words + word] = 0;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const auto raw = selected[base + slot];
    if (raw < 0 || static_cast<std::uint32_t>(raw) >= num_experts)
      continue;
    for (std::uint32_t prior = 0; prior < slot; ++prior) {
      if (selected[base + prior] == raw) {
        const auto flat = base + slot;
        const auto packed =
            (static_cast<unsigned long long>(flat) << 32U) |
            static_cast<std::uint32_t>(raw);
        atomicMin(&status->duplicate_packed, packed);
      }
    }
    const auto destination = expert_to_rank[static_cast<std::uint32_t>(raw)];
    const auto node = rank_to_node[destination];
    masks[token * mask_words + node / 64U] |=
        std::uint64_t{1} << (node % 64U);
  }
  for (std::uint32_t position = 0; position < top_k; ++position) {
    std::uint32_t best = top_k;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      bool used = false;
      for (std::uint32_t prior = 0; prior < position; ++prior)
        used = used || combine_order[base + prior] == slot;
      if (used)
        continue;
      if (best == top_k || selected[base + slot] < selected[base + best] ||
          (selected[base + slot] == selected[base + best] && slot < best))
        best = slot;
    }
    combine_order[base + position] = best;
  }
  for (std::uint32_t node = 0; node < num_nodes; ++node) {
    if ((masks[token * mask_words + node / 64U] >> (node % 64U)) & 1U)
      atomicAdd(reinterpret_cast<unsigned long long *>(&packet_count[node]),
                1ULL);
  }
}

__global__ void extract_sorted_experts_kernel(const std::uint64_t *keys,
                                               std::uint32_t *experts,
                                               std::uint64_t routes,
                                               std::uint64_t stride) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < routes)
    experts[index] = static_cast<std::uint32_t>(keys[index] / stride);
}

__global__ void histogram_and_scatter_kernel(
    const std::uint32_t *sorted_experts, const std::uint64_t *sorted_indices,
    const std::uint64_t *inclusive, std::uint64_t routes,
    std::uint32_t num_experts, std::uint64_t *histogram,
    std::uint64_t *local_offset) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= routes)
    return;
  const auto expert = sorted_experts[index];
  if (expert >= num_experts)
    return;
  local_offset[sorted_indices[index]] = inclusive[index] - 1;
  if (index + 1 == routes || sorted_experts[index + 1] != expert)
    histogram[expert] = inclusive[index];
}

struct DeviceMaterializeArgs {
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t experts;
  std::uint32_t nodes;
  std::uint32_t top_k;
  std::uint32_t mask_words;
  std::uint32_t m_tile;
  std::uint64_t tokens;
  std::uint64_t routes;
  std::uint64_t max_tokens;
  std::uint64_t max_incoming_rows;
  bool has_expert_capacity;
  bool has_workspace_capacity;
  const std::uint64_t *count_matrix;
  const std::uint64_t *summary_matrix;
  const std::uint64_t *local_histogram;
  const std::uint64_t *local_summary;
  const std::uint32_t *expert_to_rank;
  const std::uint32_t *rank_to_node;
  const std::uint32_t *rank_to_local;
  const std::uint64_t *expert_capacity;
  const std::uint64_t *workspace_capacity;
  const std::uint32_t *expert_id;
  const std::uint64_t *local_offset;
  const std::uint32_t *route_dst_rank;
  const std::uint32_t *route_dst_node;
  const std::uint32_t *route_dst_local;
  const std::uint32_t *combine_order;
  const std::uint64_t *masks;
  std::uint64_t *expert_total;
  std::uint64_t *expert_base;
  std::uint64_t *expert_tile_base;
  std::uint64_t *route_dst_row;
  std::uint32_t *route_dst_mtile;
  std::uint32_t *owned_expert_ids;
  std::uint64_t *owned_expert_count;
  std::uint64_t *owned_expert_prefix;
  std::uint64_t *mtile_offsets;
  std::uint32_t *mtile_rows;
  DeviceIncomingRowRecord *incoming_rows;
  DeviceReverseRouteRecord *reverse_routes;
  DevicePacketHeaderRecord *packet_headers;
  std::uint64_t *packet_fanout_offsets;
  DeviceFanoutRecord *packet_fanout;
  std::uint64_t *packet_primary_tile;
  std::uint32_t *packet_primary_expert;
  std::uint32_t *packet_primary_rank;
  std::uint64_t *local_route_order;
  std::uint64_t *sort_keys_in;
  std::uint64_t *sort_keys_out;
  std::uint64_t *sort_indices_in;
  std::uint64_t *sort_indices_out;
  std::uint64_t *wave_flags;
  std::uint64_t *wave_ids;
  std::uint64_t *dispatch_wave_offsets;
  std::uint32_t *dispatch_wave_destination_node;
  std::uint32_t *dispatch_wave_channel;
  std::uint64_t *dispatch_wave_first_primary_tile;
  std::uint64_t *dispatch_wave_last_primary_tile;
  std::uint64_t *dispatch_wave_packet_count;
  std::uint64_t *dispatch_wave_payload_bytes;
  DeviceSignalRecord *outgoing_signals;
  DeviceSignalRecord *incoming_signals;
  DeviceMaterializationStatus *status;
};

__device__ bool device_add(std::uint64_t lhs, std::uint64_t rhs,
                           std::uint64_t &out) {
  if (rhs > ~std::uint64_t{0} - lhs)
    return false;
  out = lhs + rhs;
  return true;
}

__device__ void materialization_fail(DeviceMaterializationStatus *status,
                                     MaterializationError error,
                                     std::uint64_t field,
                                     std::uint64_t expected,
                                     std::uint64_t actual) {
  status->error = static_cast<unsigned long long>(error);
  status->field = field;
  status->expected = expected;
  status->actual = actual;
}

__device__ std::uint32_t rank_at_coordinate(const DeviceMaterializeArgs &a,
                                            std::uint32_t node,
                                            std::uint32_t local_rank) {
  for (std::uint32_t rank = 0; rank < a.world; ++rank) {
    if (a.rank_to_node[rank] == node &&
        a.rank_to_local[rank] == local_rank)
      return rank;
  }
  return a.world;
}

// Small checked planning phase. It establishes all variable output sizes and
// canonical prefixes before any parallel writer uses them as indices.
[[maybe_unused]] __global__ void
materialize_prefix_plan_kernel(DeviceMaterializeArgs a) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *a.status = DeviceMaterializationStatus{};
  std::uint64_t max_routes_per_rank = 0;
  if (a.top_k != 0 && a.max_tokens > ~std::uint64_t{0} / a.top_k) {
    materialization_fail(a.status, MaterializationError::ArithmeticOverflow, 1,
                         ~std::uint64_t{0}, a.max_tokens);
    return;
  }
  max_routes_per_rank = a.max_tokens * a.top_k;

  for (std::uint32_t source = 0; source < a.world; ++source) {
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      const auto value = a.count_matrix[
          static_cast<std::uint64_t>(source) * a.experts + expert];
      if (value > max_routes_per_rank) {
        materialization_fail(a.status,
                             MaterializationError::InvalidCollectiveValue, 2,
                             max_routes_per_rank, value);
        return;
      }
      if (source == a.rank && value != a.local_histogram[expert]) {
        materialization_fail(a.status,
                             MaterializationError::CountRowMismatch, expert,
                             a.local_histogram[expert], value);
        return;
      }
    }
    for (std::uint32_t node = 0; node < a.nodes; ++node) {
      const auto value = a.summary_matrix[
          static_cast<std::uint64_t>(source) * a.nodes + node];
      if (value > a.max_tokens) {
        materialization_fail(a.status,
                             MaterializationError::InvalidCollectiveValue, 3,
                             a.max_tokens, value);
        return;
      }
      if (source == a.rank && value != a.local_summary[node]) {
        materialization_fail(a.status,
                             MaterializationError::SummaryRowMismatch, node,
                             a.local_summary[node], value);
        return;
      }
    }
  }

  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    std::uint64_t total = 0;
    for (std::uint32_t source = 0; source < a.world; ++source) {
      const auto value = a.count_matrix[
          static_cast<std::uint64_t>(source) * a.experts + expert];
      if (!device_add(total, value, total)) {
        materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                             4, ~std::uint64_t{0}, value);
        return;
      }
    }
    a.expert_total[expert] = total;
    if (a.has_expert_capacity && total > a.expert_capacity[expert]) {
      materialization_fail(a.status, MaterializationError::ExpertCapacity,
                           expert, a.expert_capacity[expert], total);
      return;
    }
  }

  for (std::uint32_t destination = 0; destination < a.world; ++destination) {
    std::uint64_t prefix = 0;
    std::uint64_t tile_prefix = 0;
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      if (a.expert_to_rank[expert] != destination)
        continue;
      a.expert_base[expert] = prefix;
      a.expert_tile_base[expert] = tile_prefix;
      if (!device_add(prefix, a.expert_total[expert], prefix)) {
        materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                             5, ~std::uint64_t{0}, a.expert_total[expert]);
        return;
      }
      const auto count = a.expert_total[expert];
      const auto expert_tiles = count / a.m_tile + (count % a.m_tile != 0);
      if (!device_add(tile_prefix, expert_tiles, tile_prefix)) {
        materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                             44, ~std::uint64_t{0}, expert_tiles);
        return;
      }
    }
    if (a.has_workspace_capacity &&
        prefix > a.workspace_capacity[destination]) {
      materialization_fail(a.status, MaterializationError::WorkspaceCapacity,
                           destination, a.workspace_capacity[destination],
                           prefix);
      return;
    }
    if (destination == a.rank && prefix > a.max_incoming_rows) {
      materialization_fail(a.status, MaterializationError::OutputCapacity, 6,
                           a.max_incoming_rows, prefix);
      return;
    }
  }

  std::uint64_t owned = 0;
  std::uint64_t incoming = 0;
  std::uint64_t tiles = 0;
  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    ++owned;
    if (!device_add(incoming, a.expert_total[expert], incoming)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           7, ~std::uint64_t{0}, a.expert_total[expert]);
      return;
    }
    const auto count = a.expert_total[expert];
    const auto expert_tiles = count / a.m_tile + (count % a.m_tile != 0);
    if (!device_add(tiles, expert_tiles, tiles)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           8, ~std::uint64_t{0}, expert_tiles);
      return;
    }
  }
  std::uint64_t packets = 0;
  for (std::uint32_t node = 0; node < a.nodes; ++node) {
    if (!device_add(packets, a.local_summary[node], packets)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           9, ~std::uint64_t{0}, a.local_summary[node]);
      return;
    }
  }
  if (owned > a.experts || incoming > a.max_incoming_rows ||
      tiles > a.max_incoming_rows || packets > a.routes) {
    materialization_fail(a.status, MaterializationError::OutputCapacity, 10,
                         a.max_incoming_rows, incoming);
    return;
  }
  a.status->owned_experts = owned;
  a.status->incoming_rows = incoming;
  a.status->mtile_count = tiles;
  a.status->packet_count = packets;
  a.status->fanout_count = a.routes;

  std::uint64_t owned_index = 0;
  std::uint64_t incoming_prefix = 0;
  std::uint64_t tile_index = 0;
  a.owned_expert_prefix[0] = 0;
  a.mtile_offsets[0] = 0;
  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    a.owned_expert_ids[owned_index] = expert;
    a.owned_expert_count[owned_index] = a.expert_total[expert];
    incoming_prefix += a.expert_total[expert];
    a.owned_expert_prefix[owned_index + 1] = incoming_prefix;
    const auto count = a.expert_total[expert];
    tile_index += count / a.m_tile + (count % a.m_tile != 0);
    a.mtile_offsets[owned_index + 1] = tile_index;
    ++owned_index;
  }

}

// Packet order is node-major and then token-major. One block owns a complete
// destination node, so a deterministic block scan can compact token flags and
// fan-out counts without atomics or a serial route walk. The per-node bases are
// derived from immutable histogram/summary rows and therefore do not depend on
// block scheduling.
__global__ void materialize_packet_plan_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockScan = cub::BlockScan<unsigned long long, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ unsigned long long packet_carry;
  __shared__ unsigned long long fanout_carry;
  __shared__ unsigned long long chunk_packet_base;
  __shared__ unsigned long long chunk_fanout_base;
  __shared__ unsigned long long packet_base;
  __shared__ unsigned long long fanout_base;
  __shared__ std::uint32_t proxy_rank;

  const auto node = static_cast<std::uint32_t>(blockIdx.x);
  if (node >= a.nodes || a.status->error != 0)
    return;
  if (threadIdx.x == 0) {
    packet_base = 0;
    fanout_base = 0;
    packet_carry = 0;
    fanout_carry = 0;
    for (std::uint32_t prior = 0; prior < node; ++prior)
      packet_base += a.local_summary[prior];
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      const auto destination = a.expert_to_rank[expert];
      if (a.rank_to_node[destination] < node)
        fanout_base += a.local_histogram[expert];
    }
    proxy_rank = rank_at_coordinate(a, node, a.rank_to_local[a.rank]);
  }
  __syncthreads();

  for (std::uint64_t chunk = 0; chunk < a.tokens; chunk += kThreads) {
    const auto token = chunk + threadIdx.x;
    unsigned long long present = 0;
    unsigned long long fanouts = 0;
    if (token < a.tokens) {
      present = static_cast<unsigned long long>(
          (a.masks[token * a.mask_words + node / 64U] >> (node % 64U)) & 1U);
      const auto route_base = token * a.top_k;
      for (std::uint32_t slot = 0; slot < a.top_k; ++slot)
        fanouts += a.route_dst_node[route_base + slot] == node ? 1ULL : 0ULL;
    }

    unsigned long long packet_prefix = 0;
    unsigned long long packet_aggregate = 0;
    BlockScan(scan_storage).ExclusiveSum(present, packet_prefix,
                                         packet_aggregate);
    __syncthreads();
    if (threadIdx.x == 0) {
      chunk_packet_base = packet_carry;
      packet_carry += packet_aggregate;
    }
    __syncthreads();

    unsigned long long fanout_prefix = 0;
    unsigned long long fanout_aggregate = 0;
    BlockScan(scan_storage).ExclusiveSum(fanouts, fanout_prefix,
                                         fanout_aggregate);
    __syncthreads();
    if (threadIdx.x == 0) {
      chunk_fanout_base = fanout_carry;
      fanout_carry += fanout_aggregate;
    }
    __syncthreads();

    if (present != 0) {
      const auto packet = packet_base + chunk_packet_base + packet_prefix;
      const auto fanout = fanout_base + chunk_fanout_base + fanout_prefix;
      a.packet_headers[packet] = DevicePacketHeaderRecord{
          chunk_packet_base + packet_prefix, token, a.rank, node, proxy_rank,
          0};
      a.packet_fanout_offsets[packet] = fanout;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && node + 1U == a.nodes)
    a.packet_fanout_offsets[packet_base + packet_carry] =
        fanout_base + fanout_carry;
}

__device__ void materialization_fail_once(DeviceMaterializationStatus *status,
                                          MaterializationError error,
                                          std::uint64_t field,
                                          std::uint64_t expected,
                                          std::uint64_t actual) {
  if (atomicCAS(&status->error, 0ULL,
                static_cast<unsigned long long>(error)) == 0ULL) {
    status->field = field;
    status->expected = expected;
    status->actual = actual;
  }
}

[[maybe_unused]] __global__ void
materialize_plan_init_kernel(DeviceMaterializeArgs a) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *a.status = DeviceMaterializationStatus{};
    if (a.top_k == 0 || a.max_tokens > ~std::uint64_t{0} / a.top_k) {
      materialization_fail_once(a.status,
                                MaterializationError::ArithmeticOverflow, 1,
                                ~std::uint64_t{0}, a.max_tokens);
      return;
    }
    const auto max_routes = a.max_tokens * a.top_k;
    if (a.world != 0 && max_routes > ~std::uint64_t{0} / a.world)
      materialization_fail_once(a.status,
                                MaterializationError::ArithmeticOverflow, 2,
                                ~std::uint64_t{0}, max_routes);
  }
}

[[maybe_unused]] __global__ void
materialize_expert_totals_kernel(DeviceMaterializeArgs a) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (expert >= a.experts || a.status->error != 0)
    return;
  if (a.top_k == 0 || a.max_tokens > ~std::uint64_t{0} / a.top_k) {
    materialization_fail_once(a.status,
                              MaterializationError::ArithmeticOverflow, 1,
                              ~std::uint64_t{0}, a.max_tokens);
    return;
  }
  const auto max_routes = a.max_tokens * a.top_k;
  std::uint64_t total = 0;
  for (std::uint32_t source = 0; source < a.world; ++source) {
    const auto value = a.count_matrix[
        static_cast<std::uint64_t>(source) * a.experts + expert];
    if (value > max_routes || value > ~std::uint64_t{0} - total) {
      materialization_fail_once(
          a.status, value > max_routes
                        ? MaterializationError::InvalidCollectiveValue
                        : MaterializationError::ArithmeticOverflow,
          expert, max_routes, value);
      return;
    }
    if (source == a.rank && value != a.local_histogram[expert]) {
      materialization_fail_once(a.status,
                                MaterializationError::CountRowMismatch,
                                expert, a.local_histogram[expert], value);
      return;
    }
    total += value;
  }
  if (a.has_expert_capacity && total > a.expert_capacity[expert]) {
    materialization_fail_once(a.status, MaterializationError::ExpertCapacity,
                              expert, a.expert_capacity[expert], total);
    return;
  }
  a.expert_total[expert] = total;
}

[[maybe_unused]] __global__ void
materialize_expert_prefix_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (expert >= a.experts || a.status->error != 0)
    return;
  const auto owner = a.expert_to_rank[expert];
  unsigned long long destination_partial = 0;
  unsigned long long destination_tile_partial = 0;
  unsigned long long owned_prefix_partial = 0;
  unsigned long long tile_prefix_partial = 0;
  unsigned long long owned_index_partial = 0;
  for (std::uint32_t prior = threadIdx.x; prior < expert;
       prior += kThreads) {
    if (a.expert_to_rank[prior] == owner) {
      destination_partial += a.expert_total[prior];
      const auto count = a.expert_total[prior];
      destination_tile_partial +=
          count / a.m_tile + (count % a.m_tile != 0);
    }
    if (a.expert_to_rank[prior] == a.rank) {
      ++owned_index_partial;
      owned_prefix_partial += a.expert_total[prior];
      const auto count = a.expert_total[prior];
      tile_prefix_partial += count / a.m_tile + (count % a.m_tile != 0);
    }
  }

  const auto destination_prefix =
      BlockReduce(reduce_storage).Sum(destination_partial);
  __syncthreads();
  const auto destination_tile_prefix =
      BlockReduce(reduce_storage).Sum(destination_tile_partial);
  __syncthreads();
  const auto owned_prefix =
      BlockReduce(reduce_storage).Sum(owned_prefix_partial);
  __syncthreads();
  const auto tile_prefix = BlockReduce(reduce_storage).Sum(tile_prefix_partial);
  __syncthreads();
  const auto owned_index = BlockReduce(reduce_storage).Sum(owned_index_partial);
  __syncthreads();
  if (threadIdx.x == 0) {
    a.expert_base[expert] = destination_prefix;
    a.expert_tile_base[expert] = destination_tile_prefix;
    if (owner == a.rank) {
      a.owned_expert_ids[owned_index] = expert;
      a.owned_expert_count[owned_index] = a.expert_total[expert];
      a.owned_expert_prefix[owned_index] = owned_prefix;
      a.mtile_offsets[owned_index] = tile_prefix;
    }
  }
}

[[maybe_unused]] __global__ void materialize_destination_capacity_kernel(
    DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto destination = static_cast<std::uint32_t>(blockIdx.x);
  if (destination >= a.world || a.status->error != 0)
    return;
  unsigned long long rows_partial = 0;
  for (std::uint32_t expert = threadIdx.x; expert < a.experts;
       expert += kThreads)
    if (a.expert_to_rank[expert] == destination)
      rows_partial += a.expert_total[expert];
  const auto rows = BlockReduce(reduce_storage).Sum(rows_partial);
  if (threadIdx.x == 0) {
    if (a.has_workspace_capacity && rows > a.workspace_capacity[destination])
      materialization_fail_once(
          a.status, MaterializationError::WorkspaceCapacity, destination,
          a.workspace_capacity[destination], rows);
    if (destination == a.rank && rows > a.max_incoming_rows)
      materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                                destination, a.max_incoming_rows, rows);
  }
}

[[maybe_unused]] __global__ void
materialize_plan_finalize_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  if (blockIdx.x != 0 || a.status->error != 0)
    return;
  unsigned long long owned_partial = 0;
  unsigned long long incoming_partial = 0;
  unsigned long long tiles_partial = 0;
  for (std::uint32_t expert = threadIdx.x; expert < a.experts;
       expert += kThreads) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    ++owned_partial;
    incoming_partial += a.expert_total[expert];
    const auto count = a.expert_total[expert];
    tiles_partial += count / a.m_tile + (count % a.m_tile != 0);
  }
  unsigned long long packets_partial = 0;
  for (std::uint32_t node = threadIdx.x; node < a.nodes;
       node += kThreads) {
    const auto value = a.summary_matrix[
        static_cast<std::uint64_t>(a.rank) * a.nodes + node];
    if (value != a.local_summary[node] || value > a.max_tokens) {
      materialization_fail_once(a.status,
                                MaterializationError::SummaryRowMismatch,
                                node, a.local_summary[node], value);
    } else {
      packets_partial += value;
    }
  }
  const auto owned = BlockReduce(reduce_storage).Sum(owned_partial);
  __syncthreads();
  const auto incoming = BlockReduce(reduce_storage).Sum(incoming_partial);
  __syncthreads();
  const auto tiles = BlockReduce(reduce_storage).Sum(tiles_partial);
  __syncthreads();
  const auto packets = BlockReduce(reduce_storage).Sum(packets_partial);
  __syncthreads();
  if (threadIdx.x == 0 && a.status->error == 0) {
    if (incoming > a.max_incoming_rows || tiles > a.max_incoming_rows ||
        packets > a.routes) {
      materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                                10, a.max_incoming_rows, incoming);
      return;
    }
    a.owned_expert_prefix[owned] = incoming;
    a.mtile_offsets[owned] = tiles;
    a.status->owned_experts = owned;
    a.status->incoming_rows = incoming;
    a.status->mtile_count = tiles;
    a.status->packet_count = packets;
    a.status->fanout_count = a.routes;
  }
}

__global__ void materialize_routes_kernel(DeviceMaterializeArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (a.status->error != 0 || route >= a.routes)
    return;
  const auto expert = a.expert_id[route];
  std::uint64_t source_prefix = 0;
  for (std::uint32_t source = 0; source < a.rank; ++source)
    source_prefix += a.count_matrix[
        static_cast<std::uint64_t>(source) * a.experts + expert];
  const auto row =
      a.expert_base[expert] + source_prefix + a.local_offset[route];
  a.route_dst_row[route] = row;
  const auto destination_tile =
      a.expert_tile_base[expert] +
      (row - a.expert_base[expert]) / a.m_tile;
  if (destination_tile > UINT32_MAX) {
    materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                              43, UINT32_MAX, destination_tile);
    return;
  }
  a.route_dst_mtile[route] = static_cast<std::uint32_t>(destination_tile);
  a.reverse_routes[route] = DeviceReverseRouteRecord{
      route / a.top_k, row, a.rank,
      static_cast<std::uint32_t>(route % a.top_k), expert,
      a.route_dst_rank[route]};
}

__global__ void materialize_incoming_kernel(DeviceMaterializeArgs a) {
  const auto pair = static_cast<std::uint64_t>(blockIdx.x);
  const auto expert = static_cast<std::uint32_t>(pair / a.world);
  const auto source = static_cast<std::uint32_t>(pair % a.world);
  if (a.status->error != 0 || expert >= a.experts ||
      a.expert_to_rank[expert] != a.rank)
    return;
  std::uint64_t source_prefix = 0;
  for (std::uint32_t prior = 0; prior < source; ++prior)
    source_prefix += a.count_matrix[
        static_cast<std::uint64_t>(prior) * a.experts + expert];
  const auto count = a.count_matrix[
      static_cast<std::uint64_t>(source) * a.experts + expert];
  for (std::uint64_t offset = threadIdx.x; offset < count;
       offset += blockDim.x) {
    const auto row = a.expert_base[expert] + source_prefix + offset;
    a.incoming_rows[row] =
        DeviceIncomingRowRecord{offset, row, source, expert};
  }
}

__global__ void materialize_mtiles_kernel(DeviceMaterializeArgs a) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (a.status->error != 0 || expert >= a.experts ||
      a.expert_to_rank[expert] != a.rank)
    return;
  std::uint64_t owned_index = 0;
  for (std::uint32_t prior = 0; prior < expert; ++prior)
    owned_index += a.expert_to_rank[prior] == a.rank;
  const auto count = a.expert_total[expert];
  const auto tiles = count / a.m_tile + (count % a.m_tile != 0);
  for (std::uint64_t tile = threadIdx.x; tile < tiles;
       tile += blockDim.x) {
    const auto first = tile * a.m_tile;
    const auto remaining = count - first;
    a.mtile_rows[a.mtile_offsets[owned_index] + tile] =
        static_cast<std::uint32_t>(remaining < a.m_tile ? remaining
                                                        : a.m_tile);
  }
}

template <bool IncludeReturnOrder = false>
__global__ void materialize_fanout_kernel(DeviceMaterializeArgs a) {
  const auto packet = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (a.status->error != 0 || packet >= a.status->packet_count)
    return;
  const auto header = a.packet_headers[packet];
  auto output = a.packet_fanout_offsets[packet];
  const auto base = header.local_token_id * a.top_k;
  for (std::uint32_t position = 0; position < a.top_k; ++position) {
    const auto slot = a.combine_order[base + position];
    const auto route = base + slot;
    if (a.route_dst_node[route] != header.destination_node)
      continue;
    a.packet_fanout[output++] = DeviceFanoutRecord{
        a.route_dst_row[route],
        IncludeReturnOrder ? a.sort_keys_in[route] : 0ULL, slot,
        a.route_dst_local[route], a.expert_id[route],
        a.route_dst_mtile[route]};
  }
}

// Packet storage/identity is intentionally left untouched.  These kernels
// build a transport consumption permutation whose explicit lexicographic key
// is (destination node, primary M tile, expert, destination rank, token).
// Five stable radix passes implement that full runtime-width key without
// packing unrelated runtime dimensions into a fixed-width integer.
__global__ void materialize_packet_primary_kernel(DeviceMaterializeArgs a) {
  const auto packet = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (a.status->error != 0 || packet >= a.routes)
    return;
  a.sort_indices_in[packet] = packet;
  if (packet >= a.status->packet_count) {
    a.packet_primary_tile[packet] = ~std::uint64_t{0};
    a.packet_primary_expert[packet] = ~std::uint32_t{0};
    a.packet_primary_rank[packet] = ~std::uint32_t{0};
    return;
  }
  const auto begin = a.packet_fanout_offsets[packet];
  const auto end = a.packet_fanout_offsets[packet + 1];
  if (begin >= end) {
    materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                              40, 1, 0);
    return;
  }
  std::uint64_t primary_tile = ~std::uint64_t{0};
  std::uint32_t primary_expert = ~std::uint32_t{0};
  std::uint32_t primary_rank = ~std::uint32_t{0};
  for (auto index = begin; index < end; ++index) {
    const auto target = a.packet_fanout[index];
    if (target.expert_id >= a.experts) {
      materialization_fail_once(a.status,
                                MaterializationError::InvalidCollectiveValue,
                                41, a.experts, target.expert_id);
      return;
    }
    const auto base = a.expert_base[target.expert_id];
    if (target.dst_row < base) {
      materialization_fail_once(a.status,
                                MaterializationError::InvalidCollectiveValue,
                                42, base, target.dst_row);
      return;
    }
    const auto tile = (target.dst_row - base) / a.m_tile;
    const auto destination_rank = a.expert_to_rank[target.expert_id];
    if (tile < primary_tile ||
        (tile == primary_tile && target.expert_id < primary_expert) ||
        (tile == primary_tile && target.expert_id == primary_expert &&
         destination_rank < primary_rank)) {
      primary_tile = tile;
      primary_expert = target.expert_id;
      primary_rank = destination_rank;
    }
  }
  a.packet_primary_tile[packet] = primary_tile;
  a.packet_primary_expert[packet] = primary_expert;
  a.packet_primary_rank[packet] = primary_rank;
}

enum class WaveSortField : std::uint32_t {
  Token,
  DestinationRank,
  Expert,
  MTile,
  DestinationNode,
};

template <WaveSortField Field>
__global__ void materialize_wave_sort_keys_kernel(
    DeviceMaterializeArgs a, const std::uint64_t *packet_indices) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.status->packet_count) {
    if (position < a.routes)
      a.sort_keys_in[position] = ~std::uint64_t{0};
    return;
  }
  const auto packet = packet_indices[position];
  std::uint64_t key = 0;
  if constexpr (Field == WaveSortField::Token)
    key = a.packet_headers[packet].local_token_id;
  else if constexpr (Field == WaveSortField::DestinationRank)
    key = a.packet_primary_rank[packet];
  else if constexpr (Field == WaveSortField::Expert) {
    key = a.packet_primary_expert[packet];
    // Preserve complete expert runs, but transpose the two four-GPU halves of
    // an eight-GPU node.  The natural global-expert order leaves the same
    // middle rails (notably local ranks 4 and 6) on the late side of multiple
    // 128-packet wave/context boundaries.  This bounded permutation keeps the
    // Return-friendly expert-major stream while distributing those boundaries
    // in three-bit bit-reversed order.  Other topology widths retain their
    // canonical order.
    if (a.nodes != 0U && a.world == a.nodes * 8U) {
      const auto local =
          a.rank_to_local[a.packet_primary_rank[packet]];
      const auto priority = ((local & 1U) << 2U) | (local & 2U) |
                            ((local & 4U) >> 2U);
      key += static_cast<std::uint64_t>(priority) * a.experts;
    }
  }
  else if constexpr (Field == WaveSortField::MTile)
    key = a.packet_primary_tile[packet];
  else
    key = a.packet_headers[packet].destination_node;
  a.sort_keys_in[position] = key;
}

// A one-node route has a bounded mixed-radix key, so the five stable packet
// sorts can be represented by one exact key.  Multi-node keeps the existing
// field-by-field implementation below; this specialization neither changes
// its wire order nor enters its synchronization protocol.
__global__ void materialize_one_node_wave_sort_keys_kernel(
    DeviceMaterializeArgs a, const std::uint64_t *packet_indices) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.status->packet_count) {
    if (position < a.routes)
      a.sort_keys_in[position] = ~std::uint64_t{0};
    return;
  }
  const auto packet = packet_indices[position];
  auto expert = static_cast<std::uint64_t>(a.packet_primary_expert[packet]);
  auto expert_domain = static_cast<std::uint64_t>(a.experts);
  if (a.world == 8U) {
    const auto local = a.rank_to_local[a.packet_primary_rank[packet]];
    const auto priority = ((local & 1U) << 2U) | (local & 2U) |
                          ((local & 4U) >> 2U);
    expert += static_cast<std::uint64_t>(priority) * a.experts;
    expert_domain *= 8ULL;
  }
  auto key = a.packet_primary_tile[packet];
  key = key * expert_domain + expert;
  key = key * a.world + a.packet_primary_rank[packet];
  key = key * a.max_tokens + a.packet_headers[packet].local_token_id;
  a.sort_keys_in[position] = key;
}

enum class LocalRouteSortField : std::uint32_t {
  Expert,
  MTile,
  Locality,
};

__global__ void materialize_local_route_order_init_kernel(
    DeviceMaterializeArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (a.status->error == 0 && route < a.routes)
    a.sort_indices_in[route] = route;
}

// Return payloads are independent of the packet order used to amortize the
// Forward dispatch of one token.  Build a second, route-level permutation so
// every source/proxy pair lays Return rows out by destination GPU and canonical
// row.  The key is unique because every route owns one canonical destination
// row.  Keeping the dense ordinal in DeviceFanoutRecord lets the receiving
// proxy reconstruct the same compact directory without another sort or any
// control-plane exchange.
__global__ void materialize_return_order_init_kernel(DeviceMaterializeArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (a.status->error != 0 || route >= a.routes)
    return;
  a.sort_indices_in[route] = route;
  const auto coordinate =
      static_cast<std::uint64_t>(a.route_dst_node[route]) * a.world +
      a.route_dst_local[route];
  a.sort_keys_in[route] =
      coordinate * a.max_incoming_rows + a.route_dst_row[route];
}

__global__ void materialize_return_node_offsets_kernel(
    DeviceMaterializeArgs a) {
  const auto node = static_cast<std::uint32_t>(threadIdx.x);
  std::uint64_t lower = 0ULL;
  if (a.status->error == 0 && node < a.nodes) {
    const auto key_begin =
        static_cast<std::uint64_t>(node) * a.world * a.max_incoming_rows;
    std::uint64_t upper = a.routes;
    while (lower < upper) {
      const auto middle = lower + (upper - lower) / 2ULL;
      if (a.sort_keys_out[middle] < key_begin)
        lower = middle + 1ULL;
      else
        upper = middle;
    }
  }
  // Do not let a fast lane overwrite an input key while another lane is still
  // searching it.
  __syncthreads();
  // The sorted keys are dead after this point.  Reuse their first words for
  // the tiny per-node prefix table consumed by the ordinal kernel.
  if (a.status->error == 0 && node < a.nodes)
    a.sort_keys_out[node] = lower;
}

__global__ void materialize_return_compact_rows_kernel(
    DeviceMaterializeArgs a) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.routes)
    return;
  const auto route = a.wave_flags[position];
  const auto node = a.route_dst_node[route];
  a.sort_keys_in[route] = position - a.sort_keys_out[node];
}

template <LocalRouteSortField Field>
__global__ void materialize_local_route_sort_keys_kernel(
    DeviceMaterializeArgs a, const std::uint64_t *route_indices) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.routes)
    return;
  const auto route = route_indices[position];
  std::uint64_t key = 0ULL;
  if constexpr (Field == LocalRouteSortField::Expert) {
    key = a.expert_id[route];
  } else if constexpr (Field == LocalRouteSortField::MTile) {
    const auto expert = a.expert_id[route];
    key = (a.route_dst_row[route] - a.expert_base[expert]) / a.m_tile;
  } else {
    const auto local_node = a.rank_to_node[a.rank];
    key = a.route_dst_node[route] == local_node ? 0ULL : 1ULL;
  }
  a.sort_keys_in[position] = key;
}

__global__ void materialize_one_node_local_route_sort_keys_kernel(
    DeviceMaterializeArgs a, const std::uint64_t *route_indices) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.routes)
    return;
  const auto route = route_indices[position];
  a.sort_keys_in[position] =
      static_cast<std::uint64_t>(a.route_dst_mtile[route]) * a.experts +
      a.expert_id[route];
}

inline constexpr std::uint64_t kDispatchPrimaryTilesPerWave = 1;
__global__ void materialize_wave_flags_kernel(DeviceMaterializeArgs a) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.status->packet_count)
    return;
  const auto packet = a.sort_indices_out[position];
  bool begins = position == 0 ||
                position % kDispatchMaxPacketsPerWave == 0;
  if (!begins) {
    const auto previous = a.sort_indices_out[position - 1];
    begins = a.packet_headers[packet].destination_node !=
                 a.packet_headers[previous].destination_node ||
             a.packet_primary_tile[packet] /
                     kDispatchPrimaryTilesPerWave !=
                 a.packet_primary_tile[previous] /
                     kDispatchPrimaryTilesPerWave;
  }
  a.wave_flags[position] = begins ? 1 : 0;
}

__global__ void materialize_waves_kernel(DeviceMaterializeArgs a) {
  const auto position = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
  if (a.status->error != 0 || position >= a.status->packet_count)
    return;
  const auto packet = a.sort_indices_out[position];
  const auto wave = a.wave_ids[position] - 1;
  const auto tile = a.packet_primary_tile[packet];
  if (a.wave_flags[position] != 0) {
    a.dispatch_wave_offsets[wave] = position;
    a.dispatch_wave_destination_node[wave] =
        a.packet_headers[packet].destination_node;
    a.dispatch_wave_channel[wave] = a.packet_headers[packet].channel;
    a.dispatch_wave_first_primary_tile[wave] = tile;
  }
  atomicMax(reinterpret_cast<unsigned long long *>(
                &a.dispatch_wave_last_primary_tile[wave]),
            static_cast<unsigned long long>(tile));
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &a.dispatch_wave_packet_count[wave]),
            1ULL);
  const auto fanout_count = a.packet_fanout_offsets[packet + 1] -
                            a.packet_fanout_offsets[packet];
  const auto descriptor_bytes = sizeof(DevicePacketHeaderRecord) +
                                fanout_count * sizeof(DeviceFanoutRecord);
  atomicAdd(reinterpret_cast<unsigned long long *>(
                &a.dispatch_wave_payload_bytes[wave]),
            static_cast<unsigned long long>(descriptor_bytes));
  if (position + 1 == a.status->packet_count) {
    const auto waves = a.wave_ids[position];
    a.dispatch_wave_offsets[waves] = a.status->packet_count;
    a.status->wave_count = waves;
  }
}

__global__ void materialize_signals_kernel(DeviceMaterializeArgs a) {
  const auto node = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (a.status->error != 0 || node >= a.nodes)
    return;
  const auto source_node = a.rank_to_node[a.rank];
  const auto rail = a.rank_to_local[a.rank];
  const auto proxy = rank_at_coordinate(a, node, rail);
  a.outgoing_signals[node] = DeviceSignalRecord{
      a.local_summary[node], a.local_summary[node], source_node, node, a.rank,
      proxy, rail, 0};
  const auto source_rank = rank_at_coordinate(a, node, rail);
  const auto expected = a.summary_matrix[
      static_cast<std::uint64_t>(source_rank) * a.nodes + source_node];
  a.incoming_signals[node] = DeviceSignalRecord{
      expected, expected, node, source_node, source_rank, a.rank, rail, 0};
}

#if !MEGAMOE_DEBUG_VALIDATE
__global__ void publish_materialization_status_kernel(
    const DeviceMaterializationStatus *source,
    DeviceMaterializationStatus *mapped_status,
    std::uint32_t *mapped_ready) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *mapped_status = *source;
  __threadfence_system();
  ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system> ready(
      *mapped_ready);
  ready.store(1U, ::cuda::memory_order_release);
}
#endif

__global__ void dispatch_consumer_probe_delay_kernel(
    const std::uint64_t *source, std::uint64_t delay_cycles) {
  const auto start = clock64();
  while (clock64() - start < delay_cycles) {
  }
  if (source && threadIdx.x == 0 && blockIdx.x == 0)
    asm volatile("" : : "l"(*source));
}

__global__ void delayed_route_fill_kernel(std::int32_t *routes,
                                          std::uint64_t count,
                                          std::int32_t first_expert,
                                          std::uint64_t delay_cycles) {
  const auto start = clock64();
  while (clock64() - start < delay_cycles) {
  }
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    routes[index] = first_expert + static_cast<std::int32_t>(index);
}

std::string route_coordinates(std::uint64_t flat, std::uint32_t top_k) {
  return "token=" + std::to_string(flat / top_k) +
         ", slot=" + std::to_string(flat % top_k);
}

struct WorkspaceState {
  CudaNotifyWorkspacePlan plan;
  Layout device_layout;
  HostLayout host_layout;
  RankCollectiveContext context;
  CudaExecution execution;
  std::shared_ptr<CudaAllocation> allocation;
  std::shared_ptr<const void> slot_lease;
  std::uint64_t epoch = 0;
};

DeviceBufferLease make_lease(const std::shared_ptr<WorkspaceState> &state,
                             std::size_t offset, std::uint64_t width) {
  if (width == 0)
    return {};
  auto buffer = std::make_shared<CudaBufferState>();
  buffer->owner = state;
  buffer->pointer = at<void>(state->allocation->device_ptr, offset);
  buffer->device = state->execution.device_ordinal();
  return make_device_buffer_lease(1, width, width, std::move(buffer));
}

