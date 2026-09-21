#pragma once

// Single-node, forward-only metadata construction.  Unlike local_notify(),
// route ordinals within an expert are intentionally assigned by atomics.  A
// GEMM row is independent in Forward, while the final top-k reduction still
// consumes describe_token()'s canonical combine_order.  Training must not use
// this path: dW reduces over M and therefore requires the stable row order.
__device__ void single_node_inference_local_notify(LocalNotifyArgs a) {
  auto grid = cooperative_groups::this_grid();
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;

  for (auto expert = std::uint64_t(tid); expert < a.experts;
       expert += stride)
    a.histogram[expert] = 0;
  for (auto node = std::uint64_t(tid); node < a.nodes; node += stride)
    a.packet_count[node] = 0;
  if (tid == 0)
    *a.status = DeviceStatus{~0ULL, ~0ULL};
  grid.sync();

  for (auto token = std::uint64_t(tid); token < a.tokens; token += stride)
    describe_token(a.selected, token, a.tokens, a.top_k, a.experts,
                   a.expert_to_rank, a.rank_to_node, a.nodes, a.mask_words,
                   a.combine_order, a.masks, a.packet_count, a.status);

  for (auto route = std::uint64_t(tid); route < a.routes; route += stride) {
    const auto raw = a.selected[route];
    if (raw < 0 || static_cast<unsigned>(raw) >= a.experts) {
      atomicMin(&a.status->out_of_range_packed,
                (static_cast<unsigned long long>(route) << 32U) |
                    static_cast<unsigned>(raw));
      a.expert_id[route] = a.experts;
      a.dst_rank[route] = a.dst_node[route] = a.dst_local[route] = 0;
      a.local_offset[route] = 0;
      continue;
    }
    const auto expert = static_cast<unsigned>(raw);
    const auto destination = a.expert_to_rank[expert];
    a.expert_id[route] = expert;
    a.dst_rank[route] = destination;
    a.dst_node[route] = a.rank_to_node[destination];
    a.dst_local[route] = a.rank_to_local[destination];
    a.local_offset[route] = atomicAdd(
        reinterpret_cast<unsigned long long *>(&a.histogram[expert]), 1ULL);
  }
  grid.sync();
}

__device__ void single_node_inference_materialize(
    DeviceMaterializeArgs a, std::uint64_t *sort_histogram,
    DeviceDispatchRuntime *runtime,
    DeviceMaterializationStatus *mapped_status, std::uint32_t *mapped_ready) {
  auto grid = cooperative_groups::this_grid();
  __shared__ NotifySortStorage sort_shared;
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;

  if (tid == 0) {
    *a.status = DeviceMaterializationStatus{};
    if (a.nodes != 1 || a.top_k == 0 ||
        a.max_tokens > ~std::uint64_t{0} / a.top_k)
      materialization_fail_once(
          a.status, MaterializationError::InvalidCollectiveValue, 60, 1,
          a.nodes);
  }
  grid.sync();

  const auto max_routes = a.max_tokens * a.top_k;
  for (auto expert = std::uint64_t(tid); expert < a.experts;
       expert += stride) {
    if (a.status->error)
      continue;
    std::uint64_t total = 0;
    for (std::uint32_t source = 0; source < a.world; ++source) {
      const auto value =
          a.count_matrix[std::uint64_t(source) * a.experts + expert];
      if (value > max_routes || value > ~std::uint64_t{0} - total) {
        materialization_fail_once(
            a.status,
            value > max_routes ? MaterializationError::InvalidCollectiveValue
                               : MaterializationError::ArithmeticOverflow,
            expert, max_routes, value);
        break;
      }
      if (source == a.rank && value != a.local_histogram[expert]) {
        materialization_fail_once(a.status,
                                  MaterializationError::CountRowMismatch,
                                  expert, a.local_histogram[expert], value);
        break;
      }
      total += value;
    }
    if (a.has_expert_capacity && total > a.expert_capacity[expert])
      materialization_fail_once(a.status,
                                MaterializationError::ExpertCapacity, expert,
                                a.expert_capacity[expert], total);
    a.expert_total[expert] = total;
  }
  for (auto source = std::uint64_t(tid); source < a.world;
       source += stride) {
    if (a.status->error || a.top_k == 0)
      continue;
    std::uint64_t routes = 0;
    for (std::uint32_t expert = 0; expert < a.experts; ++expert)
      routes += a.count_matrix[source * a.experts + expert];
    atomicMax(&a.status->max_source_tokens, routes / a.top_k);
  }
  grid.sync();

  // One thread owns each destination's expert prefix.  This is only O(E) per
  // rank and replaces E cooperative CTAs plus their grid barriers.
  for (auto destination = std::uint64_t(tid); destination < a.world;
       destination += stride) {
    if (a.status->error)
      continue;
    std::uint64_t rows = 0, tiles = 0, owned = 0;
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      if (a.expert_to_rank[expert] != destination)
        continue;
      const auto count = a.expert_total[expert];
      a.expert_base[expert] = rows;
      a.expert_tile_base[expert] = tiles;
      if (destination == a.rank) {
        a.owned_expert_ids[owned] = expert;
        a.owned_expert_count[owned] = count;
        a.owned_expert_prefix[owned] = rows;
        a.mtile_offsets[owned] = tiles;
        ++owned;
      }
      rows += count;
      tiles += count / a.m_tile + (count % a.m_tile != 0);
    }
    atomicMax(&a.status->max_destination_rows, rows);
    if (a.has_workspace_capacity && rows > a.workspace_capacity[destination])
      materialization_fail_once(a.status,
                                MaterializationError::WorkspaceCapacity,
                                destination,
                                a.workspace_capacity[destination], rows);
    if (destination == a.rank) {
      if (rows > a.max_incoming_rows || tiles > a.max_incoming_rows)
        materialization_fail_once(a.status,
                                  MaterializationError::OutputCapacity,
                                  destination, a.max_incoming_rows, rows);
      a.owned_expert_prefix[owned] = rows;
      a.mtile_offsets[owned] = tiles;
      a.status->owned_experts = owned;
      a.status->incoming_rows = rows;
      a.status->mtile_count = tiles;
    }
  }
  grid.sync();

  for (auto expert = std::uint64_t(tid); expert < a.experts;
       expert += stride) {
    if (a.status->error)
      continue;
    const auto total = a.expert_total[expert];
    const auto owner = a.expert_to_rank[expert];
    const auto local = a.rank_to_local[owner];
    std::uint64_t prefix = 0;
    for (std::uint32_t other = 0; other < a.experts; ++other) {
      const auto peer = a.expert_to_rank[other];
      if (a.rank_to_local[peer] < local ||
          (peer == owner && other < expert))
        prefix += a.local_histogram[other];
    }
    a.return_expert_base[expert] = prefix;
    a.return_global_expert_base[expert] = prefix;
    std::uint64_t source_prefix = 0;
    for (std::uint32_t source = 0; source < a.rank; ++source)
      source_prefix +=
          a.count_matrix[std::uint64_t(source) * a.experts + expert];
    // Expert totals have already been consumed by prefix/tile construction;
    // reuse the expert-sized array for this source interval base.
    a.expert_total[expert] = source_prefix;

    if (owner == a.rank) {
      std::uint64_t owned = 0;
      for (std::uint32_t prior = 0; prior < expert; ++prior)
        owned += a.expert_to_rank[prior] == a.rank;
      const auto count = total;
      const auto tiles = count / a.m_tile + (count % a.m_tile != 0);
      for (std::uint64_t tile = 0; tile < tiles; ++tile) {
        const auto remaining = count - tile * a.m_tile;
        a.mtile_rows[a.mtile_offsets[owned] + tile] =
            static_cast<std::uint32_t>(
                remaining < a.m_tile ? remaining : a.m_tile);
      }
    }
  }
  grid.sync();

  for (auto route = std::uint64_t(tid); route < a.routes; route += stride) {
    if (a.status->error)
      continue;
    const auto expert = a.expert_id[route];
    const auto source_prefix = a.expert_total[expert];
    const auto row =
        a.expert_base[expert] + source_prefix + a.local_offset[route];
    const auto mtile = a.expert_tile_base[expert] +
                       (row - a.expert_base[expert]) / a.m_tile;
    a.route_dst_row[route] = row;
    a.route_dst_mtile[route] = static_cast<std::uint32_t>(mtile);
    a.reverse_routes[route] = DeviceReverseRouteRecord{
        route / a.top_k, row, a.rank,
        static_cast<std::uint32_t>(route % a.top_k), expert,
        a.route_dst_rank[route]};
    if (a.routes <= 4096) {
      // Short decode/small-prefill buckets benefit from completing each
      // destination expert contiguously; their expert tiles are almost all a
      // single M block, so the generic 32-row stripe interleave buys nothing.
      a.local_route_order[a.return_global_expert_base[expert] +
                          a.local_offset[route]] = route;
    } else {
      a.local_route_order[route] = route;
      const auto stripe = 2ULL * a.expert_tile_base[expert] +
                          (source_prefix + a.local_offset[route]) / 32ULL;
      a.sort_keys_in[route] = stripe * a.experts + expert;
    }
  }
  grid.sync();

  if (a.status->error == 0 && a.routes > 4096)
    notify_radix_sort(a.sort_keys_in, a.local_route_order, a.sort_keys_out,
                      a.sort_indices_in, a.routes, sort_histogram,
                      sort_shared);

  const auto packets = a.status->error == 0 ? a.local_summary[0] : 0ULL;
  if (tid == 0) {
    const auto collective = a.summary_matrix[std::uint64_t(a.rank) * a.nodes];
    if (collective != packets || packets > a.max_tokens)
      materialization_fail_once(a.status,
                                MaterializationError::SummaryRowMismatch, 0,
                                packets, collective);
    a.status->packet_count = packets;
    a.status->fanout_count = a.routes;
    a.status->wave_count = 0;
    a.packet_fanout_offsets[packets] = a.routes;
    a.outgoing_signals[0] = DeviceSignalRecord{
        packets, packets, 0, 0, a.rank, a.rank, a.rank_to_local[a.rank], 0};
    a.incoming_signals[0] = a.outgoing_signals[0];
  }
  for (auto token = std::uint64_t(tid); token < packets; token += stride) {
    a.packet_headers[token] =
        DevicePacketHeaderRecord{token, token, a.rank, 0, a.rank, 0};
    a.packet_fanout_offsets[token] = token * a.top_k;
    for (std::uint32_t position = 0; position < a.top_k; ++position) {
      const auto slot = a.combine_order[token * a.top_k + position];
      const auto route = token * a.top_k + slot;
      const auto expert = a.expert_id[route];
      a.packet_fanout[token * a.top_k + position] = DeviceFanoutRecord{
          a.route_dst_row[route],
          a.return_expert_base[expert] + a.local_offset[route], slot,
          a.route_dst_local[route], expert, a.route_dst_mtile[route]};
    }
    a.sort_indices_out[token] = token;
  }
  grid.sync();

  if (tid == 0) {
    runtime->sizes = {
        a.routes,
        a.mask_words,
        static_cast<std::uint32_t>(a.status->owned_experts),
        a.status->incoming_rows,
        a.status->mtile_count,
        a.status->packet_count,
        a.status->fanout_count,
        1,
        1};
    runtime->wave_count = 0;
    runtime->error = a.status->error;
    if (mapped_ready) {
      *mapped_status = *a.status;
      __threadfence_system();
      ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>(
          *mapped_ready)
          .store(1U, ::cuda::memory_order_release);
    }
  }
}

__global__ void __launch_bounds__(256) single_node_inference_notify_kernel(
    DeviceMaterializeArgs a, std::uint64_t *sort_histogram,
    DeviceDispatchRuntime *runtime,
    DeviceMaterializationStatus *mapped_status, std::uint32_t *mapped_ready,
    bool, bool, LocalNotifyArgs local, NotifyCommunicationView comm,
    std::uint64_t digest0, std::uint64_t digest1) {
  programmatic_launch::begin();
  single_node_inference_local_notify(local);
  notify_exchange(local, a, comm, digest0, digest1);
  if (local.status->out_of_range_packed != ~0ULL) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      *a.status = {};
      materialization_fail(a.status,
                           MaterializationError::InvalidCollectiveValue, 50,
                           0, 1);
      *runtime = {};
      runtime->error = 1;
      if (mapped_ready) {
        *mapped_status = *a.status;
        __threadfence_system();
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>(
            *mapped_ready)
            .store(1U, ::cuda::memory_order_release);
      }
    }
    return;
  }
  single_node_inference_materialize(a, sort_histogram, runtime, mapped_status,
                                    mapped_ready);
}
