#pragma once

// The training allocation boundary only counts routes. Chunk prefixes let
// main recover stable source ordinals without a second global histogram.
__global__ void __launch_bounds__(256)
    training_count_notify_kernel(NOTIFY_ENTRY_ARGS) {
  auto grid = cooperative_groups::this_grid();
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto stride = gridDim.x * blockDim.x;
  const auto chunks = (local.routes + kTrainingCountChunkRoutes - 1U) /
                      kTrainingCountChunkRoutes;
  for (auto i = std::uint64_t(tid); i < chunks * local.experts; i += stride)
    local.chunk_counts[i] = 0U;
  for (auto i = tid; i < local.nodes; i += stride)
    local.packet_count[i] = 0U;
  if (!tid)
    *local.status = DeviceStatus{~0ULL, ~0ULL};
  grid.sync();
  for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
       chunk += gridDim.x) {
    for (unsigned offset = threadIdx.x; offset < kTrainingCountChunkRoutes;
         offset += blockDim.x) {
      const auto route = chunk * kTrainingCountChunkRoutes + offset;
      const auto expert = route < local.routes ? local.selected[route] : -1;
      const auto peers = __match_any_sync(0xffffffffU, expert);
      if (expert >= 0 && unsigned(expert) < local.experts) {
        if (threadIdx.x % 32U == unsigned(__ffs(peers) - 1))
          atomicAdd(reinterpret_cast<unsigned long long *>(local.chunk_counts) +
                        std::uint64_t(expert) * chunks + chunk,
                    static_cast<unsigned long long>(__popc(peers)));
      } else if (route < local.routes) {
        atomicMin(&local.status->out_of_range_packed,
                  (static_cast<unsigned long long>(route) << 32U) |
                      unsigned(expert));
      }
    }
  }
  grid.sync();
  for (unsigned expert = tid; expert < local.experts; expert += stride) {
    std::uint64_t prefix = 0U;
    for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
      auto &count = local.chunk_counts[std::uint64_t(expert) * chunks + chunk];
      const auto value = count;
      count = prefix;
      prefix += value;
    }
    local.histogram[expert] = prefix;
  }
  grid.sync();
  notify_exchange(local, a, comm, digest0, digest1);
  if (!tid) {
    *a.status = {};
    if (local.status->out_of_range_packed != ~0ULL)
      materialization_fail(
          a.status, MaterializationError::InvalidCollectiveValue, 50U, 0U, 1U);
  }
  grid.sync();
  for (unsigned expert = tid; expert < a.experts; expert += stride) {
    std::uint64_t total = 0U;
    for (unsigned source = 0; source < a.world; ++source)
      total += a.count_matrix[std::uint64_t(source) * a.experts + expert];
    a.expert_total[expert] = total;
    if (a.has_expert_capacity && total > a.expert_capacity[expert])
      materialization_fail_once(a.status, MaterializationError::ExpertCapacity,
                                expert, a.expert_capacity[expert], total);
  }
  grid.sync();
  for (unsigned destination = tid; destination < a.world;
       destination += stride) {
    std::uint64_t rows = 0U, tiles = 0U, padded = 0U;
    unsigned owned = 0U;
    for (unsigned expert = 0; expert < a.experts; ++expert) {
      if (a.expert_to_rank[expert] != destination)
        continue;
      const auto count = a.expert_total[expert];
      std::uint64_t source_base = rows;
      for (unsigned source = 0; source < a.rank; ++source)
        source_base +=
            a.count_matrix[std::uint64_t(source) * a.experts + expert];
      a.expert_base[expert] = source_base;
      if (destination == a.rank) {
        a.owned_expert_ids[owned] = expert;
        a.owned_expert_count[owned] = count;
        a.owned_expert_prefix[owned] = rows;
        a.mtile_offsets[owned] = tiles;
        ++owned;
      }
      rows += count;
      tiles += (count + a.m_tile - 1U) / a.m_tile;
      padded += persistent_v3::padded_expert_rows(count);
    }
    atomicMax(&a.status->max_destination_rows, rows);
    if (a.has_workspace_capacity && rows > a.workspace_capacity[destination])
      materialization_fail_once(
          a.status, MaterializationError::WorkspaceCapacity, destination,
          a.workspace_capacity[destination], rows);
    if (destination == a.rank) {
      if (rows > a.max_incoming_rows)
        materialization_fail_once(a.status,
                                  MaterializationError::OutputCapacity,
                                  destination, a.max_incoming_rows, rows);
      a.owned_expert_prefix[owned] = rows;
      a.mtile_offsets[owned] = tiles;
      a.status->owned_experts = owned;
      a.status->incoming_rows = rows;
      a.status->mtile_count = tiles;
      a.status->padded_compute_rows = padded;
    }
    std::uint64_t source_routes = 0U;
    for (unsigned expert = 0; expert < a.experts; ++expert)
      source_routes +=
          a.count_matrix[std::uint64_t(destination) * a.experts + expert];
    atomicMax(&a.status->max_source_tokens, source_routes / a.top_k);
  }
  grid.sync();
  if (!tid) {
    publish_dispatch_runtime_stage(a, 0U, runtime);
    *status = *a.status;
    __threadfence_system();
    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>(*ready)
        .store(1U, ::cuda::memory_order_release);
  }
}
