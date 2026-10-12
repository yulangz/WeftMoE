#pragma once

namespace weftmoe_pf {

// Exact tape counts and the physical PF plan share one resident entry. The
// first expert-prefix record is scratch until prepare() builds the windows.
// Every reader joins before that record can be reused.
template <bool Network>
__device__ __noinline__ void count_training_routes(const ForwardKernelArgs &a) {
  constexpr unsigned Threads = 384U;
  const auto worker = std::uint64_t(blockIdx.x) * Threads + threadIdx.x;
  const auto workers = std::uint64_t(gridDim.x) * Threads;
  const auto routes = a.dispatch.sizes.route_count;
  const auto experts = a.dispatch.identity.num_experts;
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto chunks =
      (routes + kTrainingCountChunkRoutes - 1U) / kTrainingCountChunkRoutes;
  auto *prefix = const_cast<std::uint64_t *>(a.checkpoint.counts.chunk_prefix);
  auto *counts = a.ring.source.expert_prefix;
  if (!worker) {
    a.ring.source_header->error = 0U;
    *a.checkpoint.counts.runtime = {};
  }
  for (auto i = worker; i < chunks * experts; i += workers)
    prefix[i] = 0U;
  grid_sync<0U, Threads>(a);
  for (auto chunk = std::uint64_t(blockIdx.x); chunk < chunks;
       chunk += gridDim.x) {
    for (unsigned offset = threadIdx.x; offset < kTrainingCountChunkRoutes;
         offset += Threads) {
      const auto route = chunk * kTrainingCountChunkRoutes + offset;
      const int expert = route < routes ? a.selected_experts[route] : -1;
      const auto peers = __match_any_sync(0xffffffffU, expert);
      if (expert >= 0 && unsigned(expert) < experts) {
        for (auto prior = route - route % a.dispatch.identity.top_k;
             prior < route; ++prior)
          if (a.selected_experts[prior] == expert)
            atomicExch(&a.ring.source_header->error, 1U);
        if (threadIdx.x % 32U == unsigned(__ffs(peers) - 1))
          atomicAdd(reinterpret_cast<unsigned long long *>(
                        prefix + std::uint64_t(expert) * chunks + chunk),
                    static_cast<unsigned long long>(__popc(peers)));
      } else if (route < routes) {
        atomicExch(&a.ring.source_header->error, 1U);
      }
    }
  }
  grid_sync<0U, Threads>(a);
  for (auto expert = worker; expert < experts; expert += workers) {
    const_cast<std::uint32_t *>(a.dispatch.expert_to_rank)[expert] =
        expert / owned;
    std::uint64_t sum = 0U;
    for (std::uint64_t chunk = 0; chunk < chunks; ++chunk) {
      auto &value = prefix[expert * chunks + chunk];
      const auto count = value;
      value = sum;
      sum += count;
    }
    counts[expert] = sum;
  }
  if constexpr (Network) {
    grid_sync<0U, Threads>(a);
    if (!blockIdx.x && !threadIdx.x) {
      const auto base = source_offset(a, a.rank);
      const auto offset = reinterpret_cast<const char *>(counts) -
                          reinterpret_cast<const char *>(a.ring.source_header);
      for (unsigned node = 0; node < a.dispatch.identity.num_nodes; ++node)
        if (node != a.node) {
          network_put(a, node, base, base,
                      sizeof(static_forward::SourceHeader));
          network_put(a, node, base + offset, base + offset,
                      experts * sizeof(std::uint64_t));
        }
    }
    network_barrier<0U, Threads>(a, 1U);
  } else {
    rank_barrier<0U, Threads>(a);
  }
  for (auto i = worker; i < std::uint64_t(a.world) * experts; i += workers)
    const_cast<std::uint64_t *>(a.dispatch.source_expert_count)[i] =
        ring::source_metadata(a, i / experts, counts)[i % experts];
  grid_sync<0U, Threads>(a);
  for (auto destination = worker; destination < a.world;
       destination += workers) {
    std::uint64_t rows = 0U, tiles = 0U;
    for (unsigned local = 0; local < owned; ++local) {
      const auto expert = destination * owned + local;
      std::uint64_t total = 0U, source_base = rows;
      for (unsigned source = 0; source < a.world; ++source) {
        const auto count =
            a.dispatch
                .source_expert_count[std::uint64_t(source) * experts + expert];
        total += count;
        if (source < a.rank)
          source_base += count;
      }
      const_cast<std::uint64_t *>(
          a.checkpoint.counts.expert_source_base)[expert] = source_base;
      if (destination == a.rank) {
        const_cast<std::uint32_t *>(a.dispatch.owned_expert_ids)[local] =
            expert;
        const_cast<std::uint64_t *>(a.dispatch.expert_row_count)[local] = total;
        const_cast<std::uint64_t *>(a.dispatch.expert_row_prefix)[local] = rows;
        const_cast<std::uint64_t *>(a.dispatch.mtile_row_offsets)[local] =
            tiles;
      }
      rows += total;
      tiles += (total + 127U) / 128U;
    }
    if (destination == a.rank) {
      const_cast<std::uint64_t *>(a.dispatch.expert_row_prefix)[owned] = rows;
      const_cast<std::uint64_t *>(a.dispatch.mtile_row_offsets)[owned] = tiles;
      auto &runtime = *a.checkpoint.counts.runtime;
      runtime.sizes = a.dispatch.sizes;
      runtime.sizes.incoming_row_count = rows;
      runtime.sizes.mtile_count = tiles;
      runtime.sizes.packet_count = runtime.sizes.fanout_count = 0U;
      runtime.sizes.outgoing_signal_count =
          runtime.sizes.incoming_signal_count = 0U;
      for (unsigned source = 0; source < a.world; ++source)
        runtime.error |=
            ring::source_metadata(a, source, a.ring.source_header)->error;
      runtime.error |= rows > a.max_destination_rows;
    }
  }
  // Readers on any node must finish before the full plan overwrites counts.
  if constexpr (Network)
    network_barrier<0U, Threads>(a, 1U);
  else
    rank_barrier<0U, Threads>(a);
}

template <bool Network, bool Backward = false>
__global__ __launch_bounds__(384U, 1) void notify_kernel(
    const __grid_constant__ ForwardKernelArgs a) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && __has_include(<nccl_device.h>)
  constexpr unsigned Threads = 384U;
  extern __shared__ __align__(1024) unsigned char scratch[];
  programmatic_launch::wait();
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (!blockIdx.x && !threadIdx.x) {
    *status = {};
    status->start_clock = device_globaltimer_ns();
    status->route_exchange_start_clock = status->start_clock;
    status->dispatch_blocks_done = status->reduced_tokens =
        status->fc1_start_clock = status->relay_blocks_done = ~0ULL;
  }
  if (a.retain_backward_state && !a.training_counts_ready) {
    count_training_routes<Network>(a);
    if (a.checkpoint.counts.runtime->error)
      return;
  }
  if constexpr (Network) {

    if (a.retain_backward_state)
      ring::prepare<true, true, Threads, Backward>(a, scratch);
    else
      ring::prepare<true, false, Threads>(a, scratch);

  } else {
    local::prepare(a, scratch);
  }
  if (!blockIdx.x && !threadIdx.x)
    status->route_exchange_complete_clock = device_globaltimer_ns();
#endif
}

} // namespace weftmoe_pf
