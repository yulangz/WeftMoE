#include "cuda/notify/radix.cuh"

// Measurement-only switch. Negative (the default) keeps production behaviour:
// the full exchange and materialization pipeline runs. 1 skips the cross-rank
// count exchange, 2 skips the exchange and the whole materialization, so the
// remaining time is the local routing pass alone. Results are meaningless in
// these modes; they are used to attribute notify-kernel cost.
#ifndef WEFTMOE_NOTIFY_TIMING_MODE
#define WEFTMOE_NOTIFY_TIMING_MODE -1
#endif

__device__ void materialize_notify(DeviceMaterializeArgs a,
                                   std::uint64_t *sort_histogram,
                                   DeviceDispatchRuntime *runtime,
                                   DeviceMaterializationStatus *mapped_status,
                                   std::uint32_t *mapped_ready,
                                   bool packed_wave, bool packed_local) {
  auto grid = cooperative_groups::this_grid();
  __shared__ NotifySortStorage shared;
  const auto tid = blockIdx.x * 256 + threadIdx.x;
  const auto stride = gridDim.x * 256;
  const auto route_blocks = (a.routes + 255) / 256;
  // Each stage is a device function over logical tiles. Grid barriers express
  // actual producer/consumer dependencies without additional kernel launches.
#define NOTIFY_STAGE(blocks, call)                                             \
  do {                                                                         \
    for (std::uint64_t block = blockIdx.x; block < (blocks);                   \
         block += gridDim.x) {                                                 \
      call;                                                                    \
      __syncthreads();                                                         \
    }                                                                          \
    grid.sync();                                                               \
  } while (0)
  for (auto i = std::uint64_t(tid); i < a.routes; i += stride) {
    a.wave_flags[i] = 0;
    a.dispatch_wave_last_primary_tile[i] = 0;
    a.dispatch_wave_packet_count[i] = 0;
    a.dispatch_wave_payload_bytes[i] = 0;
  }
  NOTIFY_STAGE(1U, materialize_plan_init_stage(a, block));
  NOTIFY_STAGE((a.experts + 255) / 256,
               materialize_expert_totals_stage(a, block));
  NOTIFY_STAGE(a.experts, materialize_expert_prefix_stage(a, block));
  NOTIFY_STAGE(a.world, materialize_destination_capacity_stage(a, block));
  NOTIFY_STAGE(1U, materialize_plan_finalize_stage(a, block));
  NOTIFY_STAGE((a.world + 255) / 256,
               materialize_source_capacity_stage(a, block));
  NOTIFY_STAGE(std::uint64_t(a.nodes) * a.packet_chunks,
               count_packet_blocks_stage(a, block));
  NOTIFY_STAGE(std::uint64_t(a.nodes) * a.packet_chunks,
               materialize_packet_plan_stage(a, block));
  NOTIFY_STAGE(route_blocks, materialize_routes_stage(a, block));
  NOTIFY_STAGE(a.experts, materialize_mtiles_stage(a, block));
  NOTIFY_STAGE((a.experts + 255) / 256,
               materialize_return_expert_prefix_stage(a, block));
  NOTIFY_STAGE(route_blocks, (materialize_fanout_stage(a, block),
                              materialize_packet_primary_stage(a, block)));
  if (a.status->error == 0 && a.routes) {
    if (packed_wave) {
      NOTIFY_STAGE(route_blocks, materialize_packed_wave_sort_keys_stage(
                                     a, block, a.sort_indices_in));
      notify_radix_sort(a.sort_keys_in, a.sort_indices_in, a.sort_keys_out,
                        a.local_route_order, a.status->packet_count,
                        sort_histogram, shared);
    } else {
#define WAVE_PASS(field)                                                       \
  NOTIFY_STAGE(route_blocks,                                                   \
               materialize_wave_sort_keys_stage<WaveSortField::field>(         \
                   a, block, a.sort_indices_in));                              \
  notify_radix_sort(a.sort_keys_in, a.sort_indices_in, a.sort_keys_out,        \
                    a.local_route_order, a.status->packet_count,               \
                    sort_histogram, shared)
      WAVE_PASS(Token);
      WAVE_PASS(DestinationRank);
      WAVE_PASS(Expert);
      WAVE_PASS(MTile);
      WAVE_PASS(DestinationNode);
#undef WAVE_PASS
    }
    for (auto i = std::uint64_t(tid); i < a.routes; i += stride)
      a.sort_indices_out[i] = a.sort_indices_in[i];
    grid.sync();
    NOTIFY_STAGE(route_blocks, materialize_wave_flags_stage(a, block));
    // Parallel block scans followed by the much smaller scan of block sums.
    for (std::uint64_t block = blockIdx.x; block < route_blocks;
         block += gridDim.x) {
      const auto pos = block * 256 + threadIdx.x;
      unsigned long long sum, aggregate;
      cub::BlockScan<unsigned long long, 256>(shared.temp.scan)
          .InclusiveSum(pos < a.routes ? a.wave_flags[pos] : 0ULL, sum,
                        aggregate);
      if (pos < a.routes)
        a.wave_ids[pos] = sum;
      if (!threadIdx.x)
        sort_histogram[block] = aggregate;
      __syncthreads();
    }
    grid.sync();
    if (!tid) {
      std::uint64_t prefix = 0;
      for (std::uint64_t block = 0; block < route_blocks; ++block) {
        const auto sum = sort_histogram[block];
        sort_histogram[block] = prefix;
        prefix += sum;
      }
    }
    grid.sync();
    for (auto i = std::uint64_t(tid); i < a.routes; i += stride)
      a.wave_ids[i] += sort_histogram[i / 256];
    grid.sync();
    NOTIFY_STAGE(route_blocks, materialize_waves_stage(a, block));
    for (auto i = std::uint64_t(tid); i < a.routes; i += stride)
      a.local_route_order[i] = i;
    grid.sync();
    if (packed_local) {
      NOTIFY_STAGE(route_blocks, materialize_packed_local_route_sort_keys_stage(
                                     a, block, a.local_route_order));
      notify_radix_sort(a.sort_keys_in, a.local_route_order, a.sort_keys_out,
                        a.sort_indices_in, a.routes, sort_histogram, shared);
    } else {
#define LOCAL_PASS(field)                                                      \
  NOTIFY_STAGE(                                                                \
      route_blocks,                                                            \
      materialize_local_route_sort_keys_stage<LocalRouteSortField::field>(     \
          a, block, a.local_route_order));                                     \
  notify_radix_sort(a.sort_keys_in, a.local_route_order, a.sort_keys_out,      \
                    a.sort_indices_in, a.routes, sort_histogram, shared)
      LOCAL_PASS(Expert);
      LOCAL_PASS(MTile);
      LOCAL_PASS(Locality);
#undef LOCAL_PASS
    }
  }
  NOTIFY_STAGE((a.nodes + 255) / 256, materialize_signals_stage(a, block));
  NOTIFY_STAGE(1U, publish_dispatch_runtime_stage(a, block, runtime));
  if (tid == 0 && mapped_ready) {
    *mapped_status = *a.status;
    __threadfence_system();
    ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>(
        *mapped_ready)
        .store(1U, ::cuda::memory_order_release);
  }
#undef NOTIFY_STAGE
}

#define NOTIFY_ENTRY_ARGS                                                      \
  DeviceMaterializeArgs a, std::uint64_t *histogram,                           \
      DeviceDispatchRuntime *runtime, DeviceMaterializationStatus *status,     \
      std::uint32_t *ready, bool packed_wave, bool packed_local,               \
      LocalNotifyArgs local, NotifyCommunicationView comm,                     \
      std::uint64_t digest0, std::uint64_t digest1
__global__ void __launch_bounds__(256)
    materialize_notify_kernel(NOTIFY_ENTRY_ARGS) {
  materialize_notify(a, histogram, runtime, status, ready, packed_wave,
                     packed_local);
}
__global__ void __launch_bounds__(256) single_notify_kernel(NOTIFY_ENTRY_ARGS) {
  programmatic_launch::begin();
  local_notify(local);
  cooperative_groups::this_grid().sync();
#if WEFTMOE_NOTIFY_TIMING_MODE == 2
  return;
#else
#if WEFTMOE_NOTIFY_TIMING_MODE == 1
  (void)comm;
  (void)digest0;
  (void)digest1;
#else
  notify_exchange(local, a, comm, digest0, digest1);
#endif
  if (local.status->out_of_range_packed != ~0ULL) {
    if (!blockIdx.x && !threadIdx.x) {
      *a.status = {};
      materialization_fail(
          a.status, MaterializationError::InvalidCollectiveValue, 50, 0, 1);
      *runtime = {};
      runtime->error = 1;
      if (ready) {
        *status = *a.status;
        __threadfence_system();
        ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system>(*ready)
            .store(1, ::cuda::memory_order_release);
      }
    }
    return;
  }
  materialize_notify(a, histogram, runtime, status, ready, packed_wave,
                     packed_local);
#endif
}
#undef NOTIFY_ENTRY_ARGS
