#pragma once

namespace backward_pipeline_detail {

using megamoe::cuda::persistent_v3::PaddedCounter;
using megamoe::cuda::persistent_v3::backward::BackwardMblockDescriptor;
using megamoe::cuda::persistent_v3::backward::TaskKind;
using megamoe::cuda::persistent_v3::backward::TileDomain;
using megamoe::cuda::persistent_v3::backward::TileTask;
using megamoe::cuda::persistent_v3::backward::WorkQueues;

constexpr std::uint32_t kNoMblock = ~0U;
constexpr std::uint32_t kNoTask = ~0U;
constexpr std::uint32_t kTerminalTask = ~1U;
constexpr std::uint32_t kDabRowsPerMblock = backward_tuning::kTileM;
constexpr std::uint32_t kDabRowsPerTask =
    backward_tuning::kEpilogueRowsPerTask;
constexpr std::uint32_t kDabEpilogueTasks =
    (kDabRowsPerMblock + kDabRowsPerTask - 1U) / kDabRowsPerTask;
constexpr std::uint32_t kDw2MTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileM;
constexpr std::uint32_t kDw2NTiles =
    backward_tuning::kMainIntermediate / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw1MTiles =
    (2U * backward_tuning::kMainIntermediate) /
    backward_tuning::kDwTileM;
constexpr std::uint32_t kDw1NTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw2TilesPerExpert =
    backward_tuning::kDw2TilesPerExpert;
constexpr std::uint32_t kDw1TilesPerExpert =
    backward_tuning::kDw1TilesPerExpert;

static_assert(kDabRowsPerMblock ==
              megamoe::cuda::persistent_v3::backward::
                  kDabPhaseTileCapacity);
static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileM == 0U);
static_assert(backward_tuning::kMainIntermediate %
                  backward_tuning::kDwTileN ==
              0U);
static_assert((2U * backward_tuning::kMainIntermediate) %
                  backward_tuning::kDwTileM ==
              0U);
static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileN == 0U);

__device__ __forceinline__ unsigned long long *scheduler_words(
    const BackwardKernelArgs &a) {
  return peer_at<unsigned long long>(a.local_base, a.scheduler_offset);
}

__device__ __forceinline__ TileDomain make_dx_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return TileDomain{
      words + layout.dx_ready,
      words + layout.dx_claim,
      a.mblocks,
      static_cast<std::uint32_t>(a.hidden / backward_tuning::kDxTileN),
      TaskKind::Dx};
}

__device__ __forceinline__ TileDomain make_dab_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return TileDomain{
      words + layout.dab_ready,
      words + layout.dab_claim,
      a.mblocks,
      kDabRowsPerMblock,
      TaskKind::Dab};
}

__device__ __forceinline__ TileDomain make_dw2_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return TileDomain{
      words + layout.dw2_ready,
      words + layout.dw2_claim,
      a.dispatch.sizes.owned_expert_count,
      kDw2TilesPerExpert,
      TaskKind::Dw2,
      true,
      backward_tuning::kDw2ClaimTiles};
}

__device__ __forceinline__ TileDomain make_dw1_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return TileDomain{
      words + layout.dw1_ready,
      words + layout.dw1_claim,
      a.dispatch.sizes.owned_expert_count,
      kDw1TilesPerExpert,
      TaskKind::Dw1,
      true,
      backward_tuning::kDw1ClaimTiles};
}

__device__ __forceinline__ WorkQueues make_work_queues(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  WorkQueues queues{};
  queues.dx = make_dx_domain(a, words, layout);
  queues.dab = make_dab_domain(a, words, layout);
  queues.dw2 = make_dw2_domain(a, words, layout);
  queues.dw1 = make_dw1_domain(a, words, layout);
  return queues;
}

__device__ __forceinline__ void initialize_status(
    const BackwardKernelArgs &a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  status->start_clock = ~0ULL;
  status->dispatch_complete_clock = 0ULL;
  status->fc1_start_clock = ~0ULL;
  status->fc1_complete_clock = 0ULL;
  status->relay_blocks_done = ~0ULL;  // First dX tile start.
  status->fc2_complete_clock = 0ULL;
  status->reduce_blocks_done = ~0ULL; // First returned M-block.
  status->return_complete_clock = 0ULL;
  status->backward_dw2_start_clock = ~0ULL;
  status->backward_dw2_complete_clock = 0ULL;
  status->backward_dw1_start_clock = ~0ULL;
  status->backward_dw1_complete_clock = 0ULL;
  status->gin_return_put_count = 0ULL;
  status->gin_return_payload_bytes = 0ULL;
}

__device__ __forceinline__ void publish_dispatch_if_ready(
    const BackwardKernelArgs &a, unsigned long long *words,
    const megamoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  if (a.mblocks == 0U)
    return;
  auto *cursor = reinterpret_cast<PaddedCounter *>(words + layout.dispatch_scan);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> scan(
      cursor->value);
  const auto mblock = static_cast<std::uint32_t>(
      scan.fetch_add(1ULL, ::cuda::memory_order_relaxed) % a.mblocks);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> phase(
      words[layout.phase + mblock]);
  if (phase.load(::cuda::memory_order_acquire) != 0ULL)
    return;
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + layout.descriptors);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> arrived(
      words[layout.dispatch_arrived + mblock]);
  if (arrived.load(::cuda::memory_order_acquire) !=
      descriptors[mblock].valid_rows)
    return;
  auto expected = 0ULL;
  if (!phase.compare_exchange_strong(expected, 1ULL,
                                     ::cuda::memory_order_acq_rel,
                                     ::cuda::memory_order_relaxed))
    return;
  auto dab = make_dab_domain(a, words, layout);
  const auto du_tiles =
      static_cast<std::uint32_t>(a.intermediate /
                                 backward_tuning::kDabTileN);
  megamoe::cuda::persistent_v3::backward::publish_owner_prefix(
      dab, mblock, du_tiles);
}

__device__ __forceinline__ std::uint64_t
local_route_count(const BackwardKernelArgs &a) {
  if (!a.use_gin)
    return a.dispatch.sizes.route_count;
  std::uint64_t begin = 0ULL;
  auto end = a.dispatch.sizes.route_count;
  while (begin < end) {
    const auto middle = begin + (end - begin) / 2ULL;
    const auto route = a.dispatch_waves.local_route_order[middle];
    if (a.dispatch.route_dst_node[route] == a.node)
      begin = middle + 1ULL;
    else
      end = middle;
  }
  return begin;
}

} // namespace backward_pipeline_detail

#include "cuda/persistent_v3/backward/gin_pipeline.cuh"

__global__ void backward_setup_pipeline_kernel(BackwardKernelArgs a) {
  using namespace backward_pipeline_detail;
  if (blockIdx.x != 0U)
    return;
  auto *words = scheduler_words(a);
  const auto layout =
      megamoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          static_cast<std::uint32_t>(a.hidden /
                                     backward_tuning::kDxTileN),
          a.scheduler_expert_capacity);
  for (std::uint64_t word = threadIdx.x; word < a.scheduler_words;
       word += blockDim.x)
    words[word] = 0ULL;
  __syncthreads();

  auto *descriptors = reinterpret_cast<BackwardMblockDescriptor *>(
      words + layout.descriptors);
  const auto *padded_offsets = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  const auto *headers = peer_at<const LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  for (std::uint32_t expert = threadIdx.x;
       expert < a.dispatch.sizes.owned_expert_count;
       expert += blockDim.x) {
    const auto rows = a.dispatch.expert_row_count[expert];
    words[layout.dw_reduction_rows + expert] =
        ((rows + backward_tuning::kTileK - 1ULL) /
         backward_tuning::kTileK) *
        backward_tuning::kTileK;
  }
  for (std::uint32_t mblock = threadIdx.x; mblock < a.mblocks;
       mblock += blockDim.x) {
    const auto padded_row =
        static_cast<std::uint64_t>(mblock) * backward_tuning::kTileM;
    std::uint32_t expert = 0U;
    while (expert < a.dispatch.sizes.owned_expert_count &&
           padded_row >= padded_offsets[expert + 1U])
      ++expert;
    if (expert >= a.dispatch.sizes.owned_expert_count)
      continue;
    const auto local_row = padded_row - padded_offsets[expert];
    const auto expert_rows = a.dispatch.expert_row_count[expert];
    const auto remaining_rows =
        expert_rows > local_row ? expert_rows - local_row : 0ULL;
    const auto valid_rows =
        static_cast<std::uint32_t>(
            remaining_rows < backward_tuning::kTileM
                ? remaining_rows
                : backward_tuning::kTileM);
    descriptors[mblock] = BackwardMblockDescriptor{
        static_cast<std::uint32_t>(a.dispatch.expert_row_prefix[expert] +
                                   local_row),
        static_cast<std::uint32_t>(padded_row), expert, valid_rows};
    unsigned long long context_mask = 0ULL;
    const auto compact_row = descriptors[mblock].compact_row;
    for (std::uint32_t row = 0U; row < valid_rows; ++row) {
      const auto source = headers[compact_row + row].source_rank;
      if (source / a.local_world_size != a.node)
        context_mask |= 1ULL << (source % a.local_world_size);
    }
    words[layout.return_context_mask + mblock] = context_mask;
  }
  __syncthreads();
  if (threadIdx.x == 0U)
    initialize_status(a);
  __syncthreads();
  __threadfence_system();

  const auto first_peer = a.use_gin ? a.node * a.local_world_size : 0U;
  const auto peer_count = a.use_gin ? a.local_world_size : a.world;
  for (std::uint32_t local_peer = threadIdx.x; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    auto *setup = peer_at<unsigned long long>(a.peer_bases[peer],
                                               a.setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(setup[a.rank]);
    published.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_setup = peer_at<unsigned long long>(a.local_base,
                                                   a.setup_epoch_offset);
  for (std::uint32_t local_peer = threadIdx.x; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(local_setup[peer]);
    while (published.load(::cuda::memory_order_acquire) != a.epoch)
      __nanosleep(backward_tuning::kSetupIdleNanosleep);
  }
}

// This kernel deliberately reserves the same large dynamic-SMEM footprint as
// a compute CTA. Eight blocks therefore remain one-CTA-per-SM transport
// workers instead of opportunistically co-residing on compute SMs.
__global__ __launch_bounds__(backward_tuning::kComputeThreads, 1) void
backward_local_transport_kernel(BackwardKernelArgs a) {
  using namespace backward_pipeline_detail;
  extern __shared__ __align__(1024) unsigned char reservation[];
#if __has_include(<nccl_device.h>)
  if (a.use_gin && blockIdx.x < backward_tuning::kRdmaCtas) {
    backward_gin_detail::rdma_role(a);
    return;
  }
#endif
  const auto relay_ordinal =
      static_cast<std::uint32_t>(blockIdx.x) -
      (a.use_gin ? backward_tuning::kRdmaCtas : 0U);
  (void)reservation;
  __shared__ unsigned long long route_shared;
  __shared__ std::uint32_t return_mblock_shared;
  __shared__ std::uint32_t stop_shared;
  __shared__ std::uint32_t progressed_shared;
  __shared__ std::uint32_t dispatch_destination_shared[
      backward_tuning::kDispatchBatchRows];
  __shared__ std::uint32_t dispatch_mblock_shared[
      backward_tuning::kDispatchBatchRows];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ ::cuda::barrier<::cuda::thread_scope_block>
      dispatch_full[backward_tuning::kComputeThreads / 32U];

  auto *words = scheduler_words(a);
  const auto layout =
      megamoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          static_cast<std::uint32_t>(a.hidden /
                                     backward_tuning::kDxTileN),
          a.scheduler_expert_capacity);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  constexpr std::uint32_t kWarps =
      backward_tuning::kComputeThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto copy_bytes =
      static_cast<std::uint32_t>(a.hidden * sizeof(transport::BFloat16));
  const auto local_routes = local_route_count(a);
#if __has_include(<nccl_device.h>)
  __shared__ backward_gin_detail::RelayInboundState gin_inbound;
  __shared__ backward_gin_detail::SourceCompletionState gin_completion;
  const bool owns_gin_context =
      a.use_gin && relay_ordinal < backward_tuning::kRdmaCtas;
  if (owns_gin_context) {
    backward_gin_detail::initialize_relay_state(
        a, relay_ordinal, gin_inbound, gin_completion);
  }
#endif
  if (lane == 0U)
    init(&dispatch_full[warp], 1U);
  ::cuda::device::experimental::fence_proxy_async_shared_cta();
  __syncwarp();
  if (threadIdx.x == 0U)
    atomicMin(&status->start_clock, device_globaltimer_ns());
  __syncthreads();

  while (true) {
    if (threadIdx.x == 0U) {
      auto *claim = reinterpret_cast<PaddedCounter *>(
          words + layout.dispatch_route_claim);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          cursor(claim->value);
      route_shared = cursor.fetch_add(backward_tuning::kDispatchBatchRows,
                                      ::cuda::memory_order_relaxed);
      progressed_shared = 0U;
    }
    __syncthreads();

    if (route_shared < local_routes) {
      const auto remaining = local_routes - route_shared;
      const auto batch =
          remaining < backward_tuning::kDispatchBatchRows
              ? static_cast<std::uint32_t>(remaining)
              : backward_tuning::kDispatchBatchRows;
      for (std::uint32_t local = warp; local < batch; local += kWarps) {
        const auto route = a.dispatch_waves.local_route_order[
            route_shared + local];
        if (lane == 0U) {
          const auto destination = a.dispatch.route_dst_rank[route];
          const auto compact_destination_row = a.dispatch.route_dst_row[route];
          const auto *compact_to_padded = peer_at<const unsigned int>(
              a.peer_bases[destination], a.compact_to_padded_offset);
          const auto destination_row =
              static_cast<std::uint64_t>(compact_to_padded[
                  compact_destination_row]);
          dispatch_destination_shared[local] = destination;
          dispatch_mblock_shared[local] = static_cast<std::uint32_t>(
              destination_row / backward_tuning::kTileM);
          const auto token = route / a.dispatch.identity.top_k;
          auto *destination_dr = peer_at<transport::BFloat16>(
              a.peer_bases[destination], a.padded_dr_offset);
          auto *warp_storage = reinterpret_cast<std::byte *>(reservation) +
                               static_cast<std::size_t>(warp) * copy_bytes;
          auto &full = dispatch_full[warp];
          ::cuda::device::experimental::cp_async_bulk_global_to_shared(
              warp_storage, a.dy + token * a.hidden, copy_bytes, full);
          auto token_barrier =
              ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
          full.wait(static_cast<decltype(token_barrier) &&>(token_barrier));
          ::cuda::device::experimental::cp_async_bulk_shared_to_global(
              destination_dr + destination_row * a.hidden, warp_storage,
              copy_bytes);
          ::cuda::device::experimental::cp_async_bulk_commit_group();
          ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
          ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
        }
      }
      if (lane == 0U)
        __threadfence_system();
      __syncthreads();
      for (std::uint32_t local = threadIdx.x; local < batch;
           local += blockDim.x) {
        const auto destination = dispatch_destination_shared[local];
        const auto mblock = dispatch_mblock_shared[local];
        const bool leader =
            local == 0U ||
            dispatch_destination_shared[local - 1U] != destination ||
            dispatch_mblock_shared[local - 1U] != mblock;
        if (!leader)
          continue;
        auto run_end = local + 1U;
        while (run_end < batch &&
               dispatch_destination_shared[run_end] == destination &&
               dispatch_mblock_shared[run_end] == mblock)
          ++run_end;
        auto *peer_words = peer_at<unsigned long long>(
            a.peer_bases[destination], a.scheduler_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            arrived(peer_words[layout.dispatch_arrived + mblock]);
        arrived.fetch_add(run_end - local, ::cuda::memory_order_release);
      }
      __syncthreads();
      if (threadIdx.x == 0U) {
        auto *completed_counter = reinterpret_cast<PaddedCounter *>(
            words + layout.dispatched_routes);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(completed_counter->value);
        const auto done = completed.fetch_add(
                              batch, ::cuda::memory_order_acq_rel) +
                          batch;
        if (done == local_routes)
          atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
        progressed_shared = 1U;
      }
    }
    __syncthreads();

#if __has_include(<nccl_device.h>)
    if (owns_gin_context) {
      const bool inbound_progress =
          backward_gin_detail::progress_inbound_once(a, gin_inbound);
      const bool completion_progress =
          backward_gin_detail::progress_source_completion_once(
              a, gin_completion);
      if (threadIdx.x == 0U && (inbound_progress || completion_progress))
        progressed_shared = 1U;
      __syncthreads();
    }
#endif

    if (threadIdx.x == 0U)
      publish_dispatch_if_ready(a, words, layout);
    __syncthreads();

    if (threadIdx.x == 0U) {
      return_mblock_shared = kNoMblock;
      if (a.mblocks != 0U) {
        auto *cursor = reinterpret_cast<PaddedCounter *>(
            words + layout.return_scan);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            scan(cursor->value);
        const auto mblock = static_cast<std::uint32_t>(
            scan.fetch_add(1ULL, ::cuda::memory_order_relaxed) % a.mblocks);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(words[layout.dx_completed + mblock]);
        const auto dx_ready =
            megamoe::cuda::persistent_v3::backward::backward_dx_ready_value(
                static_cast<std::uint32_t>(
                    a.hidden / backward_tuning::kDxTileN));
        if (completed.load(::cuda::memory_order_acquire) == dx_ready) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              phase(words[layout.phase + mblock]);
          auto expected = 3ULL;
          if (phase.compare_exchange_strong(expected, 4ULL,
                                            ::cuda::memory_order_acq_rel,
                                            ::cuda::memory_order_relaxed)) {
            return_mblock_shared = mblock;
            atomicMin(&status->reduce_blocks_done, device_globaltimer_ns());
          }
        }
      }
    }
    __syncthreads();

    if (return_mblock_shared != kNoMblock) {
      const auto *descriptors =
          reinterpret_cast<const BackwardMblockDescriptor *>(
              words + layout.descriptors);
      const auto descriptor = descriptors[return_mblock_shared];
      const auto *headers = peer_at<const LocalP2pRouteHeader>(
          a.local_base, a.direct_headers_offset);
      const auto *dx = peer_at<const transport::BFloat16>(
          a.local_base, a.dx_rows_offset);
      const auto *registered_dx =
          a.use_gin
              ? static_cast<const std::byte *>(a.gin_window_base) +
                    a.gin_return_staging_offset
              : static_cast<const std::byte *>(nullptr);
      const auto *dgate = peer_at<const float>(a.local_base,
                                               a.dgate_rows_offset);
      const auto warp = threadIdx.x / 32U;
      const auto lane = threadIdx.x & 31U;
      for (std::uint32_t local_row = warp;
           local_row < descriptor.valid_rows; local_row += kWarps) {
        const auto compact_row = descriptor.compact_row + local_row;
        const auto padded_row = descriptor.padded_row + local_row;
        const auto header = headers[compact_row];
        if (a.use_gin &&
            header.source_rank / a.local_world_size != a.node)
          continue;
        const auto *source =
            a.use_gin
                ? reinterpret_cast<const uint4 *>(
                      registered_dx +
                      static_cast<std::uint64_t>(padded_row) *
                          a.gin_return_row_bytes)
                : reinterpret_cast<const uint4 *>(
                      dx + static_cast<std::uint64_t>(padded_row) * a.hidden);
        auto *destination = peer_at<transport::BFloat16>(
            a.peer_bases[header.source_rank], a.contribution_offset);
        auto *target = reinterpret_cast<uint4 *>(
            destination + header.route_index * a.hidden);
        for (std::uint64_t vector = lane; vector < a.hidden / 8ULL;
             vector += 32ULL)
          target[vector] = source[vector];
        if (lane == 0U) {
          auto *destination_dgate = peer_at<float>(
              a.peer_bases[header.source_rank], a.dgate_output_offset);
          destination_dgate[header.route_index] = dgate[padded_row];
        }
      }
      __threadfence_system();
      __syncthreads();
      for (std::uint32_t local_row = threadIdx.x;
           local_row < descriptor.valid_rows; local_row += blockDim.x) {
        const auto header = headers[descriptor.compact_row + local_row];
        if (a.use_gin &&
            header.source_rank / a.local_world_size != a.node)
          continue;
        auto *ready = peer_at<unsigned long long>(
            a.peer_bases[header.source_rank], a.return_ready_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            marker(ready[header.route_index]);
        marker.store(a.epoch, ::cuda::memory_order_release);
      }
      __syncthreads();
      if (threadIdx.x == 0U) {
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.returned_mblocks);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            returned(counter->value);
        const auto done = returned.fetch_add(
                              1ULL, ::cuda::memory_order_acq_rel) +
                          1ULL;
        if (done == a.mblocks)
          atomicMax(&status->return_complete_clock, device_globaltimer_ns());
        progressed_shared = 1U;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
      auto *dispatch_counter = reinterpret_cast<PaddedCounter *>(
          words + layout.dispatched_routes);
      auto *return_counter = reinterpret_cast<PaddedCounter *>(
          words + layout.returned_mblocks);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          dispatched(dispatch_counter->value);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          returned(return_counter->value);
      stop_shared =
          dispatched.load(::cuda::memory_order_acquire) ==
                  local_routes &&
                  returned.load(::cuda::memory_order_acquire) == a.mblocks
              ? 1U
              : 0U;
#if __has_include(<nccl_device.h>)
      if (stop_shared != 0U && owns_gin_context &&
          (!backward_gin_detail::inbound_terminal(gin_inbound) ||
           !backward_gin_detail::source_completion_terminal(gin_completion)))
        stop_shared = 0U;
#endif
    }
    __syncthreads();
    if (stop_shared != 0U)
      break;
    if (progressed_shared == 0U)
      __nanosleep(backward_tuning::kQueueIdleNanosleep);
    __syncthreads();
  }
}

template <std::uint32_t DabTileN, std::uint32_t DxTileN,
          std::uint32_t DwTileN>
__global__ __launch_bounds__(backward_tuning::kComputeThreads, 1) void
backward_local_compute_kernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap dr_map,
    const __grid_constant__ CUtensorMap dab_map,
    const __grid_constant__ CUtensorMap w2_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap dr_mn_map,
    const __grid_constant__ CUtensorMap u_mn_map,
    const __grid_constant__ CUtensorMap dab_mn_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap u_store_map,
    const __grid_constant__ CUtensorMap dx_store_map,
    const __grid_constant__ CUtensorMap dw2_store_map,
    const __grid_constant__ CUtensorMap dw1_store_map) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  using namespace backward_pipeline_detail;
  static_assert(backward_tuning::kMainIntermediate % DabTileN == 0U);
  static_assert(backward_tuning::kMainHidden % DxTileN == 0U);
  static_assert(DwTileN == backward_tuning::kDwTileN);
  constexpr std::uint32_t kActivationStorageTileN =
      DabTileN > DxTileN ? DabTileN : DxTileN;
  constexpr std::uint32_t kStorageTileN =
      kActivationStorageTileN > DwTileN ? kActivationStorageTileN : DwTileN;
  constexpr std::uint32_t kDuTiles =
      backward_tuning::kMainIntermediate / DabTileN;
  constexpr std::uint32_t kDxTiles =
      backward_tuning::kMainHidden / DxTileN;
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ std::uint32_t task_kind_shared;
  __shared__ std::uint32_t task_owner_shared;
  __shared__ std::uint32_t task_tile_shared;
  __shared__ std::uint32_t task_count_shared;
  __shared__ std::uint32_t phase_shared;

  backward_compute_detail::initialize_k_mn_pipeline<kStorageTileN>(storage);
  std::uint32_t activation_pipeline = 0U;
  std::uint32_t dw_pipeline = 0U;
  auto *words = scheduler_words(a);
  const auto layout =
      megamoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity, kDxTiles,
          a.scheduler_expert_capacity);
  const auto queues = make_work_queues(a, words, layout);
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + layout.descriptors);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  bool activation_complete = false;

  while (true) {
    if (threadIdx.x == 0U) {
      TileTask task{};
      bool claimed = false;
      if (!activation_complete) {
        claimed =
            megamoe::cuda::persistent_v3::backward::try_claim_domain_once(
                queues.dx, task) ||
            megamoe::cuda::persistent_v3::backward::try_claim_domain_once(
                queues.dab, task);
        if (!claimed) {
          auto *dx_counter = reinterpret_cast<PaddedCounter *>(
              words + layout.completed_dx_tiles);
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_device>
              completed_dx(dx_counter->value);
          activation_complete =
              completed_dx.load(::cuda::memory_order_acquire) ==
              static_cast<unsigned long long>(a.mblocks) * kDxTiles;
        }
      }
      if (!claimed) {
        // Do not reserve a private dW backlog while activation work can still
        // arrive.  Once dX is globally complete, restore the tuned multi-tile
        // claims so each CTA can keep the dW K pipeline hot.
        auto dw2 = queues.dw2;
        if (!activation_complete)
          dw2.claim_granularity = 1U;
        claimed =
            megamoe::cuda::persistent_v3::backward::try_claim_domain_once(
                dw2, task);
      }
      if (!claimed) {
        auto dw1 = queues.dw1;
        if (!activation_complete)
          dw1.claim_granularity = 1U;
        claimed =
            megamoe::cuda::persistent_v3::backward::try_claim_domain_once(
                dw1, task);
      }
      if (claimed) {
        task_kind_shared = static_cast<std::uint32_t>(task.kind);
        task_owner_shared = task.owner;
        task_tile_shared = task.tile;
        task_count_shared = task.tile_count;
      } else {
        auto *dx_counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dx_tiles);
        auto *dw2_counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw2_tiles);
        auto *dw1_counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw1_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed_dx(dx_counter->value);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed_dw2(dw2_counter->value);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed_dw1(dw1_counter->value);
        task_kind_shared =
            completed_dx.load(::cuda::memory_order_acquire) ==
                    static_cast<unsigned long long>(a.mblocks) * kDxTiles &&
                    completed_dw2.load(::cuda::memory_order_acquire) ==
                        static_cast<unsigned long long>(a.active_experts) *
                            kDw2TilesPerExpert &&
                    completed_dw1.load(::cuda::memory_order_acquire) ==
                        static_cast<unsigned long long>(a.active_experts) *
                            kDw1TilesPerExpert
                ? kTerminalTask
                : kNoTask;
      }
    }
    __syncthreads();
    if (task_kind_shared == kTerminalTask)
      break;
    if (task_kind_shared == kNoTask) {
      __nanosleep(backward_tuning::kQueueIdleNanosleep);
      __syncthreads();
      continue;
    }

    const auto owner = task_owner_shared;
    const auto tile = task_tile_shared;
    if (task_kind_shared == static_cast<std::uint32_t>(TaskKind::Dab)) {
      const auto descriptor = descriptors[owner];
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            phase(words[layout.phase + owner]);
        phase_shared = static_cast<std::uint32_t>(
            phase.load(::cuda::memory_order_acquire));
      }
      __syncthreads();
      if (phase_shared == 1U) {
        if (threadIdx.x == 0U)
          atomicMin(&status->fc1_start_clock, device_globaltimer_ns());
        auto *du = peer_at<transport::BFloat16>(
            a.local_base, a.recomputed_u_offset);
        backward_compute_detail::run_k_mn_tile_tma_store<
            DabTileN, kStorageTileN>(
            dr_map, w2_map, u_store_map, descriptor.padded_row,
            descriptor.expert * static_cast<std::uint32_t>(a.hidden),
            static_cast<std::uint32_t>(a.hidden), du, a.intermediate,
            descriptor.padded_row, tile * DabTileN, storage,
            activation_pipeline);
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed(words[layout.du_completed + owner]);
          const auto done = completed.fetch_add(
                                1ULL, ::cuda::memory_order_acq_rel) +
                            1ULL;
          if (done == kDuTiles) {
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_device>
                phase(words[layout.phase + owner]);
            phase.store(2ULL, ::cuda::memory_order_release);
            megamoe::cuda::persistent_v3::backward::publish_owner_prefix(
                queues.dab, owner, kDabEpilogueTasks);
          }
        }
      } else if (phase_shared == 2U) {
        const auto local_row = tile * kDabRowsPerTask;
        const auto rows =
            kDabRowsPerMblock - local_row < kDabRowsPerTask
                ? kDabRowsPerMblock - local_row
                : kDabRowsPerTask;
        const auto valid_rows =
            descriptor.valid_rows > local_row
                ? (descriptor.valid_rows - local_row < rows
                       ? descriptor.valid_rows - local_row
                       : rows)
                : 0U;
        backward_swiglu_epilogue_warp_rows<kDabRowsPerTask>(
            a, static_cast<std::uint64_t>(descriptor.padded_row) + local_row,
            valid_rows, rows);
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
              completed(words[layout.dab_completed + owner]);
          const auto done = completed.fetch_add(
                                1ULL, ::cuda::memory_order_acq_rel) +
                            1ULL;
          if (done == kDabEpilogueTasks) {
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_device>
                phase(words[layout.phase + owner]);
            phase.store(3ULL, ::cuda::memory_order_release);
            megamoe::cuda::persistent_v3::backward::publish_owner(
                queues.dx, owner);
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_device>
                expert_completed(
                    words[layout.expert_mblocks_completed +
                          descriptor.expert]);
            const auto expert_done = expert_completed.fetch_add(
                                         1ULL,
                                         ::cuda::memory_order_acq_rel) +
                                     1ULL;
            const auto *padded_rows = peer_at<const unsigned long long>(
                a.local_base, a.padded_expert_rows_offset);
            if (expert_done ==
                padded_rows[descriptor.expert] /
                    backward_tuning::kTileM) {
              megamoe::cuda::persistent_v3::backward::publish_owner(
                  queues.dw2, descriptor.expert);
              megamoe::cuda::persistent_v3::backward::publish_owner(
                  queues.dw1, descriptor.expert);
            }
            atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
          }
        }
      }
    } else if (task_kind_shared ==
               static_cast<std::uint32_t>(TaskKind::Dx)) {
      const auto descriptor = descriptors[owner];
      if (threadIdx.x == 0U)
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
      auto *dx = peer_at<transport::BFloat16>(a.local_base,
                                              a.dx_rows_offset);
      backward_compute_detail::run_k_mn_tile_tma_store<
          DxTileN, kStorageTileN>(
          dab_map, w1_map, dx_store_map, descriptor.padded_row,
          descriptor.expert *
              static_cast<std::uint32_t>(2ULL * a.intermediate),
          static_cast<std::uint32_t>(2ULL * a.intermediate), dx, a.hidden,
          descriptor.padded_row, tile * DxTileN, storage,
          activation_pipeline);
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            owner_completed(words[layout.dx_completed + owner]);
        const auto owner_done = owner_completed.fetch_add(
                                    1ULL, ::cuda::memory_order_acq_rel) +
                                1ULL;
        phase_shared = owner_done == kDxTiles ? 1U : 0U;
      }
      __syncthreads();
      if (phase_shared != 0U && a.use_gin)
        __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0U) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            owner_completed(words[layout.dx_completed + owner]);
        if (phase_shared != 0U)
          owner_completed.store(
              megamoe::cuda::persistent_v3::backward::
                  backward_dx_ready_value(kDxTiles),
              ::cuda::memory_order_release);
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dx_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done = completed.fetch_add(
                              1ULL, ::cuda::memory_order_acq_rel) +
                          1ULL;
        if (done == static_cast<unsigned long long>(a.mblocks) * kDxTiles)
          atomicMax(&status->fc2_complete_clock, device_globaltimer_ns());
      }
    } else if (task_kind_shared ==
               static_cast<std::uint32_t>(TaskKind::Dw2)) {
      if (threadIdx.x == 0U)
        atomicMin(&status->backward_dw2_start_clock,
                  device_globaltimer_ns());
      const auto *padded_offsets = peer_at<const unsigned long long>(
          a.local_base, a.padded_expert_offsets_offset);
      const auto reduction_rows = static_cast<std::uint32_t>(
          words[layout.dw_reduction_rows + owner]);
      auto *dw2 = peer_at<transport::BFloat16>(a.local_base, a.dw2_offset);
      if (task_count_shared > 1U) {
        backward_compute_detail::run_mn_mn_chunk<
            backward_tuning::kDwTileM, DwTileN, kStorageTileN>(
            dr_mn_map, u_mn_map, dw2_store_map, tile,
            task_count_shared, kDw2NTiles,
            static_cast<std::uint32_t>(padded_offsets[owner]),
            reduction_rows,
            static_cast<std::uint32_t>(a.intermediate),
            owner * static_cast<std::uint32_t>(a.hidden), storage,
            dw_pipeline);
      } else {
        const auto current_tile = tile;
        const auto current_m_tile = current_tile / kDw2NTiles;
        const auto current_n_tile = current_tile % kDw2NTiles;
        backward_compute_detail::run_mn_mn_tile<
            backward_tuning::kDwTileM, DwTileN, kStorageTileN>(
            dr_mn_map, u_mn_map,
            current_m_tile * backward_tuning::kDwTileM,
            current_n_tile * DwTileN,
            static_cast<std::uint32_t>(padded_offsets[owner]),
            reduction_rows, dw2,
            a.intermediate,
            owner * static_cast<std::uint32_t>(a.hidden) +
                current_m_tile * backward_tuning::kDwTileM,
            dw2_store_map, storage, dw_pipeline);
      }
      if (threadIdx.x == 0U) {
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw2_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done = completed.fetch_add(
                              task_count_shared,
                              ::cuda::memory_order_acq_rel) +
                          task_count_shared;
        if (done == static_cast<unsigned long long>(a.active_experts) *
                        kDw2TilesPerExpert)
          atomicMax(&status->backward_dw2_complete_clock,
                    device_globaltimer_ns());
      }
    } else if (task_kind_shared ==
               static_cast<std::uint32_t>(TaskKind::Dw1)) {
      if (threadIdx.x == 0U)
        atomicMin(&status->backward_dw1_start_clock,
                  device_globaltimer_ns());
      const auto *padded_offsets = peer_at<const unsigned long long>(
          a.local_base, a.padded_expert_offsets_offset);
      const auto reduction_rows = static_cast<std::uint32_t>(
          words[layout.dw_reduction_rows + owner]);
      auto *dw1 = peer_at<transport::BFloat16>(a.local_base, a.dw1_offset);
      if (task_count_shared > 1U) {
        backward_compute_detail::run_mn_mn_chunk<
            backward_tuning::kDwTileM, DwTileN, kStorageTileN>(
            dab_mn_map, x_mn_map, dw1_store_map, tile,
            task_count_shared, kDw1NTiles,
            static_cast<std::uint32_t>(padded_offsets[owner]),
            reduction_rows,
            static_cast<std::uint32_t>(a.hidden),
            owner * static_cast<std::uint32_t>(2ULL * a.intermediate),
            storage, dw_pipeline);
      } else {
        const auto current_tile = tile;
        const auto current_m_tile = current_tile / kDw1NTiles;
        const auto current_n_tile = current_tile % kDw1NTiles;
        backward_compute_detail::run_mn_mn_tile<
            backward_tuning::kDwTileM, DwTileN, kStorageTileN>(
            dab_mn_map, x_mn_map,
            current_m_tile * backward_tuning::kDwTileM,
            current_n_tile * DwTileN,
            static_cast<std::uint32_t>(padded_offsets[owner]),
            reduction_rows, dw1, a.hidden,
            owner * static_cast<std::uint32_t>(2ULL * a.intermediate) +
                current_m_tile * backward_tuning::kDwTileM,
            dw1_store_map, storage, dw_pipeline);
      }
      if (threadIdx.x == 0U) {
        auto *counter = reinterpret_cast<PaddedCounter *>(
            words + layout.completed_dw1_tiles);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(counter->value);
        const auto done = completed.fetch_add(
                              task_count_shared,
                              ::cuda::memory_order_acq_rel) +
                          task_count_shared;
        if (done == static_cast<unsigned long long>(a.active_experts) *
                        kDw1TilesPerExpert)
          atomicMax(&status->backward_dw1_complete_clock,
                    device_globaltimer_ns());
      }
    }
    __syncthreads();
  }
#else
  (void)a;
  (void)dr_map;
  (void)dab_map;
  (void)w2_map;
  (void)w1_map;
  (void)dr_mn_map;
  (void)u_mn_map;
  (void)dab_mn_map;
  (void)x_mn_map;
  (void)u_store_map;
  (void)dx_store_map;
  (void)dw2_store_map;
  (void)dw1_store_map;
#endif
}
