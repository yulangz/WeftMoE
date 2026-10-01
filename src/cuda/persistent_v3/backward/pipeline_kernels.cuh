#pragma once

namespace backward_pipeline_detail {

using weftmoe::cuda::persistent_v3::PaddedCounter;
using weftmoe::cuda::persistent_v3::backward::BackwardMblockDescriptor;
using weftmoe::cuda::persistent_v3::backward::ExpertTileQueue;
using weftmoe::cuda::persistent_v3::backward::TaskKind;
using weftmoe::cuda::persistent_v3::backward::TileDomain;
using weftmoe::cuda::persistent_v3::backward::TileTask;
using weftmoe::cuda::persistent_v3::backward::WorkQueues;

constexpr std::uint32_t kNoMblock = ~0U;
constexpr std::uint32_t kNoTask = ~0U;
constexpr std::uint32_t kTerminalTask = ~1U;
constexpr std::uint32_t kDabRowsPerMblock = backward_tuning::kTileM;
constexpr std::uint32_t kDabRowsPerTask =
    2U * backward_tuning::kEpilogueRowsPerTask;
constexpr std::uint32_t kDabEpilogueTasks =
    (kDabRowsPerMblock + kDabRowsPerTask - 1U) / kDabRowsPerTask;
constexpr std::uint32_t kDw2MTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileM;
constexpr std::uint32_t kDw2NTiles =
    backward_tuning::kMainIntermediate / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw1MTiles =
    (2U * backward_tuning::kMainIntermediate) / backward_tuning::kDwTileM;
constexpr std::uint32_t kDw1NTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw2TilesPerExpert =
    backward_tuning::kDw2TilesPerExpert;
constexpr std::uint32_t kDw1TilesPerExpert =
    backward_tuning::kDw1TilesPerExpert;

static_assert(kDabRowsPerMblock ==
              weftmoe::cuda::persistent_v3::backward::kDabPhaseTileCapacity);
static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileM == 0U);
static_assert(backward_tuning::kMainIntermediate % backward_tuning::kDwTileN ==
              0U);
static_assert((2U * backward_tuning::kMainIntermediate) %
                  backward_tuning::kDwTileM ==
              0U);
static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileN == 0U);

__device__ __forceinline__ unsigned long long *
scheduler_words(const BackwardKernelArgs &a) {
  return peer_at<unsigned long long>(a.local_base, a.scheduler_offset);
}

__device__ __forceinline__ TileDomain make_dx_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout, std::uint32_t mblocks = UINT32_MAX) {
  if (mblocks == UINT32_MAX)
    mblocks = a.mblocks;
  return TileDomain{
      words + layout.dx_ready,
      words + layout.dx_claim,
      mblocks,
      static_cast<std::uint32_t>(a.hidden / backward_tuning::kDxTileN /
                                 backward_tuning::kActivationClusterN),
      TaskKind::Dx,
      words + layout.dx_ready_summary};
}

__device__ __forceinline__ TileDomain make_dab_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout, std::uint32_t mblocks = UINT32_MAX) {
  if (mblocks == UINT32_MAX)
    mblocks = a.mblocks;
  return TileDomain{words + layout.dab_ready,
                    words + layout.dab_claim,
                    mblocks,
                    kDabRowsPerMblock,
                    TaskKind::Dab,
                    words + layout.dab_ready_summary};
}

__device__ __forceinline__ ExpertTileQueue make_dw2_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return ExpertTileQueue{words + layout.dw2_ready,
                         words + layout.dw2_claim,
                         words + layout.dw2_reserved,
                         words + layout.dw2_head,
                         dispatch_sizes(a.dispatch).owned_expert_count,
                         (kDw2TilesPerExpert + 1U) / 2U,
                         TaskKind::Dw2,
                         backward_tuning::kDw2ClaimTiles};
}

__device__ __forceinline__ ExpertTileQueue make_dw1_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout) {
  return ExpertTileQueue{words + layout.dw1_ready,
                         words + layout.dw1_claim,
                         words + layout.dw1_reserved,
                         words + layout.dw1_head,
                         dispatch_sizes(a.dispatch).owned_expert_count,
                         (kDw1TilesPerExpert + 1U) / 2U,
                         TaskKind::Dw1,
                         backward_tuning::kDw1ClaimTiles};
}

__device__ __forceinline__ WorkQueues make_work_queues(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout, std::uint32_t mblocks = UINT32_MAX) {
  WorkQueues queues{};
  queues.dx = make_dx_domain(a, words, layout, mblocks);
  queues.dab = make_dab_domain(a, words, layout, mblocks);
  queues.dw2 = make_dw2_domain(a, words, layout);
  queues.dw1 = make_dw1_domain(a, words, layout);
  return queues;
}

// Local Return receives each M-block once, after its final dX tile. Reuse the
// append-only expert queue with one task per owner: readiness, not a scan of
// every M-block, now drives Return. The queue changes only transport order;
// source-side reduction still follows the immutable canonical route order.
__device__ __forceinline__ ExpertTileQueue make_return_domain(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout, std::uint32_t mblocks = UINT32_MAX) {
  if (mblocks == UINT32_MAX)
    mblocks = a.mblocks;
  return ExpertTileQueue{words + layout.return_ready,
                         words + layout.return_claim,
                         words + layout.return_reserved,
                         words + layout.return_scan,
                         mblocks,
                         1U,
                         TaskKind::Dx,
                         1U};
}

__device__ __forceinline__ void initialize_status(const BackwardKernelArgs &a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  status->start_clock = ~0ULL;
  status->dispatch_complete_clock = 0ULL;
  status->fc1_start_clock = ~0ULL;
  status->fc1_complete_clock = 0ULL;
  status->relay_blocks_done = ~0ULL; // First dX tile start.
  status->fc2_complete_clock = 0ULL;
  status->reduce_blocks_done = ~0ULL; // First returned M-block.
  status->return_complete_clock = 0ULL;
  status->backward_dw2_start_clock = ~0ULL;
  status->backward_dw2_complete_clock = 0ULL;
  status->backward_dw1_start_clock = ~0ULL;
  status->backward_dw1_complete_clock = 0ULL;
  status->backward_reduce_start_clock = ~0ULL;
  status->backward_reduce_complete_clock = 0ULL;
  status->backward_sender_complete_clock = 0ULL;
  status->backward_transport_complete_clock = 0ULL;
  status->backward_return_send_complete_clock = 0ULL;
  status->backward_return_receive_complete_clock = 0ULL;
  status->gin_return_put_count = 0ULL;
  status->gin_return_payload_bytes = 0ULL;
}

template <bool StaticSchedule = false>
__device__ __forceinline__ bool try_publish_dispatch_mblock(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout,
    std::uint32_t mblocks, std::uint32_t mblock) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> phase(
      words[layout.phase + mblock]);
  if (phase.load(::cuda::memory_order_acquire) != 0ULL)
    return false;
  const auto *descriptors = reinterpret_cast<const BackwardMblockDescriptor *>(
      words + layout.descriptors);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> arrived(
      words[layout.dispatch_arrived + mblock]);
  if (arrived.load(::cuda::memory_order_acquire) !=
      descriptors[mblock].valid_rows)
    return false;
  if constexpr (StaticSchedule) {
    // The unified transport already assigns one publisher per M-block.
    phase.store(1ULL, ::cuda::memory_order_release);
    return true;
  }
  auto expected = 0ULL;
  if (!phase.compare_exchange_strong(expected, 1ULL,
                                     ::cuda::memory_order_acq_rel,
                                     ::cuda::memory_order_relaxed))
    return false;
  auto dab = make_dab_domain(a, words, layout, mblocks);
  const auto du_tiles =
      static_cast<std::uint32_t>(a.intermediate / backward_tuning::kDabTileN /
                                 backward_tuning::kActivationClusterN);
  weftmoe::cuda::persistent_v3::backward::publish_owner_prefix(dab, mblock,
                                                               du_tiles);
  return true;
}

__device__ __forceinline__ void publish_dispatch_if_ready(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::persistent_v3::backward::BackwardSchedulerLayout
        &layout, std::uint32_t mblocks = UINT32_MAX) {
  if (mblocks == UINT32_MAX)
    mblocks = a.mblocks;
  if (mblocks == 0U)
    return;
  auto *cursor =
      reinterpret_cast<PaddedCounter *>(words + layout.dispatch_scan);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> scan(
      cursor->value);
  const auto mblock = static_cast<std::uint32_t>(
      scan.fetch_add(1ULL, ::cuda::memory_order_relaxed) % mblocks);
  try_publish_dispatch_mblock(a, words, layout, mblocks, mblock);
}

__device__ __forceinline__ std::uint64_t
local_route_count(const BackwardKernelArgs &a) {
  if (!a.use_gin)
    return dispatch_sizes(a.dispatch).route_count;
  std::uint64_t begin = 0ULL;
  auto end = dispatch_sizes(a.dispatch).route_count;
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

namespace local_notification_detail {
template <> struct Receiver<BackwardKernelArgs> {
  __device__ static void publish_dispatch(const BackwardKernelArgs &a,
                                          unsigned long long row,
                                          unsigned peer) {
    auto *base = a.peer_bases[a.node * a.local_world_size + peer];
    const auto padded =
        peer_at<const unsigned>(base, a.compact_to_padded_offset)[row];
    const auto mblock = padded / backward_tuning::kTileM;
    const auto group = __match_any_sync(
        __activemask(), 1ULL * mblock * communication::kLocalPeers + peer);
    if (threadIdx.x % 32 == unsigned(__ffs(group) - 1)) {
      const auto layout =
          persistent_v3::backward::make_backward_scheduler_layout(
              a.scheduler_mblock_capacity,
              unsigned(a.hidden / backward_tuning::kDxTileN /
                       backward_tuning::kActivationClusterN),
              a.scheduler_expert_capacity);
      auto *words = peer_at<unsigned long long>(base, a.scheduler_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>(
          words[layout.dispatch_arrived + mblock])
          .fetch_add(__popc(group), ::cuda::memory_order_release);
    }
  }
  __device__ static void apply(const BackwardKernelArgs &a,
                               unsigned long long row) {
    auto *ready =
        peer_at<unsigned long long>(a.local_base, a.return_ready_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
        ready[row])
        .store(execution_epoch(a), ::cuda::memory_order_release);
  }
};
} // namespace local_notification_detail

#include "cuda/persistent_v3/backward/gin_pipeline.cuh"

__global__ void backward_setup_pipeline_kernel(BackwardKernelArgs a) {
  if (threadIdx.x == 0U && a.runtime_epoch)
    atomicAdd(a.runtime_epoch, 1ULL);
  __syncthreads();
  backward_load_row_geometry(a);
  using namespace backward_pipeline_detail;
  if (blockIdx.x != 0U)
    return;
  auto *words = scheduler_words(a);
  const auto layout =
      weftmoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          static_cast<std::uint32_t>(a.hidden / backward_tuning::kDxTileN /
                                     backward_tuning::kActivationClusterN),
          a.scheduler_expert_capacity);
  for (std::uint64_t word = threadIdx.x; word < a.scheduler_words;
       word += blockDim.x)
    words[word] = 0ULL;
  __syncthreads();

  if (a.use_gin)
    return_queue_detail::initialize(a);
  __syncthreads();
  auto *descriptors =
      reinterpret_cast<BackwardMblockDescriptor *>(words + layout.descriptors);
  const auto *padded_offsets = peer_at<const unsigned long long>(
      a.local_base, a.padded_expert_offsets_offset);
  const auto *headers =
      peer_at<const LocalP2pRouteHeader>(a.local_base, a.direct_headers_offset);
  for (std::uint32_t expert = threadIdx.x;
       expert < dispatch_sizes(a.dispatch).owned_expert_count;
       expert += blockDim.x) {
    const auto rows = a.dispatch.expert_row_count[expert];
    words[layout.dw_reduction_rows + expert] =
        ((rows + backward_tuning::kTileK - 1ULL) / backward_tuning::kTileK) *
        backward_tuning::kTileK;
  }
  for (std::uint32_t mblock = threadIdx.x; mblock < a.mblocks;
       mblock += blockDim.x) {
    const auto padded_row =
        static_cast<std::uint64_t>(mblock) * backward_tuning::kTileM;
    std::uint32_t expert = 0U;
    while (expert < dispatch_sizes(a.dispatch).owned_expert_count &&
           padded_row >= padded_offsets[expert + 1U])
      ++expert;
    if (expert >= dispatch_sizes(a.dispatch).owned_expert_count)
      continue;
    const auto local_row = padded_row - padded_offsets[expert];
    const auto expert_rows = a.dispatch.expert_row_count[expert];
    const auto remaining_rows =
        expert_rows > local_row ? expert_rows - local_row : 0ULL;
    const auto valid_rows = static_cast<std::uint32_t>(
        remaining_rows < backward_tuning::kTileM ? remaining_rows
                                                 : backward_tuning::kTileM);
    descriptors[mblock] = BackwardMblockDescriptor{
        static_cast<std::uint32_t>(a.dispatch.expert_row_prefix[expert] +
                                   local_row),
        static_cast<std::uint32_t>(padded_row), expert, valid_rows};
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
    auto *setup =
        peer_at<unsigned long long>(a.peer_bases[peer], a.setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(setup[a.rank]);
    published.store(execution_epoch(a), ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_setup =
      peer_at<unsigned long long>(a.local_base, a.setup_epoch_offset);
  for (std::uint32_t local_peer = threadIdx.x; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(local_setup[peer]);
    while (published.load(::cuda::memory_order_acquire) != execution_epoch(a))
      __nanosleep(backward_tuning::kSetupIdleNanosleep);
  }
}

// This kernel deliberately reserves the same large dynamic-SMEM footprint as
// a compute CTA. Eight blocks therefore remain one-CTA-per-SM transport
// workers instead of opportunistically co-residing on compute SMs.
__global__ __launch_bounds__(
    backward_tuning::kComputeThreads,
    1) void backward_local_transport_kernel(BackwardKernelArgs a) {
  backward_load_row_geometry(a);
  using namespace backward_pipeline_detail;
  extern __shared__ __align__(1024) unsigned char reservation[];
#if __has_include(<nccl_device.h>)
  if (a.use_gin && blockIdx.x < backward_tuning::kRdmaCtas) {
    backward_gin_detail::rdma_role(a);
    return;
  }
#endif
  const auto relay_ordinal = static_cast<std::uint32_t>(blockIdx.x) -
                             (a.use_gin ? backward_tuning::kRdmaCtas : 0U);
  (void)reservation;
  __shared__ unsigned long long route_shared;
  __shared__ std::uint32_t return_mblock_shared;
  __shared__ std::uint32_t stop_shared;
  __shared__ std::uint32_t progressed_shared;
  __shared__ unsigned return_queue_scratch[64];
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ ::cuda::barrier<::cuda::thread_scope_block>
      dispatch_full[backward_tuning::kComputeThreads / 32U];

  auto *words = scheduler_words(a);
  const auto layout =
      weftmoe::cuda::persistent_v3::backward::make_backward_scheduler_layout(
          a.scheduler_mblock_capacity,
          static_cast<std::uint32_t>(a.hidden / backward_tuning::kDxTileN /
                                     backward_tuning::kActivationClusterN),
          a.scheduler_expert_capacity);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  constexpr std::uint32_t kWarps = backward_tuning::kComputeThreads / 32U;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  const auto copy_bytes =
      static_cast<std::uint32_t>(a.hidden * sizeof(transport::BFloat16));
  const auto local_routes = local_route_count(a);
#if __has_include(<nccl_device.h>)
  __shared__ backward_gin_detail::RelayInboundState gin_inbound;
  __shared__ backward_gin_detail::SourceCompletionState gin_completion;
  const bool owns_gin_context = a.use_gin;
  if (owns_gin_context) {
    backward_gin_detail::initialize_relay_state(a, relay_ordinal, gin_inbound,
                                                gin_completion);
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
      const auto batch = remaining < backward_tuning::kDispatchBatchRows
                             ? static_cast<std::uint32_t>(remaining)
                             : backward_tuning::kDispatchBatchRows;
      for (std::uint32_t local = warp; local < batch; local += kWarps) {
        const auto route =
            a.dispatch_waves.local_route_order[route_shared + local];
        if (lane == 0U) {
          const auto destination = a.dispatch.route_dst_rank[route];
          const auto compact_destination_row = a.dispatch.route_dst_row[route];
          const auto token = route / a.dispatch.identity.top_k;
          auto *destination_dr = a.peer_payloads[destination].dr;
          auto *warp_storage = reinterpret_cast<std::byte *>(reservation) +
                               static_cast<std::size_t>(warp) * copy_bytes;
          auto &full = dispatch_full[warp];
          ::cuda::device::experimental::cp_async_bulk_global_to_shared(
              warp_storage, a.dy + token * a.hidden, copy_bytes, full);
          auto token_barrier =
              ::cuda::device::barrier_arrive_tx(full, 1U, copy_bytes);
          full.wait(static_cast<decltype(token_barrier) &&>(token_barrier));
          ::cuda::device::experimental::cp_async_bulk_shared_to_global(
              destination_dr + compact_destination_row * a.hidden, warp_storage,
              copy_bytes);
          ::cuda::device::experimental::cp_async_bulk_commit_group();
          ::cuda::device::experimental::cp_async_bulk_wait_group_read<0>();
          ::cuda::ptx::cp_async_bulk_wait_group(::cuda::ptx::n32_t<0>{});
        }
      }
      if (lane == 0U)
        __threadfence_system();
      __syncthreads();
      local_notification_detail::publish_dispatch(
          a, batch, blockDim.x, [&](unsigned i) {
            const auto route =
                a.dispatch_waves.local_route_order[route_shared + i];
            return local_notification_detail::Entry{
                a.dispatch.route_dst_rank[route] % a.local_world_size,
                a.dispatch.route_dst_row[route]};
          });
      if (threadIdx.x == 0U) {
        auto *completed_counter =
            reinterpret_cast<PaddedCounter *>(words + layout.dispatched_routes);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(completed_counter->value);
        const auto done =
            completed.fetch_add(batch, ::cuda::memory_order_acq_rel) + batch;
        if (done == local_routes)
          atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
        progressed_shared = 1U;
      }
    }
    __syncthreads();

#if __has_include(<nccl_device.h>)
    if (owns_gin_context) {
      const bool inbound_progress = backward_gin_detail::progress_inbound_once(
          a, gin_inbound, reinterpret_cast<std::byte *>(reservation),
          dispatch_full);
      const bool completion_progress =
          backward_gin_detail::progress_source_completion_once(a,
                                                               gin_completion);
      if (threadIdx.x == 0U && (inbound_progress || completion_progress))
        progressed_shared = 1U;
      __syncthreads();
    }
#endif

    if (threadIdx.x < 32)
      local_notification_detail::progress(a);
    if (!threadIdx.x)
      publish_dispatch_if_ready(a, words, layout);
    __syncthreads();

    if (threadIdx.x == 0U) {
      return_mblock_shared = kNoMblock;
      auto mblock = kNoMblock;
      if (a.use_gin && a.mblocks != 0U) {
        // Preserve expert-major return order for GIN's batched wire queues.
        auto *cursor =
            reinterpret_cast<PaddedCounter *>(words + layout.return_scan);
        const auto candidate = static_cast<std::uint32_t>(
            ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                cursor->value)
                .fetch_add(1ULL, ::cuda::memory_order_relaxed) %
            a.mblocks);
        const auto ready =
            weftmoe::cuda::persistent_v3::backward::backward_dx_ready_value(
                static_cast<std::uint32_t>(
                    a.hidden / backward_tuning::kDxTileN /
                    backward_tuning::kActivationClusterN));
        if (::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>(
                words[layout.dx_completed + candidate])
                .load(::cuda::memory_order_acquire) == ready)
          mblock = candidate;
      } else if (!a.use_gin) {
        TileTask task{};
        if (weftmoe::cuda::persistent_v3::backward::try_claim_domain_once(
                make_return_domain(a, words, layout), task))
          mblock = task.owner;
      }
      if (mblock != kNoMblock) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            phase(words[layout.phase + mblock]);
        auto expected = 3ULL;
        if (phase.compare_exchange_strong(expected, 4ULL,
                                          ::cuda::memory_order_acq_rel,
                                          ::cuda::memory_order_relaxed)) {
          return_mblock_shared = mblock;
          atomicMin(&status->reduce_blocks_done, device_globaltimer_ns());
        } else if (!a.use_gin) {
          __trap(); // The append-only queue must publish each M-block once.
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
      const auto *dx = a.dx_rows;
      const auto *dgate = a.dgate_rows;
      const auto warp = threadIdx.x / 32U;
      const auto lane = threadIdx.x & 31U;
      for (std::uint32_t local_row = warp; local_row < descriptor.valid_rows;
           local_row += kWarps) {
        const auto compact_row = descriptor.compact_row + local_row;
        const auto padded_row = descriptor.padded_row + local_row;
        const auto header = headers[compact_row];
        if (a.use_gin && header.source_rank / a.local_world_size != a.node)
          continue;
        const auto *source = reinterpret_cast<const uint4 *>(
            dx + static_cast<std::uint64_t>(padded_row) * a.hidden);
        auto *destination = a.peer_payloads[header.source_rank].contribution;
        auto *target = reinterpret_cast<uint4 *>(destination +
                                                 header.route_index * a.hidden);
        for (std::uint64_t vector = lane; vector < a.hidden / 8ULL;
             vector += 32ULL)
          target[vector] = source[vector];
        if (lane == 0U) {
          auto *destination_dgate = a.peer_payloads[header.source_rank].dgate;
          destination_dgate[header.route_index] = dgate[padded_row];
        }
      }
      __threadfence_system();
      __syncthreads();
      local_notification_detail::publish_return(
          a, descriptor.valid_rows, blockDim.x, [&](unsigned i) {
            const auto header = headers[descriptor.compact_row + i];
            return local_notification_detail::Entry{
                header.source_rank / a.local_world_size == a.node
                    ? header.source_rank % a.local_world_size
                    : communication::kLocalPeers,
                header.route_index};
          });
      if (a.use_gin)
        return_queue_detail::publish<backward_tuning::kComputeThreads>(
            a, descriptor.compact_row, descriptor.valid_rows,
            return_queue_scratch, [] { return false; });
      if (threadIdx.x == 0U) {
        auto *counter =
            reinterpret_cast<PaddedCounter *>(words + layout.returned_mblocks);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            returned(counter->value);
        const auto done =
            returned.fetch_add(1ULL, ::cuda::memory_order_acq_rel) + 1ULL;
        if (done == a.mblocks)
          atomicMax(&status->return_complete_clock, device_globaltimer_ns());
        progressed_shared = 1U;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0U) {
      auto *dispatch_counter =
          reinterpret_cast<PaddedCounter *>(words + layout.dispatched_routes);
      auto *return_counter =
          reinterpret_cast<PaddedCounter *>(words + layout.returned_mblocks);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          dispatched(dispatch_counter->value);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          returned(return_counter->value);
      stop_shared =
          dispatched.load(::cuda::memory_order_acquire) == local_routes &&
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
  local_notification_detail::drain(a);
  __syncthreads();
}

#include "cuda/persistent_v3/backward/unified_compute.cuh"

template <std::uint32_t DabTileN, std::uint32_t DxTileN, std::uint32_t DwTileN>
__global__ __launch_bounds__(
    backward_tuning::kComputeThreads,
    1) void backward_local_compute_kernel(BackwardKernelArgs a,
                                          const __grid_constant__ CUtensorMap
                                              dr_map,
                                          const __grid_constant__ CUtensorMap
                                              dab_map,
                                          const __grid_constant__ CUtensorMap
                                              w2_map,
                                          const __grid_constant__ CUtensorMap
                                              w1_map,
                                          const __grid_constant__ CUtensorMap
                                              dr_mn_map,
                                          const __grid_constant__ CUtensorMap
                                              u_mn_map,
                                          const __grid_constant__ CUtensorMap
                                              dab_mn_map,
                                          const __grid_constant__ CUtensorMap
                                              x_mn_map,
                                          const __grid_constant__ CUtensorMap
                                              u_store_map,
                                          const __grid_constant__ CUtensorMap
                                              dx_store_map,
                                          const __grid_constant__ CUtensorMap
                                              dw2_store_map,
                                          const __grid_constant__ CUtensorMap
                                              dw1_store_map) {
  backward_load_row_geometry(a);
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  constexpr std::uint32_t kActivationStorageTileN =
      DabTileN > DxTileN ? DabTileN : DxTileN;
  constexpr std::uint32_t kStorageTileN =
      kActivationStorageTileN > DwTileN ? kActivationStorageTileN : DwTileN;
  extern __shared__ __align__(1024) unsigned char storage[];
  __shared__ backward_unified_compute_detail::SharedState state;
  backward_compute_detail::initialize_k_mn_pipeline<kStorageTileN>(storage);
  backward_compute_detail::FullCtaComputeSynchronization synchronization;
  backward_unified_compute_detail::run_compute<DabTileN, DxTileN, DwTileN>(
      a, dr_map, dab_map, w2_map, w1_map, dr_mn_map, u_mn_map, dab_mn_map,
      x_mn_map, u_store_map, dx_store_map, dw2_store_map, dw1_store_map,
      backward_unified_compute_detail::RowGeometry{a.mblocks, a.active_experts},
      storage, state, synchronization);
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
