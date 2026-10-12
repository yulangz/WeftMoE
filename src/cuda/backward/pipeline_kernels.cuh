#pragma once

namespace backward_pipeline_detail {

using weftmoe::cuda::pipeline::PaddedCounter;
using weftmoe::cuda::pipeline::backward::BackwardMblockDescriptor;
using weftmoe::cuda::pipeline::backward::TaskKind;
using weftmoe::cuda::pipeline::backward::TileTask;

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
              weftmoe::cuda::pipeline::backward::kDabPhaseTileCapacity);
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

__device__ __forceinline__ bool try_publish_dispatch_mblock(
    const BackwardKernelArgs &a, unsigned long long *words,
    const weftmoe::cuda::pipeline::backward::BackwardSchedulerLayout &layout,
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

  // The unified transport already assigns one publisher per M-block.
  phase.store(1ULL, ::cuda::memory_order_release);
  return true;
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
      const auto layout = pipeline::backward::make_backward_scheduler_layout(
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

#include "cuda/backward/gin_pipeline.cuh"

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
      weftmoe::cuda::pipeline::backward::make_backward_scheduler_layout(
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
  if (a.return_ring.depth)
    weftmoe_pf::backward_return::initialize(a);
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
#include "cuda/backward/unified_compute.cuh"
