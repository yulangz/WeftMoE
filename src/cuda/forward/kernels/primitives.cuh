#include "cuda/forward/kernel_support.cuh"
#include "cuda/static_v4/memory_order.cuh"

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) // compiled production MMA path only
// The paired Forward kernel gives FC1 and FC2 independent N shapes.
using TunedForwardFc1Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc1TileN>::type;
using TunedForwardFc2Mma =
    typename hopper::BF16MMASelector<forward_tuning::kFc2TileN>::type;
using ForwardBarrier = cutlass::arch::ClusterTransactionBarrier;

#endif

// Compute and expert storage use M128. Notify tiles may be smaller; the
// input publisher waits for every contributing Notify tile before release.
__device__ __forceinline__ std::uint64_t
forward_expert_mblock_count(const ForwardKernelArgs &a, std::uint32_t expert) {
  const auto rows = a.dispatch.expert_row_count[expert];
  return (rows + forward_tuning::kTileM - 1ULL) / forward_tuning::kTileM;
}

__device__ __forceinline__ std::uint64_t
forward_mblock_local_row(std::uint32_t local_mblock) {
  return static_cast<std::uint64_t>(local_mblock) * forward_tuning::kTileM;
}

__device__ __forceinline__ std::uint32_t
forward_mblock_valid_rows(const ForwardKernelArgs &a, std::uint32_t expert,
                          std::uint32_t local_mblock) {
  const auto local_row = forward_mblock_local_row(local_mblock);
  const auto remaining = a.dispatch.expert_row_count[expert] - local_row;
  constexpr auto extent = forward_tuning::kTileM;
  return static_cast<std::uint32_t>(remaining < extent ? remaining : extent);
}

enum ForwardMixedSchedulerWord : std::uint32_t {
  // The last three words in the aligned scheduler gap form a transport-only
  // startup rendezvous.  Keeping one immutable epoch publication and two
  // monotonic arrival counters avoids resetting a reusable barrier while a
  // slower CTA can still be observing the previous phase.
  kForwardTransportInitEpoch = 5U,
  kForwardTransportClearArrivals = 6U,
  kForwardTransportMapArrivals = 7U,
};

__device__ __forceinline__ unsigned long long *
forward_mixed_scheduler_word(const ForwardKernelArgs &a,
                             unsigned long long *scheduler,
                             ForwardMixedSchedulerWord word) {
  // Return-directory construction owns two arrays of one word per
  // (remote-node, GIN-context), beginning at 16 + 2*world.  Mixed scheduling
  // state must live after both arrays; placing it in the apparent gap before
  // that base aliases the arrays for world=2/16 once directory scans begin.
  const auto partitions = a.dispatch.identity.num_nodes - 1ULL;
  const auto mixed_base =
      16ULL + 2ULL * a.world + 2ULL * partitions * a.gin_context_count;
  return scheduler + mixed_base + static_cast<std::uint32_t>(word);
}

__device__ __forceinline__ void
forward_transport_wait_epoch(unsigned long long *word,
                             unsigned long long epoch) {
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        published(*word);
    while (published.load(::cuda::memory_order_acquire) != epoch)
      __nanosleep(256U);
  }
  __syncthreads();
}

// A fixed transport grid needs only a CTA-arrival barrier, not cooperative
// launch semantics.  Every CTA first releases all of its lanes' stores through
// one acq_rel RMW.  Waiting on the terminal counter value then acquires the
// complete RMW release sequence before the next initialization phase begins.
__device__ __forceinline__ void
forward_transport_cta_barrier(unsigned long long *arrivals,
                              unsigned int expected_ctas) {
  __syncthreads();
  if (threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> count(
        *arrivals);
    count.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
    while (count.load(::cuda::memory_order_acquire) < expected_ctas)
      __nanosleep(256U);
  }
  __syncthreads();
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
// Build the same startup-wave/expert-body order with one warp per expert.
// Prefixes are reduced across lanes once instead of searching every preceding
// tile independently for each output entry. No persistent scratch is needed.
template <bool SingleCta = false, class Emit>
__device__ __forceinline__ void
forward_initialize_mblocks(const ForwardKernelArgs &a, Emit emit) {
  constexpr unsigned startup = forward_tuning::kStartupMblocks;
  const auto lane = threadIdx.x % 32U;
  const auto experts = dispatch_sizes(a.dispatch).owned_expert_count;
  auto sum = [](unsigned long long value) {
    for (unsigned delta = 16U; delta; delta /= 2U)
      value += __shfl_down_sync(0xffffffffU, value, delta);
    return __shfl_sync(0xffffffffU, value, 0U);
  };
  const auto first_expert =
      SingleCta ? threadIdx.x / 32U
                : (blockIdx.x * blockDim.x + threadIdx.x) / 32U;
  const auto expert_stride =
      SingleCta ? blockDim.x / 32U : gridDim.x * blockDim.x / 32U;
  for (unsigned expert = first_expert; expert < experts;
       expert += expert_stride) {
    const auto blocks = forward_expert_mblock_count(a, expert);
    if (blocks == 0ULL)
      continue;
    if (a.forward_fc1_locality_group > 1U) {
      unsigned long long prefix[startup + 1U] = {};
      for (unsigned other = lane; other < experts; other += 32U) {
        const auto count = forward_expert_mblock_count(a, other);
#pragma unroll
        for (unsigned tile = 0; tile < startup; ++tile)
          prefix[tile] +=
              (count < tile ? count : tile) + (other < expert && count > tile);
        prefix[startup] +=
            (count < startup ? count : startup) +
            (other < expert && count > startup ? count - startup : 0ULL);
      }
#pragma unroll
      for (unsigned tile = 0; tile <= startup; ++tile)
        prefix[tile] = sum(prefix[tile]);
      for (unsigned long long tile = lane; tile < blocks; tile += 32U) {
        auto ordinal = prefix[startup] + tile - startup;
#pragma unroll
        for (unsigned first = 0; first < startup; ++first)
          if (tile == first)
            ordinal = prefix[first];
        emit(ordinal, expert, unsigned(tile));
      }
    } else {
      for (unsigned tile = 0; tile < blocks; ++tile) {
        unsigned long long ordinal = 0;
        for (unsigned other = lane; other < experts; other += 32U) {
          const auto count = forward_expert_mblock_count(a, other);
          ordinal +=
              (count < tile ? count : tile) + (other < expert && count > tile);
        }
        ordinal = sum(ordinal);
        if (!lane)
          emit(ordinal, expert, tile);
      }
    }
  }
}

#endif

// The cancellation nonce belongs to the host invocation. CUDA Graph replay
// advances queue epochs on device while retaining this captured nonce.
__device__ bool forward_cancelled(const ForwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(const_cast<unsigned long long *>(a.cancel_epoch),
                          0ULL) == a.epoch;
}

template <bool StaticLocal>
__device__ __forceinline__ bool
forward_mblock_rows_ready(const ForwardKernelArgs &a, std::uint32_t expert,
                          std::uint64_t local_row, std::uint32_t valid_rows) {
  if constexpr (StaticLocal) {
    if (!a.forward_stream_local_dispatch)
      return true;
  }
  if ((!a.use_gin && !a.forward_stream_local_dispatch) || valid_rows == 0U)
    return true;
  if constexpr (StaticLocal) {
    if (a.fused_inference_routes) {
      // Fused inference builds M128 geometry directly. Unlike V3 Notify's
      // smaller tiles, exactly one arrival counter covers this compute tile.
      const auto tile = a.dispatch.mtile_row_offsets[expert] +
                        local_row / forward_tuning::kTileM;
      const auto *arrived = peer_at<unsigned long long>(
          a.local_base, a.x_arrived_rows_offset);
      return static_v4::load_acquire_device(
                 arrived + tile * a.local_world_size +
                     a.rank % a.local_world_size) >= valid_rows;
    }
  }
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived =
      peer_at<unsigned long long>(a.local_base, a.x_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    unsigned long long total = 0ULL;
    for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(arrived[tile * a.local_world_size + rail]);
      total += ready.load(::cuda::memory_order_acquire);
    }
    if (total < expected)
      return false;
    tile_row = tile_row_end;
  }
  return true;
}

template <bool StaticLocal>
__device__ void wait_forward_mblock_rows(
    const ForwardKernelArgs &a, std::uint32_t expert, std::uint64_t local_row,
    std::uint32_t valid_rows,
    transport_state::GenerationCompletionVector *completion) {
  if constexpr (StaticLocal)
    return;
  if (!a.use_gin || valid_rows == 0U)
    return;
  // One elected lane polls the peer-visible rail counters; the CTA-level
  // synchronization at the call site releases the remaining lanes. Polling
  // from all 384 threads only multiplies L2/system traffic while conveying no
  // additional readiness information.
  if (threadIdx.x != 0U)
    return;
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived =
      peer_at<unsigned long long>(a.local_base, a.x_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    auto load_arrived = [&]() {
      unsigned long long total = 0ULL;
      for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(arrived[tile * a.local_world_size + rail]);
        total += ready.load(::cuda::memory_order_acquire);
      }
      return total;
    };
    auto current = load_arrived();
#if WEFTMOE_DEBUG_VALIDATE
    if (current < expected && threadIdx.x == 0U) {
      atomicAdd(&completion->tile_waiter_count, 1ULL);
      (void)atomicCAS(&completion->first_waiting_tile, ~0ULL, tile);
      if (completion->first_waiting_tile == tile) {
        completion->first_wait_expected = expected;
        completion->first_wait_arrived = current;
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
      }
    }
#endif
    while (current < expected && !forward_cancelled(a)) {
#if WEFTMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        completion->first_wait_arrived = current;
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
#endif
      current = load_arrived();
    }
    tile_row = tile_row_end;
  }
}

__device__ transport_state::TaskCompletionRecord &
completion_task(transport_state::GenerationCompletionVector *completion,
                transport_state::TaskClass kind) {
  return completion->tasks[static_cast<std::size_t>(kind)];
}

__device__ void
mark_task_space_complete(transport_state::TaskCompletionRecord &task) {
  task.claimed = task.expected;
  task.completed = task.expected;
  task.inflight = 0ULL;
}

__device__ void
claim_completion_tasks(transport_state::GenerationCompletionVector *completion,
                       transport_state::TaskClass kind,
                       unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.claimed, count);
  atomicAdd(&task.inflight, count);
}

__device__ void complete_completion_tasks(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind, unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.completed, count);
  atomicAdd(&task.inflight, 0ULL - count);
}

__device__ bool completion_vector_ready(
    const transport_state::GenerationCompletionVector &completion) {
  for (std::size_t index = 0; index < transport_state::kTaskClassCount;
       ++index) {
    const auto task = completion.tasks[index];
    if (task.claimed != task.expected || task.completed != task.expected ||
        task.inflight != 0ULL)
      return false;
  }
  return completion.directory_ready == completion.directory_expected &&
         completion.dispatch_local_completion_completed ==
             completion.dispatch_local_completion_expected &&
         completion.return_local_completion_completed ==
             completion.return_local_completion_expected &&
         completion.ring_credit_restored == completion.ring_credit_expected;
}

__device__ void publish_elected_terminal(
    transport_state::GenerationCompletionVector *completion,
    unsigned long long epoch, bool error) {
  if (!error && !completion_vector_ready(*completion))
    return;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> published(
      completion->kernel_complete_epoch);
  published.store(epoch, ::cuda::memory_order_release);
  completion->terminal_publish_count += 1ULL;
}

__device__ void forward_publish_progress(const ForwardKernelArgs &a,
                                         std::uint32_t word,
                                         unsigned long long value) {
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
  if (a.progress != nullptr && blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> field(
        a.progress[word]);
    field.store(value, ::cuda::memory_order_release);
  }
#else
  (void)a;
  (void)word;
  (void)value;
#endif
}

template <bool StaticLocal>
__device__ __forceinline__ bool remote_backend_enabled(bool enabled) {
  if constexpr (StaticLocal)
    return false;
  return enabled;
}

#include "cuda/persistent_v3/forward/dispatch_pipeline.cuh"
#include "cuda/persistent_v3/forward/finalize_pipeline.cuh"

#include "cuda/static_v4/contract.cuh"
#include "cuda/static_v4/network_access.cuh"
#include "cuda/static_v4/ring_access.cuh"
#include "cuda/static_v4/return_tx.cuh"
#include "cuda/static_v4/network_ring.cuh"
#include "cuda/persistent_v3/forward/compute_kernel.cuh"

#include "cuda/forward/initialize_pipeline.cuh"
#include "cuda/static_v4/initialize.cuh"
#include "cuda/static_v4/collective.cuh"
#include "cuda/static_v4/route_plan.cuh"
#include "cuda/static_v4/network_collective.cuh"
#include "cuda/static_v4/network_route_exchange.cuh"
#include "cuda/static_v4/single_node_route_exchange.cuh"
#include "cuda/static_v4/single_node_finalize.cuh"
#include "cuda/static_v4/ring_plan.cuh"
#include "cuda/static_v4/ring_admission.cuh"
#include "cuda/persistent_v3/forward/unified_kernel.cuh"
