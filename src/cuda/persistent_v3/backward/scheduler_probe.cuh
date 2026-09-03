#pragma once

#include <cstdint>

#include "cuda/persistent_v3/backward/scheduler.cuh"
#include "cuda/persistent_v3/queues.cuh"

namespace megamoe::cuda::persistent_v3::backward {

inline constexpr std::uint32_t kSchedulerProbeExperts = 4U;
inline constexpr std::uint32_t kSchedulerProbeMblocksPerExpert = 3U;
inline constexpr std::uint32_t kSchedulerProbeMblocks =
    kSchedulerProbeExperts * kSchedulerProbeMblocksPerExpert;
inline constexpr std::uint32_t kSchedulerProbeDabTiles = 5U;
inline constexpr std::uint32_t kSchedulerProbeDxTiles = 3U;
inline constexpr std::uint32_t kSchedulerProbeDw2Tiles = 7U;
// Cross a 64-bit claim-word boundary so the probe exercises the same
// multiword owner geometry used by production dW domains.
inline constexpr std::uint32_t kSchedulerProbeDw1Tiles = 67U;
inline constexpr std::uint32_t kSchedulerProbeDabTasks =
    kSchedulerProbeMblocks * kSchedulerProbeDabTiles;
inline constexpr std::uint32_t kSchedulerProbeDxTasks =
    kSchedulerProbeMblocks * kSchedulerProbeDxTiles;
inline constexpr std::uint32_t kSchedulerProbeDw2Tasks =
    kSchedulerProbeExperts * kSchedulerProbeDw2Tiles;
inline constexpr std::uint32_t kSchedulerProbeDw1Tasks =
    kSchedulerProbeExperts * kSchedulerProbeDw1Tiles;
inline constexpr std::uint32_t kSchedulerProbeTotalTasks =
    kSchedulerProbeDabTasks + kSchedulerProbeDxTasks +
    kSchedulerProbeDw2Tasks + kSchedulerProbeDw1Tasks;
inline constexpr std::uint32_t kSchedulerProbeMaxCtas = 256U;
inline constexpr unsigned long long kSchedulerProbeEpoch = 23ULL;

static_assert(kSchedulerProbeTotalTasks == 392U);

// This workspace intentionally uses exactly the same summary/claim geometry
// as production, only at a small scale. DAB owners are M-blocks; dW owners are
// experts. Every payload descriptor remains implicit in (owner, tile).
struct alignas(128) BackwardSchedulerProbeWorkspace {
  unsigned long long dx_ready[1];
  unsigned long long dab_ready[1];
  unsigned long long dw2_ready[1];
  unsigned long long dw1_ready[1];

  unsigned long long dx_claim[kSchedulerProbeMblocks];
  unsigned long long dab_claim[kSchedulerProbeMblocks];
  unsigned long long dw2_claim[kSchedulerProbeExperts];
  unsigned long long dw1_claim[kSchedulerProbeExperts * 2U];
  PaddedCounter scan[kTaskKindCount];

  PaddedCounter dab_tiles_completed[kSchedulerProbeMblocks];
  PaddedCounter expert_dab_tiles_completed[kSchedulerProbeExperts];
  PaddedCounter completed[kTaskKindCount];
  PaddedCounter total_completed;

  unsigned int visits[kSchedulerProbeTotalTasks];
  unsigned long long cta_claims[kSchedulerProbeMaxCtas];
  PaddedCounter duplicate_claims;
  PaddedCounter dependency_errors;
  PaddedCounter priority_errors;
  PaddedCounter timeout_errors;
  PaddedCounter terminal_epoch;
  PaddedCounter terminal_publish_count;

  unsigned long long priority_ready[kTaskKindCount];
  unsigned long long priority_claim[kTaskKindCount];
  PaddedCounter priority_scan[kTaskKindCount];
};

#if defined(__CUDACC__)
MEGAMOE_V3_DEVICE inline WorkQueues
make_scheduler_probe_queues(BackwardSchedulerProbeWorkspace &work) {
  return WorkQueues{
      TileDomain{work.dx_ready, work.dx_claim,
                 &work.scan[static_cast<std::uint32_t>(TaskKind::Dx)],
                 kSchedulerProbeMblocks, kSchedulerProbeDxTiles,
                 TaskKind::Dx},
      TileDomain{work.dab_ready, work.dab_claim,
                 &work.scan[static_cast<std::uint32_t>(TaskKind::Dab)],
                 kSchedulerProbeMblocks, kSchedulerProbeDabTiles,
                 TaskKind::Dab},
      TileDomain{work.dw2_ready, work.dw2_claim,
                 &work.scan[static_cast<std::uint32_t>(TaskKind::Dw2)],
                 kSchedulerProbeExperts, kSchedulerProbeDw2Tiles,
                 TaskKind::Dw2},
      TileDomain{work.dw1_ready, work.dw1_claim,
                 &work.scan[static_cast<std::uint32_t>(TaskKind::Dw1)],
                 kSchedulerProbeExperts, kSchedulerProbeDw1Tiles,
                 TaskKind::Dw1}};
}

MEGAMOE_V3_DEVICE inline WorkQueues
make_scheduler_priority_queues(BackwardSchedulerProbeWorkspace &work) {
  return WorkQueues{
      TileDomain{&work.priority_ready[0], &work.priority_claim[0],
                 &work.priority_scan[0], 1U, 1U, TaskKind::Dx},
      TileDomain{&work.priority_ready[1], &work.priority_claim[1],
                 &work.priority_scan[1], 1U, 1U, TaskKind::Dab},
      TileDomain{&work.priority_ready[2], &work.priority_claim[2],
                 &work.priority_scan[2], 1U, 1U, TaskKind::Dw2},
      TileDomain{&work.priority_ready[3], &work.priority_claim[3],
                 &work.priority_scan[3], 1U, 1U, TaskKind::Dw1}};
}

MEGAMOE_V3_DEVICE inline void scheduler_probe_increment(
    unsigned long long &word) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> value(
      word);
  value.fetch_add(1ULL, ::cuda::memory_order_relaxed);
}

MEGAMOE_V3_DEVICE inline std::uint32_t
scheduler_probe_visit_index(const TileTask &task) {
  switch (task.kind) {
  case TaskKind::Dx:
    return task.owner * kSchedulerProbeDxTiles + task.tile;
  case TaskKind::Dab:
    return kSchedulerProbeDxTasks +
           task.owner * kSchedulerProbeDabTiles + task.tile;
  case TaskKind::Dw2:
    return kSchedulerProbeDxTasks + kSchedulerProbeDabTasks +
           task.owner * kSchedulerProbeDw2Tiles + task.tile;
  case TaskKind::Dw1:
    return kSchedulerProbeDxTasks + kSchedulerProbeDabTasks +
           kSchedulerProbeDw2Tasks +
           task.owner * kSchedulerProbeDw1Tiles + task.tile;
  default:
    return kSchedulerProbeTotalTasks;
  }
}

MEGAMOE_V3_DEVICE inline void complete_scheduler_probe_task(
    BackwardSchedulerProbeWorkspace &work, WorkQueues queues,
    const TileTask &task) {
  const auto visit = scheduler_probe_visit_index(task);
  if (visit >= kSchedulerProbeTotalTasks) {
    scheduler_probe_increment(work.dependency_errors.value);
    return;
  }
  if (atomicAdd(&work.visits[visit], 1U) != 0U)
    scheduler_probe_increment(work.duplicate_claims.value);

  const auto kind = static_cast<std::uint32_t>(task.kind);
  if (task.kind == TaskKind::Dab) {
    const auto expert = task.owner / kSchedulerProbeMblocksPerExpert;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        owner_done(work.dab_tiles_completed[task.owner].value);
    const auto owner_previous =
        owner_done.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
    if (owner_previous + 1ULL == kSchedulerProbeDabTiles)
      publish_owner(queues.dx, task.owner);

    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        expert_done(work.expert_dab_tiles_completed[expert].value);
    const auto expert_previous =
        expert_done.fetch_add(1ULL, ::cuda::memory_order_acq_rel);
    if (expert_previous + 1ULL ==
        kSchedulerProbeMblocksPerExpert * kSchedulerProbeDabTiles) {
      publish_owner(queues.dw2, expert);
      publish_owner(queues.dw1, expert);
    }
  } else if (task.kind == TaskKind::Dx) {
    const auto done = persistent_v3::load_acquire<
        ::cuda::thread_scope_device>(
        work.dab_tiles_completed[task.owner].value);
    if (done != kSchedulerProbeDabTiles)
      scheduler_probe_increment(work.dependency_errors.value);
  } else if (task.kind == TaskKind::Dw2 || task.kind == TaskKind::Dw1) {
    const auto done = persistent_v3::load_acquire<
        ::cuda::thread_scope_device>(
        work.expert_dab_tiles_completed[task.owner].value);
    if (done !=
        kSchedulerProbeMblocksPerExpert * kSchedulerProbeDabTiles)
      scheduler_probe_increment(work.dependency_errors.value);
  }

  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      completed(work.completed[kind].value);
  completed.fetch_add(1ULL, ::cuda::memory_order_release);
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> total(
      work.total_completed.value);
  total.fetch_add(1ULL, ::cuda::memory_order_release);
}

__global__ void backward_scheduler_probe_setup_kernel(
    BackwardSchedulerProbeWorkspace *work) {
  if (blockIdx.x != 0U || threadIdx.x != 0U)
    return;

  auto priority = make_scheduler_priority_queues(*work);
  publish_owner(priority.dw1, 0U);
  publish_owner(priority.dw2, 0U);
  publish_owner(priority.dab, 0U);
  publish_owner(priority.dx, 0U);
  constexpr TaskKind expected[] = {TaskKind::Dx, TaskKind::Dab,
                                   TaskKind::Dw2, TaskKind::Dw1};
  for (const auto kind : expected) {
    TileTask task{};
    if (!try_claim_work(priority, task) || task.kind != kind)
      scheduler_probe_increment(work->priority_errors.value);
  }

  auto queues = make_scheduler_probe_queues(*work);
  for (std::uint32_t owner = 0U; owner < kSchedulerProbeMblocks; ++owner)
    publish_owner(queues.dab, owner);
}

__global__ __launch_bounds__(backward_tuning::kComputeThreads, 1)
void backward_scheduler_probe_kernel(BackwardSchedulerProbeWorkspace *work) {
  if (threadIdx.x == 0U) {
    auto queues = make_scheduler_probe_queues(*work);
    unsigned long long local_claims = 0ULL;
    unsigned long long idle_probes = 0ULL;
    while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
               work->terminal_epoch.value) < kSchedulerProbeEpoch) {
      if (persistent_v3::load_acquire<::cuda::thread_scope_device>(
              work->total_completed.value) >= kSchedulerProbeTotalTasks) {
        auto expected = 0ULL;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            terminal(work->terminal_epoch.value);
        if (terminal.compare_exchange_strong(
                expected, kSchedulerProbeEpoch, ::cuda::memory_order_acq_rel,
                ::cuda::memory_order_acquire))
          scheduler_probe_increment(work->terminal_publish_count.value);
        continue;
      }

      TileTask task{};
      if (try_claim_work(queues, task)) {
        complete_scheduler_probe_task(*work, queues, task);
        ++local_claims;
        idle_probes = 0ULL;
        continue;
      }

      if (++idle_probes == 5000000ULL) {
        scheduler_probe_increment(work->timeout_errors.value);
        auto expected = 0ULL;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            terminal(work->terminal_epoch.value);
        if (terminal.compare_exchange_strong(
                expected, kSchedulerProbeEpoch, ::cuda::memory_order_acq_rel,
                ::cuda::memory_order_acquire))
          scheduler_probe_increment(work->terminal_publish_count.value);
        break;
      }
      __nanosleep(backward_tuning::kQueueIdleNanosleep);
    }
    if (blockIdx.x < kSchedulerProbeMaxCtas)
      work->cta_claims[blockIdx.x] = local_claims;
  }
  __syncthreads();
}
#endif

} // namespace megamoe::cuda::persistent_v3::backward
