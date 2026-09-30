#pragma once

namespace weftmoe_pf {

template <unsigned Fc1TileN, unsigned Fc2TileN>
__device__ __forceinline__ void finalize(const ForwardKernelArgs &a) {
  constexpr auto kLeader = forward_tuning::kMathWarpgroups * 128U;
  if (blockIdx.x != 0U || threadIdx.x != kLeader)
    return;

  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (a.injected_status_error_rank == static_cast<int>(a.rank))
    publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
  const auto *scheduler = peer_at<const unsigned long long>(
      a.local_base, a.forward_scheduler_offset);
  const auto mblocks = scheduler[3];
  status->fc1_tasks_done =
      mblocks * ((2ULL * a.intermediate + Fc1TileN - 1ULL) / Fc1TileN);
  status->fc2_tasks_done =
      mblocks * ((a.hidden + Fc2TileN - 1ULL) / Fc2TileN);
  status->complete_clock = device_globaltimer_ns();
  WEFTMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL,
                          execution_epoch(a));
}

} // namespace weftmoe_pf
