#pragma once

// Included inside forward_unified_transport. Independent NVLink Dispatch and
// Return domains use disjoint TMA slots and their own barriers. A ready Return
// batch cannot delay subsequent inputs, regardless of the token distribution.
template <unsigned Threads, unsigned BaseThread, unsigned SegmentBytes,
          unsigned Buffers, unsigned DispatchLanes>
__device__ __forceinline__ void run_local_warps(
    const ForwardKernelArgs &a, persistent_v3::ForwardPipelineControl &control,
    unsigned long long *scheduler, Shared<Threads, SegmentBytes, Buffers, DispatchLanes> &s) {
  using Storage = Shared<Threads, SegmentBytes, Buffers, DispatchLanes>;
  static_assert(
      Storage::kCopyWarps >= 2U,
      "Independent directions require two copy warps and a publisher");
  constexpr unsigned dispatch_warps = (Storage::kCopyWarps + 1U) / 2U;
  constexpr unsigned return_warps = Storage::kCopyWarps - dispatch_warps;
  constexpr unsigned return_base = BaseThread + dispatch_warps * 32U;
  constexpr unsigned return_threads = return_warps * 32U;
  const unsigned tid = local_thread<BaseThread>();
  const unsigned warp = tid / 32U;
  init<BaseThread>(a, control, s);
  if (warp < dispatch_warps) {
    while (true) {
      if (tid == 0U)
        s.dispatch_complete =
            cancelled(a, control) || s.dispatch_cursor >= s.dispatch_end;
      copy_sync<dispatch_warps * 32U, 8U>();
      if (s.dispatch_complete)
        break;
      progress_dispatch<BaseThread, dispatch_warps, 0U, 8U>(a, s);
    }
    if (tid == 0U) {
      const auto previous =
          persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
              control.relay_dispatch_roles_complete.value, 1ULL);
      if (previous + 1ULL == gridDim.x) {
        peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
            ->dispatch_complete_clock = device_globaltimer_ns();
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, XDispatchComplete,
                                   ProxyRelay, 0U, s.dispatch_end,
                                   execution_epoch(a));
      }
    }
  } else if (warp < Storage::kCopyWarps) {
    const unsigned return_tid = local_thread<return_base>();
    while (true) {
      copy_sync<return_threads, 6U>();
      const bool progressed =
          progress_return<return_base, return_warps, dispatch_warps, 6U, true, true>(a, s);
      if (return_tid == 0U) {
        forward_return_detail::LocalReturnState state;
        state.queue_index = s.return_queue;
        s.stop = cancelled(a, control) ||
                 forward_return_detail::local_return_terminal(a, state);
      }
      copy_sync<return_threads, 6U>();
      if (s.stop)
        break;
      if (!progressed)
        __nanosleep(forward_tuning::kReturnIdleNanosleep);
    }
    WEFTMOE_TIMELINE_RECORD_IF(a.timeline, return_tid == 0U,
                               LocalReturnTerminal, LocalReturn, blockIdx.x,
                               s.return_cursor, execution_epoch(a));
    if (return_tid < 32U)
      // The live publisher owns this SPSC consumer through copies_done.
      // A second drainer must not race its non-CAS head advancement.
      drain_notifications<return_base, false>(a, [&] { return cancelled(a, control); });
    copy_sync<return_threads, 6U>();
    if (return_tid == 0U) {
      atomicMax(&peer_at<TransportDeviceStatus>(a.local_base, a.status_offset)
                     ->return_complete_clock,
                device_globaltimer_ns());
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.rdma_return_roles_complete.value, 1ULL);
      persistent_v3::fetch_add_release<::cuda::thread_scope_device>(
          control.return_receive_roles_complete.value, 1ULL);
      ::cuda::atomic_ref<unsigned, ::cuda::thread_scope_block>(s.copies_done)
          .store(1U, ::cuda::memory_order_release);
    }
  } else {
    run_publisher(a, control, scheduler, s);
  }
  domain_sync<Storage::kThreads>();
  WEFTMOE_TIMELINE_RECORD_IF(a.timeline, tid == 0U, TransportRoleDone,
                             LocalReturn, blockIdx.x, 0ULL, execution_epoch(a));
}
