#pragma once

#include "cuda/persistent_v3/forward/return_pipeline.cuh"

// Split Forward terminal protocol. Receive-completion CTAs publish remote
// contribution readiness per chunk; RDMA local completion and the compute-grid
// reduction counter are only terminal dependencies. Keeping those edges
// separate lets Reduce consume completed tokens without a transport-grid
// rendezvous.

namespace forward_finalize_detail {

__device__ __forceinline__ bool wait_for_count(
    const ForwardKernelArgs &a, persistent_v3::PaddedCounter &counter,
    unsigned long long expected) {
  while (persistent_v3::load_acquire<::cuda::thread_scope_device>(
             counter.value) < expected) {
    if (forward_cancelled(a))
      return false;
    __nanosleep(forward_tuning::kFinalizeNanosleep);
  }
  return true;
}

// One-node split Forward has Relay and Compute in independent grids, but no
// RDMA/completion-vector roles.  Relay CTA 0 performs the same fixed terminal
// agreement only after every local Return owner and every Reduce role has
// released its counter. Idle transport consumers are covered by the separate
// completed-token count. No grid-wide barrier is introduced.
__device__ __forceinline__ void local_split_transport_finalize(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    unsigned long long *scheduler, TransportDeviceStatus *status) {
  if (blockIdx.x != 0U)
    return;

  if (threadIdx.x == 0U) {
    wait_for_count(a, control.rdma_return_roles_complete,
                   a.forward_relay_cta_count);
    if (a.forward_dynamic_reduce_queue)
      wait_for_count(a, control.reduce_tokens_complete,
                     a.dispatch.identity.num_tokens);
    wait_for_count(a, control.compute_reduce_roles_complete,
                   a.forward_reduce_cta_count);
    forward_publish_progress(a, kForwardReturnObserved, scheduler[3]);
    forward_publish_progress(a, kForwardReduceComplete,
                             a.dispatch.identity.num_tokens);
    if (a.injected_status_error_rank == static_cast<int>(a.rank))
      publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
  }
  __syncthreads();

  publish_production_terminal_status<true>(
      a, status, ProductionStatusPhase::Forward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset, 0ULL);
  __syncthreads();

  if (threadIdx.x == 0U) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    const auto mblocks = scheduler[3];
    const auto fc1_ntiles =
        (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
        forward_tuning::kFc1TileN;
    const auto fc2_ntiles =
        (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
        forward_tuning::kFc2TileN;
    const auto fc1_tasks = mblocks * fc1_ntiles;
    const auto fc2_tasks = mblocks * fc2_ntiles;
    const auto fc1_k_blocks =
        (a.hidden + forward_tuning::kTileK - 1ULL) /
        forward_tuning::kTileK;
    const auto fc2_k_blocks =
        (a.intermediate + forward_tuning::kTileK - 1ULL) /
        forward_tuning::kTileK;
    status->fc1_tasks_done = fc1_tasks;
    status->fc2_tasks_done = fc2_tasks;
    status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                        fc2_tasks * fc2_k_blocks * 2ULL;
    status->wgmma_instructions =
        fc1_tasks * 2ULL * fc1_k_blocks * 2ULL *
            (forward_tuning::kTileK / TunedForwardFc1Mma::K) +
        fc2_tasks * fc2_k_blocks * 2ULL *
            (forward_tuning::kTileK / TunedForwardFc2Mma::K);
#endif
    status->complete_clock = device_globaltimer_ns();
  }
  forward_publish_progress(a, kForwardTerminal, execution_epoch(a));
  WEFTMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL,
                          execution_epoch(a));
}

template <bool StaticLocal>
__device__ __forceinline__ void split_transport_finalize(
    const ForwardKernelArgs &a,
    persistent_v3::ForwardPipelineControl &control,
    transport_state::GenerationCompletionVector *completion,
    unsigned long long *scheduler, TransportDeviceStatus *status) {
  // Receive-completion CTA 0 publishes the rank-wide terminal epoch after all
  // Relay workers have finished the arrival queues. RDMA CTA 0 may still be
  // waiting for unrelated outbound put completion and must not lengthen the
  // per-token Reduce dependency.
  if (blockIdx.x == persistent_v3::kRdmaCtaCount) {
    if (threadIdx.x == 0U)
      wait_for_count(a, control.return_receive_roles_complete,
                     a.forward_relay_cta_count);
    __syncthreads();

    const auto remote_fanout =
        forward_return_detail::source_compact_rows(a, scheduler);
    if (threadIdx.x == 0U) {
      const auto arrivals = completion_task(
          completion, transport_state::TaskClass::SourceArrival).expected;
      if (arrivals != 0ULL) {
        claim_completion_tasks(completion,
                               transport_state::TaskClass::SourceArrival,
                               arrivals);
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::SourceArrival,
                                  arrivals);
      }
      atomicAdd(&status->actual_return_peer_bytes,
                remote_fanout * a.hidden * sizeof(transport::BFloat16));
    }

    // Every completion CTA performed a system fence before its release
    // increment. The acquire above followed by this fence orders received rows
    // before the rank-wide terminal readiness epochs.
    __threadfence_system();
    __syncthreads();
    for (std::uint32_t destination = threadIdx.x; destination < a.world;
         destination += blockDim.x) {
      if (destination / a.local_world_size == a.node)
        continue;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(peer_at<unsigned long long>(a.local_base,
                                            a.outgoing_sequence_offset)
                    [destination * a.channels]);
      ready.store(execution_epoch(a), ::cuda::memory_order_release);
    }
    __syncthreads();
    WEFTMOE_TIMELINE_RECORD(a.timeline, SourceSignalObserved, SourceRelay, 0U,
                            remote_fanout, execution_epoch(a));
    if (threadIdx.x == 0U)
      persistent_v3::store_release<::cuda::thread_scope_device>(
          control.source_ready_published.value, 1ULL);
    return;
  }

  // The other receive-completion CTAs and seven non-elected RDMA CTAs finish.
  if (blockIdx.x != 0U)
    return;

  // Outbound GIN completion does not gate Reduce, so wait for it only while
  // Compute consumes the just-published source epochs.  No grid-wide barrier
  // connects the independent transport and compute grids.
  if (threadIdx.x == 0U) {
    wait_for_count(a, control.rdma_return_roles_complete,
                   persistent_v3::kRdmaCtaCount);
    if (a.forward_dynamic_reduce_queue)
      wait_for_count(a, control.reduce_tokens_complete,
                     a.dispatch.identity.num_tokens);
    wait_for_count(a, control.compute_reduce_roles_complete,
                   a.forward_reduce_cta_count);
    wait_for_count(a, control.source_ready_published, 1ULL);
    forward_publish_progress(a, kForwardReturnObserved, scheduler[3]);
    forward_publish_progress(a, kForwardReduceComplete,
                             a.dispatch.identity.num_tokens);
    if (a.injected_status_error_rank == static_cast<int>(a.rank))
      publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
#if !WEFTMOE_DEBUG_VALIDATE
    // All compute roles can reach their release only after draining FC1/FC2
    // and their claimed Reduce batch.  Queue head==num_tokens establishes the
    // exact reduction task space, so production records these three already-
    // proven static spaces once instead of issuing four contended atomics per
    // task from the GEMM/reduction hot paths.
    mark_task_space_complete(
        completion_task(completion, transport_state::TaskClass::Fc1));
    mark_task_space_complete(
        completion_task(completion, transport_state::TaskClass::Fc2));
    mark_task_space_complete(
        completion_task(completion, transport_state::TaskClass::Reduce));
#endif
    completion->ring_credit_restored = completion->ring_credit_expected;
    publish_elected_terminal(completion, execution_epoch(a),
                             status->error != 0U);
  }
  __syncthreads();

  publish_production_terminal_status<StaticLocal>(
      a, status, ProductionStatusPhase::Forward,
      a.production_error_records_offset, a.gin_production_error_records_offset,
      0ULL);
  __syncthreads();

  if (threadIdx.x == 0U) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    const auto mblocks = scheduler[3];
    const auto fc1_ntiles =
        (a.intermediate + forward_tuning::kFc1TileN - 1ULL) /
        forward_tuning::kFc1TileN;
    const auto fc2_ntiles =
        (a.hidden + forward_tuning::kFc2TileN - 1ULL) /
        forward_tuning::kFc2TileN;
    const auto fc1_tasks = mblocks * fc1_ntiles;
    const auto fc2_tasks = mblocks * fc2_ntiles;
    const auto fc1_k_blocks =
        (a.hidden + forward_tuning::kTileK - 1ULL) /
        forward_tuning::kTileK;
    const auto fc2_k_blocks =
        (a.intermediate + forward_tuning::kTileK - 1ULL) /
        forward_tuning::kTileK;
    status->fc1_tasks_done = fc1_tasks;
    status->fc2_tasks_done = fc2_tasks;
    status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                        fc2_tasks * fc2_k_blocks * 2ULL;
    status->wgmma_instructions =
        fc1_tasks * 2ULL * fc1_k_blocks * 2ULL *
            (forward_tuning::kTileK / TunedForwardFc1Mma::K) +
        fc2_tasks * fc2_k_blocks * 2ULL *
            (forward_tuning::kTileK / TunedForwardFc2Mma::K);
#endif
    status->complete_clock = device_globaltimer_ns();
  }
  forward_publish_progress(a, kForwardTerminal, execution_epoch(a));
  WEFTMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL,
                          execution_epoch(a));
}

} // namespace forward_finalize_detail
