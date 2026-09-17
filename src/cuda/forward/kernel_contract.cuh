// Internal, pointer-bearing launch contract. Never include from public API.
// Kept in this header so the aggregation translation unit preserves device
// inlining and code generation while the Forward boundary remains explicit.
struct ForwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  const transport::BFloat16 *w1;
  const transport::BFloat16 *w2;
  std::uint64_t hidden;
  std::uint64_t intermediate;
  std::uint64_t max_destination_rows;
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t channels;
  std::uint64_t epoch;
  unsigned long long *runtime_epoch = nullptr;
  std::size_t generation_setup_offset = 0;
  std::size_t completion_offset = 0;
  std::size_t context_completion_offset = 0;
  unsigned context_completion_capacity = 0;
  unsigned gin_allocated_context_count = 0;
  void *local_base;
  void **peer_bases;
#if __has_include(<nccl_device.h>)
  ncclDevComm_t gin_dev_comm;
  ncclWindow_t gin_window = nullptr;
#endif
  void *gin_window_base = nullptr;
  communication::ChunkRingLayout gin_rings;
  std::size_t local_notification_offset = 0;
  unsigned local_notification_ctas = 0;
  void *gin_sender_done = nullptr;
  unsigned gin_sender_count = 0;
  unsigned gin_receiver_base = 0;
  unsigned gin_receiver_count = 0;
  unsigned gin_total_ctas = 0;
  std::size_t forward_return_queues_offset = 0;
  std::size_t gin_send_header_offset = 0;
  std::size_t gin_receive_header_offset = 0;
  std::size_t gin_send_fanout_offset = 0;
  std::size_t gin_receive_fanout_offset = 0;
  std::size_t gin_send_payload_offset = 0;
  std::size_t gin_receive_payload_offset = 0;
  std::size_t gin_return_staging_offset = 0;
  std::size_t gin_return_receive_offset = 0;
  // Notification words carry compact row IDs and contiguous-copy hints.
  // A per-sender GIN strong signal publishes their arrival after the payloads;
  // Relay consumers claim ready rows without scanning compute tiles.
  std::size_t gin_return_send_rows_offset = 0;
  std::size_t gin_return_receive_rows_offset = 0;
  std::size_t forward_return_rows_offset = 0;
  std::size_t gin_production_error_records_offset = 0;
  std::uint32_t node = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t gin_context_count = 1;
  std::uint64_t max_tokens_per_rank = 0;
  bool use_gin = false;
  bool forward_compact_return = false;
  // A one-node split launch streams CUDA-IPC/NVLink rows through dedicated
  // Relay CTAs.  Unlike the legacy StaticLocal path, compute must consume the
  // same per-M-block ready queue used by GIN instead of assuming that the
  // complete Dispatch grid has already joined.
  bool forward_stream_local_dispatch = false;
  std::size_t canonical_rows_offset;
  std::size_t canonical_gates_offset;
  std::size_t forward_output_rows_offset;
  std::size_t direct_headers_offset;
  std::size_t contribution_offset;
  std::size_t return_ready_epoch_offset;
  transport::BFloat16 *output = nullptr;
  std::size_t ingress_published_offset;
  std::size_t outgoing_sequence_offset;
  std::size_t status_offset;
  std::size_t forward_s_offset;
  transport::BFloat16 *saved_ab = nullptr;
  std::uint64_t saved_rows = 0;
  std::size_t forward_task_claim_offset;
  std::size_t forward_fc1_done_offset;
  std::size_t forward_input_groups_offset;
  std::size_t forward_scheduler_offset;
  std::size_t forward_v3_control_offset;
  std::size_t gin_local_compute_ready_offset;
  std::size_t terminal_epoch_offset;
  std::size_t production_error_records_offset;
  const unsigned long long *cancel_epoch;
  int injected_status_error_rank = -1;
#if WEFTMOE_DEVICE_TIMELINE
  profiling::DeviceTimelineContext timeline;
#endif
#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
  unsigned long long *progress;
  std::uint32_t progress_contexts;
#endif
  DispatchWaveKernelView dispatch_waves;
  std::size_t gin_send_dispatch_directory_offset = 0;
  std::size_t gin_receive_dispatch_directory_offset = 0;
  std::size_t x_arrived_rows_offset = 0;
  // Token-ready reduction queue.  Every route publication increments the
  // owning source token's counter; the top-k completion appends the token.
  // Queue slots encode token+1; generation setup clears the active token
  // range before producers are released, avoiding a second epoch array.
  std::size_t forward_reduce_route_count_offset = 0;
  std::size_t forward_reduce_ready_token_offset = 0;
  // Forward Return has one cross-node transport policy: FULL GIN into the registered receive
  // plane, followed by per-route readiness publication.
  // Forward keeps one fixed owner per physical GIN context, while the Relay
  // team is runtime-tunable in whole context-sized groups.  Extra Relay CTAs
  // stripe node-local Dispatch and Return M-blocks; only the first group owns
  // receive directories and FULL-completion markers.
  std::uint32_t forward_relay_cta_count = 8;
  std::uint32_t forward_compute_cta_count = 0;
  std::uint32_t forward_reduce_cta_count = 0;
  // Split-GIN production normally delegates node-local Return copies to Relay
  // CTAs.  Keeping this explicit also gives the autotuner a controlled A/B
  // against the legacy compute-epilogue copy without changing wire semantics.
  bool forward_relay_local_return = false;
  // Hopper bulk async cannot copy global->global directly. Relay can instead
  // stage one contiguous hidden row through CTA shared memory and use a TMA
  // bulk store to the peer UVA destination. This is an explicit A/B knob until
  // the end-to-end tuner proves it better than the warp-vector path.
  std::uint32_t forward_relay_tma_warps = 0;
  // Batch only descriptors that are already published. This amortizes the
  // issuing warp's system fence and route-ready publication without waiting
  // for a future FC2 M-block at the tail.
  std::uint32_t forward_local_return_batch_rows = 512;
  // Publish Dispatch readiness after this many source packets. Relay keeps
  // copying the rest of the GIN wave while FC1 consumes an earlier microbatch.
  std::uint32_t forward_dispatch_publish_batch_rows = 32;
  // Preserve the first two M-block waves for startup latency, then publish
  // this many ready M-blocks from one expert in N-tile-major order. Adjacent
  // compute CTAs can consequently reuse the same W1 tile through L2 without
  // delaying Dispatch itself or requiring a cluster-wide rendezvous.
  std::uint32_t forward_fc1_locality_group = 1;
  // When locality_group is two, this half-open window uses groups of four.
  std::uint8_t forward_fc1_locality_group4_begin = 4;
  std::uint8_t forward_fc1_locality_group4_end = 12;
  bool retain_backward_state = false;
  // Production ready-token scheduler: reduction consumes only tokens whose
  // complete top-k route set has arrived instead of waiting on a fixed token
  // stripe.  The host enables this by default; the false initializer keeps
  // manually constructed test contracts conservative.
  bool forward_dynamic_reduce_queue = false;
  // Producer-light alternative: retain per-route release epochs and let
  // source-local reducers non-blockingly scan/claim complete tokens.  This
  // removes cross-peer counter RMWs from FC2 while still avoiding head-of-line
  // waits on incomplete fixed stripes.
  bool forward_dynamic_reduce_scan = false;
};

__device__ __forceinline__ std::uint64_t
execution_epoch(const ForwardKernelArgs &a) {
  return a.runtime_epoch ? *a.runtime_epoch : a.epoch;
}

// Final Return storage is dense in (node, local rank, expert, source ordinal).
// Readiness and accumulation order retain their original route identities.
__device__ __forceinline__ std::uint64_t
forward_contribution_row(const ForwardKernelArgs &a, std::uint64_t route) {
  return a.forward_compact_return
      ? a.dispatch.return_global_expert_base[a.dispatch.expert_id[route]] +
            a.dispatch.local_offset[route]
      : route;
}
