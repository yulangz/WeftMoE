// Internal, pointer-bearing launch contract. Never include from public API.
// Kept in this header so the aggregation translation unit preserves device
// inlining and code generation while the Backward boundary remains explicit.
struct BackwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *dy_input;
  const float *gates;
  const transport::BFloat16 *w1;
  const transport::BFloat16 *w2;
  std::uint64_t hidden;
  std::uint64_t intermediate;
  std::uint64_t max_destination_rows;
  std::uint64_t reduction_rows;
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t channels;
  std::uint64_t epoch;
  void *local_base;
  void **peer_bases;
#if __has_include(<nccl_device.h>)
  ncclDevComm_t gin_dev_comm;
  ncclWindow_t gin_window = nullptr;
#endif
  void *gin_window_base = nullptr;
  std::size_t gin_send_header_offset = 0;
  std::size_t gin_receive_header_offset = 0;
  std::size_t gin_send_fanout_offset = 0;
  std::size_t gin_receive_fanout_offset = 0;
  std::size_t gin_send_gate_offset = 0;
  std::size_t gin_receive_gate_offset = 0;
  std::size_t gin_send_payload_offset = 0;
  std::size_t gin_receive_payload_offset = 0;
  std::size_t gin_backward_dy_input_offset = 0;
  std::size_t gin_return_staging_offset = 0;
  std::size_t gin_return_receive_offset = 0;
  std::size_t gin_return_gate_staging_offset = 0;
  std::size_t gin_return_gate_receive_offset = 0;
  std::size_t gin_production_error_records_offset = 0;
  std::uint32_t node = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t gin_context_count = 1;
  std::uint64_t max_tokens_per_rank = 0;
  bool use_gin = false;
  bool use_socket = false;
  std::size_t canonical_x_offset;
  std::size_t direct_headers_offset;
  std::size_t dy_canonical_offset;
  std::size_t canonical_gate_offset;
  std::size_t dab_offset;
  std::size_t recompute_ab_offset;
  std::size_t backward_s_offset;
  std::size_t forward_ab_cache_offset;
  std::size_t dx_canonical_offset;
  std::size_t dx_return_offset;
  std::size_t dx_combined_offset;
  std::size_t dgate_partial_offset;
  std::size_t dgate_canonical_offset;
  std::size_t dgate_offset;
  std::size_t u_transpose_offset;
  std::size_t dab_transpose_offset;
  std::size_t dw1_offset;
  std::size_t dw2_offset;
  std::size_t scheduler_offset;
  std::size_t dw_timeline_offset;
  std::size_t claim_offset;
  std::size_t incoming_signal_offset;
  std::size_t outgoing_signal_offset;
  std::size_t gin_local_compute_ready_offset;
  std::size_t terminal_epoch_offset;
  std::size_t production_error_records_offset;
  std::size_t status_offset;
  const unsigned long long *cancel_epoch;
  unsigned long long *progress;
  std::uint32_t progress_experts;
#if MEGAMOE_DEBUG_VALIDATE
  std::uint32_t progress_contexts;
#endif
  bool use_forward_cache;
  int injected_status_error_rank = -1;
#if MEGAMOE_DEVICE_TIMELINE
  profiling::DeviceTimelineContext timeline;
#endif
#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_BUILD_TESTING
  std::size_t debug_dv_offset;
  std::size_t debug_coverage_offset;
  bool capture_stage_storage;
#endif
  DispatchWaveKernelView dispatch_waves;
  std::size_t gin_send_dispatch_directory_offset = 0;
  std::size_t gin_receive_dispatch_directory_offset = 0;
  std::size_t dy_arrived_rows_offset = 0;
  std::size_t v3_local_dy_offset = 0;
  std::size_t v3_control_offset = 0;
  std::size_t v3_activation_pair_mailboxes_offset = 0;
  std::size_t v3_mblock_mapping_offset = 0;
  std::size_t v3_mblock_dv_cursor_offset = 0;
  std::size_t v3_mblock_dv_complete_offset = 0;
  std::size_t v3_mblock_ready_offset = 0;
  std::size_t v3_mblock_dx_cursor_offset = 0;
  std::size_t v3_mblock_dx_complete_offset = 0;
  std::size_t v3_token_state_offset = 0;
  std::size_t v3_expert_padded_prefix_offset = 0;
  std::size_t v3_expert_dv_complete_offset = 0;
  std::size_t v3_expert_dw1_cursor_offset = 0;
  std::size_t v3_expert_dw2_cursor_offset = 0;
  std::size_t v3_sigmoid_lut_offset = 0;
  // Forward and Backward use the same immutable reverse-route catalog.  Keep
  // the compact row identity alive across the pair so Backward Return never
  // has to fall back to the sparse sequence*top_k wire envelope.
  std::size_t forward_return_chunk_directory_offset = 0;
  std::size_t forward_return_chunk_state_offset = 0;
  std::size_t forward_return_chunk_ready_count_offset = 0;
  std::size_t forward_return_fragment_index_offset = 0;
  std::size_t forward_return_source_compact_index_offset = 0;
  // Forward's compact->route directory is dead once its output has been
  // consumed. Backward reuses that allocation as route->compact so Reduce can
  // consume put-delivered dX directly from the local GIN receive window.
  std::size_t v3_return_route_compact_index_offset = 0;
  std::size_t forward_output_ready_epoch_offset = 0;
  std::size_t forward_scheduler_offset = 0;
  std::uint64_t forward_return_chunk_rows = 128;
  // Sixteen 2048-row groups at the main shape give each of the eight fixed
  // RDMA CTAs two large puts.  This was measurably better than 1536 rows:
  // fewer context turnovers without collapsing the return pipeline to one
  // oversized transfer per context.
  std::uint64_t v3_return_group_rows = 2048;
  // A soft scheduling preference only: every Compute CTA can still steal
  // every task class. One out of this many CTAs probes dW first while the
  // others prioritize the return-critical dV -> dX chain.
  std::uint32_t v3_weight_cta_divisor = 5;
  // Activation CTAs cycle through this many dV- and dX-preferred slots.
  // Readiness fallback remains bidirectional, so these values tune priority
  // without partitioning the work queues or introducing a phase barrier.
  std::uint32_t v3_dv_phase_slots = 1;
  std::uint32_t v3_dx_phase_slots = 2;
  // Production fixed-role dW kernels use this compile-time-supported N tile.
  // The setup CTA reads the value only to size immutable task domains/audits.
  std::uint32_t v3_dw_tile_n = 256;
  // Experimental direct producer path: dX epilogues store remote routes into
  // the same-node proxy's NCCL-registered Return window through its LSA
  // mapping, leaving Relay CTAs to publish already-contiguous payloads.
  bool v3_direct_return_staging = false;
  // Keep the network put-only while skipping the destination-side compact to
  // route-major payload copy. Relay publishes dGate/token readiness; Reduce
  // follows the setup-time route->compact map for remote dX operands.
  bool v3_direct_return_consume = false;
  // The producer supplied token-major dY in an NCCL-registered plane.  The
  // fixed RDMA CTAs can put contiguous token slices without Relay repacking.
  bool v3_direct_dy_input = false;
  // Diagnostic/compatibility path for peer-visible canonical dY publication.
  // Hopper bulk stores into an IPC peer mapping do not necessarily become
  // observable by a destination-side TMA load at the point where the source
  // thread observes bulk-group completion.  Generic vector stores plus a
  // system release provide the same fine-grained M-block contract without a
  // grid-wide rendezvous.
  bool v3_generic_dy_copy = false;
  // Strict diagnostic split: hold dV TMA issue until every Relay role has
  // completed dY delivery.  Production must leave this disabled; it exists
  // only to distinguish publication timing from activation-core corruption.
  bool v3_wait_full_dy_dispatch = false;
  // Local visibility bridge for peer-published canonical dY.  The first dV
  // task for an M-block cooperatively copies that block into the local dX
  // plane; dV TMA reads the local copy, and dX overwrites it only after every
  // dV N tile for the block has completed.
  bool v3_localize_dy = false;
};
