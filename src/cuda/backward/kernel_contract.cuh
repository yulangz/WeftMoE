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
  bool use_forward_cache;
  int injected_status_error_rank = -1;
#if MEGAMOE_DEVICE_TIMELINE
  profiling::DeviceTimelineContext timeline;
#endif
#if MEGAMOE_DEBUG_VALIDATE
  std::size_t debug_dv_offset;
  std::size_t debug_coverage_offset;
  bool capture_stage_storage;
#endif
  DispatchWaveKernelView dispatch_waves;
  std::size_t gin_send_dispatch_directory_offset = 0;
  std::size_t gin_receive_dispatch_directory_offset = 0;
  std::size_t dy_arrived_rows_offset = 0;
};
