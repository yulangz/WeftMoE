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
  std::size_t gin_send_payload_offset = 0;
  std::size_t gin_receive_payload_offset = 0;
  std::size_t gin_return_staging_offset = 0;
  std::size_t gin_return_receive_offset = 0;
  std::size_t gin_production_error_records_offset = 0;
  std::uint32_t node = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t gin_context_count = 1;
  std::uint64_t max_tokens_per_rank = 0;
  bool use_gin = false;
  bool use_socket = false;
  std::size_t canonical_rows_offset;
  std::size_t forward_output_rows_offset;
  std::size_t direct_headers_offset;
  std::size_t contribution_offset;
  std::size_t return_ready_epoch_offset;
  std::size_t combined_offset;
  std::size_t ingress_published_offset;
  std::size_t outgoing_sequence_offset;
  std::size_t status_offset;
  std::size_t forward_s_offset;
  std::size_t forward_ab_cache_offset;
  std::size_t forward_task_claim_offset;
  std::size_t forward_fc1_done_offset;
  std::size_t forward_scheduler_offset;
  std::size_t gin_local_compute_ready_offset;
  std::size_t terminal_epoch_offset;
  std::size_t production_error_records_offset;
  const unsigned long long *cancel_epoch;
  int injected_status_error_rank = -1;
#if MEGAMOE_DEBUG_VALIDATE
  unsigned long long *progress;
#endif
  bool cache_fc1;
};
