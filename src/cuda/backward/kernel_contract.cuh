// Internal pointer-bearing launch contract for the cached-state Backward
// implementation. This header is included only by the CUDA aggregation TU.
struct BackwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  DispatchWaveKernelView dispatch_waves;
  const transport::BFloat16 *dy = nullptr;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint64_t max_destination_rows = 0;
  std::uint64_t padded_capacity_rows = 0;
  std::uint64_t padded_rows = 0;
  std::uint64_t scheduler_words = 0;
  std::uint64_t epoch = 0;
  unsigned long long *runtime_epoch = nullptr;
  std::uint32_t scheduler_mblock_capacity = 0;
  std::uint32_t scheduler_expert_capacity = 0;
  std::uint32_t mblocks = 0;
  std::uint32_t active_experts = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  std::uint32_t node = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t gin_context_count = 0;
  std::uint64_t max_tokens_per_rank = 0;
  std::uint64_t backward_return_chunk_rows = 0;
  bool use_gin = false;
  void *local_base = nullptr;
  void **peer_bases = nullptr;
#if __has_include(<nccl_device.h>)
  ncclDevComm_t gin_dev_comm{};
  ncclWindow_t gin_window = nullptr;
#endif
  void *gin_window_base = nullptr;
  communication::ChunkRingLayout gin_rings;
  std::size_t local_notification_offset = 0;
  unsigned local_notification_ctas = 0;
  void *forward_unified_gin_state = nullptr;
  std::size_t forward_return_queues_offset = 0;
  std::size_t forward_return_rows_offset = 0;

  const float *saved_gates = nullptr;
  const transport::BFloat16 *saved_ab = nullptr;
  std::uint64_t saved_rows = 0;
  std::size_t compact_dr_offset = 0;
  std::size_t recomputed_u_offset = 0;
  std::size_t dab_offset = 0;
  std::size_t dx_rows_offset = 0;
  std::size_t dgate_rows_offset = 0;
  std::size_t dgate_output_offset = 0;
  transport::BFloat16 *dw1 = nullptr;
  transport::BFloat16 *dw2 = nullptr;
  std::size_t contribution_offset = 0;
  transport::BFloat16 *dx = nullptr;
  std::size_t direct_headers_offset = 0;
  std::size_t compact_to_padded_offset = 0;
  std::size_t padded_expert_rows_offset = 0;
  std::size_t padded_expert_offsets_offset = 0;
  std::size_t setup_epoch_offset = 0;
  std::size_t dispatch_ready_offset = 0;
  std::size_t return_ready_offset = 0;
  std::size_t scheduler_offset = 0;
  std::size_t status_offset = 0;
};

__device__ __forceinline__ std::uint64_t
execution_epoch(const BackwardKernelArgs &a) {
  return a.runtime_epoch ? *a.runtime_epoch : a.epoch;
}
