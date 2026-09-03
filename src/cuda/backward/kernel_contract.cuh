// Internal pointer-bearing launch contract for the cached-state Backward
// implementation. This header is included only by the CUDA aggregation TU.
struct BackwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *dy = nullptr;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint64_t max_destination_rows = 0;
  std::uint64_t padded_capacity_rows = 0;
  std::uint64_t padded_rows = 0;
  std::uint64_t epoch = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  void *local_base = nullptr;
  void **peer_bases = nullptr;
  std::size_t canonical_x_offset = 0;
  std::size_t canonical_gate_offset = 0;
  std::size_t cached_ab_offset = 0;
  std::size_t padded_x_offset = 0;
  std::size_t padded_ab_offset = 0;
  std::size_t padded_dr_offset = 0;
  std::size_t recomputed_u_offset = 0;
  std::size_t dab_offset = 0;
  std::size_t dx_rows_offset = 0;
  std::size_t dgate_rows_offset = 0;
  std::size_t dgate_output_offset = 0;
  std::size_t contribution_offset = 0;
  std::size_t combined_dx_offset = 0;
  std::size_t direct_headers_offset = 0;
  std::size_t compact_to_padded_offset = 0;
  std::size_t padded_expert_rows_offset = 0;
  std::size_t padded_expert_offsets_offset = 0;
  std::size_t setup_epoch_offset = 0;
  std::size_t dispatch_ready_offset = 0;
  std::size_t return_ready_offset = 0;
};
