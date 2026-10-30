// Internal pointer-bearing launch contract for the cached-state Backward
// implementation. This header is included only by the CUDA aggregation TU.
struct BackwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint64_t epoch = 0;
  unsigned long long *runtime_epoch = nullptr;
  std::uint32_t scheduler_mblock_capacity = 0;
  std::uint32_t scheduler_expert_capacity = 0;
  std::uint32_t rank = 0;
  std::uint32_t world = 0;
  void *local_base = nullptr;
  void **peer_bases = nullptr;

  const float *saved_gates = nullptr;
  const transport::BFloat16 *saved_ab = nullptr;
  const transport::BFloat16 *saved_x = nullptr;
  bool fp8 = false;
  bool fp32_weight_gradients = false;
  // Tape-owned column cache, produced once in place by designated dU tiles.
  std::uint8_t *fp8_x_data = nullptr;
  float *fp8_x_scales = nullptr;
  // Aliases the cache only on its first BWD, while it still holds row FP8.
  const std::uint8_t *fp8_x_row_data = nullptr;
  const float *fp8_x_row_scales = nullptr;
  const std::uint8_t *fp8_ab_data = nullptr;
  const float *fp8_ab_scales = nullptr;
  bool fp8_du_ring = false;
  std::uint64_t fp8_activation_pitch = 0;
  std::uint8_t *fp8_wgrad_data = nullptr;
  float *fp8_wgrad_scales = nullptr;
  unsigned long long *fp8_wgrad_ready = nullptr;
  std::uint64_t fp8_wgrad_rows = 0;
  std::uint64_t saved_rows = 0;
  const backward_detail::PeerPayload *peer_payloads = nullptr;
  // Stable, IPC-visible dR owned by the execution slot. Logical TMA bounds
  // follow the actual route count independently of its cached capacity.
  transport::BFloat16 *dr = nullptr;
  transport::BFloat16 *recomputed_u = nullptr;
  transport::BFloat16 *dab = nullptr;
  transport::BFloat16 *dx_rows = nullptr;
  std::uint64_t activation_stride = 0;
  float *dgate_rows = nullptr;
  float *dgate = nullptr;
  transport::BFloat16 *dw1 = nullptr;
  transport::BFloat16 *dw2 = nullptr;
  std::size_t direct_headers_offset = 0;
  std::size_t compact_to_padded_offset = 0;
  std::size_t padded_expert_rows_offset = 0;
  std::size_t padded_expert_offsets_offset = 0;
  std::size_t setup_epoch_offset = 0;
  std::size_t scheduler_offset = 0;
  std::size_t status_offset = 0;
};

__device__ __forceinline__ std::uint64_t
execution_epoch(const BackwardKernelArgs &a) {
  return a.runtime_epoch ? *a.runtime_epoch : a.epoch;
}
