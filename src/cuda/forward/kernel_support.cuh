#include "cuda/weftmoe_pf/network_layout.hpp"

constexpr std::size_t kAlign = 256;

void CUDART_CB publish_stream_completion(void *opaque) {
  static_cast<std::atomic<bool> *>(opaque)->store(true,
                                                  std::memory_order_release);
}

[[noreturn]] void transport_error(NotifyErrorCode code, const char *phase,
                                  const char *field, std::uint32_t rank,
                                  const std::string &expected,
                                  const std::string &actual) {
  throw NotifyDispatchError(
      code, std::string("phase=") + phase + ", field=" + field +
                ", rank=" + std::to_string(rank) + ", expected=" + expected +
                ", actual=" + actual);
}

void throw_cublas(const char *phase, cublasStatus_t status, std::uint32_t rank,
                  const char *operation) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;
  transport_error(NotifyErrorCode::CudaFailure, phase, operation, rank,
                  "CUBLAS_STATUS_SUCCESS",
                  std::to_string(static_cast<unsigned int>(status)));
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b, const char *field) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "uint64 product", "overflow");
  return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, const char *field) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "uint64 sum", "overflow");
  return a + b;
}

std::size_t checked_size(std::uint64_t value, const char *field) {
  if (value > std::numeric_limits<std::size_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "value <= size_t max", std::to_string(value));
  return static_cast<std::size_t>(value);
}

void validate_external_cuda_range(const void *pointer,
                                  std::uint64_t required_elements,
                                  std::uint64_t capacity_elements,
                                  std::size_t element_bytes,
                                  std::uint32_t expected_device,
                                  const char *field) {
  if (required_elements > capacity_elements)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "external-cuda-producer", field, 0,
                    "capacity >= " + std::to_string(required_elements),
                    std::to_string(capacity_elements));
  const auto required_bytes =
      checked_size(checked_mul(required_elements, element_bytes, field), field);
  const auto capacity_bytes =
      checked_size(checked_mul(capacity_elements, element_bytes, field), field);
  if (required_bytes == 0)
    return;
  if (!pointer)
    transport_error(NotifyErrorCode::InvalidHandle, "external-cuda-producer",
                    field, 0, "non-null device allocation", "null");
  cudaPointerAttributes attributes{};
  internal::throw_cuda("external-cuda-producer",
                       cudaPointerGetAttributes(&attributes, pointer), 0, 1,
                       "external", 1, ExchangeKind::ExpertCounts,
                       "cudaPointerGetAttributes");
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != static_cast<int>(expected_device))
    transport_error(
        NotifyErrorCode::WorkspaceMismatch, "external-cuda-producer", field, 0,
        "device allocation on ordinal=" + std::to_string(expected_device),
        "wrong memory/device");
  CUdeviceptr base = 0;
  std::size_t allocation_bytes = 0;
  if (cuMemGetAddressRange(&base, &allocation_bytes,
                           reinterpret_cast<CUdeviceptr>(pointer)) !=
      CUDA_SUCCESS)
    transport_error(NotifyErrorCode::CudaFailure, "external-cuda-producer",
                    field, 0, "queryable CUDA allocation",
                    "cuMemGetAddressRange failure");
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto allocation_begin = static_cast<std::uintptr_t>(base);
  if (begin < allocation_begin || begin - allocation_begin > allocation_bytes ||
      capacity_bytes > allocation_bytes - (begin - allocation_begin))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "external-cuda-producer", field, 0,
                    "declared capacity within allocation", "short range");
}

std::size_t append(std::size_t &cursor, std::uint64_t count, std::size_t width,
                   const char *field) {
  if (cursor > std::numeric_limits<std::size_t>::max() - (kAlign - 1U))
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "alignable offset", "overflow");
  const auto aligned = (cursor + kAlign - 1U) & ~(kAlign - 1U);
  const auto bytes = checked_size(checked_mul(count, width, field), field);
  if (bytes > std::numeric_limits<std::size_t>::max() - aligned)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "bounded arena", "overflow");
  cursor = aligned + bytes;
  return aligned;
}

CUtensorMap make_bf16_tma_map(void *base, std::uint64_t inner,
                              std::uint64_t outer, std::uint32_t box_outer,
                              const char *field,
                              std::uint32_t box_inner = 64U) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX || box_outer == 0 || box_inner == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-tma-map",
                    field, 0, "nonzero uint32 dimensions and valid base",
                    "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t element_strides[2] = {1U, 1U};
  const auto result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    (void)cuGetErrorString(result, &text);
    transport_error(NotifyErrorCode::CudaFailure, "forward-tma-map", field, 0,
                    "cuTensorMapEncodeTiled success",
                    text ? text : "unknown CUDA driver error");
  }
  return map;
}

CUtensorMap make_bf16_tma_store_map(void *base, std::uint64_t inner,
                                    std::uint64_t outer, const char *field,
                                    std::uint64_t row_stride_bytes = 0ULL,
                                    std::uint32_t box_rows = 128U) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX ||
      (box_rows != 32U && box_rows != 64U && box_rows != 128U))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-tma-store-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const auto dense_row_bytes =
      checked_mul(inner, sizeof(transport::BFloat16), field);
  if (row_stride_bytes == 0ULL)
    row_stride_bytes = dense_row_bytes;
  if (row_stride_bytes < dense_row_bytes || (row_stride_bytes & 15ULL) != 0ULL)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-tma-store-map", field, 0,
                    "16-byte-aligned stride covering one row",
                    std::to_string(row_stride_bytes));
  const cuuint64_t strides[1] = {row_stride_bytes};
  const cuuint32_t box[2] = {64U, box_rows};
  const cuuint32_t element_strides[2] = {1U, 1U};
  const auto result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    (void)cuGetErrorString(result, &text);
    transport_error(NotifyErrorCode::CudaFailure, "forward-tma-store-map",
                    field, 0, "cuTensorMapEncodeTiled success",
                    text ? text : "unknown CUDA driver error");
  }
  return map;
}

CUtensorMap make_bf16_tma_map_mn(void *base, std::uint64_t inner,
                                 std::uint64_t outer, const char *field,
                                 std::uint32_t box_rows = 64U) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-tma-map",
                    field, 0, "nonzero uint32 dimensions and valid base",
                    "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
  // MN-major WGMMA consumes a row-major [K,N] operand as 64 contiguous N
  // columns by one K slice. Wider N tiles assemble adjacent atoms.
  const cuuint32_t box[2] = {64U, box_rows};
  const cuuint32_t element_strides[2] = {1U, 1U};
  const auto result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    (void)cuGetErrorString(result, &text);
    transport_error(NotifyErrorCode::CudaFailure, "backward-tma-map", field, 0,
                    "cuTensorMapEncodeTiled success",
                    text ? text : "unknown CUDA driver error");
  }
  return map;
}

struct TransportDeviceStatus {
  unsigned long long packet_claim_cursor;
  unsigned long long packets_done;
  unsigned long long fanout_done;
  unsigned long long synthetic_done;
  unsigned long long combine_done;
  unsigned long long incoming_done;
  unsigned long long token_claim_cursor;
  unsigned long long reduced_tokens;
  unsigned long long credit_stalls;
  unsigned long long peak_ring_slots;
  unsigned long long dispatch_put_ticks;
  unsigned long long relay_ticks;
  unsigned long long synthetic_ticks;
  unsigned long long raw_cross_node_payload_bytes;
  unsigned long long deduplicated_cross_node_payload_bytes;
  unsigned long long actual_dispatch_peer_bytes;
  unsigned long long actual_return_peer_bytes;
  unsigned long long gin_dispatch_put_count;
  unsigned long long gin_return_put_count;
  unsigned long long gin_signal_count;
  unsigned long long gin_metadata_bytes;
  unsigned long long gin_return_payload_bytes;
  unsigned long long start_clock;
  unsigned long long route_exchange_start_clock;
  unsigned long long route_exchange_complete_clock;
  unsigned long long before_combine_start_clock;
  unsigned long long before_combine_complete_clock;
  unsigned long long after_combine_start_clock;
  unsigned long long after_combine_complete_clock;
  unsigned long long dispatch_complete_clock;
  unsigned long long complete_clock;
  unsigned long long dispatch_blocks_done;
  unsigned long long relay_blocks_done;
  unsigned long long reduce_blocks_done;
  unsigned long long fc1_tasks_done;
  unsigned long long fc2_tasks_done;
  unsigned long long fc1_start_clock;
  unsigned long long fc1_complete_clock;
  unsigned long long fc2_complete_clock;
  unsigned long long return_complete_clock;
  unsigned long long backward_dw2_start_clock;
  unsigned long long backward_dw2_complete_clock;
  unsigned long long backward_dw1_start_clock;
  unsigned long long backward_dw1_complete_clock;
  unsigned long long backward_reduce_start_clock;
  unsigned long long backward_reduce_complete_clock;
  unsigned long long backward_sender_complete_clock;
  unsigned long long backward_transport_complete_clock;
  unsigned long long backward_return_send_complete_clock;
  unsigned long long backward_return_receive_complete_clock;
  unsigned long long wgmma_instructions;
  unsigned long long tma_loads;
  unsigned long long tma_multicast_loads;
  unsigned long long cluster_rank_mask;
  unsigned int runtime_cluster_size;
  unsigned int error;
  unsigned int error_packet;
  unsigned int error_field;
  unsigned int error_origin_rank;
  unsigned int error_code;
  unsigned long long error_expected;
  unsigned long long error_actual;
};

// Fixed, pointer-free Forward terminal record. `epoch` is published last with
// system release ordering.
// The host wire exchange serializes these eight uint64 words verbatim through
// NCCL's typed uint64 AllGather; no native struct bytes cross the wire.
struct ProductionStatusRecord {
  unsigned long long generation;
  unsigned long long phase;
  unsigned long long origin_rank;
  unsigned long long error_code;
  unsigned long long field;
  unsigned long long expected;
  unsigned long long actual;
  unsigned long long epoch;
};
static_assert(sizeof(ProductionStatusRecord) == 8U * sizeof(std::uint64_t));

enum class ProductionStatusPhase : std::uint64_t {
  Forward = 1,
};

struct LocalP2pRouteHeader {
  std::uint32_t schema_version;
  std::uint32_t source_rank;
  std::uint32_t destination_rank;
  std::uint32_t channel;
  std::uint64_t generation;
  std::uint64_t source_epoch;
  // Packet sequence in legacy transports; final contribution row in the
  // unified node-local Forward path. Cross-node Return carries route IDs.
  std::uint64_t transport_index;
  std::uint64_t route_index;
  std::uint64_t local_token_id;
  std::uint64_t dst_row;
  std::uint32_t expert_id;
  std::uint32_t topk_slot;
};
static_assert(sizeof(LocalP2pRouteHeader) == 72);

// Socket is a correctness fallback and carries one route per packet, so it can
// append the gate to its header without changing the hot local/GIN record.
struct SocketRouteHeader {
  LocalP2pRouteHeader route;
  float gate;
  std::uint32_t reserved;
};
static_assert(sizeof(SocketRouteHeader) == 80);

// Forward's cross-node fanout wire record carries the route gate without
// growing the existing 32-byte metadata item or adding a fifth GIN put.  The
// production contract has far fewer than 65536 local ranks and experts; the
// canonical row and compact Return identity retain their full 64-bit range.
struct alignas(16) ForwardDispatchFanoutRecord {
  std::uint64_t dst_row;
  std::uint64_t return_compact_row;
  std::uint32_t topk_slot;
  std::uint16_t destination_local_rank;
  std::uint16_t expert_id;
  std::uint32_t mtile;
  float gate;
};
static_assert(sizeof(ForwardDispatchFanoutRecord) == 32);

// One directory entry is exchanged before a context starts publishing its
// M-tile waves.  It lets the peer consume context-local wire ranges without
// assuming that independently progressing GIN contexts complete in lockstep.
struct GinDispatchContextRecord {
  std::uint64_t generation;
  std::uint64_t epoch;
  std::uint64_t packet_begin;
  std::uint32_t packet_count;
  std::uint32_t wave_count;
  // Total route-level fanout rows for this source rank and destination node.
  // Every context carries the same total, allowing its permanent RDMA owner
  // to consume FULL Return descriptors independently during Dispatch.
  std::uint64_t return_row_count;
};
static_assert(sizeof(GinDispatchContextRecord) == 40);

enum ForwardProgressWord : std::uint32_t {
  kForwardEntered = 0,
  kForwardSetupComplete = 1,
  kForwardDispatchStored = 2,
  kForwardDispatchObserved = 3,
  kForwardFc1Complete = 4,
  kForwardFc2Complete = 5,
  kForwardReturnObserved = 6,
  kForwardReduceComplete = 7,
  kForwardTerminal = 8,
  kForwardProgressWords = 16,
};

#if WEFTMOE_DEBUG_VALIDATE || WEFTMOE_FORWARD_LIGHT_PROGRESS
void dump_generation_completion(std::ostream &output,
                                const std::uint64_t *mapped,
                                std::uint32_t contexts, const char *prefix) {
  const auto word = [&](std::size_t index) {
    return __atomic_load_n(mapped + index, __ATOMIC_ACQUIRE);
  };
  output << ' ' << prefix << ".generation=" << word(0) << ' ' << prefix
         << ".epoch=" << word(1);
  static constexpr const char *kTaskNames[] = {
      "dispatch", "relay",  "fc1",       "fc2",      "return",
      "arrival",  "reduce", "reserved0", "reserved1"};
  for (std::size_t task = 0; task < transport_state::kTaskClassCount; ++task) {
    const auto base = 2ULL + task * 4ULL;
    output << ' ' << prefix << '.' << kTaskNames[task]
           << "={expected=" << word(base) << ",claimed=" << word(base + 1ULL)
           << ",completed=" << word(base + 2ULL)
           << ",inflight=" << word(base + 3ULL) << '}';
  }
  constexpr auto kTail = 2ULL + transport_state::kTaskClassCount * 4ULL;
  output << ' ' << prefix << ".dispatch_local={expected=" << word(kTail)
         << ",completed=" << word(kTail + 1ULL) << "} " << prefix
         << ".return_local={expected=" << word(kTail + 2ULL)
         << ",completed=" << word(kTail + 3ULL) << "} " << prefix
         << ".credit={expected=" << word(kTail + 4ULL)
         << ",restored=" << word(kTail + 5ULL) << "} " << prefix
         << ".directory={expected=" << word(kTail + 6ULL)
         << ",ready=" << word(kTail + 7ULL) << "} " << prefix
         << ".bulk_return_barrier={progress=" << word(kTail + 8ULL)
         << ",compute=" << word(kTail + 9ULL) << ",grid=" << word(kTail + 10ULL)
         << "} " << prefix
         << ".bulk_return_stage={local_ready=" << word(kTail + 11ULL)
         << ",pack_complete=" << word(kTail + 12ULL)
         << ",contexts_issuing=" << word(kTail + 13ULL) << "} " << prefix
         << ".fc_scheduler={fc1=" << word(kTail + 14ULL)
         << ",fc2=" << word(kTail + 15ULL) << "} " << prefix
         << ".published_rows={local=" << word(kTail + 16ULL)
         << ",remote=" << word(kTail + 17ULL)
         << ",tile_mask=" << word(kTail + 18ULL) << "} " << prefix
         << ".tile_wait={tile=" << word(kTail + 19ULL)
         << ",expected=" << word(kTail + 20ULL)
         << ",arrived=" << word(kTail + 21ULL)
         << ",waiters=" << word(kTail + 22ULL) << '}';
#if WEFTMOE_DEBUG_VALIDATE
  output << ' ' << prefix << ".tile_wait_rails={";
  for (std::uint32_t rail = 0; rail < 8U; ++rail) {
    if (rail != 0U)
      output << ',';
    output << word(kTail + 23ULL + rail);
  }
  output << "} " << prefix << ".fc2_phase_counts={";
  for (std::uint32_t phase = 0; phase < 8U; ++phase) {
    if (phase != 0U)
      output << ',';
    output << word(kTail + 31ULL + phase);
  }
  output << "} " << prefix << ".fc2_phase_last_task={";
  for (std::uint32_t phase = 0; phase < 8U; ++phase) {
    if (phase != 0U)
      output << ',';
    output << word(kTail + 39ULL + phase);
  }
  output << "} " << prefix << ".fc2_stalled_ctas={";
  constexpr auto kCtaStarted = kTail + 47ULL;
  constexpr auto kCtaCompleted =
      kCtaStarted + transport_state::kForwardDebugCtas;
  constexpr auto kCtaLastTask =
      kCtaCompleted + transport_state::kForwardDebugCtas;
  constexpr auto kCtaPipeline =
      kCtaLastTask + transport_state::kForwardDebugCtas;
  constexpr auto kCtaProducer =
      kCtaPipeline + transport_state::kForwardDebugCtas;
  constexpr auto kCtaMath0 = kCtaProducer + transport_state::kForwardDebugCtas;
  constexpr auto kCtaMath1 = kCtaMath0 + transport_state::kForwardDebugCtas;
  bool first_stalled_cta = true;
  for (std::size_t cta = 0; cta < transport_state::kForwardDebugCtas; ++cta) {
    const auto started = word(kCtaStarted + cta);
    const auto completed = word(kCtaCompleted + cta);
    if (started == completed)
      continue;
    if (!first_stalled_cta)
      output << ',';
    first_stalled_cta = false;
    output << cta << ":" << started << '/' << completed << "@task"
           << word(kCtaLastTask + cta) << ":pipeline"
           << word(kCtaPipeline + cta) << ":producer"
           << word(kCtaProducer + cta) << ":math0" << word(kCtaMath0 + cta)
           << ":math1" << word(kCtaMath1 + cta);
  }
  constexpr auto kKernelComplete =
      kCtaMath1 + transport_state::kForwardDebugCtas;
  output << "} " << prefix << ".kernel_complete_epoch=" << word(kKernelComplete)
         << ' ' << prefix
         << ".terminal_publish_count=" << word(kKernelComplete + 1ULL);
#else
  output << ' ' << prefix << ".kernel_complete_epoch=" << word(kTail + 23ULL)
         << ' ' << prefix << ".terminal_publish_count=" << word(kTail + 24ULL);
#endif
  for (std::uint32_t context = 0; context < contexts; ++context) {
    const auto base = transport_state::kGenerationCompletionWords +
                      context * transport_state::kContextCompletionWords;
    output << ' ' << prefix << ".context[" << context
           << "]={directory=" << word(base + 2ULL)
           << ",dispatch=" << word(base + 3ULL) << '/' << word(base + 5ULL)
           << ",relay=" << word(base + 6ULL) << '/' << word(base + 8ULL)
           << ",return=" << word(base + 9ULL) << '/' << word(base + 11ULL)
           << ",inflight=" << word(base + 12ULL)
           << ",local_completion=" << word(base + 13ULL)
           << ",credit=" << word(base + 14ULL)
           << ",cursor=" << word(base + 15ULL)
           << ",last_progress=" << word(base + 16ULL)
           << ",dispatch_rows=" << word(base + 17ULL)
           << ",dispatch_tile0=" << word(base + 18ULL)
           << ",relay_rows=" << word(base + 19ULL)
           << ",relay_tile0=" << word(base + 20ULL) << '}';
  }
}
#endif

struct Layout {
  static_forward::ComputeRingLayout static_ring;
  std::size_t execution_epoch = 0;
  std::size_t input = 0;
  std::size_t ring_headers = 0;
  std::size_t ring_payload = 0;
  std::size_t ring_ready_sequence = 0;
  std::size_t link_next_sequence = 0;
  std::size_t link_credits = 0;
  std::size_t ingress_headers = 0;
  std::size_t ingress_ready_sequence = 0;
  std::size_t ingress_next_sequence = 0;
  std::size_t ingress_published_sequence = 0;
  std::size_t ingress_credits = 0;
  std::size_t outgoing_route_sequence = 0;
  std::size_t direct_route_headers = 0;
  std::size_t return_ready_epoch = 0;
  std::size_t peer_bases = 0;
  std::size_t p2p_barrier_send = 0;
  std::size_t p2p_barrier_receive = 0;
  std::size_t gin_local_compute_ready = 0;
  std::size_t terminal_epoch = 0;
  std::size_t production_error_records = 0;
  std::size_t production_status_matrix = 0;
  std::size_t packet_claim = 0;
  std::size_t canonical_rows = 0;
  // Route-local FP32 gates in the same canonical expert-row order as X.
  // Dispatch publishes each entry before the corresponding X-ready counter;
  // FC1 consumes it while materializing the gated FC2 input U = g*S.
  std::size_t canonical_gates = 0;
  std::size_t contribution = 0;
  std::size_t combine_return = 0;
  std::size_t combined = 0;
  std::size_t socket_count_matrix = 0;
  std::size_t socket_header_send = 0;
  std::size_t socket_header_receive = 0;
  std::size_t socket_payload_receive = 0;
  std::size_t forward_s = 0;
  // Training shares the canonical FC2 plane after FC1 checkpoints X. Saved
  // X/A/B and gradient outputs live outside this arena.
  std::size_t forward_training_output = 0;
  // Backward reads caller-owned X/A/B. Its scratch directory pads every expert
  // independently to 128 rows, so mutable padded tensors use a capacity larger
  // than compact Forward storage by at most 127 rows per owned expert. Compact
  // dR and source Return payloads have independent rank-local allocations.
  // dX remains disjoint so its M-blocks may return while dW2 still consumes dR.
  std::uint64_t backward_padded_capacity_rows = 0;
  std::size_t backward_setup_epoch = 0;
  std::size_t backward_dispatch_ready = 0;
  std::size_t backward_return_ready = 0;
  std::size_t backward_compact_to_padded = 0;
  std::size_t backward_padded_expert_rows = 0;
  std::size_t backward_padded_expert_offsets = 0;
  std::size_t forward_task_claim = 0;
  std::size_t forward_fc1_done = 0;
  std::size_t forward_x_arrived_rows = 0;
  // Single-node inference route exchange.  Unlike the generic Notify arena,
  // this metadata is peer mapped with the Forward pool and is rebuilt by the
  // resident kernel immediately before Dispatch.
  std::size_t fused_route_expert = 0;
  std::size_t fused_route_local_offset = 0;
  std::size_t fused_route_dst_row = 0;
  std::size_t fused_route_dst_rank = 0;
  std::size_t fused_route_dst_mtile = 0;
  std::size_t fused_combine_order = 0;
  std::size_t fused_local_expert_count = 0;
  std::size_t fused_source_expert_count = 0;
  std::size_t fused_source_routes = 0;
  std::size_t fused_source_gates = 0;
  std::size_t fused_owned_expert_ids = 0;
  std::size_t fused_expert_row_count = 0;
  std::size_t fused_expert_row_prefix = 0;
  std::size_t fused_mtile_offsets = 0;
  std::size_t fused_mtile_rows = 0;
  std::size_t fused_dispatch_runtime = 0;
  std::size_t forward_reduce_route_count = 0;
  std::size_t forward_reduce_ready_token = 0;
  std::size_t forward_return_rows = 0;
  std::size_t forward_return_queues = 0;
  std::size_t local_notifications = 0;
  unsigned local_notification_ctas = 0;
  std::size_t forward_scheduler = 0;
  std::size_t gin_sender_done = 0;
  std::size_t forward_completion = 0;
  std::size_t forward_context_completion = 0;
  std::size_t status = 0;
  std::size_t total = 0;
};

// Only these bytes participate in remote GIN registration. Compute arenas,
// weights, canonical rows, and local-P2P storage stay in ordinary CUDA
// allocations so enabling the network backend does not change their allocator.
struct GinWindowLayout {
  weftmoe_pf::NetworkLayout static_forward;
  communication::ChunkRingLayout rings;
  std::size_t send_dispatch_directory = 0;
  std::size_t receive_dispatch_directory = 0;
  std::size_t send_packet_headers = 0;
  std::size_t receive_packet_headers = 0;
  std::size_t send_fanout = 0;
  std::size_t receive_fanout = 0;
  std::size_t send_payload = 0;
  std::size_t receive_payload = 0;
  std::size_t return_staging = 0;
  std::size_t return_receive = 0;
  std::size_t return_fragment_ready = 0; // Backward arrival counters.
  std::size_t return_send_rows = 0;
  std::size_t return_receive_rows = 0;
  // FWD addresses these planes densely as BF16.  BWD reuses their larger
  // allocation as aligned [dX BF16..., dGate FP32] row records so one FULL
  // put carries both gradients and the completion signal.
  std::uint64_t backward_return_row_bytes = 0;
  std::size_t barrier = 0;
  std::size_t production_error_records = 0;
  std::size_t total = 0;
};

#if WEFTMOE_DEVICE_TIMELINE
struct TimelineAllocationLayout {
  std::size_t counts = 0;
  std::size_t overflow = 0;
  std::size_t records = 0;
  std::size_t total = 0;
};

TimelineAllocationLayout
make_timeline_allocation_layout(std::uint32_t max_ctas) {
  TimelineAllocationLayout layout;
  std::size_t cursor = 0;
  layout.counts =
      append(cursor, max_ctas, sizeof(std::uint32_t), "timeline CTA counts");
  layout.overflow =
      append(cursor, max_ctas, sizeof(std::uint32_t), "timeline CTA overflow");
  layout.records = append(
      cursor,
      checked_mul(max_ctas, profiling::kEventsPerCta, "timeline records"),
      sizeof(profiling::DeviceTimelineRecord), "timeline records");
  layout.total = cursor;
  return layout;
}
#endif

bool local_prefill_streaming(const ActivationTransportPoolConfig &config) {
  return forward_tuning::kLocalPrefillStreaming &&
         (forward_tuning::kFc1TileN == 128U ||
          forward_tuning::kFc1TileN == 256U) &&
         forward_tuning::kFc2TileN == forward_tuning::kFc1TileN &&
         forward_tuning::kStaticV4TileK == 64U &&
         !forward_tuning::kStaticV4PrefillEnabled &&
         config.max_hidden_size == forward_tuning::kMainHidden &&
         config.max_intermediate_size == forward_tuning::kMainIntermediate &&
         config.max_nodes == 1U && config.static_forward_window_tokens;
}

bool network_prefill_streaming(const ActivationTransportPoolConfig &config) {
  return config.max_nodes > 1U && config.static_forward_window_tokens &&
         forward_tuning::kFc1TileN == forward_tuning::kFc2TileN &&
         forward_tuning::kStaticV4TileK == 64U &&
         !forward_tuning::kStaticV4PrefillEnabled &&
         config.max_hidden_size == forward_tuning::kMainHidden &&
         config.max_intermediate_size == forward_tuning::kMainIntermediate;
}

unsigned contribution_windows(const ActivationTransportPoolConfig &config) {
  // The pull data plane reduces after Return drains. Keep all contributions,
  // but continue to bound Dispatch RX, X/S and Return TX independently.
  return network_prefill_streaming(config)
             ? static_cast<unsigned>(static_forward::ceil_div(
                   config.max_tokens_per_rank,
                   config.static_forward_window_tokens))
             : config.static_forward_contribution_windows;
}

static_forward::WindowShape
static_ring_shape(const ActivationTransportPoolConfig &config) {
  const auto local_world = config.max_world_size / config.max_nodes;
  // Bound the source mask to 16 bits without splitting a node's groups.
  auto sources = std::min(local_world, 16U);
  while (local_world % sources)
    --sources;
  if constexpr (WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP != 0U) {
    sources = WEFTMOE_FWD_STATIC_V4_SOURCE_GROUP;
    if (local_world % sources)
      throw std::invalid_argument(
          "static source group must divide the local EP team");
  }
  return {config.max_world_size, sources,
          static_cast<unsigned>(std::min<std::uint64_t>(
              config.static_forward_window_tokens, config.max_tokens_per_rank)),
          config.max_top_k};
}

static_forward::SourceMetadataLayout
make_source_metadata_layout(const ActivationTransportPoolConfig &config,
                            static_forward::WindowShape shape,
                            static_forward::PacketPartition packets) {
  static_forward::SourceMetadataLayout m;
  std::size_t cursor = 0;
  const auto windows =
      static_forward::ceil_div(config.max_tokens_per_rank, shape.tokens);
  const auto experts =
      std::uint64_t(config.max_world_size) * config.max_owned_experts;
  const auto routes = checked_mul(config.max_tokens_per_rank, config.max_top_k,
                                  "source metadata routes");
  const auto entries =
      checked_mul(windows, std::uint64_t(shape.tokens) * config.max_top_k,
                  "source window entries");
  const auto prefixes =
      std::uint64_t(config.max_nodes) * packets.lanes * (windows + 1U);
  m.header =
      append(cursor, 1U, sizeof(static_forward::SourceHeader), "source header");
  m.gates = append(cursor, routes, sizeof(float), "source gates");
  if (config.enable_backward_state)
    m.tape_rows = append(cursor, routes, sizeof(std::uint64_t),
                         "canonical tape row per source route");
  m.expert_prefix = append(cursor, windows * (experts + 1U),
                           sizeof(std::uint64_t), "source window experts");
  m.node_prefix = append(cursor, windows * (config.max_nodes + 1U),
                         sizeof(std::uint64_t), "source window nodes");
  m.route_order =
      append(cursor, entries, sizeof(std::uint64_t), "window routes");
  m.node_tokens = append(cursor, entries, sizeof(static_forward::NodeToken),
                         "node-deduplicated tokens");
  m.packet_prefix =
      append(cursor, prefixes, sizeof(std::uint64_t), "Dispatch prefixes");
  if (config.max_nodes > 1U) {
    const auto channels = config.max_nodes * packets.lanes;
    const auto local_world = config.max_world_size / config.max_nodes;
    const auto peer_windows =
        checked_mul(windows, config.max_nodes - 1U, "remote packet windows");
    const auto dispatch_capacity = std::min(
        routes, checked_mul(peer_windows, packets.packets(shape.tokens),
                            "Dispatch packet capacity"));
    const auto return_capacity = std::min(
        routes,
        checked_mul(peer_windows,
                    std::max<std::uint64_t>(
                        static_forward::kNetworkReturnPackets.lanes,
                        static_forward::kNetworkReturnPackets.packets(
                            std::uint64_t(shape.tokens) * config.max_top_k)),
                    "Return packet capacity"));
    m.selected =
        append(cursor, routes, sizeof(std::int32_t), "source selected experts");
    m.dispatch_offsets = append(cursor, channels + 1U, sizeof(std::uint64_t),
                                "Dispatch channel offsets");
    m.dispatch_packets = append(cursor, dispatch_capacity,
                                sizeof(static_forward::DispatchPacketPlan),
                                "Dispatch packet directory");
    m.dispatch_consumers = append(
        cursor, checked_mul(dispatch_capacity, local_world, "packet consumers"),
        sizeof(std::uint32_t), "Dispatch consumers");
    m.return_prefix =
        append(cursor, prefixes, sizeof(std::uint64_t), "Return prefixes");
    m.return_offsets = append(cursor, channels + 1U, sizeof(std::uint64_t),
                              "Return channel offsets");
    m.return_packets = append(cursor, return_capacity,
                              sizeof(static_forward::ReturnPacketPlan),
                              "Return packet directory");
    m.published_bytes = (cursor + kAlign - 1U) & ~(kAlign - 1U);
    m.locations = append(cursor, routes, sizeof(static_forward::RouteLocation),
                         "destination route locations");
  }
  m.bytes = (cursor + kAlign - 1U) & ~(kAlign - 1U);
  return m;
}

GinWindowLayout
make_gin_window_layout(const ActivationTransportPoolConfig &config) {
  GinWindowLayout layout;
  if (!config.enable_gin_rdma)
    return layout;
  std::size_t cursor = 0;
  const auto remote_nodes = config.max_nodes > 0 ? config.max_nodes - 1U : 0U;
  if (config.static_forward_window_tokens) {
    auto &network = layout.static_forward;
    constexpr auto packets = static_forward::kNetworkPackets;
    const auto shape = static_ring_shape(config);
    const auto metadata = make_source_metadata_layout(config, shape, packets);
    network.source_stride = metadata.bytes;
    network.source_bytes = metadata.published_bytes;
    network.source = append(cursor, config.max_nodes, metadata.bytes,
                            "ring source replicas");
    const auto packet =
        static_forward::packet_bytes(packets.rows, config.max_hidden_size);
    const auto returning = static_forward::packet_bytes(
        static_forward::kNetworkReturnPackets.rows, config.max_hidden_size);
    // Each channel reserves a whole worst-case Return window before FC2 can
    // write TX. This capacity depends on B/K/Q/L, not the total token count.
    const auto window_packets =
        static_forward::kNetworkReturnPackets.lane_packets(
            std::uint64_t(config.static_forward_window_tokens) *
                config.max_top_k,
            0U);
    unsigned minimum_depth = WEFTMOE_FWD_STATIC_V4_NETWORK_MIN_DEPTH;
    if (network_prefill_streaming(config))
      if (const auto *value = std::getenv("WEFTMOE_PF_MIN_RING_DEPTH")) {
        minimum_depth = static_cast<unsigned>(std::stoul(value));
        if (!minimum_depth)
          throw std::invalid_argument("PF minimum ring depth must be positive");
      }
    // Admission reserves a complete worst-case Return window; reducing the
    // heuristic floor must never reduce that correctness bound.
    const auto depth = static_cast<unsigned>(
        std::max<std::uint64_t>(minimum_depth, window_packets));
    network.rings = {config.max_nodes, packets.lanes, depth, packet, returning};
    network.ring_workers = static_cast<unsigned>(std::min<std::uint64_t>(
        network.rings.channels(), WEFTMOE_FWD_STATIC_V4_NETWORK_MAX_WORKERS));
    network.contexts = 3U * network.ring_workers;
    if (network_prefill_streaming(config)) {
      if (const auto *value = std::getenv("WEFTMOE_PF_CREDIT_WINDOWS")) {
        const auto windows = std::stoul(value);
        if (!windows || windows > 8U)
          throw std::invalid_argument("PF credit windows must be in [1, 8]");
        network.rings.depth *= static_cast<unsigned>(windows);
      }
      network.contexts = weftmoe_pf::kNetworkOwners - 1U;
      network.input =
          append(cursor,
                 checked_mul(config.max_tokens_per_rank, config.max_hidden_size,
                             "registered pull input"),
                 sizeof(transport::BFloat16), "immutable pull input");
    }
    network.ring_region =
        append(cursor, 1U, network.rings.bytes(), "static network rings");
    const auto contribution_rows =
        checked_mul(checked_mul(contribution_windows(config), shape.tokens,
                                "contribution windows"),
                    config.max_top_k, "contribution rows");
    network.contribution =
        append(cursor,
               checked_mul(contribution_rows, config.max_hidden_size,
                           "registered contribution elements"),
               sizeof(transport::BFloat16), "registered contribution ring");
  }
  const auto dense_return_row_bytes =
      checked_mul(config.max_hidden_size, sizeof(transport::BFloat16),
                  "GIN hidden row bytes");
  layout.backward_return_row_bytes =
      (dense_return_row_bytes + sizeof(float) + 15ULL) & ~15ULL;
  // The standalone transport-slice diagnostic predates the MoE operator and
  // has no compute arena. Keep its staging private to that explicit API.
  if (config.max_intermediate_size == 0) {
    const auto routes = checked_mul(config.max_tokens_per_rank,
                                    config.max_top_k, "GIN max routes");
    const auto outgoing_packets =
        checked_mul(config.max_nodes, config.max_tokens_per_rank,
                    "GIN outgoing packet capacity");
    const auto outgoing_fanout =
        checked_mul(outgoing_packets, config.max_top_k,
                    "GIN outgoing packet fanout capacity");
    constexpr std::uint32_t kGinContextsPerRail = 8;
    const auto directory_records = checked_mul(
        remote_nodes, kGinContextsPerRail, "GIN dispatch directory records");
    const auto incoming_packets =
        checked_mul(remote_nodes, config.max_tokens_per_rank,
                    "GIN incoming packet capacity");
    const auto packet_payload = checked_mul(
        outgoing_packets, config.max_hidden_size, "GIN send payload elements");
    const auto incoming_payload =
        checked_mul(incoming_packets, config.max_hidden_size,
                    "GIN receive payload elements");
    // FULL's send plane is indexed by canonical producer row and its receive
    // plane by compact source row. Backward reuses the send plane for padded
    // M128 dX rows, so reserve the largest index domain without a pre-put pack.
    const auto backward_padded_rows =
        config.enable_backward_state
            ? checked_add(config.max_destination_rows_per_rank,
                          checked_mul(config.max_owned_experts, 127ULL,
                                      "GIN Backward padded return rows"),
                          "GIN Backward padded return rows")
            : 0ULL;
    const auto direct_return_rows = std::max<std::uint64_t>(
        std::max<std::uint64_t>(
            checked_mul(remote_nodes, routes, "GIN compact return rows"),
            config.max_destination_rows_per_rank),
        backward_padded_rows);
    const auto backward_return_row_bytes =
        config.enable_backward_state
            ? (checked_add(dense_return_row_bytes, sizeof(float) + 15ULL,
                           "GIN Backward return row bytes") &
               ~15ULL)
            : dense_return_row_bytes;
    layout.backward_return_row_bytes = backward_return_row_bytes;
    layout.send_dispatch_directory =
        append(cursor, directory_records, sizeof(GinDispatchContextRecord),
               "GIN send dispatch directory");
    layout.receive_dispatch_directory =
        append(cursor, directory_records, sizeof(GinDispatchContextRecord),
               "GIN receive dispatch directory");
    layout.send_packet_headers =
        append(cursor, outgoing_packets, sizeof(LocalP2pRouteHeader),
               "GIN send packet headers");
    layout.receive_packet_headers =
        append(cursor, incoming_packets, sizeof(LocalP2pRouteHeader),
               "GIN receive packet headers");
    // Packet descriptors are materialized in transport order. A token can
    // contribute one node-deduplicated packet to every node, so packet-position
    // indexing requires nodes*tokens*top-k capacity even though the number of
    // logical routes remains tokens*top-k. The previous route-sized allocation
    // made the first remote-node packet range overrun the following planes.
    layout.send_fanout =
        append(cursor, outgoing_fanout, sizeof(ForwardDispatchFanoutRecord),
               "GIN send Forward fanout");
    layout.receive_fanout = append(
        cursor,
        checked_mul(incoming_packets, config.max_top_k,
                    "GIN receive fanout capacity"),
        sizeof(ForwardDispatchFanoutRecord), "GIN receive Forward fanout");
    layout.send_payload =
        append(cursor, packet_payload, sizeof(transport::BFloat16),
               "GIN send payload");
    layout.receive_payload =
        append(cursor, incoming_payload, sizeof(transport::BFloat16),
               "GIN receive payload");
    layout.return_staging =
        append(cursor, direct_return_rows,
               checked_size(backward_return_row_bytes,
                            "GIN direct return staging row bytes"),
               "GIN return staging");
    layout.return_receive = append(
        cursor, routes,
        checked_size(backward_return_row_bytes, "GIN return receive row bytes"),
        "GIN return receive");
    // The worst legal fragmentation begins at every row, hence routes marker
    // words are sufficient for every compact Return layout.  The main path
    // normally emits one marker per M128 producer block.
    layout.return_fragment_ready = append(cursor, routes, sizeof(std::uint64_t),
                                          "GIN return fragment ready counters");
    layout.return_send_rows =
        append(cursor, config.max_destination_rows_per_rank,
               sizeof(std::uint64_t), "GIN return send row notifications");
    layout.return_receive_rows = append(cursor, routes, sizeof(std::uint64_t),
                                        "GIN return receive row queue");
  }
  layout.barrier = append(cursor, 1, sizeof(std::uint64_t), "GIN barrier");
  layout.production_error_records =
      append(cursor, config.max_world_size, sizeof(ProductionStatusRecord),
             "GIN production error records");
  if (config.static_forward_window_tokens && !config.enable_backward_state) {
    layout.total = cursor;
    return layout;
  }
  auto &rings = layout.rings;
  // A retained tape does not own either wire ring. Both sets of controls and
  // completion state survives alternating Forward/Backward executions.
  rings.dispatch_channels = remote_nodes * communication::kDispatchChannels;
  rings.return_channels = remote_nodes *
                          (config.max_world_size / config.max_nodes) *
                          communication::kReturnPorts;
  rings.dispatch_row_bytes =
      sizeof(communication::DispatchRow) +
      config.max_top_k * sizeof(ForwardDispatchFanoutRecord) +
      dense_return_row_bytes;
  rings.return_row_bytes =
      sizeof(communication::ReturnRow) + dense_return_row_bytes;
  rings.dispatch_stride = sizeof(communication::ChunkHeader) +
                          communication::kChunkRows * rings.dispatch_row_bytes;
  rings.return_stride = sizeof(communication::ChunkHeader) +
                        communication::kChunkRows * rings.return_row_bytes;
  rings.control =
      append(cursor, rings.channels(), sizeof(communication::ChunkChannel),
             "GIN ring controls");
  const auto dispatch_slots =
      rings.dispatch_channels * communication::kRingDepth;
  const auto return_slots = rings.return_channels * communication::kRingDepth;
  rings.dispatch_tx = append(cursor, dispatch_slots, rings.dispatch_stride,
                             "GIN dispatch TX ring");
  rings.dispatch_rx = append(cursor, dispatch_slots, rings.dispatch_stride,
                             "GIN dispatch RX ring");
  rings.return_tx =
      append(cursor, return_slots, rings.return_stride, "GIN return TX ring");
  rings.return_rx =
      append(cursor, return_slots, rings.return_stride, "GIN return RX ring");
  layout.total = cursor;
  return layout;
}

Layout make_layout(const ActivationTransportPoolConfig &config) {
  if (config.max_tokens_per_rank == 0 || config.max_hidden_size == 0 ||
      config.max_top_k == 0 || config.max_world_size == 0 ||
      config.max_nodes == 0 || config.channels == 0 ||
      config.ring_slots_per_source_channel == 0 ||
      config.max_inflight_generations == 0 ||
      config.max_destination_rows_per_rank == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-query", "capacity", 0, "all capacities > 0",
                    "zero field");
  if (config.max_intermediate_size != 0 &&
      (config.max_hidden_size != forward_tuning::kMainHidden ||
       config.max_intermediate_size != forward_tuning::kMainIntermediate))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-query", "Forward shape", 0,
                    "compiled Forward hidden/intermediate dimensions",
                    std::to_string(config.max_hidden_size) + "x" +
                        std::to_string(config.max_intermediate_size));
  if (config.max_intermediate_size != 0 &&
      (config.max_destination_rows_per_rank > UINT32_MAX ||
       checked_mul(config.max_tokens_per_rank, config.max_top_k,
                   "return route indices") > UINT32_MAX))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "return row index", 0,
                    "representable as uint32", "overflow");
  Layout layout;
  std::size_t cursor = 0;
  auto &ring = layout.static_ring;
  if (config.static_forward_window_tokens) {
    if constexpr (forward_tuning::kTileM != 128U ||
                  forward_tuning::kUnifiedThreads != 384U ||
                  forward_tuning::kMathWarpgroups != 2U ||
                  forward_tuning::kUnifiedCommBase != 288U ||
                  forward_tuning::kUnifiedCommThreads != 96U)
      throw std::invalid_argument(
          "static ring requires M128 and the 256/32/96 worker partition");
    if (config.static_forward_contribution_windows == 0U)
      throw std::invalid_argument("static ring requires contribution capacity");
    if (config.max_world_size < config.max_nodes)
      throw std::invalid_argument(
          "static ring requires at least one rank per node");
    if (config.enable_socket_fallback ||
        (config.max_nodes > 1U && !config.enable_gin_rdma) ||
        config.max_intermediate_size == 0U)
      throw std::invalid_argument(
          "static ring requires Forward and GIN for remote nodes");
    ring.shape = static_ring_shape(config);
    const auto window_blocks = ring.shape.max_mblocks(config.max_owned_experts);
    ring.max_windows = ring.shape.windows(config.max_tokens_per_rank);
    ring.max_blocks =
        checked_mul(ring.max_windows, window_blocks, "static ring plan blocks");
    if (ring.max_windows > UINT32_MAX ||
        window_blocks > UINT32_MAX / (2ULL * ring.shape.tile_m))
      throw std::invalid_argument("static ring window exceeds device indices");
    ring.depth = static_cast<unsigned>(2U * window_blocks);
    ring.contribution_depth = contribution_windows(config);
    ring.state =
        append(cursor, 1, sizeof(static_forward::RingState), "ring state");
    ring.source_epochs =
        append(cursor, config.max_world_size,
               sizeof(static_forward::SourceEpoch), "source epochs");
    ring.source_reserved =
        append(cursor, config.max_world_size, sizeof(std::uint64_t),
               "source reservation heads");
    ring.slots = append(cursor, ring.depth, sizeof(static_forward::ComputeSlot),
                        "X/S/C slot controls");
    if (config.max_nodes > 1U)
      ring.return_ready = append(
          cursor,
          std::uint64_t(config.max_world_size / config.max_nodes) * ring.depth,
          sizeof(std::uint64_t), "rail-local C generations");
    ring.contributions = append(cursor, ring.contribution_depth,
                                sizeof(static_forward::ContributionSlot),
                                "contribution window controls");
    ring.contribution_ready =
        append(cursor,
               std::uint64_t(ring.contribution_depth) * ring.shape.tokens *
                   config.max_top_k,
               sizeof(std::uint64_t), "contribution route generations");
    if (config.max_nodes > 1U)
      ring.contribution_rows =
          append(cursor,
                 checked_mul(config.max_tokens_per_rank, config.max_top_k,
                             "contribution permutation routes"),
                 sizeof(std::uint32_t), "source contribution row permutation");
    ring.source = make_source_metadata_layout(config, ring.shape, ring.packets);
    if (config.max_nodes == 1U)
      ring.metadata = append(cursor, 1U, ring.source.bytes, "source metadata");
    ring.window_prefix = append(cursor, ring.max_windows + 1U,
                                sizeof(std::uint64_t), "window block prefixes");
    ring.window_sources =
        append(cursor, ring.max_windows * ring.shape.sources,
               sizeof(static_forward::WindowSource), "window source addresses");
    ring.expert_rows =
        append(cursor, ring.max_windows * config.max_owned_experts,
               sizeof(std::uint64_t), "window expert rows");
    ring.expert_blocks =
        append(cursor, ring.max_windows * (config.max_owned_experts + 1U),
               sizeof(std::uint64_t), "window expert blocks");
    ring.blocks =
        append(cursor, ring.max_blocks, sizeof(static_forward::RingBlock),
               "immutable compute blocks");
    ring.rows = append(
        cursor,
        checked_mul(ring.max_blocks, ring.shape.tile_m, "ring reverse rows"),
        sizeof(static_forward::RingRow), "immutable reverse routes");
    if (config.enable_backward_state)
      ring.tape_rows = append(cursor, ring.max_blocks * ring.shape.tile_m,
                              sizeof(std::uint64_t), "canonical tape rows");
    if (config.max_nodes > 1U)
      ring.pull_packets = append(cursor,
                                 ring.max_windows * ring.shape.sources *
                                     ring.packets.packets(ring.shape.tokens),
                                 sizeof(static_forward::PullPacket),
                                 "immutable remote pull packets");
  }
  const auto activation_rows =
      ring.depth ? std::uint64_t(ring.depth) * ring.shape.tile_m
                 : config.max_destination_rows_per_rank;
  const auto routes =
      checked_mul(config.max_tokens_per_rank, config.max_top_k, "max routes");
  const auto links = checked_mul(config.max_nodes, config.channels, "links");
  const auto ring_slots =
      checked_mul(checked_mul(config.max_nodes, config.channels, "ring links"),
                  config.ring_slots_per_source_channel, "ring slots");
  const auto rank_links =
      checked_mul(config.max_world_size, config.channels, "rank links");
  const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "ingress slots");
  layout.execution_epoch =
      append(cursor, 1, sizeof(std::uint64_t), "execution epoch");
  layout.input = append(cursor,
                        checked_mul(config.max_tokens_per_rank,
                                    config.max_hidden_size, "input elements"),
                        sizeof(transport::BFloat16), "input");
  layout.ring_headers =
      append(cursor, ring_slots, sizeof(transport::TransportSlotHeader),
             "ring headers");
  layout.ring_payload = append(
      cursor,
      checked_mul(ring_slots, config.max_hidden_size, "ring payload elements"),
      sizeof(transport::BFloat16), "ring payload");
  layout.ring_ready_sequence =
      append(cursor, ring_slots, sizeof(std::uint64_t), "ring ready sequence");
  layout.link_next_sequence =
      append(cursor, links, sizeof(std::uint64_t), "link next sequence");
  layout.link_credits =
      append(cursor, links, sizeof(std::uint32_t), "link credits");
  layout.ingress_headers =
      append(cursor, ingress_slots, sizeof(LocalP2pRouteHeader),
             "P2P ingress headers");
  layout.ingress_ready_sequence =
      append(cursor, ingress_slots, sizeof(std::uint64_t), "P2P ingress ready");
  layout.ingress_next_sequence =
      append(cursor, rank_links, sizeof(std::uint64_t), "P2P ingress next");
  layout.ingress_published_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P ingress published");
  layout.ingress_credits =
      append(cursor, rank_links, sizeof(std::uint32_t), "P2P ingress credits");
  layout.outgoing_route_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P outgoing sequence");
  layout.direct_route_headers =
      append(cursor, config.max_destination_rows_per_rank,
             sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  layout.return_ready_epoch =
      append(cursor, routes, sizeof(std::uint64_t), "P2P combine ready epoch");
  // Two pool-private pointer slots follow the world-rank table. They carry the
  // Forward completion-vector pointers without widening the kernel args.
  layout.peer_bases = append(cursor, config.max_world_size + 2ULL,
                             sizeof(void *), "P2P peer base table");
  layout.p2p_barrier_send =
      append(cursor, 2, sizeof(std::uint64_t), "P2P barrier send");
  layout.p2p_barrier_receive =
      append(cursor, checked_mul(config.max_world_size, 2, "P2P barrier words"),
             sizeof(std::uint64_t), "P2P barrier receive");
  layout.gin_local_compute_ready =
      append(cursor, config.max_world_size, sizeof(std::uint64_t),
             "GIN node-local compute ready epoch");
  layout.terminal_epoch =
      append(cursor, config.max_world_size, sizeof(std::uint64_t),
             "P2P terminal generation epoch");
  layout.production_error_records =
      append(cursor, config.max_world_size, sizeof(ProductionStatusRecord),
             "peer-visible production error records");
  layout.production_status_matrix =
      append(cursor,
             checked_mul(config.max_world_size, 8ULL,
                         "production terminal status words"),
             sizeof(std::uint64_t), "production terminal status matrix");
  layout.packet_claim =
      append(cursor, routes, sizeof(std::uint32_t), "packet claim bitmap");
  layout.canonical_rows = append(
      cursor,
      checked_mul(activation_rows, config.max_hidden_size, "canonical rows"),
      sizeof(transport::BFloat16), "canonical rows");
  layout.canonical_gates =
      append(cursor, activation_rows, sizeof(float), "canonical route gates");
  const auto route_elements =
      checked_mul(ring.depth && !local_prefill_streaming(config)
                      ? std::uint64_t(ring.contribution_depth) *
                            ring.shape.tokens * config.max_top_k
                      : routes,
                  config.max_hidden_size, "route elements");
  if (!ring.depth || !config.enable_gin_rdma)
    layout.contribution = append(cursor, route_elements,
                                 sizeof(transport::BFloat16), "contribution");
  layout.combine_return =
      ring.depth ? layout.contribution
                 : append(cursor, route_elements, sizeof(transport::BFloat16),
                          "combine return");
  if (config.max_intermediate_size == 0 || config.enable_socket_fallback)
    layout.combined =
        append(cursor,
               checked_mul(config.max_tokens_per_rank, config.max_hidden_size,
                           "combined elements"),
               sizeof(transport::BFloat16), "diagnostic combined output");
  if (config.enable_socket_fallback ||
      (config.enable_gin_rdma && config.max_intermediate_size == 0)) {
    layout.socket_count_matrix =
        append(cursor,
               checked_mul(config.max_world_size, config.max_world_size,
                           "Socket count matrix"),
               sizeof(std::uint64_t), "Socket count matrix");
    layout.socket_header_send = append(
        cursor, routes, sizeof(SocketRouteHeader), "Socket send headers");
    layout.socket_header_receive =
        append(cursor, config.max_destination_rows_per_rank,
               sizeof(SocketRouteHeader), "Socket receive headers");
    layout.socket_payload_receive = append(
        cursor,
        checked_mul(config.max_destination_rows_per_rank,
                    config.max_hidden_size, "Socket receive payload elements"),
        sizeof(transport::BFloat16), "Socket receive payload");
  }
  if (config.max_intermediate_size != 0) {
    const auto forward_elements =
        checked_mul(activation_rows, config.max_intermediate_size,
                    "forward gated U elements");
    layout.forward_s = append(cursor, forward_elements,
                              sizeof(transport::BFloat16), "forward gated U");
    if (config.enable_backward_state) {
#if defined(WEFTMOE_BENCH_TRANSPORT_ONLY)
      // Dispatch must not overwrite the diagnostic's prefilled Return fixture.
      layout.forward_training_output =
          append(cursor,
                 checked_mul(config.max_destination_rows_per_rank,
                             config.max_hidden_size,
                             "forward training output elements"),
                 sizeof(transport::BFloat16), "forward training output");
#else
      // FC2 becomes ready only after all FC1 N tiles of this M block retire.
      // X is already saved to caller-owned storage, so training can reuse the
      // same canonical plane as inference. Static Backward reuses this X ring
      // for bounded dX output, while saved X and private dR remain separate.
      layout.forward_training_output = layout.canonical_rows;
#endif
      layout.backward_padded_capacity_rows =
          checked_add(config.max_destination_rows_per_rank,
                      checked_mul(config.max_owned_experts, 127ULL,
                                  "backward expert padding rows"),
                      "backward padded row capacity");
      if (layout.backward_padded_capacity_rows > UINT32_MAX)
        transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                        "transport-pool-query", "Backward padded directory", 0,
                        "uint32 row indices",
                        std::to_string(layout.backward_padded_capacity_rows));
      layout.backward_setup_epoch =
          append(cursor, config.max_world_size, sizeof(std::uint64_t),
                 "backward setup epochs");
      layout.backward_dispatch_ready =
          append(cursor, config.max_destination_rows_per_rank,
                 sizeof(std::uint64_t), "backward dR row-ready epochs");
      layout.backward_return_ready =
          append(cursor, routes, sizeof(std::uint64_t),
                 "backward dX route-ready epochs");
      layout.backward_compact_to_padded =
          append(cursor, config.max_destination_rows_per_rank,
                 sizeof(std::uint32_t), "backward compact-to-padded rows");
      layout.backward_padded_expert_rows =
          append(cursor, config.max_owned_experts, sizeof(std::uint64_t),
                 "backward padded expert rows");
      layout.backward_padded_expert_offsets = append(
          cursor, static_cast<std::uint64_t>(config.max_owned_experts) + 1ULL,
          sizeof(std::uint64_t), "backward padded expert offsets");
    }
    // Expert segmentation can make every incoming row a distinct partial
    // M-block.  Capacity must therefore be bounded by rows, not by a dense
    // ceil(rows/64) estimate.
    const auto max_mblocks = config.max_destination_rows_per_rank;
    const auto fc1_ntiles =
        (config.max_intermediate_size + forward_tuning::kFc1TileN - 1ULL) /
        forward_tuning::kFc1TileN;
    const auto forward_claim_words =
        checked_mul(max_mblocks, 2ULL + fc1_ntiles, "forward task claims");
    auto task_claim_words = forward_claim_words;
    if (config.enable_backward_state) {
      const auto backward_mblocks = (layout.backward_padded_capacity_rows +
                                     backward_tuning::kTileM - 1ULL) /
                                    backward_tuning::kTileM;
      const auto backward_scheduler =
          pipeline::backward::make_backward_scheduler_layout(
              backward_mblocks,
              static_cast<std::uint32_t>(config.max_hidden_size /
                                         backward_tuning::kDxTileN /
                                         backward_tuning::kActivationClusterN),
              config.max_owned_experts);
      if (backward_scheduler.total_words > task_claim_words)
        task_claim_words = backward_scheduler.total_words;
    }
    layout.forward_task_claim =
        append(cursor, task_claim_words, sizeof(std::uint64_t),
               "forward/backward aliased task claims");
    layout.forward_fc1_done = append(cursor, max_mblocks, sizeof(std::uint64_t),
                                     "forward FC1 done tiles");
    // A destination tile is produced concurrently by every node-local rail.
    // Keep one row counter per local-rank/rail instead of issuing contended
    // system-scope atomics from several GPUs into one peer word. The consumer
    // sums the fixed rail vector with acquire loads. This is both deterministic
    // and avoids relying on peer-atomic combining behavior of a particular
    // NVLink/virtualization topology.
    layout.forward_x_arrived_rows =
        append(cursor,
               checked_mul(max_mblocks, config.max_world_size,
                           "forward exact X M-tile rail counters"),
               sizeof(std::uint64_t), "forward exact X M-tile arrived rows");
    const auto fused_experts =
        checked_mul(config.max_owned_experts, config.max_world_size,
                    "single-node inference experts");
    layout.fused_route_expert =
        append(cursor, routes, sizeof(std::uint32_t), "fused route experts");
    layout.fused_route_local_offset = append(
        cursor, routes, sizeof(std::uint64_t), "fused route local offsets");
    layout.fused_route_dst_row =
        append(cursor, routes, sizeof(std::uint64_t), "fused route rows");
    layout.fused_route_dst_rank =
        append(cursor, routes, sizeof(std::uint32_t), "fused route ranks");
    layout.fused_route_dst_mtile =
        append(cursor, routes, sizeof(std::uint32_t), "fused route M tiles");
    layout.fused_combine_order =
        append(cursor, routes, sizeof(std::uint32_t), "fused combine order");
    layout.fused_local_expert_count =
        append(cursor, fused_experts, sizeof(std::uint64_t),
               "fused local expert counts");
    layout.fused_source_expert_count =
        append(cursor, fused_experts, sizeof(std::uint64_t),
               "fused global expert totals");
    layout.fused_source_routes = append(
        cursor,
        checked_mul(checked_mul(config.max_owned_experts, config.max_world_size,
                                "fused source route expert/rank planes"),
                    checked_mul(config.max_tokens_per_rank, config.max_top_k,
                                "fused source routes per rank"),
                    "fused source route table"),
        // Active worlds may be smaller than the pool capacity. Provision a
        // route+gate record whenever one token window could use the local
        // fast path, independently of the capacity's source-group count.
        ring.depth && config.max_tokens_per_rank <= ring.shape.tokens
            ? sizeof(static_forward::SourceRoute)
            : sizeof(std::uint32_t),
        "fused source route table");
    layout.fused_source_gates =
        ring.depth
            ? (config.max_nodes == 1U ? ring.metadata + ring.source.gates : 0U)
            : append(cursor, routes, sizeof(float), "fused source route gates");
    layout.fused_owned_expert_ids =
        append(cursor, config.max_owned_experts, sizeof(std::uint32_t),
               "fused owned expert ids");
    layout.fused_expert_row_count =
        append(cursor, config.max_owned_experts, sizeof(std::uint64_t),
               "fused owned expert row counts");
    layout.fused_expert_row_prefix = append(
        cursor, static_cast<std::uint64_t>(config.max_owned_experts) + 1ULL,
        sizeof(std::uint64_t), "fused owned expert row prefixes");
    layout.fused_mtile_offsets = append(
        cursor, static_cast<std::uint64_t>(config.max_owned_experts) + 1ULL,
        sizeof(std::uint64_t), "fused owned expert M-tile offsets");
    layout.fused_mtile_rows = append(cursor, max_mblocks, sizeof(std::uint32_t),
                                     "fused expected M-tile rows");
    layout.fused_dispatch_runtime = append(
        cursor, 1, sizeof(DeviceDispatchRuntime), "fused dispatch runtime");
    layout.forward_reduce_route_count =
        append(cursor, config.max_tokens_per_rank, sizeof(std::uint32_t),
               "forward reduce ready route counts");
    layout.forward_reduce_ready_token =
        append(cursor, config.max_tokens_per_rank, sizeof(std::uint64_t),
               "forward reduce ready token queue");
    layout.forward_return_rows =
        append(cursor,
               checked_add(config.max_destination_rows_per_rank,
                           checked_add(config.max_destination_rows_per_rank /
                                           communication::kReturnReadyBlockRows,
                                       1ULL + config.max_world_size,
                                       "return ready blocks"),
                           "return rows and ready blocks"),
               sizeof(std::uint32_t), "forward return row queues");
    layout.local_notification_ctas = std::max(
        std::max(forward_tuning::kRelayCtas, forward_tuning::kRemoteRelayCtas) +
            pipeline::kRdmaCtaCount,
        backward_tuning::kRelayCtas + backward_tuning::kRdmaCtas);
    if constexpr (forward_tuning::kUnifiedWorkers ||
                  backward_tuning::kUnifiedWorkers) {
      int device = 0, sms = 0;
      if (cudaGetDevice(&device) != cudaSuccess ||
          cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                                 device) != cudaSuccess)
        transport_error(NotifyErrorCode::BackendUnavailable,
                        "transport-pool-query", "SM count", 0,
                        "available CUDA device", "device query failed");
      layout.local_notification_ctas =
          std::max(layout.local_notification_ctas, static_cast<unsigned>(sms));
      layout.gin_sender_done =
          append(cursor, layout.local_notification_ctas, sizeof(unsigned),
                 "unified GIN sender completion flags");
    }
    layout.local_notifications = append(
        cursor,
        1ULL * (config.max_world_size / config.max_nodes) *
            layout.local_notification_ctas,
        sizeof(communication::NotificationChannel), "local notification rings");
    layout.forward_return_queues = append(cursor, 1ULL + config.max_world_size,
                                          sizeof(communication::ReturnRowQueue),
                                          "per-peer return queue controls");
    layout.forward_scheduler =
        append(cursor, 4 + config.max_world_size * 32ULL, sizeof(std::uint64_t),
               "forward scheduler");
    layout.forward_completion =
        append(cursor, 1, sizeof(transport_state::GenerationCompletionVector),
               "forward generation completion vector");
    const auto max_context_records =
        checked_mul(config.max_nodes > 0 ? config.max_nodes - 1ULL : 0ULL, 8ULL,
                    "forward GIN context completion records");
    layout.forward_context_completion =
        append(cursor, max_context_records,
               sizeof(transport_state::ContextCompletionRecord),
               "forward GIN context completion records");
  }
  layout.status = append(cursor, 1, sizeof(TransportDeviceStatus), "status");
  layout.total = cursor;
  return layout;
}

enum class SlotState : std::uint8_t { Available, InUse, Quarantined };

struct BackwardPeerWorkspace;

struct PoolSlot {
  void *device = nullptr;
  void *gin_window = nullptr;
#if WEFTMOE_DEVICE_TIMELINE
  void *forward_timeline = nullptr;
#endif
  TransportDeviceStatus *host_status = nullptr;
  TransportDeviceStatus *host_forward_status = nullptr;
  std::uint32_t *host_forward_done = nullptr;
  std::uint32_t *device_forward_done = nullptr;
  ProductionStatusRecord *host_production_status = nullptr;
  std::uint64_t *host_forward_progress = nullptr;
  std::uint64_t *device_forward_progress = nullptr;
  std::size_t forward_progress_words = 0;
  std::size_t generation_debug_words = 0;
  std::uint32_t generation_debug_contexts = 0;
  std::uint64_t *host_cancel_epoch = nullptr;
  std::uint64_t *host_socket_counts = nullptr;
  std::uint64_t *device_cancel_epoch = nullptr;
  bool cancel_epoch_dirty = false;
  cudaEvent_t producer_ready = nullptr;
  unsigned long long producer_capture_id = 0;
  cudaEvent_t start = nullptr;
  cudaEvent_t forward_start = nullptr;
  unsigned long long forward_capture_id = 0;
  unsigned long long backward_capture_id = 0;
  cudaEvent_t forward_output_ready = nullptr;
  cudaEvent_t forward_complete = nullptr;
  // PF backward uses a communication stream and a completion join stream.
  // These streams belong to the reusable workspace, not individual calls.
  cudaStream_t backward_transport_stream = nullptr;
  cudaStream_t backward_compute_stream = nullptr;
  cudaEvent_t backward_roles_ready = nullptr;
  cudaEvent_t backward_join_transport = nullptr;
  cudaEvent_t backward_join_compute = nullptr;
  cudaEvent_t backward_start = nullptr;
  cudaEvent_t backward_compute_start = nullptr;
  cudaEvent_t backward_compute_done = nullptr;
  cudaEvent_t backward_transport_done = nullptr;
  cudaEvent_t backward_reduce_done = nullptr;
  cudaEvent_t backward_late_compute_done = nullptr;
  cudaEvent_t backward_complete = nullptr;
  cudaEvent_t dispatch_complete = nullptr;
  cudaEvent_t complete = nullptr;
  SlotState state = SlotState::Available;
  std::uint64_t epoch = 0;
  std::uint64_t transport_epoch = 0;
  bool completion_recorded = false;
  CUtensorMap forward_x_map{};
  CUtensorMap forward_s_map{};
  CUtensorMap decode_x_map[2]{};
  CUtensorMap decode_s_map[2]{};
  CUtensorMap forward_output_store_map{};
  CUtensorMap forward_training_output_store_map{};
  CUtensorMap forward_staging_store_map{};
  bool forward_maps_valid = false;
  std::shared_ptr<BackwardPeerWorkspace> backward_peer_workspace;
};

struct PeerMappings {
  std::uint32_t device = 0;
  std::shared_ptr<internal::NcclSessionState> nccl;
  std::vector<void *> bases;
  std::vector<std::uint8_t> owned;
  void close() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    for (std::size_t i = 0; i < bases.size(); ++i) {
      if (owned[i] && bases[i])
        (void)cudaIpcCloseMemHandle(bases[i]);
      owned[i] = 0;
      bases[i] = nullptr;
    }
  }
  ~PeerMappings() noexcept { close(); }
};

// Communication storage outlives any one compute arena. A larger Notify
// result may replace the arena while a saved Forward still owns the old one.
// The window and device communicator belong here, never to an IPC mapping.
struct GinResources {
  std::uint32_t device = 0;
  std::size_t bytes_per_slot = 0;
  std::vector<void *> allocations;
  std::shared_ptr<internal::NcclSessionState> nccl;
  ncclDevComm_t dev_comm{};
  std::vector<ncclWindow_t> windows;
  bool dev_comm_created = false;
  std::mutex mutex;
  cudaEvent_t last_use = nullptr;
  unsigned long long last_capture_id = 0;

  ~GinResources() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (last_use) {
      (void)cudaEventSynchronize(last_use);
      (void)cudaEventDestroy(last_use);
    }
    using Deregister = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using Destroy = ncclResult_t (*)(ncclComm_t, const ncclDevComm_t *);
    using Free = ncclResult_t (*)(void *);
    const auto deregister = reinterpret_cast<Deregister>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto destroy =
        reinterpret_cast<Destroy>(dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto free_memory =
        reinterpret_cast<Free>(dlsym(RTLD_DEFAULT, "ncclMemFree"));
    if (nccl && deregister)
      for (const auto window : windows)
        if (window)
          (void)deregister(nccl->comm, window);
    if (nccl && dev_comm_created && destroy)
      (void)destroy(nccl->comm, &dev_comm);
    if (free_memory)
      for (auto *allocation : allocations)
        if (allocation)
          (void)free_memory(allocation);
  }
};

template <typename T> T *at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}
template <typename T> const T *at(const void *base, std::size_t offset) {
  return reinterpret_cast<const T *>(static_cast<const unsigned char *>(base) +
                                     offset);
}

#if WEFTMOE_DEVICE_TIMELINE
profiling::DeviceTimelineContext
make_timeline_context(void *allocation, const TimelineAllocationLayout &layout,
                      std::uint32_t max_ctas, std::uint32_t rank,
                      profiling::TimelinePhase phase,
                      std::uint64_t generation) {
  profiling::DeviceTimelineContext context;
  context.records =
      at<profiling::DeviceTimelineRecord>(allocation, layout.records);
  context.counts = at<std::uint32_t>(allocation, layout.counts);
  context.overflow = at<std::uint32_t>(allocation, layout.overflow);
  context.max_ctas = max_ctas;
  context.capacity_per_cta = profiling::kEventsPerCta;
  context.rank = rank;
  context.phase = static_cast<std::uint32_t>(phase);
  context.generation = generation;
  return context;
}

void reset_timeline(const char *phase, void *allocation,
                    const TimelineAllocationLayout &layout,
                    std::uint32_t max_ctas, cudaStream_t stream,
                    std::uint32_t rank, std::uint32_t world,
                    std::uint64_t generation, const NcclSession &session) {
  const auto reset_bytes = checked_size(
      checked_mul(max_ctas, sizeof(std::uint32_t), "timeline count reset"),
      "timeline count reset");
  internal::throw_cuda(phase,
                       cudaMemsetAsync(at<void>(allocation, layout.counts), 0,
                                       reset_bytes, stream),
                       rank, world, session.session().identity(), generation,
                       ExchangeKind::ExpertCounts,
                       "reset device timeline counts");
  internal::throw_cuda(phase,
                       cudaMemsetAsync(at<void>(allocation, layout.overflow), 0,
                                       reset_bytes, stream),
                       rank, world, session.session().identity(), generation,
                       ExchangeKind::ExpertCounts,
                       "reset device timeline overflow");
}
#endif

struct ActivationPoolState {
  ActivationTransportPoolConfig config;
  Layout layout;
  GinWindowLayout gin_layout;
#if WEFTMOE_DEVICE_TIMELINE
  TimelineAllocationLayout timeline_layout;
  std::uint32_t timeline_max_ctas = 0;
#endif
  std::vector<std::byte> session_identity;
  std::uint32_t world = 0;
  CudaExecution execution;
  std::uint32_t device = 0;
  std::mutex mutex;
  std::weak_ptr<ActivationMegaKernelBackendState> gin_backend;
  std::shared_ptr<GinResources> gin_resources;
  std::shared_ptr<PeerMappings> peer_mappings;
  std::unique_ptr<CollectiveTensorPreparation> forward_tensor_preparation;
  std::vector<PoolSlot> slots;
  std::uint64_t epoch_high = 0;
  cudaStream_t cancel_stream = nullptr;
  cublasHandle_t backward_blas = nullptr;
  std::mutex backward_blas_mutex;

  ~ActivationPoolState() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (backward_blas)
      (void)cublasDestroy(backward_blas);
    for (auto &slot : slots) {
      if (slot.completion_recorded && slot.complete)
        (void)cudaEventSynchronize(slot.complete);
      if (slot.producer_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.producer_ready)
        (void)cudaEventDestroy(slot.producer_ready);
      if (slot.start)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.start)
        (void)cudaEventDestroy(slot.start);
      if (slot.forward_start)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_start)
        (void)cudaEventDestroy(slot.forward_start);
      if (slot.forward_output_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_output_ready)
        (void)cudaEventDestroy(slot.forward_output_ready);
      if (slot.forward_complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_complete)
        (void)cudaEventDestroy(slot.forward_complete);
      if (slot.backward_roles_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.backward_roles_ready)
        (void)cudaEventDestroy(slot.backward_roles_ready);
      if (slot.backward_join_transport)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.backward_join_transport)
        (void)cudaEventDestroy(slot.backward_join_transport);
      if (slot.backward_join_compute)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.backward_join_compute)
        (void)cudaEventDestroy(slot.backward_join_compute);
      const cudaEvent_t backward_events[] = {
          slot.backward_start,        slot.backward_compute_start,
          slot.backward_compute_done, slot.backward_transport_done,
          slot.backward_reduce_done,  slot.backward_late_compute_done,
          slot.backward_complete};
      for (const auto event : backward_events) {
        if (event)
          internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
        if (event)
          (void)cudaEventDestroy(event);
      }
      if (slot.backward_transport_stream)
        (void)cudaStreamDestroy(slot.backward_transport_stream);
      if (slot.backward_compute_stream)
        (void)cudaStreamDestroy(slot.backward_compute_stream);
      if (slot.dispatch_complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.dispatch_complete)
        (void)cudaEventDestroy(slot.dispatch_complete);
      if (slot.complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.complete)
        (void)cudaEventDestroy(slot.complete);
      if (slot.host_status)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_status)
        (void)cudaFreeHost(slot.host_status);
      if (slot.host_forward_status)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_forward_status)
        (void)cudaFreeHost(slot.host_forward_status);
      if (slot.host_production_status)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_production_status)
        (void)cudaFreeHost(slot.host_production_status);
      if (slot.host_forward_progress)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_forward_progress)
        (void)cudaFreeHost(slot.host_forward_progress);
      if (slot.host_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_cancel_epoch)
        (void)cudaFreeHost(slot.host_cancel_epoch);
      if (slot.host_socket_counts)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_socket_counts)
        (void)cudaFreeHost(slot.host_socket_counts);
      if (slot.device_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device_cancel_epoch)
        (void)cudaFree(slot.device_cancel_epoch);
      if (slot.device)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device)
        (void)cudaFree(slot.device);
#if WEFTMOE_DEVICE_TIMELINE
      if (slot.forward_timeline)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.forward_timeline)
        (void)cudaFree(slot.forward_timeline);
#endif
    }
    if (cancel_stream)
      (void)cudaStreamDestroy(cancel_stream);
  }
};

struct SlotLease {
  std::shared_ptr<ActivationPoolState> pool;
  std::size_t index = 0;
  std::uint64_t epoch = 0;
  bool quarantined = false;
  bool forward_pending = false;
  bool backward_pending = false;

  ~SlotLease() noexcept {
    if (!pool || index >= pool->slots.size())
      return;
    auto &slot = pool->slots[index];
    bool complete = true;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    if (slot.completion_recorded && slot.complete)
      complete = cudaEventSynchronize(slot.complete) == cudaSuccess;
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (slot.epoch != epoch)
      return;
    slot.state = (!complete || quarantined) ? SlotState::Quarantined
                                            : SlotState::Available;
    slot.completion_recorded = false;
  }
};

struct SlotFailureGuard {
  std::shared_ptr<SlotLease> lease;
  bool committed = false;
  ~SlotFailureGuard() noexcept {
    if (lease && !committed)
      lease->quarantined = true;
  }
};

std::shared_ptr<SlotLease>
acquire_slot(const std::shared_ptr<ActivationPoolState> &pool,
             std::uint64_t generation, std::uint32_t rank) {
  std::lock_guard<std::mutex> lock(pool->mutex);
  const auto i = static_cast<std::size_t>(
      generation % static_cast<std::uint64_t>(pool->slots.size()));
  {
    auto &slot = pool->slots[i];
    if (slot.state == SlotState::Available) {
      slot.state = SlotState::InUse;
      slot.epoch = ++pool->epoch_high;
      slot.completion_recorded = false;
      auto lease = std::make_shared<SlotLease>();
      lease->pool = pool;
      lease->index = i;
      lease->epoch = slot.epoch;
      return lease;
    }
  }
  throw NotifyDispatchError(
      NotifyErrorCode::ResourceExhausted,
      "phase=activation-pool-acquire, field=slots, rank=" +
          std::to_string(rank) + ", expected=available deterministic slot=" +
          std::to_string(i) + ", actual=busy/quarantined");
}
#include "cuda/forward/support/fallback_kernels.cuh"
#include "cuda/forward/support/fixture_kernels.cuh"
#include "cuda/forward/support/transport_kernels.cuh"
