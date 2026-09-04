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

void throw_cublas(const char *phase, cublasStatus_t status,
                  std::uint32_t rank, const char *operation) {
  if (status == CUBLAS_STATUS_SUCCESS)
    return;
  transport_error(NotifyErrorCode::CudaFailure, phase, operation, rank,
                  "CUBLAS_STATUS_SUCCESS",
                  std::to_string(static_cast<unsigned int>(status)));
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b,
                          const char *field) {
  if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "uint64 product", "overflow");
  return a * b;
}

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b,
                          const char *field) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a)
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "transport-pool-query", field, 0, "uint64 sum",
                    "overflow");
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
  const auto required_bytes = checked_size(
      checked_mul(required_elements, element_bytes, field), field);
  const auto capacity_bytes = checked_size(
      checked_mul(capacity_elements, element_bytes, field), field);
  if (required_bytes == 0)
    return;
  if (!pointer)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-cuda-producer", field, 0,
                    "non-null device allocation", "null");
  cudaPointerAttributes attributes{};
  internal::throw_cuda("external-cuda-producer",
                       cudaPointerGetAttributes(&attributes, pointer), 0, 1,
                       "external", 1, ExchangeKind::ExpertCounts,
                       "cudaPointerGetAttributes");
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != static_cast<int>(expected_device))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "external-cuda-producer", field, 0,
                    "device allocation on ordinal=" +
                        std::to_string(expected_device),
                    "wrong memory/device");
  CUdeviceptr base = 0;
  std::size_t allocation_bytes = 0;
  if (cuMemGetAddressRange(&base, &allocation_bytes,
                           reinterpret_cast<CUdeviceptr>(pointer)) !=
      CUDA_SUCCESS)
    transport_error(NotifyErrorCode::CudaFailure,
                    "external-cuda-producer", field, 0,
                    "queryable CUDA allocation",
                    "cuMemGetAddressRange failure");
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto allocation_begin = static_cast<std::uintptr_t>(base);
  if (begin < allocation_begin ||
      begin - allocation_begin > allocation_bytes ||
      capacity_bytes > allocation_bytes - (begin - allocation_begin))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "external-cuda-producer", field, 0,
                    "declared capacity within allocation", "short range");
}

std::size_t append(std::size_t &cursor, std::uint64_t count,
                   std::size_t width, const char *field) {
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
                              std::uint64_t outer,
                              std::uint32_t box_outer,
                              const char *field,
                              std::uint32_t box_inner = 64U) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX || box_outer == 0 || box_inner == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-tma-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
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
    transport_error(NotifyErrorCode::CudaFailure, "forward-tma-map", field,
                    0, "cuTensorMapEncodeTiled success",
                    text ? text : "unknown CUDA driver error");
  }
  return map;
}

CUtensorMap make_bf16_tma_store_map(void *base, std::uint64_t inner,
                                    std::uint64_t outer,
                                    const char *field,
                                    std::uint64_t row_stride_bytes = 0ULL) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX)
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
  const cuuint32_t box[2] = {64U, 128U};
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
                                 std::uint64_t outer, const char *field) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "backward-tma-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
  // MN-major WGMMA consumes a row-major [K,N] operand as 64 contiguous N
  // columns by one 64-row K slice. Wider N tiles assemble adjacent atoms.
  const cuuint32_t box[2] = {64U, 64U};
  const cuuint32_t element_strides[2] = {1U, 1U};
  const auto result = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (result != CUDA_SUCCESS) {
    const char *text = nullptr;
    (void)cuGetErrorString(result, &text);
    transport_error(NotifyErrorCode::CudaFailure, "backward-tma-map", field,
                    0, "cuTensorMapEncodeTiled success",
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
  std::uint64_t sequence;
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

// Forward Return uses immutable, generation-local compact chunks. A chunk owns a
// contiguous range in both the registered staging window and the source
// rank's return receive window.  Its rows are selected from the canonical
// reverse descriptors; FC2 completion order never changes this identity.
struct ForwardReturnChunkRecord {
  std::uint64_t row_begin;
  std::uint32_t row_count;
  std::uint32_t source_node;
  std::uint32_t context;
  std::uint32_t expected_fragments;
  std::uint64_t payload_bytes;
};
static_assert(sizeof(ForwardReturnChunkRecord) == 32);

// Compute publishes only immutable byte ranges.  The fixed RDMA team scans
// this generation-local catalog and the context named by each record is its
// sole GIN issuer.  `epoch` is stored last with device-release ordering.
struct alignas(16) ForwardDirectReturnFragment {
  std::uint64_t canonical_row;
  std::uint64_t compact_row;
  std::uint32_t source_rank;
  std::uint32_t row_count;
  std::uint32_t context;
  std::uint32_t reserved;
  std::uint64_t epoch;
};
static_assert(sizeof(ForwardDirectReturnFragment) == 48);

// One immutable word lets an FC2 producer address both the proxy's chunk
// scoreboard and the exact compact row in its registered Return window.  Main
// shapes need only hundreds of chunks and tens of thousands of compact rows;
// 24 bits for each preserves ample headroom while retaining a 16-bit rank.
constexpr std::uint32_t kReturnMappingFieldBits = 24U;
constexpr std::uint32_t kReturnMappingChunkShift = 24U;
constexpr std::uint32_t kReturnMappingProxyShift = 48U;
constexpr unsigned long long kReturnMappingFieldMask =
    (1ULL << kReturnMappingFieldBits) - 1ULL;

__device__ __forceinline__ std::uint64_t
encode_return_row_mapping(std::uint32_t proxy_rank, std::uint64_t chunk,
                          std::uint64_t compact_row) {
  return (static_cast<std::uint64_t>(proxy_rank + 1U)
          << kReturnMappingProxyShift) |
         ((chunk + 1ULL) << kReturnMappingChunkShift) | (compact_row + 1ULL);
}

__device__ __forceinline__ std::uint64_t
return_mapping_chunk(std::uint64_t mapping) {
  return ((mapping >> kReturnMappingChunkShift) & kReturnMappingFieldMask) -
         1ULL;
}

__device__ __forceinline__ std::uint64_t
return_mapping_compact_row(std::uint64_t mapping) {
  return (mapping & kReturnMappingFieldMask) - 1ULL;
}

[[maybe_unused]] constexpr std::uint32_t kForwardReturnCompletionBatch = 8U;
[[maybe_unused]] constexpr unsigned long long kReturnChunkWaiting = 0ULL;
[[maybe_unused]] constexpr unsigned long long kReturnChunkPacking = 1ULL;
[[maybe_unused]] constexpr unsigned long long kReturnChunkReady = 2ULL;
[[maybe_unused]] constexpr unsigned long long kReturnChunkInflight = 3ULL;
[[maybe_unused]] constexpr unsigned long long kReturnChunkComplete = 4ULL;

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

#if MEGAMOE_DEBUG_VALIDATE || MEGAMOE_FORWARD_LIGHT_PROGRESS
void dump_generation_completion(std::ostream &output,
                                const std::uint64_t *mapped,
                                std::uint32_t contexts,
                                const char *prefix) {
  const auto word = [&](std::size_t index) {
    return __atomic_load_n(mapped + index, __ATOMIC_ACQUIRE);
  };
  output << ' ' << prefix << ".generation=" << word(0) << ' ' << prefix
         << ".epoch=" << word(1);
  static constexpr const char *kTaskNames[] = {
      "dispatch", "relay", "fc1", "fc2", "return",
      "arrival", "reduce", "reserved0", "reserved1"};
  for (std::size_t task = 0; task < transport_state::kTaskClassCount; ++task) {
    const auto base = 2ULL + task * 4ULL;
    output << ' ' << prefix << '.' << kTaskNames[task] << "={expected="
           << word(base) << ",claimed=" << word(base + 1ULL)
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
         << ",compute=" << word(kTail + 9ULL)
         << ",grid=" << word(kTail + 10ULL) << "} " << prefix
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
#if MEGAMOE_DEBUG_VALIDATE
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
  // Optional training-only planes. A/B uses two plane-major BF16 matrices;
  // Forward output is separate so canonical post-Dispatch X stays immutable.
  std::size_t forward_ab_cache = 0;
  std::size_t forward_training_output = 0;
  // The immutable compact X/A/B checkpoint stays intact across repeated
  // Backward executions. Its compute directory pads every expert independently
  // to 128 rows, so all padded tensors use a capacity larger than compact
  // Forward storage by at most 127 rows per owned expert. dX remains disjoint
  // from dR so its M-blocks may return while dW2 still consumes dR.
  std::uint64_t backward_padded_capacity_rows = 0;
  std::size_t backward_padded_x = 0;
  std::size_t backward_padded_ab = 0;
  std::size_t backward_padded_dr = 0;
  std::size_t backward_u = 0;
  std::size_t backward_dab_rows = 0;
  std::size_t backward_dx_rows = 0;
  std::size_t backward_dgate_rows = 0;
  std::size_t backward_dgate_output = 0;
  std::size_t backward_dw1 = 0;
  std::size_t backward_dw2 = 0;
  std::size_t backward_setup_epoch = 0;
  std::size_t backward_dispatch_ready = 0;
  std::size_t backward_return_ready = 0;
  std::size_t backward_compact_to_padded = 0;
  std::size_t backward_padded_expert_rows = 0;
  std::size_t backward_padded_expert_offsets = 0;
  std::size_t forward_task_claim = 0;
  std::size_t forward_fc1_done = 0;
  std::size_t forward_fc1_published = 0;
  std::size_t forward_compute_pair_mailboxes = 0;
  std::size_t forward_x_arrived_rows = 0;
  std::size_t forward_output_ready_epoch = 0;
  std::size_t forward_return_chunk_directory = 0;
  std::size_t forward_return_chunk_state = 0;
  std::size_t forward_return_chunk_ready_count = 0;
  std::size_t forward_return_fragment_index = 0;
  std::size_t forward_return_source_compact_index = 0;
  std::size_t forward_return_source_route_index = 0;
  std::size_t forward_direct_source_compact_index = 0;
  std::size_t forward_reduce_route_count = 0;
  std::size_t forward_reduce_ready_token = 0;
  std::size_t forward_direct_return_fragment = 0;
  std::uint64_t forward_return_chunk_capacity = 0;
  std::uint64_t forward_direct_return_fragment_capacity = 0;
  std::size_t forward_scheduler = 0;
  std::size_t forward_v3_control = 0;
  std::size_t forward_completion = 0;
  std::size_t forward_context_completion = 0;
  std::size_t status = 0;
  std::size_t total = 0;
};

// Only these bytes participate in remote GIN registration. Compute arenas,
// weights, canonical rows, and local-P2P storage stay in ordinary CUDA
// allocations so enabling the network backend does not change their allocator.
struct GinWindowLayout {
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
  std::size_t return_fragment_ready = 0;
  // FWD addresses these planes densely as BF16.  BWD reuses their larger
  // allocation as aligned [dX BF16..., dGate FP32] row records so one FULL
  // put carries both gradients and the completion signal.
  std::uint64_t backward_return_row_bytes = 0;
  std::size_t barrier = 0;
  std::size_t production_error_records = 0;
  std::size_t total = 0;
};

#if MEGAMOE_DEVICE_TIMELINE
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

GinWindowLayout
make_gin_window_layout(const ActivationTransportPoolConfig &config) {
  GinWindowLayout layout;
  if (!config.enable_gin_rdma)
    return layout;
  std::size_t cursor = 0;
  const auto routes = checked_mul(config.max_tokens_per_rank, config.max_top_k,
                                  "GIN max routes");
  const auto outgoing_packets =
      checked_mul(config.max_nodes, config.max_tokens_per_rank,
                  "GIN outgoing packet capacity");
  const auto outgoing_fanout =
      checked_mul(outgoing_packets, config.max_top_k,
                  "GIN outgoing packet fanout capacity");
  const auto remote_nodes = config.max_nodes > 0 ? config.max_nodes - 1U : 0U;
  constexpr std::uint32_t kGinContextsPerRail = 8;
  const auto directory_records = checked_mul(remote_nodes, kGinContextsPerRail,
                                             "GIN dispatch directory records");
  const auto incoming_packets = checked_mul(
      remote_nodes, config.max_tokens_per_rank, "GIN incoming packet capacity");
  const auto packet_payload = checked_mul(
      outgoing_packets, config.max_hidden_size, "GIN send payload elements");
  const auto incoming_payload = checked_mul(
      incoming_packets, config.max_hidden_size, "GIN receive payload elements");
  // FULL's send plane is indexed by canonical producer row and its receive
  // plane by compact source row. Backward reuses the send plane for padded
  // M128 dX rows, so reserve the largest index domain without a pre-put pack.
  const auto backward_padded_rows =
      config.enable_backward_state
          ? checked_add(
                config.max_destination_rows_per_rank,
                checked_mul(config.max_owned_experts, 127ULL,
                            "GIN Backward padded return rows"),
                "GIN Backward padded return rows")
          : 0ULL;
  const auto direct_return_rows = std::max<std::uint64_t>(
      std::max<std::uint64_t>(
          checked_mul(remote_nodes, routes, "GIN compact return rows"),
          config.max_destination_rows_per_rank),
      backward_padded_rows);
  const auto dense_return_row_bytes = checked_mul(
      config.max_hidden_size, sizeof(transport::BFloat16),
      "GIN return row bytes");
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
  layout.send_fanout = append(cursor, outgoing_fanout,
                              sizeof(ForwardDispatchFanoutRecord),
                              "GIN send Forward fanout");
  layout.receive_fanout =
      append(cursor,
             checked_mul(incoming_packets, config.max_top_k,
                         "GIN receive fanout capacity"),
             sizeof(ForwardDispatchFanoutRecord),
             "GIN receive Forward fanout");
  layout.send_payload = append(cursor, packet_payload,
                               sizeof(transport::BFloat16), "GIN send payload");
  layout.receive_payload =
      append(cursor, incoming_payload, sizeof(transport::BFloat16),
             "GIN receive payload");
  layout.return_staging = append(
      cursor, direct_return_rows,
      checked_size(backward_return_row_bytes,
                   "GIN direct return staging row bytes"),
      "GIN return staging");
  layout.return_receive =
      append(cursor, routes,
             checked_size(backward_return_row_bytes,
                          "GIN return receive row bytes"),
             "GIN return receive");
  // The worst legal fragmentation begins at every row, hence routes marker
  // words are sufficient for every compact Return layout.  The main path
  // normally emits one marker per M128 producer block.
  layout.return_fragment_ready =
      append(cursor, routes, sizeof(std::uint64_t),
             "GIN return fragment ready counters");
  layout.barrier = append(cursor, 1, sizeof(std::uint64_t), "GIN barrier");
  layout.production_error_records =
      append(cursor, config.max_world_size, sizeof(ProductionStatusRecord),
             "GIN production error records");
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
                    "transport-pool-query", "capacity", 0,
                    "all capacities > 0", "zero field");
  if (config.max_intermediate_size != 0 &&
      (config.max_hidden_size != 7168 ||
       config.max_intermediate_size != 3072))
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-query", "Forward shape", 0,
                    "hidden=7168 and intermediate=3072",
                    std::to_string(config.max_hidden_size) + "x" +
                        std::to_string(config.max_intermediate_size));
  Layout layout;
  std::size_t cursor = 0;
  const auto routes = checked_mul(config.max_tokens_per_rank, config.max_top_k,
                                  "max routes");
  const auto links = checked_mul(config.max_nodes, config.channels, "links");
  const auto ring_slots = checked_mul(
      checked_mul(config.max_nodes, config.channels, "ring links"),
      config.ring_slots_per_source_channel, "ring slots");
  const auto rank_links = checked_mul(config.max_world_size, config.channels,
                                      "rank links");
  const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "ingress slots");
  layout.input = append(cursor,
                        checked_mul(config.max_tokens_per_rank,
                                    config.max_hidden_size, "input elements"),
                        sizeof(transport::BFloat16), "input");
  layout.ring_headers = append(cursor, ring_slots,
                               sizeof(transport::TransportSlotHeader),
                               "ring headers");
  layout.ring_payload = append(
      cursor, checked_mul(ring_slots, config.max_hidden_size,
                          "ring payload elements"),
      sizeof(transport::BFloat16), "ring payload");
  layout.ring_ready_sequence =
      append(cursor, ring_slots, sizeof(std::uint64_t), "ring ready sequence");
  layout.link_next_sequence =
      append(cursor, links, sizeof(std::uint64_t), "link next sequence");
  layout.link_credits =
      append(cursor, links, sizeof(std::uint32_t), "link credits");
  layout.ingress_headers = append(cursor, ingress_slots,
                                  sizeof(LocalP2pRouteHeader),
                                  "P2P ingress headers");
  layout.ingress_ready_sequence = append(
      cursor, ingress_slots, sizeof(std::uint64_t), "P2P ingress ready");
  layout.ingress_next_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P ingress next");
  layout.ingress_published_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P ingress published");
  layout.ingress_credits = append(cursor, rank_links, sizeof(std::uint32_t),
                                  "P2P ingress credits");
  layout.outgoing_route_sequence = append(
      cursor, rank_links, sizeof(std::uint64_t), "P2P outgoing sequence");
  layout.direct_route_headers = append(
      cursor, config.max_destination_rows_per_rank,
      sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  layout.return_ready_epoch = append(cursor, routes, sizeof(std::uint64_t),
                                     "P2P combine ready epoch");
  // Two pool-private pointer slots follow the world-rank table. They carry the
  // Forward completion-vector pointers without widening the kernel args.
  layout.peer_bases = append(cursor, config.max_world_size + 2ULL,
                             sizeof(void *), "P2P peer base table");
  layout.p2p_barrier_send =
      append(cursor, 2, sizeof(std::uint64_t), "P2P barrier send");
  layout.p2p_barrier_receive = append(cursor,
                                      checked_mul(config.max_world_size, 2,
                                                  "P2P barrier words"),
                                      sizeof(std::uint64_t),
                                      "P2P barrier receive");
  layout.gin_local_compute_ready = append(
      cursor, config.max_world_size, sizeof(std::uint64_t),
      "GIN node-local compute ready epoch");
  layout.terminal_epoch = append(cursor, config.max_world_size,
                                 sizeof(std::uint64_t),
                                 "P2P terminal generation epoch");
  layout.production_error_records = append(
      cursor, config.max_world_size, sizeof(ProductionStatusRecord),
      "peer-visible production error records");
  layout.production_status_matrix = append(
      cursor, checked_mul(config.max_world_size, 8ULL,
                          "production terminal status words"),
      sizeof(std::uint64_t), "production terminal status matrix");
  layout.packet_claim = append(cursor, routes, sizeof(std::uint32_t),
                               "packet claim bitmap");
  layout.canonical_rows = append(
      cursor, checked_mul(config.max_destination_rows_per_rank,
                          config.max_hidden_size, "canonical rows"),
      sizeof(transport::BFloat16), "canonical rows");
  layout.canonical_gates =
      append(cursor, config.max_destination_rows_per_rank, sizeof(float),
             "canonical route gates");
  const auto route_elements =
      checked_mul(routes, config.max_hidden_size, "route elements");
  layout.contribution = append(cursor, route_elements,
                               sizeof(transport::BFloat16), "contribution");
  layout.combine_return = append(cursor, route_elements,
                                 sizeof(transport::BFloat16), "combine return");
  layout.combined = append(
      cursor, checked_mul(config.max_tokens_per_rank, config.max_hidden_size,
                          "combined elements"),
      sizeof(transport::BFloat16), "combined");
  if (config.enable_socket_fallback || config.enable_gin_rdma) {
    layout.socket_count_matrix = append(
        cursor, checked_mul(config.max_world_size, config.max_world_size,
                            "Socket count matrix"),
        sizeof(std::uint64_t), "Socket count matrix");
    layout.socket_header_send = append(
        cursor, routes, sizeof(SocketRouteHeader), "Socket send headers");
    layout.socket_header_receive = append(
        cursor, config.max_destination_rows_per_rank,
        sizeof(SocketRouteHeader), "Socket receive headers");
    layout.socket_payload_receive = append(
        cursor, checked_mul(config.max_destination_rows_per_rank,
                            config.max_hidden_size,
                            "Socket receive payload elements"),
        sizeof(transport::BFloat16), "Socket receive payload");
  }
  if (config.max_intermediate_size != 0) {
    const auto forward_elements = checked_mul(
        config.max_destination_rows_per_rank,
        config.max_intermediate_size, "forward gated U elements");
    layout.forward_s = append(cursor, forward_elements,
                              sizeof(transport::BFloat16), "forward gated U");
    if (config.enable_backward_state) {
      layout.forward_ab_cache = append(
          cursor, checked_mul(forward_elements, 2ULL,
                              "forward BF16 A/B checkpoint elements"),
          sizeof(transport::BFloat16), "forward BF16 A/B checkpoints");
      layout.forward_training_output = append(
          cursor,
          checked_mul(config.max_destination_rows_per_rank,
                      config.max_hidden_size,
                      "forward training output elements"),
          sizeof(transport::BFloat16), "forward training output");
      layout.backward_padded_capacity_rows = checked_add(
          config.max_destination_rows_per_rank,
          checked_mul(config.max_owned_experts, 127ULL,
                      "backward expert padding rows"),
          "backward padded row capacity");
      const auto backward_padded_elements = checked_mul(
          layout.backward_padded_capacity_rows,
          config.max_intermediate_size, "backward padded elements");
      layout.backward_padded_x = append(
          cursor,
          checked_mul(layout.backward_padded_capacity_rows,
                      config.max_hidden_size, "backward padded X elements"),
          sizeof(transport::BFloat16), "backward padded X");
      layout.backward_padded_ab = append(
          cursor, checked_mul(backward_padded_elements, 2ULL,
                              "backward padded A/B elements"),
          sizeof(transport::BFloat16), "backward padded A/B");
      layout.backward_padded_dr = append(
          cursor,
          checked_mul(layout.backward_padded_capacity_rows,
                      config.max_hidden_size, "backward padded dR elements"),
          sizeof(transport::BFloat16), "backward padded dR");
      layout.backward_u = append(
          cursor, backward_padded_elements, sizeof(transport::BFloat16),
          "backward padded U");
      layout.backward_dab_rows = append(
          cursor, checked_mul(backward_padded_elements, 2ULL,
                              "backward interleaved dAB elements"),
          sizeof(transport::BFloat16), "backward interleaved dAB");
      layout.backward_dx_rows = append(
          cursor,
          checked_mul(layout.backward_padded_capacity_rows,
                      config.max_hidden_size, "backward dX rows"),
          sizeof(transport::BFloat16), "backward dX rows");
      layout.backward_dgate_rows = append(
          cursor, layout.backward_padded_capacity_rows, sizeof(float),
          "backward canonical dGate rows");
      layout.backward_dgate_output = append(
          cursor, routes, sizeof(float), "backward source dGate");
      if (config.max_owned_experts != 0) {
        const auto weight_matrix_elements = checked_mul(
            checked_mul(config.max_owned_experts,
                        config.max_intermediate_size,
                        "backward expert weight rows"),
            config.max_hidden_size, "backward expert weight elements");
        layout.backward_dw1 = append(
            cursor, checked_mul(2ULL, weight_matrix_elements,
                                "backward dW1 elements"),
            sizeof(transport::BFloat16), "backward dW1");
        layout.backward_dw2 = append(
            cursor, weight_matrix_elements, sizeof(transport::BFloat16),
            "backward dW2");
      }
      layout.backward_setup_epoch = append(
          cursor, config.max_world_size, sizeof(std::uint64_t),
          "backward setup epochs");
      layout.backward_dispatch_ready = append(
          cursor, config.max_destination_rows_per_rank,
          sizeof(std::uint64_t), "backward dR row-ready epochs");
      layout.backward_return_ready = append(
          cursor, routes, sizeof(std::uint64_t),
          "backward dX route-ready epochs");
      layout.backward_compact_to_padded = append(
          cursor, config.max_destination_rows_per_rank,
          sizeof(std::uint32_t), "backward compact-to-padded rows");
      layout.backward_padded_expert_rows = append(
          cursor, config.max_owned_experts, sizeof(std::uint64_t),
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
    const auto fc2_ntiles =
        (config.max_hidden_size + forward_tuning::kFc2TileN - 1ULL) /
        forward_tuning::kFc2TileN;
    const auto forward_claim_words = checked_mul(
        max_mblocks, 2ULL + fc1_ntiles + fc2_ntiles,
        "forward task claims");
    auto task_claim_words = forward_claim_words;
    if (config.enable_backward_state) {
      const auto backward_mblocks =
          (layout.backward_padded_capacity_rows +
           backward_tuning::kTileM - 1ULL) /
          backward_tuning::kTileM;
      const auto backward_scheduler =
          persistent_v3::backward::make_backward_scheduler_layout(
              backward_mblocks,
              static_cast<std::uint32_t>(config.max_hidden_size /
                                         backward_tuning::kDxTileN),
              config.max_owned_experts);
      if (backward_scheduler.total_words > task_claim_words)
        task_claim_words = backward_scheduler.total_words;
    }
    layout.forward_task_claim =
        append(cursor, task_claim_words, sizeof(std::uint64_t),
               "forward/backward aliased task claims");
    layout.forward_fc1_done = append(cursor, max_mblocks,
                                     sizeof(std::uint64_t),
                                     "forward FC1 done tiles");
    layout.forward_fc1_published = append(
        cursor, max_mblocks, sizeof(std::uint64_t),
        "forward FC1 readiness publication bitmap");
    layout.forward_compute_pair_mailboxes = append(
        cursor, persistent_v3::kForwardComputePairCapacity,
        sizeof(persistent_v3::ForwardComputePairMailbox),
        "forward compute pair mailboxes");
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
    layout.forward_output_ready_epoch =
        append(cursor, config.max_destination_rows_per_rank,
               sizeof(std::uint64_t), "forward output row ready epochs");
    // A discontinuity in the canonical reverse-row space may terminate a
    // chunk.  The safe O(routes) bound is therefore one directory record per
    // destination row rather than ceil(rows/chunk_rows).
    layout.forward_return_chunk_capacity = config.max_destination_rows_per_rank;
    layout.forward_return_chunk_directory = append(
        cursor, layout.forward_return_chunk_capacity,
        sizeof(ForwardReturnChunkRecord), "forward return chunk directory");
    layout.forward_return_chunk_state =
        append(cursor, layout.forward_return_chunk_capacity,
               sizeof(std::uint64_t), "forward return chunk state");
    layout.forward_return_chunk_ready_count = append(
        cursor,
        checked_mul(layout.forward_return_chunk_capacity, config.max_world_size,
                    "forward return chunk producer rail counters"),
        sizeof(std::uint64_t), "forward return chunk ready count");
    layout.forward_return_fragment_index =
        append(cursor, config.max_destination_rows_per_rank,
               sizeof(std::uint64_t), "forward return compact fragment index");
    layout.forward_return_source_compact_index =
        append(cursor, routes, sizeof(std::uint64_t),
               "forward source compact return index");
    layout.forward_return_source_route_index =
        append(cursor, routes, sizeof(std::uint64_t),
               "forward compact return source route index");
    layout.forward_direct_source_compact_index =
        append(cursor, routes, sizeof(std::uint64_t),
               "forward FULL source route compact index");
    layout.forward_reduce_route_count = append(
        cursor, config.max_tokens_per_rank, sizeof(std::uint32_t),
        "forward reduce ready route counts");
    layout.forward_reduce_ready_token = append(
        cursor, config.max_tokens_per_rank, sizeof(std::uint64_t),
        "forward reduce ready token queue");
    layout.forward_direct_return_fragment_capacity =
        config.max_destination_rows_per_rank;
    layout.forward_direct_return_fragment = append(
        cursor, layout.forward_direct_return_fragment_capacity,
        sizeof(ForwardDirectReturnFragment),
        "forward direct return fragment directory");
    layout.forward_scheduler =
        append(cursor, 4 + config.max_world_size * 32ULL, sizeof(std::uint64_t),
               "forward scheduler");
    layout.forward_v3_control =
        append(cursor, 1, sizeof(persistent_v3::ForwardPipelineControl),
               "forward V3 persistent control plane");
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

struct PoolSlot {
  void *device = nullptr;
  void *gin_window = nullptr;
#if MEGAMOE_DEVICE_TIMELINE
  void *forward_timeline = nullptr;
#endif
  TransportDeviceStatus *host_status = nullptr;
  TransportDeviceStatus *host_forward_status = nullptr;
  ProductionStatusRecord *host_production_status = nullptr;
  std::uint64_t *host_forward_progress = nullptr;
  std::uint64_t *device_forward_progress = nullptr;
  std::size_t forward_progress_words = 0;
  std::size_t generation_debug_words = 0;
  std::uint32_t generation_debug_contexts = 0;
  std::uint64_t *host_cancel_epoch = nullptr;
  std::uint64_t *host_socket_counts = nullptr;
  std::uint64_t *device_cancel_epoch = nullptr;
  cudaEvent_t producer_ready = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t forward_start = nullptr;
  cudaEvent_t forward_output_ready = nullptr;
  cudaEvent_t forward_complete = nullptr;
  // GIN Forward uses compile-time-isolated transport and compute kernels.
  // Per-slot non-blocking streams allow both persistent grids to remain
  // resident without serializing independent generations in the pool.
  cudaStream_t forward_transport_stream = nullptr;
  cudaStream_t forward_m64_head_stream = nullptr;
  cudaStream_t forward_compute_stream = nullptr;
  cudaEvent_t forward_roles_ready = nullptr;
  cudaEvent_t forward_transport_done = nullptr;
  cudaEvent_t forward_m64_head_done = nullptr;
  cudaEvent_t forward_compute_done = nullptr;
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
  std::uint64_t backward_epoch = 0;
  bool completion_recorded = false;
  CUtensorMap forward_x_map{};
  CUtensorMap forward_s_map{};
  CUtensorMap forward_x_m64_map{};
  CUtensorMap forward_s_m64_map{};
  CUtensorMap forward_output_store_map{};
  CUtensorMap forward_training_output_store_map{};
  CUtensorMap forward_staging_store_map{};
  bool forward_maps_valid = false;
  CUtensorMap backward_dr_map{};
  CUtensorMap backward_dab_map{};
  CUtensorMap backward_dr_mn_map{};
  CUtensorMap backward_u_mn_map{};
  CUtensorMap backward_dab_mn_map{};
  CUtensorMap backward_x_mn_map{};
  CUtensorMap backward_u_store_map{};
  CUtensorMap backward_dx_store_map{};
  CUtensorMap backward_dw2_store_map{};
  CUtensorMap backward_dw1_store_map{};
  bool backward_maps_valid = false;
};

template <typename T>
T *at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}
template <typename T>
const T *at(const void *base, std::size_t offset) {
  return reinterpret_cast<const T *>(
      static_cast<const unsigned char *>(base) + offset);
}

#if MEGAMOE_DEVICE_TIMELINE
profiling::DeviceTimelineContext make_timeline_context(
    void *allocation, const TimelineAllocationLayout &layout,
    std::uint32_t max_ctas, std::uint32_t rank,
    profiling::TimelinePhase phase, std::uint64_t generation) {
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
                    std::uint64_t generation,
                    const NcclSession &session) {
  const auto reset_bytes = checked_size(
      checked_mul(max_ctas, sizeof(std::uint32_t), "timeline count reset"),
      "timeline count reset");
  internal::throw_cuda(
      phase,
      cudaMemsetAsync(at<void>(allocation, layout.counts), 0, reset_bytes,
                      stream),
      rank, world, session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "reset device timeline counts");
  internal::throw_cuda(
      phase,
      cudaMemsetAsync(at<void>(allocation, layout.overflow), 0, reset_bytes,
                      stream),
      rank, world, session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "reset device timeline overflow");
}
#endif

struct ActivationPoolState {
  ActivationTransportPoolConfig config;
  Layout layout;
  GinWindowLayout gin_layout;
#if MEGAMOE_DEVICE_TIMELINE
  TimelineAllocationLayout timeline_layout;
  std::uint32_t timeline_max_ctas = 0;
#endif
  std::vector<std::byte> session_identity;
  std::uint32_t world = 0;
  CudaExecution execution;
  std::uint32_t device = 0;
  std::mutex mutex;
  std::weak_ptr<ActivationMegaKernelBackendState> gin_backend;
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
      if (slot.forward_roles_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_roles_ready)
        (void)cudaEventDestroy(slot.forward_roles_ready);
      if (slot.forward_transport_done)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_transport_done)
        (void)cudaEventDestroy(slot.forward_transport_done);
      if (slot.forward_m64_head_done)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_m64_head_done)
        (void)cudaEventDestroy(slot.forward_m64_head_done);
      if (slot.forward_compute_done)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.forward_compute_done)
        (void)cudaEventDestroy(slot.forward_compute_done);
      const cudaEvent_t backward_events[] = {
          slot.backward_start,         slot.backward_compute_start,
          slot.backward_compute_done,  slot.backward_transport_done,
          slot.backward_reduce_done,   slot.backward_late_compute_done,
          slot.backward_complete};
      for (const auto event : backward_events) {
        if (event)
          internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
        if (event)
          (void)cudaEventDestroy(event);
      }
      if (slot.forward_transport_stream)
        (void)cudaStreamDestroy(slot.forward_transport_stream);
      if (slot.forward_m64_head_stream)
        (void)cudaStreamDestroy(slot.forward_m64_head_stream);
      if (slot.forward_compute_stream)
        (void)cudaStreamDestroy(slot.forward_compute_stream);
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
#if MEGAMOE_DEVICE_TIMELINE
      if (slot.forward_timeline)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.forward_timeline)
        (void)cudaFree(slot.forward_timeline);
#endif
      if (slot.gin_window) {
        using MemFreeFn = ncclResult_t (*)(void *);
        const auto mem_free = reinterpret_cast<MemFreeFn>(
            dlsym(RTLD_DEFAULT, "ncclMemFree"));
        if (mem_free != nullptr)
          (void)mem_free(slot.gin_window);
      }
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

std::shared_ptr<SlotLease> acquire_slot(
    const std::shared_ptr<ActivationPoolState> &pool, std::uint64_t generation,
    std::uint32_t rank) {
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

__device__ std::uint16_t float_to_bf16_bits(float value) {
  auto bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ std::uint32_t float2_to_bf16x2_bits(float low,
                                                               float high) {
  const auto pair = static_cast<__nv_bfloat162_raw>(
      __float22bfloat162_rn(make_float2(low, high)));
  return static_cast<std::uint32_t>(pair.x) |
         (static_cast<std::uint32_t>(pair.y) << 16U);
}

__device__ __forceinline__ float2
bf16x2_bits_to_float2(std::uint32_t pair) {
  // A promoted BF16 is its 16 bits in the high half of an FP32 word. Decode
  // both packed lanes together so the FC1 epilogue needs only one shift and
  // one mask per pair instead of separately extracting two uint16_t values.
  return make_float2(__uint_as_float(pair << 16U),
                     __uint_as_float(pair & 0xffff0000U));
}

__device__ float bf16_bits_to_float(std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ float gate_after_accumulate(float sum, float gate,
                                       std::uint16_t bf16_value) {
  const float product = __fmul_rn(gate, bf16_bits_to_float(bf16_value));
  return __fadd_rn(sum, product);
}

__device__ unsigned long long device_globaltimer_ns() {
  unsigned long long value = 0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

template <typename Status>
__device__ void publish_local_structured_error(
    Status *status, std::uint32_t rank, std::uint32_t field,
    std::uint32_t code, unsigned long long expected,
    unsigned long long actual) {
  if (atomicCAS(&status->error, 0U, 1U) != 0U)
    return;
  status->error_origin_rank = rank;
  status->error_field = field;
  status->error_code = code;
  status->error_expected = expected;
  status->error_actual = actual;
  __threadfence_system();
}

__device__ transport::BFloat16 synthetic_value(
    transport::BFloat16 input, std::uint32_t expert,
    std::uint64_t hidden_index, bool identity) {
  if (identity)
    return input;
  const auto expert_term = static_cast<float>((expert % 251U) + 1U) / 128.0F;
  const auto lane_term = static_cast<float>(hidden_index % 17U) / 1024.0F;
  return transport::BFloat16{
      float_to_bf16_bits(bf16_bits_to_float(input.bits) + expert_term +
                         lane_term)};
}

struct TransportKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t max_destination_rows;
  std::uint32_t channels;
  std::uint32_t ring_slots;
  transport::TransportSlotHeader *ring_headers;
  transport::BFloat16 *ring_payload;
  unsigned long long *ring_ready;
  unsigned long long *link_next;
  unsigned int *link_credits;
  unsigned int *packet_claim;
  transport::BFloat16 *canonical_rows;
  transport::BFloat16 *contribution;
  transport::BFloat16 *combine_return;
  transport::BFloat16 *combined;
  TransportDeviceStatus *status;
  bool identity_expert;
};

__global__ void activation_transport_persistent_kernel(TransportKernelArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto route_count = a.dispatch.sizes.route_count;
  const auto packet_count = a.dispatch.sizes.packet_count;
  __shared__ unsigned long long selected_packet;
  __shared__ unsigned long long selected_token;
  __shared__ unsigned long long packet_scan_cursor;
  if (lane == 0)
    atomicCAS(&a.status->start_clock, 0ULL,
              device_globaltimer_ns());
  __syncthreads();

  // Rule-based link ownership: link L belongs to CTA L % gridDim.x. Each CTA
  // scans the canonical descriptor array once and processes every assigned
  // link in sequence order. This is bounded O(grid*packets), has no ready
  // queue, and does not use atomic return order to assign a packet or row.
  if (lane == 0)
    packet_scan_cursor = 0;
  __syncthreads();
  while (atomicAdd(&a.status->error, 0U) == 0U) {
    if (lane == 0) {
      selected_packet = ~0ULL;
      while (packet_scan_cursor < packet_count) {
        const auto candidate = packet_scan_cursor++;
        const auto candidate_header = a.dispatch.packet_headers[candidate];
        const auto candidate_link =
            static_cast<unsigned long long>(candidate_header.destination_node) *
                a.channels +
            candidate_header.channel;
        if (candidate_link % gridDim.x != blockIdx.x)
          continue;
        if (candidate_header.sequence_on_link !=
            atomicAdd(&a.link_next[candidate_link], 0ULL)) {
          a.status->error_packet = static_cast<unsigned int>(candidate);
          a.status->error_field = 2U;
          atomicExch(&a.status->error, 1U);
          break;
        }
        if (atomicCAS(&a.packet_claim[candidate], 0U, 1U) == 0U) {
          atomicAdd(&a.status->packet_claim_cursor, 1ULL);
          selected_packet = candidate;
          break;
        }
      }
    }
    __syncthreads();
    if (selected_packet == ~0ULL)
      break;
    const auto packet = selected_packet;
    const auto header = a.dispatch.packet_headers[packet];
    const auto link = static_cast<unsigned long long>(header.destination_node) *
                          a.channels +
                      header.channel;
    const auto physical =
        link * a.ring_slots + header.sequence_on_link % a.ring_slots;
    const auto gpus_per_node =
        a.dispatch.identity.world_size / a.dispatch.identity.num_nodes;
    const auto source_node =
        a.dispatch.identity.calling_rank / gpus_per_node;
    if (lane == 0) {
      const auto prior = atomicSub(&a.link_credits[link], 1U);
      if (prior == 0U) {
        atomicAdd(&a.link_credits[link], 1U);
        atomicAdd(&a.status->credit_stalls, 1ULL);
        a.status->error_packet = static_cast<unsigned int>(packet);
        a.status->error_field = 3U;
        atomicExch(&a.status->error, 1U);
      }
    }
    __syncthreads();
    if (atomicAdd(&a.status->error, 0U) != 0U)
      break;
    if (lane == 0) {
      a.ring_headers[physical] = transport::TransportSlotHeader{
          transport::kActivationTransportSchemaVersion,
          header.source_rank,
          source_node,
          header.destination_node,
          header.proxy_rank,
          header.channel,
          a.dispatch.identity.generation,
          a.dispatch.identity.generation,
          header.sequence_on_link,
          header.local_token_id,
          a.dispatch.packet_fanout_offsets[packet],
          a.dispatch.packet_fanout_offsets[packet + 1U]};
    }
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      a.ring_payload[physical * a.hidden + h] =
          a.input[header.local_token_id * a.hidden + h];
    }
    __syncthreads();
    if (lane == 0) {
      __threadfence_system();
      atomicExch_system(&a.ring_ready[physical],
                        static_cast<unsigned long long>(
                            header.sequence_on_link + 1U));
      atomicAdd(&a.status->packets_done, 1ULL);
    }
    __syncthreads();
    while (atomicAdd_system(&a.ring_ready[physical], 0ULL) !=
           header.sequence_on_link + 1U) {
    }
    __threadfence_system();
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    if (lane == 0 && header.destination_node != source_node) {
      atomicAdd(&a.status->deduplicated_cross_node_payload_bytes,
                static_cast<unsigned long long>(
                    a.hidden * sizeof(transport::BFloat16)));
      atomicAdd(&a.status->raw_cross_node_payload_bytes,
                static_cast<unsigned long long>(
                    (fanout_end - fanout_begin) * a.hidden *
                    sizeof(transport::BFloat16)));
    }
    for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = header.local_token_id * a.dispatch.identity.top_k +
                         target.topk_slot;
      if (route >= route_count) {
        if (lane == 0) {
          a.status->error_packet = static_cast<unsigned int>(packet);
          a.status->error_field = 1U;
          atomicExch(&a.status->error, 1U);
        }
        break;
      }
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
        const auto payload = a.ring_payload[physical * a.hidden + h];
        if (a.dispatch.route_dst_rank[route] ==
                a.dispatch.identity.calling_rank &&
            target.dst_row < a.max_destination_rows)
          a.canonical_rows[target.dst_row * a.hidden + h] = payload;
        const auto contribution = synthetic_value(
            payload, target.expert_id, h, a.identity_expert);
        a.contribution[route * a.hidden + h] = contribution;
        a.combine_return[route * a.hidden + h] = contribution;
      }
      if (lane == 0) {
        atomicAdd(&a.status->fanout_done, 1ULL);
        atomicAdd(&a.status->synthetic_done, 1ULL);
        atomicAdd(&a.status->combine_done, 1ULL);
      }
      __syncthreads();
    }
    if (lane == 0) {
      __threadfence_system();
      atomicAdd(&a.link_next[link], 1ULL);
      atomicAdd(&a.link_credits[link], 1U);
    }
    __syncthreads();
  }

  // Comm/Relay CTAs transition to the deterministic TopKReduce role only
  // after global packet/fan-out completion counters prove the boundary.
  while ((atomicAdd(&a.status->packets_done, 0ULL) < packet_count ||
          atomicAdd(&a.status->combine_done, 0ULL) < route_count) &&
         atomicAdd(&a.status->error, 0U) == 0U) {
  }
  if (lane == 0)
    atomicMax(&a.status->dispatch_complete_clock,
              device_globaltimer_ns());
  __threadfence();
  while (atomicAdd(&a.status->error, 0U) == 0U) {
    if (lane == 0)
      selected_token = atomicAdd(&a.status->token_claim_cursor, 1ULL);
    __syncthreads();
    const auto token = selected_token;
    if (token >= a.dispatch.identity.num_tokens)
      break;
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto original = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + original;
        sum = gate_after_accumulate(
            sum, a.gates[route],
            a.combine_return[route * a.hidden + h].bits);
      }
      a.combined[token * a.hidden + h] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
    __syncthreads();
    if (lane == 0)
      atomicAdd(&a.status->reduced_tokens, 1ULL);
  }
  if (lane == 0)
    atomicMax(&a.status->complete_clock,
              device_globaltimer_ns());
}

__global__ void socket_route_count_kernel(DeviceDispatchKernelView dispatch,
                                          unsigned long long *counts) {
  for (std::uint64_t route =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       route < dispatch.sizes.route_count;
       route += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    const auto destination = dispatch.route_dst_rank[route];
    if (destination < dispatch.identity.world_size)
      atomicAdd(&counts[destination], 1ULL);
  }
}

__global__ void socket_route_prefix_kernel(const unsigned long long *counts,
                                           unsigned long long *prefix,
                                           std::uint32_t world) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  unsigned long long total = 0;
  for (std::uint32_t rank = 0; rank < world; ++rank) {
    prefix[rank] = total;
    total += counts[rank];
  }
  prefix[world] = total;
}

__global__ void socket_stable_route_ids_kernel(
    DeviceDispatchKernelView dispatch, const unsigned long long *prefix,
    unsigned int *route_ids) {
  const auto destination = static_cast<std::uint32_t>(blockIdx.x);
  if (destination >= dispatch.identity.world_size)
    return;
  __shared__ unsigned int scan[256];
  __shared__ unsigned long long carry;
  if (threadIdx.x == 0)
    carry = 0;
  __syncthreads();
  for (std::uint64_t base = 0; base < dispatch.sizes.route_count;
       base += blockDim.x) {
    const auto route = base + threadIdx.x;
    scan[threadIdx.x] =
        route < dispatch.sizes.route_count &&
                dispatch.route_dst_rank[route] == destination
            ? 1U
            : 0U;
    __syncthreads();
    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1U) {
      const auto add = threadIdx.x >= offset ? scan[threadIdx.x - offset] : 0U;
      __syncthreads();
      scan[threadIdx.x] += add;
      __syncthreads();
    }
    if (route < dispatch.sizes.route_count &&
        dispatch.route_dst_rank[route] == destination) {
      const auto position = prefix[destination] + carry +
                            static_cast<unsigned long long>(scan[threadIdx.x] - 1U);
      route_ids[position] = static_cast<unsigned int>(route);
    }
    __syncthreads();
    if (threadIdx.x == 0)
      carry += scan[blockDim.x - 1U];
    __syncthreads();
  }
}

__global__ void socket_pack_routes_kernel(
    DeviceDispatchKernelView dispatch, const transport::BFloat16 *input,
    const float *gates, std::uint64_t hidden, std::uint64_t epoch,
    const unsigned int *route_ids, SocketRouteHeader *headers,
    transport::BFloat16 *payload) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= dispatch.sizes.route_count)
    return;
  const auto route = static_cast<std::uint64_t>(route_ids[packed]);
  const auto token = route / dispatch.identity.top_k;
  const auto slot = static_cast<std::uint32_t>(route % dispatch.identity.top_k);
  if (threadIdx.x == 0) {
    headers[packed] = SocketRouteHeader{
        LocalP2pRouteHeader{transport::kActivationTransportSchemaVersion,
                            dispatch.identity.calling_rank,
                            dispatch.route_dst_rank[route], 0U,
                            dispatch.identity.generation, epoch, packed, route,
                            token, dispatch.route_dst_row[route],
                            dispatch.expert_id[route], slot},
        gates[route], 0U};
  }
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    payload[packed * hidden + h] = input[token * hidden + h];
}

__global__ void socket_relay_transport_kernel(
    SocketRouteHeader *headers, transport::BFloat16 *compact_payload,
    transport::BFloat16 *canonical_rows, float *canonical_gates,
    std::uint64_t incoming, std::uint64_t hidden, std::uint64_t generation,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    bool identity_expert, TransportDeviceStatus *status) {
  const auto incoming_index = static_cast<std::uint64_t>(blockIdx.x);
  if (incoming_index >= incoming)
    return;
  const auto header = headers[incoming_index];
  const auto route = header.route;
  const bool valid =
      route.schema_version == transport::kActivationTransportSchemaVersion &&
      route.generation == generation &&
      route.destination_rank == destination_rank && route.dst_row < max_rows;
  if (!valid) {
    if (threadIdx.x == 0)
      publish_local_structured_error(status, destination_rank, 40U, 1U,
                                     max_rows, route.dst_row);
    return;
  }
  if (threadIdx.x == 0)
    canonical_gates[route.dst_row] = header.gate;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    const auto value = compact_payload[incoming_index * hidden + h];
    canonical_rows[route.dst_row * hidden + h] = value;
    compact_payload[incoming_index * hidden + h] =
        synthetic_value(value, route.expert_id, h, identity_expert);
  }
}

__global__ void socket_install_forward_headers_kernel(
    const SocketRouteHeader *compact_headers,
    LocalP2pRouteHeader *canonical_headers, std::uint64_t incoming,
    std::uint64_t generation, std::uint64_t epoch,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    TransportDeviceStatus *status) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= incoming)
    return;
  auto header = compact_headers[index].route;
  if (header.schema_version != transport::kActivationTransportSchemaVersion ||
      header.generation != generation ||
      header.destination_rank != destination_rank ||
      header.dst_row >= max_rows) {
    publish_local_structured_error(status, destination_rank, 41U, 1U,
                                   max_rows, header.dst_row);
    return;
  }
  header.source_epoch = epoch;
  canonical_headers[header.dst_row] = header;
}

__global__ void socket_gather_forward_returns_kernel(
    const SocketRouteHeader *headers,
    const transport::BFloat16 *canonical_output,
    transport::BFloat16 *compact, std::uint64_t incoming,
    std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= incoming)
    return;
  const auto row = headers[packed].route.dst_row;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    compact[packed * hidden + h] = canonical_output[row * hidden + h];
}

__global__ void socket_scatter_returns_kernel(
    const SocketRouteHeader *headers,
    const transport::BFloat16 *compact,
    transport::BFloat16 *by_route, std::uint64_t routes,
    std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= routes)
    return;
  const auto route = headers[packed].route.route_index;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    by_route[route * hidden + h] = compact[packed * hidden + h];
}

__global__ void socket_gate_reduce_kernel(
    DeviceDispatchKernelView dispatch, const float *gates,
    const transport::BFloat16 *by_route, transport::BFloat16 *combined,
    std::uint64_t hidden) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= dispatch.identity.num_tokens)
    return;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < dispatch.identity.top_k; ++order) {
      const auto original = dispatch.combine_order[
          token * dispatch.identity.top_k + order];
      const auto route = token * dispatch.identity.top_k + original;
      sum = gate_after_accumulate(sum, gates[route],
                                  by_route[route * hidden + h].bits);
    }
    combined[token * hidden + h] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}

// Full Forward has already applied each route gate to the BF16 FC2 input.
// Its Socket correctness fallback therefore combines returned route vectors
// with the same fixed-order, unweighted FP32 sum as the split CUDA path.
__global__ void socket_sum_forward_routes_kernel(
    DeviceDispatchKernelView dispatch,
    const transport::BFloat16 *by_route,
    transport::BFloat16 *combined, std::uint64_t hidden) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= dispatch.identity.num_tokens)
    return;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < dispatch.identity.top_k; ++order) {
      const auto slot = dispatch.combine_order[
          token * dispatch.identity.top_k + order];
      const auto route = token * dispatch.identity.top_k + slot;
      sum = __fadd_rn(
          sum, bf16_bits_to_float(by_route[route * hidden + h].bits));
    }
    combined[token * hidden + h] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}

__global__ void socket_finalize_status_kernel(TransportDeviceStatus *status,
                                              std::uint64_t routes,
                                              std::uint64_t incoming,
                                              std::uint64_t tokens) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  status->packets_done = routes;
  status->fanout_done = incoming;
  status->synthetic_done = incoming;
  status->combine_done = routes;
  status->incoming_done = incoming;
  status->reduced_tokens = tokens;
  status->complete_clock = device_globaltimer_ns();
}

#if __has_include(<nccl_device.h>)
__global__ void gin_probe_reset_kernel(ncclDevComm_t dev_comm) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const ncclGin gin(dev_comm, 0);
    gin.resetSignal(0);
    gin.resetCounter(0);
  }
}

__global__ void gin_probe_put_signal_kernel(
    ncclDevComm_t dev_comm, ncclWindow_t window, std::size_t payload_bytes,
    std::size_t receive_offset, unsigned int *device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const ncclGin gin(dev_comm, 0);
  const auto rail = ncclTeamRail(dev_comm);
  if (rail.nRanks < 2) {
    *device_error = 1U;
    return;
  }
  const auto destination = (rail.rank + 1) % rail.nRanks;
  gin.put(rail, destination, window, receive_offset, window, 0,
          payload_bytes, ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
          ncclCoopThread{}, ncclGin_None{}, ::cuda::thread_scope_thread,
          ::cuda::thread_scope_device);
  // Counter completion proves the local source region can be reused. The
  // remote indexed signal is ordered after the payload has settled.
  gin.waitCounter(ncclCoopThread{}, 0, 1);
  gin.waitSignal(ncclCoopThread{}, 0, 1);
}

__global__ void gin_link_benchmark_kernel(
    ncclDevComm_t dev_comm, ncclWindow_t window, bool full_connection,
    bool fanout_full, int destination, int local_world,
    std::size_t payload_bytes, std::size_t receive_offset,
    std::uint32_t warmup, std::uint32_t measured,
    std::uint32_t puts_per_sample, unsigned long long *duration_ns,
    unsigned int *device_error) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  const ncclGin gin(dev_comm, 0);
  const ncclTeam team = full_connection ? ncclTeamWorld(dev_comm)
                                        : ncclTeamRail(dev_comm);
  if (destination < 0 || destination >= team.nRanks || measured == 0U ||
      puts_per_sample == 0U) {
    *device_error = 1U;
    return;
  }

  unsigned long long expected = 0ULL;
  const auto samples = static_cast<unsigned long long>(warmup) + measured;
  for (unsigned long long sample = 0ULL; sample < samples; ++sample) {
    const auto begin = device_globaltimer_ns();
    for (std::uint32_t put = 0; put < puts_per_sample; ++put) {
      auto put_destination = destination;
      if (fanout_full) {
        const auto node = destination / local_world;
        const auto local = destination % local_world;
        put_destination =
            node * local_world + (local + static_cast<int>(put)) % local_world;
      }
      gin.put(team, put_destination, window, receive_offset, window, 0,
              payload_bytes, ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
              ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_system, ::cuda::thread_scope_system);
    }
    expected += puts_per_sample;
    gin.waitCounter(ncclCoopThread{}, 0, expected, 64,
                    ::cuda::memory_order_acquire);
    const auto end = device_globaltimer_ns();
    if (sample >= warmup)
      duration_ns[sample - warmup] = end - begin;
  }
  // Every rank has exactly one inverse sender.  Waiting for the matching
  // remote signal makes the final payload validation independent of whether
  // local completion happens a little earlier than the receive-side action.
  gin.waitSignal(ncclCoopThread{}, 0, expected, 64,
                 ::cuda::memory_order_acquire);
}

struct GinTransportSliceArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t epoch;
  ncclDevComm_t dev_comm;
  ncclWindow_t window;
  void *compute_base;
  void *window_base;
  std::size_t send_header_offset;
  std::size_t receive_header_offset;
  std::size_t send_fanout_offset;
  std::size_t receive_fanout_offset;
  std::size_t send_payload_offset;
  std::size_t receive_payload_offset;
  std::size_t canonical_offset;
  std::size_t return_staging_offset;
  std::size_t return_receive_offset;
  std::size_t combine_compute_offset;
  std::size_t combined_offset;
  std::size_t status_offset;
  std::size_t peer_bases_offset;
  std::uint32_t local_world_size;
  std::uint32_t node;
  std::uint64_t max_tokens_per_rank;
  bool identity_expert;
};

__global__ void gin_signal_reset_kernel(ncclDevComm_t dev_comm,
                                           std::uint32_t context_count) {
  if (blockIdx.x < context_count && threadIdx.x == 0) {
    const ncclGin gin(dev_comm, blockIdx.x);
    gin.resetSignal(0);
    gin.resetSignal(1);
    gin.resetSignal(2);
    gin.resetCounter(0);
    gin.resetCounter(1);
  }
}

__global__ void gin_transport_slice_kernel(GinTransportSliceArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  const auto rank = a.dispatch.identity.calling_rank;
  const ncclGin gin(a.dev_comm, 0);
  const auto rail = ncclTeamRail(a.dev_comm);
  auto *bytes = static_cast<std::byte *>(a.window_base);
  auto *compute_bytes = static_cast<std::byte *>(a.compute_base);
  auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.send_header_offset);
  auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
      bytes + a.receive_header_offset);
  auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.send_fanout_offset);
  auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
      bytes + a.receive_fanout_offset);
  auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.send_payload_offset);
  auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.receive_payload_offset);
  auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.return_staging_offset);
  auto *return_receive = reinterpret_cast<transport::BFloat16 *>(
      bytes + a.return_receive_offset);
  auto *combine_compute = reinterpret_cast<transport::BFloat16 *>(
      compute_bytes + a.combine_compute_offset);
  auto *combined = reinterpret_cast<transport::BFloat16 *>(
      compute_bytes + a.combined_offset);
  auto *status = reinterpret_cast<TransportDeviceStatus *>(
      compute_bytes + a.status_offset);
  auto **peer_bases = reinterpret_cast<void **>(
      compute_bytes + a.peer_bases_offset);
  __shared__ unsigned long long remote_packets;
  __shared__ unsigned long long remote_routes;
  __shared__ unsigned long long remote_incoming;
  if (lane == 0) {
    remote_packets = 0;
    remote_routes = 0;
    remote_incoming = 0;
    status->start_clock = device_globaltimer_ns();
  }
  __syncthreads();

  // Notify packets are already in deterministic (destination node, token)
  // order. One packet carries one token payload and all expert fan-out rows.
  for (std::uint64_t packet = 0;
       packet < a.dispatch.sizes.packet_count; ++packet) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    const bool local = descriptor.destination_node == a.node;
    if (!local) {
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x)
        send_payload[packet * a.hidden + h] =
            a.input[descriptor.local_token_id * a.hidden + h];
      for (auto fanout = fanout_begin + lane; fanout < fanout_end;
           fanout += blockDim.x) {
        send_fanout[fanout] = a.dispatch.packet_fanout[fanout];
      }
      if (lane == 0)
        send_headers[packet] = LocalP2pRouteHeader{
            transport::kActivationTransportSchemaVersion,
            rank,
            descriptor.proxy_rank,
            descriptor.channel,
            a.dispatch.identity.generation,
            a.epoch,
            descriptor.sequence_on_link,
            fanout_end - fanout_begin,
            descriptor.local_token_id,
            fanout_begin,
            0,
            0};
      __syncthreads();
      if (lane == 0) {
        const auto destination =
            static_cast<int>(descriptor.destination_node);
        const auto source_partition =
            a.node < descriptor.destination_node ? a.node : a.node - 1U;
        const auto receive_packet =
            source_partition * a.max_tokens_per_rank +
            descriptor.sequence_on_link;
        gin.put(rail, destination, a.window,
                a.receive_header_offset + receive_packet *
                                              sizeof(LocalP2pRouteHeader),
                a.window,
                a.send_header_offset + packet * sizeof(LocalP2pRouteHeader),
                sizeof(LocalP2pRouteHeader));
        gin.put(rail, destination, a.window,
                a.receive_fanout_offset +
                    receive_packet *
                        a.dispatch.identity.top_k * sizeof(DeviceFanoutRecord),
                a.window,
                a.send_fanout_offset +
                    fanout_begin * sizeof(DeviceFanoutRecord),
                (fanout_end - fanout_begin) * sizeof(DeviceFanoutRecord));
        gin.put(rail, destination, a.window,
                a.receive_payload_offset +
                    receive_packet * a.hidden *
                        sizeof(transport::BFloat16),
                a.window,
                a.send_payload_offset + packet * a.hidden *
                                            sizeof(transport::BFloat16),
                a.hidden * sizeof(transport::BFloat16),
                ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++remote_packets;
        remote_routes += fanout_end - fanout_begin;
      }
    } else {
      for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
        const auto target = a.dispatch.packet_fanout[fanout];
        const auto route = descriptor.local_token_id *
                               a.dispatch.identity.top_k +
                           target.topk_slot;
        for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
          const auto value = a.input[descriptor.local_token_id * a.hidden + h];
          const auto destination_rank =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *destination_canonical =
              reinterpret_cast<transport::BFloat16 *>(
                  static_cast<std::byte *>(peer_bases[destination_rank]) +
                  a.canonical_offset);
          destination_canonical[target.dst_row * a.hidden + h] = value;
          combine_compute[route * a.hidden + h] = synthetic_value(
              value, target.expert_id, h, a.identity_expert);
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }
  if (lane == 0 && remote_packets != 0)
    gin.waitCounter(ncclCoopThread{}, 0, remote_packets);
  __syncthreads();

  if (lane == 0) {
    for (std::uint32_t index = 0;
         index < a.dispatch.sizes.incoming_signal_count; ++index) {
      const auto signal = a.dispatch.incoming_proxy_signals[index];
      if (signal.source_node != a.node)
        remote_incoming += signal.expected_packets;
    }
    if (remote_incoming != 0)
      gin.waitSignal(ncclCoopThread{}, 0, remote_incoming);
  }
  __syncthreads();

  unsigned long long return_count = 0;
  for (std::uint32_t signal_index = 0;
       signal_index < a.dispatch.sizes.incoming_signal_count;
       ++signal_index) {
    const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
    if (signal.source_node == a.node)
      continue;
    const auto source_partition =
        signal.source_node < a.node ? signal.source_node
                                    : signal.source_node - 1U;
    for (std::uint64_t sequence = 0;
         sequence < signal.expected_packets; ++sequence) {
      const auto packet = source_partition * a.max_tokens_per_rank + sequence;
      const auto header = receive_headers[packet];
      const auto fanout_count = header.route_index;
      for (std::uint64_t fanout = 0; fanout < fanout_count; ++fanout) {
        const auto target = receive_fanout[
            packet * a.dispatch.identity.top_k + fanout];
        const auto destination_rank =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *destination_canonical =
            reinterpret_cast<transport::BFloat16 *>(
                static_cast<std::byte *>(peer_bases[destination_rank]) +
                a.canonical_offset);
        const auto staging =
            (packet * a.dispatch.identity.top_k + fanout) * a.hidden;
        for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
          const auto value = receive_payload[packet * a.hidden + h];
          destination_canonical[target.dst_row * a.hidden + h] = value;
          return_staging[staging + h] = synthetic_value(
              value, target.expert_id, h, a.identity_expert);
        }
        __syncthreads();
      }
      if (lane == 0 && fanout_count != 0) {
        gin.put(rail, static_cast<int>(signal.source_node), a.window,
                a.return_receive_offset +
                    header.dst_row * a.hidden *
                        sizeof(transport::BFloat16),
                a.window,
                a.return_staging_offset +
                    packet * a.dispatch.identity.top_k * a.hidden *
                        sizeof(transport::BFloat16),
                fanout_count * a.hidden * sizeof(transport::BFloat16),
                ncclGin_SignalInc{1}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++return_count;
      }
      __syncthreads();
    }
  }
  if (lane == 0 && return_count != 0)
    gin.waitCounter(ncclCoopThread{}, 0, remote_packets + return_count);
  __syncthreads();
  if (lane == 0 && remote_packets != 0)
    gin.waitSignal(ncclCoopThread{}, 1, remote_packets);
  __syncthreads();

  // Map the packed return payload back to the original route slots using the
  // same deterministic outgoing packet/fan-out order which formed the put.
  for (std::uint64_t packet = 0;
       packet < a.dispatch.sizes.packet_count; ++packet) {
    const auto descriptor = a.dispatch.packet_headers[packet];
    if (descriptor.destination_node == a.node)
      continue;
    const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
    const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
    for (auto fanout = fanout_begin; fanout < fanout_end; ++fanout) {
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = descriptor.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x)
        combine_compute[route * a.hidden + h] =
            return_receive[fanout * a.hidden + h];
      __syncthreads();
    }
  }

  for (std::uint64_t token = 0; token < a.dispatch.identity.num_tokens;
       ++token) {
    for (std::uint64_t h = lane; h < a.hidden; h += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto slot = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + slot;
        sum = gate_after_accumulate(
            sum, a.gates[route], combine_compute[route * a.hidden + h].bits);
      }
      combined[token * a.hidden + h] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
  }
  __syncthreads();
  if (lane == 0) {
    status->packets_done = a.dispatch.sizes.packet_count;
    status->fanout_done = a.dispatch.sizes.fanout_count;
    status->synthetic_done = a.dispatch.sizes.incoming_row_count;
    status->combine_done = a.dispatch.sizes.route_count;
    status->incoming_done = a.dispatch.sizes.incoming_row_count;
    status->reduced_tokens = a.dispatch.identity.num_tokens;
    status->raw_cross_node_payload_bytes =
        remote_routes * a.hidden * sizeof(transport::BFloat16);
    status->deduplicated_cross_node_payload_bytes =
        remote_packets * a.hidden * sizeof(transport::BFloat16);
    status->actual_dispatch_peer_bytes =
        status->deduplicated_cross_node_payload_bytes;
    status->actual_return_peer_bytes =
        remote_routes * a.hidden * sizeof(transport::BFloat16);
    status->complete_clock = device_globaltimer_ns();
  }
}
#endif

__global__ void peer_put_signal_kernel(const std::uint64_t *source,
                                       std::uint64_t *destination,
                                       unsigned int *signal,
                                       unsigned int *completed_blocks,
                                       std::uint64_t count,
                                       bool early_signal) {
  if (early_signal && blockIdx.x == 0 && threadIdx.x == 0) {
    __threadfence_system();
    atomicExch_system(signal, 1U);
  }
  if (early_signal && blockIdx.x != 0) {
    const auto begin = clock64();
    while (clock64() - begin < 1000000ULL) {
    }
  }
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    destination[index] = source[index];
  __syncthreads();
  if (threadIdx.x == 0) {
    // Every producer block publishes its payload before joining the completion
    // counter. Only the last block may publish the remote signal.
    __threadfence_system();
    const auto prior = atomicAdd(completed_blocks, 1U);
    if (!early_signal && prior + 1U == gridDim.x) {
      __threadfence_system();
      atomicExch_system(signal, 1U);
    }
  }
}

__global__ void peer_signal_verify_kernel(const std::uint64_t *destination,
                                          const unsigned int *signal,
                                          unsigned int *errors,
                                          std::uint64_t count,
                                          std::uint64_t expected) {
  while (atomicAdd_system(const_cast<unsigned int *>(signal), 0U) == 0U) {
  }
  __threadfence_system();
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    if (destination[index] != expected)
      atomicAdd(errors, 1U);
}

__global__ void initialize_transport_credits_kernel(unsigned int *credits,
                                                    std::uint64_t count,
                                                    unsigned int value) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    credits[index] = value;
}

__global__ void initialize_p2p_barrier_kernel(unsigned long long *record,
                                               unsigned long long slot_index,
                                               unsigned long long epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    record[0] = slot_index;
    record[1] = epoch;
  }
}

#if MEGAMOE_DEBUG_VALIDATE
__global__ void validate_p2p_barrier_kernel(
    const unsigned long long *records, std::uint32_t world,
    unsigned long long expected_slot, unsigned long long expected_epoch,
    TransportDeviceStatus *status) {
  const auto rank = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (rank >= world)
    return;
  if (records[rank * 2U] != expected_slot ||
      records[rank * 2U + 1U] != expected_epoch) {
    status->error_packet = rank;
    status->error_field = 19U;
    atomicCAS(&status->error, 0U, 1U);
  }
}
#endif

template <typename T> __device__ T *peer_at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

#if MEGAMOE_DEBUG_VALIDATE
template <typename Args>
__device__ __forceinline__ void
production_terminal_debug_progress(const Args &a, ProductionStatusPhase phase,
                                   unsigned long long value) {
  if (phase == ProductionStatusPhase::Forward && a.progress != nullptr &&
      blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        progress(a.progress[kForwardTerminal]);
    progress.store(value, ::cuda::memory_order_release);
  }
}
#else
template <typename Args>
__device__ __forceinline__ void
production_terminal_debug_progress(const Args &, ProductionStatusPhase,
                                   unsigned long long) {}
#endif

template <bool StaticLocal, typename Args, typename Status>
__device__ __forceinline__ void publish_production_terminal_status(
    const Args &a, Status *status, ProductionStatusPhase phase,
    std::size_t ordinary_record_offset, std::size_t gin_record_offset,
    unsigned long long prior_gin_signal_count) {
  if (blockIdx.x != 0U)
    return;
  production_terminal_debug_progress(a, phase, 10ULL);
  const auto local_world =
      a.local_world_size == 0U ? a.world : a.local_world_size;
  const auto node = a.local_world_size == 0U ? 0U : a.node;
  const auto first_local = node * local_world;
  // Legacy cancellation and debug checks may have claimed the status word
  // before structured fields were written. Normalize that already-owned local
  // failure at the single production boundary so it cannot be mistaken for a
  // successful record by the host agreement.
  if (threadIdx.x == 0U && status->error != 0U && status->error_code == 0U) {
    status->error_origin_rank = a.rank;
    status->error_code = status->error;
    status->error_expected = 0ULL;
    status->error_actual = status->error;
    __threadfence_system();
  }
  __syncthreads();
  const ProductionStatusRecord local{
      a.dispatch.identity.generation,
      static_cast<unsigned long long>(phase),
      a.rank,
      status->error == 0U ? 0ULL
                          : static_cast<unsigned long long>(status->error_code),
      status->error == 0U ? 0ULL
                          : static_cast<unsigned long long>(status->error_field),
      status->error_expected,
      status->error_actual,
      a.epoch};

  // The Socket fallback deliberately has no CUDA-IPC/Gin peer window. Publish
  // into owner-local storage and let the mandatory production NCCL status
  // agreement carry the same record after the host-enqueued payload drain.
  if (!StaticLocal && a.use_socket) {
    if (threadIdx.x == 0U) {
      auto *records = peer_at<ProductionStatusRecord>(
          a.local_base, ordinary_record_offset);
      auto record = local;
      record.epoch = 0ULL;
      records[a.rank] = record;
      __threadfence_system();
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(records[a.rank].epoch);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    return;
  }

  // Publish this rail's local record to every node-local peer. No record is
  // consumed until its epoch is observed with acquire ordering.
  for (std::uint32_t local_peer = threadIdx.x; local_peer < local_world;
       local_peer += blockDim.x) {
    const auto peer = first_local + local_peer;
    auto *records = peer_at<ProductionStatusRecord>(
        a.peer_bases[peer], ordinary_record_offset);
    auto record = local;
    record.epoch = 0ULL;
    records[a.rank] = record;
    __threadfence_system();
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(records[a.rank].epoch);
    published.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  production_terminal_debug_progress(a, phase, 20ULL);

#if __has_include(<nccl_device.h>)
  if (!StaticLocal && a.use_gin) {
    auto *gin_records = reinterpret_cast<ProductionStatusRecord *>(
        static_cast<std::byte *>(a.gin_window_base) + gin_record_offset);
    if (threadIdx.x == 0U) {
      auto record = local;
      record.epoch = 0ULL;
      gin_records[a.rank] = record;
      __threadfence_system();
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(gin_records[a.rank].epoch);
      published.store(a.epoch, ::cuda::memory_order_release);
      const ncclGin gin(a.gin_dev_comm, 0);
      const auto rail = ncclTeamRail(a.gin_dev_comm);
      production_terminal_debug_progress(a, phase, 30ULL);
      // Indexed signals are context-scoped, not operation-scoped.  In the
      // specialized Forward path signal 2 already contains the fixed-width
      // generation-ready handshake.  Its terminal threshold must therefore
      // include that protocol-known prefix.  It cannot be derived by reading
      // the live signal here: a faster peer may already have delivered this
      // terminal wave, which would make a sampled-base-plus-delta wait for an
      // increment that will never exist.
      //
      unsigned long long status_peers = 0ULL;
      const unsigned long long words[8] = {
          local.generation, local.phase,    local.origin_rank, local.error_code,
          local.field,      local.expected, local.actual,      local.epoch};
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == node)
          continue;
        const auto destination =
            gin_record_offset +
            static_cast<std::size_t>(a.rank) * sizeof(local);
        // Inline values remove the generic-store -> async-proxy source
        // dependency entirely.  The final epoch write carries the signal;
        // GIN orders it after all preceding puts to this context/peer, so the
        // receiver cannot observe completion before all eight words settle.
#pragma unroll
        for (std::uint32_t word = 0; word < 7U; ++word) {
          gin.putValue<unsigned long long>(
              rail, static_cast<int>(remote_node), a.gin_window,
              destination + word * sizeof(unsigned long long), words[word],
              ncclGin_None{}, ncclCoopThread{}, ncclGin_None{},
              ::cuda::thread_scope_thread, ::cuda::thread_scope_device);
          production_terminal_debug_progress(a, phase, 40ULL + word);
        }
        gin.putValue<unsigned long long>(
            rail, static_cast<int>(remote_node), a.gin_window,
            destination + 7U * sizeof(unsigned long long), words[7],
            ncclGin_SignalInc{2}, ncclCoopThread{}, ncclGin_None{},
            ::cuda::thread_scope_thread, ::cuda::thread_scope_device);
        production_terminal_debug_progress(a, phase, 47ULL);
        ++status_peers;
      }
      if (status_peers != 0ULL) {
        const auto expected = prior_gin_signal_count + status_peers;
        production_terminal_debug_progress(
            a, phase,
            0x5000000000000000ULL | (expected << 32U) | gin.readSignal(2));
        gin.waitSignal(ncclCoopThread{}, 2, expected);
      }
      production_terminal_debug_progress(a, phase, 60ULL);
    }
    __syncthreads();
    // GIN writes arrive through Hopper's async proxy. Signal observation
    // orders the remote operation, but generic CTA loads of the received
    // record require an explicit proxy-domain fence. Without it the synthetic
    // epoch publication below could expose a current epoch with a stale
    // generation-zero payload.
    asm volatile("fence.proxy.async.global;" : : : "memory");
    __syncthreads();
    production_terminal_debug_progress(a, phase, 70ULL);
    // Each same-rail proxy receives the record for its corresponding remote
    // rank, then fans it out through already-owned node-local IPC mappings.
    const auto local_rank = a.rank - first_local;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == node)
        continue;
      const auto source = remote_node * local_world + local_rank;
      const auto remote = gin_records[source];
      for (std::uint32_t local_peer = threadIdx.x; local_peer < local_world;
           local_peer += blockDim.x) {
        const auto peer = first_local + local_peer;
        auto *records = peer_at<ProductionStatusRecord>(
            a.peer_bases[peer], ordinary_record_offset);
        auto record = remote;
        record.epoch = 0ULL;
        records[source] = record;
        __threadfence_system();
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(records[source].epoch);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    __syncthreads();
    production_terminal_debug_progress(a, phase, 80ULL);
  }
#else
  (void)gin_record_offset;
#endif

  auto *observed_records = peer_at<ProductionStatusRecord>(
      a.local_base, ordinary_record_offset);
  const auto observed_world = a.world;
  const auto observed_first = 0U;
  for (std::uint32_t offset = threadIdx.x; offset < observed_world;
       offset += blockDim.x) {
    const auto source = observed_first + offset;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        observed(observed_records[source].epoch);
    while (observed.load(::cuda::memory_order_acquire) < a.epoch) {
    }
  }
  __syncthreads();
  production_terminal_debug_progress(a, phase, 90ULL);

  // Local P2P and GIN both complete the fixed status agreement on device.
  // For GIN, every rail has already put its record cross-node and fanned the
  // corresponding remote record through node-local IPC, so the matrix below
  // is the same full-world matrix a redundant host NCCL AllGather would
  // produce. Validate and select the deterministic error here before the
  // status D2H; Socket returned above and retains its host collective.
  if (StaticLocal || a.use_gin) {
    if (threadIdx.x == 0U) {
      ProductionStatusRecord selected{};
      bool has_error = false;
      const auto expected_phase = static_cast<unsigned long long>(phase);
      for (std::uint32_t source = 0; source < observed_world; ++source) {
        auto record = observed_records[source];
        if (record.generation != a.dispatch.identity.generation) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          1ULL,
                                          a.dispatch.identity.generation,
                                          record.generation,
                                          a.epoch};
        } else if (record.phase != expected_phase) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          2ULL,
                                          expected_phase,
                                          record.phase,
                                          a.epoch};
        } else if (record.origin_rank != source && record.error_code == 0ULL) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          3ULL,
                                          source,
                                          record.origin_rank,
                                          a.epoch};
        } else if (record.epoch != a.epoch) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase,
                                          source,
                                          2ULL,
                                          4ULL,
                                          a.epoch,
                                          record.epoch,
                                          a.epoch};
        }
        if (record.error_code == 0ULL)
          continue;
        if (!has_error || record.origin_rank < selected.origin_rank ||
            (record.origin_rank == selected.origin_rank &&
             record.field < selected.field)) {
          selected = record;
          has_error = true;
        }
      }
      if (has_error) {
        status->error = 1U;
        status->error_origin_rank =
            static_cast<std::uint32_t>(selected.origin_rank);
        status->error_code = static_cast<std::uint32_t>(selected.error_code);
        status->error_field = static_cast<std::uint32_t>(selected.field);
        status->error_expected = selected.expected;
        status->error_actual = selected.actual;
      }
      __threadfence_system();
    }
    __syncthreads();
  }
  production_terminal_debug_progress(a, phase, 100ULL);
}

// The storage is owned by the local rank: ingress_published is written by
// remote Dispatch producers and outgoing_sequence is written by remote Return
// producers. Initialize only the local allocation, then publish setup-ready
// to every peer. No peer may issue a put until every owner has published the
// same epoch, which prevents initialization racing a remote release store.
__global__ void initialize_forward_generation_signals_kernel(
    void *local_base, void **peer_bases, std::size_t ingress_published_offset,
    std::size_t outgoing_sequence_offset, std::size_t setup_epoch_offset,
    std::size_t gin_local_compute_ready_offset,
    std::size_t forward_x_arrived_rows_offset,
    std::uint64_t mtile_count, std::size_t forward_completion_offset,
    std::size_t forward_context_completion_offset,
    std::uint32_t context_completion_capacity, std::uint64_t generation,
    std::uint32_t rank,
    std::uint32_t world, std::uint32_t channels, unsigned long long epoch,
    std::uint32_t first_peer, std::uint32_t peer_count) {
  const auto index = static_cast<std::uint32_t>(threadIdx.x);
  auto *incoming = peer_at<unsigned long long>(local_base,
                                                ingress_published_offset);
  auto *outgoing = peer_at<unsigned long long>(local_base,
                                                outgoing_sequence_offset);
  auto *compute_ready = peer_at<unsigned long long>(
      local_base, gin_local_compute_ready_offset);
  auto *x_arrived = peer_at<unsigned long long>(
      local_base, forward_x_arrived_rows_offset);
  auto *forward_completion = peer_at<transport_state::GenerationCompletionVector>(
      local_base, forward_completion_offset);
  auto *forward_context = peer_at<transport_state::ContextCompletionRecord>(
      local_base, forward_context_completion_offset);
  if (index == 0U) {
    peer_bases[world] = forward_completion;
    peer_bases[world + 1U] = forward_context;
  }
  const auto initial = epoch - 1ULL;
  for (std::uint32_t link = index; link < world * channels;
       link += blockDim.x) {
    incoming[link] = initial;
    outgoing[link] = initial;
  }
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x)
    compute_ready[peer] = initial;
  const auto x_arrived_count = mtile_count * peer_count;
  for (std::uint64_t slot = index; slot < x_arrived_count; slot += blockDim.x) {
    x_arrived[slot] = 0ULL;
  }
  if (index == 0U) {
    *forward_completion = transport_state::GenerationCompletionVector{};
    forward_completion->generation = generation;
    forward_completion->epoch = epoch;
    forward_completion->first_waiting_tile = ~0ULL;
  }
  for (std::uint32_t context = index; context < context_completion_capacity;
       context += blockDim.x) {
    forward_context[context] = transport_state::ContextCompletionRecord{};
    forward_context[context].generation = generation;
    forward_context[context].epoch = epoch;
  }
  __syncthreads();
  __threadfence_system();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    auto *ready = peer_at<unsigned long long>(peer_bases[peer],
                                               setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ready(ready[rank]);
    peer_ready.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_ready = peer_at<unsigned long long>(local_base,
                                                   setup_epoch_offset);
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ready(local_ready[peer]);
    while (peer_ready.load(::cuda::memory_order_acquire) < epoch) {
    }
  }
  __syncthreads();
  // A second monotonic acknowledgement prevents a fast rank from returning
  // and initializing the next epoch before every slow rank has observed this
  // epoch's ready publication.
  __threadfence_system();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    auto *ack = peer_at<unsigned long long>(peer_bases[peer],
                                             setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ack(ack[world + rank]);
    peer_ack.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  for (std::uint32_t local_peer = index; local_peer < peer_count;
       local_peer += blockDim.x) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ack(local_ready[world + peer]);
    while (peer_ack.load(::cuda::memory_order_acquire) < epoch) {
    }
  }
}

struct LocalP2pKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *input;
  const float *gates;
  std::uint64_t hidden;
  std::uint64_t max_destination_rows;
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t channels;
  std::uint32_t ring_slots;
  std::uint64_t epoch;
  void **peer_bases;
  void *local_base;
  std::size_t ring_headers_offset;
  std::size_t ring_payload_offset;
  std::size_t ring_ready_offset;
  std::size_t link_next_offset;
  std::size_t link_credits_offset;
  std::size_t ingress_headers_offset;
  std::size_t ingress_ready_offset;
  std::size_t ingress_next_offset;
  std::size_t ingress_published_offset;
  std::size_t ingress_credits_offset;
  std::size_t outgoing_sequence_offset;
  std::size_t direct_headers_offset;
  std::size_t return_ready_offset;
  std::size_t canonical_rows_offset;
  std::size_t contribution_offset;
  std::size_t combine_return_offset;
  std::size_t combined_offset;
  std::size_t status_offset;
  const unsigned long long *cancel_epoch;
  bool identity_expert;
};

__device__ bool local_p2p_cancelled(const LocalP2pKernelArgs &a) {
  return atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
         a.dispatch.identity.generation;
}

__device__ bool acquire_system_credit(unsigned int *credit,
                                      TransportDeviceStatus *status,
                                      const LocalP2pKernelArgs &a) {
  while (!local_p2p_cancelled(a)) {
    const auto observed = atomicAdd_system(credit, 0U);
    if (observed != 0U &&
        atomicCAS_system(credit, observed, observed - 1U) == observed)
      return true;
    atomicAdd(&status->credit_stalls, 1ULL);
  }
  return false;
}

[[maybe_unused]] __global__ void
local_p2p_persistent_kernel(LocalP2pKernelArgs a) {
  const auto lane = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  auto *local_ring_headers = peer_at<transport::TransportSlotHeader>(
      a.local_base, a.ring_headers_offset);
  auto *local_ring_payload = peer_at<transport::BFloat16>(
      a.local_base, a.ring_payload_offset);
  auto *local_ring_ready =
      peer_at<unsigned long long>(a.local_base, a.ring_ready_offset);
  auto *local_link_next =
      peer_at<unsigned long long>(a.local_base, a.link_next_offset);
  auto *local_link_credits =
      peer_at<unsigned int>(a.local_base, a.link_credits_offset);
  auto *local_ingress_headers = peer_at<LocalP2pRouteHeader>(
      a.local_base, a.ingress_headers_offset);
  auto *local_ingress_ready =
      peer_at<unsigned long long>(a.local_base, a.ingress_ready_offset);
  auto *local_ingress_next =
      peer_at<unsigned long long>(a.local_base, a.ingress_next_offset);
  auto *local_ingress_published =
      peer_at<unsigned long long>(a.local_base, a.ingress_published_offset);
  auto *local_ingress_credits =
      peer_at<unsigned int>(a.local_base, a.ingress_credits_offset);
  auto *outgoing_sequence =
      peer_at<unsigned long long>(a.local_base, a.outgoing_sequence_offset);
  auto *local_return_ready =
      peer_at<unsigned long long>(a.local_base, a.return_ready_offset);
  auto *local_canonical = peer_at<transport::BFloat16>(
      a.local_base, a.canonical_rows_offset);
  auto *local_contribution = peer_at<transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *local_combine_return = peer_at<transport::BFloat16>(
      a.local_base, a.combine_return_offset);
  auto *local_combined =
      peer_at<transport::BFloat16>(a.local_base, a.combined_offset);
  __shared__ unsigned long long shared_value;
  __shared__ unsigned long long stage_clock;
  __shared__ unsigned long long synthetic_phase_clock;

  if (lane == 0)
    atomicCAS(&status->start_clock, 0ULL, device_globaltimer_ns());
  __syncthreads();
  if (atomicAdd(&status->error, 0U) != 0U)
    return;

  // CTA 0 is the source Comm+proxy Relay worker. Packets and fan-out are
  // already in canonical descriptor order, so one elected producer owns every
  // sequence transition and no competing atomic-return order is observable.
  if (blockIdx.x == 0) {
    for (std::uint64_t packet = 0;
         packet < a.dispatch.sizes.packet_count && !local_p2p_cancelled(a);
         ++packet) {
      const auto packet_header = a.dispatch.packet_headers[packet];
      const auto link = static_cast<std::uint64_t>(packet_header.channel);
      if (packet_header.destination_node != 0 ||
          packet_header.channel >= a.channels ||
          packet_header.sequence_on_link != local_link_next[link]) {
        if (lane == 0) {
          status->error_packet = static_cast<unsigned int>(packet);
          status->error_field = 20U;
          atomicCAS(&status->error, 0U, 1U);
        }
        break;
      }
      if (lane == 0)
        shared_value = acquire_system_credit(&local_link_credits[link], status,
                                             a)
                           ? 1ULL
                           : 0ULL;
      __syncthreads();
      if (shared_value == 0)
        break;
      if (lane == 0) {
        const auto available = atomicAdd_system(&local_link_credits[link], 0U);
        atomicMax(&status->peak_ring_slots,
                  static_cast<unsigned long long>(a.ring_slots - available));
      }
      const auto physical =
          link * a.ring_slots + packet_header.sequence_on_link % a.ring_slots;
      if (lane == 0)
        stage_clock = device_globaltimer_ns();
      if (lane == 0) {
        local_ring_headers[physical] = transport::TransportSlotHeader{
            transport::kActivationTransportSchemaVersion,
            a.rank,
            0,
            0,
            packet_header.proxy_rank,
            packet_header.channel,
            a.dispatch.identity.generation,
            a.epoch,
            packet_header.sequence_on_link,
            packet_header.local_token_id,
            a.dispatch.packet_fanout_offsets[packet],
            a.dispatch.packet_fanout_offsets[packet + 1U]};
      }
      for (std::uint64_t hidden = lane; hidden < a.hidden;
           hidden += blockDim.x)
        local_ring_payload[physical * a.hidden + hidden] =
            a.input[packet_header.local_token_id * a.hidden + hidden];
      __syncthreads();
      if (lane == 0) {
        __threadfence_system();
        atomicExch_system(&local_ring_ready[physical],
                          packet_header.sequence_on_link + 1U);
        atomicAdd(&status->packets_done, 1ULL);
        atomicAdd(&status->dispatch_put_ticks,
                  device_globaltimer_ns() - stage_clock);
        stage_clock = device_globaltimer_ns();
      }
      __syncthreads();

      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (lane == 0) {
        atomicAdd(&status->deduplicated_cross_node_payload_bytes,
                  a.hidden * sizeof(transport::BFloat16));
        atomicAdd(&status->raw_cross_node_payload_bytes,
                  (fanout_end - fanout_begin) * a.hidden *
                      sizeof(transport::BFloat16));
      }
      for (auto fanout = fanout_begin;
           fanout < fanout_end && !local_p2p_cancelled(a); ++fanout) {
        const auto target = a.dispatch.packet_fanout[fanout];
        const auto route = packet_header.local_token_id *
                               a.dispatch.identity.top_k +
                           target.topk_slot;
        if (route >= a.dispatch.sizes.route_count) {
          if (lane == 0) {
            status->error_packet = static_cast<unsigned int>(packet);
            status->error_field = 21U;
            atomicCAS(&status->error, 0U, 1U);
          }
          break;
        }
        const auto destination_rank = a.dispatch.route_dst_rank[route];
        if (destination_rank >= a.world ||
            target.destination_local_rank != destination_rank ||
            target.dst_row >= a.max_destination_rows) {
          if (lane == 0) {
            status->error_packet = static_cast<unsigned int>(packet);
            status->error_field = 21U;
            atomicCAS(&status->error, 0U, 1U);
          }
          break;
        }
        auto *destination_base = a.peer_bases[destination_rank];
        auto *destination_credit = peer_at<unsigned int>(
            destination_base, a.ingress_credits_offset);
        const auto ingress_link =
            static_cast<std::uint64_t>(a.rank) * a.channels +
            packet_header.channel;
        const auto no_wrap_batch = a.ring_slots >= a.max_destination_rows;
        if (lane == 0)
          shared_value =
              (no_wrap_batch || acquire_system_credit(
                                    &destination_credit[ingress_link], status, a))
                  ? outgoing_sequence[destination_rank * a.channels +
                                      packet_header.channel]
                  : ~0ULL;
        __syncthreads();
        if (shared_value == ~0ULL)
          break;
        if (lane == 0 && !no_wrap_batch) {
          const auto available =
              atomicAdd_system(&destination_credit[ingress_link], 0U);
          atomicMax(&status->peak_ring_slots,
                    static_cast<unsigned long long>(a.ring_slots - available));
        }
        const auto sequence = shared_value;
        const auto ingress_physical =
            ingress_link * a.ring_slots + sequence % a.ring_slots;
        auto *destination_canonical = peer_at<transport::BFloat16>(
            destination_base, a.canonical_rows_offset);
        for (std::uint64_t hidden = lane; hidden < a.hidden;
             hidden += blockDim.x)
          destination_canonical[target.dst_row * a.hidden + hidden] =
              local_ring_payload[physical * a.hidden + hidden];
        __syncthreads();
        if (lane == 0) {
          auto *destination_headers = peer_at<LocalP2pRouteHeader>(
              destination_base, a.ingress_headers_offset);
          destination_headers[ingress_physical] = LocalP2pRouteHeader{
              transport::kActivationTransportSchemaVersion,
              a.rank,
              destination_rank,
              packet_header.channel,
              a.dispatch.identity.generation,
              a.epoch,
              sequence,
              route,
              packet_header.local_token_id,
              target.dst_row,
              target.expert_id,
              target.topk_slot};
          if (!no_wrap_batch) {
            __threadfence_system();
            auto *destination_ready = peer_at<unsigned long long>(
                destination_base, a.ingress_ready_offset);
            atomicExch_system(&destination_ready[ingress_physical],
                              sequence + 1U);
          } else if ((sequence + 1U) % 64U == 0U) {
            __threadfence_system();
            auto *destination_published = peer_at<unsigned long long>(
                destination_base, a.ingress_published_offset);
            atomicExch_system(&destination_published[ingress_link],
                              sequence + 1U);
          }
          outgoing_sequence[destination_rank * a.channels +
                            packet_header.channel] = sequence + 1U;
          atomicAdd(&status->fanout_done, 1ULL);
        }
        __syncthreads();
      }
      if (lane == 0) {
        local_link_next[link] = packet_header.sequence_on_link + 1U;
        atomicAdd_system(&local_link_credits[link], 1U);
        atomicAdd(&status->relay_ticks,
                  device_globaltimer_ns() - stage_clock);
      }
      __syncthreads();
    }
    if (a.ring_slots >= a.max_destination_rows) {
      for (std::uint32_t destination = lane; destination < a.world;
           destination += blockDim.x) {
        const auto final_sequence =
            outgoing_sequence[destination * a.channels];
        if (final_sequence != 0U) {
          const auto ingress_link =
              static_cast<std::uint64_t>(a.rank) * a.channels;
          auto *destination_published = peer_at<unsigned long long>(
              a.peer_bases[destination], a.ingress_published_offset);
          __threadfence_system();
          atomicExch_system(&destination_published[ingress_link],
                            final_sequence);
        }
      }
    }
  }

  // CTA 1+source consumes the independent bounded ingress link from source.
  // It observes only the release sequence before reading the canonical row.
  if (blockIdx.x > 0 && blockIdx.x <= a.world) {
    const auto source = static_cast<std::uint32_t>(blockIdx.x - 1U);
    std::uint64_t expected = 0;
    if (lane == 0)
      shared_value = 0;
    __syncthreads();
    for (std::uint64_t row = lane; row < a.dispatch.sizes.incoming_row_count;
         row += blockDim.x)
      if (a.dispatch.incoming_rows[row].source_rank == source)
        atomicAdd(&shared_value, 1ULL);
    __syncthreads();
    expected = shared_value;
    const auto ingress_link = static_cast<std::uint64_t>(source) * a.channels;
    if (lane == 0)
      synthetic_phase_clock = device_globaltimer_ns();
    __syncthreads();
    for (std::uint64_t sequence = 0;
         sequence < expected && !local_p2p_cancelled(a); ++sequence) {
      const auto physical =
          ingress_link * a.ring_slots + sequence % a.ring_slots;
      if (lane == 0) {
        if (a.ring_slots >= a.max_destination_rows) {
          while (atomicAdd_system(&local_ingress_published[ingress_link],
                                  0ULL) < sequence + 1U &&
                 !local_p2p_cancelled(a)) {
          }
        } else {
          while (atomicAdd_system(&local_ingress_ready[physical], 0ULL) !=
                     sequence + 1U &&
                 !local_p2p_cancelled(a)) {
          }
        }
        shared_value = local_p2p_cancelled(a) ? 0ULL : 1ULL;
      }
      __syncthreads();
      if (shared_value == 0)
        break;
      const auto header = local_ingress_headers[physical];
      if (lane == 0)
        stage_clock = device_globaltimer_ns();
      const auto incoming = header.dst_row < a.dispatch.sizes.incoming_row_count
                                ? a.dispatch.incoming_rows[header.dst_row]
                                : DeviceIncomingRowRecord{};
      if (header.schema_version != transport::kActivationTransportSchemaVersion ||
          header.source_rank != source || header.destination_rank != a.rank ||
          header.generation != a.dispatch.identity.generation ||
          header.sequence != sequence ||
          header.dst_row >= a.dispatch.sizes.incoming_row_count ||
          incoming.source_rank != source || incoming.expert_id != header.expert_id) {
        if (lane == 0) {
          status->error_packet = static_cast<unsigned int>(sequence);
          status->error_field = 22U;
          atomicCAS(&status->error, 0U, 1U);
        }
        break;
      }
      auto *source_base = a.peer_bases[source];
      auto *source_contribution = peer_at<transport::BFloat16>(
          source_base, a.contribution_offset);
      auto *source_combine = peer_at<transport::BFloat16>(
          source_base, a.combine_return_offset);
      for (std::uint64_t hidden = lane; hidden < a.hidden;
           hidden += blockDim.x) {
        const auto contribution = synthetic_value(
            local_canonical[header.dst_row * a.hidden + hidden],
            header.expert_id, hidden, a.identity_expert);
        source_contribution[header.route_index * a.hidden + hidden] =
            contribution;
        source_combine[header.route_index * a.hidden + hidden] = contribution;
      }
      __syncthreads();
      if (lane == 0) {
        __threadfence_system();
        if (a.ring_slots < a.max_destination_rows) {
          auto *source_ready = peer_at<unsigned long long>(
              source_base, a.return_ready_offset);
          atomicExch_system(&source_ready[header.route_index],
                            header.source_epoch);
          auto *source_status = peer_at<TransportDeviceStatus>(
              source_base, a.status_offset);
          atomicAdd_system(&source_status->synthetic_done, 1ULL);
          atomicAdd_system(&source_status->combine_done, 1ULL);
        }
        local_ingress_next[ingress_link] = sequence + 1U;
        if (a.ring_slots < a.max_destination_rows) {
          atomicExch_system(&local_ingress_ready[physical], 0ULL);
          atomicAdd_system(&local_ingress_credits[ingress_link], 1U);
        }
        atomicAdd(&status->incoming_done, 1ULL);
      }
      __syncthreads();
    }
    if (a.ring_slots >= a.max_destination_rows && expected != 0U && lane == 0) {
      __threadfence_system();
      auto *source_status = peer_at<TransportDeviceStatus>(
          a.peer_bases[source], a.status_offset);
      atomicAdd_system(&source_status->synthetic_done, expected);
      atomicAdd_system(&source_status->combine_done, expected);
    }
    if (lane == 0)
      atomicMax(&status->synthetic_ticks,
                device_globaltimer_ns() - synthetic_phase_clock);
  }

  while (!local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U &&
         (atomicAdd(&status->packets_done, 0ULL) <
              a.dispatch.sizes.packet_count ||
          atomicAdd(&status->incoming_done, 0ULL) <
              a.dispatch.sizes.incoming_row_count ||
          atomicAdd_system(&status->combine_done, 0ULL) <
              a.dispatch.sizes.route_count)) {
  }
  if (local_p2p_cancelled(a) && lane == 0)
    atomicCAS(&status->error, 0U, 2U);
  if (lane == 0)
    atomicMax(&status->dispatch_complete_clock, device_globaltimer_ns());
  __threadfence_system();

  while (!local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    if (lane == 0)
      shared_value = atomicAdd(&status->token_claim_cursor, 1ULL);
    __syncthreads();
    const auto token = shared_value;
    if (token >= a.dispatch.identity.num_tokens)
      break;
    for (std::uint64_t hidden = lane; hidden < a.hidden;
         hidden += blockDim.x) {
      float sum = 0.0F;
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto original = a.dispatch.combine_order[
            token * a.dispatch.identity.top_k + order];
        const auto route = token * a.dispatch.identity.top_k + original;
        if (a.ring_slots < a.max_destination_rows)
          while (atomicAdd_system(&local_return_ready[route], 0ULL) != a.epoch &&
                 !local_p2p_cancelled(a)) {
          }
        sum = gate_after_accumulate(
            sum, a.gates[route],
            local_combine_return[route * a.hidden + hidden].bits);
      }
      local_combined[token * a.hidden + hidden] =
          transport::BFloat16{float_to_bf16_bits(sum)};
    }
    __syncthreads();
    if (lane == 0)
      atomicAdd(&status->reduced_tokens, 1ULL);
  }
  if (lane == 0)
    atomicMax(&status->complete_clock, device_globaltimer_ns());
}

// Trusted-team release path. Notify has already assigned a unique canonical
// row to every route. Separate all-SM grids bulk-copy each phase, and tiny
// kernels publish one system-scope completion per peer rather than per route.
#if !MEGAMOE_DEBUG_VALIDATE
__global__ void local_p2p_bulk_dispatch_kernel(LocalP2pKernelArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x);
  if (route >= a.dispatch.sizes.route_count)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (route == 0 && threadIdx.x == 0)
    status->start_clock = device_globaltimer_ns();
  const auto destination = a.dispatch.route_dst_rank[route];
  const auto row = a.dispatch.route_dst_row[route];
  if (destination >= a.world || row >= a.max_destination_rows) {
    if (threadIdx.x == 0) {
      status->error_packet = static_cast<unsigned int>(route);
      status->error_field = 30U;
      atomicCAS(&status->error, 0U, 1U);
    }
    return;
  }
  const auto token = route / a.dispatch.identity.top_k;
  auto *destination_base = a.peer_bases[destination];
  auto *destination_rows = peer_at<transport::BFloat16>(
      destination_base, a.canonical_rows_offset);
  const auto *source = a.input + token * a.hidden;
  auto *target = destination_rows + row * a.hidden;
  if ((a.hidden & 7ULL) == 0ULL) {
    const auto *source128 = reinterpret_cast<const uint4 *>(source);
    auto *target128 = reinterpret_cast<uint4 *>(target);
    const auto vector_count = a.hidden / 8ULL;
    for (std::uint64_t vector = threadIdx.x; vector < vector_count;
         vector += blockDim.x * 2ULL) {
      target128[vector] = source128[vector];
      const auto second = vector + blockDim.x;
      if (second < vector_count)
        target128[second] = source128[second];
    }
  } else {
    for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
         hidden += blockDim.x)
      target[hidden] = source[hidden];
  }
  if (threadIdx.x == 0) {
    auto *destination_headers = peer_at<LocalP2pRouteHeader>(
        destination_base, a.direct_headers_offset);
    destination_headers[row] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        a.rank,
        destination,
        static_cast<std::uint32_t>(route % a.channels),
        a.dispatch.identity.generation,
        a.epoch,
        route,
        route,
        token,
        row,
        a.dispatch.expert_id[route],
        static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
    if (destination != a.rank)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                a.hidden * sizeof(transport::BFloat16));
  }
  __threadfence_system();
}

__global__ void local_p2p_publish_dispatch_kernel(LocalP2pKernelArgs a) {
  const auto destination = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (destination < a.world) {
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[destination], a.ingress_published_offset);
    __threadfence_system();
    atomicExch_system(&signal[a.rank * a.channels], a.epoch);
  }
  if (threadIdx.x == 0) {
    status->packets_done = a.dispatch.sizes.packet_count;
    status->fanout_done = a.dispatch.sizes.route_count;
    status->raw_cross_node_payload_bytes =
        a.dispatch.sizes.route_count * a.hidden *
        sizeof(transport::BFloat16);
    status->deduplicated_cross_node_payload_bytes =
        a.dispatch.sizes.packet_count * a.hidden *
        sizeof(transport::BFloat16);
    status->dispatch_put_ticks = device_globaltimer_ns() - status->start_clock;
  }
}

__global__ void local_p2p_wait_dispatch_kernel(LocalP2pKernelArgs a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  for (std::uint32_t source = threadIdx.x; source < a.world;
       source += blockDim.x) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.ingress_published_offset);
    while (atomicAdd_system(&signals[source * a.channels], 0ULL) != a.epoch &&
           !local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (local_p2p_cancelled(a))
      atomicCAS(&status->error, 0U, 2U);
    __threadfence_system();
    status->dispatch_complete_clock = device_globaltimer_ns();
  }
}

__global__ void local_p2p_bulk_return_kernel(LocalP2pKernelArgs a) {
  const auto row = static_cast<std::uint64_t>(blockIdx.x);
  if (row >= a.dispatch.sizes.incoming_row_count)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (atomicAdd(&status->error, 0U) != 0U)
    return;
  auto *local_headers = peer_at<LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *local_canonical = peer_at<transport::BFloat16>(
      a.local_base, a.canonical_rows_offset);
  const auto header = local_headers[row];
  auto *source_contribution = peer_at<transport::BFloat16>(
      a.peer_bases[header.source_rank], a.contribution_offset);
  for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
       hidden += blockDim.x)
    source_contribution[header.route_index * a.hidden + hidden] =
        synthetic_value(local_canonical[row * a.hidden + hidden],
                        header.expert_id, hidden, a.identity_expert);
  if (threadIdx.x == 0 && header.source_rank != a.rank)
    atomicAdd(&status->actual_return_peer_bytes,
              a.hidden * sizeof(transport::BFloat16));
  __threadfence_system();
}

__global__ void local_p2p_publish_return_kernel(LocalP2pKernelArgs a) {
  const auto source = static_cast<std::uint32_t>(threadIdx.x);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (source < a.world) {
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[source], a.outgoing_sequence_offset);
    __threadfence_system();
    atomicExch_system(&signal[a.rank * a.channels], a.epoch);
  }
  if (threadIdx.x == 0)
    status->incoming_done = a.dispatch.sizes.incoming_row_count;
}

__global__ void local_p2p_wait_return_kernel(LocalP2pKernelArgs a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  for (std::uint32_t destination = threadIdx.x; destination < a.world;
       destination += blockDim.x) {
    auto *signals = peer_at<unsigned long long>(
        a.local_base, a.outgoing_sequence_offset);
    while (atomicAdd_system(&signals[destination * a.channels], 0ULL) !=
               a.epoch &&
           !local_p2p_cancelled(a) && atomicAdd(&status->error, 0U) == 0U) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (local_p2p_cancelled(a))
      atomicCAS(&status->error, 0U, 2U);
    __threadfence_system();
    status->synthetic_ticks =
        device_globaltimer_ns() - status->dispatch_complete_clock;
    status->relay_ticks = status->synthetic_ticks;
    status->synthetic_done = a.dispatch.sizes.route_count;
    status->combine_done = a.dispatch.sizes.route_count;
  }
}

__global__ void local_p2p_bulk_reduce_kernel(LocalP2pKernelArgs a) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= a.dispatch.identity.num_tokens)
    return;
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  if (atomicAdd(&status->error, 0U) != 0U)
    return;
  auto *contribution = peer_at<transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *combined = peer_at<transport::BFloat16>(
      a.local_base, a.combined_offset);
  for (std::uint64_t hidden = threadIdx.x; hidden < a.hidden;
       hidden += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < a.dispatch.identity.top_k; ++order) {
      const auto original = a.dispatch.combine_order[
          token * a.dispatch.identity.top_k + order];
      const auto route = token * a.dispatch.identity.top_k + original;
      sum = gate_after_accumulate(
          sum, a.gates[route],
          contribution[route * a.hidden + hidden].bits);
    }
    combined[token * a.hidden + hidden] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
  if (threadIdx.x == 0)
    atomicAdd(&status->reduced_tokens, 1ULL);
}

__global__ void local_p2p_finish_kernel(LocalP2pKernelArgs a) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
    status->complete_clock = device_globaltimer_ns();
  }
}
#endif

__device__ std::uint32_t forward_fixture_mix32(std::uint32_t value) {
  value ^= value >> 16U;
  value *= 0x7feb352dU;
  value ^= value >> 15U;
  value *= 0x846ca68bU;
  value ^= value >> 16U;
  return value;
}

__device__ std::uint16_t forward_fixture_weight_bits(
    std::uint32_t expert, std::uint32_t tensor_kind, std::uint64_t row,
    std::uint64_t column) {
  auto key = forward_fixture_mix32(expert ^ 0x9e3779b9U);
  key ^= forward_fixture_mix32(tensor_kind ^ 0x85ebca6bU);
  key ^= forward_fixture_mix32(static_cast<std::uint32_t>(row) ^
                               0xc2b2ae35U);
  key ^= forward_fixture_mix32(static_cast<std::uint32_t>(column) ^
                               0x27d4eb2fU);
  const auto signed_value =
      static_cast<int>(forward_fixture_mix32(key) & 1U) * 2 - 1;
  return float_to_bf16_bits(static_cast<float>(signed_value) *
                            (1.0F / 256.0F));
}

constexpr std::uint64_t kWeightHashChunkElements = 1ULL << 22U;
constexpr std::uint64_t kWeightHashSeedLow = 0x243f6a8885a308d3ULL;
constexpr std::uint64_t kWeightHashSeedHigh = 0x13198a2e03707344ULL;
constexpr std::uint64_t kWeightHashMultiplierLow = 0x9e3779b185ebca87ULL;
constexpr std::uint64_t kWeightHashMultiplierHigh = 0xc2b2ae3d27d4eb4fULL;

struct WeightHashPair {
  std::uint64_t low;
  std::uint64_t high;
};

__device__ std::uint32_t weight_hash_mix(std::uint32_t value) {
  return forward_fixture_mix32(value);
}

__global__ void hash_forward_weight_chunks_kernel(
    const transport::BFloat16 *values, std::uint64_t elements,
    WeightHashPair *chunks) {
  __shared__ std::uint64_t low[256];
  __shared__ std::uint64_t high[256];
  const auto begin = static_cast<std::uint64_t>(blockIdx.x) *
                     kWeightHashChunkElements;
  const auto end = min(elements, begin + kWeightHashChunkElements);
  std::uint64_t local_low = 0;
  std::uint64_t local_high = 0;
  for (auto index = begin + threadIdx.x; index < end; index += blockDim.x) {
    const auto bits = static_cast<std::uint32_t>(values[index].bits);
    const auto logical_index = static_cast<std::uint32_t>(index);
    const auto mixed_low = weight_hash_mix(
        bits ^ (logical_index * 0x9e3779b1U) ^ 0xa5a5f00dU);
    const auto mixed_high = weight_hash_mix(
        (bits << 16U) ^ (logical_index * 0x85ebca77U) ^ 0x5a5a0ff1U);
    local_low += static_cast<std::uint64_t>(mixed_low) *
                 kWeightHashMultiplierLow;
    local_high += static_cast<std::uint64_t>(mixed_high) *
                  kWeightHashMultiplierHigh;
  }
  low[threadIdx.x] = local_low;
  high[threadIdx.x] = local_high;
  __syncthreads();
  for (std::uint32_t offset = blockDim.x / 2U; offset != 0U; offset /= 2U) {
    if (threadIdx.x < offset) {
      low[threadIdx.x] += low[threadIdx.x + offset];
      high[threadIdx.x] += high[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    chunks[blockIdx.x] = {low[0], high[0]};
}

__global__ void finalize_forward_weight_hash_kernel(
    const WeightHashPair *chunks, std::uint64_t chunk_count,
    std::uint64_t elements, WeightHashPair *result) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  WeightHashPair value{
      kWeightHashSeedLow ^ (elements * 0x100000001b3ULL),
      kWeightHashSeedHigh ^ (elements * 0x9e3779b97f4a7c15ULL)};
  for (std::uint64_t chunk = 0; chunk < chunk_count; ++chunk) {
    value.low += chunks[chunk].low;
    value.high += chunks[chunk].high;
  }
  *result = value;
}

__global__ void initialize_forward_weights_kernel(
    transport::BFloat16 *w1, std::uint64_t w1_elements,
    transport::BFloat16 *w2, std::uint64_t w2_elements,
    std::uint32_t global_expert_begin, std::uint64_t hidden,
    std::uint64_t intermediate) {
  const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < w1_elements; index += stride) {
    const auto expert_stride = 2ULL * intermediate * hidden;
    const auto expert = global_expert_begin +
                        static_cast<std::uint32_t>(index / expert_stride);
    const auto within = index % expert_stride;
    w1[index].bits = forward_fixture_weight_bits(
        expert, 1U, within / hidden, within % hidden);
  }
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < w2_elements; index += stride) {
    const auto expert_stride = hidden * intermediate;
    const auto expert = global_expert_begin +
                        static_cast<std::uint32_t>(index / expert_stride);
    const auto within = index % expert_stride;
    w2[index].bits = forward_fixture_weight_bits(
        expert, 2U, within / intermediate, within % intermediate);
  }
}

#include "cuda/forward/kernel_contract.cuh"
