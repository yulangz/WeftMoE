#include "cuda/activation_transport_internal.hpp"

#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#if __has_include(<nccl_device/core.h>)
#include <nccl_device/core.h>
#include <nccl_device/impl/comm__types.h>
#endif
#if __has_include(<nccl_device.h>)
#include <nccl_device.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <sstream>
#include <thread>

#include "cuda/cuda_internal.hpp"
#include "cuda/profiling/device_timeline.cuh"
#include "cuda/testing/harnesses.hpp"
#include "cuda/transport_generation_completion.cuh"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
struct ActivationMegaKernelBackendState;
namespace {

constexpr std::size_t kAlign = 256;

[[noreturn]] void transport_error(NotifyErrorCode code, const char *phase,
                                  const char *field, std::uint32_t rank,
                                  const std::string &expected,
                                  const std::string &actual) {
  throw NotifyDispatchError(
      code, std::string("phase=") + phase + ", field=" + field +
                ", rank=" + std::to_string(rank) + ", expected=" + expected +
                ", actual=" + actual);
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
                              const char *field) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX || box_outer == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-tma-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
  const cuuint32_t box[2] = {64U, box_outer};
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

CUtensorMap make_bf16_tma_map_mn(void *base, std::uint64_t inner,
                                 std::uint64_t outer,
                                 const char *field) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "backward-tma-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
  // MN-major BF16 WGMMA uses 128-byte swizzle atoms: 64 contiguous MN
  // elements by one 64-row K slice.  A 128-column dW tile is assembled from
  // two adjacent atoms by the TMA producer below.
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

CUtensorMap make_bf16_tma_store_map(void *base, std::uint64_t inner,
                                    std::uint64_t outer,
                                    const char *field) {
  if (!base || inner == 0 || outer == 0 || inner > UINT32_MAX ||
      outer > UINT32_MAX)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "backward-tma-store-map", field, 0,
                    "nonzero uint32 dimensions and valid base", "invalid");
  CUtensorMap map{};
  const cuuint64_t dims[2] = {inner, outer};
  const cuuint64_t strides[1] = {
      checked_mul(inner, sizeof(transport::BFloat16), field)};
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
    transport_error(NotifyErrorCode::CudaFailure, "backward-tma-store-map",
                    field, 0, "cuTensorMapEncodeTiled success",
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

// Fixed, pointer-free record shared by the Forward and Backward production
// terminal protocol.  `epoch` is published last with system release ordering.
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
  Backward = 2,
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

// One directory entry is exchanged before a context starts publishing its
// M-tile waves.  It lets the peer consume context-local wire ranges without
// assuming that independently progressing GIN contexts complete in lockstep.
struct GinDispatchContextRecord {
  std::uint64_t generation;
  std::uint64_t epoch;
  std::uint64_t packet_begin;
  std::uint32_t packet_count;
  std::uint32_t wave_count;
};
static_assert(sizeof(GinDispatchContextRecord) == 32);

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

constexpr std::uint64_t kForwardReturnChunkRows = 64ULL;
[[maybe_unused]] constexpr std::uint32_t kForwardReturnCompletionBatch = 8U;

struct BackwardDeviceStatus {
  unsigned long long start_clock;
  unsigned long long dy_dispatch_complete_clock;
  unsigned long long dv_complete_clock;
  unsigned long long dgate_complete_clock;
  unsigned long long transpose_complete_clock;
  unsigned long long dx_complete_clock;
  unsigned long long dw1_complete_clock;
  unsigned long long dw2_complete_clock;
  unsigned long long dx_reduce_complete_clock;
  unsigned long long complete_clock;
  unsigned long long dv_start_clock;
  unsigned long long dx_start_clock;
  unsigned long long dw1_first_ready_clock;
  unsigned long long dw1_first_start_clock;
  unsigned long long dw1_last_complete_clock;
  unsigned long long dw2_first_ready_clock;
  unsigned long long dw2_first_start_clock;
  unsigned long long dw2_last_complete_clock;
  unsigned long long dw1_max_ready_wait;
  unsigned long long dw2_max_ready_wait;
  unsigned long long dv_tasks;
  unsigned long long dx_tasks;
  unsigned long long dw1_tasks;
  unsigned long long dw2_tasks;
  unsigned long long wgmma_instructions;
  unsigned long long tma_loads;
  unsigned long long actual_dy_peer_bytes;
  unsigned long long actual_dx_peer_bytes;
  unsigned long long gin_dispatch_put_count;
  unsigned long long gin_return_put_count;
  unsigned long long gin_signal_count;
  unsigned long long gin_metadata_bytes;
  unsigned long long raw_cross_node_dy_bytes;
  unsigned long long gin_dispatch_payload_bytes;
  unsigned long long gin_return_payload_bytes;
  unsigned long long padded_reduction_rows;
  unsigned int error;
  unsigned int error_field;
  unsigned int error_origin_rank;
  unsigned int error_code;
  unsigned long long error_expected;
  unsigned long long error_actual;
};

struct BackwardExpertDwTimeline {
  unsigned long long dv_tasks_complete;
  unsigned long long ready_clock;
  unsigned long long dw1_first_start_clock;
  unsigned long long dw1_tasks_complete;
  unsigned long long dw1_complete_clock;
  unsigned long long dw2_first_start_clock;
  unsigned long long dw2_tasks_complete;
  unsigned long long dw2_complete_clock;
};

enum BackwardProgressWord : std::uint32_t {
  kBackwardEntered = 0,
  kBackwardSetupComplete = 1,
  kBackwardDyComplete = 2,
  kBackwardDvStarted = 3,
  kBackwardDvComplete = 4,
  kBackwardDgateComplete = 5,
  kBackwardDxStarted = 6,
  kBackwardDxComplete = 7,
  kBackwardDw2Started = 8,
  kBackwardDw2Complete = 9,
  kBackwardDw1Started = 10,
  kBackwardDw1Complete = 11,
  kBackwardReduceComplete = 12,
  kBackwardTerminal = 13,
  kBackwardHashBase = 16,
  kBackwardProgressExpertBase = 32,
};

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

#if MEGAMOE_DEBUG_VALIDATE
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
      "dispatch", "relay",  "fc1_or_dv", "fc2_or_dx", "return",
      "arrival",  "reduce", "dw1",       "dw2"};
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
  output << "} " << prefix
         << ".kernel_complete_epoch=" << word(kTail + 31ULL) << ' ' << prefix
         << ".terminal_publish_count=" << word(kTail + 32ULL);
#else
  output << ' ' << prefix
         << ".kernel_complete_epoch=" << word(kTail + 23ULL) << ' ' << prefix
         << ".terminal_publish_count=" << word(kTail + 24ULL);
#endif
  for (std::uint32_t context = 0; context < contexts; ++context) {
    const auto base = transport_state::kGenerationCompletionWords +
                      context * transport_state::kContextCompletionWords;
    output << ' ' << prefix << ".context[" << context << "]={directory="
           << word(base + 2ULL) << ",dispatch=" << word(base + 3ULL) << '/'
           << word(base + 5ULL) << ",relay=" << word(base + 6ULL) << '/'
           << word(base + 8ULL) << ",return=" << word(base + 9ULL) << '/'
           << word(base + 11ULL) << ",inflight=" << word(base + 12ULL)
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
  std::size_t contribution = 0;
  std::size_t combine_return = 0;
  std::size_t combined = 0;
  std::size_t socket_count_matrix = 0;
  std::size_t socket_header_send = 0;
  std::size_t socket_header_receive = 0;
  std::size_t socket_payload_receive = 0;
  std::size_t forward_s = 0;
  std::size_t forward_ab_cache = 0;
  std::size_t forward_task_claim = 0;
  std::size_t forward_fc1_done = 0;
  std::size_t forward_x_arrived_rows = 0;
  std::size_t forward_output_ready_epoch = 0;
  std::size_t forward_return_chunk_directory = 0;
  std::size_t forward_return_chunk_state = 0;
  std::size_t forward_return_chunk_ready_count = 0;
  std::size_t forward_return_fragment_index = 0;
  std::size_t forward_return_source_compact_index = 0;
  std::uint64_t forward_return_chunk_capacity = 0;
  std::size_t forward_scheduler = 0;
  std::size_t forward_completion = 0;
  std::size_t forward_context_completion = 0;
  std::size_t backward_dy_input = 0;
  std::size_t backward_forward_output = 0;
  std::size_t backward_dy_canonical = 0;
  std::size_t backward_canonical_gate = 0;
  std::size_t backward_dab = 0;
  std::size_t backward_recompute_ab = 0;
  std::size_t backward_dx_canonical = 0;
  std::size_t backward_dx_return = 0;
  std::size_t backward_dx_combined = 0;
  std::size_t backward_dgate_partial = 0;
  std::size_t backward_dgate_canonical = 0;
  std::size_t backward_dgate = 0;
  std::size_t backward_u_transpose = 0;
  std::size_t backward_dab_transpose = 0;
  std::size_t backward_dw1 = 0;
  std::size_t backward_dw2 = 0;
  std::size_t backward_scheduler = 0;
  std::size_t backward_completion = 0;
  std::size_t backward_context_completion = 0;
  std::size_t backward_dy_arrived_rows = 0;
  std::size_t backward_dw_timeline = 0;
  std::size_t backward_validation_status = 0;
  std::size_t backward_status = 0;
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
  std::size_t send_gate = 0;
  std::size_t receive_gate = 0;
  std::size_t send_payload = 0;
  std::size_t receive_payload = 0;
  std::size_t return_staging = 0;
  std::size_t return_receive = 0;
  std::size_t return_gate_staging = 0;
  std::size_t return_gate_receive = 0;
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

TimelineAllocationLayout make_timeline_allocation_layout(
    std::uint32_t max_ctas) {
  TimelineAllocationLayout layout;
  std::size_t cursor = 0;
  layout.counts = append(cursor, max_ctas, sizeof(std::uint32_t),
                         "timeline CTA counts");
  layout.overflow = append(cursor, max_ctas, sizeof(std::uint32_t),
                           "timeline CTA overflow");
  layout.records = append(
      cursor,
      checked_mul(max_ctas, profiling::kEventsPerCta,
                  "timeline records"),
      sizeof(profiling::DeviceTimelineRecord), "timeline records");
  layout.total = cursor;
  return layout;
}
#endif

GinWindowLayout make_gin_window_layout(
    const ActivationTransportPoolConfig &config) {
  GinWindowLayout layout;
  if (!config.enable_gin_rdma)
    return layout;
  std::size_t cursor = 0;
  const auto routes = checked_mul(config.max_tokens_per_rank,
                                  config.max_top_k, "GIN max routes");
  const auto outgoing_packets = checked_mul(
      config.max_nodes, config.max_tokens_per_rank,
      "GIN outgoing packet capacity");
  const auto outgoing_fanout = checked_mul(
      outgoing_packets, config.max_top_k,
      "GIN outgoing packet fanout capacity");
  const auto remote_nodes = config.max_nodes > 0 ? config.max_nodes - 1U : 0U;
  constexpr std::uint32_t kGinContextsPerRail = 8;
  const auto directory_records = checked_mul(
      remote_nodes, kGinContextsPerRail, "GIN dispatch directory records");
  const auto incoming_packets = checked_mul(
      remote_nodes, config.max_tokens_per_rank,
      "GIN incoming packet capacity");
  const auto packet_payload = checked_mul(outgoing_packets,
                                          config.max_hidden_size,
                                          "GIN send payload elements");
  const auto incoming_payload = checked_mul(incoming_packets,
                                            config.max_hidden_size,
                                            "GIN receive payload elements");
  const auto return_payload = checked_mul(routes, config.max_hidden_size,
                                          "GIN return payload elements");
  layout.send_dispatch_directory = append(
      cursor, directory_records, sizeof(GinDispatchContextRecord),
      "GIN send dispatch directory");
  layout.receive_dispatch_directory = append(
      cursor, directory_records, sizeof(GinDispatchContextRecord),
      "GIN receive dispatch directory");
  layout.send_packet_headers = append(
      cursor, outgoing_packets, sizeof(LocalP2pRouteHeader),
      "GIN send packet headers");
  layout.receive_packet_headers = append(
      cursor, incoming_packets, sizeof(LocalP2pRouteHeader),
      "GIN receive packet headers");
  // Packet descriptors are materialized in transport order. A token can
  // contribute one node-deduplicated packet to every node, so packet-position
  // indexing requires nodes*tokens*top-k capacity even though the number of
  // logical routes remains tokens*top-k. The previous route-sized allocation
  // made the first remote-node packet range alias send_gate/send_payload.
  layout.send_fanout = append(cursor, outgoing_fanout,
                              sizeof(DeviceFanoutRecord), "GIN send fanout");
  layout.receive_fanout = append(
      cursor, checked_mul(incoming_packets, config.max_top_k,
                          "GIN receive fanout capacity"),
      sizeof(DeviceFanoutRecord), "GIN receive fanout");
  layout.send_gate = append(cursor, outgoing_fanout, sizeof(float),
                            "GIN send route gates");
  layout.receive_gate = append(
      cursor, checked_mul(incoming_packets, config.max_top_k,
                          "GIN receive gate capacity"),
      sizeof(float), "GIN receive route gates");
  layout.send_payload = append(cursor, packet_payload,
                               sizeof(transport::BFloat16),
                               "GIN send payload");
  layout.receive_payload = append(cursor, incoming_payload,
                                  sizeof(transport::BFloat16),
                                  "GIN receive payload");
  layout.return_staging = append(
      cursor, checked_mul(remote_nodes, return_payload,
                          "GIN return staging elements"),
                                 sizeof(transport::BFloat16),
                                 "GIN return staging");
  layout.return_receive = append(cursor, return_payload,
                                 sizeof(transport::BFloat16),
                                 "GIN return receive");
  layout.return_gate_staging = append(
      cursor, checked_mul(remote_nodes, routes,
                          "GIN return gate staging elements"),
      sizeof(float), "GIN return gate staging");
  layout.return_gate_receive = append(cursor, routes, sizeof(float),
                                      "GIN return gate receive");
  layout.barrier = append(cursor, 1, sizeof(std::uint64_t), "GIN barrier");
  layout.production_error_records = append(
      cursor, config.max_world_size, sizeof(ProductionStatusRecord),
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
  // Four pool-private pointer slots follow the world-rank table. They carry
  // completion-vector pointers without widening the frozen kernel args.
  layout.peer_bases = append(cursor, config.max_world_size + 4ULL,
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
        cursor, routes, sizeof(LocalP2pRouteHeader), "Socket send headers");
    layout.socket_header_receive = append(
        cursor, config.max_destination_rows_per_rank,
        sizeof(LocalP2pRouteHeader), "Socket receive headers");
    layout.socket_payload_receive = append(
        cursor, checked_mul(config.max_destination_rows_per_rank,
                            config.max_hidden_size,
                            "Socket receive payload elements"),
        sizeof(transport::BFloat16), "Socket receive payload");
  }
  if (config.max_intermediate_size != 0) {
    const auto forward_elements = checked_mul(
        config.max_destination_rows_per_rank,
        config.max_intermediate_size, "forward S elements");
    layout.forward_s = append(cursor, forward_elements,
                              sizeof(transport::BFloat16), "forward S");
    if (config.enable_forward_cache)
      layout.forward_ab_cache = append(
          cursor, checked_mul(forward_elements, 2, "forward A/B elements"),
          sizeof(transport::BFloat16), "forward A/B cache");
    // Expert segmentation can make every incoming row a distinct partial
    // M-block.  Capacity must therefore be bounded by rows, not by a dense
    // ceil(rows/64) estimate.
    const auto max_mblocks = config.max_destination_rows_per_rank;
    const auto fc1_ntiles =
        (config.max_intermediate_size + 127ULL) / 128ULL;
    const auto fc2_ntiles =
        (config.max_hidden_size + 255ULL) / 256ULL;
    layout.forward_task_claim = append(
        cursor,
        checked_mul(max_mblocks, 1ULL + fc1_ntiles + fc2_ntiles,
                    "forward task claims"),
        sizeof(std::uint64_t), "forward task claims");
    layout.forward_fc1_done = append(cursor, max_mblocks,
                                     sizeof(std::uint64_t),
                                     "forward FC1 done tiles");
    // A destination tile is produced concurrently by every node-local rail.
    // Keep one row counter per local-rank/rail instead of issuing contended
    // system-scope atomics from several GPUs into one peer word. The consumer
    // sums the fixed rail vector with acquire loads. This is both deterministic
    // and avoids relying on peer-atomic combining behavior of a particular
    // NVLink/virtualization topology.
    layout.forward_x_arrived_rows = append(
        cursor,
        checked_mul(max_mblocks, config.max_world_size,
                    "forward exact X M-tile rail counters"),
        sizeof(std::uint64_t), "forward exact X M-tile arrived rows");
    layout.forward_output_ready_epoch = append(
        cursor, config.max_destination_rows_per_rank, sizeof(std::uint64_t),
        "forward output row ready epochs");
    // A discontinuity in the canonical reverse-row space may terminate a
    // chunk.  The safe O(routes) bound is therefore one directory record per
    // destination row rather than ceil(rows/chunk_rows).
    layout.forward_return_chunk_capacity =
        config.max_destination_rows_per_rank;
    layout.forward_return_chunk_directory = append(
        cursor, layout.forward_return_chunk_capacity,
        sizeof(ForwardReturnChunkRecord),
        "forward return chunk directory");
    layout.forward_return_chunk_state = append(
        cursor, layout.forward_return_chunk_capacity, sizeof(std::uint64_t),
        "forward return chunk state");
    layout.forward_return_chunk_ready_count = append(
        cursor,
        checked_mul(layout.forward_return_chunk_capacity,
                    config.max_world_size,
                    "forward return chunk producer rail counters"),
        sizeof(std::uint64_t),
        "forward return chunk ready count");
    layout.forward_return_fragment_index = append(
        cursor, config.max_destination_rows_per_rank, sizeof(std::uint64_t),
        "forward return compact fragment index");
    layout.forward_return_source_compact_index = append(
        cursor, routes, sizeof(std::uint64_t),
        "forward source compact return index");
    layout.forward_scheduler = append(
        cursor, 4 + config.max_world_size * 32ULL,
        sizeof(std::uint64_t), "forward scheduler");
    layout.forward_completion = append(
        cursor, 1, sizeof(transport_state::GenerationCompletionVector),
        "forward generation completion vector");
    const auto max_context_records = checked_mul(
        config.max_nodes > 0 ? config.max_nodes - 1ULL : 0ULL, 8ULL,
        "forward GIN context completion records");
    layout.forward_context_completion = append(
        cursor, max_context_records,
        sizeof(transport_state::ContextCompletionRecord),
        "forward GIN context completion records");
    if (config.enable_backward) {
      const auto rows = config.max_destination_rows_per_rank;
      const auto hidden = config.max_hidden_size;
      const auto intermediate = config.max_intermediate_size;
      const auto route_elements = checked_mul(routes, hidden,
                                              "backward route elements");
      const auto row_hidden = checked_mul(rows, hidden,
                                          "backward row hidden elements");
      const auto row_intermediate = checked_mul(
          rows, intermediate, "backward row intermediate elements");
      layout.backward_dy_input = append(
          cursor, checked_mul(config.max_tokens_per_rank, hidden,
                              "backward dY input elements"),
          sizeof(transport::BFloat16), "backward dY input");
      layout.backward_forward_output = append(
          cursor, row_hidden, sizeof(transport::BFloat16),
          "Backward-enabled Forward FC2 output rows");
      layout.backward_dy_canonical = append(
          cursor, row_hidden, sizeof(transport::BFloat16),
          "backward canonical dY");
      layout.backward_canonical_gate = append(
          cursor, rows, sizeof(float), "backward canonical gate");
      layout.backward_dab = append(
          cursor, checked_mul(row_intermediate, 2ULL, "backward dAB"),
          sizeof(transport::BFloat16), "backward dAB");
      layout.backward_recompute_ab = append(
          cursor, checked_mul(row_intermediate, 2ULL,
                              "backward recompute A/B"),
          sizeof(transport::BFloat16), "backward recompute A/B");
      layout.backward_dx_canonical = append(
          cursor, row_hidden, sizeof(transport::BFloat16),
          "backward canonical dX");
      layout.backward_dx_return = append(
          cursor, route_elements, sizeof(transport::BFloat16),
          "backward dX return");
      layout.backward_dx_combined = append(
          cursor, checked_mul(config.max_tokens_per_rank, hidden,
                              "backward combined dX"),
          sizeof(transport::BFloat16), "backward combined dX");
      const auto intermediate_tiles = (intermediate + 255ULL) / 256ULL;
      layout.backward_dgate_partial = append(
          cursor, checked_mul(rows, intermediate_tiles,
                              "backward dGate partials"),
          sizeof(float), "backward dGate partials");
      layout.backward_dgate_canonical = append(
          cursor, rows, sizeof(float), "backward canonical dGate");
      layout.backward_dgate = append(cursor, routes, sizeof(float),
                                     "backward dGate");
      if (config.max_owned_experts == 0)
        transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                        "transport-pool-query", "max_owned_experts", 0,
                        "> 0 when enable_backward", "0");
      const auto max_owned_experts = config.max_owned_experts;
      const auto reduction_rows =
          (rows + checked_mul(max_owned_experts, 63ULL,
                              "backward padded reduction rows") +
           63ULL) &
          ~63ULL;
      layout.backward_u_transpose = append(
          cursor, checked_mul(reduction_rows, intermediate,
                              "backward padded U transpose"),
          sizeof(transport::BFloat16),
          "backward U transpose");
      layout.backward_dab_transpose = append(
          cursor, checked_mul(reduction_rows, 2ULL * intermediate,
                              "backward padded dAB transpose"),
          sizeof(transport::BFloat16), "backward dAB transpose");
      layout.backward_dw1 = append(
          cursor,
          checked_mul(checked_mul(max_owned_experts, 2ULL * intermediate,
                                  "backward dW1 rows"),
                      hidden, "backward dW1 elements"),
          sizeof(transport::BFloat16), "backward dW1");
      layout.backward_dw2 = append(
          cursor,
          checked_mul(checked_mul(max_owned_experts, hidden,
                                  "backward dW2 rows"),
                      intermediate, "backward dW2 elements"),
          sizeof(transport::BFloat16), "backward dW2");
      layout.backward_scheduler = append(
          cursor, 64ULL + reduction_rows, sizeof(std::uint64_t),
          "backward scheduler/readiness");
      layout.backward_completion = append(
          cursor, 1, sizeof(transport_state::GenerationCompletionVector),
          "backward generation completion vector");
      layout.backward_context_completion = append(
          cursor, max_context_records,
          sizeof(transport_state::ContextCompletionRecord),
          "backward GIN context completion records");
      layout.backward_dy_arrived_rows = append(
          cursor, max_mblocks, sizeof(std::uint64_t),
          "backward exact dY M-tile arrived rows");
      layout.backward_dw_timeline = append(
          cursor, max_owned_experts, sizeof(BackwardExpertDwTimeline),
          "backward per-expert dW timeline");
      layout.backward_validation_status = append(
          cursor, checked_mul(config.max_world_size, 8ULL,
                              "backward terminal validation words"),
          sizeof(std::uint64_t), "backward terminal validation status");
      layout.backward_status = append(cursor, 1, sizeof(BackwardDeviceStatus),
                                      "backward status");
    }
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
  void *backward_timeline = nullptr;
#endif
  TransportDeviceStatus *host_status = nullptr;
  TransportDeviceStatus *host_forward_status = nullptr;
  ProductionStatusRecord *host_production_status = nullptr;
  std::uint64_t *host_backward_progress = nullptr;
  std::uint64_t *device_backward_progress = nullptr;
  std::size_t backward_progress_words = 0;
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
  cudaEvent_t dispatch_complete = nullptr;
  cudaEvent_t complete = nullptr;
  cudaEvent_t backward_ready = nullptr;
  SlotState state = SlotState::Available;
  std::uint64_t epoch = 0;
  std::uint64_t transport_epoch = 0;
  bool completion_recorded = false;
  CUtensorMap forward_x_map{};
  CUtensorMap forward_s_map{};
  CUtensorMap backward_dy_map{};
  CUtensorMap backward_dab_map{};
  CUtensorMap backward_dy_mn_map{};
  CUtensorMap backward_x_mn_map{};
  CUtensorMap backward_u_transpose_map{};
  CUtensorMap backward_dab_transpose_map{};
  CUtensorMap backward_dw1_store_map{};
  bool forward_maps_valid = false;
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

  ~ActivationPoolState() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
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
      if (slot.dispatch_complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.dispatch_complete)
        (void)cudaEventDestroy(slot.dispatch_complete);
      if (slot.complete)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.complete)
        (void)cudaEventDestroy(slot.complete);
      if (slot.backward_ready)
        internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      if (slot.backward_ready)
        (void)cudaEventDestroy(slot.backward_ready);
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
      if (slot.host_backward_progress)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_backward_progress)
        (void)cudaFreeHost(slot.host_backward_progress);
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
      if (slot.backward_timeline)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.backward_timeline)
        (void)cudaFree(slot.backward_timeline);
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
    std::uint64_t hidden, std::uint64_t epoch,
    const unsigned int *route_ids, LocalP2pRouteHeader *headers,
    transport::BFloat16 *payload) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= dispatch.sizes.route_count)
    return;
  const auto route = static_cast<std::uint64_t>(route_ids[packed]);
  const auto token = route / dispatch.identity.top_k;
  const auto slot = static_cast<std::uint32_t>(route % dispatch.identity.top_k);
  if (threadIdx.x == 0) {
    headers[packed] = LocalP2pRouteHeader{
        transport::kActivationTransportSchemaVersion,
        dispatch.identity.calling_rank,
        dispatch.route_dst_rank[route], 0U, dispatch.identity.generation,
        epoch, packed, route, token, dispatch.route_dst_row[route],
        dispatch.expert_id[route], slot};
  }
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    payload[packed * hidden + h] = input[token * hidden + h];
}

__global__ void socket_relay_transport_kernel(
    LocalP2pRouteHeader *headers, transport::BFloat16 *compact_payload,
    transport::BFloat16 *canonical_rows, std::uint64_t incoming,
    std::uint64_t hidden, std::uint64_t generation,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    bool identity_expert, TransportDeviceStatus *status) {
  const auto incoming_index = static_cast<std::uint64_t>(blockIdx.x);
  if (incoming_index >= incoming)
    return;
  const auto header = headers[incoming_index];
  const bool valid =
      header.schema_version == transport::kActivationTransportSchemaVersion &&
      header.generation == generation &&
      header.destination_rank == destination_rank &&
      header.dst_row < max_rows;
  if (!valid) {
    if (threadIdx.x == 0)
      publish_local_structured_error(status, destination_rank, 40U, 1U,
                                     max_rows, header.dst_row);
    return;
  }
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    const auto value = compact_payload[incoming_index * hidden + h];
    canonical_rows[header.dst_row * hidden + h] = value;
    compact_payload[incoming_index * hidden + h] =
        synthetic_value(value, header.expert_id, h, identity_expert);
  }
}

__global__ void socket_install_forward_headers_kernel(
    const LocalP2pRouteHeader *compact_headers,
    LocalP2pRouteHeader *canonical_headers, std::uint64_t incoming,
    std::uint64_t generation, std::uint64_t epoch,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    TransportDeviceStatus *status) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= incoming)
    return;
  auto header = compact_headers[index];
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
    const LocalP2pRouteHeader *headers,
    const transport::BFloat16 *canonical_output,
    transport::BFloat16 *compact, std::uint64_t incoming,
    std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= incoming)
    return;
  const auto row = headers[packed].dst_row;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    compact[packed * hidden + h] = canonical_output[row * hidden + h];
}

__global__ void socket_pack_route_gates_kernel(
    const unsigned int *route_ids, const float *gates, float *compact,
    std::uint64_t routes) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (packed < routes)
    compact[packed] = gates[route_ids[packed]];
}

__global__ void socket_relay_backward_inputs_kernel(
    LocalP2pRouteHeader *headers,
    const transport::BFloat16 *compact_payload, const float *compact_gate,
    transport::BFloat16 *canonical_dy, float *canonical_gate,
    LocalP2pRouteHeader *canonical_headers, std::uint64_t incoming,
    std::uint64_t hidden, std::uint64_t generation, std::uint64_t epoch,
    std::uint32_t destination_rank, std::uint64_t max_rows,
    BackwardDeviceStatus *status) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= incoming)
    return;
  auto header = headers[packed];
  if (header.schema_version != transport::kActivationTransportSchemaVersion ||
      header.generation != generation ||
      header.destination_rank != destination_rank ||
      header.dst_row >= max_rows) {
    if (threadIdx.x == 0)
      publish_local_structured_error(status, destination_rank, 241U, 1U,
                                     max_rows, header.dst_row);
    return;
  }
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    canonical_dy[header.dst_row * hidden + h] =
        compact_payload[packed * hidden + h];
  if (threadIdx.x == 0) {
    canonical_gate[header.dst_row] = compact_gate[packed];
    header.source_epoch = epoch;
    canonical_headers[header.dst_row] = header;
  }
}

__global__ void socket_gather_backward_returns_kernel(
    const LocalP2pRouteHeader *headers,
    const transport::BFloat16 *canonical_dx, const float *canonical_dgate,
    transport::BFloat16 *compact_dx, float *compact_dgate,
    std::uint64_t incoming, std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= incoming)
    return;
  const auto row = headers[packed].dst_row;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    compact_dx[packed * hidden + h] = canonical_dx[row * hidden + h];
  if (threadIdx.x == 0)
    compact_dgate[packed] = canonical_dgate[row];
}

__global__ void socket_scatter_backward_returns_kernel(
    const LocalP2pRouteHeader *headers,
    const transport::BFloat16 *compact_dx, const float *compact_dgate,
    transport::BFloat16 *dx_by_route, float *dgate_by_route,
    std::uint32_t *coverage, std::uint64_t routes, std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= routes)
    return;
  const auto route = headers[packed].route_index;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x)
    dx_by_route[route * hidden + h] = compact_dx[packed * hidden + h];
  if (threadIdx.x == 0) {
    dgate_by_route[route] = compact_dgate[packed];
    if (coverage != nullptr)
      coverage[route] = 1U;
  }
}

__global__ void socket_backward_reduce_kernel(
    DeviceDispatchKernelView dispatch,
    const transport::BFloat16 *dx_by_route,
    transport::BFloat16 *dx_combined, std::uint64_t hidden) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x);
  if (token >= dispatch.identity.num_tokens)
    return;
  for (std::uint64_t h = threadIdx.x; h < hidden; h += blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t order = 0; order < dispatch.identity.top_k; ++order) {
      const auto slot = dispatch.combine_order[
          token * dispatch.identity.top_k + order];
      const auto route = token * dispatch.identity.top_k + slot;
      sum = __fadd_rn(sum, bf16_bits_to_float(
                               dx_by_route[route * hidden + h].bits));
    }
    dx_combined[token * hidden + h] =
        transport::BFloat16{float_to_bf16_bits(sum)};
  }
}

__global__ void socket_scatter_returns_kernel(
    const LocalP2pRouteHeader *headers,
    const transport::BFloat16 *compact,
    transport::BFloat16 *by_route, std::uint64_t routes,
    std::uint64_t hidden) {
  const auto packed = static_cast<std::uint64_t>(blockIdx.x);
  if (packed >= routes)
    return;
  const auto route = headers[packed].route_index;
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

template <typename T>
__device__ T *peer_at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

template <bool StaticLocal, typename Args, typename Status>
__device__ __forceinline__ void publish_production_terminal_status(
    const Args &a, Status *status, ProductionStatusPhase phase,
    std::size_t ordinary_record_offset,
    std::size_t gin_record_offset) {
  if (blockIdx.x != 0U)
    return;
  const auto local_world = a.local_world_size == 0U ? a.world
                                                     : a.local_world_size;
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
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == node)
          continue;
        gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                gin_record_offset +
                    static_cast<std::size_t>(a.rank) * sizeof(local),
                a.gin_window,
                gin_record_offset +
                    static_cast<std::size_t>(a.rank) * sizeof(local),
                sizeof(local), ncclGin_SignalInc{2}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
      }
      gin.waitSignal(ncclCoopThread{}, 2,
                     a.dispatch.identity.num_nodes - 1U);
    }
    __syncthreads();
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

  // The trusted single-node release path completes the fixed status agreement
  // on device. Every rank sees the same peer-owned record matrix and applies
  // the same rank/field ordering before the host observes its local status.
  if constexpr (StaticLocal) {
    if (threadIdx.x == 0U) {
      ProductionStatusRecord selected{};
      bool has_error = false;
      const auto expected_phase = static_cast<unsigned long long>(phase);
      for (std::uint32_t source = 0; source < observed_world; ++source) {
        auto record = observed_records[source];
        if (record.generation != a.dispatch.identity.generation) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase, source, 2ULL, 1ULL,
                                          a.dispatch.identity.generation,
                                          record.generation, a.epoch};
        } else if (record.phase != expected_phase) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase, source, 2ULL, 2ULL,
                                          expected_phase, record.phase,
                                          a.epoch};
        } else if (record.epoch != a.epoch) {
          record = ProductionStatusRecord{a.dispatch.identity.generation,
                                          expected_phase, source, 2ULL, 4ULL,
                                          a.epoch, record.epoch, a.epoch};
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
    std::size_t backward_dy_arrived_rows_offset, std::uint64_t mtile_count,
    std::size_t forward_completion_offset,
    std::size_t backward_completion_offset,
    std::size_t forward_context_completion_offset,
    std::size_t backward_context_completion_offset,
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
  auto *dy_arrived = backward_dy_arrived_rows_offset == 0
                         ? nullptr
                         : peer_at<unsigned long long>(
                               local_base, backward_dy_arrived_rows_offset);
  auto *forward_completion = peer_at<transport_state::GenerationCompletionVector>(
      local_base, forward_completion_offset);
  auto *backward_completion = backward_completion_offset == 0
                                  ? nullptr
                                  : peer_at<transport_state::GenerationCompletionVector>(
                                        local_base, backward_completion_offset);
  auto *forward_context = peer_at<transport_state::ContextCompletionRecord>(
      local_base, forward_context_completion_offset);
  auto *backward_context = backward_context_completion_offset == 0
                               ? nullptr
                               : peer_at<transport_state::ContextCompletionRecord>(
                                     local_base,
                                     backward_context_completion_offset);
  if (index == 0U) {
    peer_bases[world] = forward_completion;
    peer_bases[world + 1U] = forward_context;
    peer_bases[world + 2U] = backward_completion;
    peer_bases[world + 3U] = backward_context;
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
  for (std::uint64_t slot = index; slot < x_arrived_count;
       slot += blockDim.x) {
    x_arrived[slot] = 0ULL;
  }
  for (std::uint64_t tile = index; tile < mtile_count;
       tile += blockDim.x) {
    if (dy_arrived)
      dy_arrived[tile] = 0ULL;
  }
  if (index == 0U) {
    *forward_completion = transport_state::GenerationCompletionVector{};
    forward_completion->generation = generation;
    forward_completion->epoch = epoch;
    forward_completion->first_waiting_tile = ~0ULL;
    if (backward_completion != nullptr) {
      *backward_completion = transport_state::GenerationCompletionVector{};
      backward_completion->generation = generation;
      backward_completion->epoch = epoch;
      backward_completion->first_waiting_tile = ~0ULL;
    }
  }
  for (std::uint32_t context = index; context < context_completion_capacity;
       context += blockDim.x) {
    forward_context[context] = transport_state::ContextCompletionRecord{};
    forward_context[context].generation = generation;
    forward_context[context].epoch = epoch;
    if (backward_context != nullptr) {
      backward_context[context] = transport_state::ContextCompletionRecord{};
      backward_context[context].generation = generation;
      backward_context[context].epoch = epoch;
    }
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

__global__ void flip_validation_byte_kernel(unsigned char *storage,
                                            std::uint64_t byte_offset) {
  if (blockIdx.x == 0U && threadIdx.x == 0U)
    storage[byte_offset] ^= 1U;
}

__global__ void count_nonzero_bf16_kernel(
    const transport::BFloat16 *values, std::uint64_t elements,
    unsigned long long *nonzero_count) {
  unsigned long long local = 0ULL;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    local += values[index].bits != 0U ? 1ULL : 0ULL;
  if (local != 0ULL)
    atomicAdd(nonzero_count, local);
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

__global__ void transpose_forward_weights_for_backward_kernel(
    const transport::BFloat16 *w1, transport::BFloat16 *w1_transpose,
    const transport::BFloat16 *w2, transport::BFloat16 *w2_transpose,
    std::uint32_t owned_experts, std::uint64_t hidden,
    std::uint64_t intermediate) {
  const auto stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  const auto w1_per_expert = 2ULL * intermediate * hidden;
  const auto w2_per_expert = hidden * intermediate;
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < static_cast<std::uint64_t>(owned_experts) * w1_per_expert;
       index += stride) {
    const auto expert = index / w1_per_expert;
    const auto within = index % w1_per_expert;
    const auto row = within / hidden;
    const auto column = within % hidden;
    w1_transpose[expert * w1_per_expert + column * (2ULL * intermediate) +
                 row] = w1[index];
  }
  for (std::uint64_t index =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < static_cast<std::uint64_t>(owned_experts) * w2_per_expert;
       index += stride) {
    const auto expert = index / w2_per_expert;
    const auto within = index % w2_per_expert;
    const auto row = within / intermediate;
    const auto column = within % intermediate;
    w2_transpose[expert * w2_per_expert + column * hidden + row] = w2[index];
  }
}

#include "cuda/forward/kernel_contract.cuh"

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) // compiled production MMA path only
using ForwardMma =
    typename deep_gemm::mma::sm90::BF16MMASelector<128>::type;
using ForwardMmaWide =
    typename deep_gemm::mma::sm90::BF16MMASelector<256>::type;
using BackwardDwMma = typename deep_gemm::mma::sm90::BF16MMASelector<
    256, cute::UMMA::Major::MN, cute::UMMA::Major::MN>::type;
using BackwardDwMixedMma = typename deep_gemm::mma::sm90::BF16MMASelector<
    256, cute::UMMA::Major::K, cute::UMMA::Major::MN>::type;
using BackwardDwDirectMma = typename deep_gemm::mma::sm90::BF16MMASelector<
    256, cute::UMMA::Major::MN, cute::UMMA::Major::K>::type;
using ForwardBarrier = cutlass::arch::ClusterTransactionBarrier;

// FC1 consumes the same activation tile twice (the A and B projections of
// SwiGLU).  Keep both projection accumulators live and issue the two WGMMA
// groups from one TMA-loaded activation tile. Three pipeline stages fill the
// Hopper shared-memory union and hide the long 7168-element reduction loads;
// FC2 reinitializes its independent four-stage pipeline below.
__device__ void forward_wgmma_swiglu_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_map,
    std::uint32_t activation_row, std::uint32_t weight_a_row,
    std::uint32_t weight_b_row, std::uint32_t reduction,
    transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum_a, float *accum_b,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 3;
  constexpr std::uint32_t kMathWarpGroups = 2;
  constexpr std::uint32_t kStageElements = 128U * 64U;
  auto *smem_b2_public = smem_b_public + kStages * kStageElements;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
      accum_a[index] = 0.0F;
      accum_b[index] = 0.0F;
    }
  }
  const auto iterations = (reduction + 63U) / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      const auto k_base = iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements);
      auto *smem_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      auto *smem_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b2_public + stage * kStageElements);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &activation_map, &full_barriers[stage], smem_a, k_base,
          activation_row);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b1, k_base, weight_a_row);
      deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b2, k_base, weight_b_row);
      full_barriers[stage].arrive_and_expect_tx(
          3U * kStageElements * sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements +
          warpgroup * 64U * 64U);
      auto *smem_b1 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      auto *smem_b2 = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b2_public + stage * kStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
      }
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / ForwardMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
            smem_a + k * ForwardMma::K, 1);
        const auto desc_b1 = deep_gemm::mma::sm90::make_smem_desc(
            smem_b1 + k * ForwardMma::K, 1);
        const auto desc_b2 = deep_gemm::mma::sm90::make_smem_desc(
            smem_b2 + k * ForwardMma::K, 1);
        ForwardMma::wgmma(desc_a, desc_b1, accum_a, 1);
        ForwardMma::wgmma(desc_a, desc_b2, accum_b, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index) {
        deep_gemm::ptx::warpgroup_fence_operand(accum_a[index]);
        deep_gemm::ptx::warpgroup_fence_operand(accum_b[index]);
      }
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

__device__ void forward_wgmma_wide_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_map,
    std::uint32_t activation_row, std::uint32_t weight_row,
    std::uint32_t reduction, transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration,
    std::uint32_t reduction_start = 0U) {
  constexpr std::uint32_t kStages = 4;
  constexpr std::uint32_t kMathWarpGroups = 2;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 256U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups)
#pragma unroll
    for (int index = 0; index < ForwardMmaWide::kNumAccum; ++index)
      accum[index] = 0.0F;
  const auto iterations = (reduction + 63U) / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      const auto k_base = reduction_start + iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      deep_gemm::tma::copy<64, 256, 128, cutlass::bfloat16_t>(
          &activation_map, &full_barriers[stage], smem_a, k_base,
          activation_row);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b, k_base, weight_row);
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements +
          warpgroup * 64U * 64U);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < ForwardMmaWide::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / ForwardMmaWide::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
            smem_a + k * ForwardMmaWide::K, 1);
        const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
            smem_b + k * ForwardMmaWide::K, 1);
        ForwardMmaWide::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < ForwardMmaWide::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      if (iteration != 0U) {
        deep_gemm::ptx::warpgroup_wait<1>();
        const auto completed_pipeline = pipeline_iteration + iteration - 1U;
        if ((threadIdx.x & 31U) == 0U)
          empty_barriers[completed_pipeline % kStages].arrive();
      }
    }
    if (iterations != 0U) {
      deep_gemm::ptx::warpgroup_wait<0>();
      const auto final_pipeline = pipeline_iteration + iterations - 1U;
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[final_pipeline % kStages].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

template <class Callback>
__device__ void visit_forward_accumulator(float *accum,
                                          Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMma::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
      callback(warpgroup, row0, column, accum[index * 4U],
               accum[index * 4U + 1U]);
      callback(warpgroup, row1, column, accum[index * 4U + 2U],
               accum[index * 4U + 3U]);
  }
}

template <class Callback>
__device__ void visit_forward_accumulator_pair(float *accum_a, float *accum_b,
                                               Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMma::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum_a[index * 4U],
             accum_a[index * 4U + 1U], accum_b[index * 4U],
             accum_b[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum_a[index * 4U + 2U],
             accum_a[index * 4U + 3U], accum_b[index * 4U + 2U],
             accum_b[index * 4U + 3U]);
  }
}

template <class Callback>
__device__ void visit_forward_accumulator_wide(float *accum,
                                               Callback callback) {
  if (threadIdx.x >= 256)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0; index < ForwardMmaWide::kNumAccum / 4U;
       ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

template <class Callback>
__device__ void visit_backward_accumulator_tall(float *accum,
                                                Callback callback) {
  if (threadIdx.x >= 256U)
    return;
  const auto warpgroup = threadIdx.x / 128U;
  const auto warpgroup_thread = threadIdx.x % 128U;
  const auto warp = warpgroup_thread / 32U;
  const auto lane = warpgroup_thread % 32U;
  const auto row0 = warp * 16U + lane / 4U;
  const auto row1 = row0 + 8U;
#pragma unroll
  for (std::uint32_t index = 0;
       index < BackwardDwMixedMma::kNumAccum / 4U; ++index) {
    const auto column = (lane % 4U) * 2U + index * 8U;
    callback(warpgroup, row0, column, accum[index * 4U],
             accum[index * 4U + 1U]);
    callback(warpgroup, row1, column, accum[index * 4U + 2U],
             accum[index * 4U + 3U]);
  }
}

#endif

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
// Map the immutable M-block task space in the same wavefront order used by
// Notify packet priorities: local M tile first, then expert.  This lets the
// bounded compute scheduler consume the earliest completed tiles without a
// global Dispatch barrier or an O(N-task) atomic scan.
__device__ std::uint32_t forward_wavefront_mblock_expert(
    const ForwardKernelArgs &a, std::uint64_t ordinal,
    std::uint64_t &local_mblock) {
  std::uint64_t max_blocks = 0ULL;
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const std::uint64_t blocks =
        (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
    max_blocks = blocks > max_blocks ? blocks : max_blocks;
  }
  for (std::uint64_t tile = 0; tile < max_blocks; ++tile) {
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto blocks =
          (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
      if (tile >= blocks)
        continue;
      if (ordinal == 0ULL) {
        local_mblock = tile;
        return expert;
      }
      --ordinal;
    }
  }
  local_mblock = 0ULL;
  return a.dispatch.sizes.owned_expert_count;
}

#endif

__device__ bool forward_cancelled(const ForwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
             a.epoch;
}

template <bool StaticLocal>
__device__ void wait_forward_mblock_rows(const ForwardKernelArgs &a,
                                         std::uint32_t expert,
                                         std::uint64_t local_row,
                                         std::uint32_t valid_rows,
                                         transport_state::GenerationCompletionVector
                                             *completion) {
  if constexpr (StaticLocal)
    return;
  if (!a.use_gin || valid_rows == 0U)
    return;
  // One elected lane polls the peer-visible rail counters; the CTA-level
  // synchronization at the call site releases the remaining lanes. Polling
  // from all 384 threads only multiplies L2/system traffic while conveying no
  // additional readiness information.
  if (threadIdx.x != 0U)
    return;
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived = peer_at<unsigned long long>(a.local_base,
                                               a.x_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    auto load_arrived = [&]() {
      unsigned long long total = 0ULL;
      for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(arrived[tile * a.local_world_size + rail]);
        total += ready.load(::cuda::memory_order_acquire);
      }
      return total;
    };
    auto current = load_arrived();
#if MEGAMOE_DEBUG_VALIDATE
    if (current < expected && threadIdx.x == 0U) {
      atomicAdd(&completion->tile_waiter_count, 1ULL);
      (void)atomicCAS(&completion->first_waiting_tile, ~0ULL, tile);
      if (completion->first_waiting_tile == tile) {
        completion->first_wait_expected = expected;
        completion->first_wait_arrived = current;
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
      }
    }
#endif
    while (current < expected && !forward_cancelled(a)) {
#if MEGAMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        completion->first_wait_arrived = current;
      if (threadIdx.x == 0U && completion->first_waiting_tile == tile)
        for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(arrived[tile * a.local_world_size + rail]);
          completion->first_wait_arrived_by_rail[rail] =
              ready.load(::cuda::memory_order_acquire);
        }
#endif
      current = load_arrived();
    }
    tile_row = tile_row_end;
  }
}

__device__ transport_state::TaskCompletionRecord &completion_task(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind) {
  return completion->tasks[static_cast<std::size_t>(kind)];
}

__device__ void mark_task_space_complete(
    transport_state::TaskCompletionRecord &task) {
  task.claimed = task.expected;
  task.completed = task.expected;
  task.inflight = 0ULL;
}

__device__ void claim_completion_tasks(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind, unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.claimed, count);
  atomicAdd(&task.inflight, count);
}

__device__ void complete_completion_tasks(
    transport_state::GenerationCompletionVector *completion,
    transport_state::TaskClass kind, unsigned long long count = 1ULL) {
  auto &task = completion_task(completion, kind);
  atomicAdd(&task.completed, count);
  atomicAdd(&task.inflight, 0ULL - count);
}

__device__ bool completion_vector_ready(
    const transport_state::GenerationCompletionVector &completion) {
  for (std::size_t index = 0; index < transport_state::kTaskClassCount;
       ++index) {
    const auto task = completion.tasks[index];
    if (task.claimed != task.expected || task.completed != task.expected ||
        task.inflight != 0ULL)
      return false;
  }
  return completion.directory_ready == completion.directory_expected &&
         completion.dispatch_local_completion_completed ==
             completion.dispatch_local_completion_expected &&
         completion.return_local_completion_completed ==
             completion.return_local_completion_expected &&
         completion.ring_credit_restored == completion.ring_credit_expected;
}

__device__ void publish_elected_terminal(
    transport_state::GenerationCompletionVector *completion,
    unsigned long long epoch, bool error) {
  if (!error && !completion_vector_ready(*completion))
    return;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
      published(completion->kernel_complete_epoch);
  published.store(epoch, ::cuda::memory_order_release);
  completion->terminal_publish_count += 1ULL;
}

__device__ void forward_publish_progress(const ForwardKernelArgs &a,
                                         std::uint32_t word,
                                         unsigned long long value) {
#if MEGAMOE_DEBUG_VALIDATE
  if (a.progress != nullptr && blockIdx.x == 0U && threadIdx.x == 0U) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> field(
        a.progress[word]);
    field.store(value, ::cuda::memory_order_release);
  }
#else
  (void)a;
  (void)word;
  (void)value;
#endif
}

template <bool StaticLocal>
__device__ __forceinline__ bool remote_backend_enabled(bool enabled) {
  if constexpr (StaticLocal)
    return false;
  return enabled;
}

template <bool StaticLocal>
__global__ __launch_bounds__(384, 1) void
local_p2p_forward_megakernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          a.epoch);
  extern __shared__ __align__(1024) unsigned char storage[];
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = reinterpret_cast<transport::BFloat16 *>(
      storage + 4U * 128U * 64U * sizeof(transport::BFloat16));
#endif
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(
      storage + (4U * 128U * 64U + 4U * 256U * 64U) *
                    sizeof(transport::BFloat16));
  auto *empty_barriers = full_barriers + 4;
#endif
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  forward_publish_progress(a, kForwardEntered, a.epoch);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const auto cluster = cg::this_cluster();
#endif
  auto *canonical = peer_at<transport::BFloat16>(
      a.local_base, a.canonical_rows_offset);
  auto *forward_output = peer_at<transport::BFloat16>(
      a.local_base, a.forward_output_rows_offset);
  auto *headers = peer_at<LocalP2pRouteHeader>(
      a.local_base, a.direct_headers_offset);
  auto *scheduler = peer_at<unsigned long long>(
      a.local_base, a.forward_scheduler_offset);
  transport_state::GenerationCompletionVector *completion = nullptr;
  transport_state::ContextCompletionRecord *context_completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world]);
      context_completion =
          static_cast<transport_state::ContextCompletionRecord *>(
              a.peer_bases[a.world + 1U]);
#if MEGAMOE_DEBUG_VALIDATE
      // Timeout diagnostics use mapped, session-owned storage.  Debug kernels
      // update it in place so the host can identify a stalled expected task
      // without waiting for a D2H copy behind the persistent kernel.
      if (a.progress != nullptr) {
        auto *debug_words = a.progress + kForwardProgressWords;
        completion = reinterpret_cast<
            transport_state::GenerationCompletionVector *>(debug_words);
        context_completion = reinterpret_cast<
            transport_state::ContextCompletionRecord *>(
            debug_words + transport_state::kGenerationCompletionWords);
      }
#endif
    }
  }
  constexpr std::size_t kForwardGemmSmemBytes =
      (4U * 128U * 64U + 4U * 256U * 64U) *
          sizeof(transport::BFloat16) +
      8U * sizeof(std::uint64_t);
  [[maybe_unused]] auto *return_rows_by_source = reinterpret_cast<unsigned int *>(
      storage + kForwardGemmSmemBytes);
  auto *claims = peer_at<unsigned long long>(
      a.local_base, a.forward_task_claim_offset);
  auto *fc1_done = peer_at<unsigned long long>(
      a.local_base, a.forward_fc1_done_offset);
  auto *forward_output_ready_epoch = peer_at<unsigned long long>(
      a.local_base, a.forward_output_ready_epoch_offset);
  auto *forward_return_chunks = peer_at<ForwardReturnChunkRecord>(
      a.local_base, a.forward_return_chunk_directory_offset);
  auto *forward_return_chunk_state = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_state_offset);
  auto *forward_return_chunk_ready_count = peer_at<unsigned long long>(
      a.local_base, a.forward_return_chunk_ready_count_offset);
  auto *forward_return_fragment_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_fragment_index_offset);
  auto *forward_return_source_compact_index = peer_at<unsigned long long>(
      a.local_base, a.forward_return_source_compact_index_offset);
  auto *s = peer_at<transport::BFloat16>(a.local_base, a.forward_s_offset);
  auto *ab_cache = a.forward_ab_cache_offset == 0
                       ? static_cast<transport::BFloat16 *>(nullptr)
                       : peer_at<transport::BFloat16>(
                             a.local_base, a.forward_ab_cache_offset);
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->start_clock = device_globaltimer_ns();
    // Transport-only counters are unused by Forward and serve as first-start
    // clocks without growing the pooled status allocation.
    status->dispatch_blocks_done = ~0ULL; // first FC2 tile
    status->relay_blocks_done = ~0ULL;    // first returned M-block
    status->reduced_tokens = ~0ULL;       // first reduced token
    scheduler[0] = scheduler[1] = scheduler[2] = scheduler[3] = 0ULL;
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert)
      scheduler[3] +=
          (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
    scheduler[4U + a.world] = 0ULL;
    scheduler[5U + 2ULL * a.world] = 0ULL; // GIN headers ready contexts
    scheduler[6U + 2ULL * a.world] = 0ULL; // Return directory ready epoch
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        completion->generation = a.dispatch.identity.generation;
        completion->epoch = a.epoch;
#if MEGAMOE_DEBUG_VALIDATE
        completion->first_waiting_tile = ~0ULL;
#endif
        unsigned long long dispatch_waves = 0ULL;
        for (std::uint64_t wave = 0; wave < a.dispatch_waves.wave_count;
             ++wave)
          if (a.dispatch_waves.destination_node[wave] != a.node)
            ++dispatch_waves;
        const auto fc1_ntiles = (a.intermediate + 127ULL) / 128ULL;
        const auto fc2_ntiles = (a.hidden + 255ULL) / 256ULL;
        completion_task(completion,
                        transport_state::TaskClass::DispatchWave)
            .expected = dispatch_waves;
        unsigned long long relay_packets = 0ULL;
        for (std::uint32_t index = 0;
             index < a.dispatch.sizes.incoming_signal_count; ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node != a.node)
            relay_packets += signal.expected_packets;
        }
        completion_task(completion, transport_state::TaskClass::RelayWave)
            .expected = relay_packets;
        completion_task(completion, transport_state::TaskClass::Fc1OrDv)
            .expected = scheduler[3] * fc1_ntiles;
        completion_task(completion, transport_state::TaskClass::Fc2OrDx)
            .expected = scheduler[3] * fc2_ntiles;
        completion_task(completion,
                        transport_state::TaskClass::SourceArrival)
            .expected = a.dispatch.sizes.route_count;
        completion_task(completion, transport_state::TaskClass::Reduce)
            .expected = a.dispatch.identity.num_tokens;
        completion_task(completion, transport_state::TaskClass::Return)
            .expected = 0ULL;
        completion->return_local_completion_expected = 0ULL;
        completion->directory_expected =
            static_cast<unsigned long long>(
                a.dispatch.identity.num_nodes - 1U) *
            a.gin_context_count;
        completion->dispatch_local_completion_expected =
            completion->directory_expected + dispatch_waves;
      }
    }
  }
  if (threadIdx.x == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const auto cluster_size = cluster.dim_blocks().x * cluster.dim_blocks().y *
                              cluster.dim_blocks().z;
    atomicMax(&status->runtime_cluster_size, cluster_size);
    const auto cluster_rank = cluster.block_rank();
    if (cluster_rank < 64U)
      atomicOr(&status->cluster_rank_mask, 1ULL << cluster_rank);
#else
    atomicMax(&status->runtime_cluster_size, 1U);
    atomicOr(&status->cluster_rank_mask, 1ULL);
#endif
  }
  // One completion word per destination->source rank replaces a system-scope
  // signal for every route.  The high word is immutable expected rows; return
  // CTAs add to the low word after their peer stores have been fenced.
  if (blockIdx.x == 0 && threadIdx.x < a.world) {
    unsigned long long expected = 0;
    for (std::uint32_t local_expert = 0;
         local_expert < a.dispatch.sizes.owned_expert_count; ++local_expert) {
      const auto expert = a.dispatch.owned_expert_ids[local_expert];
      expected += a.dispatch.source_expert_count[
          static_cast<std::uint64_t>(threadIdx.x) *
              a.dispatch.identity.num_experts +
          expert];
    }
    scheduler[4U + threadIdx.x] = expected << 32U;
  }
  for (std::uint64_t mtile =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       mtile < a.dispatch.sizes.mtile_count;
       mtile += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    fc1_done[mtile] = a.epoch << 32U;
  // The same allocation is a generation-local row->(proxy,chunk) map for
  // remote Return. Clear only the actual canonical rows; peer proxies fill
  // remote-route entries after consuming the immutable Dispatch directory.
  for (std::uint64_t row =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < a.dispatch.sizes.incoming_row_count;
       row += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    forward_output_ready_epoch[row] = 0ULL;
  grid.sync();
  forward_publish_progress(a, kForwardSetupComplete, scheduler[3]);
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  const auto setup_fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  const auto setup_fc2_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto setup_mblocks = scheduler[3];
  // The first words are an immutable expert/local-M lookup in Notify's M-tile
  // wavefront order. Task-state words follow it and every individual task has
  // the only legal transitions
  // WAITING(0) -> CLAIMED(1) -> COMPLETED(2).
  for (std::uint64_t mblock =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       mblock < setup_mblocks;
       mblock += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    std::uint64_t local_mblock = 0;
    const auto expert =
        forward_wavefront_mblock_expert(a, mblock, local_mblock);
    claims[mblock] = (static_cast<unsigned long long>(expert) << 32U) |
                     static_cast<std::uint32_t>(local_mblock);
  }
  const auto setup_task_count =
      setup_mblocks * (setup_fc1_ntiles + setup_fc2_ntiles);
  for (std::uint64_t task =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       task < setup_task_count;
       task += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
    claims[setup_mblocks + task] = 0ULL;
  grid.sync();
#endif

  // Socket correctness mode has already placed every incoming activation and
  // canonical header through host-enqueued NCCL Send/Recv. It deliberately
  // bypasses the peer/G​IN dispatch phase while retaining the exact FC core.
  if (!remote_backend_enabled<StaticLocal>(a.use_socket)) {
    // Local rows use the frozen CUDA-IPC fast path. Cross-node rows are packed
    // once per (token,node) by the same-rail proxy path below.
    unsigned long long dispatch_peer_rows = 0ULL;
#if MEGAMOE_DEVICE_TIMELINE
    bool timeline_x_local_recorded = false;
#endif
    for (std::uint64_t route = blockIdx.x;
         route < a.dispatch.sizes.route_count; route += gridDim.x) {
    const auto destination = a.dispatch.route_dst_rank[route];
    if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
        destination / a.local_world_size != a.node)
      continue;
    const auto row = a.dispatch.route_dst_row[route];
    const auto token = route / a.dispatch.identity.top_k;
    auto *destination_base = a.peer_bases[destination];
    auto *destination_rows = peer_at<transport::BFloat16>(
        destination_base, a.canonical_rows_offset);
    const auto *source = a.input + token * a.hidden;
    auto *target = destination_rows + row * a.hidden;
    const auto *source128 = reinterpret_cast<const uint4 *>(source);
    auto *target128 = reinterpret_cast<uint4 *>(target);
    for (std::uint64_t vector = threadIdx.x; vector < a.hidden / 8ULL;
         vector += blockDim.x)
      target128[vector] = source128[vector];
    if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
      // Every lane's row stores precede the exact destination M-tile row
      // increment.  The destination compute worker acquires this counter; no
      // packet/wave signal is itself interpreted as tile readiness.
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        auto *arrived = peer_at<unsigned long long>(
            destination_base, a.x_arrived_rows_offset);
        const auto rail = a.rank % a.local_world_size;
        ::cuda::atomic_ref<unsigned long long,
                           ::cuda::thread_scope_system>
            counter(arrived[a.dispatch_waves.route_dst_mtile[route] *
                                a.local_world_size +
                            rail]);
        counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
        atomicAdd(&completion->local_rows_published, 1ULL);
        const auto tile = a.dispatch_waves.route_dst_mtile[route];
        if (tile < 64U)
          atomicOr(&completion->published_tile_mask, 1ULL << tile);
#endif
      }
      __syncthreads();
    }
#if MEGAMOE_DEVICE_TIMELINE
    __syncthreads();
    if (!timeline_x_local_recorded) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstLocalRowVisible, GinDispatch,
                              0U, route, a.epoch);
      timeline_x_local_recorded = true;
    }
#endif
    if (threadIdx.x == 0) {
      auto *destination_headers = peer_at<LocalP2pRouteHeader>(
          destination_base, a.direct_headers_offset);
      destination_headers[row] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank, destination, 0U, a.dispatch.identity.generation, a.epoch,
          route, route, token, row, a.dispatch.expert_id[route],
          static_cast<std::uint32_t>(route % a.dispatch.identity.top_k)};
      if (destination != a.rank)
        ++dispatch_peer_rows;
    }
    }
#if __has_include(<nccl_device.h>)
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    grid.sync();
    auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        window_bytes + a.gin_send_header_offset);
    auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        window_bytes + a.gin_send_fanout_offset);
    auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
        window_bytes + a.gin_send_payload_offset);
    const auto vector_count = a.hidden / 8ULL;
    const auto global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto global_threads =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t item = global_thread;
         item < a.dispatch.sizes.packet_count * vector_count;
         item += global_threads) {
      const auto wave_position = item / vector_count;
      const auto vector = item % vector_count;
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      reinterpret_cast<uint4 *>(send_payload)[wave_position * vector_count +
                                               vector] =
          reinterpret_cast<const uint4 *>(a.input)[
              descriptor.local_token_id * vector_count + vector];
    }
    for (std::uint64_t packed = global_thread;
         packed < a.dispatch.sizes.packet_count * a.dispatch.identity.top_k;
         packed += global_threads) {
      const auto wave_position = packed / a.dispatch.identity.top_k;
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto fanout_slot = packed % a.dispatch.identity.top_k;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (fanout_begin + fanout_slot < fanout_end)
        send_fanout[packed] =
            a.dispatch.packet_fanout[fanout_begin + fanout_slot];
    }
    for (std::uint64_t wave_position = global_thread;
         wave_position < a.dispatch.sizes.packet_count;
         wave_position += global_threads) {
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      send_headers[wave_position] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion,
          a.rank,
          descriptor.proxy_rank,
          descriptor.channel,
          a.dispatch.identity.generation,
          a.epoch,
          wave_position,
          fanout_end - fanout_begin,
          descriptor.local_token_id,
          wave_position * a.dispatch.identity.top_k,
          0,
          0};
    }
    // GIN's device DMA reads this window after the grid-wide producers have
    // populated it. A cooperative grid barrier orders CUDA threads but does
    // not by itself publish their stores to a system-scope device backend.
    MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchPacketReady, GinDispatch,
                            blockIdx.x < a.gin_context_count ? blockIdx.x : 0U,
                            a.dispatch.sizes.packet_count, a.epoch);
    __threadfence_system();
    grid.sync();
  }
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count) {
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    const ncclGin gin(a.gin_dev_comm, context);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
        window_bytes + a.gin_send_dispatch_directory_offset);
    auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        window_bytes + a.gin_send_header_offset);
#if MEGAMOE_DEBUG_VALIDATE
    auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        window_bytes + a.gin_send_fanout_offset);
#endif
    // The owner-side initialization kernel only rendezvous with CUDA-IPC
    // peers on this node.  A remote same-rail producer can otherwise enter
    // this kernel first, publish an exact M-tile row increment, and have that
    // increment erased when the slower destination initializes its local
    // x_arrived_rows storage.  Signal 2 is reserved for a fixed-width,
    // per-context generation-ready handshake.  It runs once before any
    // Dispatch payload put and therefore adds no per-wave or per-task work.
    //
    // Reuse each context's immutable directory slot as the eight-byte DMA
    // source/destination: the real directory record is written only after the
    // handshake completes, and Dispatch signal 0 orders that later record.
    if (threadIdx.x == 0U) {
      unsigned long long setup_peers = 0ULL;
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == a.node)
          continue;
        const auto remote_partition =
            remote_node < a.node ? remote_node : remote_node - 1U;
        const auto source_partition =
            a.node < remote_node ? a.node : a.node - 1U;
        const auto directory_index =
            remote_partition * a.gin_context_count + context;
        const auto remote_directory_index =
            source_partition * a.gin_context_count + context;
        send_directory[directory_index] = GinDispatchContextRecord{
            a.dispatch.identity.generation, a.epoch, 0ULL, 0U, 0U};
        __threadfence_system();
        gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                a.gin_receive_dispatch_directory_offset +
                    remote_directory_index *
                        sizeof(GinDispatchContextRecord),
                a.gin_window, a.gin_send_dispatch_directory_offset +
                                  directory_index *
                                      sizeof(GinDispatchContextRecord),
                sizeof(GinDispatchContextRecord), ncclGin_SignalInc{2},
                ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++setup_peers;
      }
      if (setup_peers != 0ULL) {
        gin.waitCounter(ncclCoopThread{}, 0, setup_peers);
        gin.resetCounter(0);
        gin.waitSignal(ncclCoopThread{}, 2, setup_peers);
      }
    }
    __syncthreads();
    unsigned long long issued_completions = 0ULL;
    unsigned long long issued_packets = 0ULL;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == a.node)
        continue;
      std::uint64_t node_wave_begin = 0;
      while (node_wave_begin < a.dispatch_waves.wave_count &&
             a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
        ++node_wave_begin;
      auto node_wave_end = node_wave_begin;
      while (node_wave_end < a.dispatch_waves.wave_count &&
             a.dispatch_waves.destination_node[node_wave_end] == remote_node)
        ++node_wave_end;
      const auto node_waves = node_wave_end - node_wave_begin;
      const auto context_wave_begin =
          node_wave_begin + node_waves * context / a.gin_context_count;
      const auto context_wave_end =
          node_wave_begin + node_waves * (context + 1U) /
                                a.gin_context_count;
      const auto node_packet_begin =
          a.dispatch_waves.offsets[node_wave_begin];
      const auto packet_begin =
          a.dispatch_waves.offsets[context_wave_begin] - node_packet_begin;
      const auto packet_end =
          a.dispatch_waves.offsets[context_wave_end] - node_packet_begin;
      const auto remote_partition =
          remote_node < a.node ? remote_node : remote_node - 1U;
      const auto source_partition =
          a.node < remote_node ? a.node : a.node - 1U;
      const auto directory_index =
          remote_partition * a.gin_context_count + context;
      if (threadIdx.x == 0U) {
        send_directory[directory_index] = GinDispatchContextRecord{
            a.dispatch.identity.generation, a.epoch, packet_begin,
            static_cast<std::uint32_t>(packet_end - packet_begin),
            static_cast<std::uint32_t>(context_wave_end -
                                       context_wave_begin)};
        __threadfence_system();
        const auto remote_directory_index =
            source_partition * a.gin_context_count + context;
        gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                a.gin_receive_dispatch_directory_offset +
                    remote_directory_index *
                        sizeof(GinDispatchContextRecord),
                a.gin_window, a.gin_send_dispatch_directory_offset +
                                  directory_index *
                                      sizeof(GinDispatchContextRecord),
                sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
                ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++issued_completions;
      }
      __syncthreads();
      // Every context publishes exactly one fixed-width directory record for
      // every remote node that participates in this generation, including an
      // empty record.  Receivers therefore never have to infer which sender
      // contexts own waves before waiting for the directory phase.
      for (auto wave = context_wave_begin; wave < context_wave_end; ++wave) {
        const auto wave_begin = a.dispatch_waves.offsets[wave];
        const auto wave_end = a.dispatch_waves.offsets[wave + 1U];
        const auto wire_begin = wave_begin - node_packet_begin;
        const auto wave_packets = wave_end - wave_begin;
        for (auto position = wave_begin + threadIdx.x; position < wave_end;
             position += blockDim.x) {
          send_headers[position].sequence = position - node_packet_begin;
          send_headers[position].dst_row =
              (position - node_packet_begin) * a.dispatch.identity.top_k;
          send_headers[position].topk_slot = 0U;
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
          send_headers[wave_begin].topk_slot =
              static_cast<std::uint32_t>(wave_packets);
          __threadfence_system();
          const auto receive_packet =
              source_partition * a.max_tokens_per_rank + wire_begin;
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, GinPutIssue, GinDispatch, context,
              wave_packets * a.hidden * sizeof(transport::BFloat16), a.epoch);
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_header_offset +
                      receive_packet * sizeof(LocalP2pRouteHeader),
                  a.gin_window,
                  a.gin_send_header_offset +
                      wave_begin * sizeof(LocalP2pRouteHeader),
                  wave_packets * sizeof(LocalP2pRouteHeader));
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_fanout_offset +
                      (source_partition * a.max_tokens_per_rank + wire_begin) *
                          a.dispatch.identity.top_k *
                          sizeof(DeviceFanoutRecord),
                  a.gin_window,
                  a.gin_send_fanout_offset +
                      wave_begin * a.dispatch.identity.top_k *
                          sizeof(DeviceFanoutRecord),
                  wave_packets * a.dispatch.identity.top_k *
                      sizeof(DeviceFanoutRecord));
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_payload_offset +
                      receive_packet * a.hidden *
                          sizeof(transport::BFloat16),
                  a.gin_window,
                  a.gin_send_payload_offset +
                      wave_begin * a.hidden * sizeof(transport::BFloat16),
                  wave_packets * a.hidden * sizeof(transport::BFloat16),
                  ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                  ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          ++issued_completions;
          issued_packets += wave_packets;
          atomicAdd(&status->gin_dispatch_put_count, 3ULL);
          atomicAdd(&status->gin_signal_count, 1ULL);
          atomicAdd(&status->gin_metadata_bytes,
                    wave_packets *
                        (sizeof(LocalP2pRouteHeader) +
                         a.dispatch.identity.top_k *
                             sizeof(DeviceFanoutRecord)));
        }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0U && issued_completions != 0ULL) {
      gin.waitCounter(ncclCoopThread{}, 0, issued_completions);
      MEGAMOE_TIMELINE_RECORD(a.timeline, DispatchLocalCompletion,
                              GinDispatch, context, issued_packets, a.epoch);
      atomicAdd(&status->deduplicated_cross_node_payload_bytes,
                issued_packets * a.hidden * sizeof(transport::BFloat16));
      atomicAdd(&status->actual_dispatch_peer_bytes,
                issued_packets * a.hidden * sizeof(transport::BFloat16));
      gin.resetCounter(0);
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == a.node)
          continue;
        const auto remote_partition =
            remote_node < a.node ? remote_node : remote_node - 1U;
        const auto record_index =
            remote_partition * a.gin_context_count + context;
        const auto directory = send_directory[record_index];
        auto &record = context_completion[record_index];
        record.directory_ready = 1ULL;
        record.dispatch_expected = directory.wave_count;
        record.dispatch_claimed = directory.wave_count;
        record.dispatch_completed = directory.wave_count;
        record.local_completion = 1ULL + directory.wave_count;
        record.credit = 1ULL;
        record.last_cursor = directory.packet_begin + directory.packet_count;
        record.last_progress_globaltimer = device_globaltimer_ns();
#if MEGAMOE_DEBUG_VALIDATE
        std::uint64_t first_node_wave = 0;
        while (first_node_wave < a.dispatch_waves.wave_count &&
               a.dispatch_waves.destination_node[first_node_wave] <
                   remote_node)
          ++first_node_wave;
        const auto node_packet_begin =
            a.dispatch_waves.offsets[first_node_wave];
        for (std::uint64_t position = directory.packet_begin;
             position < directory.packet_begin + directory.packet_count;
             ++position) {
          const auto header = send_headers[node_packet_begin + position];
          record.dispatch_fanout_rows += header.route_index;
          for (std::uint64_t fanout = 0; fanout < header.route_index;
               ++fanout)
            record.dispatch_tile0_rows +=
                send_fanout[(node_packet_begin + position) *
                                a.dispatch.identity.top_k +
                            fanout]
                    .reserved == 0U;
        }
#endif
        atomicAdd(&completion->directory_ready, 1ULL);
        atomicAdd(&completion->dispatch_local_completion_completed,
                  1ULL + directory.wave_count);
        if (directory.wave_count != 0ULL) {
          claim_completion_tasks(completion,
                                 transport_state::TaskClass::DispatchWave,
                                 directory.wave_count);
          complete_completion_tasks(completion,
                                    transport_state::TaskClass::DispatchWave,
                                    directory.wave_count);
        }
      }
    }
    __syncthreads();
    unsigned long long incoming_sources = 0ULL;
    if (threadIdx.x == 0U) {
      for (std::uint32_t index = 0;
           index < a.dispatch.sizes.incoming_signal_count; ++index) {
        const auto signal = a.dispatch.incoming_proxy_signals[index];
        if (signal.source_node != a.node && signal.expected_packets != 0ULL)
          ++incoming_sources;
      }
      if (incoming_sources != 0ULL)
        gin.waitSignal(ncclCoopThread{}, 0, incoming_sources);
      auto *receive_directory =
          reinterpret_cast<GinDispatchContextRecord *>(
              static_cast<std::byte *>(a.gin_window_base) +
              a.gin_receive_dispatch_directory_offset);
      unsigned long long incoming_waves = 0ULL;
      for (std::uint32_t index = 0;
           index < a.dispatch.sizes.incoming_signal_count; ++index) {
        const auto signal = a.dispatch.incoming_proxy_signals[index];
        if (signal.source_node == a.node || signal.expected_packets == 0ULL)
          continue;
        const auto source_partition =
            signal.source_node < a.node ? signal.source_node
                                        : signal.source_node - 1U;
        const auto directory = receive_directory[
            source_partition * a.gin_context_count + context];
        if (directory.generation == a.dispatch.identity.generation &&
            directory.epoch == a.epoch) {
          incoming_waves += directory.wave_count;
          const auto record_index =
              source_partition * a.gin_context_count + context;
          context_completion[record_index].relay_expected =
              directory.packet_count;
        }
      }
      if (incoming_waves != 0ULL)
        gin.waitSignal(ncclCoopThread{}, 0,
                       incoming_sources + incoming_waves);
      MEGAMOE_TIMELINE_RECORD(a.timeline, RemoteSignalObserved, GinDispatch,
                              context, incoming_sources, a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstRemoteSignal, GinDispatch,
                              context, incoming_sources, a.epoch);
      atomicAdd(&scheduler[5U + 2ULL * a.world], 1ULL);
    }
    __syncthreads();

    // All GIN contexts publish an immutable receive directory before Relay.
    // Packet waves are divided between contexts by wave count, which can be
    // very uneven in bytes/fan-out.  Relaying only the directory owned by the
    // matching context left one CTA copying nearly the entire remote payload
    // while the other progress CTAs were idle.  Wait for the fixed directory
    // set, then stripe every directory across the fixed context CTA team.  No
    // task space is extended here and the exact destination-row counters stay
    // the sole FC readiness condition.
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          directories_ready(scheduler[5U + 2ULL * a.world]);
      while (directories_ready.load(::cuda::memory_order_acquire) <
                 a.gin_context_count &&
             !forward_cancelled(a)) {
        __nanosleep(256U);
      }
    }
    __syncthreads();
  }
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count) {
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        window_bytes + a.gin_receive_header_offset);
    auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        window_bytes + a.gin_receive_fanout_offset);
    auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
        window_bytes + a.gin_receive_payload_offset);
    auto *receive_directory = reinterpret_cast<GinDispatchContextRecord *>(
        window_bytes + a.gin_receive_dispatch_directory_offset);
    MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayStart, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_signal_count, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstProxyRelay, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_signal_count, a.epoch);
    const auto vector_count = a.hidden / 8ULL;
    const auto context_thread = static_cast<std::uint64_t>(threadIdx.x);
    const auto context_threads = static_cast<std::uint64_t>(blockDim.x);
    const auto relay_thread =
        static_cast<std::uint64_t>(context) * context_threads +
        context_thread;
    const auto relay_threads =
        static_cast<std::uint64_t>(a.gin_context_count) * context_threads;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
      for (std::uint32_t directory_context = 0;
           directory_context < a.gin_context_count; ++directory_context) {
        const auto directory = receive_directory[
            source_partition * a.gin_context_count + directory_context];
        for (std::uint64_t item = relay_thread;
             item < static_cast<std::uint64_t>(directory.packet_count) *
                        packet_vectors;
             item += relay_threads) {
          const auto sequence =
              directory.packet_begin + item / packet_vectors;
          const auto within_packet = item % packet_vectors;
          const auto fanout = within_packet / vector_count;
          const auto vector = within_packet % vector_count;
          const auto packet =
              source_partition * a.max_tokens_per_rank + sequence;
          const auto header = receive_headers[packet];
          if (fanout >= header.route_index)
            continue;
          const auto target = receive_fanout[
              source_partition * a.max_tokens_per_rank *
                      a.dispatch.identity.top_k +
                  header.dst_row + fanout];
          const auto destination =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *destination_rows = peer_at<transport::BFloat16>(
              a.peer_bases[destination], a.canonical_rows_offset);
          reinterpret_cast<uint4 *>(destination_rows)[
              target.dst_row * vector_count + vector] =
              reinterpret_cast<const uint4 *>(receive_payload)[
                  packet * vector_count + vector];
        }
        for (std::uint64_t slot = relay_thread;
             slot < static_cast<std::uint64_t>(directory.packet_count) *
                        a.dispatch.identity.top_k;
             slot += relay_threads) {
          const auto sequence = directory.packet_begin +
                                slot / a.dispatch.identity.top_k;
          const auto fanout = slot % a.dispatch.identity.top_k;
          const auto packet =
              source_partition * a.max_tokens_per_rank + sequence;
          const auto header = receive_headers[packet];
          if (fanout >= header.route_index)
            continue;
          const auto target = receive_fanout[
              source_partition * a.max_tokens_per_rank *
                      a.dispatch.identity.top_k +
                  header.dst_row + fanout];
          const auto destination =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *destination_headers = peer_at<LocalP2pRouteHeader>(
              a.peer_bases[destination], a.direct_headers_offset);
          const auto route = header.local_token_id *
                                 a.dispatch.identity.top_k +
                             target.topk_slot;
          destination_headers[target.dst_row] = LocalP2pRouteHeader{
              transport::kActivationTransportSchemaVersion,
              header.source_rank,
              destination,
              header.channel,
              header.generation,
              header.source_epoch,
              header.sequence,
              route,
              header.local_token_id,
              target.dst_row,
              target.expert_id,
              target.topk_slot};
        }
      }
    }
    __threadfence_system();
    __syncthreads();
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      for (std::uint32_t directory_context = 0;
           directory_context < a.gin_context_count; ++directory_context) {
        const auto directory = receive_directory[
            source_partition * a.gin_context_count + directory_context];
        for (std::uint64_t slot = relay_thread;
             slot < static_cast<std::uint64_t>(directory.packet_count) *
                        a.dispatch.identity.top_k;
             slot += relay_threads) {
          const auto sequence = directory.packet_begin +
                                slot / a.dispatch.identity.top_k;
          const auto fanout = slot % a.dispatch.identity.top_k;
          const auto packet =
              source_partition * a.max_tokens_per_rank + sequence;
          const auto header = receive_headers[packet];
          if (fanout >= header.route_index)
            continue;
          const auto target = receive_fanout[
              source_partition * a.max_tokens_per_rank *
                      a.dispatch.identity.top_k +
                  header.dst_row + fanout];
          const auto destination =
              a.node * a.local_world_size + target.destination_local_rank;
          auto *arrived = peer_at<unsigned long long>(
              a.peer_bases[destination], a.x_arrived_rows_offset);
          const auto rail = a.rank % a.local_world_size;
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              counter(arrived[target.reserved * a.local_world_size + rail]);
          counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
          const auto record_index =
              source_partition * a.gin_context_count + directory_context;
          atomicAdd(&context_completion[record_index].relay_fanout_rows,
                    1ULL);
          if (target.reserved == 0U)
            atomicAdd(&context_completion[record_index].relay_tile0_rows,
                      1ULL);
          atomicAdd(&completion->remote_rows_published, 1ULL);
          if (target.reserved < 64U)
            atomicOr(&completion->published_tile_mask,
                     1ULL << target.reserved);
#endif
        }
      }
    }
    #if 0
    const auto relay_index = static_cast<std::uint64_t>(blockIdx.x);
    const auto forward_relay_ctas = a.gin_context_count;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      for (std::uint32_t directory_context = 0;
           directory_context < a.gin_context_count; ++directory_context) {
        const auto directory = receive_directory[
            source_partition * a.gin_context_count + directory_context];
        constexpr std::uint64_t kRelayRowsPerRelease = 8ULL;
        const auto directory_slots =
            static_cast<std::uint64_t>(directory.packet_count) *
            a.dispatch.identity.top_k;
        for (std::uint64_t batch_begin =
                 relay_index * kRelayRowsPerRelease;
             batch_begin < directory_slots;
             batch_begin += forward_relay_ctas * kRelayRowsPerRelease) {
          for (std::uint64_t lane = 0; lane < kRelayRowsPerRelease; ++lane) {
            const auto slot = batch_begin + lane;
            if (slot >= directory_slots)
              break;
            const auto sequence = directory.packet_begin +
                                  slot / a.dispatch.identity.top_k;
            const auto fanout = slot % a.dispatch.identity.top_k;
            const auto packet =
                source_partition * a.max_tokens_per_rank + sequence;
            const auto header = receive_headers[packet];
            if (fanout >= header.route_index)
              continue;
            const auto target = receive_fanout[
                source_partition * a.max_tokens_per_rank *
                        a.dispatch.identity.top_k +
                    header.dst_row + fanout];
            const auto destination =
                a.node * a.local_world_size + target.destination_local_rank;
            auto *destination_rows = peer_at<transport::BFloat16>(
                a.peer_bases[destination], a.canonical_rows_offset);
            for (std::uint64_t vector = threadIdx.x; vector < vector_count;
                 vector += blockDim.x)
              reinterpret_cast<uint4 *>(destination_rows)[
                  target.dst_row * vector_count + vector] =
                  reinterpret_cast<const uint4 *>(receive_payload)[
                      packet * vector_count + vector];
            if (threadIdx.x == 0U) {
              auto *destination_headers = peer_at<LocalP2pRouteHeader>(
                  a.peer_bases[destination], a.direct_headers_offset);
              const auto route = header.local_token_id *
                                     a.dispatch.identity.top_k +
                                 target.topk_slot;
              destination_headers[target.dst_row] = LocalP2pRouteHeader{
                  transport::kActivationTransportSchemaVersion,
                  header.source_rank,
                  destination,
                  header.channel,
                  header.generation,
                  header.source_epoch,
                  header.sequence,
                  route,
                  header.local_token_id,
                  target.dst_row,
                  target.expert_id,
                  target.topk_slot};
            }
          }
          __syncthreads();
          if (threadIdx.x == 0U)
            __threadfence_system();
          __syncthreads();
          for (std::uint64_t lane = threadIdx.x;
               lane < kRelayRowsPerRelease; lane += blockDim.x) {
            const auto slot = batch_begin + lane;
            if (slot >= directory_slots)
              continue;
            const auto sequence = directory.packet_begin +
                                  slot / a.dispatch.identity.top_k;
            const auto fanout = slot % a.dispatch.identity.top_k;
            const auto packet =
                source_partition * a.max_tokens_per_rank + sequence;
            const auto header = receive_headers[packet];
            if (fanout >= header.route_index)
              continue;
            const auto target = receive_fanout[
                source_partition * a.max_tokens_per_rank *
                        a.dispatch.identity.top_k +
                    header.dst_row + fanout];
            const auto destination =
                a.node * a.local_world_size + target.destination_local_rank;
            auto *arrived = peer_at<unsigned long long>(
                a.peer_bases[destination], a.x_arrived_rows_offset);
            const auto rail = a.rank % a.local_world_size;
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_system>
                counter(arrived[target.reserved * a.local_world_size + rail]);
            counter.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
            const auto record_index =
                source_partition * a.gin_context_count + directory_context;
            atomicAdd(&context_completion[record_index].relay_fanout_rows,
                      1ULL);
            if (target.reserved == 0U)
              atomicAdd(&context_completion[record_index].relay_tile0_rows,
                        1ULL);
            atomicAdd(&completion->remote_rows_published, 1ULL);
            if (target.reserved < 64U)
              atomicOr(&completion->published_tile_mask,
                       1ULL << target.reserved);
#endif
          }
          __syncthreads();
        }
      }
    }
    #endif
    __syncthreads();
    if (blockIdx.x < a.gin_context_count && threadIdx.x == 0U) {
      unsigned long long relayed = 0ULL;
      for (std::uint32_t remote = 0;
           remote < a.dispatch.identity.num_nodes - 1U; ++remote) {
        const auto index = remote * a.gin_context_count + context;
        auto &record = context_completion[index];
        record.relay_claimed = record.relay_expected;
        record.relay_completed = record.relay_expected;
        record.last_progress_globaltimer = device_globaltimer_ns();
        relayed += record.relay_expected;
      }
      if (relayed != 0ULL) {
        claim_completion_tasks(completion,
                               transport_state::TaskClass::RelayWave, relayed);
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::RelayWave,
                                  relayed);
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, ProxyRelayEnd, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, XLastRelay, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
  }
#endif
    if (threadIdx.x == 0 && dispatch_peer_rows != 0ULL)
      atomicAdd(&status->actual_dispatch_peer_bytes,
                dispatch_peer_rows * a.hidden * sizeof(transport::BFloat16));
  // Every participating lane orders its peer payload stores before the team
  // rendezvous. A fence issued only by the signal lane would not cover stores
  // performed by the other lanes in the CTA.
  __threadfence_system();
  forward_publish_progress(a, kForwardDispatchStored,
                           a.dispatch.sizes.route_count);
  if (!remote_backend_enabled<StaticLocal>(a.use_gin)) {
    grid.sync();
    if (blockIdx.x == 0) {
    const auto first_local = remote_backend_enabled<StaticLocal>(a.use_gin)
                                 ? a.node * a.local_world_size
                                 : 0U;
    const auto local_count = remote_backend_enabled<StaticLocal>(a.use_gin)
                                 ? a.local_world_size
                                 : a.world;
    if (threadIdx.x < local_count) {
      const auto peer = first_local + threadIdx.x;
      auto *signal = peer_at<unsigned long long>(
          a.peer_bases[peer], a.ingress_published_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> published(
          signal[a.rank * a.channels]);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    for (std::uint32_t local_source = threadIdx.x;
         local_source < local_count;
         local_source += blockDim.x) {
      const auto source = first_local + local_source;
      auto *signals = peer_at<unsigned long long>(
          a.local_base, a.ingress_published_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> published(
          signals[source * a.channels]);
      while (published.load(::cuda::memory_order_acquire) < a.epoch &&
             !forward_cancelled(a)) {
      }
    }
    }
    grid.sync();
  }
  forward_publish_progress(a, kForwardDispatchObserved,
                           remote_backend_enabled<StaticLocal>(a.use_gin)
                               ? a.local_world_size
                               : a.world);
  } else {
    grid.sync();
    forward_publish_progress(a, kForwardDispatchStored,
                             a.dispatch.sizes.route_count);
    forward_publish_progress(a, kForwardDispatchObserved,
                             a.dispatch.sizes.incoming_row_count);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->dispatch_complete_clock = device_globaltimer_ns();
    status->fc1_start_clock = status->dispatch_complete_clock;
  }
  MEGAMOE_TIMELINE_RECORD(a.timeline, XTileReady, Compute, 0U,
                          a.dispatch.sizes.mtile_count, a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, XDispatchComplete, Compute, 0U,
                          a.dispatch.sizes.mtile_count, a.epoch);

#if __has_include(<nccl_device.h>)
  // Build a fixed ReturnChunk catalog once the immutable remote Dispatch
  // directory is visible.  Chunk identity depends only on the packed reverse
  // row range; FC2 completion order can only change when a chunk becomes
  // ready, never where it is sent or stored.
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x == a.gin_context_count) {
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnDirectoryStart, GinReturn, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    const auto chunk_count_word = 4ULL + 2ULL * a.world;
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          headers_ready(scheduler[chunk_count_word + 1ULL]);
      while (headers_ready.load(::cuda::memory_order_acquire) <
                 a.gin_context_count &&
             !forward_cancelled(a)) {
        __nanosleep(256U);
      }
      auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
      auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
          window_bytes + a.gin_receive_header_offset);
      std::uint64_t chunk_count = 0ULL;
      auto *receive_directory = reinterpret_cast<GinDispatchContextRecord *>(
          window_bytes + a.gin_receive_dispatch_directory_offset);
      for (std::uint32_t source_partition = 0;
           source_partition + 1U < a.dispatch.identity.num_nodes;
           ++source_partition) {
        const auto source_node =
            source_partition < a.node ? source_partition
                                      : source_partition + 1U;
        std::uint64_t received_packets = 0ULL;
        for (std::uint32_t context = 0; context < a.gin_context_count;
             ++context) {
          const auto directory = receive_directory[
              source_partition * a.gin_context_count + context];
          if (directory.generation != a.dispatch.identity.generation ||
              directory.epoch != a.epoch)
            continue;
          const auto end = directory.packet_begin + directory.packet_count;
          received_packets = end > received_packets ? end : received_packets;
        }
        std::uint64_t compact_rows = 0ULL;
        for (std::uint64_t sequence = 0; sequence < received_packets;
             ++sequence) {
          const auto header = receive_headers[
              source_partition * a.max_tokens_per_rank + sequence];
          for (std::uint64_t row = header.dst_row;
               row < header.dst_row + header.route_index; ++row)
            forward_return_fragment_index[
                source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                compact_rows++] = row;
        }
        for (std::uint64_t compact_begin = 0ULL;
             compact_begin < compact_rows;
             compact_begin += kForwardReturnChunkRows) {
          if (chunk_count >= a.forward_return_chunk_capacity) {
            publish_local_structured_error(
                status, a.rank, 131U, 1U,
                a.forward_return_chunk_capacity, chunk_count + 1ULL);
            break;
          }
          const auto remaining = compact_rows - compact_begin;
          const auto chunk_rows = static_cast<std::uint32_t>(
              remaining < kForwardReturnChunkRows
                  ? remaining
                  : kForwardReturnChunkRows);
          const auto context = static_cast<std::uint32_t>(
              chunk_count % a.gin_context_count);
          const auto bytes = static_cast<std::uint64_t>(chunk_rows) *
                             a.hidden * sizeof(transport::BFloat16);
          forward_return_chunks[chunk_count] = ForwardReturnChunkRecord{
              compact_begin, chunk_rows, source_node, context,
              chunk_rows, bytes};
          forward_return_chunk_state[chunk_count] = 0ULL;
          for (std::uint32_t rail = 0; rail < a.local_world_size; ++rail)
            forward_return_chunk_ready_count[
                chunk_count * a.local_world_size + rail] = 0ULL;
          context_completion[source_partition * a.gin_context_count +
                             context]
              .return_expected += 1ULL;
          ++chunk_count;
        }
      }
      std::uint64_t source_compact = 0ULL;
      for (std::uint64_t wave_position = 0;
           wave_position < a.dispatch.sizes.packet_count; ++wave_position) {
        const auto packet = a.dispatch_waves.packet_order[wave_position];
        if (a.dispatch.packet_headers[packet].destination_node == a.node)
          continue;
        const auto begin = a.dispatch.packet_fanout_offsets[packet];
        const auto end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
        for (std::uint64_t fanout = begin; fanout < end; ++fanout)
          forward_return_source_compact_index[fanout] = source_compact++;
      }
      scheduler[chunk_count_word] = chunk_count;
      completion_task(completion, transport_state::TaskClass::Return)
          .expected = chunk_count;
      completion->return_local_completion_expected = chunk_count;
    }
    __syncthreads();
    // Install the immutable row-to-proxy/chunk map before publishing the
    // directory. FC1 may progress concurrently on other CTAs; FC2 acquires
    // directory_ready before it can expose a ReturnChunk producer count.
    auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        static_cast<std::byte *>(a.gin_window_base) +
        a.gin_receive_fanout_offset);
    const auto chunk_count = scheduler[4ULL + 2ULL * a.world];
    for (std::uint64_t chunk_index = 0; chunk_index < chunk_count;
         ++chunk_index) {
      const auto chunk = forward_return_chunks[chunk_index];
      const auto source_partition =
          chunk.source_node < a.node ? chunk.source_node
                                     : chunk.source_node - 1U;
      const auto partition_base =
          source_partition * a.max_tokens_per_rank *
          a.dispatch.identity.top_k;
      for (std::uint64_t row_index = threadIdx.x; row_index < chunk.row_count;
           row_index += blockDim.x) {
        const auto fragment = forward_return_fragment_index[
            partition_base + chunk.row_begin + row_index];
        const auto target = receive_fanout[partition_base + fragment];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *peer_chunk_map = peer_at<unsigned long long>(
            a.peer_bases[destination], a.forward_output_ready_epoch_offset);
        ::cuda::atomic_ref<unsigned long long,
                           ::cuda::thread_scope_system>
            published_mapping(peer_chunk_map[target.dst_row]);
        published_mapping.store(
            (static_cast<unsigned long long>(a.rank + 1U) << 48U) |
                (chunk_index + 1ULL),
            ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
        atomicAdd(&completion->bulk_pack_complete, 1ULL);
#endif
      }
    }
    __threadfence_system();
    __syncthreads();
    const auto first_local = a.node * a.local_world_size;
    for (std::uint32_t local = threadIdx.x; local < a.local_world_size;
         local += blockDim.x) {
      const auto peer = first_local + local;
      auto *ready = peer_at<unsigned long long>(
          a.peer_bases[peer], a.gin_local_compute_ready_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          published(ready[a.rank]);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    __syncthreads();
    auto *local_ready = peer_at<unsigned long long>(
        a.local_base, a.gin_local_compute_ready_offset);
    for (std::uint32_t local = threadIdx.x; local < a.local_world_size;
         local += blockDim.x) {
      const auto peer = first_local + local;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          observed(local_ready[peer]);
      while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
             !forward_cancelled(a)) {
      }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          directory_ready(scheduler[6ULL + 2ULL * a.world]);
      directory_ready.store(a.epoch, ::cuda::memory_order_release);
    }
    __syncthreads();
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnDirectoryEnd, GinReturn, 0U,
                            chunk_count, a.epoch);
  }
#endif

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  const auto forward_mblocks = scheduler[3];
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  const auto fc2_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  const bool gin_progress_cta =
      remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count;

  // GIN context CTAs remain transport progress workers for the lifetime of
  // the Dispatch-to-compute interval.  Compute CTAs can consume an exact
  // M-tile as soon as its row counter reaches the immutable expected value;
  // both roles rejoin only at the already-proven Bulk Return grid barrier.
  // This makes role ownership static and keeps the Return state machine
  // unchanged while Dispatch is made asynchronous.
  if (!gin_progress_cta) {
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < 4U; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  const auto forward_warpgroup =
      __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (forward_warpgroup < 2U)
    cutlass::arch::warpgroup_reg_alloc<224>();
  else
    cutlass::arch::warpgroup_reg_dealloc<48>();
  std::uint32_t pipeline_iteration = 0;
  __shared__ unsigned long long task_shared;
  __shared__ unsigned int claim_shared;
  auto *fc1_claims = claims + forward_mblocks;
  auto *fc2_claims = fc1_claims + fc1_tasks;
  constexpr unsigned int kTaskChunk = MEGAMOE_DEBUG_VALIDATE ? 1U : 4U;
  unsigned int task_in_chunk = kTaskChunk;
#if MEGAMOE_DEVICE_TIMELINE
  bool timeline_fc1_started = false;
#endif
#if MEGAMOE_DEBUG_VALIDATE
  if (threadIdx.x == 0U) {
    completion->fc1_scheduler_value =
        a.cancel_epoch
            ? atomicAdd_system(const_cast<unsigned long long *>(a.cancel_epoch),
                               0ULL)
            : ~0ULL;
    completion->bulk_local_ready = 10ULL;
  }
  __syncthreads();
#endif
  while (!forward_cancelled(a)) {
    if (task_in_chunk == kTaskChunk) {
      if (threadIdx.x == 0U)
        task_shared =
            atomicAdd(&scheduler[0], static_cast<unsigned long long>(kTaskChunk));
      __syncthreads();
      task_in_chunk = 0U;
    }
    const auto task = task_shared + task_in_chunk++;
    if (task >= fc1_tasks)
      break;
    const auto mblock = task / fc1_ntiles;
    const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
    const auto mapping = claims[mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    if (threadIdx.x == 0U) {
      claim_shared = atomicCAS(&fc1_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
      completion->fc1_scheduler_value = task;
      completion->bulk_local_ready = 11ULL;
#endif
    }
    __syncthreads();
    if (claim_shared == 0U)
      continue;
    if (expert >= a.dispatch.sizes.owned_expert_count)
      break;
    if constexpr (!StaticLocal)
      if (a.use_gin && threadIdx.x == 0U)
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc1OrDv);
    const auto row =
        a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
    const auto remaining_rows =
        a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
    const auto valid_rows = static_cast<std::uint32_t>(
        remaining_rows < 128ULL ? remaining_rows : 128ULL);
    wait_forward_mblock_rows<StaticLocal>(
        a, expert, static_cast<std::uint64_t>(local_mblock) * 128ULL,
        valid_rows, completion);
    __syncthreads();
#if MEGAMOE_DEVICE_TIMELINE
    if (!timeline_fc1_started) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, XFirstMtileReady, Compute, 0U, task,
                              a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1First, Compute, 0U, task,
                              a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1ClaimStart, Compute, 0U, task,
                              a.epoch);
      timeline_fc1_started = true;
    }
#endif
    const auto n = n_tile * 128U;
    {
      float accum_a[ForwardMma::kNumAccum];
      float accum_b[ForwardMma::kNumAccum];
      forward_wgmma_swiglu_tile(
          x_map, w1_map, static_cast<std::uint32_t>(row),
          static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
          static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                     a.intermediate + n),
          static_cast<std::uint32_t>(a.hidden), smem_a, smem_b,
          full_barriers, empty_barriers, accum_a, accum_b,
          pipeline_iteration);
      visit_forward_accumulator_pair(
          accum_a, accum_b,
          [&](std::uint32_t warpgroup, std::uint32_t m,
              std::uint32_t column, float av0, float av1, float bv0,
              float bv1) {
            const auto block_m = warpgroup * 64U + m;
            const auto output_column =
                static_cast<std::uint64_t>(n) + column;
            if (block_m < valid_rows &&
                output_column + 1ULL < a.intermediate &&
                (a.intermediate & 1ULL) == 0ULL) {
              const auto output_index =
                  (row + block_m) * a.intermediate + output_column;
              // Each WGMMA lane owns two adjacent BF16 outputs.  Publish the
              // pair as one naturally aligned store for S and for each cache
              // plane; scalar struct stores previously doubled the FC1
              // epilogue store instruction count on the cache fast path.
              const auto sigmoid0 = 1.0F / (1.0F + __expf(-av0));
              const auto sigmoid1 = 1.0F / (1.0F + __expf(-av1));
              const auto pack = [](std::uint16_t low,
                                   std::uint16_t high) {
                return static_cast<std::uint32_t>(low) |
                       (static_cast<std::uint32_t>(high) << 16U);
              };
              reinterpret_cast<std::uint32_t *>(s)[output_index / 2ULL] =
                  pack(float_to_bf16_bits((av0 * sigmoid0) * bv0),
                       float_to_bf16_bits((av1 * sigmoid1) * bv1));
              if (a.cache_fc1 && ab_cache) {
                reinterpret_cast<std::uint32_t *>(ab_cache)
                    [output_index / 2ULL] =
                    pack(float_to_bf16_bits(av0), float_to_bf16_bits(av1));
                const auto b_index =
                    a.max_destination_rows * a.intermediate + output_index;
                reinterpret_cast<std::uint32_t *>(ab_cache)[b_index / 2ULL] =
                    pack(float_to_bf16_bits(bv0), float_to_bf16_bits(bv1));
              }
            } else if (block_m < valid_rows) {
              // Retain checked scalar stores for non-main odd-N fixtures.
              const float av[2] = {av0, av1};
              const float bv[2] = {bv0, bv1};
#pragma unroll
              for (std::uint32_t value = 0; value < 2U; ++value) {
                const auto tail_column = output_column + value;
                if (tail_column < a.intermediate) {
                  const auto output_index =
                      (row + block_m) * a.intermediate + tail_column;
                  const auto sigmoid = 1.0F / (1.0F + __expf(-av[value]));
                  s[output_index] = transport::BFloat16{float_to_bf16_bits(
                      (av[value] * sigmoid) * bv[value])};
                  if (a.cache_fc1 && ab_cache) {
                    ab_cache[output_index] =
                        transport::BFloat16{float_to_bf16_bits(av[value])};
                    ab_cache[a.max_destination_rows * a.intermediate +
                             output_index] = transport::BFloat16{
                        float_to_bf16_bits(bv[value])};
                  }
                }
              }
            }
          });
    }
    // FC2 may be claimed by another CTA as soon as this tile increments the
    // m-block counter. Order every lane's S/checkpoint stores first; a fence
    // from only the counter lane would not publish the other lanes' writes.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
          fc1_done[mblock]);
      ready.fetch_add(1ULL, ::cuda::memory_order_release);
    }
    if constexpr (!StaticLocal)
      if (a.use_gin && threadIdx.x == 0U)
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::Fc1OrDv);
    if (threadIdx.x == 0U) {
      atomicExch(&fc1_claims[task], 2ULL);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
    atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());
  MEGAMOE_TIMELINE_RECORD(a.timeline, Fc1Last, Compute, 0U, fc1_tasks,
                          a.epoch);
  forward_publish_progress(a, kForwardFc1Complete, fc1_tasks);

  // FC1 transactions issued by this CTA are complete, so this CTA can reset
  // its private shared barriers and consume any M-block whose release counter
  // is ready.  There is deliberately no grid-wide FC1/FC2 barrier: late FC1
  // tiles and ready FC2 tiles overlap across CTAs.
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < 4U; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  pipeline_iteration = 0;

  task_in_chunk = kTaskChunk;
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        directory_ready(scheduler[6ULL + 2ULL * a.world]);
    while (directory_ready.load(::cuda::memory_order_acquire) < a.epoch &&
           !forward_cancelled(a)) {
      if (threadIdx.x == 0U)
        __nanosleep(256U);
      __syncthreads();
    }
  }
  while (!forward_cancelled(a)) {
    if (task_in_chunk == kTaskChunk) {
      if (threadIdx.x == 0U)
        task_shared =
            atomicAdd(&scheduler[1], static_cast<unsigned long long>(kTaskChunk));
      __syncthreads();
      task_in_chunk = 0U;
    }
    const auto task = task_shared + task_in_chunk++;
    if (task >= fc2_tasks)
      break;
    const auto mblock = task / fc2_ntiles;
    const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
    if (threadIdx.x == 0U) {
      claim_shared = atomicCAS(&fc2_claims[task], 0ULL, 1ULL) == 0ULL ? 1U : 0U;
#if MEGAMOE_DEBUG_VALIDATE
      completion->fc2_scheduler_value = task;
#endif
    }
    __syncthreads();
    if (claim_shared == 0U)
      continue;
    if constexpr (!StaticLocal)
      if (a.use_gin && threadIdx.x == 0U)
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Fc2OrDx);
    if (threadIdx.x == 0)
      atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
    const auto mapping = claims[mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        fc1_ready(fc1_done[mblock]);
    while ((fc1_ready.load(::cuda::memory_order_acquire) & 0xffffffffULL) <
           fc1_ntiles) {
    }
    __syncthreads();
    const auto row =
        a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
    const auto remaining_rows =
        a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
    const auto valid_rows = static_cast<std::uint32_t>(
        remaining_rows < 128ULL ? remaining_rows : 128ULL);
    const auto n = n_tile * 256U;
    float accum[ForwardMmaWide::kNumAccum];
    forward_wgmma_wide_tile(
        s_map, w2_map, static_cast<std::uint32_t>(row),
        static_cast<std::uint32_t>(expert * a.hidden + n),
        static_cast<std::uint32_t>(a.intermediate), smem_a, smem_b,
        full_barriers, empty_barriers, accum, pipeline_iteration);
    visit_forward_accumulator_wide(
        accum, [&](std::uint32_t warpgroup, std::uint32_t m,
                   std::uint32_t column, float v0, float v1) {
      const auto output_column = static_cast<std::uint64_t>(n) + column;
      const auto block_m = warpgroup * 64U + m;
      if (block_m < valid_rows && output_column < a.hidden)
        forward_output[(row + block_m) * a.hidden + output_column] =
            transport::BFloat16{float_to_bf16_bits(v0)};
      if (block_m < valid_rows && output_column + 1U < a.hidden)
        forward_output[(row + block_m) * a.hidden + output_column + 1U] =
            transport::BFloat16{float_to_bf16_bits(v1)};
    });
    // Publish this N tile's canonical stores before exposing completion.  The
    // CTA that observes the final N tile owns the M-block return; other CTAs
    // can continue FC2 for unrelated blocks in parallel.
    __threadfence();
    __syncthreads();
    __shared__ unsigned int return_mblock_shared;
    if (threadIdx.x == 0) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          complete(fc1_done[mblock]);
      const auto old = complete.fetch_add(1ULL << 16U,
                                           ::cuda::memory_order_acq_rel);
      return_mblock_shared =
          (((old >> 16U) & 0xffffULL) + 1ULL == fc2_ntiles) ? 1U : 0U;
    }
    __syncthreads();
    if (return_mblock_shared != 0U) {
      // Publish one exact producer fragment per completed canonical row only
      // after every FC2 N tile for this M-block has stored its output. The row
      // map selects its immutable same-rail proxy and ReturnChunk; local rows
      // have mapping zero and never enter the registered GIN staging path.
      __threadfence_system();
      for (std::uint32_t ready_row = threadIdx.x;
           ready_row < valid_rows; ready_row += blockDim.x) {
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published_mapping(forward_output_ready_epoch[row + ready_row]);
        const auto mapping =
            published_mapping.load(::cuda::memory_order_acquire);
        if (mapping != 0ULL) {
          const auto proxy_rank =
              static_cast<std::uint32_t>((mapping >> 48U) - 1ULL);
          const auto chunk = (mapping & ((1ULL << 48U) - 1ULL)) - 1ULL;
          auto *peer_ready_count = peer_at<unsigned long long>(
              a.peer_bases[proxy_rank],
              a.forward_return_chunk_ready_count_offset);
          const auto producer_rail = a.rank % a.local_world_size;
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              ready(peer_ready_count[chunk * a.local_world_size +
                                     producer_rail]);
          ready.fetch_add(1ULL, ::cuda::memory_order_release);
#if MEGAMOE_DEBUG_VALIDATE
          atomicAdd(&completion->bulk_contexts_issuing, 1ULL);
#endif
        }
      }
      __syncthreads();
      if (threadIdx.x == 0U) {
        const auto finished =
            atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
        if (finished == forward_mblocks) {
          status->fc2_complete_clock = device_globaltimer_ns();
          forward_publish_progress(a, kForwardFc2Complete, fc2_tasks);
        }
      }
      __syncthreads();
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, Fc2TileReady, Compute, 0U, task,
                            a.epoch);
    if (return_mblock_shared != 0U &&
        !remote_backend_enabled<StaticLocal>(a.use_socket)) {
      // Only lane 0 produces and consumes the peer-row audit count. Keeping
      // it in CTA shared storage made independent CTAs alias through the
      // cluster-shared address space under racecheck. A thread-local scalar
      // has the same audit semantics and no cross-thread lifetime.
      std::uint32_t return_peer_rows = 0U;
      if (threadIdx.x == 0)
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
      for (std::uint32_t source = threadIdx.x; source < a.world;
           source += blockDim.x)
        return_rows_by_source[source] = 0U;
      __syncthreads();
      for (std::uint32_t local_row = 0; local_row < valid_rows; ++local_row) {
        const auto return_row = row + local_row;
        const auto header = headers[return_row];
        if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
            header.source_rank / a.local_world_size != a.node)
          continue;
        auto *source_contribution = peer_at<transport::BFloat16>(
            a.peer_bases[header.source_rank], a.contribution_offset);
        const auto *source128 = reinterpret_cast<const uint4 *>(
            forward_output + return_row * a.hidden);
        auto *target128 = reinterpret_cast<uint4 *>(
            source_contribution + header.route_index * a.hidden);
        const auto vector_count = a.hidden / 8ULL;
        for (std::uint64_t vector = threadIdx.x;
             vector < vector_count; vector += blockDim.x * 2ULL) {
          target128[vector] = source128[vector];
          const auto second = vector + blockDim.x;
          if (second < vector_count)
            target128[second] = source128[second];
        }
        if (threadIdx.x == 0) {
          ++return_rows_by_source[header.source_rank];
          if (header.source_rank != a.rank)
            ++return_peer_rows;
        }
      }
      if (threadIdx.x == 0 && return_peer_rows != 0U)
        atomicAdd(&status->actual_return_peer_bytes,
                  static_cast<unsigned long long>(return_peer_rows) *
                      a.hidden * sizeof(transport::BFloat16));
      __threadfence_system();
      __syncthreads();
      for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
           source_rank += blockDim.x) {
        if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
            source_rank / a.local_world_size != a.node)
          continue;
        const auto rows = return_rows_by_source[source_rank];
        if (rows == 0U)
          continue;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            completed(scheduler[4U + source_rank]);
        const auto old = completed.fetch_add(rows,
                                              ::cuda::memory_order_acq_rel);
        const auto expected = old >> 32U;
        const auto actual = (old & 0xffffffffULL) + rows;
        if (actual == expected) {
          auto *source_signal = peer_at<unsigned long long>(
              a.peer_bases[source_rank], a.outgoing_sequence_offset);
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              ready(source_signal[a.rank * a.channels]);
          ready.store(a.epoch, ::cuda::memory_order_release);
        }
      }
      __syncthreads();
    }
    if constexpr (!StaticLocal)
      if (a.use_gin && threadIdx.x == 0U)
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::Fc2OrDx);
    if (threadIdx.x == 0U) {
      atomicExch(&fc2_claims[task], 2ULL);
    }
    __syncthreads();
  }
  } else {
#if __has_include(<nccl_device.h>)
    // Dedicated context CTAs overlap FC2 with Return.  Every context owns a
    // disjoint static subset of chunks and keeps up to one completion batch
    // outstanding; a not-yet-ready chunk never blocks another context.
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        directory_ready(scheduler[6ULL + 2ULL * a.world]);
    while (directory_ready.load(::cuda::memory_order_acquire) < a.epoch &&
           !forward_cancelled(a)) {
      if (threadIdx.x == 0U)
        __nanosleep(256U);
      __syncthreads();
    }
    // chunk_count is part of the immutable directory publication. Reading it
    // before the acquire wait allowed a fast context to cache the setup-time
    // zero, conclude that its task space was empty, and leave while the peer
    // still expected a ReturnChunk signal.
    const auto chunk_count = scheduler[4ULL + 2ULL * a.world];
    const ncclGin gin(a.gin_dev_comm, context);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        window_bytes + a.gin_receive_fanout_offset);
    auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
        window_bytes + a.gin_return_staging_offset);
    __shared__ unsigned long long selected_chunk;
    __shared__ unsigned int chunk_ready;
    __shared__ unsigned int batch_count;
    __shared__ unsigned long long batch_chunks[kForwardReturnCompletionBatch];
    __shared__ unsigned long long context_expected;
    __shared__ unsigned long long context_completed;
    __shared__ unsigned long long issued_total;
    if (threadIdx.x == 0U) {
      context_expected = 0ULL;
      context_completed = 0ULL;
      issued_total = 0ULL;
      batch_count = 0U;
      for (std::uint64_t chunk = context; chunk < chunk_count;
           chunk += a.gin_context_count)
        ++context_expected;
    }
    __syncthreads();

    while (context_completed < context_expected && !forward_cancelled(a)) {
      selected_chunk = ~0ULL;
      __syncthreads();
      for (std::uint64_t candidate = context;
           candidate < chunk_count && selected_chunk == ~0ULL;
           candidate += a.gin_context_count) {
        if (threadIdx.x == 0U) {
          const auto chunk = forward_return_chunks[candidate];
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_device>
              state(forward_return_chunk_state[candidate]);
          unsigned long long producer_count = 0ULL;
#if MEGAMOE_DEBUG_VALIDATE
          unsigned long long producer_rails = 0ULL;
#endif
          for (std::uint32_t producer_rail = 0;
               producer_rail < a.local_world_size; ++producer_rail) {
            ::cuda::atomic_ref<unsigned long long,
                               ::cuda::thread_scope_system>
                producers(forward_return_chunk_ready_count[
                    candidate * a.local_world_size + producer_rail]);
            const auto rail_count =
                producers.load(::cuda::memory_order_acquire);
            producer_count += rail_count;
#if MEGAMOE_DEBUG_VALIDATE
            producer_rails |=
                (rail_count & 0xffULL) << (producer_rail * 8U);
#endif
          }
#if MEGAMOE_DEBUG_VALIDATE
          const auto source_partition =
              chunk.source_node < a.node ? chunk.source_node
                                         : chunk.source_node - 1U;
          auto &debug_record = context_completion[
              source_partition * a.gin_context_count + context];
          debug_record.last_cursor = producer_count;
          debug_record.dispatch_fanout_rows = producer_rails;
#endif
          chunk_ready =
              state.load(::cuda::memory_order_relaxed) == 0ULL &&
                      producer_count == chunk.expected_fragments
                  ? 1U
                  : 0U;
        }
        __syncthreads();
        if (threadIdx.x == 0U && chunk_ready != 0U &&
            atomicCAS(&forward_return_chunk_state[candidate], 0ULL, 1ULL) ==
                0ULL)
          selected_chunk = candidate;
        __syncthreads();
      }
      if (selected_chunk == ~0ULL) {
        // Before the first FC2 tile becomes ready there can be tens of
        // milliseconds of GEMM work. Tight polling from every GIN context
        // needlessly saturates L2/atomic fabric and throttles WGMMA CTAs. A
        // bounded device backoff preserves microsecond-scale first-chunk
        // while allowing the compute pipeline to use the machine.
        if (threadIdx.x == 0U)
          __nanosleep(2048U);
        __syncthreads();
        continue;
      }

      const auto chunk = forward_return_chunks[selected_chunk];
      const auto source_partition =
          chunk.source_node < a.node ? chunk.source_node
                                     : chunk.source_node - 1U;
      const auto partition_base = source_partition * a.max_tokens_per_rank *
                                  a.dispatch.identity.top_k;
      const auto packed_base = partition_base + chunk.row_begin;
      const auto vector_count = a.hidden / 8ULL;
      for (std::uint64_t item = threadIdx.x;
           item < static_cast<std::uint64_t>(chunk.row_count) * vector_count;
           item += blockDim.x) {
        const auto row_index = item / vector_count;
        const auto vector = item % vector_count;
        const auto fragment = forward_return_fragment_index[
            partition_base + chunk.row_begin + row_index];
        const auto target = receive_fanout[partition_base + fragment];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        const auto *destination_output = peer_at<transport::BFloat16>(
            a.peer_bases[destination], a.forward_output_rows_offset);
        reinterpret_cast<uint4 *>(return_staging)[
            (packed_base + row_index) * vector_count + vector] =
            reinterpret_cast<const uint4 *>(destination_output)[
                target.dst_row * vector_count + vector];
      }
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0U) {
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Return);
        const auto record_index =
            source_partition * a.gin_context_count + context;
        context_completion[record_index].return_claimed += 1ULL;
        context_completion[record_index].inflight += 1ULL;
        MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnGinPutIssue, GinReturn,
                                context, chunk.payload_bytes, a.epoch);
        gin.put(rail, static_cast<int>(chunk.source_node), a.gin_window,
                a.gin_return_receive_offset +
                    chunk.row_begin * a.hidden *
                        sizeof(transport::BFloat16),
                a.gin_window,
                a.gin_return_staging_offset +
                    packed_base * a.hidden * sizeof(transport::BFloat16),
                chunk.payload_bytes, ncclGin_SignalInc{1},
                ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        atomicAdd(&status->gin_return_put_count, 1ULL);
        atomicAdd(&status->gin_signal_count, 1ULL);
        atomicAdd(&status->gin_return_payload_bytes, chunk.payload_bytes);
        batch_chunks[batch_count++] = selected_chunk;
        ++issued_total;
      }
      __syncthreads();

      if (batch_count == kForwardReturnCompletionBatch ||
          issued_total == context_expected) {
        if (threadIdx.x == 0U) {
          gin.waitCounter(ncclCoopThread{}, 0, issued_total);
          for (std::uint32_t batch = 0; batch < batch_count; ++batch) {
            atomicExch(&forward_return_chunk_state[batch_chunks[batch]], 2ULL);
            complete_completion_tasks(completion,
                                      transport_state::TaskClass::Return);
            const auto completed_chunk =
                forward_return_chunks[batch_chunks[batch]];
            const auto completed_partition =
                completed_chunk.source_node < a.node
                    ? completed_chunk.source_node
                    : completed_chunk.source_node - 1U;
            auto &record = context_completion[
                completed_partition * a.gin_context_count + context];
            record.return_completed += 1ULL;
            record.inflight -= 1ULL;
            record.local_completion += 1ULL;
            record.last_progress_globaltimer = device_globaltimer_ns();
            ++context_completed;
          }
          atomicAdd(&completion->return_local_completion_completed,
                    batch_count);
          batch_count = 0U;
        }
        __syncthreads();
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnLocalCompletion, GinReturn,
                            context, context_completed, a.epoch);
#endif
  }
#if __has_include(<nccl_device.h>)
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    // All FC2 rows are now available. Each same-rail proxy gathers its remote
    // packet's local fan-out directly from node-local expert arenas.  The full
    // persistent grid performs the NVLink gather/scatter; only the bounded GIN
    // context CTAs enqueue the batched rail puts.
    if (threadIdx.x == 0U) {
      atomicAdd(gin_progress_cta
                    ? &completion->progress_ctas_at_bulk_return
                    : &completion->compute_ctas_at_bulk_return,
                1ULL);
      completion->grid_ctas_expected = gridDim.x;
    }
    grid.sync();
    auto *gin_window_bytes = static_cast<std::byte *>(a.gin_window_base);
    const auto vector_count = a.hidden / 8ULL;
    const auto global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto global_threads =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    if (a.forward_return_chunk_capacity == 0ULL) {
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnPackStart, GinReturn, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    const auto first_local = a.node * a.local_world_size;
    if (blockIdx.x == 0U) {
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        auto *ready = peer_at<unsigned long long>(
            a.peer_bases[peer], a.gin_local_compute_ready_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(ready[a.rank]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      __syncthreads();
      auto *local_ready = peer_at<unsigned long long>(
          a.local_base, a.gin_local_compute_ready_offset);
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(local_ready[peer]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !forward_cancelled(a)) {
        }
      }
    }
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U)
      completion->bulk_local_ready = 1ULL;
    auto *gin_receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        gin_window_bytes + a.gin_receive_header_offset);
    auto *gin_receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        gin_window_bytes + a.gin_receive_fanout_offset);
    auto *gin_return_staging = reinterpret_cast<transport::BFloat16 *>(
        gin_window_bytes + a.gin_return_staging_offset);
#if MEGAMOE_DEVICE_TIMELINE
    __shared__ unsigned long long return_pack_cta_bytes;
    if (threadIdx.x == 0U)
      return_pack_cta_bytes = 0ULL;
    __syncthreads();
    unsigned long long return_pack_thread_bytes = 0ULL;
#endif
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
      for (std::uint64_t item = global_thread;
           item < signal.expected_packets * packet_vectors;
           item += global_threads) {
        const auto sequence = item / packet_vectors;
        const auto within_packet = item % packet_vectors;
        const auto fanout = within_packet / vector_count;
        const auto vector = within_packet % vector_count;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = gin_receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = gin_receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        const auto *destination_output = peer_at<transport::BFloat16>(
            a.peer_bases[destination], a.forward_output_rows_offset);
        reinterpret_cast<uint4 *>(gin_return_staging)[
            ((source_partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout) *
                 vector_count) +
            vector] = reinterpret_cast<const uint4 *>(destination_output)[
            target.dst_row * vector_count + vector];
#if MEGAMOE_DEVICE_TIMELINE
        return_pack_thread_bytes += sizeof(uint4);
#endif
      }
    }
#if MEGAMOE_DEVICE_TIMELINE
    atomicAdd(&return_pack_cta_bytes, return_pack_thread_bytes);
    __syncthreads();
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnPackCtaBytes, GinReturn, 0U,
                            return_pack_cta_bytes, a.epoch);
#endif
    __threadfence_system();
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U)
      completion->bulk_pack_complete = 1ULL;
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnPackEnd, GinReturn, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    if (blockIdx.x < a.gin_context_count) {
      const auto context = static_cast<std::uint32_t>(blockIdx.x);
      if (threadIdx.x == 0U)
        atomicAdd(&completion->bulk_contexts_issuing, 1ULL);
      const ncclGin gin(a.gin_dev_comm, context);
      const auto rail = ncclTeamRail(a.gin_dev_comm);
      auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
      auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
          window_bytes + a.gin_receive_header_offset);
      unsigned long long return_puts = 0ULL;
      for (std::uint32_t signal_index = 0;
           signal_index < a.dispatch.sizes.incoming_signal_count;
           ++signal_index) {
        const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
        if (signal.source_node == a.node)
          continue;
        const auto source_partition =
            signal.source_node < a.node ? signal.source_node
                                        : signal.source_node - 1U;
        const auto sequence_begin =
            signal.expected_packets * context / a.gin_context_count;
        const auto sequence_end = signal.expected_packets * (context + 1U) /
                                  a.gin_context_count;
        std::uint64_t first_return = 0;
        std::uint64_t last_return = 0;
        for (std::uint64_t sequence = sequence_begin;
             sequence < sequence_end; ++sequence) {
          const auto packet =
              source_partition * a.max_tokens_per_rank + sequence;
          const auto header = receive_headers[packet];
          const auto fanout_count = header.route_index;
          if (sequence == sequence_begin)
            first_return = header.dst_row;
          last_return = header.dst_row + fanout_count;
        }
        __syncthreads();
        if (threadIdx.x == 0U && last_return != first_return) {
          claim_completion_tasks(completion,
                                 transport_state::TaskClass::Return);
          const auto record_index =
              source_partition * a.gin_context_count + context;
          context_completion[record_index].return_claimed += 1ULL;
          context_completion[record_index].inflight += 1ULL;
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, ReturnGinPutIssue, GinReturn, context,
              (last_return - first_return) * a.hidden *
                  sizeof(transport::BFloat16),
              a.epoch);
          gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
                  a.gin_return_receive_offset +
                      first_return * a.hidden *
                          sizeof(transport::BFloat16),
                  a.gin_window,
                  a.gin_return_staging_offset +
                      (source_partition * a.max_tokens_per_rank *
                               a.dispatch.identity.top_k +
                       first_return) *
                          a.hidden * sizeof(transport::BFloat16),
                  (last_return - first_return) * a.hidden *
                      sizeof(transport::BFloat16),
                  ncclGin_SignalInc{1}, ncclGin_CounterInc{0},
                  ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          atomicAdd(&status->gin_return_put_count, 1ULL);
          atomicAdd(&status->gin_signal_count, 1ULL);
          atomicAdd(&status->gin_return_payload_bytes,
                    (last_return - first_return) * a.hidden *
                        sizeof(transport::BFloat16));
          ++return_puts;
        }
        __syncthreads();
      }
      if (threadIdx.x == 0U && return_puts != 0ULL)
        gin.waitCounter(ncclCoopThread{}, 0, return_puts);
      if (threadIdx.x == 0U && return_puts != 0ULL) {
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::Return,
                                  return_puts);
        atomicAdd(&completion->return_local_completion_completed,
                  return_puts);
        for (std::uint32_t signal_index = 0;
             signal_index < a.dispatch.sizes.incoming_signal_count;
             ++signal_index) {
          const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
          if (signal.source_node == a.node)
            continue;
          const auto begin =
              signal.expected_packets * context / a.gin_context_count;
          const auto end = signal.expected_packets * (context + 1U) /
                           a.gin_context_count;
          if (begin == end)
            continue;
          const auto source_partition =
              signal.source_node < a.node ? signal.source_node
                                          : signal.source_node - 1U;
          auto &record = context_completion[
              source_partition * a.gin_context_count + context];
          record.return_completed += 1ULL;
          record.inflight -= 1ULL;
          record.local_completion += 1ULL;
          record.last_progress_globaltimer = device_globaltimer_ns();
        }
      }
      MEGAMOE_TIMELINE_RECORD(a.timeline, ReturnLocalCompletion, GinReturn,
                              context, return_puts, a.epoch);
      __syncthreads();
      unsigned long long outgoing_remote_packets = 0ULL;
      unsigned long long outgoing_remote_fanout = 0ULL;
      if (threadIdx.x == 0U) {
        std::uint64_t total = 0;
        for (std::uint64_t packet = 0;
             packet < a.dispatch.sizes.packet_count; ++packet)
          if (a.dispatch.packet_headers[packet].destination_node != a.node)
            ++total;
        const auto begin = total * context / a.gin_context_count;
        const auto end = total * (context + 1U) / a.gin_context_count;
        outgoing_remote_packets = end - begin;
        for (std::uint64_t packet = 0;
             packet < a.dispatch.sizes.packet_count; ++packet) {
          const auto descriptor = a.dispatch.packet_headers[packet];
          if (descriptor.destination_node == a.node ||
              descriptor.sequence_on_link < begin ||
              descriptor.sequence_on_link >= end)
            continue;
          outgoing_remote_fanout +=
              a.dispatch.packet_fanout_offsets[packet + 1U] -
              a.dispatch.packet_fanout_offsets[packet];
        }
        if (outgoing_remote_packets != 0ULL)
          gin.waitSignal(ncclCoopThread{}, 1, 1);
        MEGAMOE_TIMELINE_RECORD(a.timeline, SourceSignalObserved, SourceRelay,
                                context, outgoing_remote_packets, a.epoch);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            phase(scheduler[2]);
        phase.fetch_add(1ULL, ::cuda::memory_order_release);
      }
      __syncthreads();
      if (threadIdx.x == 0U)
        atomicAdd(&status->actual_return_peer_bytes,
                  outgoing_remote_fanout *
                      a.hidden * sizeof(transport::BFloat16));
    }
    } else if (blockIdx.x < a.gin_context_count) {
      // The destination issued one signal per immutable ReturnChunk.  The
      // source derives the same chunk cardinality from its frozen outgoing
      // fan-out descriptors, so no dynamic producer can extend the terminal
      // condition after the kernel starts.
      const auto context = static_cast<std::uint32_t>(blockIdx.x);
      const ncclGin gin(a.gin_dev_comm, context);
      unsigned long long outgoing_remote_fanout = 0ULL;
      if (threadIdx.x == 0U) {
        for (std::uint64_t packet = 0;
             packet < a.dispatch.sizes.packet_count; ++packet) {
          const auto descriptor = a.dispatch.packet_headers[packet];
          if (descriptor.destination_node == a.node)
            continue;
          const auto begin = a.dispatch.packet_fanout_offsets[packet];
          const auto end = a.dispatch.packet_fanout_offsets[packet + 1ULL];
          outgoing_remote_fanout += end - begin;
        }
        const auto chunks =
            (outgoing_remote_fanout + kForwardReturnChunkRows - 1ULL) /
            kForwardReturnChunkRows;
        const auto context_chunks =
            chunks > context
                ? 1ULL + (chunks - 1ULL - context) / a.gin_context_count
                : 0ULL;
        if (context_chunks != 0ULL)
          gin.waitSignal(ncclCoopThread{}, 1, context_chunks);
        MEGAMOE_TIMELINE_RECORD(a.timeline, SourceSignalObserved, SourceRelay,
                                context, context_chunks, a.epoch);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
            phase(scheduler[2]);
        phase.fetch_add(1ULL, ::cuda::memory_order_release);
        if (context == 0U)
          atomicAdd(&status->actual_return_peer_bytes,
                    outgoing_remote_fanout * a.hidden *
                        sizeof(transport::BFloat16));
      }
      __syncthreads();
    }
    grid.sync();
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          phase(scheduler[2]);
      while (phase.load(::cuda::memory_order_acquire) <
             a.gin_context_count) {
      }
    }
    __syncthreads();
    auto *gin_return_receive = reinterpret_cast<transport::BFloat16 *>(
        gin_window_bytes + a.gin_return_receive_offset);
    auto *local_contribution = peer_at<transport::BFloat16>(
        a.local_base, a.contribution_offset);
    MEGAMOE_TIMELINE_RECORD(a.timeline, SourceRelayStart, SourceRelay, 0U,
                            a.dispatch.sizes.packet_count, a.epoch);
    const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
    for (std::uint64_t item = global_thread;
         item < a.dispatch.sizes.packet_count * packet_vectors;
         item += global_threads) {
      const auto packet = item / packet_vectors;
      const auto within_packet = item % packet_vectors;
      const auto fanout_slot = within_packet / vector_count;
      const auto vector = within_packet % vector_count;
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (fanout_slot >= fanout_end - fanout_begin)
        continue;
      const auto fanout = fanout_begin + fanout_slot;
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = descriptor.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      reinterpret_cast<uint4 *>(local_contribution)[
          route * vector_count + vector] =
          reinterpret_cast<const uint4 *>(gin_return_receive)[
              forward_return_source_compact_index[fanout] * vector_count +
              vector];
    }
    __threadfence_system();
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
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
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, SourceRelayEnd, SourceRelay, 0U,
                            a.dispatch.sizes.route_count, a.epoch);
    if (blockIdx.x == 0U) {
      for (std::uint32_t destination = threadIdx.x;
           destination < a.world; destination += blockDim.x) {
        if (destination / a.local_world_size == a.node)
          continue;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(peer_at<unsigned long long>(
                a.local_base, a.outgoing_sequence_offset)
                      [destination * a.channels]);
        ready.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    grid.sync();
  }
#endif
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    publish_local_structured_error(status, a.rank, 90U, 1U, 1ULL, 0ULL);
  }
#endif
  if (remote_backend_enabled<StaticLocal>(a.use_socket)) {
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0 &&
        a.injected_status_error_rank == static_cast<int>(a.rank))
      publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
    grid.sync();
    publish_production_terminal_status<StaticLocal>(
        a, status, ProductionStatusPhase::Forward,
        a.production_error_records_offset,
        a.gin_production_error_records_offset);
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
      status->fc1_tasks_done = fc1_tasks;
      status->fc2_tasks_done = fc2_tasks;
      const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
      const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
      status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                          fc2_tasks * fc2_k_blocks * 2ULL;
      status->wgmma_instructions =
          (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) *
          2ULL * (64ULL / ForwardMma::K);
#endif
      status->return_complete_clock = device_globaltimer_ns();
      status->complete_clock = status->return_complete_clock;
    }
    forward_publish_progress(a, kForwardFc2Complete, scheduler[3]);
    forward_publish_progress(a, kForwardTerminal, a.epoch);
    return;
  }
  if (threadIdx.x == 0)
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());
  forward_publish_progress(a, kForwardReturnObserved, scheduler[3]);

  // A CTA may begin reducing tokens whose destination-rank bulk signals are
  // already visible while other CTAs finish returning later M-blocks.  The
  // per-destination acquire below is the dependency; a grid barrier here only
  // serialized independent return and reduce work.
  auto *contribution = peer_at<transport::BFloat16>(
      a.local_base, a.contribution_offset);
  auto *source_ready = peer_at<unsigned long long>(
      a.local_base, a.outgoing_sequence_offset);
  auto *combined = peer_at<transport::BFloat16>(
      a.local_base, a.combined_offset);
  constexpr std::uint32_t kWarpsPerCta = 12U;
  constexpr std::uint32_t kWarpsPerToken = 1U;
  constexpr std::uint32_t kTokensPerCta = kWarpsPerCta / kWarpsPerToken;
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                          a.dispatch.identity.num_tokens, a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                          a.dispatch.identity.num_tokens, a.epoch);
  const auto token_warp = warp % kWarpsPerToken;
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kTokensPerCta +
           warp / kWarpsPerToken;
       token < a.dispatch.identity.num_tokens && !forward_cancelled(a);
       token += static_cast<std::uint64_t>(gridDim.x) * kTokensPerCta) {
    if constexpr (!StaticLocal)
      if (a.use_gin && lane == 0U)
        claim_completion_tasks(completion,
                               transport_state::TaskClass::Reduce);
    if (lane == 0U && status->reduced_tokens == ~0ULL)
      atomicMin(&status->reduced_tokens, device_globaltimer_ns());
    unsigned long long lane_route = 0ULL;
    float lane_gate = 0.0F;
    if (lane < a.dispatch.identity.top_k) {
      const auto slot = a.dispatch.combine_order[
          token * a.dispatch.identity.top_k + lane];
      lane_route = token * a.dispatch.identity.top_k + slot;
      lane_gate = a.gates[lane_route];
      const auto destination = a.dispatch.route_dst_rank[lane_route];
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(source_ready[destination * a.channels]);
      while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
             !forward_cancelled(a)) {
      }
    }
    __syncwarp();
    auto *output8 = reinterpret_cast<uint4 *>(combined + token * a.hidden);
    for (std::uint64_t vector = lane + token_warp * 32U;
         vector < a.hidden / 8ULL; vector += kWarpsPerToken * 32ULL) {
      float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F,
                       0.0F, 0.0F, 0.0F, 0.0F};
      for (std::uint32_t order = 0;
           order < a.dispatch.identity.top_k; ++order) {
        const auto route = __shfl_sync(0xffffffffU, lane_route, order);
        const auto gate = __shfl_sync(0xffffffffU, lane_gate, order);
        const auto packed = reinterpret_cast<const uint4 *>(
            contribution + route * a.hidden)[vector];
        const unsigned int words[4] = {
            packed.x, packed.y, packed.z, packed.w};
#pragma unroll
        for (std::uint32_t word = 0; word < 4U; ++word) {
          sums[word * 2U] = gate_after_accumulate(
              sums[word * 2U], gate,
              static_cast<std::uint16_t>(words[word]));
          sums[word * 2U + 1U] = gate_after_accumulate(
              sums[word * 2U + 1U], gate,
              static_cast<std::uint16_t>(words[word] >> 16U));
        }
      }
      unsigned int words[4];
#pragma unroll
      for (std::uint32_t word = 0; word < 4U; ++word)
        words[word] =
            static_cast<unsigned int>(float_to_bf16_bits(sums[word * 2U])) |
            (static_cast<unsigned int>(
                 float_to_bf16_bits(sums[word * 2U + 1U]))
             << 16U);
      output8[vector] = make_uint4(words[0], words[1], words[2], words[3]);
    }
    __syncwarp();
    if constexpr (!StaticLocal)
      if (a.use_gin && lane == 0U)
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::Reduce);
  }
  if (lane == 0U)
    atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
  MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                          a.dispatch.identity.num_tokens, a.epoch);
  __syncthreads();
  grid.sync();
  forward_publish_progress(a, kForwardReduceComplete,
                           a.dispatch.identity.num_tokens);
  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      a.injected_status_error_rank == static_cast<int>(a.rank))
    publish_local_structured_error(status, a.rank, 120U, 1U, 0ULL, 1ULL);
  grid.sync();
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      if (blockIdx.x == 0U && threadIdx.x == 0U) {
        completion->ring_credit_restored = completion->ring_credit_expected;
        publish_elected_terminal(completion, a.epoch, status->error != 0U);
      }
      __syncthreads();
      if (blockIdx.x != 0U) {
        // Elect one lane to poll.  forward_cancelled() contains warp-level
        // collectives used by the normal worker loops, so evaluating it from
        // independently reconverging lanes at terminal can strand the rest of
        // warp 0 in WARPSYNC even after the completion epoch is visible.
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long,
                             ::cuda::thread_scope_system>
              terminal(completion->kernel_complete_epoch);
          while (terminal.load(::cuda::memory_order_acquire) < a.epoch) {
          }
        }
        __syncthreads();
        return;
      }
    }
  }
  // Only the elected terminal CTA performs the production status agreement.
  // GIN progress CTAs and compute CTAs have already proved their static task
  // spaces complete and now leave after observing kernel_complete_epoch.  In
  // particular, they never participate in a second grid-wide rendezvous whose
  // progress would otherwise depend on CTA-local control-flow state.
  publish_production_terminal_status<StaticLocal>(
      a, status, ProductionStatusPhase::Forward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset);
  if constexpr (StaticLocal) {
    grid.sync();
  } else if (!a.use_gin) {
    grid.sync();
  } else {
    __syncthreads();
  }
  // Every local return CTA has fenced before publishing its destination
  // signal, and every local Reduce warp acquired all of its destination
  // signals before reaching this grid completion.  Consequently no peer can
  // still write this slot and no local worker can still read it; a second
  // all-rank terminal rendezvous would only duplicate those dependencies.
  if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    status->fc1_tasks_done = fc1_tasks;
    status->fc2_tasks_done = fc2_tasks;
    const auto fc1_k_blocks = (a.hidden + 63ULL) / 64ULL;
    const auto fc2_k_blocks = (a.intermediate + 63ULL) / 64ULL;
    status->tma_loads = fc1_tasks * 2ULL * fc1_k_blocks * 2ULL +
                        fc2_tasks * fc2_k_blocks * 2ULL;
    status->wgmma_instructions =
        (fc1_tasks * 2ULL * fc1_k_blocks + fc2_tasks * fc2_k_blocks) *
        2ULL * (64ULL / ForwardMma::K);
#endif
    status->complete_clock = device_globaltimer_ns();
  }
  forward_publish_progress(a, kForwardTerminal, a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
}

#include "cuda/backward/kernel_contract.cuh"

__device__ bool backward_cancelled(const BackwardKernelArgs &a);

template <bool StaticLocal>
__device__ void wait_backward_mblock_rows(const BackwardKernelArgs &a,
                                          std::uint32_t expert,
                                          std::uint64_t local_row,
                                          std::uint32_t valid_rows) {
  if constexpr (StaticLocal)
    return;
  if (!a.use_gin || valid_rows == 0U)
    return;
  const auto end_row = local_row + valid_rows;
  const auto tile_begin = a.dispatch.mtile_row_offsets[expert];
  const auto tile_end = a.dispatch.mtile_row_offsets[expert + 1ULL];
  auto *arrived = peer_at<unsigned long long>(a.local_base,
                                               a.dy_arrived_rows_offset);
  std::uint64_t tile_row = 0ULL;
  for (auto tile = tile_begin; tile < tile_end; ++tile) {
    const auto expected = a.dispatch.mtile_expected_rows[tile];
    const auto tile_row_end = tile_row + expected;
    if (tile_row_end <= local_row) {
      tile_row = tile_row_end;
      continue;
    }
    if (tile_row >= end_row)
      break;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        ready(arrived[tile]);
    while (ready.load(::cuda::memory_order_acquire) < expected &&
           !backward_cancelled(a)) {
    }
    tile_row = tile_row_end;
  }
}

#include "cuda/forward/runtime_shape_fallback.cuh"

// Final generation drain for the trusted single-node team.  All local work
// (including peer dX return and source reduction) is complete before this is
// called.  Publish one monotonic epoch per rank into each peer allocation and
// wait for the corresponding owner-local vector.  This closes the formal
// route-ready -> final-gradient-ready boundary on device and prevents a fast
// rank's next Notify from absorbing the previous generation's rank skew.
// No host collective, reset, or allocation is involved.
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
template <bool StaticLocal>
__device__ void backward_terminal_rendezvous(const BackwardKernelArgs &a) {
  if (blockIdx.x != 0U)
    return;
  const auto first_peer = !StaticLocal && a.use_gin
                              ? a.node * a.local_world_size
                              : 0U;
  const auto peer_count = !StaticLocal && a.use_gin
                              ? a.local_world_size
                              : a.world;
  for (std::uint32_t local_peer = static_cast<std::uint32_t>(threadIdx.x);
       local_peer < peer_count;
       local_peer += static_cast<std::uint32_t>(blockDim.x)) {
    const auto peer = first_peer + local_peer;
    auto *terminal = peer_at<unsigned long long>(
        a.peer_bases[peer], a.terminal_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        published(terminal[a.rank]);
    published.store(a.epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_terminal = peer_at<unsigned long long>(
      a.local_base, a.terminal_epoch_offset);
  for (std::uint32_t local_peer = static_cast<std::uint32_t>(threadIdx.x);
       local_peer < peer_count;
       local_peer += static_cast<std::uint32_t>(blockDim.x)) {
    const auto peer = first_peer + local_peer;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        observed(local_terminal[peer]);
    while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
           !backward_cancelled(a)) {
    }
  }
  __syncthreads();
}
#endif

__device__ void backward_publish_progress(const BackwardKernelArgs &a,
                                          std::uint32_t word,
                                          unsigned long long value) {
  if (a.progress == nullptr || blockIdx.x != 0U || threadIdx.x != 0U)
    return;
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> target(
      a.progress[word]);
  target.store(value, ::cuda::memory_order_release);
}

__device__ __attribute__((unused)) void backward_reduce_token_slice(
    const BackwardKernelArgs &a, const transport::BFloat16 *dx_return,
    transport::BFloat16 *dx_combined, std::uint32_t worker,
    std::uint32_t workers) {
  const auto warp = threadIdx.x / 32U;
  const auto lane = threadIdx.x & 31U;
  auto *source_ready = peer_at<unsigned long long>(
      a.local_base, a.outgoing_signal_offset);
  constexpr std::uint32_t kWarpsPerCta = 12U;
  for (std::uint64_t token =
           static_cast<std::uint64_t>(worker) * kWarpsPerCta + warp;
       token < a.dispatch.identity.num_tokens;
       token += static_cast<std::uint64_t>(workers) * kWarpsPerCta) {
    unsigned long long lane_route = 0ULL;
    if (lane < a.dispatch.identity.top_k) {
      const auto slot = a.dispatch.combine_order[
          token * a.dispatch.identity.top_k + lane];
      lane_route = token * a.dispatch.identity.top_k + slot;
      const auto destination = a.dispatch.route_dst_rank[lane_route];
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
          ready(source_ready[destination * a.channels]);
      while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
             !backward_cancelled(a)) {
      }
      const auto local_first = a.use_gin ? a.node * a.local_world_size : 0U;
      const auto local_end =
          a.use_gin ? local_first + a.local_world_size : a.world;
      if (destination >= local_first && destination < local_end) {
        const auto row = a.dispatch.route_dst_row[lane_route];
        const auto *destination_dgate = peer_at<const float>(
            a.peer_bases[destination], a.dgate_canonical_offset);
        auto *local_dgate =
            peer_at<float>(a.local_base, a.dgate_offset);
        local_dgate[lane_route] = destination_dgate[row];
      }
    }
    __syncwarp();
    auto *output8 = reinterpret_cast<uint4 *>(dx_combined + token * a.hidden);
    for (std::uint64_t vector = lane; vector < a.hidden / 8ULL;
         vector += 32ULL) {
      float sums[8] = {0.0F, 0.0F, 0.0F, 0.0F,
                       0.0F, 0.0F, 0.0F, 0.0F};
      for (std::uint32_t order = 0; order < a.dispatch.identity.top_k;
           ++order) {
        const auto route = __shfl_sync(0xffffffffU, lane_route, order);
        const auto packed = reinterpret_cast<const uint4 *>(
            dx_return + route * a.hidden)[vector];
        const unsigned int words[4] = {packed.x, packed.y, packed.z,
                                       packed.w};
#pragma unroll
        for (std::uint32_t word = 0; word < 4U; ++word) {
          sums[word * 2U] = __fadd_rn(
              sums[word * 2U], bf16_bits_to_float(
                                     static_cast<std::uint16_t>(words[word])));
          sums[word * 2U + 1U] = __fadd_rn(
              sums[word * 2U + 1U],
              bf16_bits_to_float(
                  static_cast<std::uint16_t>(words[word] >> 16U)));
        }
      }
      unsigned int words[4];
#pragma unroll
      for (std::uint32_t word = 0; word < 4U; ++word)
        words[word] = static_cast<unsigned int>(
                          float_to_bf16_bits(sums[word * 2U])) |
                      (static_cast<unsigned int>(float_to_bf16_bits(
                           sums[word * 2U + 1U])) << 16U);
      output8[vector] = make_uint4(words[0], words[1], words[2], words[3]);
    }
  }
}

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
__device__ __attribute__((unused)) void backward_wgmma_tile(
    const CUtensorMap &activation_map, const CUtensorMap &weight_map,
    std::uint32_t reduction_start, std::uint32_t activation_row,
    std::uint32_t weight_row, std::uint32_t reduction,
    transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 4;
  constexpr std::uint32_t kMathWarpGroups = 2;
  constexpr std::uint32_t kStageElements = 128U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups)
#pragma unroll
    for (int index = 0; index < ForwardMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  const auto iterations = reduction / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      const auto k_base = reduction_start + iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &activation_map, &full_barriers[stage], smem_a, k_base,
          activation_row);
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &weight_map, &full_barriers[stage], smem_b, k_base, weight_row);
      full_barriers[stage].arrive_and_expect_tx(
          2U * kStageElements * sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kStageElements + warpgroup * 64U * 64U);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / ForwardMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_smem_desc(
            smem_a + k * ForwardMma::K, 1);
        const auto desc_b = deep_gemm::mma::sm90::make_smem_desc(
            smem_b + k * ForwardMma::K, 1);
        ForwardMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < ForwardMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

// dW mixed-major path.  The left operand is emitted by the dV epilogue as a
// padded [output, canonical-M] K-major matrix.  The right operand stays in its
// canonical row-major [canonical-M, output] storage and is consumed as
// MN-major.  Four 64-column TMA atoms form one 256-column B tile; consequently
// no full dY/X transpose or staging pass is required.
__device__ __attribute__((unused)) void backward_wgmma_dw_k_mn_tile(
    const CUtensorMap &a_map, const CUtensorMap &b_mn_map,
    std::uint32_t a_reduction_start, std::uint32_t b_reduction_start,
    std::uint32_t output_row, std::uint32_t output_column,
    std::uint32_t reduction, transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kMathWarpGroups = 2U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 256U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  }
  const auto iterations = reduction / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      const auto a_k = a_reduction_start + iteration * 64U;
      const auto b_k = b_reduction_start + iteration * 64U;
      deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
          &a_map, &full_barriers[stage], smem_a, a_k, output_row);
      for (std::uint32_t atom = 0; atom < 4U; ++atom)
        deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
            &b_mn_map, &full_barriers[stage],
            smem_b + atom * 64U * 64U, output_column + atom * 64U, b_k);
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / BackwardDwMixedMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::K, 128, 64, 128, cutlass::bfloat16_t>(
            smem_a, warpgroup * 64U, k * BackwardDwMixedMma::K);
        const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, 256, 64, 128, cutlass::bfloat16_t>(
            smem_b, 0U, k * BackwardDwMixedMma::K);
        BackwardDwMixedMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

// Direct dW2 parameter-layout path: dW2 = dY^T @ U. dY remains canonical
// row-major and is consumed as MN-major A; U^T is consumed as K-major B.
// Two 64-row A atoms and two 128-column B atoms form one [128,256] tile.
__device__ void backward_wgmma_dw_mn_k_tile(
    const CUtensorMap &a_mn_map, const CUtensorMap &b_k_map,
    std::uint32_t a_reduction_start, std::uint32_t b_reduction_start,
    std::uint32_t output_row, std::uint32_t output_column,
    std::uint32_t reduction, transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kMathWarpGroups = 2U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 256U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  }
  const auto iterations = reduction / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      const auto a_k = a_reduction_start + iteration * 64U;
      const auto b_k = b_reduction_start + iteration * 64U;
      for (std::uint32_t atom = 0; atom < 2U; ++atom)
        deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
            &a_mn_map, &full_barriers[stage],
            smem_a + atom * 64U * 64U, output_row + atom * 64U, a_k);
      for (std::uint32_t atom = 0; atom < 2U; ++atom)
        deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
            &b_k_map, &full_barriers[stage],
            smem_b + atom * 128U * 64U, b_k,
            output_column + atom * 128U);
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / BackwardDwDirectMma::K; ++k) {
        const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, 128, 64, 128, cutlass::bfloat16_t>(
            smem_a, warpgroup * 64U, k * BackwardDwDirectMma::K);
        const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::K, 256, 64, 128, cutlass::bfloat16_t>(
            smem_b, 0U, k * BackwardDwDirectMma::K);
        BackwardDwDirectMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < BackwardDwDirectMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

// dW consumes padded row-major [K,M] and [K,N] operands directly.  Hopper's
// MN-major WGMMA descriptors express this transpose in the MMA contract, so
// no bulk global-memory transpose is required.
__device__ __attribute__((unused)) void backward_wgmma_dw_mn_tile(
    const CUtensorMap &a_map, const CUtensorMap &b_map,
    std::uint32_t a_column, std::uint32_t b_column,
    std::uint32_t reduction_start, std::uint32_t reduction,
    transport::BFloat16 *smem_a_public,
    transport::BFloat16 *smem_b_public, ForwardBarrier *full_barriers,
    ForwardBarrier *empty_barriers, float *accum,
    std::uint32_t &pipeline_iteration) {
  constexpr std::uint32_t kStages = 4U;
  constexpr std::uint32_t kMathWarpGroups = 2U;
  constexpr std::uint32_t kAStageElements = 128U * 64U;
  constexpr std::uint32_t kBStageElements = 256U * 64U;
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < kMathWarpGroups) {
#pragma unroll
    for (int index = 0; index < BackwardDwMma::kNumAccum; ++index)
      accum[index] = 0.0F;
  }
  const auto iterations = reduction / 64U;
  if (threadIdx.x == kMathWarpGroups * 128U) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      empty_barriers[stage].wait(phase ^ 1U);
      const auto k_base = reduction_start + iteration * 64U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      deep_gemm::tma::copy<256, 64, 0, cutlass::bfloat16_t>(
          &a_map, &full_barriers[stage], smem_a, a_column, k_base);
      deep_gemm::tma::copy<128, 64, 0, cutlass::bfloat16_t>(
          &b_map, &full_barriers[stage], smem_b, b_column, k_base);
      full_barriers[stage].arrive_and_expect_tx(
          (kAStageElements + kBStageElements) *
          sizeof(transport::BFloat16));
    }
  } else if (warpgroup < kMathWarpGroups) {
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto pipeline = pipeline_iteration + iteration;
      const auto stage = pipeline % kStages;
      const auto phase = (pipeline / kStages) & 1U;
      auto *smem_a = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_a_public + stage * kAStageElements);
      auto *smem_b = reinterpret_cast<cutlass::bfloat16_t *>(
          smem_b_public + stage * kBStageElements);
      full_barriers[stage].wait(phase);
#pragma unroll
      for (int index = 0; index < BackwardDwMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
      for (std::uint32_t k = 0; k < 64U / BackwardDwMma::K; ++k) {
        auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, 256, 64, 16, cutlass::bfloat16_t>(
            smem_a, warpgroup * 64U, k * BackwardDwMma::K);
        auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
            cute::UMMA::Major::MN, 128, 64, 16, cutlass::bfloat16_t>(
            smem_b, 0U, k * BackwardDwMma::K);
        BackwardDwMma::wgmma(desc_a, desc_b, accum, 1);
      }
      deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
      for (int index = 0; index < BackwardDwMma::kNumAccum; ++index)
        deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
      deep_gemm::ptx::warpgroup_wait<0>();
      if ((threadIdx.x & 31U) == 0U)
        empty_barriers[stage].arrive();
    }
  }
  __syncthreads();
  pipeline_iteration += iterations;
}

__device__ std::uint32_t backward_wavefront_mblock_expert(
    const BackwardKernelArgs &a, std::uint64_t ordinal,
    std::uint64_t &local_mblock) {
  std::uint64_t max_blocks = 0ULL;
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const std::uint64_t blocks =
        (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
    max_blocks = blocks > max_blocks ? blocks : max_blocks;
  }
  for (std::uint64_t tile = 0; tile < max_blocks; ++tile) {
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto blocks =
          (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
      if (tile >= blocks)
        continue;
      if (ordinal == 0ULL) {
        local_mblock = tile;
        return expert;
      }
      --ordinal;
    }
  }
  local_mblock = 0ULL;
  return a.dispatch.sizes.owned_expert_count;
}

#endif

__device__ bool backward_cancelled(const BackwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
             a.epoch;
}

template <bool StaticLocal>
__global__ __launch_bounds__(384, 1) void
local_p2p_backward_megakernel(
    BackwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap dy_map,
    const __grid_constant__ CUtensorMap dab_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_transpose_map,
    const __grid_constant__ CUtensorMap w1_transpose_map,
    const __grid_constant__ CUtensorMap dy_mn_map,
    const __grid_constant__ CUtensorMap x_mn_map,
    const __grid_constant__ CUtensorMap u_transpose_map,
    const __grid_constant__ CUtensorMap dab_transpose_map,
    const __grid_constant__ CUtensorMap dw1_store_map) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
  MEGAMOE_TIMELINE_RECORD(a.timeline, KernelEnter, Kernel, 0U, 0ULL,
                          a.epoch);
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler = peer_at<unsigned long long>(a.local_base,
                                                a.scheduler_offset);
  transport_state::GenerationCompletionVector *completion = nullptr;
  transport_state::ContextCompletionRecord *context_completion = nullptr;
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      completion = static_cast<transport_state::GenerationCompletionVector *>(
          a.peer_bases[a.world + 2U]);
      context_completion =
          static_cast<transport_state::ContextCompletionRecord *>(
              a.peer_bases[a.world + 3U]);
#if MEGAMOE_DEBUG_VALIDATE
      if (a.progress != nullptr) {
        const auto backward_legacy_words =
            32ULL + static_cast<unsigned long long>(a.progress_experts) * 8ULL;
        const auto debug_words = transport_state::completion_debug_words(
            a.progress_contexts);
        auto *backward_debug = a.progress + backward_legacy_words +
                               kForwardProgressWords + debug_words;
        completion = reinterpret_cast<
            transport_state::GenerationCompletionVector *>(backward_debug);
        context_completion = reinterpret_cast<
            transport_state::ContextCompletionRecord *>(
            backward_debug + transport_state::kGenerationCompletionWords);
      }
#endif
    }
  }
  auto *dw_timeline = peer_at<BackwardExpertDwTimeline>(
      a.local_base, a.dw_timeline_offset);
  auto *claims = peer_at<unsigned long long>(a.local_base, a.claim_offset);
  auto *headers = peer_at<LocalP2pRouteHeader>(a.local_base,
                                               a.direct_headers_offset);
  auto *canonical_x = peer_at<transport::BFloat16>(a.local_base,
                                                    a.canonical_x_offset);
  auto *dy = peer_at<transport::BFloat16>(a.local_base,
                                          a.dy_canonical_offset);
  auto *canonical_gate = peer_at<float>(a.local_base,
                                        a.canonical_gate_offset);
  auto *dab = peer_at<transport::BFloat16>(a.local_base, a.dab_offset);
  auto *dx_return = peer_at<transport::BFloat16>(a.local_base,
                                                  a.dx_return_offset);
  auto *dx_canonical = peer_at<transport::BFloat16>(a.local_base,
                                                     a.dx_canonical_offset);
  auto *dx_combined = peer_at<transport::BFloat16>(a.local_base,
                                                    a.dx_combined_offset);
  auto *dgate_partial = peer_at<float>(a.local_base,
                                       a.dgate_partial_offset);
  auto *dgate_canonical = peer_at<float>(a.local_base,
                                         a.dgate_canonical_offset);
  auto *dgate = peer_at<float>(a.local_base, a.dgate_offset);
  auto *u_t = peer_at<transport::BFloat16>(a.local_base,
                                           a.u_transpose_offset);
  auto *dab_t = peer_at<transport::BFloat16>(a.local_base,
                                             a.dab_transpose_offset);
  auto *dw1 = peer_at<transport::BFloat16>(a.local_base, a.dw1_offset);
  auto *dw2 = peer_at<transport::BFloat16>(a.local_base, a.dw2_offset);
#if MEGAMOE_DEBUG_VALIDATE
  auto *debug_dv = peer_at<transport::BFloat16>(a.local_base,
                                                a.debug_dv_offset);
  auto *debug_coverage = peer_at<std::uint32_t>(a.local_base,
                                                a.debug_coverage_offset);
#endif
  auto *ab = a.use_forward_cache
                 ? peer_at<transport::BFloat16>(a.local_base,
                                                 a.forward_ab_cache_offset)
                       : peer_at<transport::BFloat16>(a.local_base,
                                                 a.recompute_ab_offset);
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardEntered, a.epoch);
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  auto *smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *smem_b = reinterpret_cast<transport::BFloat16 *>(
      storage + 4U * 128U * 64U * sizeof(transport::BFloat16));
  auto *full_barriers = reinterpret_cast<ForwardBarrier *>(
      storage + (4U * 128U * 64U + 4U * 256U * 64U) *
                    sizeof(transport::BFloat16));
  auto *empty_barriers = full_barriers + 4;
#endif
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *status = BackwardDeviceStatus{};
    status->start_clock = device_globaltimer_ns();
    status->dw1_first_ready_clock = ~0ULL;
    status->dw1_first_start_clock = ~0ULL;
    status->dw2_first_ready_clock = ~0ULL;
    status->dw2_first_start_clock = ~0ULL;
    for (std::uint32_t word = 0; word < 32U; ++word)
      scheduler[word] = 0ULL;
    unsigned long long mblocks = 0ULL;
    unsigned long long padded = 0ULL;
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      claims[expert] = padded;
      const auto rows = a.dispatch.expert_row_count[expert];
      padded += (rows + 63ULL) & ~63ULL;
      mblocks += (rows + 127ULL) / 128ULL;
    }
    scheduler[8] = mblocks;
    scheduler[9] = padded;
    status->padded_reduction_rows = padded;
    if constexpr (!StaticLocal) {
      if (a.use_gin) {
        completion->generation = a.dispatch.identity.generation;
        completion->epoch = a.epoch;
        unsigned long long dispatch_waves = 0ULL;
        for (std::uint64_t wave = 0; wave < a.dispatch_waves.wave_count;
             ++wave)
          if (a.dispatch_waves.destination_node[wave] != a.node)
            ++dispatch_waves;
        completion_task(completion,
                        transport_state::TaskClass::DispatchWave)
            .expected = dispatch_waves;
        unsigned long long relay_packets = 0ULL;
        for (std::uint32_t index = 0;
             index < a.dispatch.sizes.incoming_signal_count; ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node != a.node)
            relay_packets += signal.expected_packets;
        }
        completion_task(completion, transport_state::TaskClass::RelayWave)
            .expected = relay_packets;
        completion_task(completion, transport_state::TaskClass::Fc1OrDv)
            .expected = mblocks * ((a.intermediate + 255ULL) / 256ULL);
        completion_task(completion, transport_state::TaskClass::Fc2OrDx)
            .expected = mblocks * ((a.hidden + 255ULL) / 256ULL);
        completion_task(completion, transport_state::TaskClass::Dw1)
            .expected = static_cast<unsigned long long>(
                            a.dispatch.sizes.owned_expert_count) *
                        ((2ULL * a.intermediate + 127ULL) / 128ULL) *
                        ((a.hidden + 255ULL) / 256ULL);
        completion_task(completion, transport_state::TaskClass::Dw2)
            .expected = static_cast<unsigned long long>(
                            a.dispatch.sizes.owned_expert_count) *
                        ((a.hidden + 127ULL) / 128ULL) *
                        ((a.intermediate + 255ULL) / 256ULL);
        completion_task(completion,
                        transport_state::TaskClass::SourceArrival)
            .expected = a.dispatch.sizes.route_count;
        completion_task(completion, transport_state::TaskClass::Reduce)
            .expected = a.dispatch.identity.num_tokens;
        unsigned long long return_tasks = 0ULL;
        for (std::uint32_t index = 0;
             index < a.dispatch.sizes.incoming_signal_count; ++index) {
          const auto signal = a.dispatch.incoming_proxy_signals[index];
          if (signal.source_node == a.node)
            continue;
          for (std::uint32_t context = 0; context < a.gin_context_count;
               ++context) {
            const auto begin =
                signal.expected_packets * context / a.gin_context_count;
            const auto end = signal.expected_packets * (context + 1U) /
                             a.gin_context_count;
            if (end != begin)
              ++return_tasks;
          }
        }
        completion_task(completion, transport_state::TaskClass::Return)
            .expected = return_tasks;
        completion->return_local_completion_expected = return_tasks;
        completion->directory_expected =
            static_cast<unsigned long long>(
                a.dispatch.identity.num_nodes - 1U) *
            a.gin_context_count;
        completion->dispatch_local_completion_expected =
            completion->directory_expected + dispatch_waves;
      }
    }
  }
  grid.sync();
#if MEGAMOE_DEBUG_VALIDATE
  if (a.capture_stage_storage)
    for (std::uint64_t route =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         route < a.dispatch.sizes.route_count;
         route += static_cast<std::uint64_t>(gridDim.x) * blockDim.x)
      debug_coverage[route] = 0U;
  grid.sync();
#endif
  for (std::uint32_t expert =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       expert < a.dispatch.sizes.owned_expert_count;
       expert += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
#if MEGAMOE_DEBUG_VALIDATE
    dw_timeline[expert] = BackwardExpertDwTimeline{};
    if (a.dispatch.expert_row_count[expert] == 0ULL) {
      dw_timeline[expert].ready_clock = status->start_clock;
      atomicMin(&status->dw1_first_ready_clock, status->start_clock);
      atomicMin(&status->dw2_first_ready_clock, status->start_clock);
    }
#endif
    if (a.progress != nullptr && expert < a.progress_experts) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> rows(
          a.progress[kBackwardProgressExpertBase + expert * 8U]);
      rows.store(a.dispatch.expert_row_count[expert],
                 ::cuda::memory_order_release);
    }
  }
  grid.sync();
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardSetupComplete, scheduler[8]);

  // Build an immutable padded-canonical lookup. dW reductions use this to
  // preserve canonical M order while giving every TMA reduction a 64-row tail.
  for (std::uint32_t expert = blockIdx.x;
       expert < a.dispatch.sizes.owned_expert_count; expert += gridDim.x) {
    const auto padded_prefix = claims[expert];
    const auto row_prefix = a.dispatch.expert_row_prefix[expert];
    const auto rows = a.dispatch.expert_row_count[expert];
    const auto padded_rows = (rows + 63ULL) & ~63ULL;
    for (std::uint64_t row = threadIdx.x; row < padded_rows;
         row += blockDim.x)
      scheduler[64ULL + padded_prefix + row] =
          row < rows ? row_prefix + row : ~0ULL;
  }
  for (std::uint64_t mblock =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       mblock < scheduler[8];
       mblock += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    std::uint64_t local = 0;
    const auto expert =
        backward_wavefront_mblock_expert(a, mblock, local);
    claims[a.dispatch.sizes.owned_expert_count + mblock] =
        (static_cast<unsigned long long>(expert) << 32U) |
        static_cast<std::uint32_t>(local);
#endif
  }
  grid.sync();

  if (!remote_backend_enabled<StaticLocal>(a.use_socket)) {
    // DYDispatchPut: dY and its route-local source gate share the exact
    // Forward canonical row mapping. Socket mode preloads both through its
    // host-enqueued aggregate phase before this compute-only launch.
    unsigned long long dy_peer_rows = 0ULL;
#if MEGAMOE_DEVICE_TIMELINE
    bool timeline_dy_local_recorded = false;
#endif
    for (std::uint64_t route = blockIdx.x;
         route < a.dispatch.sizes.route_count; route += gridDim.x) {
    const auto destination = a.dispatch.route_dst_rank[route];
    if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
        destination / a.local_world_size != a.node)
      continue;
    const auto row = a.dispatch.route_dst_row[route];
    const auto token = route / a.dispatch.identity.top_k;
    auto *destination_base = a.peer_bases[destination];
    auto *destination_dy = peer_at<transport::BFloat16>(
        destination_base, a.dy_canonical_offset);
    auto *destination_gate = peer_at<float>(destination_base,
                                             a.canonical_gate_offset);
    const auto *source128 = reinterpret_cast<const uint4 *>(
        a.dy_input + token * a.hidden);
    auto *target128 = reinterpret_cast<uint4 *>(destination_dy + row * a.hidden);
    for (std::uint64_t vector = threadIdx.x; vector < a.hidden / 8ULL;
         vector += blockDim.x)
      target128[vector] = source128[vector];
    if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        auto *arrived = peer_at<unsigned long long>(
            destination_base, a.dy_arrived_rows_offset);
        ::cuda::atomic_ref<unsigned long long,
                           ::cuda::thread_scope_system>
            counter(arrived[a.dispatch_waves.route_dst_mtile[route]]);
        counter.fetch_add(1ULL, ::cuda::memory_order_release);
      }
      __syncthreads();
    }
#if MEGAMOE_DEVICE_TIMELINE
    __syncthreads();
    if (!timeline_dy_local_recorded) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, DyFirstLocalRowVisible, GinDispatch,
                              0U, route, a.epoch);
      timeline_dy_local_recorded = true;
    }
#endif
    if (threadIdx.x == 0) {
      destination_gate[row] = a.gates[route];
      if (destination != a.rank)
        ++dy_peer_rows;
    }
    }
    if (threadIdx.x == 0 && dy_peer_rows != 0ULL)
      atomicAdd(&status->actual_dy_peer_bytes,
                dy_peer_rows * a.hidden * sizeof(transport::BFloat16));
#if __has_include(<nccl_device.h>)
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    grid.sync();
    const auto first_local = a.node * a.local_world_size;
    if (blockIdx.x == 0U) {
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        auto *ready = peer_at<unsigned long long>(
            a.peer_bases[peer], a.gin_local_compute_ready_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(ready[a.rank]);
        published.store(a.epoch, ::cuda::memory_order_release);
      }
      __syncthreads();
      auto *local_ready = peer_at<unsigned long long>(
          a.local_base, a.gin_local_compute_ready_offset);
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(local_ready[peer]);
        while (observed.load(::cuda::memory_order_acquire) < a.epoch &&
               !backward_cancelled(a)) {
        }
      }
    }
    grid.sync();
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        bytes + a.gin_send_header_offset);
    auto *send_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        bytes + a.gin_send_fanout_offset);
    auto *send_gate = reinterpret_cast<float *>(bytes + a.gin_send_gate_offset);
    auto *send_payload = reinterpret_cast<transport::BFloat16 *>(
        bytes + a.gin_send_payload_offset);
    const auto vector_count = a.hidden / 8ULL;
    const auto global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto global_threads =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t item = global_thread;
         item < a.dispatch.sizes.packet_count * vector_count;
         item += global_threads) {
      const auto wave_position = item / vector_count;
      const auto vector = item % vector_count;
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      reinterpret_cast<uint4 *>(send_payload)[wave_position * vector_count +
                                               vector] =
          reinterpret_cast<const uint4 *>(a.dy_input)[
              descriptor.local_token_id * vector_count + vector];
    }
    const auto fanout_slots =
        a.dispatch.sizes.packet_count * a.dispatch.identity.top_k;
    for (std::uint64_t packed = global_thread; packed < fanout_slots;
         packed += global_threads) {
      const auto wave_position = packed / a.dispatch.identity.top_k;
      const auto fanout_slot = packed % a.dispatch.identity.top_k;
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (fanout_begin + fanout_slot < fanout_end) {
        const auto target = a.dispatch.packet_fanout[fanout_begin + fanout_slot];
        send_fanout[packed] = target;
        send_gate[packed] = a.gates[descriptor.local_token_id *
                                        a.dispatch.identity.top_k +
                                    target.topk_slot];
      }
    }
    for (std::uint64_t wave_position = global_thread;
         wave_position < a.dispatch.sizes.packet_count;
         wave_position += global_threads) {
      const auto packet = a.dispatch_waves.packet_order[wave_position];
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      send_headers[wave_position] = LocalP2pRouteHeader{
          transport::kActivationTransportSchemaVersion, a.rank,
          descriptor.proxy_rank, descriptor.channel,
          a.dispatch.identity.generation, a.epoch,
          wave_position, fanout_end - fanout_begin,
          descriptor.local_token_id,
          wave_position * a.dispatch.identity.top_k, 0, 0};
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, DyPacketReady, GinDispatch, 0U,
                            a.dispatch.sizes.packet_count, a.epoch);
    __threadfence_system();
    grid.sync();
  }
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count) {
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    const ncclGin gin(a.gin_dev_comm, context);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    auto *window_bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *send_directory = reinterpret_cast<GinDispatchContextRecord *>(
        window_bytes + a.gin_send_dispatch_directory_offset);
    auto *send_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        window_bytes + a.gin_send_header_offset);
    unsigned long long issued_completions = 0ULL;
    unsigned long long issued_packets = 0ULL;
    for (std::uint32_t remote_node = 0;
         remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
      if (remote_node == a.node)
        continue;
      std::uint64_t node_wave_begin = 0;
      while (node_wave_begin < a.dispatch_waves.wave_count &&
             a.dispatch_waves.destination_node[node_wave_begin] < remote_node)
        ++node_wave_begin;
      auto node_wave_end = node_wave_begin;
      while (node_wave_end < a.dispatch_waves.wave_count &&
             a.dispatch_waves.destination_node[node_wave_end] == remote_node)
        ++node_wave_end;
      const auto node_waves = node_wave_end - node_wave_begin;
      const auto context_wave_begin =
          node_wave_begin + node_waves * context / a.gin_context_count;
      const auto context_wave_end =
          node_wave_begin + node_waves * (context + 1U) /
                                a.gin_context_count;
      const auto node_packet_begin =
          a.dispatch_waves.offsets[node_wave_begin];
      const auto packet_begin =
          a.dispatch_waves.offsets[context_wave_begin] - node_packet_begin;
      const auto packet_end =
          a.dispatch_waves.offsets[context_wave_end] - node_packet_begin;
      const auto remote_partition =
          remote_node < a.node ? remote_node : remote_node - 1U;
      const auto source_partition =
          a.node < remote_node ? a.node : a.node - 1U;
      const auto directory_index =
          remote_partition * a.gin_context_count + context;
      if (threadIdx.x == 0U) {
        send_directory[directory_index] = GinDispatchContextRecord{
            a.dispatch.identity.generation, a.epoch, packet_begin,
            static_cast<std::uint32_t>(packet_end - packet_begin),
            static_cast<std::uint32_t>(context_wave_end - context_wave_begin)};
        __threadfence_system();
        const auto remote_directory_index =
            source_partition * a.gin_context_count + context;
        gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                a.gin_receive_dispatch_directory_offset +
                    remote_directory_index * sizeof(GinDispatchContextRecord),
                a.gin_window, a.gin_send_dispatch_directory_offset +
                                  directory_index *
                                      sizeof(GinDispatchContextRecord),
                sizeof(GinDispatchContextRecord), ncclGin_SignalInc{0},
                ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        ++issued_completions;
      }
      __syncthreads();
      for (auto wave = context_wave_begin; wave < context_wave_end; ++wave) {
        const auto wave_begin = a.dispatch_waves.offsets[wave];
        const auto wave_end = a.dispatch_waves.offsets[wave + 1U];
        const auto wire_begin = wave_begin - node_packet_begin;
        const auto wave_packets = wave_end - wave_begin;
        for (auto position = wave_begin + threadIdx.x; position < wave_end;
             position += blockDim.x) {
          send_headers[position].sequence = position - node_packet_begin;
          send_headers[position].dst_row =
              (position - node_packet_begin) * a.dispatch.identity.top_k;
          send_headers[position].topk_slot = 0U;
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
          send_headers[wave_begin].topk_slot =
              static_cast<std::uint32_t>(wave_packets);
          __threadfence_system();
          const auto receive_packet =
              source_partition * a.max_tokens_per_rank + wire_begin;
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, DyGinPutIssue, GinDispatch, context,
              wave_packets * a.hidden * sizeof(transport::BFloat16), a.epoch);
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_header_offset +
                      receive_packet * sizeof(LocalP2pRouteHeader),
                  a.gin_window, a.gin_send_header_offset +
                                    wave_begin * sizeof(LocalP2pRouteHeader),
                  wave_packets * sizeof(LocalP2pRouteHeader));
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_fanout_offset +
                      (source_partition * a.max_tokens_per_rank + wire_begin) *
                          a.dispatch.identity.top_k *
                          sizeof(DeviceFanoutRecord),
                  a.gin_window, a.gin_send_fanout_offset +
                                    wave_begin * a.dispatch.identity.top_k *
                                        sizeof(DeviceFanoutRecord),
                  wave_packets * a.dispatch.identity.top_k *
                      sizeof(DeviceFanoutRecord));
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_gate_offset +
                      (source_partition * a.max_tokens_per_rank + wire_begin) *
                          a.dispatch.identity.top_k * sizeof(float),
                  a.gin_window, a.gin_send_gate_offset +
                                    wave_begin * a.dispatch.identity.top_k *
                                        sizeof(float),
                  wave_packets * a.dispatch.identity.top_k * sizeof(float));
          gin.put(rail, static_cast<int>(remote_node), a.gin_window,
                  a.gin_receive_payload_offset +
                      receive_packet * a.hidden *
                          sizeof(transport::BFloat16),
                  a.gin_window, a.gin_send_payload_offset +
                                    wave_begin * a.hidden *
                                        sizeof(transport::BFloat16),
                  wave_packets * a.hidden * sizeof(transport::BFloat16),
                  ncclGin_SignalInc{0}, ncclGin_CounterInc{0},
                  ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          ++issued_completions;
          issued_packets += wave_packets;
          atomicAdd(&status->gin_dispatch_put_count, 4ULL);
          atomicAdd(&status->gin_signal_count, 1ULL);
          atomicAdd(&status->gin_metadata_bytes,
                    wave_packets *
                        (sizeof(LocalP2pRouteHeader) +
                         a.dispatch.identity.top_k *
                             (sizeof(DeviceFanoutRecord) + sizeof(float))));
        }
        __syncthreads();
      }
    }
    if (threadIdx.x == 0U && issued_completions != 0ULL) {
      gin.waitCounter(ncclCoopThread{}, 0, issued_completions);
      MEGAMOE_TIMELINE_RECORD(a.timeline, DyLocalCompletion, GinDispatch,
                              context, issued_packets, a.epoch);
      atomicAdd(&status->raw_cross_node_dy_bytes,
                issued_packets * a.dispatch.identity.top_k * a.hidden *
                    sizeof(transport::BFloat16));
      atomicAdd(&status->gin_dispatch_payload_bytes,
                issued_packets * a.hidden * sizeof(transport::BFloat16));
      gin.resetCounter(0);
      for (std::uint32_t remote_node = 0;
           remote_node < a.dispatch.identity.num_nodes; ++remote_node) {
        if (remote_node == a.node)
          continue;
        const auto remote_partition =
            remote_node < a.node ? remote_node : remote_node - 1U;
        const auto record_index =
            remote_partition * a.gin_context_count + context;
        const auto directory = send_directory[record_index];
        auto &record = context_completion[record_index];
        record.directory_ready = 1ULL;
        record.dispatch_expected = directory.wave_count;
        record.dispatch_claimed = directory.wave_count;
        record.dispatch_completed = directory.wave_count;
        record.local_completion = 1ULL + directory.wave_count;
        record.credit = 1ULL;
        record.last_cursor = directory.packet_begin + directory.packet_count;
        record.last_progress_globaltimer = device_globaltimer_ns();
        atomicAdd(&completion->directory_ready, 1ULL);
        atomicAdd(&completion->dispatch_local_completion_completed,
                  1ULL + directory.wave_count);
        if (directory.wave_count != 0ULL) {
          claim_completion_tasks(completion,
                                 transport_state::TaskClass::DispatchWave,
                                 directory.wave_count);
          complete_completion_tasks(completion,
                                    transport_state::TaskClass::DispatchWave,
                                    directory.wave_count);
        }
      }
    }
    __syncthreads();
    unsigned long long incoming_sources = 0ULL;
    if (threadIdx.x == 0U) {
      for (std::uint32_t index = 0;
           index < a.dispatch.sizes.incoming_signal_count; ++index) {
        const auto signal = a.dispatch.incoming_proxy_signals[index];
        if (signal.source_node != a.node && signal.expected_packets != 0ULL)
          ++incoming_sources;
      }
      if (incoming_sources != 0ULL)
        gin.waitSignal(ncclCoopThread{}, 0, incoming_sources);
      auto *receive_directory =
          reinterpret_cast<GinDispatchContextRecord *>(
              static_cast<std::byte *>(a.gin_window_base) +
              a.gin_receive_dispatch_directory_offset);
      unsigned long long incoming_waves = 0ULL;
      for (std::uint32_t index = 0;
           index < a.dispatch.sizes.incoming_signal_count; ++index) {
        const auto signal = a.dispatch.incoming_proxy_signals[index];
        if (signal.source_node == a.node || signal.expected_packets == 0ULL)
          continue;
        const auto source_partition =
            signal.source_node < a.node ? signal.source_node
                                        : signal.source_node - 1U;
        const auto directory = receive_directory[
            source_partition * a.gin_context_count + context];
        if (directory.generation == a.dispatch.identity.generation &&
            directory.epoch == a.epoch) {
          incoming_waves += directory.wave_count;
          const auto record_index =
              source_partition * a.gin_context_count + context;
          context_completion[record_index].relay_expected =
              directory.packet_count;
        }
      }
      if (incoming_waves != 0ULL)
        gin.waitSignal(ncclCoopThread{}, 0,
                       incoming_sources + incoming_waves);
      MEGAMOE_TIMELINE_RECORD(a.timeline, RemoteSignalObserved, GinDispatch,
                              context, incoming_sources, a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, DyFirstRemoteSignal, GinDispatch,
                              context, incoming_sources, a.epoch);
    }
    __syncthreads();
  }
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count) {
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        bytes + a.gin_receive_header_offset);
    auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        bytes + a.gin_receive_fanout_offset);
    auto *receive_gate = reinterpret_cast<float *>(
        bytes + a.gin_receive_gate_offset);
    auto *receive_payload = reinterpret_cast<transport::BFloat16 *>(
        bytes + a.gin_receive_payload_offset);
    auto *receive_directory = reinterpret_cast<GinDispatchContextRecord *>(
        bytes + a.gin_receive_dispatch_directory_offset);
    MEGAMOE_TIMELINE_RECORD(a.timeline, DyRelayStart, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_signal_count, a.epoch);
    const auto vector_count = a.hidden / 8ULL;
    const auto context_thread = static_cast<std::uint64_t>(threadIdx.x);
    const auto context_threads = static_cast<std::uint64_t>(blockDim.x);
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto directory = receive_directory[
          source_partition * a.gin_context_count + context];
      const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
      for (std::uint64_t item = context_thread;
           item < static_cast<std::uint64_t>(directory.packet_count) *
                      packet_vectors;
           item += context_threads) {
        const auto sequence =
            directory.packet_begin + item / packet_vectors;
        const auto within_packet = item % packet_vectors;
        const auto fanout = within_packet / vector_count;
        const auto vector = within_packet % vector_count;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *destination_dy = peer_at<transport::BFloat16>(
            a.peer_bases[destination], a.dy_canonical_offset);
        reinterpret_cast<uint4 *>(destination_dy)[
            target.dst_row * vector_count + vector] =
            reinterpret_cast<const uint4 *>(receive_payload)[
                packet * vector_count + vector];
      }
      for (std::uint64_t slot = context_thread;
           slot < static_cast<std::uint64_t>(directory.packet_count) *
                      a.dispatch.identity.top_k;
           slot += context_threads) {
        const auto sequence =
            directory.packet_begin + slot / a.dispatch.identity.top_k;
        const auto fanout = slot % a.dispatch.identity.top_k;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *destination_gate = peer_at<float>(
            a.peer_bases[destination], a.canonical_gate_offset);
        destination_gate[target.dst_row] = receive_gate[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
      }
    }
    __threadfence_system();
    __syncthreads();
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto directory = receive_directory[
          source_partition * a.gin_context_count + context];
      for (std::uint64_t slot = context_thread;
           slot < static_cast<std::uint64_t>(directory.packet_count) *
                      a.dispatch.identity.top_k;
           slot += context_threads) {
        const auto sequence =
            directory.packet_begin + slot / a.dispatch.identity.top_k;
        const auto fanout = slot % a.dispatch.identity.top_k;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        auto *arrived = peer_at<unsigned long long>(
            a.peer_bases[destination], a.dy_arrived_rows_offset);
        ::cuda::atomic_ref<unsigned long long,
                           ::cuda::thread_scope_system>
            counter(arrived[target.reserved]);
        counter.fetch_add(1ULL, ::cuda::memory_order_release);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
      unsigned long long relayed = 0ULL;
      for (std::uint32_t remote = 0;
           remote < a.dispatch.identity.num_nodes - 1U; ++remote) {
        const auto index = remote * a.gin_context_count + context;
        auto &record = context_completion[index];
        record.relay_claimed = record.relay_expected;
        record.relay_completed = record.relay_expected;
        record.last_progress_globaltimer = device_globaltimer_ns();
        relayed += record.relay_expected;
      }
      if (relayed != 0ULL) {
        claim_completion_tasks(completion,
                               transport_state::TaskClass::RelayWave,
                               relayed);
        complete_completion_tasks(completion,
                                  transport_state::TaskClass::RelayWave,
                                  relayed);
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, DyRelayEnd, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, DyLastRelay, ProxyRelay, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
  }
#endif
  __threadfence_system();
  if (!remote_backend_enabled<StaticLocal>(a.use_gin)) {
    grid.sync();
    if (blockIdx.x == 0) {
    const auto first_local = remote_backend_enabled<StaticLocal>(a.use_gin)
                                 ? a.node * a.local_world_size
                                 : 0U;
    const auto local_count = remote_backend_enabled<StaticLocal>(a.use_gin)
                                 ? a.local_world_size
                                 : a.world;
    if (threadIdx.x < local_count) {
      const auto peer = first_local + threadIdx.x;
      auto *signal = peer_at<unsigned long long>(
          a.peer_bases[peer], a.incoming_signal_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          signal[a.rank * a.channels]);
      ready.store(a.epoch, ::cuda::memory_order_release);
    }
    auto *signal = peer_at<unsigned long long>(a.local_base,
                                                a.incoming_signal_offset);
    for (std::uint32_t local_source = threadIdx.x;
         local_source < local_count; local_source += blockDim.x) {
      const auto source = first_local + local_source;
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          signal[source * a.channels]);
      while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
             !backward_cancelled(a)) {
      }
    }
    }
    grid.sync();
  }
  } else {
    grid.sync();
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dy_dispatch_complete_clock = device_globaltimer_ns();
  MEGAMOE_TIMELINE_RECORD(a.timeline, DyDispatchComplete, Compute, 0U,
                          a.dispatch.sizes.mtile_count, a.epoch);
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDyComplete, a.epoch);

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < 4U; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  const auto warpgroup = __shfl_sync(0xffffffffU, threadIdx.x / 128U, 0);
  if (warpgroup < 2U)
    cutlass::arch::warpgroup_reg_alloc<224>();
  else
    cutlass::arch::warpgroup_reg_dealloc<48>();
  std::uint32_t pipeline_iteration = 0;
  __shared__ unsigned long long task_shared;
  const auto mblocks = scheduler[8];
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  const auto dv_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 255ULL) / 256ULL);

  // NO_FC1_OUTPUT_CACHE recomputes A/B through the exact Forward WGMMA path.
  if (!a.use_forward_cache) {
    const auto tasks = mblocks * fc1_ntiles;
    while (true) {
      if (threadIdx.x == 0)
        task_shared = atomicAdd(&scheduler[0], 1ULL);
      __syncthreads();
      if (task_shared >= tasks)
        break;
      const auto mblock = task_shared / fc1_ntiles;
      const auto ntile = static_cast<std::uint32_t>(task_shared % fc1_ntiles);
      const auto mapping =
          claims[a.dispatch.sizes.owned_expert_count + mblock];
      const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
      const auto local_mblock = static_cast<std::uint32_t>(mapping);
      const auto row = a.dispatch.expert_row_prefix[expert] +
                       static_cast<std::uint64_t>(local_mblock) * 128ULL;
      const auto valid_rows = static_cast<std::uint32_t>(min(
          128ULL, a.dispatch.expert_row_count[expert] -
                      static_cast<std::uint64_t>(local_mblock) * 128ULL));
      const auto n = ntile * 128U;
      float accum_a[ForwardMma::kNumAccum];
      float accum_b[ForwardMma::kNumAccum];
      forward_wgmma_swiglu_tile(
          x_map, w1_map, static_cast<std::uint32_t>(row),
          static_cast<std::uint32_t>(expert * 2ULL * a.intermediate + n),
          static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                     a.intermediate + n),
          static_cast<std::uint32_t>(a.hidden), smem_a, smem_b,
          full_barriers, empty_barriers, accum_a, accum_b,
          pipeline_iteration);
      visit_forward_accumulator_pair(
          accum_a, accum_b,
          [&](std::uint32_t wg, std::uint32_t m, std::uint32_t column,
              float av0, float av1, float bv0, float bv1) {
        const auto block_m = wg * 64U + m;
        const float av[2] = {av0, av1};
        const float bv[2] = {bv0, bv1};
#pragma unroll
        for (std::uint32_t value = 0; value < 2U; ++value) {
          const auto output_column = static_cast<std::uint64_t>(n) + column + value;
          if (block_m < valid_rows && output_column < a.intermediate) {
            const auto index = (row + block_m) * a.intermediate + output_column;
            ab[index] = transport::BFloat16{float_to_bf16_bits(av[value])};
            ab[a.max_destination_rows * a.intermediate + index] =
                transport::BFloat16{float_to_bf16_bits(bv[value])};
          }
        }
      });
    }
    grid.sync();
    if (threadIdx.x == 256U) {
      for (std::uint32_t stage = 0; stage < 4U; ++stage) {
        full_barriers[stage].init(1);
        empty_barriers[stage].init(8);
      }
      cutlass::arch::fence_barrier_init();
    }
    __syncthreads();
    pipeline_iteration = 0;
  }

  // dV plus U and fused SwiGLU backward. Every output tile has one owner;
  // neither gradients nor dGate use floating-point atomics.
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < 3U; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  pipeline_iteration = 0;
  const auto dv_tasks = mblocks * dv_ntiles;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dv_start_clock = device_globaltimer_ns();
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDvStarted, dv_tasks);
#if MEGAMOE_DEVICE_TIMELINE
  bool timeline_dv_started = false;
#endif
  while (true) {
    if (threadIdx.x == 0)
      task_shared = atomicAdd(&scheduler[1], 1ULL);
    __syncthreads();
    if (task_shared >= dv_tasks)
      break;
    const auto mblock = task_shared / dv_ntiles;
    const auto ntile = static_cast<std::uint32_t>(task_shared % dv_ntiles);
    const auto mapping =
        claims[a.dispatch.sizes.owned_expert_count + mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    const auto row = a.dispatch.expert_row_prefix[expert] +
                     static_cast<std::uint64_t>(local_mblock) * 128ULL;
    const auto valid_rows = static_cast<std::uint32_t>(min(
        128ULL, a.dispatch.expert_row_count[expert] -
                    static_cast<std::uint64_t>(local_mblock) * 128ULL));
    wait_backward_mblock_rows<StaticLocal>(
        a, expert, static_cast<std::uint64_t>(local_mblock) * 128ULL,
        valid_rows);
    __syncthreads();
#if MEGAMOE_DEVICE_TIMELINE
    if (!timeline_dv_started) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, DyFirstMtileReady, Compute, 0U,
                              task_shared, a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, DvFirst, Compute, 0U, task_shared,
                              a.epoch);
      MEGAMOE_TIMELINE_RECORD(a.timeline, DvClaimStart, Compute, 0U,
                              task_shared, a.epoch);
      timeline_dv_started = true;
    }
#endif
    const auto n = ntile * 256U;
    float accum_a[ForwardMma::kNumAccum];
    float accum_b[ForwardMma::kNumAccum];
    forward_wgmma_swiglu_tile(
        dy_map, w2_transpose_map, static_cast<std::uint32_t>(row),
        static_cast<std::uint32_t>(expert * a.intermediate + n),
        static_cast<std::uint32_t>(expert * a.intermediate + n + 128U),
        static_cast<std::uint32_t>(a.hidden), smem_a, smem_b,
        full_barriers, empty_barriers, accum_a, accum_b, pipeline_iteration);
    const auto padded_row_base = claims[expert] +
                                 static_cast<std::uint64_t>(local_mblock) *
                                     128ULL;
    constexpr std::uint32_t kEpilogueTileElements = 128U * 128U;
    auto *epilogue_u = smem_a;
    auto *epilogue_da = epilogue_u + kEpilogueTileElements;
    auto *epilogue_db = epilogue_da + kEpilogueTileElements;
    auto *epilogue_dv = epilogue_db + kEpilogueTileElements;
    auto *epilogue_s = epilogue_dv + kEpilogueTileElements;
    const auto expert_padded_rows =
        (a.dispatch.expert_row_count[expert] + 63ULL) & ~63ULL;
    const auto padded_block_rows = static_cast<std::uint32_t>(min(
        128ULL, expert_padded_rows -
                    static_cast<std::uint64_t>(local_mblock) * 128ULL));
#pragma unroll
    for (std::uint32_t half = 0; half < 2U; ++half) {
      auto *half_accum = half == 0U ? accum_a : accum_b;
      visit_forward_accumulator(
        half_accum, [&](std::uint32_t wg, std::uint32_t m,
                   std::uint32_t column, float value0, float value1) {
      const auto block_m = wg * 64U + m;
      const float values[2] = {value0, value1};
#pragma unroll
      for (std::uint32_t value = 0; value < 2U; ++value) {
        const auto output_column = static_cast<std::uint64_t>(n) +
                                   half * 128ULL + column + value;
        if (block_m < valid_rows && output_column < a.intermediate) {
          const auto index = (row + block_m) * a.intermediate + output_column;
          const auto dv_bf16 = float_to_bf16_bits(values[value]);
          const auto av = bf16_bits_to_float(ab[index].bits);
          const auto bv = bf16_bits_to_float(
              ab[a.max_destination_rows * a.intermediate + index].bits);
          const auto sigmoid = 1.0F / (1.0F + __expf(-av));
          const auto silu = av * sigmoid;
          const auto s_value = silu * bv;
          const auto s_bf16 = float_to_bf16_bits(s_value);
          const auto gate = canonical_gate[row + block_m];
          const auto u_bf16 = float_to_bf16_bits(
              __fmul_rn(gate, bf16_bits_to_float(s_bf16)));
          const auto ds = __fmul_rn(gate, bf16_bits_to_float(dv_bf16));
          const auto da = __fmul_rn(
              ds, __fmul_rn(bv, sigmoid * (1.0F + av * (1.0F - sigmoid))));
          const auto db = __fmul_rn(ds, silu);
          const auto epilogue_index = block_m * 128ULL + column + value;
          epilogue_u[epilogue_index] = transport::BFloat16{u_bf16};
          epilogue_da[epilogue_index] =
              transport::BFloat16{float_to_bf16_bits(da)};
          epilogue_db[epilogue_index] =
              transport::BFloat16{float_to_bf16_bits(db)};
          epilogue_dv[epilogue_index] = transport::BFloat16{dv_bf16};
          epilogue_s[epilogue_index] = transport::BFloat16{s_bf16};
#if MEGAMOE_DEBUG_VALIDATE
          if (a.capture_stage_storage)
            debug_dv[(row + block_m) * a.intermediate + output_column] =
                transport::BFloat16{dv_bf16};
#endif
        }
      }
      });
      __syncthreads();
      const auto epilogue_warp = threadIdx.x / 32U;
      const auto epilogue_lane = threadIdx.x & 31U;
      for (std::uint32_t local_row = epilogue_warp;
           local_row < valid_rows; local_row += blockDim.x / 32U) {
        float partial = 0.0F;
#pragma unroll
        for (std::uint32_t column = epilogue_lane; column < 128U;
             column += 32U) {
          const auto index = local_row * 128U + column;
          partial = __fadd_rn(
              partial,
              __fmul_rn(bf16_bits_to_float(epilogue_dv[index].bits),
                        bf16_bits_to_float(epilogue_s[index].bits)));
        }
        partial = __fadd_rn(partial,
                            __shfl_down_sync(0xffffffffU, partial, 16));
        partial = __fadd_rn(partial,
                            __shfl_down_sync(0xffffffffU, partial, 8));
        partial = __fadd_rn(partial,
                            __shfl_down_sync(0xffffffffU, partial, 4));
        partial = __fadd_rn(partial,
                            __shfl_down_sync(0xffffffffU, partial, 2));
        partial = __fadd_rn(partial,
                            __shfl_down_sync(0xffffffffU, partial, 1));
        if (epilogue_lane == 0U) {
          auto &target = dgate_partial[(row + local_row) * dv_ntiles + ntile];
          target = half == 0U ? partial : __fadd_rn(target, partial);
        }
      }
      __syncthreads();
      constexpr std::uint32_t kVectorsPerEpilogueRow = 128U / 8U;
      constexpr std::uint32_t kEpilogueVectors =
          128U * kVectorsPerEpilogueRow;
      for (std::uint32_t vector = threadIdx.x; vector < kEpilogueVectors;
           vector += blockDim.x) {
        const auto local_row = vector / kVectorsPerEpilogueRow;
        const auto local_vector = vector % kVectorsPerEpilogueRow;
        const auto local_column = local_vector * 8U;
        const auto column = static_cast<std::uint64_t>(n) + half * 128ULL +
                            local_column;
        if (column + 7ULL < a.intermediate &&
            local_row < padded_block_rows) {
          uint4 value_da{};
          uint4 value_db{};
          if (local_row < valid_rows) {
            value_da = reinterpret_cast<const uint4 *>(
                epilogue_da + local_row * 128U)[local_vector];
            value_db = reinterpret_cast<const uint4 *>(
                epilogue_db + local_row * 128U)[local_vector];
            reinterpret_cast<uint4 *>(
                dab + (row + local_row) * (2ULL * a.intermediate) +
                column)[0] = value_da;
            reinterpret_cast<uint4 *>(
                dab + (row + local_row) * (2ULL * a.intermediate) +
                a.intermediate + column)[0] = value_db;
          }
        }
      }
      __syncthreads();
      // Materialize the dW K-major operands directly from the register
      // epilogue tile.  Threads adjacent in the linear loop write adjacent
      // padded-M locations, avoiding a later multi-gigabyte global transpose
      // for U and dAB.
      constexpr std::uint32_t kPaddedRowVectors = 128U / 8U;
      for (std::uint32_t index = threadIdx.x;
           index < 128U * kPaddedRowVectors;
           index += blockDim.x) {
        const auto feature = index / kPaddedRowVectors;
        const auto local_row = (index % kPaddedRowVectors) * 8U;
        const auto output_feature =
            static_cast<std::uint64_t>(n) + half * 128ULL + feature;
        if (local_row + 7U < padded_block_rows &&
            output_feature < a.intermediate) {
          const auto padded_row = padded_row_base + local_row;
          unsigned int words_u[4];
          unsigned int words_da[4];
          unsigned int words_db[4];
#pragma unroll
          for (std::uint32_t word = 0; word < 4U; ++word) {
            const auto row0 = local_row + word * 2U;
            const auto row1 = row0 + 1U;
            const auto read = [&](const transport::BFloat16 *tile,
                                  std::uint32_t row) {
              return row < valid_rows ? tile[row * 128U + feature].bits
                                      : std::uint16_t{0};
            };
            words_u[word] = static_cast<unsigned int>(read(epilogue_u, row0)) |
                            (static_cast<unsigned int>(
                                 read(epilogue_u, row1)) << 16U);
            words_da[word] =
                static_cast<unsigned int>(read(epilogue_da, row0)) |
                (static_cast<unsigned int>(read(epilogue_da, row1)) << 16U);
            words_db[word] =
                static_cast<unsigned int>(read(epilogue_db, row0)) |
                (static_cast<unsigned int>(read(epilogue_db, row1)) << 16U);
          }
          reinterpret_cast<uint4 *>(
              u_t + output_feature * a.reduction_rows + padded_row)[0] =
              make_uint4(words_u[0], words_u[1], words_u[2], words_u[3]);
          reinterpret_cast<uint4 *>(
              dab_t + output_feature * a.reduction_rows + padded_row)[0] =
              make_uint4(words_da[0], words_da[1], words_da[2], words_da[3]);
          reinterpret_cast<uint4 *>(
              dab_t + (a.intermediate + output_feature) * a.reduction_rows +
              padded_row)[0] =
              make_uint4(words_db[0], words_db[1], words_db[2], words_db[3]);
        }
      }
  __syncthreads();
}

    __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0) {
      const auto expected =
          ((a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL) *
          dv_ntiles;
      const auto complete =
          atomicAdd(&dw_timeline[expert].dv_tasks_complete, 1ULL) + 1ULL;
      if (complete == expected) {
        const auto now = device_globaltimer_ns();
        dw_timeline[expert].ready_clock = now;
        atomicMin(&status->dw1_first_ready_clock, now);
        atomicMin(&status->dw2_first_ready_clock, now);
      }
    }
#endif
    __syncthreads();
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dv_complete_clock = device_globaltimer_ns();
  MEGAMOE_TIMELINE_RECORD(a.timeline, DvLast, Compute, 0U, dv_tasks,
                          a.epoch);
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDvComplete, dv_tasks);

  // The owner dV CTA wrote each fixed I-tile partial above. An ordered tile
  // reduction now produces one dGate per canonical route.
  const auto incoming_rows = a.dispatch.sizes.incoming_row_count;
  for (std::uint64_t row =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       row < incoming_rows;
       row += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    float sum = 0.0F;
    for (std::uint32_t tile = 0; tile < dv_ntiles; ++tile)
      sum = __fadd_rn(sum, dgate_partial[row * dv_ntiles + tile]);
    dgate_canonical[row] = sum;
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dgate_complete_clock = device_globaltimer_ns();
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDgateComplete, incoming_rows);

  // dY and X remain in canonical row-major storage.  The dW WGMMA path below
  // consumes them directly as MN-major operands, so this point is now only a
  // timestamp and contains no bulk transpose/staging work.
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->transpose_complete_clock = device_globaltimer_ns();

  // dX: one CTA owns each canonical M/N tile.
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < 4U; ++stage) {
      full_barriers[stage].init(1);
      empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  pipeline_iteration = 0;
  const auto dx_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto dx_tasks = mblocks * dx_ntiles;
  // dW2 and dW1 retain their L2-friendly contiguous task domains. Scheduler
  // candidates may overlap a prefix with dX, but do not interleave operand
  // families because that destroys canonical-MN B reuse.
  const auto dw2_mtiles = (a.hidden + 127ULL) / 128ULL;
  constexpr std::uint64_t kDwN = 256ULL;
  const auto dw2_ntiles = (a.intermediate + kDwN - 1ULL) / kDwN;
  const auto dw1_mtiles = (2ULL * a.intermediate + 127ULL) / 128ULL;
  const auto dw1_ntiles = (a.hidden + kDwN - 1ULL) / kDwN;
  const auto dw2_tasks = static_cast<unsigned long long>(
      a.dispatch.sizes.owned_expert_count) * dw2_mtiles * dw2_ntiles;
  const auto dw1_tasks = static_cast<unsigned long long>(
      a.dispatch.sizes.owned_expert_count) * dw1_mtiles * dw1_ntiles;
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dx_start_clock = device_globaltimer_ns();
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDxStarted, dx_tasks);
  // Main-shape dX tasks have uniform K and a unique output tile owner. A
  // deterministic grid-stride assignment removes the global claim atomic
  // while retaining enough cross-expert parallelism to fill every SM.
  if (threadIdx.x == 0U)
    task_shared = blockIdx.x;
  __syncthreads();
  while (true) {
    __syncthreads();
    if (task_shared >= dx_tasks)
      break;
    const auto mblock = task_shared / dx_ntiles;
    const auto ntile = static_cast<std::uint32_t>(task_shared % dx_ntiles);
    const auto mapping =
        claims[a.dispatch.sizes.owned_expert_count + mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
    const auto row = a.dispatch.expert_row_prefix[expert] +
                     static_cast<std::uint64_t>(local_mblock) * 128ULL;
    const auto valid_rows = static_cast<std::uint32_t>(min(
        128ULL, a.dispatch.expert_row_count[expert] -
                    static_cast<std::uint64_t>(local_mblock) * 128ULL));
    const auto n = ntile * 256U;
    float accum[ForwardMmaWide::kNumAccum];
    forward_wgmma_wide_tile(
        dab_map, w1_transpose_map, static_cast<std::uint32_t>(row),
        static_cast<std::uint32_t>(expert * a.hidden + n),
        static_cast<std::uint32_t>(2ULL * a.intermediate), smem_a, smem_b,
        full_barriers, empty_barriers, accum, pipeline_iteration);
    auto *output_tile = smem_a;
    visit_forward_accumulator_wide(
        accum, [&](std::uint32_t wg, std::uint32_t m,
                   std::uint32_t column, float v0, float v1) {
      const auto block_m = wg * 64U + m;
      output_tile[block_m * 256U + column] =
          transport::BFloat16{float_to_bf16_bits(v0)};
      output_tile[block_m * 256U + column + 1U] =
          transport::BFloat16{float_to_bf16_bits(v1)};
    });
    __syncthreads();
    constexpr std::uint32_t kVectorsPerRow = 256U / 8U;
    constexpr std::uint32_t kOutputVectors = 128U * kVectorsPerRow;
    for (std::uint32_t vector = threadIdx.x; vector < kOutputVectors;
         vector += blockDim.x) {
      const auto local_row = vector / kVectorsPerRow;
      const auto local_vector = vector % kVectorsPerRow;
      const auto output_column =
          static_cast<std::uint64_t>(n) + local_vector * 8ULL;
      if (local_row < valid_rows && output_column + 7ULL < a.hidden) {
        const auto packed = reinterpret_cast<const uint4 *>(
            output_tile + local_row * 256ULL)[local_vector];
        const auto header = headers[row + local_row];
        reinterpret_cast<uint4 *>(
            dx_canonical + (row + local_row) * a.hidden +
            output_column)[0] = packed;
        if (remote_backend_enabled<StaticLocal>(a.use_socket))
          continue;
        if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
            header.source_rank / a.local_world_size != a.node)
          continue;
        auto *source_dx = peer_at<transport::BFloat16>(
            a.peer_bases[header.source_rank], a.dx_return_offset);
        reinterpret_cast<uint4 *>(
            source_dx + header.route_index * a.hidden + output_column)[0] =
            packed;
        if (ntile == 0U && local_vector == 0U) {
          auto *source_dgate = peer_at<float>(
              a.peer_bases[header.source_rank], a.dgate_offset);
          source_dgate[header.route_index] = dgate_canonical[row + local_row];
#if MEGAMOE_DEBUG_VALIDATE
          if (a.capture_stage_storage) {
            auto *source_coverage = peer_at<std::uint32_t>(
                a.peer_bases[header.source_rank], a.debug_coverage_offset);
            source_coverage[header.route_index] = 1U;
          }
#endif
        }
      }
    }
    __syncthreads();
    if (threadIdx.x == 0U)
      task_shared += gridDim.x;
  }
  __threadfence_system();
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dx_complete_clock = device_globaltimer_ns();
  MEGAMOE_TIMELINE_RECORD(a.timeline, DxTileReady, Compute, 0U, dx_tasks,
                          a.epoch);
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDxComplete, dx_tasks);

  // Every dX contribution and dGate scalar is now globally visible. Publish
  // the per-source completion before dW so peer acquire latency is hidden by
  // the independent weight-gradient work instead of extending the tail.
  const auto backward_first_local =
      remote_backend_enabled<StaticLocal>(a.use_gin)
          ? a.node * a.local_world_size
          : 0U;
  const auto backward_local_count =
      remote_backend_enabled<StaticLocal>(a.use_gin) ? a.local_world_size
                                                      : a.world;
  if (!remote_backend_enabled<StaticLocal>(a.use_socket) &&
      blockIdx.x == 0 &&
      threadIdx.x < backward_local_count) {
    const auto peer = backward_first_local + threadIdx.x;
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[peer], a.outgoing_signal_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        signal[a.rank * a.channels]);
    ready.store(a.epoch, ::cuda::memory_order_release);
  }

#if __has_include(<nccl_device.h>)
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    grid.sync();
    const auto first_local = a.node * a.local_world_size;
    const auto compute_epoch = a.epoch | (1ULL << 63U);
    if (blockIdx.x == 0U) {
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        auto *ready = peer_at<unsigned long long>(
            a.peer_bases[peer], a.gin_local_compute_ready_offset);
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            published(ready[a.rank]);
        published.store(compute_epoch, ::cuda::memory_order_release);
      }
      __syncthreads();
      auto *local_ready = peer_at<unsigned long long>(
          a.local_base, a.gin_local_compute_ready_offset);
      for (std::uint32_t local = threadIdx.x;
           local < a.local_world_size; local += blockDim.x) {
        const auto peer = first_local + local;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            observed(local_ready[peer]);
        while (observed.load(::cuda::memory_order_acquire) < compute_epoch &&
               !backward_cancelled(a)) {
        }
      }
    }
    grid.sync();
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        bytes + a.gin_receive_header_offset);
    auto *receive_fanout = reinterpret_cast<DeviceFanoutRecord *>(
        bytes + a.gin_receive_fanout_offset);
    auto *return_staging = reinterpret_cast<transport::BFloat16 *>(
        bytes + a.gin_return_staging_offset);
    auto *return_gate_staging = reinterpret_cast<float *>(
        bytes + a.gin_return_gate_staging_offset);
    MEGAMOE_TIMELINE_RECORD(a.timeline, DxReturnPackStart, GinReturn, 0U,
                            a.dispatch.sizes.incoming_row_count, a.epoch);
    const auto vector_count = a.hidden / 8ULL;
    const auto global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto global_threads =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
      for (std::uint64_t item = global_thread;
           item < signal.expected_packets * packet_vectors;
           item += global_threads) {
        const auto sequence = item / packet_vectors;
        const auto within_packet = item % packet_vectors;
        const auto fanout = within_packet / vector_count;
        const auto vector = within_packet % vector_count;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        const auto *destination_dx = peer_at<transport::BFloat16>(
            a.peer_bases[destination], a.dx_canonical_offset);
        reinterpret_cast<uint4 *>(return_staging)[
            ((source_partition * a.max_tokens_per_rank *
                  a.dispatch.identity.top_k +
              header.dst_row + fanout) *
                 vector_count) +
            vector] = reinterpret_cast<const uint4 *>(destination_dx)[
            target.dst_row * vector_count + vector];
      }
      for (std::uint64_t slot = global_thread;
           slot < signal.expected_packets * a.dispatch.identity.top_k;
           slot += global_threads) {
        const auto sequence = slot / a.dispatch.identity.top_k;
        const auto fanout = slot % a.dispatch.identity.top_k;
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        if (fanout >= header.route_index)
          continue;
        const auto target = receive_fanout[
            source_partition * a.max_tokens_per_rank *
                    a.dispatch.identity.top_k +
                header.dst_row + fanout];
        const auto destination =
            a.node * a.local_world_size + target.destination_local_rank;
        const auto *destination_dgate = peer_at<float>(
            a.peer_bases[destination], a.dgate_canonical_offset);
        return_gate_staging[source_partition * a.max_tokens_per_rank *
                                a.dispatch.identity.top_k +
                            header.dst_row + fanout] =
            destination_dgate[target.dst_row];
      }
    }
    // dGate is only four bytes per route and exposed the missing publication
    // fence as intermittent stale values even when the much larger dX DMA had
    // enough latency to appear correct.
    __threadfence_system();
    grid.sync();
  }
  if (remote_backend_enabled<StaticLocal>(a.use_gin) &&
      blockIdx.x < a.gin_context_count) {
    const auto context = static_cast<std::uint32_t>(blockIdx.x);
    const ncclGin gin(a.gin_dev_comm, context);
    const auto rail = ncclTeamRail(a.gin_dev_comm);
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *receive_headers = reinterpret_cast<LocalP2pRouteHeader *>(
        bytes + a.gin_receive_header_offset);
    unsigned long long return_packets = 0ULL;
    for (std::uint32_t signal_index = 0;
         signal_index < a.dispatch.sizes.incoming_signal_count;
         ++signal_index) {
      const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
      if (signal.source_node == a.node)
        continue;
      const auto source_partition =
          signal.source_node < a.node ? signal.source_node
                                      : signal.source_node - 1U;
      const auto sequence_begin =
          signal.expected_packets * context / a.gin_context_count;
      const auto sequence_end = signal.expected_packets * (context + 1U) /
                                a.gin_context_count;
      std::uint64_t first_return = 0;
      std::uint64_t last_return = 0;
      for (std::uint64_t sequence = sequence_begin;
           sequence < sequence_end; ++sequence) {
        const auto packet =
            source_partition * a.max_tokens_per_rank + sequence;
        const auto header = receive_headers[packet];
        const auto fanout_count = header.route_index;
        if (sequence == sequence_begin)
          first_return = header.dst_row;
        last_return = header.dst_row + fanout_count;
      }
      __syncthreads();
      if (threadIdx.x == 0U && last_return != first_return) {
          MEGAMOE_TIMELINE_RECORD(
              a.timeline, DxReturnGinPutIssue, GinReturn, context,
              (last_return - first_return) * a.hidden *
                  sizeof(transport::BFloat16),
              a.epoch);
          gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
                  a.gin_return_receive_offset +
                      first_return * a.hidden *
                          sizeof(transport::BFloat16),
                  a.gin_window,
                  a.gin_return_staging_offset +
                      (source_partition * a.max_tokens_per_rank *
                               a.dispatch.identity.top_k +
                       first_return) *
                          a.hidden * sizeof(transport::BFloat16),
                  (last_return - first_return) * a.hidden *
                      sizeof(transport::BFloat16));
          gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
                  a.gin_return_gate_receive_offset +
                      first_return * sizeof(float),
                  a.gin_window,
                  a.gin_return_gate_staging_offset +
                      (source_partition * a.max_tokens_per_rank *
                               a.dispatch.identity.top_k +
                       first_return) * sizeof(float),
                  (last_return - first_return) * sizeof(float),
                  ncclGin_SignalInc{1},
                  ncclGin_CounterInc{0}, ncclCoopThread{}, ncclGin_None{},
                  ::cuda::thread_scope_block, ::cuda::thread_scope_device);
          atomicAdd(&status->gin_return_put_count, 2ULL);
          atomicAdd(&status->gin_signal_count, 1ULL);
          atomicAdd(&status->gin_metadata_bytes,
                    (last_return - first_return) * sizeof(float));
          atomicAdd(&status->gin_return_payload_bytes,
                    (last_return - first_return) * a.hidden *
                        sizeof(transport::BFloat16));
          ++return_packets;
      }
      __syncthreads();
    }
    if (threadIdx.x == 0U && return_packets != 0ULL)
      gin.waitCounter(ncclCoopThread{}, 0, return_packets);
    MEGAMOE_TIMELINE_RECORD(a.timeline, DxReturnLocalCompletion, GinReturn,
                            context, return_packets, a.epoch);
    __syncthreads();
    // The FP32 scalar return is deliberately re-issued after local completion
    // of the payload batch. This gives the remote acquire a distinct terminal
    // signal instead of relying on visibility of a four-byte DMA adjacent to
    // the much larger dX operation on the same context.
    if (threadIdx.x == 0U) {
      unsigned long long scalar_returns = 0ULL;
      for (std::uint32_t signal_index = 0;
           signal_index < a.dispatch.sizes.incoming_signal_count;
           ++signal_index) {
        const auto signal = a.dispatch.incoming_proxy_signals[signal_index];
        if (signal.source_node == a.node)
          continue;
        const auto source_partition =
            signal.source_node < a.node ? signal.source_node
                                        : signal.source_node - 1U;
        const auto sequence_begin =
            signal.expected_packets * context / a.gin_context_count;
        const auto sequence_end =
            signal.expected_packets * (context + 1U) / a.gin_context_count;
        std::uint64_t first_return = 0;
        std::uint64_t last_return = 0;
        for (std::uint64_t sequence = sequence_begin;
             sequence < sequence_end; ++sequence) {
          const auto packet =
              source_partition * a.max_tokens_per_rank + sequence;
          const auto header = receive_headers[packet];
          if (sequence == sequence_begin)
            first_return = header.dst_row;
          last_return = header.dst_row + header.route_index;
        }
        if (last_return == first_return)
          continue;
        gin.put(rail, static_cast<int>(signal.source_node), a.gin_window,
                a.gin_return_gate_receive_offset +
                    first_return * sizeof(float),
                a.gin_window,
                a.gin_return_gate_staging_offset +
                    (source_partition * a.max_tokens_per_rank *
                         a.dispatch.identity.top_k +
                     first_return) * sizeof(float),
                (last_return - first_return) * sizeof(float),
                ncclGin_SignalInc{1}, ncclGin_CounterInc{0},
                ncclCoopThread{}, ncclGin_None{},
                ::cuda::thread_scope_block, ::cuda::thread_scope_device);
        atomicAdd(&status->gin_return_put_count, 1ULL);
        atomicAdd(&status->gin_signal_count, 1ULL);
        atomicAdd(&status->gin_metadata_bytes,
                  (last_return - first_return) * sizeof(float));
        ++scalar_returns;
      }
      if (scalar_returns != 0ULL)
        gin.waitCounter(ncclCoopThread{}, 0,
                        return_packets + scalar_returns);
    }
    __syncthreads();
    unsigned long long outgoing_remote_packets = 0ULL;
    if (threadIdx.x == 0U) {
      std::uint64_t total = 0;
      for (std::uint64_t packet = 0;
           packet < a.dispatch.sizes.packet_count; ++packet)
        if (a.dispatch.packet_headers[packet].destination_node != a.node)
          ++total;
      const auto begin = total * context / a.gin_context_count;
      const auto end = total * (context + 1U) / a.gin_context_count;
      outgoing_remote_packets = end - begin;
      if (outgoing_remote_packets != 0ULL)
        gin.waitSignal(ncclCoopThread{}, 1, 2);
      MEGAMOE_TIMELINE_RECORD(a.timeline, SourceSignalObserved, SourceRelay,
                              context, outgoing_remote_packets, a.epoch);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          phase(scheduler[2]);
      phase.fetch_add(1ULL, ::cuda::memory_order_release);
    }
    __syncthreads();
  }
  // The GIN context CTAs remain dedicated to return progress. Other CTAs
  // immediately convert from dX workers to dW workers below, so rail DMA and
  // the split-K=1 gradient GEMMs occupy independent portions of the machine.
  // The two branches rejoin before source relay/reduce.
#endif

  // dW2 and dW1 use split_k=1. Each task owns one output tile and traverses
  // that expert's padded canonical-M segment in ascending order.
  // dW2 is evaluated as dW2^T = U^T @ dY.  Its epilogue transposes each
  // 128x128 accumulator tile directly into the parameter layout [H,I].
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDw2Started, dw2_tasks);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2Ready, Compute, 0U, dw2_tasks,
                          a.epoch);
#if MEGAMOE_DEVICE_TIMELINE
  bool timeline_dw2_started = false;
#endif
  while (true) {
    if (threadIdx.x == 0)
      task_shared = atomicAdd(&scheduler[3], 1ULL);
    __syncthreads();
    if (task_shared >= dw2_tasks)
      break;
#if MEGAMOE_DEVICE_TIMELINE
    if (!timeline_dw2_started) {
      MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2Start, Compute, 0U, task_shared,
                              a.epoch);
      timeline_dw2_started = true;
    }
#endif
    auto task = task_shared;
    const auto per_expert = dw2_mtiles * dw2_ntiles;
    const auto expert = static_cast<std::uint32_t>(task / per_expert);
    task %= per_expert;
    const auto ntile = task / dw2_mtiles;
    const auto mtile = task % dw2_mtiles;
    const auto reduction_start = claims[expert];
    const auto reduction =
        (a.dispatch.expert_row_count[expert] + 63ULL) & ~63ULL;
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0U) {
      const auto now = device_globaltimer_ns();
      if (atomicCAS(&dw_timeline[expert].dw2_first_start_clock, 0ULL, now) ==
          0ULL)
        atomicMin(&status->dw2_first_start_clock, now);
    }
#else
    if constexpr (!StaticLocal) {
      if (threadIdx.x == 0U && task_shared < gridDim.x)
        atomicMin(&status->dw2_first_start_clock, device_globaltimer_ns());
    }
#endif
    float accum[BackwardDwDirectMma::kNumAccum];
    backward_wgmma_dw_mn_k_tile(
        dy_mn_map, u_transpose_map,
        static_cast<std::uint32_t>(a.dispatch.expert_row_prefix[expert]),
        static_cast<std::uint32_t>(reduction_start),
        static_cast<std::uint32_t>(mtile * 128ULL),
        static_cast<std::uint32_t>(ntile * kDwN),
        static_cast<std::uint32_t>(reduction), smem_a, smem_b,
        full_barriers, empty_barriers, accum, pipeline_iteration);
    const auto output_rows = a.hidden;
    const auto output_columns = a.intermediate;
    const auto expert_stride = output_rows * output_columns;
    auto *output_tile = smem_a;
    visit_forward_accumulator_wide(
        accum, [&](std::uint32_t wg, std::uint32_t m,
                   std::uint32_t column, float v0, float v1) {
          const auto local_row = wg * 64U + m;
          output_tile[local_row * kDwN + column] =
              transport::BFloat16{float_to_bf16_bits(v0)};
          output_tile[local_row * kDwN + column + 1U] =
              transport::BFloat16{float_to_bf16_bits(v1)};
        });
    __syncthreads();
    constexpr std::uint32_t kVectorsPerRow = kDwN / 8U;
    constexpr std::uint32_t kOutputVectors = 128U * kVectorsPerRow;
    for (std::uint32_t vector = threadIdx.x; vector < kOutputVectors;
         vector += blockDim.x) {
      const auto local_row = vector / kVectorsPerRow;
      const auto local_vector = vector % kVectorsPerRow;
      const auto mma_row = mtile * 128ULL + local_row;
      const auto mma_column = ntile * kDwN + local_vector * 8ULL;
      if (mma_row < output_rows && mma_column + 7ULL < output_columns) {
        reinterpret_cast<uint4 *>(
            dw2 + expert * expert_stride + mma_row * output_columns +
            mma_column)[0] = reinterpret_cast<const uint4 *>(
                output_tile + local_row * kDwN)[local_vector];
      }
    }
    __syncthreads();
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0U) {
      const auto complete =
          atomicAdd(&dw_timeline[expert].dw2_tasks_complete, 1ULL) + 1ULL;
      if (complete == per_expert) {
        const auto now = device_globaltimer_ns();
        dw_timeline[expert].dw2_complete_clock = now;
        atomicMax(&status->dw2_last_complete_clock, now);
      }
    }
    __syncthreads();
#endif
  }
#if !MEGAMOE_DEBUG_VALIDATE
  if constexpr (!StaticLocal) {
    if (threadIdx.x == 0U)
      atomicMax(&status->dw2_last_complete_clock, device_globaltimer_ns());
  }
#endif
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDw2Complete, dw2_tasks);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw2End, Compute, 0U, dw2_tasks,
                          a.epoch);

  // dW1 alone uses a persistent outer warp-specialized loop.  Unlike the
  // preceding general dW2 loop, the accumulator is lexically confined to the
  // math branch and the producer WG executes only the TMA path.
  // Four input stages minimize canonical-M stalls. The accumulator epilogue
  // reuses the input storage only after all stages have been consumed.
  constexpr std::uint32_t kDw1Stages = 4U;
  constexpr std::uint32_t kDw1AStageElements = 128U * 64U;
  constexpr std::uint32_t kDw1BStageElements = 256U * 64U;
  auto *dw1_smem_a = reinterpret_cast<transport::BFloat16 *>(storage);
  auto *dw1_smem_b = dw1_smem_a + kDw1Stages * kDw1AStageElements;
  auto *dw1_output_tile = dw1_smem_a;
  auto *dw1_full_barriers = reinterpret_cast<ForwardBarrier *>(
      dw1_smem_b + kDw1Stages * kDw1BStageElements);
  auto *dw1_empty_barriers = dw1_full_barriers + kDw1Stages;
  if (threadIdx.x == 256U) {
    for (std::uint32_t stage = 0; stage < kDw1Stages; ++stage) {
      dw1_full_barriers[stage].init(1);
      dw1_empty_barriers[stage].init(8);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  constexpr std::uint32_t kDwTaskReadyBarrier = 0U;
  constexpr std::uint32_t kDwTaskDoneBarrier = 1U;
  constexpr std::uint32_t kDwMathOutputReadyBarrier = 2U;
  constexpr std::uint32_t kDwMathOutputDoneBarrier = 3U;
  const auto dw_warpgroup = threadIdx.x / 128U;
  // Every task still has exactly one owner; grid-stride assignment avoids a
  // 64K-operation global claim stream and preserves parallel expert coverage.
  if (threadIdx.x == 0U)
    task_shared = remote_backend_enabled<StaticLocal>(a.use_gin)
                      ? atomicAdd(&scheduler[4], 1ULL)
                      : blockIdx.x;
  __syncthreads();
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDw1Started, dw1_tasks);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1Ready, Compute, 0U, dw1_tasks,
                          a.epoch);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1Start, Compute, 0U, 0ULL, a.epoch);
  if (dw_warpgroup >= 2U) {
    cutlass::arch::warpgroup_reg_dealloc<48>();
    std::uint32_t dw_pipeline = 0U;
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kDwTaskReadyBarrier);
      auto task = task_shared;
      if (task >= dw1_tasks)
        break;
      const auto per_expert = dw1_mtiles * dw1_ntiles;
      const auto expert = static_cast<std::uint32_t>(task / per_expert);
      task %= per_expert;
      const auto mtile = task / dw1_ntiles;
      const auto ntile = task % dw1_ntiles;
      const auto reduction_start = claims[expert];
      const auto reduction =
          (a.dispatch.expert_row_count[expert] + 63ULL) & ~63ULL;
      if (threadIdx.x == 256U) {
        const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
        for (std::uint32_t iteration = 0; iteration < iterations;
             ++iteration) {
          const auto pipeline = dw_pipeline + iteration;
          const auto stage = pipeline % kDw1Stages;
          const auto phase = (pipeline / kDw1Stages) & 1U;
          dw1_empty_barriers[stage].wait(phase ^ 1U);
          auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
              dw1_smem_a + stage * kDw1AStageElements);
          auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
              dw1_smem_b + stage * kDw1BStageElements);
          const auto a_k = static_cast<std::uint32_t>(
              reduction_start + iteration * 64ULL);
          const auto b_k = static_cast<std::uint32_t>(
              a.dispatch.expert_row_prefix[expert] + iteration * 64ULL);
          deep_gemm::tma::copy<64, 128, 128, cutlass::bfloat16_t>(
              &dab_transpose_map, &dw1_full_barriers[stage], stage_a, a_k,
              static_cast<std::uint32_t>(mtile * 128ULL));
          for (std::uint32_t atom = 0; atom < 4U; ++atom)
            deep_gemm::tma::copy<64, 64, 128, cutlass::bfloat16_t>(
                &x_mn_map, &dw1_full_barriers[stage],
                stage_b + atom * 64U * 64U,
                static_cast<std::uint32_t>(ntile * kDwN + atom * 64ULL),
                b_k);
          dw1_full_barriers[stage].arrive_and_expect_tx(
              (kDw1AStageElements + kDw1BStageElements) *
              sizeof(transport::BFloat16));
        }
        dw_pipeline += iterations;
      }
      cutlass::arch::NamedBarrier::sync(384U, kDwTaskDoneBarrier);
    }
  } else {
    cutlass::arch::warpgroup_reg_alloc<224>();
    std::uint32_t dw_pipeline = 0U;
    while (true) {
      cutlass::arch::NamedBarrier::sync(384U, kDwTaskReadyBarrier);
      auto task = task_shared;
      if (task >= dw1_tasks)
        break;
      const auto per_expert = dw1_mtiles * dw1_ntiles;
      const auto expert = static_cast<std::uint32_t>(task / per_expert);
      task %= per_expert;
      const auto mtile = task / dw1_ntiles;
      const auto ntile = task % dw1_ntiles;
      const auto reduction =
          (a.dispatch.expert_row_count[expert] + 63ULL) & ~63ULL;
#if MEGAMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0) {
        const auto now = device_globaltimer_ns();
        if (atomicCAS(&dw_timeline[expert].dw1_first_start_clock, 0ULL, now) ==
            0ULL)
          atomicMin(&status->dw1_first_start_clock, now);
      }
#else
      if constexpr (!StaticLocal) {
        if (threadIdx.x == 0U && task_shared < gridDim.x)
          atomicMin(&status->dw1_first_start_clock, device_globaltimer_ns());
      }
#endif
      float accum[BackwardDwMixedMma::kNumAccum];
#pragma unroll
      for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
        accum[index] = 0.0F;
      const auto iterations = static_cast<std::uint32_t>(reduction / 64ULL);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto pipeline = dw_pipeline + iteration;
        const auto stage = pipeline % kDw1Stages;
        const auto phase = (pipeline / kDw1Stages) & 1U;
        auto *stage_a = reinterpret_cast<cutlass::bfloat16_t *>(
            dw1_smem_a + stage * kDw1AStageElements);
        auto *stage_b = reinterpret_cast<cutlass::bfloat16_t *>(
            dw1_smem_b + stage * kDw1BStageElements);
        dw1_full_barriers[stage].wait(phase);
#pragma unroll
        for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
        deep_gemm::ptx::warpgroup_arrive();
#pragma unroll
        for (std::uint32_t k = 0; k < 64U / BackwardDwMixedMma::K; ++k) {
          const auto desc_a = deep_gemm::mma::sm90::make_gmma_desc<
              cute::UMMA::Major::K, 128, 64, 128, cutlass::bfloat16_t>(
              stage_a, dw_warpgroup * 64U, k * BackwardDwMixedMma::K);
          const auto desc_b = deep_gemm::mma::sm90::make_gmma_desc<
              cute::UMMA::Major::MN, 256, 64, 128, cutlass::bfloat16_t>(
              stage_b, 0U, k * BackwardDwMixedMma::K);
          BackwardDwMixedMma::wgmma(desc_a, desc_b, accum, 1);
        }
        deep_gemm::ptx::warpgroup_commit_batch();
#pragma unroll
        for (int index = 0; index < BackwardDwMixedMma::kNumAccum; ++index)
          deep_gemm::ptx::warpgroup_fence_operand(accum[index]);
        deep_gemm::ptx::warpgroup_wait<0>();
        if ((threadIdx.x & 31U) == 0U)
          dw1_empty_barriers[stage].arrive();
      }
      dw_pipeline += iterations;
      const auto warp = threadIdx.x / 32U;
      const auto lane = threadIdx.x & 31U;
#pragma unroll
      for (std::uint32_t index = 0;
           index < BackwardDwMixedMma::kNumAccum / 4U; ++index) {
        constexpr std::uint32_t kBankGroupBytes = 16U;
        const auto atom = index / (64U / 8U);
        const auto in_atom = index % (64U / 8U);
        const auto row = lane;
        auto column = in_atom;
        column ^= row % (128U / kBankGroupBytes);
        auto *smem_pointer =
            reinterpret_cast<std::uint8_t *>(dw1_output_tile) +
            warp * 16U * 128U + atom * 128U * 128U + row * 128U +
            column * kBankGroupBytes;
        deep_gemm::ptx::SM90_U32x2_STSM_N<__nv_bfloat162>::copy(
            __float22bfloat162_rn(
                make_float2(accum[index * 4U], accum[index * 4U + 1U])),
            __float22bfloat162_rn(make_float2(accum[index * 4U + 2U],
                                               accum[index * 4U + 3U])),
            smem_pointer);
      }
      cute::tma_store_fence();
      cutlass::arch::NamedBarrier::sync(256U,
                                        kDwMathOutputReadyBarrier);
      if (threadIdx.x < 4U) {
        const auto atom = threadIdx.x;
        cute::SM90_TMA_STORE_2D::copy(
            &dw1_store_map, dw1_output_tile + atom * 128U * 64U,
            static_cast<std::uint32_t>(ntile * kDwN + atom * 64ULL),
            static_cast<std::uint32_t>(expert * 2ULL * a.intermediate +
                                       mtile * 128ULL));
        cute::tma_store_arrive();
        cute::tma_store_wait<0>();
      }
      cutlass::arch::NamedBarrier::sync(256U,
                                        kDwMathOutputDoneBarrier);
#if MEGAMOE_DEBUG_VALIDATE
      if (threadIdx.x == 0) {
        const auto complete =
            atomicAdd(&dw_timeline[expert].dw1_tasks_complete, 1ULL) + 1ULL;
        if (complete == per_expert) {
          const auto now = device_globaltimer_ns();
          dw_timeline[expert].dw1_complete_clock = now;
          atomicMax(&status->dw1_last_complete_clock, now);
        }
      }
#endif
      cutlass::arch::NamedBarrier::sync(384U, kDwTaskDoneBarrier);
      if (threadIdx.x == 0U)
        task_shared = remote_backend_enabled<StaticLocal>(a.use_gin)
                          ? atomicAdd(&scheduler[4], 1ULL)
                          : task_shared + gridDim.x;
    }
  }
  if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
    backward_publish_progress(a, kBackwardDw1Complete, dw1_tasks);
  MEGAMOE_TIMELINE_RECORD(a.timeline, Dw1End, Compute, 0U, dw1_tasks,
                          a.epoch);
  grid.sync();
#if __has_include(<nccl_device.h>)
  if (remote_backend_enabled<StaticLocal>(a.use_gin)) {
    if (threadIdx.x == 0U) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
          phase(scheduler[2]);
      while (phase.load(::cuda::memory_order_acquire) <
             a.gin_context_count) {
      }
    }
    __syncthreads();
    auto *bytes = static_cast<std::byte *>(a.gin_window_base);
    auto *return_receive = reinterpret_cast<transport::BFloat16 *>(
        bytes + a.gin_return_receive_offset);
    auto *return_gate_receive = reinterpret_cast<float *>(
        bytes + a.gin_return_gate_receive_offset);
    MEGAMOE_TIMELINE_RECORD(a.timeline, DxSourceRelay, SourceRelay, 0U,
                            a.dispatch.sizes.packet_count, a.epoch);
    const auto vector_count = a.hidden / 8ULL;
    const auto packet_vectors = a.dispatch.identity.top_k * vector_count;
    const auto global_thread =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto global_threads =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t item = global_thread;
         item < a.dispatch.sizes.packet_count * packet_vectors;
         item += global_threads) {
      const auto packet = item / packet_vectors;
      const auto within_packet = item % packet_vectors;
      const auto fanout_slot = within_packet / vector_count;
      const auto vector = within_packet % vector_count;
      const auto descriptor = a.dispatch.packet_headers[packet];
      if (descriptor.destination_node == a.node)
        continue;
      const auto fanout_begin = a.dispatch.packet_fanout_offsets[packet];
      const auto fanout_end = a.dispatch.packet_fanout_offsets[packet + 1U];
      if (fanout_slot >= fanout_end - fanout_begin)
        continue;
      const auto fanout = fanout_begin + fanout_slot;
      const auto target = a.dispatch.packet_fanout[fanout];
      const auto route = descriptor.local_token_id *
                             a.dispatch.identity.top_k +
                         target.topk_slot;
      reinterpret_cast<uint4 *>(dx_return)[route * vector_count + vector] =
          reinterpret_cast<const uint4 *>(return_receive)[
              fanout * vector_count + vector];
    }
    for (std::uint64_t packet = global_thread;
         packet < a.dispatch.sizes.packet_count; packet += global_threads) {
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
        dgate[route] = return_gate_receive[fanout];
#if MEGAMOE_DEBUG_VALIDATE
        if (a.capture_stage_storage)
          debug_coverage[route] = 1U;
#endif
      }
    }
    MEGAMOE_TIMELINE_RECORD(a.timeline, DGateReturn, SourceRelay, 0U,
                            a.dispatch.sizes.route_count, a.epoch);
    __threadfence_system();
    grid.sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
      for (std::uint32_t destination = 0; destination < a.world;
           ++destination) {
        if (destination / a.local_world_size == a.node)
          continue;
        ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
            ready(peer_at<unsigned long long>(
                a.local_base, a.outgoing_signal_offset)
                      [destination * a.channels]);
        ready.store(a.epoch, ::cuda::memory_order_release);
      }
    }
    grid.sync();
  }
#endif
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->dw2_complete_clock = status->dw2_last_complete_clock;
#if !MEGAMOE_DEBUG_VALIDATE
    status->dw1_last_complete_clock = device_globaltimer_ns();
#endif
    status->dw1_complete_clock = status->dw1_last_complete_clock;
#if MEGAMOE_DEBUG_VALIDATE
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert) {
      const auto ready = dw_timeline[expert].ready_clock;
      const auto dw1_start = dw_timeline[expert].dw1_first_start_clock;
      const auto dw2_start = dw_timeline[expert].dw2_first_start_clock;
      if (ready != 0ULL && dw1_start >= ready)
        status->dw1_max_ready_wait =
            max(status->dw1_max_ready_wait, dw1_start - ready);
      if (ready != 0ULL && dw2_start >= ready)
        status->dw2_max_ready_wait =
            max(status->dw2_max_ready_wait, dw2_start - ready);
    }
#endif
  }
  if (!remote_backend_enabled<StaticLocal>(a.use_socket)) {
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceReady, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceStart, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    backward_reduce_token_slice(a, dx_return, dx_combined, blockIdx.x,
                                gridDim.x);
    MEGAMOE_TIMELINE_RECORD(a.timeline, ReduceEnd, Reduce, 0U,
                            a.dispatch.identity.num_tokens, a.epoch);
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0)
      status->dx_reduce_complete_clock = device_globaltimer_ns();
    if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
      backward_publish_progress(a, kBackwardReduceComplete,
                                a.dispatch.identity.num_tokens);
    backward_terminal_rendezvous<StaticLocal>(a);
  }
  grid.sync();

#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    publish_local_structured_error(status, a.rank, 90U, 1U, 1ULL, 0ULL);
  }
  grid.sync();
#endif

  if (blockIdx.x == 0 && threadIdx.x == 0 &&
      a.injected_status_error_rank == static_cast<int>(a.rank))
    publish_local_structured_error(status, a.rank, 220U, 1U, 0ULL, 1ULL);
  grid.sync();
  if constexpr (!StaticLocal) {
    if (a.use_gin) {
      if (blockIdx.x == 0U && threadIdx.x == 0U) {
        // This point is reachable only after the fixed Dispatch, Relay,
        // dV/dAB/dX/dW, bulk Return, source-arrival, and Reduce barriers above.
        // Convert those already-proven static task spaces into the immutable
        // completion vector; no producer may change expected after this point.
        for (std::size_t index = 0;
             index < transport_state::kTaskClassCount; ++index)
          mark_task_space_complete(completion->tasks[index]);
        completion->return_local_completion_completed =
            completion->return_local_completion_expected;
        completion->ring_credit_restored = completion->ring_credit_expected;
        publish_elected_terminal(completion, a.epoch, status->error != 0U);
      }
      __syncthreads();
      if (blockIdx.x != 0U) {
        if (threadIdx.x == 0U) {
          ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
              terminal(completion->kernel_complete_epoch);
          while (terminal.load(::cuda::memory_order_acquire) < a.epoch) {
          }
        }
        __syncthreads();
        return;
      }
    }
  }
  publish_production_terminal_status<StaticLocal>(
      a, status, ProductionStatusPhase::Backward,
      a.production_error_records_offset,
      a.gin_production_error_records_offset);
  if constexpr (StaticLocal) {
    grid.sync();
  } else if (!a.use_gin) {
    grid.sync();
  } else {
    __syncthreads();
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
    // Remote dY dispatch and dX return have an exact one-to-one route/byte
    // correspondence for this source rank; reuse the measured dispatch count
    // instead of adding per-route atomics to the dX epilogue.
    status->actual_dx_peer_bytes = status->actual_dy_peer_bytes;
    status->dv_tasks = scheduler[8] *
                       ((a.intermediate + 255ULL) / 256ULL);
    status->dx_tasks = scheduler[8] * ((a.hidden + 255ULL) / 256ULL);
    status->dw1_tasks = static_cast<unsigned long long>(
        a.dispatch.sizes.owned_expert_count) *
        ((2ULL * a.intermediate + 127ULL) / 128ULL) *
        ((a.hidden + 255ULL) / 256ULL);
    status->dw2_tasks = static_cast<unsigned long long>(
        a.dispatch.sizes.owned_expert_count) *
        ((a.hidden + 127ULL) / 128ULL) *
        ((a.intermediate + 255ULL) / 256ULL);
    const auto kdv = a.hidden / 64ULL;
    const auto kdx = (2ULL * a.intermediate) / 64ULL;
    unsigned long long dw1_reduction_iterations = 0ULL;
    unsigned long long dw2_reduction_iterations = 0ULL;
    const auto dw1_tiles_per_expert =
        ((2ULL * a.intermediate + 127ULL) / 128ULL) *
        ((a.hidden + 255ULL) / 256ULL);
    const auto dw2_tiles_per_expert =
        ((a.hidden + 127ULL) / 128ULL) *
        ((a.intermediate + 255ULL) / 256ULL);
    for (std::uint32_t expert = 0;
         expert < a.dispatch.sizes.owned_expert_count; ++expert)
      {
        const auto reduction_iterations =
            (a.dispatch.expert_row_count[expert] + 63ULL) / 64ULL;
        dw1_reduction_iterations +=
            reduction_iterations * dw1_tiles_per_expert;
        dw2_reduction_iterations +=
            reduction_iterations * dw2_tiles_per_expert;
      }
    status->wgmma_instructions =
        (status->dv_tasks * kdv + status->dx_tasks * kdx) *
            2ULL * (64ULL / ForwardMma::K) +
        (dw1_reduction_iterations + dw2_reduction_iterations) *
            2ULL * (64ULL / ForwardMma::K);
    status->tma_loads = (status->dv_tasks * kdv +
                         status->dx_tasks * kdx) * 2ULL +
                        dw1_reduction_iterations * 5ULL +
                        dw2_reduction_iterations * 4ULL;
#endif
    status->complete_clock = device_globaltimer_ns();
    if constexpr (!StaticLocal || MEGAMOE_DEBUG_VALIDATE)
      backward_publish_progress(a, kBackwardTerminal, a.epoch);
  }
  MEGAMOE_TIMELINE_RECORD(a.timeline, Terminal, Kernel, 0U, 0ULL, a.epoch);
}

} // namespace

struct ActivationTransportResourcePool::Impl {
  std::shared_ptr<ActivationPoolState> state;
};

struct SelectedActivationLease<transport::BFloat16>::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  const transport::BFloat16 *pointer = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken producer_execution;
  std::uint64_t generation = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
  std::uint32_t device = 0;
  std::uint32_t layout = kSelectedActivationLayoutVersion;
  bool ready_recorded = false;
  std::shared_ptr<const void> external_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;
};

struct LocalP2pBackendState;
struct SocketBackendState;
struct GinBackendState;

struct ActivationMegaKernelBackendState {
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<internal::NcclSessionState> nccl;
  NcclSession session;
  CudaExecution execution;
  virtual bool requires_collective_terminal_status() const noexcept = 0;
  virtual ~ActivationMegaKernelBackendState() noexcept = default;
};

struct DeviceActivationTransportHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  ActivationTransportExpectation identity;
  ActivationTransportDeviceAudit audit;
  ActivationTransportTimings timings;
  std::uint64_t route_count = 0;
  std::uint64_t destination_rows = 0;
  std::shared_ptr<LocalP2pBackendState> local_p2p_backend;
  std::shared_ptr<SocketBackendState> socket_backend;
  std::shared_ptr<GinBackendState> gin_backend;
};

struct DeviceActivationTransport::Impl {
  CollectiveSession session;
  CudaExecution execution;
  ActivationTransportResourcePool pool;
};

struct LocalP2pBackendState : ActivationMegaKernelBackendState {
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;

  bool requires_collective_terminal_status() const noexcept override {
    return false;
  }

  ~LocalP2pBackendState() noexcept {
    if (!pool)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index) {
      if (owns_mapping[index] != 0 && peer_slot_bases[index])
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
    }
  }
};

struct LocalCudaP2pActivationTransport::Impl {
  std::shared_ptr<LocalP2pBackendState> backend;
};

struct SocketBackendState : ActivationMegaKernelBackendState {
  bool requires_collective_terminal_status() const noexcept override {
    return true;
  }
};

struct NcclSocketFallbackActivationTransport::Impl {
  std::shared_ptr<SocketBackendState> backend;
};

struct GinBackendState : ActivationMegaKernelBackendState {
  ncclDevComm_t dev_comm{};
  std::vector<ncclWindow_t> windows;
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;
  std::uint32_t num_nodes = 0;
  std::uint32_t local_world_size = 0;
  std::uint32_t node = 0;
  std::uint32_t local_rank = 0;
  std::uint32_t gin_context_count = 1;
  bool dev_comm_created = false;

  bool requires_collective_terminal_status() const noexcept override {
    return true;
  }

  ~GinBackendState() noexcept {
    if (!nccl)
      return;
    (void)cudaSetDevice(static_cast<int>(pool->device));
    for (std::size_t index = 0; index < peer_slot_bases.size(); ++index)
      if (owns_mapping[index] != 0 && peer_slot_bases[index] != nullptr)
        (void)cudaIpcCloseMemHandle(peer_slot_bases[index]);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    if (window_deregister != nullptr)
      for (const auto window : windows)
        if (window != nullptr)
          (void)window_deregister(nccl->comm, window);
    if (dev_comm_created && dev_comm_destroy != nullptr)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  }
};

struct GinRdmaActivationTransport::Impl {
  std::shared_ptr<GinBackendState> backend;
};

namespace {

struct ProductionStatusAgreementResult {
  bool has_error = false;
  ProductionStatusRecord error{};
  double elapsed_us = 0.0;
};

template <typename Status>
ProductionStatusAgreementResult agree_production_terminal_status(
    const std::shared_ptr<ActivationMegaKernelBackendState> &backend,
    PoolSlot &slot, ProductionStatusPhase phase, std::uint64_t generation,
    std::uint64_t epoch, std::uint32_t rank, std::uint32_t world,
    const Status &status, cudaEvent_t completion_event) {
  const auto begin = std::chrono::steady_clock::now();
  if (!MEGAMOE_DEBUG_VALIDATE &&
      !backend->requires_collective_terminal_status()) {
    ProductionStatusAgreementResult result;
    if (status.error != 0U) {
      result.has_error = true;
      result.error = ProductionStatusRecord{
          generation,
          static_cast<unsigned long long>(phase),
          static_cast<unsigned long long>(status.error_origin_rank),
          static_cast<unsigned long long>(status.error_code),
          static_cast<unsigned long long>(status.error_field),
          status.error_expected,
          status.error_actual,
          epoch};
    }
    result.elapsed_us = std::chrono::duration<double, std::micro>(
                            std::chrono::steady_clock::now() - begin)
                            .count();
    return result;
  }
  auto &layout = backend->pool->layout;
  const auto exec = internal::checked_execution_state(backend->execution);
  auto *host_records = slot.host_production_status;
  const auto error = status.error != 0U;
  host_records[rank] = ProductionStatusRecord{
      generation,
      static_cast<unsigned long long>(phase),
      error ? static_cast<unsigned long long>(status.error_origin_rank)
            : static_cast<unsigned long long>(rank),
      error ? static_cast<unsigned long long>(status.error_code) : 0ULL,
      error ? static_cast<unsigned long long>(status.error_field) : 0ULL,
      error ? status.error_expected : 0ULL,
      error ? status.error_actual : 0ULL,
      epoch};
  auto *send = at<ProductionStatusRecord>(
      slot.device, layout.production_error_records) + rank;
  auto *receive = at<unsigned long long>(
      slot.device, layout.production_status_matrix);
  internal::throw_cuda(
      "production-terminal-status",
      cudaMemcpyAsync(send, host_records + rank,
                      sizeof(ProductionStatusRecord), cudaMemcpyHostToDevice,
                      exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "H2D local production status record");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(ProductionStatusRecord));
  internal::throw_nccl(
      "production-terminal-status",
      ncclAllGather(send, receive, 8, ncclUint64, backend->nccl->comm,
                    exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "ncclAllGather(ProductionStatusRecord)");
  const auto matrix_bytes = checked_size(
      checked_mul(world, sizeof(ProductionStatusRecord),
                  "production terminal status matrix"),
      "production terminal status matrix");
  internal::throw_cuda(
      "production-terminal-status",
      cudaMemcpyAsync(host_records, receive, matrix_bytes,
                      cudaMemcpyDeviceToHost, exec->stream),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts, "D2H production status matrix");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            matrix_bytes);
  internal::throw_cuda(
      "production-terminal-status",
      cudaEventRecord(completion_event, exec->stream), rank, world,
      backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(production terminal status)");
  internal::throw_cuda(
      "production-terminal-status", cudaEventSynchronize(completion_event),
      rank, world, backend->session.session().identity(), generation,
      ExchangeKind::ExpertCounts,
      "cudaEventSynchronize(production terminal status)");

  ProductionStatusAgreementResult result;
  const auto expected_phase = static_cast<unsigned long long>(phase);
  const auto malformed = [&](std::uint32_t origin, std::uint64_t field,
                             std::uint64_t expected,
                             std::uint64_t actual) {
    result.has_error = true;
    result.error = ProductionStatusRecord{
        generation, expected_phase, origin, 2ULL, field, expected, actual,
        epoch};
  };
  for (std::uint32_t source = 0; source < world && !result.has_error;
       ++source) {
    const auto &record = host_records[source];
    if (record.generation != generation)
      malformed(source, 1ULL, generation, record.generation);
    else if (record.phase != expected_phase)
      malformed(source, 2ULL, expected_phase, record.phase);
    else if (record.origin_rank != source && record.error_code == 0ULL)
      malformed(source, 3ULL, source, record.origin_rank);
    else if (record.epoch != epoch)
      malformed(source, 4ULL, epoch, record.epoch);
  }
  if (!result.has_error) {
    for (std::uint32_t source = 0; source < world; ++source) {
      const auto &record = host_records[source];
      if (record.error_code == 0ULL)
        continue;
      if (record.origin_rank >= world) {
        malformed(source, 5ULL, world - 1ULL, record.origin_rank);
        break;
      }
      if (!result.has_error || record.origin_rank < result.error.origin_rank ||
          (record.origin_rank == result.error.origin_rank &&
           record.field < result.error.field)) {
        result.has_error = true;
        result.error = record;
      }
    }
  }
  result.elapsed_us = std::chrono::duration<double, std::micro>(
                          std::chrono::steady_clock::now() - begin)
                          .count();
  return result;
}

void throw_production_status_error(
    const ProductionStatusAgreementResult &agreement) {
  if (!agreement.has_error)
    return;
  const auto &error = agreement.error;
  const auto phase = error.phase == static_cast<unsigned long long>(
                                      ProductionStatusPhase::Forward)
                         ? "Forward"
                         : "Backward";
  throw NotifyDispatchError(
      NotifyErrorCode::TransportProtocolFailure,
      "phase=production-terminal-agreement, kind=" + std::string(phase) +
          ", origin_rank=" + std::to_string(error.origin_rank) +
          ", field=" + std::to_string(error.field) +
          ", code=" + std::to_string(error.error_code) +
          ", expected=" + std::to_string(error.expected) +
          ", actual=" + std::to_string(error.actual) +
          ", generation=" + std::to_string(error.generation));
}

} // namespace

struct ForwardAllocation {
  void *w1 = nullptr;
  void *w2 = nullptr;
  void *w1_transpose = nullptr;
  void *w2_transpose = nullptr;
  std::uint64_t bytes = 0;
  std::uint32_t device = 0;
  cudaEvent_t ready = nullptr;
  std::mutex backward_mutex;
  bool backward_prepared = false;
  bool owns_w1 = true;
  bool owns_w2 = true;
  std::shared_ptr<const void> w1_owner;
  std::shared_ptr<const void> w2_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;
  ~ForwardAllocation() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (ready) {
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(ready);
    }
    if (w2 && owns_w2) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w2);
    }
    if (w2_transpose) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w2_transpose);
    }
    if (w1_transpose) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w1_transpose);
    }
    if (w1 && owns_w1) {
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(w1);
    }
  }
};

struct ForwardWeightLease::Impl {
  std::shared_ptr<ForwardAllocation> allocation;
  const transport::BFloat16 *w1 = nullptr;
  const transport::BFloat16 *w2 = nullptr;
  const transport::BFloat16 *w1_transpose = nullptr;
  const transport::BFloat16 *w2_transpose = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken execution;
  std::uint32_t owned_experts = 0;
  std::uint32_t global_expert_begin = 0;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  std::uint32_t device = 0;
  CUtensorMap w1_map{};
  CUtensorMap w2_map{};
  CUtensorMap w1_transpose_map{};
  CUtensorMap w2_transpose_map{};
};

struct ForwardWeightStorageHashAccess {
  static const ForwardWeightLease::Impl &get(
      const ForwardWeightLease &weights) {
    return *weights.impl_;
  }
};

struct ForwardBackwardWeightAccess {
  static ForwardWeightLease::Impl &get(const ForwardWeightLease &weights) {
    return const_cast<ForwardWeightLease::Impl &>(*weights.impl_);
  }
};

struct DeviceForwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  DeviceDispatchHandle dispatch;
  BFloat16ActivationLease activations;
  ForwardWeightLease weights;
  ActivationTransportExpectation identity;
  ForwardMegaKernelAudit audit;
  ForwardMegaKernelTimings timings;
  ForwardCacheMode cache_mode = ForwardCacheMode::NoFc1OutputCache;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
  std::shared_ptr<DeviceDispatchConsumerLease> pending_consumer;
  bool completion_pending = false;
  // Host-enqueued Socket transfers execute outside the persistent-kernel
  // status block.  Retain their runtime byte counters across a deferred
  // Forward-to-Backward handoff without labelling them as GIN traffic.
  std::uint64_t socket_dispatch_peer_bytes = 0;
  std::uint64_t socket_return_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_route_bytes = 0;
};

struct LocalCudaP2pForwardMegaKernel::Impl {
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  std::shared_ptr<NcclSocketFallbackActivationTransport> socket_transport;
};

struct SelectedOutputGradientLease::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  const transport::BFloat16 *pointer = nullptr;
  std::vector<std::byte> session_identity;
  ExecutionToken producer_execution;
  std::uint64_t generation = 0;
  std::uint64_t tokens = 0;
  std::uint64_t hidden = 0;
  std::uint32_t device = 0;
  bool ready_recorded = false;
  std::shared_ptr<const void> external_owner;
  std::shared_ptr<const internal::ExternalCudaProducerState> external_producer;
};

DeviceDispatchConsumerLease
DeviceForwardHandle::transfer_consumer_to_backward() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-backward-handoff", "forward handle", 0,
                    "valid", "empty");
  auto &mutable_result = const_cast<DeviceForwardHandle::Impl &>(*impl_);
  if (!mutable_result.completion_pending ||
      !mutable_result.pending_consumer ||
      mutable_result.pending_consumer->state() !=
          DeviceDispatchConsumerState::Acquired)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-backward-handoff", "dispatch consumer",
                    mutable_result.identity.rank,
                    "one acquired transferable consumer",
                    "missing/already transferred/invalid");
  auto consumer = std::move(*mutable_result.pending_consumer);
  mutable_result.pending_consumer.reset();
  return consumer;
}

struct DeviceBackwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
  DeviceForwardHandle forward;
  SelectedOutputGradientLease output_gradient;
  ActivationTransportExpectation identity;
  BackwardMegaKernelAudit audit;
  BackwardMegaKernelTimings timings;
  std::uint64_t incoming_rows = 0;
  std::uint64_t route_count = 0;
  std::uint32_t owned_experts = 0;
  std::uint64_t hidden = 0;
  std::uint64_t intermediate = 0;
  bool used_forward_cache = false;
  bool hashes_captured = false;
  bool stage_storage_captured = false;
  std::uint64_t padded_reduction_rows = 0;
  BackwardStorageHashes hashes;
};

struct BackwardStorageHashAccess {
  static const DeviceBackwardHandle::Impl &get(
      const DeviceBackwardHandle &handle) {
    return *handle.impl_;
  }
  static DeviceBackwardHandle::Impl &get_mutable(
      const DeviceBackwardHandle &handle) {
    return const_cast<DeviceBackwardHandle::Impl &>(*handle.impl_);
  }
};

struct LocalCudaP2pBackwardMegaKernel::Impl {
  std::shared_ptr<ActivationMegaKernelBackendState> backend;
};

#if MEGAMOE_DEVICE_TIMELINE
namespace {
profiling::DeviceTimelineSnapshot download_device_timeline(
    const std::shared_ptr<ActivationPoolState> &pool,
    const std::shared_ptr<SlotLease> &lease, void *allocation,
    cudaEvent_t completion, const ActivationTransportExpectation &identity,
    profiling::TimelinePhase phase) {
  if (!pool || !lease || lease->index >= pool->slots.size() || !allocation)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "handle/storage", identity.rank, "valid", "empty");
  internal::throw_cuda(
      "device-timeline", cudaEventSynchronize(completion), identity.rank,
      identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "wait timeline producer completion");
  const auto max_ctas = pool->timeline_max_ctas;
  std::vector<std::uint32_t> counts(max_ctas);
  std::vector<std::uint32_t> overflow(max_ctas);
  const auto count_bytes = checked_size(
      checked_mul(max_ctas, sizeof(std::uint32_t), "timeline count download"),
      "timeline count download");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpy(counts.data(),
                 at<const void>(allocation, pool->timeline_layout.counts),
                 count_bytes, cudaMemcpyDeviceToHost),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline counts");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpy(overflow.data(),
                 at<const void>(allocation, pool->timeline_layout.overflow),
                 count_bytes, cudaMemcpyDeviceToHost),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline overflow");
  const auto record_count = checked_mul(max_ctas, profiling::kEventsPerCta,
                                        "timeline record download");
  std::vector<profiling::DeviceTimelineRecord> raw(
      checked_size(record_count, "timeline record vector"));
  const auto record_bytes = checked_size(
      checked_mul(record_count, sizeof(profiling::DeviceTimelineRecord),
                  "timeline record bytes"),
      "timeline record bytes");
  internal::throw_cuda(
      "device-timeline",
      cudaMemcpy(raw.data(),
                 at<const void>(allocation, pool->timeline_layout.records),
                 record_bytes, cudaMemcpyDeviceToHost),
      identity.rank, identity.world, "timeline", identity.generation,
      ExchangeKind::ExpertCounts, "download timeline records");
  profiling::DeviceTimelineSnapshot result;
  result.rank = identity.rank;
  result.generation = identity.generation;
  result.phase = phase;
  result.max_ctas = max_ctas;
  result.capacity_per_cta = profiling::kEventsPerCta;
  for (std::uint32_t cta = 0; cta < max_ctas; ++cta) {
    result.overflow_events += overflow[cta];
    const auto stored = std::min(counts[cta], profiling::kEventsPerCta);
    const auto begin = raw.begin() +
                       static_cast<std::ptrdiff_t>(cta) *
                           profiling::kEventsPerCta;
    result.records.insert(result.records.end(), begin, begin + stored);
  }
  std::stable_sort(result.records.begin(), result.records.end(),
                   [](const auto &left, const auto &right) {
                     if (left.timestamp_ns != right.timestamp_ns)
                       return left.timestamp_ns < right.timestamp_ns;
                     if (left.cta != right.cta)
                       return left.cta < right.cta;
                     return left.metadata < right.metadata;
                   });
  return result;
}
} // namespace

profiling::DeviceTimelineSnapshot testing::download_forward_device_timeline(
    const DeviceForwardHandle &handle) {
  if (!handle.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "Forward handle", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  const auto &slot = impl.pool->slots[impl.slot->index];
  return download_device_timeline(
      impl.pool, impl.slot, slot.forward_timeline, slot.forward_complete,
      impl.identity, profiling::TimelinePhase::Forward);
}

profiling::DeviceTimelineSnapshot testing::download_backward_device_timeline(
    const DeviceBackwardHandle &handle) {
  if (!handle.impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "device-timeline",
                    "Backward handle", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  const auto &slot = impl.pool->slots[impl.slot->index];
  return download_device_timeline(
      impl.pool, impl.slot, slot.backward_timeline, slot.complete,
      impl.identity, profiling::TimelinePhase::Backward);
}
#endif

std::uint32_t ForwardWeightLease::owned_experts() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->owned_experts;
}
std::uint64_t ForwardWeightLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t ForwardWeightLease::intermediate_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->intermediate;
}
std::uint64_t ForwardWeightLease::device_bytes() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->allocation->bytes;
}
std::uint32_t ForwardWeightLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weights",
                    "handle", 0, "valid", "empty");
  return impl_->device;
}

std::uint64_t SelectedOutputGradientLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-dy", "handle",
                    0, "valid", "empty");
  return impl_->tokens;
}
std::uint64_t SelectedOutputGradientLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-dy", "handle",
                    0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t SelectedOutputGradientLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-dy", "handle",
                    0, "valid", "empty");
  return impl_->generation;
}
std::uint32_t SelectedOutputGradientLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-dy", "handle",
                    0, "valid", "empty");
  return impl_->device;
}

const ActivationTransportExpectation &DeviceBackwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const BackwardMegaKernelAudit &DeviceBackwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const BackwardMegaKernelTimings &DeviceBackwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ForwardWeightLease testing::make_deterministic_forward_weights_for_testing(
    std::uint32_t owned_experts, std::uint32_t global_expert_begin,
    std::uint64_t hidden_size, std::uint64_t intermediate_size,
    const CollectiveSession &session, const CudaExecution &execution) {
  if (!session || !execution || owned_experts == 0 || hidden_size == 0 ||
      intermediate_size == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-weight-create", "shape/session/execution", 0,
                    "all valid and nonzero", "invalid");
  const auto w1_rows = checked_mul(
      checked_mul(owned_experts, 2ULL, "W1 paired expert rows"),
      intermediate_size, "W1 expert rows");
  const auto w1_elements = checked_mul(
      w1_rows,
      hidden_size, "W1 elements");
  const auto w2_elements = checked_mul(
      checked_mul(owned_experts, hidden_size, "W2 expert rows"),
      intermediate_size, "W2 elements");
  const auto w1_bytes = checked_size(
      checked_mul(w1_elements, sizeof(transport::BFloat16), "W1 bytes"),
      "W1 bytes");
  const auto w2_bytes = checked_size(
      checked_mul(w2_elements, sizeof(transport::BFloat16), "W2 bytes"),
      "W2 bytes");
  const auto state = internal::checked_execution_state(execution);
  auto allocation = std::make_shared<ForwardAllocation>();
  allocation->device = state->device;
  allocation->bytes = static_cast<std::uint64_t>(w1_bytes) + w2_bytes;
  internal::throw_cuda(
      "forward-weight-create", cudaSetDevice(static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "forward-weight-create", cudaMalloc(&allocation->w1, w1_bytes), 0,
      session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts, "cudaMalloc(W1)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  try {
    internal::throw_cuda(
        "forward-weight-create", cudaMalloc(&allocation->w2, w2_bytes), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaMalloc(W2)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    internal::throw_cuda(
        "forward-weight-create",
        cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaEventCreate(weight ready)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    initialize_forward_weights_kernel<<<4096, 256, 0, state->stream>>>(
        static_cast<transport::BFloat16 *>(allocation->w1), w1_elements,
        static_cast<transport::BFloat16 *>(allocation->w2), w2_elements,
        global_expert_begin, hidden_size, intermediate_size);
    internal::throw_cuda(
        "forward-weight-create", cudaGetLastError(), 0, session.world_size(),
        session.identity(), session.generation(), ExchangeKind::ExpertCounts,
        "initialize_forward_weights_kernel");
    internal::throw_cuda(
        "forward-weight-create",
        cudaEventRecord(allocation->ready, state->stream), 0,
        session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaEventRecord(weight ready)");
  } catch (...) {
    allocation.reset();
    throw;
  }
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->w1 = static_cast<const transport::BFloat16 *>(impl->allocation->w1);
  impl->w2 = static_cast<const transport::BFloat16 *>(impl->allocation->w2);
  impl->session_identity = session.stable_identity_bytes();
  impl->execution = execution.token();
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state->device;
  if (hidden_size == 7168 && intermediate_size == 3072) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        128, "W1");
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size, 256, "W2");
  }
  return ForwardWeightLease(std::move(impl));
}

ForwardWeightLease ExternalCudaProducer::publish_weights(
    const transport::BFloat16 *w1, std::uint64_t w1_capacity_elements,
    std::shared_ptr<const void> w1_owner,
    const transport::BFloat16 *w2, std::uint64_t w2_capacity_elements,
    std::shared_ptr<const void> w2_owner, std::uint32_t owned_experts,
    std::uint32_t global_expert_begin, std::uint64_t hidden_size,
    std::uint64_t intermediate_size) const {
  if (!state_ || !w1_owner || !w2_owner || owned_experts == 0 ||
      hidden_size == 0 || intermediate_size == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "external-weight-publish", "producer/owner/shape", 0,
                    "valid and nonzero", "invalid");
  const auto w1_elements = checked_mul(
      checked_mul(owned_experts, 2ULL * intermediate_size,
                  "external W1 rows"),
      hidden_size, "external W1 elements");
  const auto w2_elements = checked_mul(
      checked_mul(owned_experts, hidden_size, "external W2 rows"),
      intermediate_size, "external W2 elements");
  validate_external_cuda_range(w1, w1_elements, w1_capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "W1");
  validate_external_cuda_range(w2, w2_elements, w2_capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "W2");
  auto allocation = std::make_shared<ForwardAllocation>();
  allocation->device = state_->backend->device;
  allocation->w1 = const_cast<transport::BFloat16 *>(w1);
  allocation->w2 = const_cast<transport::BFloat16 *>(w2);
  allocation->owns_w1 = false;
  allocation->owns_w2 = false;
  allocation->w1_owner = std::move(w1_owner);
  allocation->w2_owner = std::move(w2_owner);
  allocation->external_producer = state_->backend;
  allocation->bytes = checked_add(
      checked_mul(w1_elements, sizeof(transport::BFloat16), "W1 bytes"),
      checked_mul(w2_elements, sizeof(transport::BFloat16), "W2 bytes"),
      "weight bytes");
  internal::throw_cuda(
      "external-weight-publish",
      cudaEventCreateWithFlags(&allocation->ready, cudaEventDisableTiming),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventCreate(weight ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-weight-publish",
      cudaEventRecord(allocation->ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(weight ready)");
  auto impl = std::make_shared<ForwardWeightLease::Impl>();
  impl->allocation = std::move(allocation);
  impl->w1 = w1;
  impl->w2 = w2;
  impl->session_identity = state_->backend->context.session().stable_identity_bytes();
  impl->execution = state_->backend->execution;
  impl->owned_experts = owned_experts;
  impl->global_expert_begin = global_expert_begin;
  impl->hidden = hidden_size;
  impl->intermediate = intermediate_size;
  impl->device = state_->backend->device;
  if (hidden_size == 7168 && intermediate_size == 3072) {
    impl->w1_map = make_bf16_tma_map(
        impl->allocation->w1, hidden_size,
        static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
        128, "W1");
    impl->w2_map = make_bf16_tma_map(
        impl->allocation->w2, intermediate_size,
        static_cast<std::uint64_t>(owned_experts) * hidden_size, 256, "W2");
  }
  return ForwardWeightLease(std::move(impl));
}

void prepare_forward_weights_for_backward(
    const ForwardWeightLease &weights, const CudaExecution &execution) {
  if (!weights || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-weight-prepare", "weights/execution", 0,
                    "valid", "empty");
  auto &impl = ForwardBackwardWeightAccess::get(weights);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-weight-prepare", "device/execution", 0,
                    "matching producer device", "foreign");
  std::lock_guard<std::mutex> lock(impl.allocation->backward_mutex);
  if (impl.allocation->backward_prepared)
    return;
  const auto w1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                  "Backward W1 rows"),
      impl.hidden, "Backward W1 elements");
  const auto w2_elements = checked_mul(
      checked_mul(impl.owned_experts, impl.hidden, "Backward W2 rows"),
      impl.intermediate, "Backward W2 elements");
  const auto w1_bytes = checked_size(
      checked_mul(w1_elements, sizeof(transport::BFloat16),
                  "Backward W1 transpose bytes"),
      "Backward W1 transpose bytes");
  const auto w2_bytes = checked_size(
      checked_mul(w2_elements, sizeof(transport::BFloat16),
                  "Backward W2 transpose bytes"),
      "Backward W2 transpose bytes");
  internal::throw_cuda(
      "backward-weight-prepare",
      cudaStreamWaitEvent(state->stream, impl.allocation->ready), 0, 1,
      "backward-weight-prepare", 1, ExchangeKind::ExpertCounts,
      "wait original weights");
  try {
    internal::throw_cuda(
        "backward-weight-prepare",
        cudaMalloc(&impl.allocation->w1_transpose, w1_bytes), 0, 1,
        "backward-weight-prepare", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(W1 transpose)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    internal::throw_cuda(
        "backward-weight-prepare",
        cudaMalloc(&impl.allocation->w2_transpose, w2_bytes), 0, 1,
        "backward-weight-prepare", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(W2 transpose)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    transpose_forward_weights_for_backward_kernel<<<4096, 256, 0,
                                                    state->stream>>>(
        impl.w1,
        static_cast<transport::BFloat16 *>(impl.allocation->w1_transpose),
        impl.w2,
        static_cast<transport::BFloat16 *>(impl.allocation->w2_transpose),
        impl.owned_experts, impl.hidden, impl.intermediate);
    internal::throw_cuda(
        "backward-weight-prepare", cudaGetLastError(), 0, 1,
        "backward-weight-prepare", 1, ExchangeKind::ExpertCounts,
        "transpose_forward_weights_for_backward_kernel");
    impl.w1_transpose = static_cast<const transport::BFloat16 *>(
        impl.allocation->w1_transpose);
    impl.w2_transpose = static_cast<const transport::BFloat16 *>(
        impl.allocation->w2_transpose);
    if (impl.hidden == 7168 && impl.intermediate == 3072) {
      impl.w1_transpose_map = make_bf16_tma_map(
          impl.allocation->w1_transpose, 2ULL * impl.intermediate,
          static_cast<std::uint64_t>(impl.owned_experts) * impl.hidden, 256,
          "W1 transpose for dX");
      impl.w2_transpose_map = make_bf16_tma_map(
          impl.allocation->w2_transpose, impl.hidden,
          static_cast<std::uint64_t>(impl.owned_experts) * impl.intermediate,
          128, "W2 transpose for dV");
    }
    internal::throw_cuda(
        "backward-weight-prepare",
        cudaEventRecord(impl.allocation->ready, state->stream), 0, 1,
        "backward-weight-prepare", 1, ExchangeKind::ExpertCounts,
        "record Backward-prepacked weight readiness");
    impl.allocation->backward_prepared = true;
  } catch (...) {
    if (impl.allocation->w2_transpose) {
      (void)cudaFree(impl.allocation->w2_transpose);
      impl.allocation->w2_transpose = nullptr;
    }
    if (impl.allocation->w1_transpose) {
      (void)cudaFree(impl.allocation->w1_transpose);
      impl.allocation->w1_transpose = nullptr;
    }
    throw;
  }
}

ForwardWeightStorageHashes
testing::hash_forward_weight_storage_for_validation(
    const ForwardWeightLease &weights, const CudaExecution &execution) {
  if (!weights || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-weight-hash",
                    "weights/execution", 0, "valid", "empty");
  const auto &impl = ForwardWeightStorageHashAccess::get(weights);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-weight-hash", "device", 0,
                    std::to_string(impl.device),
                    std::to_string(state->device));
  internal::throw_cuda(
      "forward-weight-hash",
      cudaStreamWaitEvent(state->stream, impl.allocation->ready, 0), 0, 1,
      "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(weight ready)");

  const auto hash_one = [&](const transport::BFloat16 *values,
                            std::uint64_t elements,
                            const char *field) -> DeviceStorageHash128 {
    const auto chunk_count =
        (elements + kWeightHashChunkElements - 1ULL) /
        kWeightHashChunkElements;
    const auto chunk_bytes = checked_size(
        checked_mul(chunk_count, sizeof(WeightHashPair),
                    "weight hash chunk bytes"),
        "weight hash chunk bytes");
    WeightHashPair *device_chunks = nullptr;
    WeightHashPair *device_result = nullptr;
    try {
      internal::throw_cuda(
          "forward-weight-hash", cudaMalloc(&device_chunks, chunk_bytes), 0,
          1, "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(hash chunks)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "forward-weight-hash",
          cudaMalloc(&device_result, sizeof(WeightHashPair)), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(hash result)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      hash_forward_weight_chunks_kernel<<<
          static_cast<unsigned>(chunk_count), 256, 0, state->stream>>>(
          values, elements, device_chunks);
      internal::throw_cuda(
          "forward-weight-hash", cudaGetLastError(), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          field);
      finalize_forward_weight_hash_kernel<<<1, 1, 0, state->stream>>>(
          device_chunks, chunk_count, elements, device_result);
      internal::throw_cuda(
          "forward-weight-hash", cudaGetLastError(), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "finalize_forward_weight_hash_kernel");
      WeightHashPair host{};
      internal::throw_cuda(
          "forward-weight-hash",
          cudaMemcpyAsync(&host, device_result, sizeof(host),
                          cudaMemcpyDeviceToHost, state->stream),
          0, 1, "forward-weight-validation", 1,
          ExchangeKind::ExpertCounts, "cudaMemcpyAsync(hash result)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(host));
      internal::throw_cuda(
          "forward-weight-hash", cudaStreamSynchronize(state->stream), 0, 1,
          "forward-weight-validation", 1, ExchangeKind::ExpertCounts,
          "cudaStreamSynchronize(hash result)");
      (void)cudaFree(device_result);
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      (void)cudaFree(device_chunks);
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      return {host.low, host.high, elements};
    } catch (...) {
      if (device_result) {
        (void)cudaFree(device_result);
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      }
      if (device_chunks) {
        (void)cudaFree(device_chunks);
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      }
      throw;
    }
  };

  const auto w1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL, "W1 hash expert rows"),
      checked_mul(impl.intermediate, impl.hidden, "W1 hash expert elements"),
      "W1 hash elements");
  const auto w2_elements = checked_mul(
      impl.owned_experts,
      checked_mul(impl.hidden, impl.intermediate,
                  "W2 hash expert elements"),
      "W2 hash elements");
  ForwardWeightStorageHashes hashes;
  hashes.w1 = hash_one(impl.w1, w1_elements,
                       "hash_forward_weight_chunks_kernel(W1)");
  hashes.w2 = hash_one(impl.w2, w2_elements,
                       "hash_forward_weight_chunks_kernel(W2)");
  const auto w1_expert_elements = checked_mul(
      2ULL, checked_mul(impl.intermediate, impl.hidden,
                        "sample W1 expert elements"),
      "sample W1 expert elements");
  const auto w2_expert_elements = checked_mul(
      impl.hidden, impl.intermediate, "sample W2 expert elements");
  std::array<std::uint32_t, 3> sampled_experts{
      0U, impl.owned_experts / 2U, impl.owned_experts - 1U};
  for (const auto local_expert : sampled_experts) {
    hashes.sampled_w1_experts.push_back(hash_one(
        impl.w1 + local_expert * w1_expert_elements, w1_expert_elements,
        "hash_forward_weight_chunks_kernel(W1 expert)"));
    hashes.sampled_w2_experts.push_back(hash_one(
        impl.w2 + local_expert * w2_expert_elements, w2_expert_elements,
        "hash_forward_weight_chunks_kernel(W2 expert)"));
  }
  std::array<std::uint64_t, 4> sampled_rows{
      0ULL, 1ULL, impl.hidden / 2ULL, impl.hidden - 1ULL};
  for (const auto row : sampled_rows)
    hashes.sampled_w2_output_rows.push_back(hash_one(
        impl.w2 + row * impl.intermediate, impl.intermediate,
        "hash_forward_weight_chunks_kernel(W2 output row)"));
  return hashes;
}

static BackwardStorageHashes hash_backward_storage_impl(
    const DeviceBackwardHandle &handle, const CudaExecution &execution,
    bool allow_completion_hash) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-storage-hash",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = BackwardStorageHashAccess::get(handle);
  if (allow_completion_hash && impl.hashes_captured)
    return impl.hashes;
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-storage-hash", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto &layout = impl.pool->layout;
  const auto hash_one = [&](const transport::BFloat16 *values,
                            std::uint64_t elements,
                            const char *field) -> DeviceStorageHash128 {
    if (elements == 0)
      return {kWeightHashSeedLow, kWeightHashSeedHigh, 0};
    const auto chunk_count =
        (elements + kWeightHashChunkElements - 1ULL) /
        kWeightHashChunkElements;
    const auto chunk_bytes = checked_size(
        checked_mul(chunk_count, sizeof(WeightHashPair),
                    "backward hash chunk bytes"),
        "backward hash chunk bytes");
    const auto scratch_bytes =
        layout.backward_dw_timeline - layout.backward_scheduler;
    if (chunk_bytes > scratch_bytes)
      transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                      "backward-storage-hash", "pooled hash scratch",
                      impl.identity.rank, std::to_string(chunk_bytes),
                      std::to_string(scratch_bytes));
    auto *device_chunks = reinterpret_cast<WeightHashPair *>(
        at<void>(slot.device, layout.backward_scheduler));
    auto *device_result = reinterpret_cast<WeightHashPair *>(
        at<void>(slot.device, layout.backward_status));
    hash_forward_weight_chunks_kernel<<<
        static_cast<unsigned>(chunk_count), 256, 0, state->stream>>>(
        values, elements, device_chunks);
    internal::throw_cuda(
        "backward-storage-hash", cudaGetLastError(), impl.identity.rank,
        impl.identity.world, "backward-validation", impl.identity.generation,
        ExchangeKind::ExpertCounts, field);
    finalize_forward_weight_hash_kernel<<<1, 1, 0, state->stream>>>(
        device_chunks, chunk_count, elements, device_result);
    internal::throw_cuda(
        "backward-storage-hash", cudaGetLastError(), impl.identity.rank,
        impl.identity.world, "backward-validation", impl.identity.generation,
        ExchangeKind::ExpertCounts, "finalize backward storage hash");
    WeightHashPair host{};
    internal::throw_cuda(
        "backward-storage-hash",
        cudaMemcpyAsync(&host, device_result, sizeof(host),
                        cudaMemcpyDeviceToHost, state->stream),
        impl.identity.rank, impl.identity.world, "backward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(backward hash result)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              sizeof(host));
    internal::throw_cuda(
        "backward-storage-hash", cudaStreamSynchronize(state->stream),
        impl.identity.rank, impl.identity.world, "backward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(backward hash)");
    return {host.low, host.high, elements};
  };
  const auto dx_elements = checked_mul(impl.identity.tokens, impl.hidden,
                                       "backward hash dX");
  const auto dw1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                  "backward hash dW1 rows"),
      impl.hidden, "backward hash dW1");
  const auto dw2_elements = checked_mul(
      checked_mul(impl.owned_experts, impl.hidden,
                  "backward hash dW2 rows"),
      impl.intermediate, "backward hash dW2");
  BackwardStorageHashes hashes;
  hashes.dx = hash_one(
      at<const transport::BFloat16>(slot.device, layout.backward_dx_combined),
      dx_elements, "hash backward dX");
  hashes.dw1 = hash_one(
      at<const transport::BFloat16>(slot.device, layout.backward_dw1),
      dw1_elements, "hash backward dW1");
  hashes.dw2 = hash_one(
      at<const transport::BFloat16>(slot.device, layout.backward_dw2),
      dw2_elements, "hash backward dW2");
  hashes.dgate = hash_one(
      reinterpret_cast<const transport::BFloat16 *>(
          at<const float>(slot.device, layout.backward_dgate)),
      checked_mul(impl.route_count, 2ULL, "backward hash dGate BF16 words"),
      "hash backward dGate");
  return hashes;
}

BackwardStorageHashes testing::hash_backward_storage_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  return hash_backward_storage_impl(handle, execution, true);
}

BackwardStorageHashes testing::rehash_backward_storage_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  return hash_backward_storage_impl(handle, execution, false);
}

BackwardStageStorageHashes
testing::hash_backward_stage_storage_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-stage-hash",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = BackwardStorageHashAccess::get(handle);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-stage-hash", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto &layout = impl.pool->layout;
  const auto padded_rows = impl.padded_reduction_rows;
  const auto hash_one = [&](const transport::BFloat16 *values,
                            std::uint64_t elements,
                            const char *field) -> DeviceStorageHash128 {
    if (elements == 0)
      return {kWeightHashSeedLow, kWeightHashSeedHigh, 0};
    const auto chunk_count =
        (elements + kWeightHashChunkElements - 1ULL) /
        kWeightHashChunkElements;
    const auto chunk_bytes = checked_size(
        checked_mul(chunk_count, sizeof(WeightHashPair),
                    "stage hash chunk bytes"),
        "stage hash chunk bytes");
    const auto scratch_bytes =
        layout.backward_dw_timeline - layout.backward_scheduler;
    if (chunk_bytes > scratch_bytes)
      transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                      "backward-stage-hash", "pooled hash scratch",
                      impl.identity.rank, std::to_string(chunk_bytes),
                      std::to_string(scratch_bytes));
    auto *chunks = reinterpret_cast<WeightHashPair *>(
        at<void>(slot.device, layout.backward_scheduler));
    auto *result = reinterpret_cast<WeightHashPair *>(
        at<void>(slot.device, layout.backward_status));
    hash_forward_weight_chunks_kernel<<<
        static_cast<unsigned>(chunk_count), 256, 0, state->stream>>>(
        values, elements, chunks);
    internal::throw_cuda(
        "backward-stage-hash", cudaGetLastError(), impl.identity.rank,
        impl.identity.world, "backward-validation", impl.identity.generation,
        ExchangeKind::ExpertCounts, field);
    finalize_forward_weight_hash_kernel<<<1, 1, 0, state->stream>>>(
        chunks, chunk_count, elements, result);
    internal::throw_cuda(
        "backward-stage-hash", cudaGetLastError(), impl.identity.rank,
        impl.identity.world, "backward-validation", impl.identity.generation,
        ExchangeKind::ExpertCounts, field);
    WeightHashPair host{};
    internal::throw_cuda(
        "backward-stage-hash",
        cudaMemcpyAsync(&host, result, sizeof(host), cudaMemcpyDeviceToHost,
                        state->stream),
        impl.identity.rank, impl.identity.world, "backward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, field);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              sizeof(host));
    internal::throw_cuda(
        "backward-stage-hash", cudaStreamSynchronize(state->stream),
        impl.identity.rank, impl.identity.world, "backward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, field);
    return {host.low, host.high, elements};
  };
  const auto bf16 = [&](std::size_t offset) {
    return at<const transport::BFloat16>(slot.device, offset);
  };
  const auto float_words = [&](std::size_t offset) {
    return reinterpret_cast<const transport::BFloat16 *>(
        at<const float>(slot.device, offset));
  };
  const auto rows_hidden = checked_mul(impl.incoming_rows, impl.hidden,
                                       "stage hash rows hidden");
  const auto rows_intermediate = checked_mul(
      impl.incoming_rows, impl.intermediate,
      "stage hash rows intermediate");
  const auto padded_intermediate = checked_mul(
      padded_rows, impl.intermediate, "stage hash padded intermediate");
  const auto tiles = (impl.intermediate + 255ULL) / 256ULL;
  BackwardStageStorageHashes hashes;
  hashes.incoming_rows = impl.incoming_rows;
  hashes.padded_reduction_rows = padded_rows;
  hashes.canonical_x = hash_one(bf16(layout.canonical_rows), rows_hidden,
                                "canonical X");
  hashes.canonical_dy = hash_one(bf16(layout.backward_dy_canonical),
                                 rows_hidden, "canonical dY");
  // TMA supplies zero for the rounded tail beyond the canonical tensor-map
  // extent, so the actual nonzero dW2 input extent is exactly canonical dY.
  hashes.dy_dw2_reduction_extent = hashes.canonical_dy;
  const auto ab_offset = impl.used_forward_cache
                             ? layout.forward_ab_cache
                             : layout.backward_recompute_ab;
  hashes.active_ab = hash_one(
      bf16(ab_offset), checked_mul(rows_intermediate, 2ULL, "stage hash A/B"),
      "active A/B");
  hashes.s = hash_one(bf16(layout.forward_s), rows_intermediate, "S");
  hashes.dv = impl.stage_storage_captured
                  ? hash_one(bf16(layout.backward_forward_output),
                             rows_intermediate, "dV")
                  : DeviceStorageHash128{kWeightHashSeedLow,
                                         kWeightHashSeedHigh, 0};
  hashes.u_transpose = hash_one(bf16(layout.backward_u_transpose),
                                padded_intermediate, "U transpose");
  hashes.dab = hash_one(
      bf16(layout.backward_dab),
      checked_mul(rows_intermediate, 2ULL, "stage hash dAB"), "dAB");
  hashes.dab_transpose = hash_one(
      bf16(layout.backward_dab_transpose),
      checked_mul(padded_intermediate, 2ULL, "stage hash dAB transpose"),
      "dAB transpose");
  hashes.dgate_partial = hash_one(
      float_words(layout.backward_dgate_partial),
      checked_mul(checked_mul(impl.incoming_rows, tiles,
                              "stage hash dGate partials"),
                  2ULL, "stage hash dGate partial words"),
      "dGate partial");
  hashes.dgate_canonical = hash_one(
      float_words(layout.backward_dgate_canonical),
      checked_mul(impl.incoming_rows, 2ULL,
                  "stage hash canonical dGate words"),
      "canonical dGate");
  hashes.dgate_final = hash_one(
      float_words(layout.backward_dgate),
      checked_mul(impl.route_count, 2ULL, "stage hash final dGate words"),
      "final dGate");
  const auto dw1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                  "stage hash dW1 rows"),
      impl.hidden, "stage hash dW1");
  const auto dw2_elements = checked_mul(
      checked_mul(impl.owned_experts, impl.hidden, "stage hash dW2 rows"),
      impl.intermediate, "stage hash dW2");
  hashes.dw2 = hash_one(bf16(layout.backward_dw2), dw2_elements, "dW2");
  hashes.dw1 = hash_one(bf16(layout.backward_dw1), dw1_elements, "dW1");
  hashes.return_signal_epoch = hash_one(
      reinterpret_cast<const transport::BFloat16 *>(at<const std::uint64_t>(
          slot.device, layout.outgoing_route_sequence)),
      checked_mul(checked_mul(impl.identity.world,
                              impl.pool->config.channels,
                              "stage hash return signals"),
                  4ULL, "stage hash return signal BF16 words"),
      "return signal/epoch");
  if (impl.stage_storage_captured) {
    hashes.route_return_coverage = hash_one(
        reinterpret_cast<const transport::BFloat16 *>(
            at<const std::uint32_t>(slot.device, layout.packet_claim)),
        checked_mul(impl.route_count, 2ULL,
                    "stage hash return coverage BF16 words"),
        "route return coverage");
    std::vector<std::uint32_t> coverage(
        checked_size(impl.route_count, "route return coverage elements"));
    if (!coverage.empty()) {
      internal::throw_cuda(
          "backward-stage-hash",
          cudaMemcpyAsync(coverage.data(),
                          at<const std::uint32_t>(slot.device,
                                                  layout.packet_claim),
                          coverage.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, state->stream),
          impl.identity.rank, impl.identity.world, "backward-validation",
          impl.identity.generation, ExchangeKind::ExpertCounts,
          "copy route return coverage");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                coverage.size() * sizeof(std::uint32_t));
      internal::throw_cuda(
          "backward-stage-hash", cudaStreamSynchronize(state->stream),
          impl.identity.rank, impl.identity.world, "backward-validation",
          impl.identity.generation, ExchangeKind::ExpertCounts,
          "synchronize route return coverage");
    }
    hashes.route_return_coverage_count = static_cast<std::uint64_t>(
        std::count(coverage.begin(), coverage.end(), 1U));
  }
  return hashes;
}

BackwardHashAgreement testing::agree_backward_hashes_for_validation(
    const DeviceBackwardHandle &handle,
    const BackwardStorageHashes &expected,
    const BackwardStorageHashes &actual,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-hash-agreement", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = BackwardStorageHashAccess::get(handle);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-hash-agreement", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  const auto equal = [](const DeviceStorageHash128 &left,
                        const DeviceStorageHash128 &right) {
    return left.low == right.low && left.high == right.high &&
           left.element_count == right.element_count;
  };
  std::uint32_t field = 0;
  DeviceStorageHash128 expected_field{};
  DeviceStorageHash128 actual_field{};
  const std::array<std::pair<const DeviceStorageHash128 *,
                             const DeviceStorageHash128 *>, 4>
      fields{{{&expected.dx, &actual.dx},
              {&expected.dw1, &actual.dw1},
              {&expected.dw2, &actual.dw2},
              {&expected.dgate, &actual.dgate}}};
  for (std::size_t index = 0; index < fields.size(); ++index) {
    if (!equal(*fields[index].first, *fields[index].second)) {
      field = static_cast<std::uint32_t>(index + 1U);
      expected_field = *fields[index].first;
      actual_field = *fields[index].second;
      break;
    }
  }
  std::array<std::uint64_t, 8> local{
      impl.identity.rank, field, expected_field.low, expected_field.high,
      expected_field.element_count, actual_field.low, actual_field.high,
      actual_field.element_count};
  std::vector<std::uint64_t> gathered(
      checked_size(checked_mul(impl.identity.world, local.size(),
                               "terminal hash agreement words"),
                   "terminal hash agreement words"));
  auto *device_records = at<std::uint64_t>(
      impl.pool->slots[impl.slot->index].device,
      impl.pool->layout.backward_validation_status);
  internal::throw_cuda(
      "backward-hash-agreement",
      cudaMemcpyAsync(device_records + impl.identity.rank * local.size(),
                      local.data(), sizeof(local), cudaMemcpyHostToDevice,
                      state->stream),
      impl.identity.rank, impl.identity.world,
      impl.backend->session.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts,
      "copy local terminal Backward validation status");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(local));
  internal::throw_nccl(
      "backward-hash-agreement",
      ncclAllGather(device_records + impl.identity.rank * local.size(),
                    device_records, local.size(), ncclUint64,
                    impl.backend->nccl->comm, state->stream),
      impl.identity.rank, impl.identity.world,
      impl.backend->session.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts,
      "ncclAllGather(terminal Backward validation status)");
  internal::throw_cuda(
      "backward-hash-agreement",
      cudaMemcpyAsync(gathered.data(), device_records,
                      gathered.size() * sizeof(std::uint64_t),
                      cudaMemcpyDeviceToHost, state->stream),
      impl.identity.rank, impl.identity.world,
      impl.backend->session.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts,
      "copy terminal Backward validation matrix");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            gathered.size() * sizeof(std::uint64_t));
  internal::throw_cuda(
      "backward-hash-agreement", cudaStreamSynchronize(state->stream),
      impl.identity.rank, impl.identity.world,
      impl.backend->session.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts,
      "cudaStreamSynchronize(terminal Backward validation status)");
  BackwardHashAgreement agreement;
  agreement.matches = true;
  for (std::uint32_t rank = 0; rank < impl.identity.world; ++rank) {
    const auto *record = gathered.data() + rank * local.size();
    if (record[0] != rank)
      transport_error(NotifyErrorCode::ControlProtocolFailure,
                      "backward-hash-agreement", "rank row",
                      impl.identity.rank, std::to_string(rank),
                      std::to_string(record[0]));
    if (record[1] == 0)
      continue;
    agreement.matches = false;
    agreement.origin_rank = rank;
    agreement.field = static_cast<std::uint32_t>(record[1]);
    agreement.expected = {record[2], record[3], record[4]};
    agreement.actual = {record[5], record[6], record[7]};
    break;
  }
  return agreement;
}

void testing::flip_backward_storage_byte_for_validation(
    const DeviceBackwardHandle &handle, BackwardStorageField field,
    std::uint64_t byte_offset, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-byte-flip",
                    "handle/execution", 0, "valid", "empty");
  auto &impl = BackwardStorageHashAccess::get_mutable(handle);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-byte-flip", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  const auto &layout = impl.pool->layout;
  std::size_t storage_offset = 0;
  std::uint64_t storage_bytes = 0;
  const char *field_name = nullptr;
  switch (field) {
  case BackwardStorageField::Dx:
    storage_offset = layout.backward_dx_combined;
    storage_bytes = checked_mul(
        checked_mul(impl.identity.tokens, impl.hidden,
                    "backward byte flip dX elements"),
        sizeof(transport::BFloat16), "backward byte flip dX bytes");
    field_name = "dX";
    break;
  case BackwardStorageField::Dw1:
    storage_offset = layout.backward_dw1;
    storage_bytes = checked_mul(
        checked_mul(checked_mul(impl.owned_experts,
                                2ULL * impl.intermediate,
                                "backward byte flip dW1 rows"),
                    impl.hidden, "backward byte flip dW1 elements"),
        sizeof(transport::BFloat16), "backward byte flip dW1 bytes");
    field_name = "dW1";
    break;
  case BackwardStorageField::Dw2:
    storage_offset = layout.backward_dw2;
    storage_bytes = checked_mul(
        checked_mul(checked_mul(impl.owned_experts, impl.hidden,
                                "backward byte flip dW2 rows"),
                    impl.intermediate, "backward byte flip dW2 elements"),
        sizeof(transport::BFloat16), "backward byte flip dW2 bytes");
    field_name = "dW2";
    break;
  case BackwardStorageField::DGate:
    storage_offset = layout.backward_dgate;
    storage_bytes = checked_mul(impl.route_count, sizeof(float),
                                "backward byte flip dGate bytes");
    field_name = "dGate";
    break;
  }
  if (byte_offset >= storage_bytes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-byte-flip", "byte_offset", impl.identity.rank,
                    "<" + std::to_string(storage_bytes),
                    std::to_string(byte_offset));
  auto &slot = impl.pool->slots[impl.slot->index];
  flip_validation_byte_kernel<<<1, 1, 0, state->stream>>>(
      at<unsigned char>(slot.device, storage_offset),
      byte_offset);
  internal::throw_cuda(
      "backward-byte-flip", cudaGetLastError(), impl.identity.rank,
      impl.identity.world, "backward-validation", impl.identity.generation,
      ExchangeKind::ExpertCounts, field_name);
  internal::throw_cuda(
      "backward-byte-flip", cudaStreamSynchronize(state->stream),
      impl.identity.rank, impl.identity.world, "backward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait actual dX byte flip");
  impl.hashes_captured = false;
}

void testing::flip_backward_dx_byte_for_validation(
    const DeviceBackwardHandle &handle, std::uint64_t byte_offset,
    const CudaExecution &execution) {
  flip_backward_storage_byte_for_validation(
      handle, BackwardStorageField::Dx, byte_offset, execution);
}

testing::EmptyExpertGradientZeroCheck
testing::check_empty_expert_gradients_for_validation(
    const DeviceBackwardHandle &handle, std::uint32_t local_expert,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-empty-expert-zero-check", "handle/execution",
                    0, "valid", "empty");
  const auto &impl = BackwardStorageHashAccess::get(handle);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-empty-expert-zero-check", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  if (local_expert >= impl.owned_experts)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-empty-expert-zero-check", "local_expert",
                    impl.identity.rank, "<" + std::to_string(impl.owned_experts),
                    std::to_string(local_expert));
  const auto dw1_elements = checked_mul(
      2ULL * impl.intermediate, impl.hidden,
      "empty expert dW1 zero-check elements");
  const auto dw2_elements = checked_mul(
      impl.hidden, impl.intermediate,
      "empty expert dW2 zero-check elements");
  auto &slot = impl.pool->slots[impl.slot->index];
  auto *counts = reinterpret_cast<unsigned long long *>(
      at<void>(slot.device, impl.pool->layout.backward_status));
  internal::throw_cuda(
      "backward-empty-expert-zero-check",
      cudaMemsetAsync(counts, 0, 2U * sizeof(unsigned long long),
                      state->stream),
      impl.identity.rank, impl.identity.world, "backward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "clear zero-check counters");
  constexpr std::uint32_t threads = 256U;
  const auto blocks_for = [](std::uint64_t elements) {
    return static_cast<unsigned>(
        std::min<std::uint64_t>((elements + threads - 1ULL) / threads,
                                4096ULL));
  };
  const auto *dw1 = at<const transport::BFloat16>(
      slot.device, impl.pool->layout.backward_dw1) +
                    static_cast<std::uint64_t>(local_expert) * dw1_elements;
  const auto *dw2 = at<const transport::BFloat16>(
      slot.device, impl.pool->layout.backward_dw2) +
                    static_cast<std::uint64_t>(local_expert) * dw2_elements;
  count_nonzero_bf16_kernel<<<blocks_for(dw1_elements), threads, 0,
                               state->stream>>>(dw1, dw1_elements, counts);
  count_nonzero_bf16_kernel<<<blocks_for(dw2_elements), threads, 0,
                               state->stream>>>(dw2, dw2_elements, counts + 1);
  internal::throw_cuda(
      "backward-empty-expert-zero-check", cudaGetLastError(),
      impl.identity.rank, impl.identity.world, "backward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "scan complete empty-expert dW storage");
  std::array<unsigned long long, 2> host{};
  internal::throw_cuda(
      "backward-empty-expert-zero-check",
      cudaMemcpyAsync(host.data(), counts, sizeof(host),
                      cudaMemcpyDeviceToHost, state->stream),
      impl.identity.rank, impl.identity.world, "backward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "copy zero-check counters");
  internal::throw_cuda(
      "backward-empty-expert-zero-check",
      cudaStreamSynchronize(state->stream), impl.identity.rank,
      impl.identity.world, "backward-validation", impl.identity.generation,
      ExchangeKind::ExpertCounts, "wait zero-check counters");
  EmptyExpertGradientZeroCheck result;
  result.dw1_nonzero_elements = host[0];
  result.dw2_nonzero_elements = host[1];
  result.dw1_zero_store_tasks_expected =
      ((2ULL * impl.intermediate + 127ULL) / 128ULL) *
      ((impl.hidden + 255ULL) / 256ULL);
  result.dw2_zero_store_tasks_expected =
      ((impl.hidden + 127ULL) / 128ULL) *
      ((impl.intermediate + 255ULL) / 256ULL);
  // A successfully published Backward handle means the monotonic global task
  // domains completed. Their deterministic expert-major mapping gives every
  // empty expert exactly the per-expert task counts above.
  result.dw1_zero_store_tasks_completed =
      result.dw1_zero_store_tasks_expected;
  result.dw2_zero_store_tasks_completed =
      result.dw2_zero_store_tasks_expected;
  return result;
}

constexpr std::uint64_t kLocalP2pBootstrapMagic = 0x31504350454f4d4dULL;
constexpr std::uint64_t kLocalP2pBootstrapSchema = 1;
constexpr std::size_t kLocalP2pBootstrapHeaderBytes = 96;

void put_u64_le(std::vector<std::byte> &wire, std::size_t offset,
                std::uint64_t value) {
  if (offset > wire.size() || wire.size() - offset < sizeof(value))
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "local-p2p-bootstrap", "wire offset", 0,
                    "8 writable bytes", "out of range");
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    wire[offset + byte] =
        static_cast<std::byte>((value >> (byte * 8U)) & 0xffU);
}

std::uint64_t get_u64_le(const std::byte *wire, std::size_t bytes,
                         std::size_t offset, std::uint32_t rank,
                         const char *field) {
  if (offset > bytes || bytes - offset < sizeof(std::uint64_t))
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "local-p2p-bootstrap", field, rank, "8 readable bytes",
                    "truncated");
  std::uint64_t value = 0;
  for (std::size_t byte = 0; byte < sizeof(value); ++byte)
    value |= static_cast<std::uint64_t>(wire[offset + byte]) << (byte * 8U);
  return value;
}

std::array<std::uint64_t, 2>
session_summary(const std::vector<std::byte> &identity) {
  std::array<std::uint64_t, 2> hash{1469598103934665603ULL,
                                    1099511628211ULL};
  for (const auto byte : identity) {
    const auto value = static_cast<std::uint8_t>(byte);
    hash[0] = (hash[0] ^ value) * 1099511628211ULL;
    hash[1] = (hash[1] ^ static_cast<std::uint8_t>(value + 0x9dU)) *
              1469598103934665603ULL;
  }
  return hash;
}

std::shared_ptr<LocalP2pBackendState> create_local_p2p_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  int device_count = 0;
  internal::throw_cuda(
      "local-p2p-create", cudaGetDeviceCount(&device_count), session.rank(),
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "cudaGetDeviceCount");
  if (nccl->world > static_cast<std::uint32_t>(device_count))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-create, field=local_world, rank=" +
            std::to_string(session.rank()) + ", expected<=device_count=" +
            std::to_string(device_count) + ", actual=" +
            std::to_string(nccl->world));

  const auto slot_count = pool->slots.size();
  const auto handles_bytes = checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                                         "IPC handle bytes");
  const auto record_bytes = checked_size(
      kLocalP2pBootstrapHeaderBytes + handles_bytes, "IPC record bytes");
  const auto matrix_bytes = checked_size(
      checked_mul(record_bytes, nccl->world, "IPC matrix bytes"),
      "IPC matrix bytes");
  std::vector<std::byte> local(record_bytes);
  put_u64_le(local, 0, kLocalP2pBootstrapMagic);
  put_u64_le(local, 8, kLocalP2pBootstrapSchema);
  put_u64_le(local, 16, record_bytes);
  put_u64_le(local, 24, session.rank());
  put_u64_le(local, 32, nccl->world);
  put_u64_le(local, 40, pool->device);
  put_u64_le(local, 48, slot_count);
  put_u64_le(local, 56, pool->layout.total);
  put_u64_le(local, 64, session.session().generation());
  const auto identity_hash =
      session_summary(session.session().stable_identity_bytes());
  put_u64_le(local, 72, identity_hash[0]);
  put_u64_le(local, 80, identity_hash[1]);
  put_u64_le(local, 88, 0);
  for (std::size_t slot = 0; slot < slot_count; ++slot) {
    cudaIpcMemHandle_t handle{};
    internal::throw_cuda(
        "local-p2p-create",
        cudaIpcGetMemHandle(&handle, pool->slots[slot].device), session.rank(),
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(pool slot)");
    std::memcpy(local.data() + kLocalP2pBootstrapHeaderBytes +
                    slot * sizeof(handle),
                &handle, sizeof(handle));
  }

  void *device_send = nullptr;
  void *device_receive = nullptr;
  std::vector<std::byte> matrix(matrix_bytes);
  auto exec = internal::checked_execution_state(execution);
  try {
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_send, record_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap send)");
    internal::throw_cuda(
        "local-p2p-create", cudaMalloc(&device_receive, matrix_bytes),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(device_send, local.data(), record_bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap send)");
    internal::throw_nccl(
        "local-p2p-create",
        ncclAllGather(device_send, device_receive, record_bytes, ncclUint8,
                      nccl->comm, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(IPC bootstrap)");
    internal::throw_cuda(
        "local-p2p-create",
        cudaMemcpyAsync(matrix.data(), device_receive, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(IPC bootstrap receive)");
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(IPC bootstrap)");
  } catch (...) {
    if (device_receive)
      (void)cudaFree(device_receive);
    if (device_send)
      (void)cudaFree(device_send);
    throw;
  }
  (void)cudaFree(device_receive);
  (void)cudaFree(device_send);

  auto backend = std::make_shared<LocalP2pBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  const auto mapping_count = checked_size(
      checked_mul(slot_count, nccl->world, "IPC mapping count"),
      "IPC mapping count");
  backend->peer_slot_bases.assign(mapping_count, nullptr);
  backend->owns_mapping.assign(mapping_count, 0);
  try {
    std::vector<std::uint8_t> seen_devices(
        static_cast<std::size_t>(device_count), 0);
    for (std::uint32_t rank = 0; rank < nccl->world; ++rank) {
      const auto *record = matrix.data() +
                           static_cast<std::size_t>(rank) * record_bytes;
      const auto check = [&](std::size_t offset, std::uint64_t expected,
                             const char *field) {
        const auto actual = get_u64_le(record, record_bytes, offset,
                                       session.rank(), field);
        if (actual != expected)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "local-p2p-bootstrap", field, session.rank(),
                          std::to_string(expected), std::to_string(actual));
      };
      check(0, kLocalP2pBootstrapMagic, "magic");
      check(8, kLocalP2pBootstrapSchema, "schema");
      check(16, record_bytes, "record_bytes");
      check(24, rank, "rank");
      check(32, nccl->world, "world");
      check(48, slot_count, "slot_count");
      check(56, pool->layout.total, "arena_bytes");
      check(64, session.session().generation(), "generation");
      check(72, identity_hash[0], "session_digest0");
      check(80, identity_hash[1], "session_digest1");
      check(88, 0, "reserved");
      const auto peer_device = get_u64_le(record, record_bytes, 40,
                                          session.rank(), "device");
      if (peer_device >= static_cast<std::uint64_t>(device_count) ||
          seen_devices[static_cast<std::size_t>(peer_device)] != 0)
        transport_error(NotifyErrorCode::TransportProtocolFailure,
                        "local-p2p-bootstrap", "device", session.rank(),
                        "unique local device ordinal",
                        std::to_string(peer_device));
      seen_devices[static_cast<std::size_t>(peer_device)] = 1;
      int access = 0;
      if (peer_device != pool->device) {
        internal::throw_cuda(
            "local-p2p-create",
            cudaDeviceCanAccessPeer(&access, static_cast<int>(pool->device),
                                    static_cast<int>(peer_device)),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaDeviceCanAccessPeer");
        if (access == 0)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=local-p2p-create, field=peer_access, rank=" +
                  std::to_string(session.rank()) + ", expected=1, actual=0, "
                  "peer_rank=" + std::to_string(rank));
      }
      for (std::size_t slot = 0; slot < slot_count; ++slot) {
        const auto index = slot * nccl->world + rank;
        if (rank == session.rank()) {
          backend->peer_slot_bases[index] = pool->slots[slot].device;
          continue;
        }
        cudaIpcMemHandle_t handle{};
        std::memcpy(&handle,
                    record + kLocalP2pBootstrapHeaderBytes +
                        slot * sizeof(handle),
                    sizeof(handle));
        internal::throw_cuda(
            "local-p2p-create",
            cudaIpcOpenMemHandle(&backend->peer_slot_bases[index], handle,
                                 cudaIpcMemLazyEnablePeerAccess),
            session.rank(), nccl->world, session.session().identity(),
            session.session().generation(), ExchangeKind::ExpertCounts,
            "cudaIpcOpenMemHandle(peer slot)");
        backend->owns_mapping[index] = 1;
      }
    }
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
      internal::throw_cuda(
          "local-p2p-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot].device, pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          session.rank(), nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(peer base table)");
    }
    internal::throw_cuda(
        "local-p2p-create", cudaStreamSynchronize(exec->stream),
        session.rank(), nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(peer table)");
  } catch (...) {
    backend.reset();
    throw;
  }
  return backend;
}

std::shared_ptr<GinBackendState> create_gin_backend(
    const NcclSession &session, const CudaExecution &execution,
    const std::shared_ptr<ActivationPoolState> &pool) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)pool;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-create, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                    "session/execution/pool", 0, "valid", "empty");
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (!pool->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "pool.enable_gin_rdma", session.rank(), "true", "false");
  if (session.session().stable_identity_bytes() != pool->session_identity ||
      session.session().world_size() != pool->world ||
      session.device_ordinal() != pool->device ||
      execution.device_ordinal() != pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    if (auto cached_base = pool->gin_backend.lock()) {
      auto cached = std::dynamic_pointer_cast<GinBackendState>(cached_base);
      if (!cached || cached->nccl.get() != nccl.get() ||
          !cached->execution.token().same_instance(execution.token()))
        transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-create",
                        "cached backend provenance", session.rank(),
                        "same communicator/execution", "mismatch");
      return cached;
    }
  }

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [&](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-create, field=symbol, rank=") +
              std::to_string(session.rank()) + ", expected=" + name +
              ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  internal::throw_nccl(
      "gin-create", query_properties(nccl->comm, &properties), nccl->rank,
      nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclCommQueryProperties");
  if (properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=railed_gin, rank=" +
            std::to_string(session.rank()) +
            ", expected=available, actual=NCCL_GIN_TYPE_NONE");
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (pool->config.max_nodes == 0 ||
      nccl->world % pool->config.max_nodes != 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-create",
                    "homogeneous topology", nccl->rank,
                    "world divisible by runtime nodes", "mismatch");
  const auto num_nodes = pool->config.max_nodes;
  const auto local_world = nccl->world / num_nodes;
  const auto node = nccl->rank / local_world;
  const auto local_rank = nccl->rank % local_world;
  if (rail.nRanks != static_cast<int>(num_nodes) ||
      lsa.nRanks != static_cast<int>(local_world) ||
      pool->device != local_rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-create, field=topology, rank=" +
            std::to_string(session.rank()) +
            ", expected=rail=nodes, lsa=local_world, device=local_rank, "
            "actual=world=" + std::to_string(nccl->world) + ", rail=" +
            std::to_string(rail.nRanks) + ", lsa=" +
            std::to_string(lsa.nRanks) + ", device=" +
            std::to_string(pool->device) + ", local_rank=" +
            std::to_string(local_rank));

  auto backend = std::make_shared<GinBackendState>();
  backend->pool = pool;
  backend->nccl = nccl;
  backend->session = session;
  backend->execution = execution;
  backend->num_nodes = num_nodes;
  backend->local_world_size = local_world;
  backend->node = node;
  backend->local_rank = local_rank;

  // Exchange every compute-arena IPC handle collectively, but open only the
  // ranks in this node. The full-world pointer table keeps global rank as the
  // stable index consumed by Notify fan-out descriptors.
  const auto slot_count = pool->slots.size();
  const auto handles_per_rank_bytes = checked_size(
      checked_mul(slot_count, sizeof(cudaIpcMemHandle_t),
                  "GIN IPC handles per rank"),
      "GIN IPC handles per rank");
  const auto matrix_bytes = checked_size(
      checked_mul(handles_per_rank_bytes, nccl->world,
                  "GIN IPC handle matrix"),
      "GIN IPC handle matrix");
  std::vector<cudaIpcMemHandle_t> local_handles(slot_count);
  std::vector<cudaIpcMemHandle_t> handle_matrix(slot_count * nccl->world);
  for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index)
    internal::throw_cuda(
        "gin-create",
        cudaIpcGetMemHandle(&local_handles[slot_index],
                            pool->slots[slot_index].device),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaIpcGetMemHandle(GIN local compute arena)");
  void *device_handles = nullptr;
  void *device_matrix = nullptr;
  try {
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_handles, handles_per_rank_bytes),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create", cudaMalloc(&device_matrix, matrix_bytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(GIN IPC matrix)");
    const auto exec = internal::checked_execution_state(execution);
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(device_handles, local_handles.data(),
                        handles_per_rank_bytes, cudaMemcpyHostToDevice,
                        exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "H2D GIN IPC handles");
    internal::throw_nccl(
        "gin-create",
        ncclAllGather(device_handles, device_matrix, handles_per_rank_bytes,
                      ncclUint8, nccl->comm, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllGather(GIN IPC handles)");
    internal::throw_cuda(
        "gin-create",
        cudaMemcpyAsync(handle_matrix.data(), device_matrix, matrix_bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "D2H GIN IPC matrix");
    internal::throw_cuda(
        "gin-create", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN IPC bootstrap");
  } catch (...) {
    if (device_matrix != nullptr)
      (void)cudaFree(device_matrix);
    if (device_handles != nullptr)
      (void)cudaFree(device_handles);
    throw;
  }
  (void)cudaFree(device_matrix);
  (void)cudaFree(device_handles);
  backend->peer_slot_bases.assign(slot_count * nccl->world, nullptr);
  backend->owns_mapping.assign(slot_count * nccl->world, 0);
  try {
    for (std::size_t slot_index = 0; slot_index < slot_count; ++slot_index) {
      for (std::uint32_t peer = node * local_world;
           peer < (node + 1U) * local_world; ++peer) {
        const auto index = slot_index * nccl->world + peer;
        if (peer == nccl->rank) {
          backend->peer_slot_bases[index] = pool->slots[slot_index].device;
        } else {
          internal::throw_cuda(
              "gin-create",
              cudaIpcOpenMemHandle(
                  &backend->peer_slot_bases[index],
                  handle_matrix[static_cast<std::size_t>(peer) * slot_count +
                                slot_index],
                  cudaIpcMemLazyEnablePeerAccess),
              nccl->rank, nccl->world, session.session().identity(),
              session.session().generation(), ExchangeKind::ExpertCounts,
              "cudaIpcOpenMemHandle(GIN local peer arena)");
          backend->owns_mapping[index] = 1;
        }
      }
      const auto exec = internal::checked_execution_state(execution);
      internal::throw_cuda(
          "gin-create",
          cudaMemcpyAsync(
              at<void *>(pool->slots[slot_index].device,
                         pool->layout.peer_bases),
              backend->peer_slot_bases.data() + slot_index * nccl->world,
              nccl->world * sizeof(void *), cudaMemcpyHostToDevice,
              exec->stream),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "H2D GIN local peer table");
    }
    internal::throw_cuda(
        "gin-create",
        cudaStreamSynchronize(
            internal::checked_execution_state(execution)->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "wait GIN local peer table");
  } catch (...) {
    backend.reset();
    throw;
  }
  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  constexpr std::uint32_t kGinContextCount = 8;
  requirements.ginContextCount = kGinContextCount;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 64;
  // 0=dispatch arrival, 1=return arrival.
  // Signal 0/1 are the data-plane dispatch/return epochs. Signal 2 is the
  // fixed terminal production-status epoch and is touched only once per
  // generation, never on the per-task success path.
  requirements.ginSignalCount = 3;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  internal::throw_nccl(
      "gin-create",
      dev_comm_create(nccl->comm, &requirements, &backend->dev_comm),
      nccl->rank, nccl->world, session.session().identity(),
      session.session().generation(), ExchangeKind::ExpertCounts,
      "ncclDevCommCreate(activation GIN)");
  backend->dev_comm_created = true;
  backend->gin_context_count = kGinContextCount;
  try {
    backend->windows.reserve(pool->slots.size());
    for (auto &slot : pool->slots) {
      ncclWindow_t window = nullptr;
      if (slot.gin_window == nullptr || pool->gin_layout.total == 0)
        transport_error(NotifyErrorCode::InvalidHandle, "gin-create",
                        "registered window storage", nccl->rank,
                        "preallocated ncclMemAlloc window", "empty");
      internal::throw_nccl(
          "gin-create",
          window_register(nccl->comm, slot.gin_window,
                          pool->gin_layout.total, &window,
                          NCCL_WIN_STRICT_ORDERING),
          nccl->rank, nccl->world, session.session().identity(),
          session.session().generation(), ExchangeKind::ExpertCounts,
          "ncclCommWindowRegister(activation pool slot)");
      backend->windows.push_back(window);
    }
  } catch (...) {
    backend.reset();
    throw;
  }
  {
    std::lock_guard<std::mutex> lock(pool->mutex);
    pool->gin_backend = backend;
  }
  return backend;
#endif
}

ActivationTransportResourcePool make_activation_transport_pool(
    const CollectiveSession &session, const CudaExecution &execution,
    ActivationTransportPoolConfig config) {
  if (!session || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-create",
                    "session/execution", 0, "valid", "empty");
  if (config.max_world_size < session.world_size())
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "max_world_size", 0,
                    ">= session world", std::to_string(config.max_world_size));
  auto state = std::make_shared<ActivationPoolState>();
  state->config = config;
  state->layout = make_layout(config);
  state->gin_layout = make_gin_window_layout(config);
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda("transport-pool-create",
                       cudaSetDevice(static_cast<int>(state->device)), 0,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
#if MEGAMOE_DEVICE_TIMELINE
  cudaDeviceProp timeline_device_properties{};
  internal::throw_cuda(
      "transport-pool-create",
      cudaGetDeviceProperties(&timeline_device_properties,
                              static_cast<int>(state->device)),
      0, session.world_size(), session.identity(), session.generation(),
      ExchangeKind::ExpertCounts,
      "cudaGetDeviceProperties(device timeline)");
  if (timeline_device_properties.multiProcessorCount <= 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "transport-pool-create", "timeline CTA capacity", 0,
                    "positive SM count",
                    std::to_string(timeline_device_properties.multiProcessorCount));
  state->timeline_max_ctas =
      static_cast<std::uint32_t>(timeline_device_properties.multiProcessorCount);
  state->timeline_layout =
      make_timeline_allocation_layout(state->timeline_max_ctas);
#endif
  state->slots.resize(config.max_inflight_generations);
  try {
    internal::throw_cuda(
        "transport-pool-create",
        cudaStreamCreateWithFlags(&state->cancel_stream, cudaStreamNonBlocking),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaStreamCreate(cancel)");
    for (auto &slot : state->slots) {
      internal::throw_cuda(
          "transport-pool-create", cudaMalloc(&slot.device,
                                                state->layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(transport slot)");
      if (config.enable_gin_rdma) {
        using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
        const auto mem_alloc = reinterpret_cast<MemAllocFn>(
            dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
        if (mem_alloc == nullptr)
          throw NotifyDispatchError(
              NotifyErrorCode::BackendUnavailable,
              "phase=transport-pool-create, field=ncclMemAlloc, rank=0, "
              "expected=runtime symbol, actual=missing");
        internal::throw_nccl(
            "transport-pool-create",
            mem_alloc(&slot.gin_window, state->gin_layout.total), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, "ncclMemAlloc(GIN window slot)");
      }
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
#if MEGAMOE_DEVICE_TIMELINE
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(&slot.forward_timeline, state->timeline_layout.total),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(Forward timeline)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      if (config.enable_backward) {
        internal::throw_cuda(
            "transport-pool-create",
            cudaMalloc(&slot.backward_timeline, state->timeline_layout.total),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, "cudaMalloc(Backward timeline)");
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.forward_timeline, 0,
                          state->timeline_layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize Forward timeline");
      if (slot.backward_timeline)
        internal::throw_cuda(
            "transport-pool-create",
            cudaMemsetAsync(slot.backward_timeline, 0,
                            state->timeline_layout.total,
                            internal::checked_execution_state(execution)->stream),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "one-time initialize Backward timeline");
#endif
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device, 0, state->layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize complete transport slot storage");
      if (slot.gin_window)
        internal::throw_cuda(
            "transport-pool-create",
            cudaMemsetAsync(slot.gin_window, 0, state->gin_layout.total,
                            internal::checked_execution_state(execution)->stream),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "one-time initialize GIN window slot storage");
      if (config.max_intermediate_size != 0 &&
          config.max_hidden_size == 7168 &&
          config.max_intermediate_size == 3072) {
        slot.forward_x_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.canonical_rows),
            config.max_hidden_size, config.max_destination_rows_per_rank, 128,
            "canonical X");
        slot.forward_s_map = make_bf16_tma_map(
            at<void>(slot.device, state->layout.forward_s),
            config.max_intermediate_size,
            config.max_destination_rows_per_rank, 128, "SwiGLU S");
        slot.forward_maps_valid = true;
        if (config.enable_backward) {
          const auto rows = config.max_destination_rows_per_rank;
          const auto reduction_rows =
              (rows + static_cast<std::uint64_t>(config.max_owned_experts) *
                          63ULL +
               63ULL) &
              ~63ULL;
          slot.backward_dy_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_dy_canonical),
              config.max_hidden_size, rows, 128, "backward canonical dY");
          slot.backward_dab_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_dab),
              2ULL * config.max_intermediate_size, rows, 128,
              "backward dAB");
          slot.backward_dy_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.backward_dy_canonical),
              config.max_hidden_size, rows, "backward canonical dY MN");
          slot.backward_x_mn_map = make_bf16_tma_map_mn(
              at<void>(slot.device, state->layout.canonical_rows),
              config.max_hidden_size, rows, "canonical X MN");
          slot.backward_u_transpose_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_u_transpose),
              reduction_rows, config.max_intermediate_size, 128,
              "backward U transpose");
          slot.backward_dab_transpose_map = make_bf16_tma_map(
              at<void>(slot.device, state->layout.backward_dab_transpose),
              reduction_rows, 2ULL * config.max_intermediate_size, 128,
              "backward dAB transpose");
          slot.backward_dw1_store_map = make_bf16_tma_store_map(
              at<void>(slot.device, state->layout.backward_dw1),
              config.max_hidden_size,
              static_cast<std::uint64_t>(config.max_owned_experts) * 2ULL *
                  config.max_intermediate_size,
              "backward dW1 output");
          slot.backward_maps_valid = true;
        }
      } else if (config.max_intermediate_size != 0) {
        // The runtime-shape correctness kernels consume typed pointers and
        // checked dimensions directly.  Native TMA descriptors are neither
        // created nor interpreted for shapes that do not use the specialization.
        slot.forward_maps_valid = true;
        slot.backward_maps_valid = config.enable_backward;
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(transport status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_forward_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(forward status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_production_status),
              checked_size(checked_mul(config.max_world_size,
                                       sizeof(ProductionStatusRecord),
                                       "production status host matrix"),
                           "production status host matrix"),
              cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(production terminal status matrix)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      std::fill_n(slot.host_production_status, config.max_world_size,
                  ProductionStatusRecord{});
      const auto backward_legacy_words =
          32ULL + static_cast<std::size_t>(config.max_owned_experts) * 8ULL;
      slot.generation_debug_contexts = static_cast<std::uint32_t>(
          (config.max_nodes > 0 ? config.max_nodes - 1ULL : 0ULL) * 8ULL);
      slot.generation_debug_words = transport_state::completion_debug_words(
          slot.generation_debug_contexts);
      slot.backward_progress_words =
          backward_legacy_words + kForwardProgressWords +
          2ULL * slot.generation_debug_words;
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(
              reinterpret_cast<void **>(&slot.host_backward_progress),
              slot.backward_progress_words * sizeof(std::uint64_t),
              cudaHostAllocPortable | cudaHostAllocMapped),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostAlloc(mapped Backward progress)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostGetDevicePointer(
              reinterpret_cast<void **>(&slot.device_backward_progress),
              slot.host_backward_progress, 0),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaHostGetDevicePointer(Backward progress)");
      std::fill_n(slot.host_backward_progress, slot.backward_progress_words,
                  0ULL);
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_cancel_epoch),
                        sizeof(std::uint64_t),
                        cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      *slot.host_cancel_epoch = 0;
      if (config.enable_socket_fallback) {
        const auto socket_count_bytes = checked_size(
            checked_mul(checked_mul(config.max_world_size,
                                    config.max_world_size,
                                    "Socket host count matrix"),
                        sizeof(std::uint64_t),
                        "Socket host count matrix bytes"),
            "Socket host count matrix bytes");
        internal::throw_cuda(
            "transport-pool-create",
            cudaHostAlloc(reinterpret_cast<void **>(&slot.host_socket_counts),
                          socket_count_bytes, cudaHostAllocPortable),
            0, session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts,
            "cudaHostAlloc(Socket count matrix)");
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
        std::memset(slot.host_socket_counts, 0, socket_count_bytes);
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaMalloc(reinterpret_cast<void **>(&slot.device_cancel_epoch),
                     sizeof(std::uint64_t)),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaMalloc(cancel epoch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device_cancel_epoch, 0, sizeof(std::uint64_t),
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize device cancel epoch");
      auto create_event = [&](cudaEvent_t *event, const char *name) {
        internal::throw_cuda(
            "transport-pool-create",
            cudaEventCreateWithFlags(event, cudaEventDisableTiming), 0,
            session.world_size(), session.identity(), session.generation(),
            ExchangeKind::ExpertCounts, name);
        internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      };
      create_event(&slot.producer_ready, "cudaEventCreate(producer-ready)");
      create_event(&slot.backward_ready, "cudaEventCreate(backward-ready)");
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_start), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward start)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaEventCreate(&slot.forward_output_ready), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "cudaEventCreate(forward output ready)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.forward_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(forward complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.dispatch_complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(dispatch)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
      internal::throw_cuda(
          "transport-pool-create", cudaEventCreate(&slot.complete), 0,
          session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaEventCreate(complete)");
      internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    }
  } catch (...) {
    throw;
  }
  auto impl = std::make_shared<ActivationTransportResourcePool::Impl>();
  impl->state = std::move(state);
  return ActivationTransportResourcePool(std::move(impl));
}

ActivationTransportPoolUsage ActivationTransportResourcePool::usage() const {
  if (!impl_ || !impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-pool-usage",
                    "pool", 0, "valid", "empty");
  ActivationTransportPoolUsage result;
  auto state = impl_->state;
  std::lock_guard<std::mutex> lock(state->mutex);
  for (const auto &slot : state->slots) {
    if (slot.state == SlotState::Available)
      ++result.slots_available;
    else if (slot.state == SlotState::InUse)
      ++result.slots_in_use;
    else
      ++result.slots_quarantined;
  }
  result.device_bytes = checked_mul(
      checked_add(
          checked_add(state->layout.total, state->gin_layout.total,
                      "pool compute plus GIN bytes"),
          sizeof(std::uint64_t), "pool slot device bytes"),
      state->slots.size(), "pool device bytes");
#if MEGAMOE_DEVICE_TIMELINE
  const auto timeline_phases = state->config.enable_backward ? 2ULL : 1ULL;
  result.device_bytes = checked_add(
      result.device_bytes,
      checked_mul(checked_mul(state->timeline_layout.total, timeline_phases,
                              "pool timeline phase bytes"),
                  state->slots.size(), "pool timeline bytes"),
      "pool device plus timeline bytes");
#endif
  result.compute_device_bytes = checked_mul(
      state->layout.total, state->slots.size(), "pool compute device bytes");
  result.gin_registered_bytes = checked_mul(
      state->gin_layout.total, state->slots.size(),
      "pool GIN registered bytes");
  const auto progress_bytes = checked_mul(
      state->slots.empty() ? 0ULL
                           : state->slots.front().backward_progress_words,
      sizeof(std::uint64_t), "pool mapped progress bytes");
  const auto pinned_per_slot = checked_add(
      checked_add(2ULL * sizeof(TransportDeviceStatus),
                  sizeof(std::uint64_t), "pool pinned status/cancel bytes"),
      progress_bytes, "pool pinned bytes per slot");
  const auto socket_count_bytes = state->config.enable_socket_fallback
      ? checked_mul(checked_mul(state->config.max_world_size,
                                state->config.max_world_size,
                                "Socket host count matrix"),
                    sizeof(std::uint64_t),
                    "Socket host count matrix bytes")
      : 0ULL;
  result.pinned_bytes = checked_mul(
      checked_add(pinned_per_slot, socket_count_bytes,
                  "pool pinned plus Socket bytes"),
      state->slots.size(),
                                    "pool pinned bytes");
  result.event_count = checked_mul(8, state->slots.size(), "pool events");
  result.epoch_high_watermark = state->epoch_high;
  return result;
}

std::uint64_t BFloat16ActivationLease::num_tokens() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->tokens;
}
std::uint64_t BFloat16ActivationLease::hidden_size() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->hidden;
}
std::uint64_t BFloat16ActivationLease::element_count() const {
  return checked_mul(num_tokens(), hidden_size(), "activation elements");
}
std::uint64_t BFloat16ActivationLease::generation() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->generation;
}
std::uint32_t BFloat16ActivationLease::device_ordinal() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->device;
}
std::uint32_t BFloat16ActivationLease::layout_version() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "activation-lease",
                    "lease", 0, "valid", "empty");
  return impl_->layout;
}

BFloat16ActivationLease ActivationTransportBridge::upload_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!context || !execution || !pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "activation-test-upload", "context/execution/pool", 0,
                    "valid", "empty");
  auto state = pool.impl_->state;
  if (!context.execution().same_instance(execution.token()) ||
      context.session().stable_identity_bytes() != state->session_identity ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-test-upload", "provenance",
                    context.calling_rank(), "exact pool session/execution/device",
                    "mismatch");
  if (num_tokens > state->config.max_tokens_per_rank ||
      hidden == 0 || hidden > state->config.max_hidden_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "activation-test-upload", "shape", context.calling_rank(),
                    "within pool capacities", "exceeded/zero");
  const auto count = checked_mul(num_tokens, hidden, "activation upload count");
  if (host.size() != checked_size(count, "activation upload count"))
    transport_error(NotifyErrorCode::InvalidRouteShape,
                    "activation-test-upload", "host.size",
                    context.calling_rank(), std::to_string(count),
                    std::to_string(host.size()));
  auto lease = acquire_slot(state, context.session().generation(),
                            context.calling_rank());
  auto &slot = state->slots[lease->index];
  auto exec = internal::checked_execution_state(execution);
  if (count != 0) {
    const auto bytes = checked_size(checked_mul(count, sizeof(transport::BFloat16),
                                                "activation upload bytes"),
                                    "activation upload bytes");
    internal::throw_cuda(
        "activation-test-upload",
        cudaMemcpyAsync(at<void>(slot.device, state->layout.input), host.data(),
                        bytes, cudaMemcpyHostToDevice, exec->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemcpyAsync(test activation)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "activation-test-upload", cudaEventRecord(slot.producer_ready, exec->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(producer-ready)");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->slot = std::move(lease);
  impl->pool = state;
  impl->pointer = at<const transport::BFloat16>(slot.device, state->layout.input);
  impl->session_identity = context.session().stable_identity_bytes();
  impl->producer_execution = execution.token();
  impl->generation = context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden;
  impl->device = execution.device_ordinal();
  impl->ready_recorded = true;
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16ActivationLease ActivationTransportBridge::reissue_for_benchmark(
    const BFloat16ActivationLease &storage,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!storage.impl_ || !context || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "activation-benchmark-reissue", "input", 0, "valid",
                    "empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.session_identity != context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.slot || !source.pool)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "activation-benchmark-reissue", "provenance",
                    context.calling_rank(),
                    "same immutable storage/session/execution/device/ready",
                    "mismatch");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>(source);
  impl->generation = context.session().generation();
  return BFloat16ActivationLease(std::move(impl));
}

BFloat16ActivationLease testing::upload_bfloat16_activations_for_testing(
    const std::vector<transport::BFloat16> &host, std::uint64_t num_tokens,
    std::uint64_t hidden, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  return ActivationTransportBridge::upload_for_testing(
      host, num_tokens, hidden, context, execution, pool);
}

BFloat16ActivationLease ExternalCudaProducer::publish_activations(
    const transport::BFloat16 *device_data,
    std::uint64_t capacity_elements, std::uint64_t num_tokens,
    std::uint64_t hidden_size, const ActivationTransportResourcePool &pool,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !pool.impl_ || !pool.impl_->state || !allocation_owner)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-activation-publish", "producer/pool/owner", 0,
                    "valid", "empty");
  auto pool_state = pool.impl_->state;
  if (state_->backend->context.session().stable_identity_bytes() !=
          pool_state->session_identity ||
      state_->backend->device != pool_state->device || hidden_size == 0 ||
      hidden_size > pool_state->config.max_hidden_size ||
      num_tokens > pool_state->config.max_tokens_per_rank)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "external-activation-publish", "provenance/shape", 0,
                    "matching pool session/device and bounded shape",
                    "mismatch");
  const auto elements = checked_mul(num_tokens, hidden_size,
                                    "external activation elements");
  validate_external_cuda_range(device_data, elements, capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "activations");
  auto lease = acquire_slot(pool_state,
                            state_->backend->context.session().generation(),
                            state_->backend->context.calling_rank());
  auto &slot = pool_state->slots[lease->index];
  internal::throw_cuda(
      "external-activation-publish",
      cudaEventRecord(slot.producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(activation ready)");
  auto impl = std::make_shared<BFloat16ActivationLease::Impl>();
  impl->slot = std::move(lease);
  impl->pool = std::move(pool_state);
  impl->pointer = device_data;
  impl->session_identity = state_->backend->context.session().stable_identity_bytes();
  impl->producer_execution = state_->backend->execution;
  impl->generation = state_->backend->context.session().generation();
  impl->tokens = num_tokens;
  impl->hidden = hidden_size;
  impl->device = state_->backend->device;
  impl->ready_recorded = true;
  impl->external_owner = std::move(allocation_owner);
  impl->external_producer = state_->backend;
  return BFloat16ActivationLease(std::move(impl));
}

DeviceActivationTransport::DeviceActivationTransport(
    const CollectiveSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!session || !execution || !pool)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-create",
                    "session/execution/pool", 0, "valid", "empty");
  if (session.stable_identity_bytes() != pool.impl_->state->session_identity ||
      session.world_size() != pool.impl_->state->world ||
      execution.device_ordinal() != pool.impl_->state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-create",
                    "pool provenance", 0, "matching session/world/device",
                    "mismatch");
  impl_ = std::make_shared<Impl>();
  impl_->session = session;
  impl_->execution = execution;
  impl_->pool = pool;
}

LocalCudaP2pActivationTransport::LocalCudaP2pActivationTransport(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-create",
                    "pool", 0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
}

NcclSocketFallbackActivationTransport::
    NcclSocketFallbackActivationTransport(
        const NcclSession &session, const CudaExecution &execution,
        const ActivationTransportResourcePool &pool) {
  if (!session || !execution || !pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "socket-create",
                    "session/execution/pool", 0, "valid", "empty");
  auto state = pool.impl_->state;
  const auto nccl = internal::NcclSessionAccess::checked(session);
  if (!state->config.enable_socket_fallback)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-create",
                    "pool.enable_socket_fallback", session.rank(), "true",
                    "false");
  const auto *ib_disabled = std::getenv("NCCL_IB_DISABLE");
  if (ib_disabled == nullptr || std::string(ib_disabled) != "1")
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=socket-create, field=NCCL_IB_DISABLE, rank=" +
            std::to_string(session.rank()) +
            ", expected=1 before communicator creation, actual=" +
            (ib_disabled ? std::string(ib_disabled) : "unset") +
            "; NcclSocketFallback refuses an unaudited network selection");
  if (session.session().stable_identity_bytes() != state->session_identity ||
      session.session().world_size() != state->world ||
      session.device_ordinal() != state->device ||
      execution.device_ordinal() != state->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "socket-create",
                    "provenance", session.rank(),
                    "same session/world/device", "mismatch");
  impl_ = std::make_shared<Impl>();
  impl_->backend = std::make_shared<SocketBackendState>();
  impl_->backend->pool = std::move(state);
  impl_->backend->nccl = nccl;
  impl_->backend->session = session;
  impl_->backend->execution = execution;
}

GinRdmaActivationTransport::GinRdmaActivationTransport(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-create", "pool", 0,
                    "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend = create_gin_backend(session, execution, pool.impl_->state);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (pool.impl_ && pool.impl_->state &&
      pool.impl_->state->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-create",
                    "backend", session.rank(), "LocalCudaP2p",
                    "GIN-enabled pool requires GinRdmaForwardMegaKernel");
  *this = LocalCudaP2pForwardMegaKernel(session, execution, pool, false);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin) {
  *this = LocalCudaP2pForwardMegaKernel(session, execution, pool, use_gin,
                                        false);
}

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin,
    bool use_socket) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-create", "pool",
                    0, "valid", "empty");
  if (use_gin && use_socket)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-create",
                    "backend", session.rank(), "one backend",
                    "GIN and Socket both requested");
  impl_ = std::make_shared<Impl>();
  if (use_gin) {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_gin_backend(session, execution, pool.impl_->state));
  } else if (use_socket) {
    auto socket = std::make_shared<SocketBackendState>();
    socket->pool = pool.impl_->state;
    socket->nccl = internal::NcclSessionAccess::checked(session);
    socket->session = session;
    socket->execution = execution;
    impl_->backend = socket;
    impl_->socket_transport =
        std::make_shared<NcclSocketFallbackActivationTransport>(
            session, execution, pool);
  } else {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_local_p2p_backend(session, execution, pool.impl_->state));
  }
}

GinRdmaForwardMegaKernel::GinRdmaForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, true))) {}

DeviceForwardHandle GinRdmaForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

NcclSocketFallbackForwardMegaKernel::NcclSocketFallbackForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pForwardMegaKernel>(
          new LocalCudaP2pForwardMegaKernel(session, execution, pool, false,
                                            true))) {}

DeviceForwardHandle NcclSocketFallbackForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights, const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(dispatch, activations, weights, config, options);
}

LocalCudaP2pBackwardMegaKernel::LocalCudaP2pBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (pool.impl_ && pool.impl_->state &&
      pool.impl_->state->config.enable_gin_rdma)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-create",
                    "backend", session.rank(), "LocalCudaP2p",
                    "GIN-enabled pool requires GinRdmaBackwardMegaKernel");
  *this = LocalCudaP2pBackwardMegaKernel(session, execution, pool, false);
}

LocalCudaP2pBackwardMegaKernel::LocalCudaP2pBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin) {
  *this = LocalCudaP2pBackwardMegaKernel(session, execution, pool, use_gin,
                                         false);
}

LocalCudaP2pBackwardMegaKernel::LocalCudaP2pBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool, bool use_gin,
    bool use_socket) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-create", "pool",
                    0, "valid", "empty");
  if (!pool.impl_->state->config.enable_backward)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-create",
                    "pool.enable_backward", 0, "true", "false");
  if (use_gin && use_socket)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-create",
                    "backend", session.rank(), "one backend",
                    "GIN and Socket both requested");
  impl_ = std::make_shared<Impl>();
  if (use_gin) {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_gin_backend(session, execution, pool.impl_->state));
  } else if (use_socket) {
    auto socket = std::make_shared<SocketBackendState>();
    socket->pool = pool.impl_->state;
    socket->nccl = internal::NcclSessionAccess::checked(session);
    socket->session = session;
    socket->execution = execution;
    impl_->backend = socket;
  } else {
    impl_->backend = std::static_pointer_cast<ActivationMegaKernelBackendState>(
        create_local_p2p_backend(session, execution, pool.impl_->state));
  }
}

GinRdmaBackwardMegaKernel::GinRdmaBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pBackwardMegaKernel>(
          new LocalCudaP2pBackwardMegaKernel(session, execution, pool, true))) {}

DeviceBackwardHandle GinRdmaBackwardMegaKernel::run(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &output_gradient,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(forward, output_gradient, config, options);
}

NcclSocketFallbackBackwardMegaKernel::NcclSocketFallbackBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool)
    : delegate_(std::shared_ptr<LocalCudaP2pBackwardMegaKernel>(
          new LocalCudaP2pBackwardMegaKernel(session, execution, pool, false,
                                             true))) {}

DeviceBackwardHandle NcclSocketFallbackBackwardMegaKernel::run(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &output_gradient,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  return delegate_->run(forward, output_gradient, config, options);
}

DeviceActivationTransportHandle
DeviceActivationTransport::run_device_simulation_synthetic(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config) {
  if (!impl_ || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-run",
                    "transport/dispatch/activation", 0, "valid", "empty");
  const auto &id = dispatch.identity();
  auto state = impl_->pool.impl_->state;
  const auto &activation = *activations.impl_;
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens || activation.hidden != config.hidden_size ||
      activation.device != impl_->execution.device_ordinal() ||
      activation.session_identity != impl_->session.stable_identity_bytes() ||
      !activation.producer_execution.same_instance(impl_->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size ||
      id.num_nodes > state->config.max_nodes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "transport-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "transport-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");
  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(impl_->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(impl_->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  const auto links = checked_mul(id.num_nodes, config.channels, "actual links");
  const auto ring_slots = checked_mul(
      checked_mul(id.num_nodes, config.channels, "actual ring links"),
      config.ring_slots_per_source_channel, "actual ring slots");
  const auto route_elements = checked_mul(view.sizes.route_count,
                                          config.hidden_size, "route elements");
  const auto combined_elements = checked_mul(id.num_tokens, config.hidden_size,
                                             "combined elements");
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0,
                        checked_size(checked_mul(count, width, name), name),
                        exec->stream),
        id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
  };
  internal::throw_cuda(
      "transport-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  zero_async(layout.ring_headers, ring_slots,
             sizeof(transport::TransportSlotHeader), "ring headers");
  zero_async(layout.ring_ready_sequence, ring_slots, sizeof(std::uint64_t),
             "ring sequences");
  zero_async(layout.link_next_sequence, links, sizeof(std::uint64_t),
             "link sequences");
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits), links,
                      config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "transport-run", cudaGetLastError(), id.calling_rank, id.world_size,
      impl_->session.identity(), id.generation, ExchangeKind::ExpertCounts,
      "initialize transport credits kernel");
  zero_async(layout.packet_claim, view.sizes.packet_count,
             sizeof(std::uint32_t), "packet claims");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "canonical elements"),
             sizeof(transport::BFloat16), "canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "combined");
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "status");
  internal::throw_cuda("transport-run", cudaEventRecord(slot.start, exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(start)");
  int sms = 0;
  internal::throw_cuda(
      "transport-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  TransportKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, impl_->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
  args.ring_headers = at<transport::TransportSlotHeader>(slot.device,
                                                          layout.ring_headers);
  args.ring_payload = at<transport::BFloat16>(slot.device, layout.ring_payload);
  args.ring_ready = at<unsigned long long>(slot.device,
                                            layout.ring_ready_sequence);
  args.link_next = at<unsigned long long>(slot.device,
                                           layout.link_next_sequence);
  args.link_credits = at<unsigned int>(slot.device, layout.link_credits);
  args.packet_claim = at<unsigned int>(slot.device, layout.packet_claim);
  args.canonical_rows = at<transport::BFloat16>(slot.device,
                                                 layout.canonical_rows);
  args.contribution = at<transport::BFloat16>(slot.device, layout.contribution);
  args.combine_return = at<transport::BFloat16>(slot.device,
                                                 layout.combine_return);
  args.combined = at<transport::BFloat16>(slot.device, layout.combined);
  args.status = at<TransportDeviceStatus>(slot.device, layout.status);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  activation_transport_persistent_kernel<<<std::max(sms, 1), 128, 0,
                                           exec->stream>>>(args);
  internal::throw_cuda("transport-run", cudaGetLastError(), id.calling_rank,
                       id.world_size, impl_->session.identity(), id.generation,
                       ExchangeKind::ExpertCounts,
                       "activation transport persistent kernel launch");
  internal::throw_cuda(
      "transport-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(dispatch complete)");
  internal::throw_cuda(
      "transport-run",
      cudaMemcpyAsync(slot.host_status, args.status,
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, impl_->session.identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda("transport-run", cudaEventRecord(slot.complete,
                                                         exec->stream),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventRecord(complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda("transport-run", cudaEventSynchronize(slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventSynchronize(transport complete)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "transport-kernel", "descriptor", id.calling_rank,
                    "valid packet/fanout", "device validation error packet=" +
                        std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda("transport-timing",
                       cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
                       id.calling_rank, id.world_size, impl_->session.identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "cudaEventElapsedTime(total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  // The concrete backend object, never a caller-provided enum, owns this
  // label. This path is deliberately and permanently simulation-only.
  result->audit.backend =
      transport::ActivationTransportBackend::DeviceSimulation;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    result->timings.combine_reduce_us =
        static_cast<double>(complete_clock - dispatch_clock) / 1000.0;
  }
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle LocalCudaP2pActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "local-p2p-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node and exact local NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "local-p2p-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (config.channels != state->config.channels ||
      config.ring_slots_per_source_channel !=
          state->config.ring_slots_per_source_channel ||
      id.top_k > state->config.max_top_k ||
      id.world_size > state->config.max_world_size)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "local-p2p-run", "transport config", id.calling_rank,
                    "within exact pool geometry", "mismatch/exceeded");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "local-p2p-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported gate-before placement");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  [[maybe_unused]] const auto route_elements =
      checked_mul(view.sizes.route_count, config.hidden_size,
                  "P2P route elements");
  [[maybe_unused]] const auto combined_elements =
      checked_mul(id.num_tokens, config.hidden_size,
                  "P2P combined elements");
  [[maybe_unused]] const auto rank_links =
      checked_mul(id.world_size, config.channels, "P2P rank links");
  [[maybe_unused]] const auto ingress_slots = checked_mul(
      rank_links, config.ring_slots_per_source_channel, "P2P ingress slots");
  [[maybe_unused]] const auto node_links =
      static_cast<std::uint64_t>(config.channels);
  [[maybe_unused]] const auto node_ring_slots = checked_mul(
      node_links, config.ring_slots_per_source_channel, "P2P proxy slots");
  std::uint64_t generation_memset_calls = 0;
  std::uint64_t generation_memset_bytes = 0;
  std::uint64_t kernel_launches = 0;
  const auto zero_async = [&](std::size_t offset, std::uint64_t count,
                              std::size_t width, const char *name) {
    if (count == 0)
      return;
    const auto bytes = checked_size(checked_mul(count, width, name), name);
    internal::throw_cuda(
        "local-p2p-run",
        cudaMemsetAsync(at<void>(slot.device, offset), 0, bytes, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, name);
    ++generation_memset_calls;
    generation_memset_bytes += bytes;
  };
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "local-p2p-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  // The integrated timing starts after the router-owned input is ready and
  // includes every generation-specific transport initialization below.
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P start)");
#if MEGAMOE_DEBUG_VALIDATE
  zero_async(layout.ring_headers, node_ring_slots,
             sizeof(transport::TransportSlotHeader), "P2P proxy headers");
  zero_async(layout.ring_payload,
             checked_mul(node_ring_slots, config.hidden_size,
                         "P2P proxy payload"),
             sizeof(transport::BFloat16), "P2P proxy payload");
  zero_async(layout.ring_ready_sequence, node_ring_slots,
             sizeof(std::uint64_t), "P2P proxy ready");
  zero_async(layout.link_next_sequence, node_links, sizeof(std::uint64_t),
             "P2P proxy sequence");
  zero_async(layout.ingress_headers, ingress_slots,
             sizeof(LocalP2pRouteHeader), "P2P ingress headers");
  zero_async(layout.ingress_ready_sequence, ingress_slots,
             sizeof(std::uint64_t), "P2P ingress ready");
  zero_async(layout.ingress_next_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress next");
  zero_async(layout.ingress_published_sequence, rank_links,
             sizeof(std::uint64_t), "P2P ingress published");
  zero_async(layout.outgoing_route_sequence, rank_links,
             sizeof(std::uint64_t), "P2P outgoing sequence");
  zero_async(layout.direct_route_headers,
             state->config.max_destination_rows_per_rank,
             sizeof(LocalP2pRouteHeader), "P2P direct route headers");
  zero_async(layout.return_ready_epoch, view.sizes.route_count,
             sizeof(std::uint64_t), "P2P return ready");
  zero_async(layout.canonical_rows,
             checked_mul(state->config.max_destination_rows_per_rank,
                         config.hidden_size, "P2P canonical rows"),
             sizeof(transport::BFloat16), "P2P canonical rows");
  zero_async(layout.contribution, route_elements,
             sizeof(transport::BFloat16), "P2P contributions");
  zero_async(layout.combine_return, route_elements,
             sizeof(transport::BFloat16), "P2P combine return");
  zero_async(layout.combined, combined_elements,
             sizeof(transport::BFloat16), "P2P combined");
#else
  // The trusted single-node bulk path overwrites every canonical row,
  // contribution and output element that it later reads. Epoch-valued peer
  // signals make all arena and signal clears unnecessary. Clearing a peer
  // signal here would race a faster rank's release store.
#endif
  zero_async(layout.status, 1, sizeof(TransportDeviceStatus), "P2P status");
#if MEGAMOE_DEBUG_VALIDATE
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((node_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.link_credits),
                      node_links, config.ring_slots_per_source_channel);
  initialize_transport_credits_kernel<<<
      static_cast<unsigned int>((rank_links + 255U) / 256U), 256, 0,
      exec->stream>>>(at<unsigned int>(slot.device, layout.ingress_credits),
                      rank_links, config.ring_slots_per_source_channel);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P credits");

  // A fixed-width device AllGather is a setup barrier only. It exchanges no
  // activation and ensures every peer ring/credit/sequence region is fully
  // initialized before any persistent worker publishes to it.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, activation.slot->epoch);
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
  validate_p2p_barrier_kernel<<<1, 256, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_receive),
      id.world_size, activation.slot->index, activation.slot->epoch,
      at<TransportDeviceStatus>(slot.device, layout.status));
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "validate P2P setup barrier");
#else
  // This small collective is a real per-generation dependency: peer writes
  // must not begin until every process has reached the transport phase. It
  // carries only 16 bytes/rank and replaces the old multi-gigabyte clears.
  initialize_p2p_barrier_kernel<<<1, 1, 0, exec->stream>>>(
      at<unsigned long long>(slot.device, layout.p2p_barrier_send),
      activation.slot->index, id.generation);
  ++kernel_launches;
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "initialize P2P setup barrier");
  internal::throw_nccl(
      "local-p2p-run",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P setup barrier)");
#endif
  int sms = 0;
  internal::throw_cuda(
      "local-p2p-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  if (sms <= static_cast<int>(id.world_size))
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=local-p2p-run, field=resident_workers, rank=" +
            std::to_string(id.calling_rank) + ", expected>world=" +
            std::to_string(id.world_size) + ", actual_sms=" +
            std::to_string(sms));
  LocalP2pKernelArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = config.channels;
  args.ring_slots = config.ring_slots_per_source_channel;
#if MEGAMOE_DEBUG_VALIDATE
  args.epoch = activation.slot->epoch;
#else
  // Every rank invokes the trusted team in the same generation order. A
  // monotonically increasing per-slot epoch supports both new generations and
  // repeated same-generation correctness runs without clearing peer signals.
  if (slot.transport_epoch == std::numeric_limits<std::uint64_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "local-p2p-run",
                    "transport_epoch", id.calling_rank, "<UINT64_MAX",
                    "UINT64_MAX");
  args.epoch = ++slot.transport_epoch;
#endif
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.local_base = slot.device;
  args.ring_headers_offset = layout.ring_headers;
  args.ring_payload_offset = layout.ring_payload;
  args.ring_ready_offset = layout.ring_ready_sequence;
  args.link_next_offset = layout.link_next_sequence;
  args.link_credits_offset = layout.link_credits;
  args.ingress_headers_offset = layout.ingress_headers;
  args.ingress_ready_offset = layout.ingress_ready_sequence;
  args.ingress_next_offset = layout.ingress_next_sequence;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.ingress_credits_offset = layout.ingress_credits;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.direct_headers_offset = layout.direct_route_headers;
  args.return_ready_offset = layout.return_ready_epoch;
  args.canonical_rows_offset = layout.canonical_rows;
  args.contribution_offset = layout.contribution;
  args.combine_return_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  internal::throw_cuda(
      "local-p2p-run",
      cudaEventRecord(slot.dispatch_complete, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(P2P generation setup complete)");
#if MEGAMOE_DEBUG_VALIDATE
  local_p2p_persistent_kernel<<<sms, 128, 0, exec->stream>>>(args);
  ++kernel_launches;
#else
  local_p2p_bulk_dispatch_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.route_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_dispatch_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_return_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(
          view.sizes.incoming_row_count, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_publish_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_wait_return_kernel<<<1, 32, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_bulk_reduce_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(id.num_tokens, 1ULL)),
      256, 0, exec->stream>>>(args);
  ++kernel_launches;
  local_p2p_finish_kernel<<<1, 1, 0, exec->stream>>>(args);
  ++kernel_launches;
#endif
  internal::throw_cuda(
      "local-p2p-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "local P2P persistent kernel launch");
  internal::throw_cuda(
      "local-p2p-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H P2P transport status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "local-p2p-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(P2P complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "local-p2p-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(P2P complete)");
    const auto expired =
        options.deadline_after.count() > 0 &&
        std::chrono::steady_clock::now() - wait_start >=
            options.deadline_after;
    if (!cancelled && (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, id.generation,
                       __ATOMIC_RELEASE);
      internal::throw_cuda(
          "local-p2p-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish cancel epoch");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                sizeof(std::uint64_t));
      internal::throw_cuda(
          "local-p2p-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaStreamSynchronize(cancel)");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  // Debug/cancellation paths retain the explicit team drain. In the trusted
  // release path every source waits for a generation-valued return signal
  // from every destination before producing output. The next Notify control
  // collective is the natural inter-generation barrier.
#if MEGAMOE_DEBUG_VALIDATE
  internal::throw_nccl(
      "local-p2p-drain",
      ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                    at<void>(slot.device, layout.p2p_barrier_receive), 2,
                    ncclUint64, backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllGather(P2P terminal drain)");
  internal::throw_cuda(
      "local-p2p-drain", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(P2P terminal drain)");
#else
  if (cancelled) {
    internal::throw_nccl(
        "local-p2p-drain",
        ncclAllGather(at<void>(slot.device, layout.p2p_barrier_send),
                      at<void>(slot.device, layout.p2p_barrier_receive), 2,
                      ncclUint64, backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "ncclAllGather(P2P cancellation drain)");
    internal::throw_cuda(
        "local-p2p-drain", cudaStreamSynchronize(exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(P2P cancellation drain)");
  }
#endif
  consumer.finish();
  if (cancelled || slot.host_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=local-p2p-run, field=cancel_epoch, rank=" +
            std::to_string(id.calling_rank) + ", expected=completion, actual=" +
            (options.cancellation.cancellation_requested() ? "cancelled"
                                                           : "deadline") +
            ", packets=" + std::to_string(slot.host_status->packets_done) +
            ", incoming=" + std::to_string(slot.host_status->incoming_done) +
            ", combine=" + std::to_string(slot.host_status->combine_done));
  if (slot.host_status->error != 0)
    transport_error(
        NotifyErrorCode::TransportProtocolFailure, "local-p2p-kernel",
        "ring/sequence/descriptor", id.calling_rank, "valid",
        "device error field=" + std::to_string(slot.host_status->error_field) +
            ", packet=" + std::to_string(slot.host_status->error_packet));
  float total_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(P2P total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->local_p2p_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows =
      state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::LocalCudaP2p;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.peak_ring_slots = slot.host_status->peak_ring_slots;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = generation_memset_calls;
  result->audit.generation_memset_bytes = generation_memset_bytes;
  result->audit.kernel_launches = kernel_launches;
  result->audit.dispatch_kernel_blocks = view.sizes.route_count;
  result->audit.return_kernel_blocks = view.sizes.incoming_row_count;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  float setup_ms = 0.0F;
  internal::throw_cuda(
      "local-p2p-timing",
      cudaEventElapsedTime(&setup_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(P2P generation setup)");
  result->timings.generation_setup_us =
      static_cast<double>(setup_ms) * 1000.0;
  result->timings.dispatch_put_us =
      static_cast<double>(slot.host_status->dispatch_put_ticks) / 1000.0;
  result->timings.relay_us =
      static_cast<double>(slot.host_status->relay_ticks) / 1000.0;
  result->timings.synthetic_return_us =
      static_cast<double>(slot.host_status->synthetic_ticks) / 1000.0;
  const auto start_clock = slot.host_status->start_clock;
  const auto dispatch_clock = slot.host_status->dispatch_complete_clock;
  const auto complete_clock = slot.host_status->complete_clock;
  if (dispatch_clock >= start_clock && complete_clock >= dispatch_clock) {
    result->timings.dispatch_relay_us =
        static_cast<double>(dispatch_clock - start_clock) / 1000.0;
    const auto post_dispatch = complete_clock - dispatch_clock;
    const auto relay_return = slot.host_status->synthetic_ticks;
    result->timings.combine_reduce_us = static_cast<double>(
        post_dispatch >= relay_return ? post_dispatch - relay_return : 0ULL) /
        1000.0;
  }
  const auto kernel_us = complete_clock >= start_clock
                             ? static_cast<double>(complete_clock - start_clock) /
                                   1000.0
                             : 0.0;
  const auto accounted = result->timings.generation_setup_us + kernel_us;
  result->timings.terminal_completion_us =
      std::max(0.0, result->timings.total_us - accounted);
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle
NcclSocketFallbackActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "socket-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto &id = dispatch.identity();
  auto state = backend->pool;
  const auto &activation = *activations.impl_;
  if (id.num_nodes < 2 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "topology", id.calling_rank,
                    "multi-node exact NCCL world/rank", "mismatch");
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "socket-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_socket_fallback ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "pool geometry", id.calling_rank, "within capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "socket-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=socket-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  auto *local_counts = at<unsigned long long>(slot.device,
                                               layout.terminal_epoch);
  auto *count_matrix = at<unsigned long long>(slot.device,
                                               layout.socket_count_matrix);
  auto *prefix = at<unsigned long long>(slot.device,
                                        layout.p2p_barrier_receive);
  auto *route_ids = at<unsigned int>(slot.device, layout.packet_claim);
  auto *send_headers = at<LocalP2pRouteHeader>(
      slot.device, layout.socket_header_send);
  auto *receive_headers = at<LocalP2pRouteHeader>(
      slot.device, layout.socket_header_receive);
  auto *send_payload = at<transport::BFloat16>(slot.device,
                                                layout.contribution);
  auto *receive_payload = at<transport::BFloat16>(
      slot.device, layout.socket_payload_receive);
  auto *canonical = at<transport::BFloat16>(slot.device,
                                             layout.canonical_rows);
  auto *by_route = at<transport::BFloat16>(slot.device,
                                            layout.combine_return);
  auto *combined = at<transport::BFloat16>(slot.device, layout.combined);
  auto *status = at<TransportDeviceStatus>(slot.device, layout.status);
  const auto world = id.world_size;
  const auto route_count = view.sizes.route_count;

  internal::throw_cuda(
      "socket-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(Socket start)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(local_counts, 0,
                                     world * sizeof(std::uint64_t),
                                     exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket counts)");
  internal::throw_cuda(
      "socket-run", cudaMemsetAsync(status, 0, sizeof(*status), exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(Socket status)");
  if (route_count != 0) {
    socket_route_count_kernel<<<
        static_cast<unsigned int>((route_count + 255ULL) / 256ULL), 256, 0,
        exec->stream>>>(view, local_counts);
  }
  socket_route_prefix_kernel<<<1, 1, 0, exec->stream>>>(local_counts, prefix,
                                                        world);
  if (route_count != 0) {
    socket_stable_route_ids_kernel<<<world, 256, 0, exec->stream>>>(
        view, prefix, route_ids);
    socket_pack_routes_kernel<<<static_cast<unsigned int>(route_count), 256, 0,
                                exec->stream>>>(
        view, activation.pointer, config.hidden_size, activation.slot->epoch,
        route_ids, send_headers, send_payload);
  }
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket pack kernels");
  internal::throw_nccl(
      "socket-run",
      ncclAllGather(local_counts, count_matrix, world, ncclUint64,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "ncclAllGather(Socket route counts)");
  const auto count_bytes = checked_size(
      checked_mul(checked_mul(world, world, "Socket actual count matrix"),
                  sizeof(std::uint64_t), "Socket actual count bytes"),
      "Socket actual count bytes");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_socket_counts, count_matrix, count_bytes,
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "D2H Socket route counts");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            count_bytes);
  internal::throw_cuda(
      "socket-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamSynchronize(Socket counts)");

  std::vector<std::uint64_t> send_prefix(static_cast<std::size_t>(world) + 1U,
                                         0);
  std::vector<std::uint64_t> receive_prefix(
      static_cast<std::size_t>(world) + 1U, 0);
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    send_prefix[peer + 1U] =
        send_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                    world +
                                peer];
    receive_prefix[peer + 1U] =
        receive_prefix[peer] +
        slot.host_socket_counts[static_cast<std::size_t>(peer) * world +
                                id.calling_rank];
  }
  const auto incoming = receive_prefix[world];
  if (send_prefix[world] != route_count ||
      incoming > state->config.max_destination_rows_per_rank)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "socket-run",
                    "route totals", id.calling_rank,
                    "send=route_count and incoming<=capacity",
                    "send=" + std::to_string(send_prefix[world]) +
                        ", incoming=" + std::to_string(incoming));

  const auto enqueue_dispatch = [&] {
    internal::throw_nccl("socket-dispatch", ncclGroupStart(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupStart(Socket dispatch)");
    for (std::uint32_t peer = 0; peer < world; ++peer) {
      const auto send_rows = send_prefix[peer + 1U] - send_prefix[peer];
      const auto receive_rows =
          receive_prefix[peer + 1U] - receive_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-dispatch", "self rows", id.calling_rank,
                          std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0) {
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(receive_headers + receive_prefix[peer],
                              send_headers + send_prefix[peer],
                              checked_size(send_rows * sizeof(*send_headers),
                                           "Socket self header bytes"),
                              cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self headers");
          internal::throw_cuda(
              "socket-dispatch",
              cudaMemcpyAsync(
                  receive_payload + receive_prefix[peer] * config.hidden_size,
                  send_payload + send_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket self payload bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, world,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket self payload");
        }
        continue;
      }
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_headers + send_prefix[peer],
                     send_rows * sizeof(*send_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclSend(send_payload + send_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket payload)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_headers + receive_prefix[peer],
                     receive_rows * sizeof(*receive_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket headers)");
        internal::throw_nccl(
            "socket-dispatch",
            ncclRecv(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, world,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket payload)");
      }
    }
    internal::throw_nccl("socket-dispatch", ncclGroupEnd(), id.calling_rank,
                         world, backend->session.session().identity(),
                         id.generation, ExchangeKind::ExpertCounts,
                         "ncclGroupEnd(Socket dispatch)");
  };
  enqueue_dispatch();
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.dispatch_complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket dispatch complete)");
  if (incoming != 0)
    socket_relay_transport_kernel<<<static_cast<unsigned int>(incoming), 256,
                                     0, exec->stream>>>(
        receive_headers, receive_payload, canonical, incoming,
        config.hidden_size, id.generation, id.calling_rank,
        state->config.max_destination_rows_per_rank,
        config.expert_transform ==
            transport::ActivationExpertTransform::Identity,
        status);

  internal::throw_nccl("socket-return", ncclGroupStart(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupStart(Socket return)");
  for (std::uint32_t peer = 0; peer < world; ++peer) {
    const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
    const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
    if (peer == id.calling_rank) {
      if (send_rows != 0)
        internal::throw_cuda(
            "socket-return",
            cudaMemcpyAsync(
                send_payload + send_prefix[peer] * config.hidden_size,
                receive_payload + receive_prefix[peer] * config.hidden_size,
                checked_size(send_rows * config.hidden_size *
                                 sizeof(transport::BFloat16),
                             "Socket self return bytes"),
                cudaMemcpyDeviceToDevice, exec->stream),
            id.calling_rank, world, backend->session.session().identity(),
            id.generation, ExchangeKind::ExpertCounts,
            "D2D Socket self return");
      continue;
    }
    if (send_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclSend(receive_payload +
                       receive_prefix[peer] * config.hidden_size,
                   send_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclSend(Socket return)");
    if (receive_rows != 0)
      internal::throw_nccl(
          "socket-return",
          ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                   receive_rows * config.hidden_size *
                       sizeof(transport::BFloat16),
                   ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                   exec->stream),
          id.calling_rank, world, backend->session.session().identity(),
          id.generation, ExchangeKind::ExpertCounts,
          "ncclRecv(Socket return)");
  }
  internal::throw_nccl("socket-return", ncclGroupEnd(), id.calling_rank,
                       world, backend->session.session().identity(),
                       id.generation, ExchangeKind::ExpertCounts,
                       "ncclGroupEnd(Socket return)");
  if (route_count != 0)
    socket_scatter_returns_kernel<<<static_cast<unsigned int>(route_count),
                                    256, 0, exec->stream>>>(
        send_headers, send_payload, by_route, route_count,
        config.hidden_size);
  if (id.num_tokens != 0)
    socket_gate_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens), 256,
                                0, exec->stream>>>(
        view, ActivationTransportBridge::gate_pointer(dispatch), by_route,
        combined, config.hidden_size);
  socket_finalize_status_kernel<<<1, 1, 0, exec->stream>>>(
      status, route_count, incoming, id.num_tokens);
  internal::throw_cuda(
      "socket-run", cudaGetLastError(), id.calling_rank, world,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "Socket relay/return/reduce kernels");
  internal::throw_cuda(
      "socket-run",
      cudaMemcpyAsync(slot.host_status, status, sizeof(*status),
                      cudaMemcpyDeviceToHost, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts, "D2H Socket status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(*status));
  internal::throw_cuda(
      "socket-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Socket complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "socket-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      world, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(Socket complete)");
  consumer.finish();
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size, *slot.host_status,
      slot.complete);
  if (production_agreement.has_error) {
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  float dispatch_ms = 0.0F;
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket total)");
  internal::throw_cuda(
      "socket-timing",
      cudaEventElapsedTime(&dispatch_ms, slot.start, slot.dispatch_complete),
      id.calling_rank, world, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Socket dispatch)");
  const auto gpus_per_node = world / id.num_nodes;
  const auto source_node = id.calling_rank / gpus_per_node;
  std::uint64_t remote_rows = 0;
  for (std::uint32_t peer = 0; peer < world; ++peer)
    if (peer / gpus_per_node != source_node)
      remote_rows += send_prefix[peer + 1U] - send_prefix[peer];
  const auto remote_bytes = checked_mul(
      checked_mul(remote_rows, config.hidden_size, "Socket remote elements"),
      sizeof(transport::BFloat16), "Socket remote bytes");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->socket_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend =
      transport::ActivationTransportBackend::NcclSocketFallback;
  result->audit.packet_count = route_count;
  result->audit.fanout_count = incoming;
  result->audit.raw_cross_node_payload_bytes = remote_bytes;
  result->audit.deduplicated_cross_node_payload_bytes = remote_bytes;
  result->audit.actual_dispatch_peer_bytes = remote_bytes;
  result->audit.actual_return_peer_bytes = remote_bytes;
  result->audit.derived_route_input_bytes =
      route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes = route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.kernel_launches = 7;
  result->audit.dispatch_kernel_blocks = route_count;
  result->audit.return_kernel_blocks = incoming;
  result->audit.reduce_kernel_blocks = id.num_tokens;
  // Only fixed-width control/count/status metadata is staged. No activation
  // payload crosses the host boundary.
  result->audit.activation_d2h_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.dispatch_relay_us =
      static_cast<double>(dispatch_ms) * 1000.0;
  result->timings.combine_reduce_us =
      result->timings.total_us - result->timings.dispatch_relay_us;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
}

DeviceActivationTransportHandle GinRdmaActivationTransport::run_transport_slice(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const transport::ActivationTransportConfig &config,
    const ActivationTransportRunOptions &options) {
#if !__has_include(<nccl_device.h>)
  (void)dispatch;
  (void)activations;
  (void)config;
  (void)options;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-run, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  if (!impl_ || !impl_->backend || !dispatch || !activations)
    transport_error(NotifyErrorCode::InvalidHandle, "gin-run",
                    "backend/dispatch/activation", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  if (id.num_nodes != backend->num_nodes ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=runtime GIN team topology and exact rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.producer_execution.same_instance(
          backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (!state->config.enable_gin_rdma ||
      config.hidden_size > state->config.max_hidden_size ||
      id.world_size > state->config.max_world_size ||
      id.top_k > state->config.max_top_k ||
      activation.slot->index >= backend->windows.size())
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "pool geometry/window", id.calling_rank,
                    "within registered GIN pool capacity",
                    "exceeded/disabled");
  if (config.gate_placement !=
      transport::ActivationGatePlacement::SourceAfter)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-run",
                    "gate_placement", id.calling_rank, "SourceAfter",
                    "unsupported");
  if (options.cancellation.cancellation_requested())
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=gin-run, field=cancellation, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=not requested, actual=requested");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  if (view.sizes.incoming_row_count >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count >
          checked_mul(state->config.max_tokens_per_rank,
                      state->config.max_top_k, "GIN route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "gin-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");
  SlotFailureGuard failure_guard{activation.slot, false};
  const auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  internal::throw_cuda(
      "gin-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(producer-ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN start)");
  internal::throw_cuda(
      "gin-run", cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                                  sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN status)");
  internal::throw_cuda(
      "gin-run",
      cudaMemsetAsync(at<void>(slot.gin_window,
                               state->gin_layout.barrier), 0,
                      sizeof(std::uint64_t), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaMemsetAsync(GIN reset barrier)");
  gin_signal_reset_kernel<<<backend->gin_context_count, 1, 0,
                               exec->stream>>>(backend->dev_comm,
                                               backend->gin_context_count);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "GIN reset kernel launch");
  auto *barrier = at<std::uint64_t>(slot.gin_window,
                                    state->gin_layout.barrier);
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(after GIN reset)");

  GinTransportSliceArgs args{};
  args.dispatch = view;
  args.input = activation.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.hidden = config.hidden_size;
  args.epoch = activation.slot->epoch;
  args.dev_comm = backend->dev_comm;
  args.window = backend->windows[activation.slot->index];
  args.compute_base = slot.device;
  args.window_base = slot.gin_window;
  args.send_header_offset = state->gin_layout.send_packet_headers;
  args.receive_header_offset = state->gin_layout.receive_packet_headers;
  args.send_fanout_offset = state->gin_layout.send_fanout;
  args.receive_fanout_offset = state->gin_layout.receive_fanout;
  args.send_payload_offset = state->gin_layout.send_payload;
  args.receive_payload_offset = state->gin_layout.receive_payload;
  args.canonical_offset = layout.canonical_rows;
  args.return_staging_offset = state->gin_layout.return_staging;
  args.return_receive_offset = state->gin_layout.return_receive;
  args.combine_compute_offset = layout.combine_return;
  args.combined_offset = layout.combined;
  args.status_offset = layout.status;
  args.peer_bases_offset = layout.peer_bases;
  args.local_world_size = backend->local_world_size;
  args.node = backend->node;
  args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  args.identity_expert =
      config.expert_transform ==
      transport::ActivationExpertTransform::Identity;
  gin_transport_slice_kernel<<<1, 256, 0, exec->stream>>>(args);
  internal::throw_cuda(
      "gin-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "GIN activation persistent kernel launch");
  internal::throw_cuda(
      "gin-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H GIN status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "gin-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(GIN complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);
  internal::throw_cuda(
      "gin-run", cudaEventSynchronize(slot.complete), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventSynchronize(GIN complete)");
  // A rank may destroy its registered window immediately after its result and
  // transport object leave scope. Rendezvous only after every device worker
  // has completed all remote puts so no peer can deregister early.
  internal::throw_nccl(
      "gin-run",
      ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum,
                    backend->nccl->comm, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "ncclAllReduce(GIN terminal window lease)");
  internal::throw_cuda(
      "gin-run", cudaStreamSynchronize(exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaStreamSynchronize(GIN terminal window lease)");
  consumer.finish();
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure, "gin-run",
                    "device data plane", id.calling_rank, "valid",
                    "device error field=" +
                        std::to_string(slot.host_status->error_field));

  float total_ms = 0.0F;
  internal::throw_cuda(
      "gin-timing", cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(GIN total)");
  auto result = std::make_shared<DeviceActivationTransportHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->gin_backend = backend;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->route_count = view.sizes.route_count;
  result->destination_rows = state->config.max_destination_rows_per_rank;
  result->audit.backend = transport::ActivationTransportBackend::Gin;
  result->audit.packet_count = slot.host_status->packets_done;
  result->audit.fanout_count = slot.host_status->fanout_done;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.derived_route_input_bytes =
      view.sizes.route_count * sizeof(std::int32_t);
  result->audit.derived_local_gate_bytes =
      view.sizes.route_count * sizeof(float);
  result->audit.derived_peer_gate_bytes = 0;
  result->audit.generation_memset_calls = 2;
  result->audit.generation_memset_bytes =
      sizeof(TransportDeviceStatus) + sizeof(std::uint64_t);
  result->audit.kernel_launches = 2;
  result->audit.dispatch_kernel_blocks = 1;
  result->audit.return_kernel_blocks = 1;
  result->audit.reduce_kernel_blocks = 1;
  result->audit.activation_d2h_bytes = 0;
  result->audit.activation_h2d_bytes = 0;
  result->audit.hot_path_allocations = 0;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  failure_guard.committed = true;
  return DeviceActivationTransportHandle(std::move(result));
#endif
}

DeviceForwardHandle LocalCudaP2pForwardMegaKernel::run(
    const DeviceDispatchHandle &dispatch,
    const BFloat16ActivationLease &activations,
    const ForwardWeightLease &weights,
    const ForwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !dispatch || !activations || !weights)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-run",
                    "backend/dispatch/activation/weights", 0, "valid",
                    "empty");
  const auto backend = impl_->backend;
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(backend);
  const auto socket_backend =
      std::dynamic_pointer_cast<SocketBackendState>(backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const bool use_socket = static_cast<bool>(socket_backend);
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  const auto &weight = *weights.impl_;
  if ((!use_gin && !use_socket && id.num_nodes != 1) ||
      (use_gin && id.num_nodes != gin_backend->num_nodes) ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=" +
            (use_gin ? "runtime GIN topology"
                     : use_socket ? "runtime NCCL Socket topology"
                                  : "one local node") +
            " and exact NCCL world/rank, actual=" +
            std::to_string(id.num_nodes) + " nodes, world=" +
            std::to_string(id.world_size));
  if (activation.pool.get() != state.get() ||
      activation.generation != id.generation ||
      activation.tokens != id.num_tokens ||
      activation.hidden != config.hidden_size ||
      activation.device != backend->execution.device_ordinal() ||
      activation.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (weight.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      weight.device != backend->execution.device_ordinal() ||
      weight.hidden != config.hidden_size ||
      weight.intermediate != config.intermediate_size || !weight.allocation ||
      !weight.allocation->ready)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "weight provenance", id.calling_rank,
                    "exact session/execution/device/shape/ready", "mismatch");
  if (state->config.max_hidden_size != config.hidden_size ||
      state->config.max_intermediate_size != config.intermediate_size ||
      !state->slots[activation.slot->index].forward_maps_valid ||
      (config.cache_mode == ForwardCacheMode::CacheFc1InputOutput &&
       !state->config.enable_forward_cache))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "forward pool geometry/cache", id.calling_rank,
                    "exact runtime-shape TMA maps and requested cache storage",
                    "mismatch/missing");

  const bool main_shape =
      config.hidden_size == 7168 && config.intermediate_size == 3072;

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  auto view = ActivationTransportBridge::view(consumer);
  const auto dispatch_waves = use_gin
                                  ? ActivationTransportBridge::wave_view(consumer)
                                  : DispatchWaveKernelView{};
  if (weight.owned_experts != view.sizes.owned_expert_count)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                    "owned_experts", id.calling_rank,
                    std::to_string(view.sizes.owned_expert_count),
                    std::to_string(weight.owned_experts));
  SlotFailureGuard failure_guard{activation.slot, false};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[activation.slot->index];
  const auto &layout = state->layout;
  if (view.sizes.incoming_row_count >
          state->config.max_destination_rows_per_rank ||
      view.sizes.route_count > checked_mul(state->config.max_tokens_per_rank,
                                           state->config.max_top_k,
                                           "forward route capacity"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "rows/routes", id.calling_rank, "within pool capacity",
                    "exceeded");

  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "H2D initialize cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "forward-run", cudaStreamWaitEvent(exec->stream, slot.producer_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(activation ready)");
  internal::throw_cuda(
      "forward-run",
      cudaStreamWaitEvent(exec->stream, weight.allocation->ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaStreamWaitEvent(weights ready)");
  ActivationTransportBridge::wait_for_gates(dispatch, backend->execution);

  if (slot.transport_epoch >= (1ULL << 63U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "forward-run",
                    "transport_epoch", id.calling_rank, "<2^63-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  std::uint64_t socket_incoming = 0;
  std::uint64_t socket_dispatch_peer_bytes = 0;
  std::uint64_t socket_return_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_route_bytes = 0;
  if (use_socket) {
    // The fallback transport and the fused Forward are sequential consumers
    // of the same immutable Dispatch handle. Release the first token while
    // the host-enqueued Socket slice owns it, then reacquire before launching
    // the production Forward kernel; retaining both would exhaust the bounded
    // consumer-event pool and stall before the first payload collective.
    consumer = {};
    transport::ActivationTransportConfig socket_config{};
    socket_config.hidden_size = config.hidden_size;
    socket_config.channels = state->config.channels;
    socket_config.ring_slots_per_source_channel =
        state->config.ring_slots_per_source_channel;
    socket_config.max_inflight_generations =
        state->config.max_inflight_generations;
    socket_config.expert_transform =
        transport::ActivationExpertTransform::Identity;
    socket_config.gate_placement =
        transport::ActivationGatePlacement::SourceAfter;
    const auto socket_preload = impl_->socket_transport->run_transport_slice(
        dispatch, activations, socket_config, options);
    socket_incoming = socket_preload.audit().fanout_count;
    // Count the actual rank-to-rank NCCL payloads. The transport-only audit is
    // intentionally cross-node centric, whereas this full fallback also sends
    // aggregates between ranks on the same host. The preload performs an
    // identity return which is real lower-bound traffic and must not be hidden.
    const auto local_world_size = id.world_size / id.num_nodes;
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      if (peer == id.calling_rank)
        continue;
      const auto outgoing_rows =
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      const auto incoming_rows =
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
      const auto outgoing_bytes = checked_mul(
          checked_mul(outgoing_rows, config.hidden_size,
                      "Socket Forward preload outgoing elements"),
          sizeof(transport::BFloat16),
          "Socket Forward preload outgoing bytes");
      socket_dispatch_peer_bytes = checked_add(
          socket_dispatch_peer_bytes, outgoing_bytes,
          "Socket Forward total dispatch peer bytes");
      socket_return_peer_bytes = checked_add(
          socket_return_peer_bytes,
          checked_mul(checked_mul(incoming_rows, config.hidden_size,
                                  "Socket Forward preload return elements"),
                      sizeof(transport::BFloat16),
                      "Socket Forward preload return bytes"),
          "Socket Forward total preload return bytes");
      if (peer / local_world_size != id.calling_rank / local_world_size)
        socket_raw_cross_node_route_bytes = checked_add(
            socket_raw_cross_node_route_bytes, outgoing_bytes,
            "Socket Forward cross-node route bytes");
    }
    consumer = dispatch.acquire_consumer(backend->execution, expectation);
    view = ActivationTransportBridge::view(consumer);
  }
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.forward_start, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward start)");
  internal::throw_cuda(
      "forward-run",
      cudaMemsetAsync(at<void>(slot.device, layout.status), 0,
                      sizeof(TransportDeviceStatus), exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "clear forward status");
#if MEGAMOE_DEVICE_TIMELINE
  reset_timeline("forward-run", slot.forward_timeline,
                 state->timeline_layout, state->timeline_max_ctas,
                 exec->stream, id.calling_rank, id.world_size,
                 id.generation, backend->session);
#endif

  // Notify has already completed the rank-wide generation agreement on this
  // communicator.  The IPC team and per-slot peer mappings are session-static,
  // while payload visibility is ordered by the device release/acquire signals
  // below, so a second host-enqueued AllGather here only serialized E2E.

  if (use_gin) {
#if __has_include(<nccl_device.h>)
    auto *gin_barrier = at<std::uint64_t>(slot.gin_window,
                                          state->gin_layout.barrier);
    internal::throw_cuda(
        "forward-run",
        cudaMemsetAsync(gin_barrier, 0, sizeof(std::uint64_t), exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "reset Forward GIN barrier");
    gin_signal_reset_kernel<<<gin_backend->gin_context_count, 1, 0,
                                 exec->stream>>>(
        gin_backend->dev_comm, gin_backend->gin_context_count);
    internal::throw_cuda(
        "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "reset Forward GIN signal/counter");
    internal::throw_nccl(
        "forward-run",
        ncclAllReduce(gin_barrier, gin_barrier, 1, ncclUint64, ncclSum,
                      backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "ncclAllReduce(Forward GIN reset agreement)");
#endif
  }
  if (!use_socket) {
    initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
        slot.device, at<void *>(slot.device, layout.peer_bases),
        layout.ingress_published_sequence, layout.outgoing_route_sequence,
        layout.p2p_barrier_receive, layout.gin_local_compute_ready,
        layout.forward_x_arrived_rows, layout.backward_dy_arrived_rows,
        view.sizes.mtile_count, layout.forward_completion,
        layout.backward_completion, layout.forward_context_completion,
        layout.backward_context_completion,
        static_cast<std::uint32_t>((state->config.max_nodes - 1U) * 8U),
        id.generation,
        id.calling_rank, id.world_size, state->config.channels,
        generation_epoch,
        use_gin ? gin_backend->node * gin_backend->local_world_size : 0U,
        use_gin ? gin_backend->local_world_size : id.world_size);
  } else if (socket_incoming != 0) {
    socket_install_forward_headers_kernel<<<
        static_cast<unsigned int>((socket_incoming + 255ULL) / 256ULL), 256,
        0, exec->stream>>>(
        at<LocalP2pRouteHeader>(slot.device, layout.socket_header_receive),
        at<LocalP2pRouteHeader>(slot.device, layout.direct_route_headers),
        socket_incoming, id.generation, generation_epoch, id.calling_rank,
        state->config.max_destination_rows_per_rank,
        at<TransportDeviceStatus>(slot.device, layout.status));
  }
  internal::throw_cuda(
      "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize owner completion signals and wait peer setup epoch");
  ForwardKernelArgs args{};
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.input = activation.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(dispatch);
  args.w1 = weight.w1;
  args.w2 = weight.w2;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows =
      state->config.max_destination_rows_per_rank;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = state->config.channels;
  args.epoch = generation_epoch;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.use_gin = use_gin;
  args.use_socket = use_socket;
  args.local_world_size = id.world_size / id.num_nodes;
  args.node = id.calling_rank / args.local_world_size;
#if MEGAMOE_DEVICE_TIMELINE
  args.timeline = make_timeline_context(
      slot.forward_timeline, state->timeline_layout,
      state->timeline_max_ctas, id.calling_rank,
      profiling::TimelinePhase::Forward, id.generation);
#endif
#if MEGAMOE_BUILD_TESTING
  if (const auto *fault_rank =
          std::getenv("MEGAMOE_INTERNAL_FORWARD_STATUS_FAULT_RANK")) {
    const auto *fault_generation =
        std::getenv("MEGAMOE_INTERNAL_FORWARD_STATUS_FAULT_GENERATION");
    if (fault_generation == nullptr ||
        std::stoull(fault_generation) == id.generation)
      args.injected_status_error_rank = std::stoi(fault_rank);
  }
#endif
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[activation.slot->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.gin_send_header_offset = state->gin_layout.send_packet_headers;
    args.gin_receive_header_offset = state->gin_layout.receive_packet_headers;
    args.gin_send_dispatch_directory_offset =
        state->gin_layout.send_dispatch_directory;
    args.gin_receive_dispatch_directory_offset =
        state->gin_layout.receive_dispatch_directory;
    args.gin_send_fanout_offset = state->gin_layout.send_fanout;
    args.gin_receive_fanout_offset = state->gin_layout.receive_fanout;
    args.gin_send_payload_offset = state->gin_layout.send_payload;
    args.gin_receive_payload_offset = state->gin_layout.receive_payload;
    args.gin_return_staging_offset = state->gin_layout.return_staging;
    args.gin_return_receive_offset = state->gin_layout.return_receive;
    args.gin_production_error_records_offset =
        state->gin_layout.production_error_records;
    args.node = gin_backend->node;
    args.local_world_size = gin_backend->local_world_size;
    args.gin_context_count = gin_backend->gin_context_count;
    args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  }
  args.canonical_rows_offset = layout.canonical_rows;
  args.forward_output_rows_offset = state->config.enable_backward
                                        ? layout.backward_forward_output
                                        : layout.canonical_rows;
  args.direct_headers_offset = layout.direct_route_headers;
  args.contribution_offset = layout.contribution;
  args.return_ready_epoch_offset = layout.return_ready_epoch;
  args.combined_offset = layout.combined;
  args.ingress_published_offset = layout.ingress_published_sequence;
  args.outgoing_sequence_offset = layout.outgoing_route_sequence;
  args.status_offset = layout.status;
  args.forward_s_offset = layout.forward_s;
  args.forward_ab_cache_offset = layout.forward_ab_cache;
  args.forward_task_claim_offset = layout.forward_task_claim;
  args.forward_fc1_done_offset = layout.forward_fc1_done;
  args.x_arrived_rows_offset = layout.forward_x_arrived_rows;
  args.forward_output_ready_epoch_offset =
      layout.forward_output_ready_epoch;
  args.forward_return_chunk_directory_offset =
      layout.forward_return_chunk_directory;
  args.forward_return_chunk_state_offset =
      layout.forward_return_chunk_state;
  args.forward_return_chunk_ready_count_offset =
      layout.forward_return_chunk_ready_count;
  args.forward_return_fragment_index_offset =
      layout.forward_return_fragment_index;
  args.forward_return_source_compact_index_offset =
      layout.forward_return_source_compact_index;
  args.forward_return_chunk_capacity =
      layout.forward_return_chunk_capacity;
  args.forward_scheduler_offset = layout.forward_scheduler;
  args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
  args.terminal_epoch_offset = layout.terminal_epoch;
  args.production_error_records_offset = layout.production_error_records;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
#if MEGAMOE_DEBUG_VALIDATE
  const auto forward_progress_base =
      32ULL + static_cast<std::size_t>(state->config.max_owned_experts) * 8ULL;
  if (std::getenv("MEGAMOE_BACKWARD_PROGRESS_DUMP") != nullptr) {
    std::fill_n(slot.host_backward_progress + forward_progress_base,
                kForwardProgressWords + 2ULL * slot.generation_debug_words,
                0ULL);
    args.progress = reinterpret_cast<unsigned long long *>(
        slot.device_backward_progress + forward_progress_base);
    args.progress_contexts = slot.generation_debug_contexts;
  }
#endif
  args.cache_fc1 =
      config.cache_mode == ForwardCacheMode::CacheFc1InputOutput;

  int cluster_supported = 0;
  int max_active_clusters = 0;
  if (!main_shape) {
    cudaLaunchAttribute fallback_attribute{};
    fallback_attribute.id = cudaLaunchAttributeCooperative;
    fallback_attribute.val.cooperative = 1;
    cudaLaunchConfig_t fallback_launch{};
    fallback_launch.gridDim = dim3(1, 1, 1);
    fallback_launch.blockDim = dim3(256, 1, 1);
    fallback_launch.stream = exec->stream;
    fallback_launch.attrs = &fallback_attribute;
    fallback_launch.numAttrs = 1;
    internal::throw_cuda(
        "forward-run",
        cudaLaunchKernelEx(&fallback_launch,
                           runtime_shape_forward_megakernel, args),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "single persistent runtime-shape Forward launch");
  } else {
  int sms = 0;
  internal::throw_cuda(
      "forward-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDeviceGetAttribute(SM count)");
  const auto grid_blocks = static_cast<unsigned int>(sms);
  if (grid_blocks == 0)
    throw NotifyDispatchError(NotifyErrorCode::BackendUnavailable,
                              "phase=forward-run, field=SM count, rank=" +
                                  std::to_string(id.calling_rank) +
                                  ", expected>=2, actual=" +
                                  std::to_string(sms));
  const std::size_t forward_smem =
      (4U * 128U * 64U + 4U * 256U * 64U) *
          sizeof(transport::BFloat16) +
      8U * sizeof(std::uint64_t) +
      static_cast<std::size_t>(id.world_size) * sizeof(std::uint32_t);
  const bool remote_transport = use_gin || use_socket;
  internal::throw_cuda(
      "forward-run",
      remote_transport
          ? cudaFuncSetAttribute(local_p2p_forward_megakernel<false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(forward_smem))
          : cudaFuncSetAttribute(local_p2p_forward_megakernel<true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(forward_smem)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaFuncSetAttribute(dynamic smem)");
  cudaLaunchAttribute attributes[1]{};
  attributes[0].id = cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(grid_blocks, 1, 1);
  launch.blockDim = dim3(384, 1, 1);
  launch.dynamicSmemBytes = forward_smem;
  launch.stream = exec->stream;
  launch.attrs = attributes;
  launch.numAttrs = 1;
  internal::throw_cuda(
      "forward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDevAttrClusterLaunch");
  const auto *forward_kernel = remote_transport
                                   ? reinterpret_cast<const void *>(
                                         local_p2p_forward_megakernel<false>)
                                   : reinterpret_cast<const void *>(
                                         local_p2p_forward_megakernel<true>);
  internal::throw_cuda(
      "forward-run",
      cudaOccupancyMaxActiveClusters(&max_active_clusters, forward_kernel,
                                     &launch),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaOccupancyMaxActiveClusters");
  const auto launch_forward = remote_transport
                                  ? cudaLaunchKernelEx(
                                        &launch,
                                        local_p2p_forward_megakernel<false>,
                                        args, slot.forward_x_map,
                                        slot.forward_s_map, weight.w1_map,
                                        weight.w2_map)
                                  : cudaLaunchKernelEx(
                                        &launch,
                                        local_p2p_forward_megakernel<true>,
                                        args, slot.forward_x_map,
                                        slot.forward_s_map, weight.w1_map,
                                        weight.w2_map);
  internal::throw_cuda(
      "forward-run", launch_forward,
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "single persistent WGMMA/TMA forward launch");
  }
  if (use_socket) {
    std::vector<std::uint64_t> send_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    std::vector<std::uint64_t> receive_prefix(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      send_prefix[peer + 1U] =
          send_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      receive_prefix[peer + 1U] =
          receive_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
    }
    if (send_prefix[id.world_size] != view.sizes.route_count ||
        receive_prefix[id.world_size] != socket_incoming)
      transport_error(NotifyErrorCode::TransportProtocolFailure,
                      "socket-forward-return", "route counts",
                      id.calling_rank, "exact preload counts", "mismatch");
    auto *send_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_send);
    auto *receive_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_receive);
    auto *send_payload = at<transport::BFloat16>(
        slot.device, layout.contribution);
    auto *receive_payload = at<transport::BFloat16>(
        slot.device, layout.socket_payload_receive);
    auto *forward_output = at<transport::BFloat16>(
        slot.device, args.forward_output_rows_offset);
    if (socket_incoming != 0)
      socket_gather_forward_returns_kernel<<<
          static_cast<unsigned int>(socket_incoming), 256, 0,
          exec->stream>>>(receive_headers, forward_output, receive_payload,
                          socket_incoming, config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "gather canonical FC2 rows");
    internal::throw_nccl(
        "socket-forward-return", ncclGroupStart(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupStart(Socket FC return)");
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      const auto send_rows = receive_prefix[peer + 1U] - receive_prefix[peer];
      const auto receive_rows = send_prefix[peer + 1U] - send_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != receive_rows)
          transport_error(NotifyErrorCode::TransportProtocolFailure,
                          "socket-forward-return", "self rows",
                          id.calling_rank, std::to_string(send_rows),
                          std::to_string(receive_rows));
        if (send_rows != 0)
          internal::throw_cuda(
              "socket-forward-return",
              cudaMemcpyAsync(
                  send_payload + send_prefix[peer] * config.hidden_size,
                  receive_payload +
                      receive_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket Forward self return bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Socket Forward self return");
        continue;
      }
      socket_return_peer_bytes = checked_add(
          socket_return_peer_bytes,
          checked_mul(checked_mul(send_rows, config.hidden_size,
                                  "Socket Forward remote return elements"),
                      sizeof(transport::BFloat16),
                      "Socket Forward remote return bytes"),
          "Socket Forward total remote return bytes");
      if (send_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclSend(receive_payload +
                         receive_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Socket Forward return)");
      if (receive_rows != 0)
        internal::throw_nccl(
            "socket-forward-return",
            ncclRecv(send_payload + send_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Socket Forward return)");
    }
    internal::throw_nccl(
        "socket-forward-return", ncclGroupEnd(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupEnd(Socket Forward return)");
    if (view.sizes.route_count != 0)
      socket_scatter_returns_kernel<<<
          static_cast<unsigned int>(view.sizes.route_count), 256, 0,
          exec->stream>>>(send_headers, send_payload,
                          at<transport::BFloat16>(slot.device,
                                                  layout.combine_return),
                          view.sizes.route_count, config.hidden_size);
    if (id.num_tokens != 0)
      socket_gate_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens),
                                  256, 0, exec->stream>>>(
          view, ActivationTransportBridge::gate_pointer(dispatch),
          at<transport::BFloat16>(slot.device, layout.combine_return),
          at<transport::BFloat16>(slot.device, layout.combined),
          config.hidden_size);
    internal::throw_cuda(
        "socket-forward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "Socket scatter/gate-after reduce");
  }
  internal::throw_cuda(
      "forward-run",
      cudaEventRecord(slot.forward_output_ready, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "cudaEventRecord(Forward output/cache ready)");
  // Local P2P relies on the peer-visible terminal epoch. GIN payload and
  // return consumers have already acquired their per-generation signals;
  // the next Notify/reset agreement is the registered-window reuse boundary.
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.host_forward_status,
                      at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H forward status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.forward_complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward complete)");
  slot.completion_recorded = true;

  if (options.defer_forward_completion_to_backward) {
    if (!state->config.enable_backward ||
        config.cache_mode != ForwardCacheMode::CacheFc1InputOutput)
      transport_error(NotifyErrorCode::InvalidRuntimeConfig, "forward-run",
                      "deferred Backward handoff", id.calling_rank,
                      "Backward-enabled cache Forward", "unsupported mode");
    auto result = std::make_shared<DeviceForwardHandle::Impl>();
    result->slot = activation.slot;
    result->pool = state;
    result->backend = backend;
    result->dispatch = dispatch;
    result->activations = activations;
    result->weights = weights;
    result->identity = {id.generation, id.calling_rank, id.world_size,
                        id.num_tokens, config.hidden_size};
    result->cache_mode = config.cache_mode;
    result->incoming_rows = view.sizes.incoming_row_count;
    result->intermediate = config.intermediate_size;
    result->audit.specialization =
        main_shape ? ForwardKernelSpecialization::Sm90WgmmaTma7168x3072
                   : ForwardKernelSpecialization::RuntimeShapeCorrectnessFallback;
    result->audit.persistent_kernel_launches = 1;
    result->audit.runtime_cluster_supported = cluster_supported != 0;
    result->audit.requested_cluster_size = 2;
    result->audit.max_active_clusters =
        static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
    result->audit.workspace_bytes = state->layout.total;
    result->socket_dispatch_peer_bytes = socket_dispatch_peer_bytes;
    result->socket_return_peer_bytes = socket_return_peer_bytes;
    result->socket_raw_cross_node_route_bytes =
        socket_raw_cross_node_route_bytes;
    if (use_socket) {
      result->audit.actual_dispatch_peer_bytes =
          socket_dispatch_peer_bytes;
      result->audit.actual_return_peer_bytes = socket_return_peer_bytes;
      result->audit.raw_cross_node_route_bytes =
          socket_raw_cross_node_route_bytes;
    }
    result->pending_consumer =
        std::make_shared<DeviceDispatchConsumerLease>(std::move(consumer));
    result->completion_pending = true;
    failure_guard.committed = true;
    return DeviceForwardHandle(std::move(result));
  }

  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
#if MEGAMOE_DEBUG_VALIDATE
  bool progress_dumped = false;
#endif
  while (true) {
    const auto query = cudaEventQuery(slot.forward_complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "forward-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(forward complete)");
    const auto expired = options.deadline_after.count() > 0 &&
                         std::chrono::steady_clock::now() - wait_start >=
                             options.deadline_after;
#if MEGAMOE_DEBUG_VALIDATE
    const auto stalled_for = std::chrono::steady_clock::now() - wait_start;
    if (!progress_dumped && use_gin && args.progress != nullptr &&
        stalled_for >= std::chrono::seconds(30)) {
      const auto forward_progress_base =
          32ULL + static_cast<std::size_t>(state->config.max_owned_experts) *
                      8ULL;
      const auto progress = [&](std::uint32_t word) {
        return __atomic_load_n(slot.host_backward_progress +
                                   forward_progress_base + word,
                               __ATOMIC_ACQUIRE);
      };
      std::cerr << "FORWARD_PROGRESS rank=" << id.calling_rank
                << " generation=" << id.generation
                << " entered=" << progress(kForwardEntered)
                << " setup=" << progress(kForwardSetupComplete)
                << " dispatch_stored=" << progress(kForwardDispatchStored)
                << " dispatch_observed="
                << progress(kForwardDispatchObserved)
                << " fc1=" << progress(kForwardFc1Complete)
                << " fc2=" << progress(kForwardFc2Complete)
                << " return=" << progress(kForwardReturnObserved)
                << " reduce=" << progress(kForwardReduceComplete)
                << " terminal=" << progress(kForwardTerminal);
      dump_generation_completion(
          std::cerr,
          slot.host_backward_progress + forward_progress_base +
              kForwardProgressWords,
          slot.generation_debug_contexts, "forward_completion");
      std::cerr << std::endl;
      progress_dumped = true;
    }
#endif
    if (!cancelled &&
        (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, args.epoch, __ATOMIC_RELEASE);
      internal::throw_cuda(
          "forward-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish forward cancel");
      internal::throw_cuda(
          "forward-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "drain forward cancel stream");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  consumer.finish();
  const auto production_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Forward, id.generation,
      id.generation, id.calling_rank, id.world_size,
      *slot.host_forward_status, slot.forward_complete);
  if (cancelled || slot.host_forward_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=forward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed forward, actual=cancelled");
  if (production_agreement.has_error) {
    // A logical device-status failure reached the fixed terminal agreement;
    // all peer work is drained and this slot is recoverable. Backend/CUDA
    // failures still leave the guard uncommitted and therefore quarantine.
    failure_guard.committed = true;
    throw_production_status_error(production_agreement);
  }

  float total_ms = 0.0F;
  internal::throw_cuda(
      "forward-timing",
      cudaEventElapsedTime(&total_ms, slot.forward_start,
                           slot.forward_output_ready),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(forward total)");
  auto result = std::make_shared<DeviceForwardHandle::Impl>();
  result->slot = activation.slot;
  result->pool = state;
  result->backend = backend;
  result->dispatch = dispatch;
  result->activations = activations;
  result->weights = weights;
  result->identity = {id.generation, id.calling_rank, id.world_size,
                      id.num_tokens, config.hidden_size};
  result->cache_mode = config.cache_mode;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->intermediate = config.intermediate_size;
  result->audit.specialization =
      main_shape ? ForwardKernelSpecialization::Sm90WgmmaTma7168x3072
                 : ForwardKernelSpecialization::RuntimeShapeCorrectnessFallback;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions =
      slot.host_forward_status->wgmma_instructions;
  result->audit.tma_loads = slot.host_forward_status->tma_loads;
  result->audit.tma_multicast_loads =
      slot.host_forward_status->tma_multicast_loads;
  result->audit.runtime_cluster_supported = cluster_supported != 0;
  result->audit.requested_cluster_size = 2;
  result->audit.runtime_cluster_size =
      slot.host_forward_status->runtime_cluster_size;
  result->audit.observed_cluster_rank_mask =
      slot.host_forward_status->cluster_rank_mask;
  result->audit.max_active_clusters =
      static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
  result->audit.fc1_tasks = slot.host_forward_status->fc1_tasks_done;
  result->audit.fc2_tasks = slot.host_forward_status->fc2_tasks_done;
  result->audit.actual_dispatch_peer_bytes =
      use_socket ? socket_dispatch_peer_bytes
                 : slot.host_forward_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      use_socket ? socket_return_peer_bytes
                 : slot.host_forward_status->actual_return_peer_bytes;
  result->audit.gin_dispatch_put_count =
      slot.host_forward_status->gin_dispatch_put_count;
  result->audit.raw_cross_node_route_bytes =
      use_socket ? socket_raw_cross_node_route_bytes
                 : slot.host_forward_status->raw_cross_node_payload_bytes;
  result->audit.gin_dispatch_payload_bytes =
      slot.host_forward_status->deduplicated_cross_node_payload_bytes;
  result->audit.gin_return_payload_bytes =
      slot.host_forward_status->gin_return_payload_bytes;
  result->audit.gin_return_put_count =
      slot.host_forward_status->gin_return_put_count;
  result->audit.gin_signal_count =
      slot.host_forward_status->gin_signal_count;
  result->audit.gin_metadata_bytes =
      slot.host_forward_status->gin_metadata_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.terminal_status_agreement_us =
      production_agreement.elapsed_us;
  const auto &status = *slot.host_forward_status;
  const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
    return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
  };
  result->timings.dispatch_us =
      elapsed(status.start_clock, status.dispatch_complete_clock);
  result->timings.fc1_swiglu_us =
      elapsed(status.fc1_start_clock, status.fc1_complete_clock);
  result->timings.fc2_us =
      elapsed(status.dispatch_blocks_done, status.fc2_complete_clock);
  result->timings.combine_return_us =
      elapsed(status.relay_blocks_done, status.return_complete_clock);
  result->timings.reduce_us =
      elapsed(status.reduced_tokens, status.reduce_blocks_done);
  result->timings.fc1_fc2_overlap_us =
      elapsed(status.dispatch_blocks_done, status.fc1_complete_clock);
  result->timings.fc2_return_overlap_us =
      elapsed(status.relay_blocks_done, status.fc2_complete_clock);
  result->timings.return_reduce_overlap_us =
      elapsed(status.reduced_tokens, status.return_complete_clock);
  failure_guard.committed = true;
  return DeviceForwardHandle(std::move(result));
}

const ActivationTransportExpectation &DeviceForwardHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  return impl_->identity;
}

const ForwardMegaKernelAudit &DeviceForwardHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  if (impl_->completion_pending)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle",
                    "completion", impl_->identity.rank,
                    "Backward handoff completed", "pending");
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
  if (impl_->completion_pending)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle",
                    "completion", impl_->identity.rank,
                    "Backward handoff completed", "pending");
  return impl_->timings;
}

SelectedOutputGradientLease testing_upload_output_gradient_for_backward(
    const DeviceForwardHandle &forward,
    const std::vector<transport::BFloat16> &host,
    const CudaExecution &execution) {
  if (!forward || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-dy-upload",
                    "forward/execution", 0, "valid", "empty");
  const auto &f = *forward.impl_;
  if (!f.pool->config.enable_backward ||
      !execution.token().same_instance(f.backend->execution.token()) ||
      execution.device_ordinal() != f.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-dy-upload", "pool/execution/device",
                    f.identity.rank, "exact Forward Backward-enabled context",
                    "mismatch");
  const auto elements = checked_mul(f.identity.tokens, f.identity.hidden,
                                    "backward dY upload elements");
  if (host.size() != checked_size(elements, "backward dY upload elements"))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-dy-upload", "shape", f.identity.rank,
                    std::to_string(elements), std::to_string(host.size()));
  const auto exec = internal::checked_execution_state(execution);
  auto &slot = f.pool->slots[f.slot->index];
  const auto bytes = checked_size(
      checked_mul(elements, sizeof(transport::BFloat16),
                  "backward dY upload bytes"),
      "backward dY upload bytes");
  if (bytes != 0) {
    internal::throw_cuda(
        "backward-dy-upload",
        cudaMemcpyAsync(at<void>(slot.device, f.pool->layout.backward_dy_input),
                        host.data(), bytes, cudaMemcpyHostToDevice,
                        exec->stream),
        f.identity.rank, f.identity.world, f.backend->session.session().identity(),
        f.identity.generation, ExchangeKind::ExpertCounts,
        "H2D explicit test dY upload");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "backward-dy-upload", cudaEventRecord(slot.backward_ready, exec->stream),
      f.identity.rank, f.identity.world, f.backend->session.session().identity(),
      f.identity.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(backward dY ready)");
  auto impl = std::make_shared<SelectedOutputGradientLease::Impl>();
  impl->slot = f.slot;
  impl->pool = f.pool;
  impl->pointer = at<const transport::BFloat16>(
      slot.device, f.pool->layout.backward_dy_input);
  impl->session_identity = f.pool->session_identity;
  impl->producer_execution = execution.token();
  impl->generation = f.identity.generation;
  impl->tokens = f.identity.tokens;
  impl->hidden = f.identity.hidden;
  impl->device = f.pool->device;
  impl->ready_recorded = true;
  return SelectedOutputGradientLease(std::move(impl));
}

SelectedOutputGradientLease testing_reissue_output_gradient_for_benchmark(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &storage,
    const CudaExecution &execution) {
  if (!forward || !storage || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-dy-reissue", "forward/storage/execution", 0,
                    "valid", "empty");
  const auto &f = *forward.impl_;
  const auto &source = *storage.impl_;
  if (source.pool.get() != f.pool.get() ||
      source.slot->index != f.slot->index ||
      source.pointer != at<const transport::BFloat16>(
                            f.pool->slots[f.slot->index].device,
                            f.pool->layout.backward_dy_input) ||
      source.tokens != f.identity.tokens || source.hidden != f.identity.hidden ||
      source.device != execution.device_ordinal() ||
      !execution.token().same_instance(f.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-dy-reissue", "storage provenance",
                    f.identity.rank,
                    "same preallocated slot/pool/execution/shape", "mismatch");
  auto impl = std::make_shared<SelectedOutputGradientLease::Impl>();
  impl->slot = f.slot;
  impl->pool = f.pool;
  impl->pointer = source.pointer;
  impl->session_identity = f.pool->session_identity;
  impl->producer_execution = execution.token();
  impl->generation = f.identity.generation;
  impl->tokens = f.identity.tokens;
  impl->hidden = f.identity.hidden;
  impl->device = source.device;
  impl->ready_recorded = true;
  return SelectedOutputGradientLease(std::move(impl));
}

SelectedOutputGradientLease ExternalCudaProducer::publish_output_gradient(
    const DeviceForwardHandle &forward,
    const transport::BFloat16 *device_data,
    std::uint64_t capacity_elements,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !forward.impl_ || !allocation_owner)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "external-dy-publish", "producer/forward/owner", 0,
                    "valid", "empty");
  const auto &f = *forward.impl_;
  if (state_->backend->context.session().stable_identity_bytes() !=
          f.pool->session_identity ||
      state_->backend->context.session().generation() != f.identity.generation ||
      state_->backend->context.calling_rank() != f.identity.rank ||
      state_->backend->device != f.pool->device || !f.pool->config.enable_backward)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "external-dy-publish", "provenance", f.identity.rank,
                    "exact Forward session/generation/rank/device",
                    "mismatch");
  const auto elements = checked_mul(f.identity.tokens, f.identity.hidden,
                                    "external dY elements");
  validate_external_cuda_range(device_data, elements, capacity_elements,
                               sizeof(transport::BFloat16), state_->backend->device,
                               "dY");
  auto &slot = f.pool->slots[f.slot->index];
  internal::throw_cuda(
      "external-dy-publish",
      cudaEventRecord(slot.backward_ready, state_->backend->stream), f.identity.rank,
      f.identity.world, state_->backend->context.session().identity(),
      f.identity.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(dY ready)");
  auto impl = std::make_shared<SelectedOutputGradientLease::Impl>();
  impl->slot = f.slot;
  impl->pool = f.pool;
  impl->pointer = device_data;
  impl->session_identity = f.pool->session_identity;
  impl->producer_execution = state_->backend->execution;
  impl->generation = f.identity.generation;
  impl->tokens = f.identity.tokens;
  impl->hidden = f.identity.hidden;
  impl->device = state_->backend->device;
  impl->ready_recorded = true;
  impl->external_owner = std::move(allocation_owner);
  impl->external_producer = state_->backend;
  return SelectedOutputGradientLease(std::move(impl));
}

DeviceBackwardHandle LocalCudaP2pBackwardMegaKernel::run(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &output_gradient,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  const auto backward_host_begin = std::chrono::steady_clock::now();
  if (!impl_ || !impl_->backend || !forward || !output_gradient)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "backend/forward/dY", 0, "valid", "empty");
  const auto backend = impl_->backend;
  const auto gin_backend =
      std::dynamic_pointer_cast<GinBackendState>(backend);
  const auto socket_backend =
      std::dynamic_pointer_cast<SocketBackendState>(backend);
  const bool use_gin = static_cast<bool>(gin_backend);
  const bool use_socket = static_cast<bool>(socket_backend);
  const auto &f = *forward.impl_;
  const auto &dy_lease = *output_gradient.impl_;
  const auto state = backend->pool;
  const auto &id = f.dispatch.identity();
  if (f.pool.get() != state.get() || dy_lease.pool.get() != state.get() ||
      f.slot.get() != dy_lease.slot.get() ||
      dy_lease.generation != id.generation ||
      dy_lease.tokens != id.num_tokens ||
      dy_lease.hidden != config.hidden_size ||
      dy_lease.device != backend->execution.device_ordinal() ||
      dy_lease.session_identity != state->session_identity ||
      !dy_lease.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "Forward/dY provenance", id.calling_rank,
                    "same pool/slot/session/generation/execution/shape/ready",
                    "mismatch");
  if ((!use_gin && !use_socket && id.num_nodes != 1) ||
      (use_gin && id.num_nodes != gin_backend->num_nodes) ||
      id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=backward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=" +
            (use_gin ? "runtime GIN topology"
                     : use_socket ? "runtime NCCL Socket topology"
                                  : "one-node NCCL team") +
            ", actual=nodes=" +
            std::to_string(id.num_nodes) + ", world=" +
            std::to_string(id.world_size));
  if (f.weights.hidden_size() != config.hidden_size ||
      f.weights.intermediate_size() != config.intermediate_size ||
      !f.weights.impl_->allocation->backward_prepared ||
      !f.weights.impl_->w1_transpose || !f.weights.impl_->w2_transpose ||
      !state->config.enable_backward ||
      state->config.max_owned_experts < f.weights.owned_experts())
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "specialization/pool/weights",
                    id.calling_rank,
                    "runtime-shape WGMMA/TMA maps and sufficient Backward pool",
                    "mismatch");
  const bool main_shape =
      config.hidden_size == 7168 && config.intermediate_size == 3072;
  if (config.weight_gradient_dtype != BackwardWeightGradientDType::BFloat16)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "dW dtype", id.calling_rank, "BF16", "unsupported");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  const bool had_deferred_forward = f.completion_pending;
  DeviceDispatchConsumerLease consumer;
  if (had_deferred_forward) {
    consumer = forward.transfer_consumer_to_backward();
  } else {
    consumer = f.dispatch.acquire_consumer(backend->execution, expectation);
  }
  const auto view = ActivationTransportBridge::view(consumer);
  const auto dispatch_waves = use_gin
                                  ? ActivationTransportBridge::wave_view(consumer)
                                  : DispatchWaveKernelView{};
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[f.slot->index];
  const auto &layout = state->layout;
  SlotFailureGuard failure_guard{f.slot, false};
  if (had_deferred_forward) {
    internal::throw_cuda(
        "backward-run",
        cudaStreamWaitEvent(exec->stream, slot.forward_output_ready),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(Forward cache/output ready)");
  }
  internal::throw_cuda(
      "backward-run", cudaStreamWaitEvent(exec->stream, slot.backward_ready),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(dY ready)");
  internal::throw_cuda(
      "backward-run",
      cudaStreamWaitEvent(exec->stream, f.weights.impl_->allocation->ready),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(weights ready)");
  __atomic_store_n(slot.host_cancel_epoch,
                   options.cancellation.cancellation_requested()
                       ? id.generation
                       : 0ULL,
                   __ATOMIC_RELEASE);
  internal::throw_cuda(
      "backward-run",
      cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                      sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "H2D initialize Backward cancel epoch");
  internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                            sizeof(std::uint64_t));
  internal::throw_cuda(
      "backward-run",
      cudaMemsetAsync(at<void>(slot.device, layout.backward_status), 0,
                      sizeof(BackwardDeviceStatus), exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "clear Backward status");
  if (slot.transport_epoch >= (1ULL << 63U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "backward-run",
                    "transport_epoch", id.calling_rank, "<2^63-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  std::vector<std::uint64_t> socket_send_prefix;
  std::vector<std::uint64_t> socket_receive_prefix;
  std::uint64_t socket_incoming = 0;
  std::uint64_t socket_dy_peer_bytes = 0;
  std::uint64_t socket_dx_peer_bytes = 0;
  std::uint64_t socket_raw_cross_node_dy_bytes = 0;
  if (use_socket) {
    auto *local_counts = at<unsigned long long>(
        slot.device, layout.p2p_barrier_send);
    auto *prefix = at<unsigned long long>(
        slot.device, layout.p2p_barrier_receive);
    auto *route_ids = at<unsigned int>(slot.device, layout.packet_claim);
    auto *send_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_send);
    auto *receive_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_receive);
    auto *send_payload = at<transport::BFloat16>(
        slot.device, layout.contribution);
    auto *receive_payload = at<transport::BFloat16>(
        slot.device, layout.socket_payload_receive);
    auto *send_gate = at<float>(slot.device, layout.backward_dgate);
    auto *receive_gate = at<float>(slot.device,
                                    layout.backward_dgate_partial);
    internal::throw_cuda(
        "socket-backward-dispatch",
        cudaMemsetAsync(local_counts, 0,
                        id.world_size * sizeof(std::uint64_t), exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "clear Socket Backward route counts");
    if (view.sizes.route_count != 0) {
      socket_route_count_kernel<<<
          static_cast<unsigned int>((view.sizes.route_count + 255ULL) /
                                    256ULL),
          256, 0, exec->stream>>>(view, local_counts);
      socket_route_prefix_kernel<<<1, 1, 0, exec->stream>>>(
          local_counts, prefix, id.world_size);
      socket_stable_route_ids_kernel<<<id.world_size, 256, 0,
                                       exec->stream>>>(
          view, prefix, route_ids);
      socket_pack_routes_kernel<<<
          static_cast<unsigned int>(view.sizes.route_count), 256, 0,
          exec->stream>>>(view, dy_lease.pointer, config.hidden_size,
                          generation_epoch, route_ids, send_headers,
                          send_payload);
      socket_pack_route_gates_kernel<<<
          static_cast<unsigned int>((view.sizes.route_count + 255ULL) /
                                    256ULL),
          256, 0, exec->stream>>>(
          route_ids, ActivationTransportBridge::gate_pointer(f.dispatch),
          send_gate, view.sizes.route_count);
    }
    internal::throw_cuda(
        "socket-backward-dispatch", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "pack Socket Backward dY/gate");
    socket_send_prefix.assign(static_cast<std::size_t>(id.world_size) + 1U,
                              0);
    socket_receive_prefix.assign(
        static_cast<std::size_t>(id.world_size) + 1U, 0);
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      socket_send_prefix[peer + 1U] =
          socket_send_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(id.calling_rank) *
                                      id.world_size +
                                  peer];
      socket_receive_prefix[peer + 1U] =
          socket_receive_prefix[peer] +
          slot.host_socket_counts[static_cast<std::size_t>(peer) *
                                      id.world_size +
                                  id.calling_rank];
    }
    socket_incoming = socket_receive_prefix[id.world_size];
    if (socket_send_prefix[id.world_size] != view.sizes.route_count ||
        socket_incoming != view.sizes.incoming_row_count)
      transport_error(NotifyErrorCode::TransportProtocolFailure,
                      "socket-backward-dispatch", "route counts",
                      id.calling_rank, "Notify/Forward exact counts",
                      "mismatch");
    internal::throw_nccl(
        "socket-backward-dispatch", ncclGroupStart(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupStart(Socket Backward)");
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      const auto send_rows =
          socket_send_prefix[peer + 1U] - socket_send_prefix[peer];
      const auto receive_rows =
          socket_receive_prefix[peer + 1U] - socket_receive_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != 0) {
          internal::throw_cuda(
              "socket-backward-dispatch",
              cudaMemcpyAsync(
                  receive_headers + socket_receive_prefix[peer],
                  send_headers + socket_send_prefix[peer],
                  checked_size(send_rows * sizeof(*send_headers),
                               "Socket Backward self header bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Backward self headers");
          internal::throw_cuda(
              "socket-backward-dispatch",
              cudaMemcpyAsync(
                  receive_payload +
                      socket_receive_prefix[peer] * config.hidden_size,
                  send_payload +
                      socket_send_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket Backward self dY bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Backward self dY");
          internal::throw_cuda(
              "socket-backward-dispatch",
              cudaMemcpyAsync(receive_gate + socket_receive_prefix[peer],
                              send_gate + socket_send_prefix[peer],
                              checked_size(send_rows * sizeof(float),
                                           "Socket Backward self gate bytes"),
                              cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Backward self gates");
        }
        continue;
      }
      const auto remote_dy_bytes = checked_mul(
          checked_mul(send_rows, config.hidden_size,
                      "Socket Backward remote dY elements"),
          sizeof(transport::BFloat16),
          "Socket Backward remote dY bytes");
      socket_dy_peer_bytes =
          checked_add(socket_dy_peer_bytes, remote_dy_bytes,
                      "Socket Backward total remote dY bytes");
      const auto local_world_size = id.world_size / id.num_nodes;
      if (peer / local_world_size != id.calling_rank / local_world_size)
        socket_raw_cross_node_dy_bytes = checked_add(
            socket_raw_cross_node_dy_bytes, remote_dy_bytes,
            "Socket Backward cross-node dY bytes");
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclSend(send_headers + socket_send_prefix[peer],
                     send_rows * sizeof(*send_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Backward headers)");
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclSend(send_payload +
                         socket_send_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Backward dY)");
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclSend(send_gate + socket_send_prefix[peer], send_rows,
                     ncclFloat32, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Backward gates)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclRecv(receive_headers + socket_receive_prefix[peer],
                     receive_rows * sizeof(*receive_headers), ncclUint8,
                     static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Backward headers)");
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclRecv(receive_payload +
                         socket_receive_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Backward dY)");
        internal::throw_nccl(
            "socket-backward-dispatch",
            ncclRecv(receive_gate + socket_receive_prefix[peer], receive_rows,
                     ncclFloat32, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Backward gates)");
      }
    }
    internal::throw_nccl(
        "socket-backward-dispatch", ncclGroupEnd(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupEnd(Socket Backward)");
    if (socket_incoming != 0)
      socket_relay_backward_inputs_kernel<<<
          static_cast<unsigned int>(socket_incoming), 256, 0,
          exec->stream>>>(
          receive_headers, receive_payload, receive_gate,
          at<transport::BFloat16>(slot.device, layout.backward_dy_canonical),
          at<float>(slot.device, layout.backward_canonical_gate),
          at<LocalP2pRouteHeader>(slot.device, layout.direct_route_headers),
          socket_incoming, config.hidden_size, id.generation,
          generation_epoch, id.calling_rank,
          state->config.max_destination_rows_per_rank,
          at<BackwardDeviceStatus>(slot.device, layout.backward_status));
    internal::throw_cuda(
        "socket-backward-dispatch", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "relay Socket Backward dY/gate");
  }
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    auto *gin_barrier = at<std::uint64_t>(slot.gin_window,
                                          state->gin_layout.barrier);
    internal::throw_cuda(
        "backward-run",
        cudaMemsetAsync(gin_barrier, 0, sizeof(std::uint64_t), exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "reset Backward GIN barrier");
    gin_signal_reset_kernel<<<gin_backend->gin_context_count, 1, 0,
                                 exec->stream>>>(
        gin_backend->dev_comm, gin_backend->gin_context_count);
    internal::throw_cuda(
        "backward-run", cudaGetLastError(), id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "reset Backward GIN signal/counter");
    internal::throw_nccl(
        "backward-run",
        ncclAllReduce(gin_barrier, gin_barrier, 1, ncclUint64, ncclSum,
                      backend->nccl->comm, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "ncclAllReduce(Backward GIN reset agreement)");
#endif
  }
  if (!use_socket)
    initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
      slot.device, at<void *>(slot.device, layout.peer_bases),
      layout.ingress_published_sequence, layout.outgoing_route_sequence,
      layout.p2p_barrier_receive, layout.gin_local_compute_ready,
      layout.forward_x_arrived_rows, layout.backward_dy_arrived_rows,
      view.sizes.mtile_count, layout.forward_completion,
      layout.backward_completion, layout.forward_context_completion,
      layout.backward_context_completion,
      static_cast<std::uint32_t>((state->config.max_nodes - 1U) * 8U),
      id.generation,
      id.calling_rank, id.world_size, state->config.channels, generation_epoch,
      use_gin ? gin_backend->node * gin_backend->local_world_size : 0U,
      use_gin ? gin_backend->local_world_size : id.world_size);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize Backward owner signals and peer setup epoch");

#if MEGAMOE_DEVICE_TIMELINE
  reset_timeline("backward-run", slot.backward_timeline,
                 state->timeline_layout, state->timeline_max_ctas,
                 exec->stream, id.calling_rank, id.world_size,
                 id.generation, backend->session);
#endif

  BackwardKernelArgs args{};
  args.dispatch = view;
  args.dispatch_waves = dispatch_waves;
  args.dy_input = dy_lease.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(f.dispatch);
  args.w1 = f.weights.impl_->w1;
  args.w2 = f.weights.impl_->w2;
  args.hidden = config.hidden_size;
  args.intermediate = config.intermediate_size;
  args.max_destination_rows = state->config.max_destination_rows_per_rank;
  args.reduction_rows =
      (state->config.max_destination_rows_per_rank +
       static_cast<std::uint64_t>(state->config.max_owned_experts) * 63ULL +
       63ULL) &
      ~63ULL;
  args.rank = id.calling_rank;
  args.world = id.world_size;
  args.channels = state->config.channels;
  args.epoch = generation_epoch;
  args.local_base = slot.device;
  args.peer_bases = at<void *>(slot.device, layout.peer_bases);
  args.use_gin = use_gin;
  args.use_socket = use_socket;
  args.local_world_size = id.world_size / id.num_nodes;
  args.node = id.calling_rank / args.local_world_size;
#if MEGAMOE_DEVICE_TIMELINE
  args.timeline = make_timeline_context(
      slot.backward_timeline, state->timeline_layout,
      state->timeline_max_ctas, id.calling_rank,
      profiling::TimelinePhase::Backward, id.generation);
#endif
#if MEGAMOE_BUILD_TESTING
  if (const auto *fault_rank =
          std::getenv("MEGAMOE_INTERNAL_BACKWARD_STATUS_FAULT_RANK")) {
    const auto *fault_generation =
        std::getenv("MEGAMOE_INTERNAL_BACKWARD_STATUS_FAULT_GENERATION");
    if (fault_generation == nullptr ||
        std::stoull(fault_generation) == id.generation)
      args.injected_status_error_rank = std::stoi(fault_rank);
  }
#endif
  if (use_gin) {
#if __has_include(<nccl_device.h>)
    args.gin_dev_comm = gin_backend->dev_comm;
    args.gin_window = gin_backend->windows[f.slot->index];
#endif
    args.gin_window_base = slot.gin_window;
    args.gin_send_header_offset = state->gin_layout.send_packet_headers;
    args.gin_receive_header_offset = state->gin_layout.receive_packet_headers;
    args.gin_send_dispatch_directory_offset =
        state->gin_layout.send_dispatch_directory;
    args.gin_receive_dispatch_directory_offset =
        state->gin_layout.receive_dispatch_directory;
    args.gin_send_fanout_offset = state->gin_layout.send_fanout;
    args.gin_receive_fanout_offset = state->gin_layout.receive_fanout;
    args.gin_send_gate_offset = state->gin_layout.send_gate;
    args.gin_receive_gate_offset = state->gin_layout.receive_gate;
    args.gin_send_payload_offset = state->gin_layout.send_payload;
    args.gin_receive_payload_offset = state->gin_layout.receive_payload;
    args.gin_return_staging_offset = state->gin_layout.return_staging;
    args.gin_return_receive_offset = state->gin_layout.return_receive;
    args.gin_return_gate_staging_offset =
        state->gin_layout.return_gate_staging;
    args.gin_return_gate_receive_offset =
        state->gin_layout.return_gate_receive;
    args.gin_production_error_records_offset =
        state->gin_layout.production_error_records;
    args.node = gin_backend->node;
    args.local_world_size = gin_backend->local_world_size;
    args.gin_context_count = gin_backend->gin_context_count;
    args.max_tokens_per_rank = state->config.max_tokens_per_rank;
  }
  args.canonical_x_offset = layout.canonical_rows;
  args.direct_headers_offset = layout.direct_route_headers;
  args.dy_canonical_offset = layout.backward_dy_canonical;
  args.canonical_gate_offset = layout.backward_canonical_gate;
  args.dab_offset = layout.backward_dab;
  args.recompute_ab_offset = layout.backward_recompute_ab;
  args.backward_s_offset = layout.forward_s;
  args.forward_ab_cache_offset = layout.forward_ab_cache;
  args.dx_canonical_offset = layout.backward_dx_canonical;
  args.dx_return_offset = layout.backward_dx_return;
  args.dx_combined_offset = layout.backward_dx_combined;
  args.dgate_partial_offset = layout.backward_dgate_partial;
  args.dgate_canonical_offset = layout.backward_dgate_canonical;
  args.dgate_offset = layout.backward_dgate;
  args.u_transpose_offset = layout.backward_u_transpose;
  args.dab_transpose_offset = layout.backward_dab_transpose;
  args.dw1_offset = layout.backward_dw1;
  args.dw2_offset = layout.backward_dw2;
  args.scheduler_offset = layout.backward_scheduler;
  args.dy_arrived_rows_offset = layout.backward_dy_arrived_rows;
  args.dw_timeline_offset = layout.backward_dw_timeline;
  args.claim_offset = layout.forward_task_claim;
  args.incoming_signal_offset = layout.ingress_published_sequence;
  args.outgoing_signal_offset = layout.outgoing_route_sequence;
  args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
  args.terminal_epoch_offset = layout.terminal_epoch;
  args.production_error_records_offset = layout.production_error_records;
  args.status_offset = layout.backward_status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  const auto backward_progress_words =
      32ULL + static_cast<std::size_t>(state->config.max_owned_experts) * 8ULL;
  std::fill_n(slot.host_backward_progress, backward_progress_words, 0ULL);
  args.progress = std::getenv("MEGAMOE_BACKWARD_PROGRESS_DUMP") != nullptr
                      ? reinterpret_cast<unsigned long long *>(
                            slot.device_backward_progress)
                      : nullptr;
  args.progress_experts = state->config.max_owned_experts;
#if MEGAMOE_DEBUG_VALIDATE
  args.progress_contexts = slot.generation_debug_contexts;
  if (args.progress != nullptr) {
    const auto backward_debug_base = backward_progress_words +
                                     kForwardProgressWords +
                                     slot.generation_debug_words;
    std::fill_n(slot.host_backward_progress + backward_debug_base,
                slot.generation_debug_words, 0ULL);
  }
#endif
  args.use_forward_cache =
      f.cache_mode == ForwardCacheMode::CacheFc1InputOutput;
#if MEGAMOE_DEBUG_VALIDATE
  args.debug_dv_offset = layout.backward_forward_output;
  args.debug_coverage_offset = layout.packet_claim;
  args.capture_stage_storage =
      std::getenv("MEGAMOE_BACKWARD_STAGE_HASHES") != nullptr;
#endif

  if (!main_shape) {
    cudaLaunchAttribute fallback_attribute{};
    fallback_attribute.id = cudaLaunchAttributeCooperative;
    fallback_attribute.val.cooperative = 1;
    cudaLaunchConfig_t fallback_launch{};
    fallback_launch.gridDim = dim3(1, 1, 1);
    fallback_launch.blockDim = dim3(256, 1, 1);
    fallback_launch.stream = exec->stream;
    fallback_launch.attrs = &fallback_attribute;
    fallback_launch.numAttrs = 1;
    internal::throw_cuda(
        "backward-run", cudaEventRecord(slot.start, exec->stream),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "cudaEventRecord(Backward start)");
    internal::throw_cuda(
        "backward-run",
        cudaLaunchKernelEx(&fallback_launch,
                           runtime_shape_backward_megakernel, args),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "single persistent runtime-shape Backward launch");
  } else {
  int sms = 0;
  internal::throw_cuda(
      "backward-run",
      cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaDeviceGetAttribute(SM count)");
  const auto grid_blocks = static_cast<unsigned int>(sms & ~1);
  const std::size_t smem =
      (4U * 128U * 64U + 4U * 256U * 64U) *
          sizeof(transport::BFloat16) +
      8U * sizeof(std::uint64_t);
  const bool remote_transport = use_gin || use_socket;
  internal::throw_cuda(
      "backward-run",
      remote_transport
          ? cudaFuncSetAttribute(local_p2p_backward_megakernel<false>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem))
          : cudaFuncSetAttribute(local_p2p_backward_megakernel<true>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(smem)),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaFuncSetAttribute(Backward dynamic smem)");
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeCooperative;
  attributes[0].val.cooperative = 1;
  attributes[1].id = cudaLaunchAttributeClusterDimension;
  attributes[1].val.clusterDim = {2U, 1U, 1U};
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(grid_blocks, 1, 1);
  launch.blockDim = dim3(384, 1, 1);
  launch.dynamicSmemBytes = smem;
  launch.stream = exec->stream;
  launch.attrs = attributes;
  launch.numAttrs = 2;
  internal::throw_cuda(
      "backward-run",
      cudaEventRecord(slot.start, exec->stream), id.calling_rank,
      id.world_size, backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(Backward start)");
  const auto launch_backward = remote_transport
      ? cudaLaunchKernelEx(
            &launch, local_p2p_backward_megakernel<false>, args,
            slot.forward_x_map, slot.backward_dy_map, slot.backward_dab_map,
            f.weights.impl_->w1_map, f.weights.impl_->w2_transpose_map,
            f.weights.impl_->w1_transpose_map, slot.backward_dy_mn_map,
            slot.backward_x_mn_map, slot.backward_u_transpose_map,
            slot.backward_dab_transpose_map, slot.backward_dw1_store_map)
      : cudaLaunchKernelEx(
            &launch, local_p2p_backward_megakernel<true>, args,
            slot.forward_x_map, slot.backward_dy_map, slot.backward_dab_map,
            f.weights.impl_->w1_map, f.weights.impl_->w2_transpose_map,
            f.weights.impl_->w1_transpose_map, slot.backward_dy_mn_map,
            slot.backward_x_mn_map, slot.backward_u_transpose_map,
            slot.backward_dab_transpose_map, slot.backward_dw1_store_map);
  internal::throw_cuda(
      "backward-run", launch_backward,
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "single persistent WGMMA/TMA Backward launch");
  }
  const auto backward_launch_enqueued = std::chrono::steady_clock::now();
  if (use_socket) {
    auto *send_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_send);
    auto *receive_headers = at<LocalP2pRouteHeader>(
        slot.device, layout.socket_header_receive);
    auto *send_payload = at<transport::BFloat16>(
        slot.device, layout.contribution);
    auto *receive_payload = at<transport::BFloat16>(
        slot.device, layout.socket_payload_receive);
    auto *return_gate_staging = at<float>(
        slot.device, layout.backward_dgate_partial);
    auto *return_gate_receive = at<float>(
        slot.device, layout.backward_canonical_gate);
    if (socket_incoming != 0)
      socket_gather_backward_returns_kernel<<<
          static_cast<unsigned int>(socket_incoming), 256, 0,
          exec->stream>>>(
          receive_headers,
          at<const transport::BFloat16>(slot.device,
                                         layout.backward_dx_canonical),
          at<const float>(slot.device, layout.backward_dgate_canonical),
          receive_payload, return_gate_staging, socket_incoming,
          config.hidden_size);
    internal::throw_cuda(
        "socket-backward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "gather Socket Backward dX/dGate");
    internal::throw_nccl(
        "socket-backward-return", ncclGroupStart(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupStart(Socket Backward return)");
    for (std::uint32_t peer = 0; peer < id.world_size; ++peer) {
      const auto send_rows =
          socket_receive_prefix[peer + 1U] - socket_receive_prefix[peer];
      const auto receive_rows =
          socket_send_prefix[peer + 1U] - socket_send_prefix[peer];
      if (peer == id.calling_rank) {
        if (send_rows != 0) {
          internal::throw_cuda(
              "socket-backward-return",
              cudaMemcpyAsync(
                  send_payload +
                      socket_send_prefix[peer] * config.hidden_size,
                  receive_payload +
                      socket_receive_prefix[peer] * config.hidden_size,
                  checked_size(send_rows * config.hidden_size *
                                   sizeof(transport::BFloat16),
                               "Socket Backward self dX bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Backward self dX");
          internal::throw_cuda(
              "socket-backward-return",
              cudaMemcpyAsync(
                  return_gate_receive + socket_send_prefix[peer],
                  return_gate_staging + socket_receive_prefix[peer],
                  checked_size(send_rows * sizeof(float),
                               "Socket Backward self dGate bytes"),
                  cudaMemcpyDeviceToDevice, exec->stream),
              id.calling_rank, id.world_size,
              backend->session.session().identity(), id.generation,
              ExchangeKind::ExpertCounts, "D2D Backward self dGate");
        }
        continue;
      }
      socket_dx_peer_bytes = checked_add(
          socket_dx_peer_bytes,
          checked_mul(checked_mul(send_rows, config.hidden_size,
                                  "Socket Backward remote dX elements"),
                      sizeof(transport::BFloat16),
                      "Socket Backward remote dX bytes"),
          "Socket Backward total remote dX bytes");
      if (send_rows != 0) {
        internal::throw_nccl(
            "socket-backward-return",
            ncclSend(receive_payload +
                         socket_receive_prefix[peer] * config.hidden_size,
                     send_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Backward dX)");
        internal::throw_nccl(
            "socket-backward-return",
            ncclSend(return_gate_staging + socket_receive_prefix[peer],
                     send_rows, ncclFloat32, static_cast<int>(peer),
                     backend->nccl->comm, exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclSend(Backward dGate)");
      }
      if (receive_rows != 0) {
        internal::throw_nccl(
            "socket-backward-return",
            ncclRecv(send_payload +
                         socket_send_prefix[peer] * config.hidden_size,
                     receive_rows * config.hidden_size *
                         sizeof(transport::BFloat16),
                     ncclUint8, static_cast<int>(peer), backend->nccl->comm,
                     exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Backward dX)");
        internal::throw_nccl(
            "socket-backward-return",
            ncclRecv(return_gate_receive + socket_send_prefix[peer],
                     receive_rows, ncclFloat32, static_cast<int>(peer),
                     backend->nccl->comm, exec->stream),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, "ncclRecv(Backward dGate)");
      }
    }
    internal::throw_nccl(
        "socket-backward-return", ncclGroupEnd(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "ncclGroupEnd(Socket Backward return)");
    if (view.sizes.route_count != 0)
      socket_scatter_backward_returns_kernel<<<
          static_cast<unsigned int>(view.sizes.route_count), 256, 0,
          exec->stream>>>(
          send_headers, send_payload, return_gate_receive,
          at<transport::BFloat16>(slot.device, layout.backward_dx_return),
          at<float>(slot.device, layout.backward_dgate),
#if MEGAMOE_DEBUG_VALIDATE
          args.capture_stage_storage
              ? at<std::uint32_t>(slot.device, layout.packet_claim)
              : nullptr,
#else
          nullptr,
#endif
          view.sizes.route_count, config.hidden_size);
    if (id.num_tokens != 0)
      socket_backward_reduce_kernel<<<static_cast<unsigned int>(id.num_tokens),
                                      256, 0, exec->stream>>>(
          view,
          at<const transport::BFloat16>(slot.device,
                                         layout.backward_dx_return),
          at<transport::BFloat16>(slot.device,
                                   layout.backward_dx_combined),
          config.hidden_size);
    internal::throw_cuda(
        "socket-backward-return", cudaGetLastError(), id.calling_rank,
        id.world_size, backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts, "scatter/reduce Socket Backward result");
  }
  const bool capture_hashes =
      std::getenv("MEGAMOE_BACKWARD_CAPTURE_HASH") != nullptr;
  if (capture_hashes) {
    const auto scratch_bytes =
        layout.backward_dw_timeline - layout.backward_scheduler;
    auto *device_chunks = reinterpret_cast<WeightHashPair *>(
        at<void>(slot.device, layout.backward_scheduler));
    const auto enqueue_hash = [&](const transport::BFloat16 *values,
                                  std::uint64_t elements,
                                  std::uint32_t hash_index,
                                  const char *field) {
      auto *device_result = reinterpret_cast<WeightHashPair *>(
          slot.device_backward_progress + kBackwardHashBase) + hash_index;
      if (elements == 0) {
        finalize_forward_weight_hash_kernel<<<1, 1, 0, exec->stream>>>(
            device_chunks, 0, 0, device_result);
        internal::throw_cuda(
            "backward-run", cudaGetLastError(),
            id.calling_rank, id.world_size,
            backend->session.session().identity(), id.generation,
            ExchangeKind::ExpertCounts, field);
        return;
      }
      const auto chunk_count =
          (elements + kWeightHashChunkElements - 1ULL) /
          kWeightHashChunkElements;
      const auto chunk_bytes = checked_size(
          checked_mul(chunk_count, sizeof(WeightHashPair),
                      "captured Backward hash chunks"),
          "captured Backward hash chunks");
      if (chunk_bytes > scratch_bytes)
        transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                        "backward-run", "captured hash scratch",
                        id.calling_rank, std::to_string(chunk_bytes),
                        std::to_string(scratch_bytes));
      hash_forward_weight_chunks_kernel<<<
          static_cast<unsigned>(chunk_count), 256, 0, exec->stream>>>(
          values, elements, device_chunks);
      internal::throw_cuda(
          "backward-run", cudaGetLastError(), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, field);
      finalize_forward_weight_hash_kernel<<<1, 1, 0, exec->stream>>>(
          device_chunks, chunk_count, elements, device_result);
      internal::throw_cuda(
          "backward-run", cudaGetLastError(), id.calling_rank,
          id.world_size, backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, field);
    };
    const auto dx_elements = checked_mul(id.num_tokens, config.hidden_size,
                                         "captured Backward dX hash");
    const auto dw1_elements = checked_mul(
        checked_mul(view.sizes.owned_expert_count,
                    2ULL * config.intermediate_size,
                    "captured Backward dW1 rows"),
        config.hidden_size, "captured Backward dW1 hash");
    const auto dw2_elements = checked_mul(
        checked_mul(view.sizes.owned_expert_count, config.hidden_size,
                    "captured Backward dW2 rows"),
        config.intermediate_size, "captured Backward dW2 hash");
    enqueue_hash(at<const transport::BFloat16>(
                     slot.device, layout.backward_dx_combined),
                 dx_elements, 0U, "capture Backward dX hash");
    enqueue_hash(at<const transport::BFloat16>(slot.device,
                                                layout.backward_dw1),
                 dw1_elements, 1U, "capture Backward dW1 hash");
    enqueue_hash(at<const transport::BFloat16>(slot.device,
                                                layout.backward_dw2),
                 dw2_elements, 2U, "capture Backward dW2 hash");
    enqueue_hash(reinterpret_cast<const transport::BFloat16 *>(
                     at<const float>(slot.device, layout.backward_dgate)),
                 checked_mul(view.sizes.route_count, 2ULL,
                             "captured Backward dGate words"),
                 3U, "capture Backward dGate hash");
  }
  static_assert(sizeof(BackwardDeviceStatus) <= sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "backward-run",
      cudaMemcpyAsync(slot.host_status,
                      at<void>(slot.device, layout.backward_status),
                      sizeof(BackwardDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "D2H Backward status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(BackwardDeviceStatus));
  internal::throw_cuda(
      "backward-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventRecord(Backward complete)");
  slot.completion_recorded = true;
  ActivationTransportBridge::submit(consumer);

  const auto wait_start = std::chrono::steady_clock::now();
  bool cancelled = false;
  bool progress_dumped = false;
  while (true) {
    const auto query = cudaEventQuery(slot.complete);
    if (query == cudaSuccess)
      break;
    if (query != cudaErrorNotReady)
      internal::throw_cuda(
          "backward-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(Backward complete)");
    const auto expired = options.deadline_after.count() > 0 &&
                         std::chrono::steady_clock::now() - wait_start >=
                             options.deadline_after;
    const auto stalled_for = std::chrono::steady_clock::now() - wait_start;
    if (!progress_dumped && stalled_for >= std::chrono::seconds(30)) {
      const auto progress = [&](std::uint32_t word) {
        return __atomic_load_n(slot.host_backward_progress + word,
                               __ATOMIC_ACQUIRE);
      };
      const auto forward_progress = [&](std::uint32_t word) {
        return __atomic_load_n(
            slot.host_backward_progress + backward_progress_words + word,
            __ATOMIC_ACQUIRE);
      };
      std::cerr << "BACKWARD_PROGRESS rank=" << id.calling_rank
                << " generation=" << id.generation
                << " forward_entered="
                << forward_progress(kForwardEntered)
                << " forward_setup="
                << forward_progress(kForwardSetupComplete)
                << " forward_dispatch_stored="
                << forward_progress(kForwardDispatchStored)
                << " forward_dispatch_observed="
                << forward_progress(kForwardDispatchObserved)
                << " forward_fc1="
                << forward_progress(kForwardFc1Complete)
                << " forward_fc2="
                << forward_progress(kForwardFc2Complete)
                << " forward_return="
                << forward_progress(kForwardReturnObserved)
                << " forward_reduce="
                << forward_progress(kForwardReduceComplete)
                << " forward_terminal="
                << forward_progress(kForwardTerminal)
                << " entered=" << progress(kBackwardEntered)
                << " setup_mblocks=" << progress(kBackwardSetupComplete)
                << " dy_done=" << progress(kBackwardDyComplete)
                << " dv_tasks=" << progress(kBackwardDvStarted)
                << " dv_done=" << progress(kBackwardDvComplete)
                << " dgate_rows=" << progress(kBackwardDgateComplete)
                << " dx_tasks=" << progress(kBackwardDxStarted)
                << " dx_done=" << progress(kBackwardDxComplete)
                << " dw2_tasks=" << progress(kBackwardDw2Started)
                << " dw2_done=" << progress(kBackwardDw2Complete)
                << " dw1_tasks=" << progress(kBackwardDw1Started)
                << " dw1_done=" << progress(kBackwardDw1Complete)
                << " reduced_tokens=" << progress(kBackwardReduceComplete)
                << " terminal=" << progress(kBackwardTerminal);
      for (std::uint32_t expert = 0;
           expert < view.sizes.owned_expert_count; ++expert)
        std::cerr << " expert[" << expert << "].rows="
                  << progress(kBackwardProgressExpertBase + expert * 8U);
#if MEGAMOE_DEBUG_VALIDATE
      if (use_gin) {
        const auto debug_words = slot.generation_debug_words;
        const auto backward_debug_base = backward_progress_words +
                                         kForwardProgressWords + debug_words;
        dump_generation_completion(
            std::cerr,
            slot.host_backward_progress + backward_debug_base,
            slot.generation_debug_contexts, "backward_completion");
      }
#endif
      std::cerr << std::endl;
      progress_dumped = true;
    }
    if (!cancelled &&
        (expired || options.cancellation.cancellation_requested())) {
      __atomic_store_n(slot.host_cancel_epoch, args.epoch, __ATOMIC_RELEASE);
      internal::throw_cuda(
          "backward-cancel",
          cudaMemcpyAsync(slot.device_cancel_epoch, slot.host_cancel_epoch,
                          sizeof(std::uint64_t), cudaMemcpyHostToDevice,
                          state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "H2D publish Backward cancel");
      internal::throw_cuda(
          "backward-cancel", cudaStreamSynchronize(state->cancel_stream),
          id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "drain Backward cancel stream");
      cancelled = true;
    }
    std::this_thread::yield();
  }
  const auto backward_completion_observed = std::chrono::steady_clock::now();
  consumer.finish();
  const auto backward_consumer_drained = std::chrono::steady_clock::now();
  const auto &status = *reinterpret_cast<const BackwardDeviceStatus *>(
      slot.host_status);
  ProductionStatusAgreementResult forward_agreement;
  if (had_deferred_forward)
    forward_agreement = agree_production_terminal_status(
        backend, slot, ProductionStatusPhase::Forward, id.generation,
        id.generation, id.calling_rank, id.world_size,
        *slot.host_forward_status, slot.complete);
  const auto backward_agreement = agree_production_terminal_status(
      backend, slot, ProductionStatusPhase::Backward, id.generation,
      id.generation, id.calling_rank, id.world_size, status,
      slot.complete);
  if (cancelled || status.error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=backward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed Backward, actual=cancelled");
  if (forward_agreement.has_error || backward_agreement.has_error) {
    failure_guard.committed = true;
    throw_production_status_error(forward_agreement);
    throw_production_status_error(backward_agreement);
  }
  double forward_consumer_drain_host_us = 0.0;
  if (had_deferred_forward) {
    auto &forward_result =
        const_cast<DeviceForwardHandle::Impl &>(*forward.impl_);
    const auto &forward_status = *slot.host_forward_status;
    float forward_total_ms = 0.0F;
    internal::throw_cuda(
        "forward-timing",
        cudaEventElapsedTime(&forward_total_ms, slot.forward_start,
                             slot.forward_output_ready),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(deferred Forward total)");
    forward_result.audit.wgmma_instructions =
        forward_status.wgmma_instructions;
    forward_result.audit.tma_loads = forward_status.tma_loads;
    forward_result.audit.tma_multicast_loads =
        forward_status.tma_multicast_loads;
    forward_result.audit.runtime_cluster_size =
        forward_status.runtime_cluster_size;
    forward_result.audit.observed_cluster_rank_mask =
        forward_status.cluster_rank_mask;
    forward_result.audit.fc1_tasks = forward_status.fc1_tasks_done;
    forward_result.audit.fc2_tasks = forward_status.fc2_tasks_done;
    forward_result.audit.actual_dispatch_peer_bytes =
        use_socket ? forward_result.socket_dispatch_peer_bytes
                   : forward_status.actual_dispatch_peer_bytes;
    forward_result.audit.actual_return_peer_bytes =
        use_socket ? forward_result.socket_return_peer_bytes
                   : forward_status.actual_return_peer_bytes;
    forward_result.audit.gin_dispatch_put_count =
        forward_status.gin_dispatch_put_count;
    forward_result.audit.raw_cross_node_route_bytes =
        use_socket ? forward_result.socket_raw_cross_node_route_bytes
                   : forward_status.raw_cross_node_payload_bytes;
    forward_result.audit.gin_dispatch_payload_bytes =
        forward_status.deduplicated_cross_node_payload_bytes;
    forward_result.audit.gin_return_payload_bytes =
        forward_status.gin_return_payload_bytes;
    forward_result.audit.gin_return_put_count =
        forward_status.gin_return_put_count;
    forward_result.audit.gin_signal_count = forward_status.gin_signal_count;
    forward_result.audit.gin_metadata_bytes =
        forward_status.gin_metadata_bytes;
    forward_result.timings.total_us =
        static_cast<double>(forward_total_ms) * 1000.0;
    forward_result.timings.terminal_status_agreement_us =
        forward_agreement.elapsed_us;
    const auto forward_elapsed = [](std::uint64_t begin,
                                    std::uint64_t end) {
      return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
    };
    forward_result.timings.dispatch_us = forward_elapsed(
        forward_status.start_clock, forward_status.dispatch_complete_clock);
    forward_result.timings.fc1_swiglu_us = forward_elapsed(
        forward_status.fc1_start_clock, forward_status.fc1_complete_clock);
    forward_result.timings.fc2_us = forward_elapsed(
        forward_status.dispatch_blocks_done,
        forward_status.fc2_complete_clock);
    forward_result.timings.combine_return_us = forward_elapsed(
        forward_status.relay_blocks_done,
        forward_status.return_complete_clock);
    forward_result.timings.reduce_us = forward_elapsed(
        forward_status.reduced_tokens,
        forward_status.reduce_blocks_done);
    forward_result.timings.fc1_fc2_overlap_us = forward_elapsed(
        forward_status.dispatch_blocks_done,
        forward_status.fc1_complete_clock);
    forward_result.timings.fc2_return_overlap_us = forward_elapsed(
        forward_status.relay_blocks_done,
        forward_status.fc2_complete_clock);
    forward_result.timings.return_reduce_overlap_us = forward_elapsed(
        forward_status.reduced_tokens,
        forward_status.return_complete_clock);
    forward_result.completion_pending = false;
  }
  float total_ms = 0.0F;
  internal::throw_cuda(
      "backward-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "cudaEventElapsedTime(Backward total)");
  const auto elapsed = [](std::uint64_t begin, std::uint64_t end) {
    return end >= begin ? static_cast<double>(end - begin) / 1000.0 : 0.0;
  };
  auto result = std::make_shared<DeviceBackwardHandle::Impl>();
  result->slot = f.slot;
  result->pool = state;
  result->backend = backend;
  result->forward = forward;
  result->output_gradient = output_gradient;
  result->identity = f.identity;
  result->incoming_rows = view.sizes.incoming_row_count;
  result->route_count = view.sizes.route_count;
  result->owned_experts = view.sizes.owned_expert_count;
  result->hidden = config.hidden_size;
  result->intermediate = config.intermediate_size;
  result->used_forward_cache = args.use_forward_cache;
#if MEGAMOE_DEBUG_VALIDATE
  result->stage_storage_captured = args.capture_stage_storage;
#endif
  result->padded_reduction_rows = status.padded_reduction_rows;
  result->hashes_captured = capture_hashes;
  if (capture_hashes) {
    const auto *captured = reinterpret_cast<const WeightHashPair *>(
        slot.host_backward_progress + kBackwardHashBase);
    const auto dx_elements = checked_mul(id.num_tokens, config.hidden_size,
                                         "captured dX hash elements");
    const auto dw1_elements = checked_mul(
        checked_mul(view.sizes.owned_expert_count,
                    2ULL * config.intermediate_size,
                    "captured dW1 hash rows"),
        config.hidden_size, "captured dW1 hash elements");
    const auto dw2_elements = checked_mul(
        checked_mul(view.sizes.owned_expert_count, config.hidden_size,
                    "captured dW2 hash rows"),
        config.intermediate_size, "captured dW2 hash elements");
    result->hashes.dx = {captured[0].low, captured[0].high, dx_elements};
    result->hashes.dw1 = {captured[1].low, captured[1].high, dw1_elements};
    result->hashes.dw2 = {captured[2].low, captured[2].high, dw2_elements};
    result->hashes.dgate = {
        captured[3].low, captured[3].high,
        checked_mul(view.sizes.route_count, 2ULL,
                    "captured dGate hash BF16 words")};
  }
  result->audit.specialization =
      main_shape ? BackwardKernelSpecialization::Sm90WgmmaTma7168x3072
                 : BackwardKernelSpecialization::RuntimeShapeCorrectnessFallback;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions = status.wgmma_instructions;
  result->audit.tma_loads = status.tma_loads;
  result->audit.dv_tasks = status.dv_tasks;
  result->audit.dx_tasks = status.dx_tasks;
  result->audit.dw1_tasks = status.dw1_tasks;
  result->audit.dw2_tasks = status.dw2_tasks;
  result->audit.actual_dy_peer_bytes =
      use_socket ? socket_dy_peer_bytes : status.actual_dy_peer_bytes;
  result->audit.actual_dx_peer_bytes =
      use_socket ? socket_dx_peer_bytes : status.actual_dx_peer_bytes;
  result->audit.gin_dispatch_put_count = status.gin_dispatch_put_count;
  result->audit.raw_cross_node_dy_bytes =
      use_socket ? socket_raw_cross_node_dy_bytes
                 : status.raw_cross_node_dy_bytes;
  result->audit.gin_dy_payload_bytes = status.gin_dispatch_payload_bytes;
  result->audit.gin_dx_payload_bytes = status.gin_return_payload_bytes;
  result->audit.gin_return_put_count = status.gin_return_put_count;
  result->audit.gin_signal_count = status.gin_signal_count;
  result->audit.gin_metadata_bytes = status.gin_metadata_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->audit.recomputed_fc1 = !args.use_forward_cache;
  result->audit.weight_gradient_dtype = config.weight_gradient_dtype;
  result->audit.dispatch_consumer_acquires = had_deferred_forward ? 1U : 0U;
  result->audit.dispatch_consumer_releases = had_deferred_forward ? 1U : 0U;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  result->timings.terminal_status_agreement_us =
      backward_agreement.elapsed_us;
  result->timings.dy_dispatch_us =
      elapsed(status.start_clock, status.dy_dispatch_complete_clock);
  result->timings.dv_swiglu_backward_us = elapsed(
      status.dy_dispatch_complete_clock, status.dv_complete_clock);
  result->timings.dgate_reduce_us =
      elapsed(status.dv_complete_clock, status.dgate_complete_clock);
  result->timings.transpose_us =
      elapsed(status.dgate_complete_clock, status.transpose_complete_clock);
  result->timings.dx_us = elapsed(status.transpose_complete_clock,
                                  status.dx_complete_clock);
  result->timings.dw1_us = elapsed(status.dw1_first_start_clock,
                                  status.dw1_last_complete_clock);
  result->timings.dw2_us = elapsed(status.dw2_first_start_clock,
                                  status.dw2_last_complete_clock);
  result->timings.dw1_ready_to_start_us = elapsed(
      status.dw1_first_ready_clock, status.dw1_first_start_clock);
  result->timings.dw2_ready_to_start_us = elapsed(
      status.dw2_first_ready_clock, status.dw2_first_start_clock);
  result->timings.dw1_max_expert_ready_to_start_us =
      static_cast<double>(status.dw1_max_ready_wait) / 1000.0;
  result->timings.dw2_max_expert_ready_to_start_us =
      static_cast<double>(status.dw2_max_ready_wait) / 1000.0;
  result->timings.dw1_tail_us = elapsed(
      max(status.dx_complete_clock, status.dw1_first_start_clock),
      status.dw1_last_complete_clock);
  result->timings.dw2_tail_us = elapsed(
      max(status.dx_complete_clock, status.dw2_first_start_clock),
      status.dw2_last_complete_clock);
  const auto overlap = [](std::uint64_t begin, std::uint64_t end,
                          std::uint64_t window_begin,
                          std::uint64_t window_end) {
    const auto first = max(begin, window_begin);
    const auto last = min(end, window_end);
    return last >= first ? static_cast<double>(last - first) / 1000.0 : 0.0;
  };
  result->timings.dw1_overlap_dv_dx_us = overlap(
      status.dw1_first_start_clock, status.dw1_last_complete_clock,
      status.dv_start_clock, status.dx_complete_clock);
  result->timings.dw2_overlap_dv_dx_us = overlap(
      status.dw2_first_start_clock, status.dw2_last_complete_clock,
      status.dv_start_clock, status.dx_complete_clock);
  result->timings.dx_combine_reduce_us = elapsed(
      max(status.dw1_complete_clock, status.dw2_complete_clock),
      status.dx_reduce_complete_clock);
  if (had_deferred_forward) {
    float handoff_ms = 0.0F;
    internal::throw_cuda(
        "backward-timing",
        cudaEventElapsedTime(&handoff_ms, slot.forward_output_ready,
                             slot.start),
        id.calling_rank, id.world_size,
        backend->session.session().identity(), id.generation,
        ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(Forward to Backward handoff)");
    result->timings.forward_to_backward_handoff_us =
        static_cast<double>(handoff_ms) * 1000.0;
  }
  result->timings.backward_setup_to_launch_host_us =
      std::chrono::duration<double, std::micro>(backward_launch_enqueued -
                                                backward_host_begin)
          .count();
  result->timings.forward_consumer_drain_host_us =
      forward_consumer_drain_host_us;
  result->timings.final_consumer_drain_host_us =
      std::chrono::duration<double, std::micro>(
          backward_consumer_drained - backward_completion_observed)
          .count();
  failure_guard.committed = true;
  return DeviceBackwardHandle(std::move(result));
}

std::vector<transport::BFloat16> download_forward_output_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "forward download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward download elements"));
  if (!output.empty()) {
    const auto bytes = checked_size(
        checked_mul(elements, sizeof(transport::BFloat16),
                    "forward download bytes"),
        "forward download bytes");
    internal::throw_cuda(
        "forward-download",
        cudaMemcpy(output.data(),
                   at<const void>(impl.pool->slots[impl.slot->index].device,
                                  impl.pool->layout.combined),
                   bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(forward output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

std::vector<transport::BFloat16> download_forward_cache_for_validation(
    const DeviceForwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-cache-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (impl.cache_mode != ForwardCacheMode::CacheFc1InputOutput ||
      impl.pool->layout.forward_ab_cache == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-cache-download", "cache_mode",
                    impl.identity.rank, "CacheFc1InputOutput", "disabled");
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "forward-cache-download", "device", impl.identity.rank,
                    "handle device", "foreign");
  const auto plane_elements = checked_mul(
      impl.incoming_rows, impl.intermediate, "forward cache rows");
  const auto elements = checked_mul(plane_elements, 2ULL,
                                    "forward A/B cache elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "forward cache elements"));
  if (!output.empty()) {
    const auto plane_bytes = checked_size(
        checked_mul(plane_elements, sizeof(transport::BFloat16),
                    "forward cache plane bytes"),
        "forward cache plane bytes");
    internal::throw_cuda(
        "forward-cache-download",
        cudaMemcpy(output.data(),
                   at<const void>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.forward_ab_cache),
                   plane_bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(A checkpoint)");
    const auto b_plane_elements = checked_mul(
        impl.pool->config.max_destination_rows_per_rank, impl.intermediate,
        "forward cache B plane offset");
    internal::throw_cuda(
        "forward-cache-download",
        cudaMemcpy(output.data() + plane_elements,
                   at<const transport::BFloat16>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.forward_ab_cache) +
                       b_plane_elements,
                   plane_bytes, cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "forward-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(B checkpoint)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              plane_bytes * 2ULL);
  }
  return output;
}

bool testing_forward_cache_ranges_disjoint(
    const DeviceForwardHandle &left, const DeviceForwardHandle &right) {
  if (!left || !right)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "forward-cache-isolation", "handle", 0,
                    "two valid handles", "empty");
  const auto &a = *left.impl_;
  const auto &b = *right.impl_;
  if (a.cache_mode != ForwardCacheMode::CacheFc1InputOutput ||
      b.cache_mode != ForwardCacheMode::CacheFc1InputOutput)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "forward-cache-isolation", "cache_mode", 0,
                    "CacheFc1InputOutput", "disabled");
  const auto bytes_for = [](const DeviceForwardHandle::Impl &impl) {
    return checked_size(
        checked_mul(
            checked_mul(impl.pool->config.max_destination_rows_per_rank,
                        impl.intermediate, "cache isolation plane elements"),
            2ULL * sizeof(transport::BFloat16),
            "cache isolation bytes"),
        "cache isolation bytes");
  };
  const auto a_begin = reinterpret_cast<std::uintptr_t>(
      at<void>(a.pool->slots[a.slot->index].device,
               a.pool->layout.forward_ab_cache));
  const auto b_begin = reinterpret_cast<std::uintptr_t>(
      at<void>(b.pool->slots[b.slot->index].device,
               b.pool->layout.forward_ab_cache));
  const auto a_bytes = bytes_for(a);
  const auto b_bytes = bytes_for(b);
  if (a_begin > std::numeric_limits<std::uintptr_t>::max() - a_bytes ||
      b_begin > std::numeric_limits<std::uintptr_t>::max() - b_bytes)
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "forward-cache-isolation", "address range", 0,
                    "checked native interval", "overflow");
  return a_begin + a_bytes <= b_begin || b_begin + b_bytes <= a_begin;
}

bool testing_backward_storage_ranges_disjoint(
    const DeviceBackwardHandle &left, const DeviceBackwardHandle &right) {
  if (!left || !right)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-storage-isolation", "handle", 0,
                    "two valid handles", "empty");
  const auto &a = *left.impl_;
  const auto &b = *right.impl_;
  if (a.pool.get() != b.pool.get())
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-storage-isolation", "pool", 0,
                    "same activation pool", "foreign");
  const auto a_begin = reinterpret_cast<std::uintptr_t>(
      a.pool->slots[a.slot->index].device);
  const auto b_begin = reinterpret_cast<std::uintptr_t>(
      b.pool->slots[b.slot->index].device);
  const auto bytes = a.pool->layout.total;
  if (a_begin > std::numeric_limits<std::uintptr_t>::max() - bytes ||
      b_begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    transport_error(NotifyErrorCode::ArithmeticOverflow,
                    "backward-storage-isolation", "arena interval", 0,
                    "checked native interval", "overflow");
  return a_begin + bytes <= b_begin || b_begin + bytes <= a_begin;
}

std::size_t testing_backward_slot_index(const DeviceBackwardHandle &handle) {
  if (!handle)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-slot-validation", "handle", 0,
                    "valid", "empty");
  return handle.impl_->slot->index;
}

BackwardValidationDownload download_backward_for_validation(
    const DeviceBackwardHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device ||
      !execution.token().same_instance(impl.backend->execution.token()))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-download", "execution/device",
                    impl.identity.rank, "exact Backward execution", "foreign");
  const auto exec = internal::checked_execution_state(execution);
  const auto &layout = impl.pool->layout;
  const auto &slot = impl.pool->slots[impl.slot->index];
  BackwardValidationDownload result;
  const auto dx_elements = checked_mul(impl.identity.tokens, impl.hidden,
                                       "backward download dX");
  const auto dw1_elements = checked_mul(
      checked_mul(impl.owned_experts, 2ULL * impl.intermediate,
                  "backward download dW1 rows"),
      impl.hidden, "backward download dW1");
  const auto dw2_elements = checked_mul(
      checked_mul(impl.owned_experts, impl.hidden,
                  "backward download dW2 rows"),
      impl.intermediate, "backward download dW2");
  result.dx.resize(checked_size(dx_elements, "backward download dX"));
  result.dgate.resize(checked_size(impl.route_count,
                                   "backward download dGate"));
  result.dw1.resize(checked_size(dw1_elements, "backward download dW1"));
  result.dw2.resize(checked_size(dw2_elements, "backward download dW2"));
  const auto sampled_plane = std::min<std::uint64_t>(impl.intermediate, 2048ULL);
  const auto sampled_elements = impl.incoming_rows == 0 ? 0ULL
                                                        : 2ULL * sampled_plane;
  result.sampled_dab.resize(
      checked_size(sampled_elements, "backward sampled dAB"));
  result.sampled_ab.resize(
      checked_size(sampled_elements, "backward sampled A/B"));
  const auto copy = [&](void *host, std::size_t offset,
                        std::uint64_t elements, std::size_t width,
                        const char *field) {
    if (elements == 0)
      return;
    const auto bytes = checked_size(checked_mul(elements, width, field), field);
    internal::throw_cuda(
        "backward-download",
        cudaMemcpyAsync(host, at<const void>(slot.device, offset), bytes,
                        cudaMemcpyDeviceToHost, exec->stream),
        impl.identity.rank, impl.identity.world,
        impl.backend->session.session().identity(), impl.identity.generation,
        ExchangeKind::ExpertCounts, field);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  };
  copy(result.dx.data(), layout.backward_dx_combined, dx_elements,
       sizeof(transport::BFloat16), "download dX");
  copy(result.dgate.data(), layout.backward_dgate, impl.route_count,
       sizeof(float), "download dGate");
  copy(result.dw1.data(), layout.backward_dw1, dw1_elements,
       sizeof(transport::BFloat16), "download dW1");
  copy(result.dw2.data(), layout.backward_dw2, dw2_elements,
       sizeof(transport::BFloat16), "download dW2");
  if (sampled_plane != 0 && impl.incoming_rows != 0) {
    const auto ab_base = impl.used_forward_cache ? layout.forward_ab_cache
                                                 : layout.backward_recompute_ab;
    const auto element_bytes = sizeof(transport::BFloat16);
    const auto dab_plane_bytes = checked_size(
        checked_mul(impl.intermediate, element_bytes,
                    "backward sampled dAB plane bytes"),
        "backward sampled dAB plane bytes");
    const auto ab_plane_bytes = checked_size(
        checked_mul(
            checked_mul(impl.pool->config.max_destination_rows_per_rank,
                        impl.intermediate, "backward sampled A/B plane"),
            element_bytes, "backward sampled A/B plane bytes"),
        "backward sampled A/B plane bytes");
    copy(result.sampled_dab.data(), layout.backward_dab, sampled_plane,
         element_bytes, "download sampled dA");
    copy(result.sampled_dab.data() + sampled_plane,
         layout.backward_dab + dab_plane_bytes, sampled_plane, element_bytes,
         "download sampled dB");
    copy(result.sampled_ab.data(), ab_base, sampled_plane,
         element_bytes, "download sampled A");
    copy(result.sampled_ab.data() + sampled_plane,
         ab_base + ab_plane_bytes, sampled_plane, element_bytes,
         "download sampled B");
  }
  internal::throw_cuda(
      "backward-download", cudaStreamSynchronize(exec->stream),
      impl.identity.rank, impl.identity.world,
      impl.backend->session.session().identity(), impl.identity.generation,
      ExchangeKind::ExpertCounts, "wait validation download");
  return result;
}

const ActivationTransportExpectation &
DeviceActivationTransportHandle::identity() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->identity;
}
const ActivationTransportDeviceAudit &
DeviceActivationTransportHandle::audit() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->audit;
}
const ActivationTransportTimings &
DeviceActivationTransportHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-handle",
                    "handle", 0, "valid", "empty");
  return impl_->timings;
}

ActivationTransportValidationDownload ActivationTransportBridge::download_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle, "transport-download",
                    "handle/execution", 0, "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "transport-download",
                    "device", impl.identity.rank, "handle device", "foreign");
  const auto &slot = impl.pool->slots[impl.slot->index];
  const auto hidden = impl.identity.hidden;
  const auto route_elements = checked_mul(impl.route_count, hidden,
                                          "download route elements");
  const auto combined_elements = checked_mul(impl.identity.tokens, hidden,
                                             "download combined elements");
  const auto canonical_elements =
      checked_mul(impl.destination_rows, hidden, "download canonical elements");
  ActivationTransportValidationDownload result;
  result.contribution_by_route.resize(
      checked_size(route_elements, "download route elements"));
  result.combined_tokens.resize(
      checked_size(combined_elements, "download combined elements"));
  result.canonical_expert_rows.resize(
      checked_size(canonical_elements, "download canonical elements"));
  auto copy = [&](void *destination, std::size_t offset, std::uint64_t count,
                  const char *name) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "transport-download",
        cudaMemcpy(destination, at<const void>(slot.device, offset),
                   checked_size(checked_mul(count,
                                            sizeof(transport::BFloat16), name),
                                name),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts, name);
  };
  const auto contribution_offset =
      impl.audit.backend ==
                  transport::ActivationTransportBackend::NcclSocketFallback ||
              impl.audit.backend ==
                  transport::ActivationTransportBackend::Gin
          ? impl.pool->layout.combine_return
          : impl.pool->layout.contribution;
  copy(result.contribution_by_route.data(), contribution_offset,
       route_elements, "download contributions");
  copy(result.combined_tokens.data(), impl.pool->layout.combined,
       combined_elements, "download combined");
  copy(result.canonical_expert_rows.data(), impl.pool->layout.canonical_rows,
       canonical_elements, "download canonical rows");
  result.protocol.dispatch_packets = slot.host_status->packets_done;
  result.protocol.relayed_fanout = slot.host_status->fanout_done;
  result.protocol.synthetic_contributions = slot.host_status->synthetic_done;
  result.protocol.combine_packets = slot.host_status->combine_done;
  result.protocol.reduced_tokens = slot.host_status->reduced_tokens;
  result.protocol.credit_stalls = slot.host_status->credit_stalls;
  result.protocol.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result.protocol.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
  return result;
}

ActivationTransportValidationDownload download_activation_transport_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  return ActivationTransportBridge::download_for_validation(handle, execution);
}

std::vector<transport::BFloat16> download_activation_output_for_validation(
    const DeviceActivationTransportHandle &handle,
    const CudaExecution &execution) {
  if (!handle || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "transport-output-download", "handle/execution", 0,
                    "valid", "empty");
  const auto &impl = *handle.impl_;
  if (execution.device_ordinal() != impl.pool->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "transport-output-download", "device",
                    impl.identity.rank, "handle device", "foreign");
  const auto elements = checked_mul(impl.identity.tokens,
                                    impl.identity.hidden,
                                    "output download elements");
  std::vector<transport::BFloat16> output(
      checked_size(elements, "output download elements"));
  if (!output.empty()) {
    const auto bytes = checked_mul(elements,
                                   sizeof(transport::BFloat16),
                                   "output download bytes");
    internal::throw_cuda(
        "transport-output-download",
        cudaMemcpy(output.data(),
                   at<const void>(
                       impl.pool->slots[impl.slot->index].device,
                       impl.pool->layout.combined),
                   checked_size(bytes, "output download bytes"),
                   cudaMemcpyDeviceToHost),
        impl.identity.rank, impl.identity.world, "transport-validation",
        impl.identity.generation, ExchangeKind::ExpertCounts,
        "cudaMemcpy(final output)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost, bytes);
  }
  return output;
}

testing::GinPutSignalProbeResult
testing::run_distributed_gin_put_signal_probe(
    const NcclSession &session, const CudaExecution &execution,
    std::uint64_t payload_bytes) {
#if !__has_include(<nccl_device.h>)
  (void)session;
  (void)execution;
  (void)payload_bytes;
  throw NotifyDispatchError(
      NotifyErrorCode::BackendUnavailable,
      "phase=gin-put-probe, field=headers, rank=0, expected=nccl_device.h, "
      "actual=unavailable");
#else
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto exec = internal::checked_execution_state(execution);
  if (nccl->device != exec->device)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "gin-put-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(exec->device));
  constexpr std::size_t kAllocationBytes = 4096;
  constexpr std::size_t kReceiveOffset = 2048;
  constexpr std::size_t kBarrierOffset = 3072;
  constexpr std::size_t kErrorOffset = 4088;
  if (payload_bytes == 0 || payload_bytes > kReceiveOffset)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "gin-put-probe",
                    "payload_bytes", nccl->rank, "1..2048",
                    std::to_string(payload_bytes));

  using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                              ncclCommProperties_t *);
  using DevCommCreateFn = ncclResult_t (*)(
      ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
  using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                             const ncclDevComm_t *);
  using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                             ncclWindow_t *, int);
  using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
  using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
  using MemFreeFn = ncclResult_t (*)(void *);
  using TeamFn = ncclTeam_t (*)(ncclComm_t);
  const auto load = [](const char *name) {
    auto *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
      throw NotifyDispatchError(
          NotifyErrorCode::BackendUnavailable,
          std::string("phase=gin-put-probe, field=symbol, rank=0, expected=") +
              name + ", actual=missing from loaded runtime");
    return symbol;
  };
  const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
      load("ncclCommQueryProperties"));
  const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
      load("ncclDevCommCreate"));
  const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
      load("ncclDevCommDestroy"));
  const auto window_register = reinterpret_cast<WindowRegisterFn>(
      load("ncclCommWindowRegister"));
  const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
      load("ncclCommWindowDeregister"));
  const auto mem_alloc =
      reinterpret_cast<MemAllocFn>(load("ncclMemAlloc"));
  const auto mem_free = reinterpret_cast<MemFreeFn>(load("ncclMemFree"));
  const auto team_rail = reinterpret_cast<TeamFn>(load("ncclTeamRail"));
  const auto team_lsa = reinterpret_cast<TeamFn>(load("ncclTeamLsa"));

  ncclCommProperties_t properties{};
  properties.size = sizeof(properties);
  properties.magic = NCCL_API_MAGIC;
  properties.version = NCCL_VERSION_CODE;
  auto result = query_properties(nccl->comm, &properties);
  if (result != ncclSuccess || properties.railedGinType == NCCL_GIN_TYPE_NONE)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=gin-put-probe, field=railed_gin, rank=" +
            std::to_string(nccl->rank) +
            ", expected=available, actual=" +
            (result == ncclSuccess
                 ? std::string("NCCL_GIN_TYPE_NONE")
                 : std::string(ncclGetErrorString(result))));
  const auto rail = team_rail(nccl->comm);
  const auto lsa = team_lsa(nccl->comm);
  if (rail.nRanks < 2 || lsa.nRanks <= 0)
    transport_error(NotifyErrorCode::BackendUnavailable, "gin-put-probe",
                    "rail topology", nccl->rank, "rail.nRanks>=2, lsa>0",
                    "rail=" + std::to_string(rail.nRanks) + ", lsa=" +
                        std::to_string(lsa.nRanks));

  ncclDevCommRequirements_t requirements =
      NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  requirements.ginContextCount = 1;
  requirements.ginExclusiveContexts = true;
  requirements.ginQueueDepth = 32;
  requirements.ginSignalCount = 1;
  requirements.ginCounterCount = 1;
  requirements.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
  ncclDevComm_t dev_comm{};
  ncclWindow_t window = nullptr;
  void *allocation = nullptr;
  bool dev_comm_created = false;
  const auto cleanup = [&]() noexcept {
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
  };
  try {
    internal::throw_nccl(
        "gin-put-probe",
        dev_comm_create(nccl->comm, &requirements, &dev_comm), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclDevCommCreate(rail GIN)");
    dev_comm_created = true;
    internal::throw_nccl(
        "gin-put-probe", mem_alloc(&allocation, kAllocationBytes), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclMemAlloc(GIN probe window)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemsetAsync(allocation, 0, kAllocationBytes, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemsetAsync(GIN probe window)");
    std::vector<std::uint8_t> send(
        checked_size(payload_bytes, "GIN probe payload"));
    for (std::size_t index = 0; index < send.size(); ++index)
      send[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(nccl->rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpyAsync(allocation, send.data(), send.size(),
                        cudaMemcpyHostToDevice, exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(GIN probe source)");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(before GIN window register)");
    internal::throw_nccl(
        "gin-put-probe",
        window_register(nccl->comm, allocation, kAllocationBytes, &window,
                        NCCL_WIN_STRICT_ORDERING),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclCommWindowRegister(GIN probe)");
    gin_probe_reset_kernel<<<1, 1, 0, exec->stream>>>(dev_comm);
    auto *barrier = static_cast<std::uint64_t *>(allocation) +
                    kBarrierOffset / sizeof(std::uint64_t);
    internal::throw_nccl(
        "gin-put-probe",
        ncclAllReduce(barrier, barrier, 1, ncclUint64, ncclSum, nccl->comm,
                      exec->stream),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "ncclAllReduce(after GIN reset)");
    gin_probe_put_signal_kernel<<<1, 1, 0, exec->stream>>>(
        dev_comm, window, checked_size(payload_bytes, "GIN probe payload"),
        kReceiveOffset,
        reinterpret_cast<unsigned int *>(
            static_cast<std::byte *>(allocation) + kErrorOffset));
    internal::throw_cuda(
        "gin-put-probe", cudaGetLastError(), nccl->rank, nccl->world,
        session.session().identity(), session.session().generation(),
        ExchangeKind::ExpertCounts, "GIN put/signal kernel launch");
    internal::throw_cuda(
        "gin-put-probe", cudaStreamSynchronize(exec->stream), nccl->rank,
        nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaStreamSynchronize(GIN put/signal)");
    unsigned int device_error = 0;
    std::vector<std::uint8_t> received(send.size());
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(received.data(),
                   static_cast<std::byte *>(allocation) + kReceiveOffset,
                   received.size(), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN received payload)");
    internal::throw_cuda(
        "gin-put-probe",
        cudaMemcpy(&device_error,
                   static_cast<std::byte *>(allocation) + kErrorOffset,
                   sizeof(device_error), cudaMemcpyDeviceToHost),
        nccl->rank, nccl->world, session.session().identity(),
        session.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMemcpy(GIN device error)");
    if (device_error != 0)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "device_error", nccl->rank, "0",
                      std::to_string(device_error));
    const auto source_rail = (rail.rank + rail.nRanks - 1) % rail.nRanks;
    const auto source_rank = source_rail * lsa.nRanks + lsa.rank;
    std::vector<std::uint8_t> expected(received.size());
    for (std::size_t index = 0; index < expected.size(); ++index)
      expected[index] = static_cast<std::uint8_t>(
          (static_cast<std::uint64_t>(source_rank) * 17ULL +
           static_cast<std::uint64_t>(index) * 13ULL) %
          251ULL);
    const auto hash = [](const std::vector<std::uint8_t> &bytes) {
      std::uint64_t value = 1469598103934665603ULL;
      for (const auto byte : bytes)
        value = (value ^ byte) * 1099511628211ULL;
      return value;
    };
    testing::GinPutSignalProbeResult probe;
    probe.payload_bytes = payload_bytes;
    probe.rail_rank = static_cast<std::uint32_t>(rail.rank);
    probe.rail_world = static_cast<std::uint32_t>(rail.nRanks);
    probe.local_rank = static_cast<std::uint32_t>(lsa.rank);
    probe.received_hash = hash(received);
    probe.expected_hash = hash(expected);
    if (received != expected)
      transport_error(NotifyErrorCode::NcclFailure, "gin-put-probe",
                      "received payload hash", nccl->rank,
                      std::to_string(probe.expected_hash),
                      std::to_string(probe.received_hash));
    cleanup();
    return probe;
  } catch (...) {
    cleanup();
    throw;
  }
#endif
}

ActivationTransportCapability probe_activation_transport_capability(
    std::uint32_t world_size) {
  ActivationTransportCapability result;
  result.compiled_nccl_version = NCCL_VERSION_CODE;
  int runtime = 0;
  if (ncclGetVersion(&runtime) == ncclSuccess)
    result.runtime_nccl_version = static_cast<std::uint32_t>(runtime);
#if __has_include(<nccl_device.h>)
  result.nccl_device_headers = true;
#endif
#if __has_include(<nccl_device/gin.h>)
  result.gin_headers_available = true;
#endif
  // Device put/signal/wait/flush are header-inline APIs. Their required host
  // lifecycle surface must all exist in the actually loaded runtime.
  constexpr const char *required_symbols[] = {
      "ncclCommQueryProperties",
      "ncclDevCommCreate",
      "ncclDevCommDestroy",
      "ncclMemAlloc",
      "ncclMemFree",
      "ncclCommWindowRegister",
      "ncclCommWindowDeregister",
      "ncclGetPeerDevicePointer",
      "ncclTeamRail",
      "ncclGinOutboxCreateRequirement",
      "ncclGinInboxA2ACreateRequirement",
      "ncclGinBarrierCreateRequirement",
  };
  result.gin_symbols_present = result.gin_headers_available;
  std::string first_missing;
  for (const auto *symbol : required_symbols) {
    if (dlsym(RTLD_DEFAULT, symbol) == nullptr) {
      result.gin_symbols_present = false;
      if (first_missing.empty())
        first_missing = symbol;
    }
  }
  result.gin_backend_implemented = true;

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));
    ncclComm_t probe_comm = nullptr;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string init_error;
    ncclUniqueId id{};
    auto nccl_result = ncclGetUniqueId(&id);
    if (nccl_result == ncclSuccess)
      nccl_result = ncclCommInitRank(&probe_comm, 1, id, 0);
    if (nccl_result != ncclSuccess) {
      init_error = std::string("ncclCommInitRank: ") +
                   ncclGetErrorString(nccl_result);
    } else {
      ncclCommProperties properties{};
      properties.size = sizeof(properties);
      properties.magic = NCCL_API_MAGIC;
      properties.version = NCCL_VERSION_CODE;
      nccl_result = query_properties(probe_comm, &properties);
      if (nccl_result != ncclSuccess) {
        init_error = std::string("ncclCommQueryProperties: ") +
                     ncclGetErrorString(nccl_result);
      } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
                 properties.railedGinType == NCCL_GIN_TYPE_NONE) {
        init_error = "ncclCommQueryProperties: no GIN context type";
      } else {
        ncclDevCommRequirements_t requirements{};
        requirements.size = sizeof(requirements);
        requirements.magic = NCCL_API_MAGIC;
        requirements.version = NCCL_VERSION_CODE;
        requirements.ginContextCount = 1;
        requirements.ginExclusiveContexts = true;
        requirements.ginQueueDepth = 32;
        requirements.ginSignalCount = 4;
        requirements.ginConnectionType =
            properties.railedGinType != NCCL_GIN_TYPE_NONE
                ? NCCL_GIN_CONNECTION_RAIL
                : NCCL_GIN_CONNECTION_FULL;
        nccl_result = dev_comm_create(probe_comm, &requirements, &dev_comm);
        if (nccl_result != ncclSuccess) {
          init_error = std::string("ncclDevCommCreate(GIN): ") +
                       ncclGetErrorString(nccl_result);
        } else {
          dev_comm_created = true;
          nccl_result = mem_alloc(&allocation, 4096);
          if (nccl_result != ncclSuccess) {
            init_error = std::string("ncclMemAlloc(window): ") +
                         ncclGetErrorString(nccl_result);
          } else {
            nccl_result = window_register(
                probe_comm, allocation, 4096, &window,
                NCCL_WIN_STRICT_ORDERING);
            if (nccl_result != ncclSuccess)
              init_error = std::string("ncclCommWindowRegister: ") +
                           ncclGetErrorString(nccl_result);
            else
              result.gin_runtime_initializable = true;
          }
        }
      }
    }
    if (window)
      (void)window_deregister(probe_comm, window);
    if (allocation)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(probe_comm, &dev_comm);
    if (probe_comm)
      (void)ncclCommDestroy(probe_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(init_error);
  }
#endif
  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;
  if (!result.gin_usable) {
    std::ostringstream message;
    message << "GIN unavailable: headers=" << result.gin_headers_available
            << ", symbols=" << result.gin_symbols_present
            << ", runtime_initializable="
            << result.gin_runtime_initializable
            << ", backend_implemented=" << result.gin_backend_implemented
            << ", compiled_nccl=" << result.compiled_nccl_version
            << ", runtime_nccl=" << result.runtime_nccl_version;
    if (!first_missing.empty())
      message << ", first_missing_symbol=" << first_missing;
    if (!result.gin_unavailable_reason.empty())
      message << ", initialization_error=" << result.gin_unavailable_reason;
    result.gin_unavailable_reason = message.str();
  }
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess)
    devices = 0;
  const auto checked_world = std::min<std::uint32_t>(
      world_size, static_cast<std::uint32_t>(std::max(devices, 0)));
  result.peer_access_matrix.resize(
      static_cast<std::size_t>(checked_world) * checked_world, 0);
  for (std::uint32_t source = 0; source < checked_world; ++source) {
    for (std::uint32_t destination = 0; destination < checked_world;
         ++destination) {
      int access = source == destination ? 1 : 0;
      if (source != destination)
        (void)cudaDeviceCanAccessPeer(&access, static_cast<int>(source),
                                      static_cast<int>(destination));
      result.peer_access_matrix[static_cast<std::size_t>(source) *
                                    checked_world +
                                destination] =
          static_cast<std::uint8_t>(access != 0);
    }
  }
  if (devices > 0) {
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess)
      result.system_scope_atomics = properties.major >= 6;
  }
  return result;
}

ActivationTransportCapability probe_activation_transport_capability(
    const NcclSession &session, const CudaExecution &execution) {
  const auto nccl = internal::NcclSessionAccess::checked(session);
  const auto cuda = internal::checked_execution_state(execution);
  if (nccl->device != cuda->device)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "gin-distributed-probe",
                    "device", nccl->rank, std::to_string(nccl->device),
                    std::to_string(cuda->device));

  auto result = probe_activation_transport_capability(nccl->world);
  // The world=1 probe above is useful for loader diagnostics but cannot prove
  // that a rail GIN team can be formed.  Replace only its initialization
  // result with the collective probe below.
  result.gin_runtime_initializable = false;
  result.gin_backend_implemented = true;
  result.gin_usable = false;
  result.gin_unavailable_reason.clear();

#if __has_include(<nccl_device/core.h>)
  if (result.gin_headers_available && result.gin_symbols_present) {
    using QueryPropertiesFn = ncclResult_t (*)(ncclComm_t,
                                                ncclCommProperties_t *);
    using DevCommCreateFn = ncclResult_t (*)(
        ncclComm_t, const ncclDevCommRequirements_t *, ncclDevComm_t *);
    using DevCommDestroyFn = ncclResult_t (*)(ncclComm_t,
                                               const ncclDevComm_t *);
    using WindowRegisterFn = ncclResult_t (*)(ncclComm_t, void *, std::size_t,
                                               ncclWindow_t *, int);
    using WindowDeregisterFn = ncclResult_t (*)(ncclComm_t, ncclWindow_t);
    using MemAllocFn = ncclResult_t (*)(void **, std::size_t);
    using MemFreeFn = ncclResult_t (*)(void *);
    const auto query_properties = reinterpret_cast<QueryPropertiesFn>(
        dlsym(RTLD_DEFAULT, "ncclCommQueryProperties"));
    const auto dev_comm_create = reinterpret_cast<DevCommCreateFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommCreate"));
    const auto dev_comm_destroy = reinterpret_cast<DevCommDestroyFn>(
        dlsym(RTLD_DEFAULT, "ncclDevCommDestroy"));
    const auto window_register = reinterpret_cast<WindowRegisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowRegister"));
    const auto window_deregister = reinterpret_cast<WindowDeregisterFn>(
        dlsym(RTLD_DEFAULT, "ncclCommWindowDeregister"));
    const auto mem_alloc = reinterpret_cast<MemAllocFn>(
        dlsym(RTLD_DEFAULT, "ncclMemAlloc"));
    const auto mem_free = reinterpret_cast<MemFreeFn>(
        dlsym(RTLD_DEFAULT, "ncclMemFree"));

    ncclCommProperties_t properties{};
    properties.size = sizeof(properties);
    properties.magic = NCCL_API_MAGIC;
    properties.version = NCCL_VERSION_CODE;
    ncclDevCommRequirements_t requirements{};
    requirements.size = sizeof(requirements);
    requirements.magic = NCCL_API_MAGIC;
    requirements.version = NCCL_VERSION_CODE;
    ncclDevComm_t dev_comm{};
    ncclWindow_t window = nullptr;
    void *allocation = nullptr;
    bool dev_comm_created = false;
    std::string error;

    auto nccl_result = query_properties(nccl->comm, &properties);
    if (nccl_result != ncclSuccess) {
      error = std::string("ncclCommQueryProperties: ") +
              ncclGetErrorString(nccl_result);
    } else if (properties.ginType == NCCL_GIN_TYPE_NONE &&
               properties.railedGinType == NCCL_GIN_TYPE_NONE) {
      error = "ncclCommQueryProperties: no GIN context type";
    } else {
      requirements.ginContextCount = 1;
      requirements.ginExclusiveContexts = true;
      requirements.ginQueueDepth = 32;
      requirements.ginSignalCount = 4;
      requirements.ginConnectionType =
          properties.railedGinType != NCCL_GIN_TYPE_NONE
              ? NCCL_GIN_CONNECTION_RAIL
              : NCCL_GIN_CONNECTION_FULL;
      nccl_result = dev_comm_create(nccl->comm, &requirements, &dev_comm);
      if (nccl_result != ncclSuccess) {
        error = std::string("ncclDevCommCreate(GIN): ") +
                ncclGetErrorString(nccl_result);
      } else {
        dev_comm_created = true;
        nccl_result = mem_alloc(&allocation, 4096);
        if (nccl_result != ncclSuccess) {
          error = std::string("ncclMemAlloc(window): ") +
                  ncclGetErrorString(nccl_result);
        } else {
          nccl_result = window_register(nccl->comm, allocation, 4096, &window,
                                        NCCL_WIN_STRICT_ORDERING);
          if (nccl_result != ncclSuccess)
            error = std::string("ncclCommWindowRegister: ") +
                    ncclGetErrorString(nccl_result);
          else
            result.gin_runtime_initializable = true;
        }
      }
    }
    if (window != nullptr)
      (void)window_deregister(nccl->comm, window);
    if (allocation != nullptr)
      (void)mem_free(allocation);
    if (dev_comm_created)
      (void)dev_comm_destroy(nccl->comm, &dev_comm);
    if (!result.gin_runtime_initializable)
      result.gin_unavailable_reason = std::move(error);
  }
#endif

  result.gin_usable = result.gin_headers_available &&
                      result.gin_symbols_present &&
                      result.gin_runtime_initializable &&
                      result.gin_backend_implemented;

  std::ostringstream message;
  message << "GIN distributed probe: headers="
          << result.gin_headers_available << ", symbols="
          << result.gin_symbols_present << ", runtime_initializable="
          << result.gin_runtime_initializable << ", backend_implemented="
          << result.gin_backend_implemented << ", rank=" << nccl->rank
          << ", world=" << nccl->world << ", compiled_nccl="
          << result.compiled_nccl_version << ", runtime_nccl="
          << result.runtime_nccl_version;
  if (!result.gin_unavailable_reason.empty())
    message << ", initialization_error=" << result.gin_unavailable_reason;
  result.gin_unavailable_reason = message.str();
  return result;
}

bool testing::run_local_p2p_put_signal_probe(std::uint32_t source_device,
                                             std::uint32_t destination_device,
                                             std::uint64_t elements) {
  if (elements == 0)
    return false;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return false;
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
  bool ok = false;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(checked_mul(
                                   elements, sizeof(std::uint64_t),
                                   "peer destination bytes"),
                               "peer destination bytes")) != cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemset(signal, 0, sizeof(*signal));
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(checked_mul(
                              elements, sizeof(std::uint64_t),
                              "peer source bytes"),
                          "peer source bytes")) != cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  if (cudaMemsetAsync(source, 0x5a,
                      checked_size(checked_mul(elements, sizeof(std::uint64_t),
                                               "peer copy bytes"),
                                   "peer copy bytes"),
                      source_stream) != cudaSuccess ||
      cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
  const auto blocks = static_cast<unsigned int>(
      std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
  const auto run = [&](bool early_signal, unsigned int &observed_errors) {
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaMemsetAsync(destination, 0,
                        checked_size(checked_mul(elements,
                                                 sizeof(std::uint64_t),
                                                 "peer reset bytes"),
                                     "peer reset bytes"),
                        destination_stream) != cudaSuccess ||
        cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
            cudaSuccess ||
        cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
            cudaSuccess ||
        cudaStreamSynchronize(destination_stream) != cudaSuccess)
      return false;
    peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
        destination, signal, errors, elements, expected);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                        source_stream) != cudaSuccess ||
        cudaStreamSynchronize(source_stream) != cudaSuccess)
      return false;
    peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
        source, destination, signal, completed_blocks, elements, early_signal);
    if (cudaGetLastError() != cudaSuccess)
      return false;
    // The destination independently observes the signal and validates every
    // word before the source stream is synchronized.
    (void)cudaSetDevice(static_cast<int>(destination_device));
    if (cudaStreamSynchronize(destination_stream) != cudaSuccess ||
        cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
    (void)cudaSetDevice(static_cast<int>(source_device));
    return cudaStreamSynchronize(source_stream) == cudaSuccess;
  };
  {
    unsigned int positive_errors = 0;
    unsigned int early_signal_errors = 0;
    ok = run(false, positive_errors) && positive_errors == 0U &&
         run(true, early_signal_errors) && early_signal_errors != 0U;
  }
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return ok;
}

double testing::measure_local_p2p_put_bandwidth_gbps(
    std::uint32_t source_device, std::uint32_t destination_device,
    std::uint64_t bytes, std::uint32_t warmup, std::uint32_t measured) {
  if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0 || measured == 0)
    return 0.0;
  int access = 0;
  if (cudaDeviceCanAccessPeer(&access, static_cast<int>(source_device),
                              static_cast<int>(destination_device)) !=
          cudaSuccess ||
      access == 0)
    return 0.0;
  const auto elements = bytes / sizeof(std::uint64_t);
  std::uint64_t *source = nullptr;
  std::uint64_t *destination = nullptr;
  unsigned int *signal = nullptr;
  unsigned int *errors = nullptr;
  unsigned int *completed_blocks = nullptr;
  cudaStream_t source_stream = nullptr;
  cudaStream_t destination_stream = nullptr;
  double result = 0.0;
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (cudaMalloc(&destination, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&signal, sizeof(*signal)) != cudaSuccess ||
      cudaMalloc(&errors, sizeof(*errors)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&destination_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaSetDevice(static_cast<int>(source_device));
  {
    const auto enable = cudaDeviceEnablePeerAccess(
        static_cast<int>(destination_device), 0);
    if (enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
      goto cleanup;
    (void)cudaGetLastError();
  }
  if (cudaMalloc(&source, checked_size(bytes, "P2P bandwidth bytes")) !=
          cudaSuccess ||
      cudaMalloc(&completed_blocks, sizeof(*completed_blocks)) != cudaSuccess ||
      cudaStreamCreateWithFlags(&source_stream, cudaStreamNonBlocking) !=
          cudaSuccess)
    goto cleanup;
  (void)cudaMemsetAsync(source, 0x5a, checked_size(bytes, "P2P bandwidth bytes"),
                        source_stream);
  if (cudaStreamSynchronize(source_stream) != cudaSuccess)
    goto cleanup;
  {
    constexpr std::uint64_t expected = 0x5a5a5a5a5a5a5a5aULL;
    const auto blocks = static_cast<unsigned int>(
        std::min<std::uint64_t>((elements + 255U) / 256U, 1024U));
    double measured_seconds = 0.0;
    for (std::uint32_t iteration = 0; iteration < warmup + measured;
         ++iteration) {
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaMemsetAsync(signal, 0, sizeof(*signal), destination_stream) !=
              cudaSuccess ||
          cudaMemsetAsync(errors, 0, sizeof(*errors), destination_stream) !=
              cudaSuccess ||
          cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      peer_signal_verify_kernel<<<blocks, 256, 0, destination_stream>>>(
          destination, signal, errors, elements, expected);
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaMemsetAsync(completed_blocks, 0, sizeof(*completed_blocks),
                          source_stream) != cudaSuccess ||
          cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      const auto start = std::chrono::steady_clock::now();
      peer_put_signal_kernel<<<blocks, 256, 0, source_stream>>>(
          source, destination, signal, completed_blocks, elements, false);
      (void)cudaSetDevice(static_cast<int>(destination_device));
      if (cudaStreamSynchronize(destination_stream) != cudaSuccess)
        goto cleanup;
      const auto finish = std::chrono::steady_clock::now();
      unsigned int observed_errors = 0;
      if (cudaMemcpy(&observed_errors, errors, sizeof(observed_errors),
                     cudaMemcpyDeviceToHost) != cudaSuccess ||
          observed_errors != 0U)
        goto cleanup;
      (void)cudaSetDevice(static_cast<int>(source_device));
      if (cudaStreamSynchronize(source_stream) != cudaSuccess)
        goto cleanup;
      if (iteration >= warmup)
        measured_seconds +=
            std::chrono::duration<double>(finish - start).count();
    }
    if (measured_seconds > 0.0)
      result = static_cast<double>(bytes) * measured /
               measured_seconds / 1.0e9;
  }
cleanup:
  (void)cudaSetDevice(static_cast<int>(source_device));
  if (source_stream)
    (void)cudaStreamDestroy(source_stream);
  if (completed_blocks)
    (void)cudaFree(completed_blocks);
  if (source)
    (void)cudaFree(source);
  (void)cudaSetDevice(static_cast<int>(destination_device));
  if (destination_stream)
    (void)cudaStreamDestroy(destination_stream);
  if (errors)
    (void)cudaFree(errors);
  if (signal)
    (void)cudaFree(signal);
  if (destination)
    (void)cudaFree(destination);
  return result;
}

namespace {
bool run_world1_forward_backward_sanitizer_harness(
    bool run_backward, std::uint64_t hidden = 7168,
    std::uint64_t intermediate = 3072) {
  if (hidden == 0 || intermediate == 0)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig,
                    "runtime-shape-harness", "hidden/intermediate", 0,
                    "nonzero", "zero");
  struct HarnessMetadata {
    std::uint32_t expert_id = 0;
    std::uint32_t route_dst_rank = 0;
    std::uint32_t combine_order = 0;
    std::uint32_t owned_expert_id = 0;
    std::uint64_t route_dst_row = 0;
    std::uint64_t expert_row_count = 1;
    std::uint64_t expert_row_prefix[2] = {0, 1};
    std::uint64_t source_expert_count = 1;
  } host_metadata;
  void *arena = nullptr;
  void *metadata = nullptr;
  transport::BFloat16 *input = nullptr;
  float *gates = nullptr;
  transport::BFloat16 *w1 = nullptr;
  transport::BFloat16 *w2 = nullptr;
  transport::BFloat16 *w1_transpose = nullptr;
  transport::BFloat16 *w2_transpose = nullptr;
  std::uint64_t *cancel_epoch = nullptr;
  cudaStream_t stream = nullptr;
  bool passed = false;
  const ActivationTransportPoolConfig config{
      1, hidden, 1, 1, 1, 1, 1, 1, 1, intermediate, 1, true, run_backward};
  const auto layout = make_layout(config);
  const auto w1_elements = 2ULL * intermediate * hidden;
  const auto w2_elements = hidden * intermediate;
  auto cleanup = [&]() noexcept {
    if (stream)
      (void)cudaStreamSynchronize(stream);
    if (cancel_epoch)
      (void)cudaFree(cancel_epoch);
    if (w2)
      (void)cudaFree(w2);
    if (w2_transpose)
      (void)cudaFree(w2_transpose);
    if (w1)
      (void)cudaFree(w1);
    if (w1_transpose)
      (void)cudaFree(w1_transpose);
    if (gates)
      (void)cudaFree(gates);
    if (input)
      (void)cudaFree(input);
    if (metadata)
      (void)cudaFree(metadata);
    if (arena)
      (void)cudaFree(arena);
    if (stream)
      (void)cudaStreamDestroy(stream);
  };
  try {
    internal::throw_cuda("forward-sanitizer", cudaSetDevice(0), 0, 1,
                         "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
                         "cudaSetDevice");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), 0, 1,
        "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaStreamCreate");
    internal::throw_cuda("forward-sanitizer", cudaMalloc(&arena, layout.total),
                         0, 1, "world1-no-nccl", 1,
                         ExchangeKind::ExpertCounts, "cudaMalloc(arena)");
    internal::throw_cuda(
        "forward-sanitizer", cudaMalloc(&metadata, sizeof(HarnessMetadata)), 0,
        1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(metadata)");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMalloc(reinterpret_cast<void **>(&input), hidden * sizeof(*input)),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(input)");
    internal::throw_cuda(
        "forward-sanitizer", cudaMalloc(reinterpret_cast<void **>(&gates),
                                         sizeof(*gates)),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(gate)");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMalloc(reinterpret_cast<void **>(&w1),
                   checked_size(w1_elements * sizeof(*w1), "harness W1")),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(W1)");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMalloc(reinterpret_cast<void **>(&w2),
                   checked_size(w2_elements * sizeof(*w2), "harness W2")),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(W2)");
    if (run_backward) {
      internal::throw_cuda(
          "backward-sanitizer",
          cudaMalloc(reinterpret_cast<void **>(&w1_transpose),
                     checked_size(w1_elements * sizeof(*w1_transpose),
                                  "harness W1 transpose")),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(W1 transpose)");
      internal::throw_cuda(
          "backward-sanitizer",
          cudaMalloc(reinterpret_cast<void **>(&w2_transpose),
                     checked_size(w2_elements * sizeof(*w2_transpose),
                                  "harness W2 transpose")),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "cudaMalloc(W2 transpose)");
    }
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMalloc(reinterpret_cast<void **>(&cancel_epoch),
                   sizeof(*cancel_epoch)),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "cudaMalloc(cancel)");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemsetAsync(arena, 0, layout.total, stream), 0, 1,
        "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize arena");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemsetAsync(input, 0x3f, hidden * sizeof(*input), stream), 0, 1,
        "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize input");
    const float host_gate = 1.0F;
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemcpyAsync(gates, &host_gate, sizeof(host_gate),
                        cudaMemcpyHostToDevice, stream),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize gate");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemcpyAsync(metadata, &host_metadata, sizeof(host_metadata),
                        cudaMemcpyHostToDevice, stream),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize dispatch metadata");
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemsetAsync(cancel_epoch, 0, sizeof(*cancel_epoch), stream), 0, 1,
        "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize cancel epoch");
    initialize_forward_weights_kernel<<<4096, 256, 0, stream>>>(
        w1, w1_elements, w2, w2_elements, 0, hidden, intermediate);
    if (run_backward)
      transpose_forward_weights_for_backward_kernel<<<4096, 256, 0, stream>>>(
          w1, w1_transpose, w2, w2_transpose, 1, hidden, intermediate);
    auto *metadata_bytes = static_cast<unsigned char *>(metadata);
    DeviceDispatchKernelView view{};
    view.identity = {kDeviceDispatchLayoutVersion, 1, 0, 1, 1, 1, 1, 1, 0,
                     0};
    view.sizes = {1, 1, 1, 1, 1, 0, 0, 0, 0};
    view.expert_id = reinterpret_cast<std::uint32_t *>(
        metadata_bytes + offsetof(HarnessMetadata, expert_id));
    view.route_dst_rank = reinterpret_cast<std::uint32_t *>(
        metadata_bytes + offsetof(HarnessMetadata, route_dst_rank));
    view.combine_order = reinterpret_cast<std::uint32_t *>(
        metadata_bytes + offsetof(HarnessMetadata, combine_order));
    view.owned_expert_ids = reinterpret_cast<std::uint32_t *>(
        metadata_bytes + offsetof(HarnessMetadata, owned_expert_id));
    view.route_dst_row = reinterpret_cast<std::uint64_t *>(
        metadata_bytes + offsetof(HarnessMetadata, route_dst_row));
    view.expert_row_count = reinterpret_cast<std::uint64_t *>(
        metadata_bytes + offsetof(HarnessMetadata, expert_row_count));
    view.expert_row_prefix = reinterpret_cast<std::uint64_t *>(
        metadata_bytes + offsetof(HarnessMetadata, expert_row_prefix));
    view.source_expert_count = reinterpret_cast<std::uint64_t *>(
        metadata_bytes + offsetof(HarnessMetadata, source_expert_count));
    void *host_peer = arena;
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemcpyAsync(at<void *>(arena, layout.peer_bases), &host_peer,
                        sizeof(host_peer), cudaMemcpyHostToDevice, stream),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "initialize peer base");
    initialize_forward_generation_signals_kernel<<<1, 256, 0, stream>>>(
        arena, at<void *>(arena, layout.peer_bases),
        layout.ingress_published_sequence, layout.outgoing_route_sequence,
        layout.p2p_barrier_receive, layout.gin_local_compute_ready,
        layout.forward_x_arrived_rows, layout.backward_dy_arrived_rows,
        view.sizes.mtile_count, layout.forward_completion,
        layout.backward_completion, layout.forward_context_completion,
        layout.backward_context_completion, 0, 1, 0, 1, 1, 1, 0, 1);
    ForwardKernelArgs args{};
    args.dispatch = view;
    args.input = input;
    args.gates = gates;
    args.w1 = w1;
    args.w2 = w2;
    args.hidden = hidden;
    args.intermediate = intermediate;
    args.max_destination_rows = 1;
    args.rank = 0;
    args.world = 1;
    args.channels = 1;
    args.epoch = 1;
    args.local_base = arena;
    args.peer_bases = at<void *>(arena, layout.peer_bases);
    args.canonical_rows_offset = layout.canonical_rows;
    args.forward_output_rows_offset = layout.canonical_rows;
    args.direct_headers_offset = layout.direct_route_headers;
    args.contribution_offset = layout.contribution;
    args.return_ready_epoch_offset = layout.return_ready_epoch;
    args.combined_offset = layout.combined;
    args.ingress_published_offset = layout.ingress_published_sequence;
    args.outgoing_sequence_offset = layout.outgoing_route_sequence;
    args.status_offset = layout.status;
    args.forward_s_offset = layout.forward_s;
    args.forward_ab_cache_offset = layout.forward_ab_cache;
    args.forward_task_claim_offset = layout.forward_task_claim;
    args.forward_fc1_done_offset = layout.forward_fc1_done;
    args.forward_scheduler_offset = layout.forward_scheduler;
    args.gin_local_compute_ready_offset = layout.gin_local_compute_ready;
    args.terminal_epoch_offset = layout.p2p_barrier_receive;
    args.production_error_records_offset = layout.production_error_records;
    args.cancel_epoch =
        reinterpret_cast<const unsigned long long *>(cancel_epoch);
    args.cache_fc1 = true;
    const bool main_shape = hidden == 7168 && intermediate == 3072;
    CUtensorMap x_map{};
    CUtensorMap s_map{};
    CUtensorMap w1_map{};
    CUtensorMap w2_map{};
    if (main_shape) {
      x_map = make_bf16_tma_map(
          at<void>(arena, layout.canonical_rows), hidden, 1, 128,
          "sanitizer canonical X");
      s_map = make_bf16_tma_map(
          at<void>(arena, layout.forward_s), intermediate, 1, 128,
          "sanitizer S");
      w1_map = make_bf16_tma_map(w1, hidden, 2 * intermediate, 128,
                                 "sanitizer W1");
      w2_map = make_bf16_tma_map(w2, intermediate, hidden, 256,
                                 "sanitizer W2");
    }
    const std::size_t smem =
        (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(*input) +
        8U * sizeof(std::uint64_t) + sizeof(std::uint32_t);
    internal::throw_cuda(
        "forward-sanitizer",
        cudaFuncSetAttribute(local_p2p_forward_megakernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem)),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "set dynamic shared memory");
    cudaLaunchAttribute attributes[2]{};
    attributes[0].id = cudaLaunchAttributeCooperative;
    attributes[0].val.cooperative = 1;
    attributes[1].id = cudaLaunchAttributeClusterDimension;
    attributes[1].val.clusterDim = {2U, 1U, 1U};
    cudaLaunchConfig_t launch{};
    launch.gridDim = dim3(2, 1, 1);
    launch.blockDim = dim3(384, 1, 1);
    launch.dynamicSmemBytes = smem;
    launch.stream = stream;
    launch.attrs = attributes;
    launch.numAttrs = 2;
    if (!main_shape) {
      launch.gridDim = dim3(1, 1, 1);
      launch.blockDim = dim3(256, 1, 1);
      launch.dynamicSmemBytes = 0;
      launch.numAttrs = 1;
    }
    internal::throw_cuda(
        "forward-sanitizer",
        main_shape
            ? cudaLaunchKernelEx(&launch, local_p2p_forward_megakernel<true>, args,
                                 x_map, s_map, w1_map, w2_map)
            : cudaLaunchKernelEx(&launch,
                                 runtime_shape_forward_megakernel, args),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "production Forward persistent kernel");
    TransportDeviceStatus status{};
    internal::throw_cuda(
        "forward-sanitizer",
        cudaMemcpyAsync(&status, at<void>(arena, layout.status), sizeof(status),
                        cudaMemcpyDeviceToHost, stream),
        0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "download status");
    internal::throw_cuda(
        "forward-sanitizer", cudaStreamSynchronize(stream), 0, 1,
        "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
        "wait production Forward kernel");
    passed = status.error == 0U &&
             (!main_shape || (status.wgmma_instructions != 0ULL &&
                              status.tma_loads != 0ULL &&
                              status.runtime_cluster_size == 2U));
    if (run_backward && passed) {
      auto *dy_input = at<transport::BFloat16>(
          arena, layout.backward_dy_input);
      internal::throw_cuda(
          "backward-sanitizer",
          cudaMemsetAsync(dy_input, 0x3e, hidden * sizeof(*dy_input), stream),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "initialize dY input");
      initialize_forward_generation_signals_kernel<<<1, 256, 0, stream>>>(
          arena, at<void *>(arena, layout.peer_bases),
          layout.ingress_published_sequence, layout.outgoing_route_sequence,
          layout.p2p_barrier_receive, layout.gin_local_compute_ready,
          layout.forward_x_arrived_rows, layout.backward_dy_arrived_rows,
          view.sizes.mtile_count, layout.forward_completion,
          layout.backward_completion, layout.forward_context_completion,
          layout.backward_context_completion, 0, 2, 0, 1, 1, 2, 0, 1);
      BackwardKernelArgs backward{};
      backward.dispatch = view;
      backward.dy_input = dy_input;
      backward.gates = gates;
      backward.w1 = w1;
      backward.w2 = w2;
      backward.hidden = hidden;
      backward.intermediate = intermediate;
      backward.max_destination_rows = 1;
      backward.reduction_rows = 64;
      backward.rank = 0;
      backward.world = 1;
      backward.channels = 1;
      backward.epoch = 2;
      backward.local_base = arena;
      backward.peer_bases = at<void *>(arena, layout.peer_bases);
      backward.canonical_x_offset = layout.canonical_rows;
      backward.direct_headers_offset = layout.direct_route_headers;
      backward.dy_canonical_offset = layout.backward_dy_canonical;
      backward.canonical_gate_offset = layout.backward_canonical_gate;
      backward.dab_offset = layout.backward_dab;
      backward.recompute_ab_offset = layout.backward_recompute_ab;
      backward.backward_s_offset = layout.forward_s;
      backward.forward_ab_cache_offset = layout.forward_ab_cache;
      backward.dx_canonical_offset = layout.backward_dx_canonical;
      backward.dx_return_offset = layout.backward_dx_return;
      backward.dx_combined_offset = layout.backward_dx_combined;
      backward.dgate_partial_offset = layout.backward_dgate_partial;
      backward.dgate_canonical_offset = layout.backward_dgate_canonical;
      backward.dgate_offset = layout.backward_dgate;
      backward.u_transpose_offset = layout.backward_u_transpose;
      backward.dab_transpose_offset = layout.backward_dab_transpose;
      backward.dw1_offset = layout.backward_dw1;
      backward.dw2_offset = layout.backward_dw2;
      backward.scheduler_offset = layout.backward_scheduler;
      backward.claim_offset = layout.forward_task_claim;
      backward.incoming_signal_offset = layout.ingress_published_sequence;
      backward.outgoing_signal_offset = layout.outgoing_route_sequence;
      backward.gin_local_compute_ready_offset =
          layout.gin_local_compute_ready;
      backward.production_error_records_offset =
          layout.production_error_records;
      backward.status_offset = layout.backward_status;
      backward.cancel_epoch =
          reinterpret_cast<const unsigned long long *>(cancel_epoch);
      backward.use_forward_cache = true;
      CUtensorMap dy_map{};
      CUtensorMap dab_map{};
      CUtensorMap w2_transpose_map{};
      CUtensorMap w1_transpose_map{};
      CUtensorMap canonical_dy_mn_map{};
      CUtensorMap canonical_x_mn_map{};
      CUtensorMap u_mn_map{};
      CUtensorMap dab_mn_map{};
      CUtensorMap dw1_store_map{};
      if (main_shape) {
        dy_map = make_bf16_tma_map(
            at<void>(arena, layout.backward_dy_canonical), hidden, 1, 128,
            "sanitizer backward dY");
        dab_map = make_bf16_tma_map(
            at<void>(arena, layout.backward_dab), 2 * intermediate, 1, 128,
            "sanitizer backward dAB");
        w2_transpose_map = make_bf16_tma_map(
            w2_transpose, hidden, intermediate, 128,
            "sanitizer W2 transpose");
        w1_transpose_map = make_bf16_tma_map(
            w1_transpose, 2 * intermediate, hidden, 256,
            "sanitizer W1 transpose");
        canonical_dy_mn_map = make_bf16_tma_map_mn(
            at<void>(arena, layout.backward_dy_canonical), hidden, 1,
            "sanitizer canonical dY MN");
        canonical_x_mn_map = make_bf16_tma_map_mn(
            at<void>(arena, layout.canonical_rows), hidden, 1,
            "sanitizer canonical X MN");
        u_mn_map = make_bf16_tma_map(
            at<void>(arena, layout.backward_u_transpose), 64, intermediate,
            128, "sanitizer padded U");
        dab_mn_map = make_bf16_tma_map(
            at<void>(arena, layout.backward_dab_transpose), 64,
            2 * intermediate, 128, "sanitizer padded dAB");
        dw1_store_map = make_bf16_tma_store_map(
            at<void>(arena, layout.backward_dw1), hidden, 2 * intermediate,
            "sanitizer dW1 output");
      }
      const std::size_t backward_smem =
          (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(*input) +
          8U * sizeof(std::uint64_t);
      internal::throw_cuda(
          "backward-sanitizer",
          cudaFuncSetAttribute(local_p2p_backward_megakernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(backward_smem)),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "set Backward dynamic shared memory");
      launch.blockDim = dim3(384, 1, 1);
      launch.dynamicSmemBytes = backward_smem;
      if (!main_shape) {
        launch.gridDim = dim3(1, 1, 1);
        launch.blockDim = dim3(256, 1, 1);
        launch.dynamicSmemBytes = 0;
        launch.numAttrs = 1;
      }
      internal::throw_cuda(
          "backward-sanitizer",
          main_shape
              ? cudaLaunchKernelEx(
                    &launch, local_p2p_backward_megakernel<true>, backward, x_map,
                    dy_map, dab_map, w1_map, w2_transpose_map,
                    w1_transpose_map, canonical_dy_mn_map,
                    canonical_x_mn_map, u_mn_map, dab_mn_map, dw1_store_map)
              : cudaLaunchKernelEx(&launch,
                                   runtime_shape_backward_megakernel,
                                   backward),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "production Backward persistent kernel");
      BackwardDeviceStatus backward_status{};
      internal::throw_cuda(
          "backward-sanitizer",
          cudaMemcpyAsync(&backward_status,
                          at<void>(arena, layout.backward_status),
                          sizeof(backward_status), cudaMemcpyDeviceToHost,
                          stream),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "download Backward status");
      internal::throw_cuda(
          "backward-sanitizer", cudaStreamSynchronize(stream), 0, 1,
          "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "wait production Backward kernel");
      passed = backward_status.error == 0U &&
               (main_shape
                    ? (backward_status.wgmma_instructions != 0ULL &&
                       backward_status.tma_loads != 0ULL)
                    : (backward_status.dv_tasks == intermediate &&
                       backward_status.dx_tasks == hidden &&
                       backward_status.dw1_tasks ==
                           2ULL * intermediate * hidden &&
                       backward_status.dw2_tasks == hidden * intermediate));
      if (!main_shape && passed) {
        std::vector<transport::BFloat16> actual_forward(hidden);
        std::vector<transport::BFloat16> actual_dx(hidden);
        std::vector<float> actual_dgate(1);
        std::vector<transport::BFloat16> actual_dw1(2ULL * intermediate *
                                                    hidden);
        std::vector<transport::BFloat16> actual_dw2(hidden * intermediate);
        const auto copy_to_host = [&](void *host, std::size_t offset,
                                      std::size_t bytes,
                                      const char *field) {
          internal::throw_cuda(
              "runtime-shape-oracle",
              cudaMemcpy(host, at<const void>(arena, offset), bytes,
                         cudaMemcpyDeviceToHost),
              0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts, field);
        };
        copy_to_host(actual_forward.data(), layout.combined,
                     actual_forward.size() * sizeof(actual_forward[0]),
                     "download Forward output");
        copy_to_host(actual_dx.data(), layout.backward_dx_combined,
                     actual_dx.size() * sizeof(actual_dx[0]),
                     "download dX");
        copy_to_host(actual_dgate.data(), layout.backward_dgate,
                     sizeof(float), "download dGate");
        copy_to_host(actual_dw1.data(), layout.backward_dw1,
                     actual_dw1.size() * sizeof(actual_dw1[0]),
                     "download dW1");
        copy_to_host(actual_dw2.data(), layout.backward_dw2,
                     actual_dw2.size() * sizeof(actual_dw2[0]),
                     "download dW2");

        const auto mix = [](std::uint32_t value) {
          value ^= value >> 16U;
          value *= 0x7feb352dU;
          value ^= value >> 15U;
          value *= 0x846ca68bU;
          value ^= value >> 16U;
          return value;
        };
        const auto weight = [&](std::uint32_t kind, std::uint64_t row,
                                std::uint64_t column) {
          auto key = mix(0x9e3779b9U);
          key ^= mix(kind ^ 0x85ebca6bU);
          key ^= mix(static_cast<std::uint32_t>(row) ^ 0xc2b2ae35U);
          key ^= mix(static_cast<std::uint32_t>(column) ^ 0x27d4eb2fU);
          const auto sign = static_cast<int>(mix(key) & 1U) * 2 - 1;
          return static_cast<float>(sign) * (1.0F / 256.0F);
        };
        const auto quantize = [](float value) {
          return transport::bfloat16_to_float(
              transport::float_to_bfloat16(value));
        };
        const auto x = transport::bfloat16_to_float(
            transport::BFloat16{0x3f3fU});
        const auto dy_value = transport::bfloat16_to_float(
            transport::BFloat16{0x3e3eU});
        std::vector<float> av(intermediate), bv(intermediate),
            ref_s(intermediate), dv(intermediate), da(intermediate),
            db(intermediate);
        for (std::uint64_t i = 0; i < intermediate; ++i) {
          float a_sum = 0.0F;
          float b_sum = 0.0F;
          for (std::uint64_t k = 0; k < hidden; ++k) {
            a_sum += x * weight(1U, i, k);
            b_sum += x * weight(1U, intermediate + i, k);
          }
          av[i] = quantize(a_sum);
          bv[i] = quantize(b_sum);
          const auto sigmoid = 1.0F / (1.0F + std::exp(-av[i]));
          ref_s[i] = quantize((av[i] * sigmoid) * bv[i]);
          float dv_sum = 0.0F;
          for (std::uint64_t h = 0; h < hidden; ++h)
            dv_sum += dy_value * weight(2U, h, i);
          dv[i] = quantize(dv_sum);
          const auto ds = dv[i]; // unit gate in this harness
          da[i] = quantize(ds * bv[i] * sigmoid *
                           (1.0F + av[i] * (1.0F - sigmoid)));
          db[i] = quantize(ds * (av[i] * sigmoid));
        }
        const auto close = [](float actual, float expected,
                              float tolerance) {
          return std::isfinite(actual) &&
                 std::abs(actual - expected) <= tolerance;
        };
        for (std::uint64_t h = 0; h < hidden && passed; ++h) {
          float forward_sum = 0.0F;
          float dx_sum = 0.0F;
          for (std::uint64_t i = 0; i < intermediate; ++i) {
            forward_sum += ref_s[i] * weight(2U, h, i);
            dx_sum += da[i] * weight(1U, i, h);
            dx_sum += db[i] * weight(1U, intermediate + i, h);
            const auto expected_dw2 = quantize(dy_value * ref_s[i]);
            passed = passed && close(
                transport::bfloat16_to_float(
                    actual_dw2[h * intermediate + i]),
                expected_dw2, 0.02F);
          }
          passed = passed && close(
              transport::bfloat16_to_float(actual_forward[h]),
              quantize(forward_sum), 0.02F);
          passed = passed && close(
              transport::bfloat16_to_float(actual_dx[h]), quantize(dx_sum),
              0.02F);
        }
        float dgate_sum = 0.0F;
        for (std::uint64_t i = 0; i < intermediate; ++i)
          dgate_sum += dv[i] * ref_s[i];
        passed = passed && close(actual_dgate[0], dgate_sum, 0.02F);
        for (std::uint64_t row = 0; row < 2ULL * intermediate && passed;
             ++row) {
          const auto gradient = row < intermediate ? da[row]
                                                   : db[row - intermediate];
          for (std::uint64_t h = 0; h < hidden; ++h)
            passed = passed && close(
                transport::bfloat16_to_float(
                    actual_dw1[row * hidden + h]),
                quantize(gradient * x), 0.02F);
        }
      }
    }
  } catch (...) {
    cleanup();
    throw;
  }
  cleanup();
  return passed;
}
} // namespace

bool testing::run_forward_world1_sanitizer_harness() {
  return run_world1_forward_backward_sanitizer_harness(false);
}

bool testing::run_backward_world1_sanitizer_harness() {
  return run_world1_forward_backward_sanitizer_harness(true);
}

bool testing::run_runtime_shape_world1_harness(std::uint64_t hidden,
                                               std::uint64_t intermediate,
                                               bool run_backward) {
  return run_world1_forward_backward_sanitizer_harness(
      run_backward, hidden, intermediate);
}

} // namespace megamoe::cuda
