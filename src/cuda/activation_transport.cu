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
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {
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

std::size_t checked_size(std::uint64_t value, const char *field) {
  if (value > std::numeric_limits<std::size_t>::max())
    transport_error(NotifyErrorCode::ArithmeticOverflow, "transport-pool-query",
                    field, 0, "value <= size_t max", std::to_string(value));
  return static_cast<std::size_t>(value);
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
  unsigned int reserved;
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
  unsigned int error;
  unsigned int error_field;
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
  std::size_t packet_claim = 0;
  std::size_t canonical_rows = 0;
  std::size_t contribution = 0;
  std::size_t combine_return = 0;
  std::size_t combined = 0;
  std::size_t forward_s = 0;
  std::size_t forward_ab_cache = 0;
  std::size_t forward_task_claim = 0;
  std::size_t forward_fc1_done = 0;
  std::size_t forward_scheduler = 0;
  std::size_t backward_dy_input = 0;
  std::size_t backward_forward_output = 0;
  std::size_t backward_dy_canonical = 0;
  std::size_t backward_canonical_gate = 0;
  std::size_t backward_dab = 0;
  std::size_t backward_recompute_ab = 0;
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
  std::size_t backward_dw_timeline = 0;
  std::size_t backward_status = 0;
  std::size_t status = 0;
  std::size_t total = 0;
};

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
  layout.peer_bases = append(cursor, config.max_world_size, sizeof(void *),
                             "P2P peer base table");
  layout.p2p_barrier_send =
      append(cursor, 2, sizeof(std::uint64_t), "P2P barrier send");
  layout.p2p_barrier_receive = append(cursor,
                                      checked_mul(config.max_world_size, 2,
                                                  "P2P barrier words"),
                                      sizeof(std::uint64_t),
                                      "P2P barrier receive");
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
        checked_mul(max_mblocks, fc1_ntiles + fc2_ntiles,
                    "forward task claims"),
        sizeof(std::uint64_t), "forward task claims");
    layout.forward_fc1_done = append(cursor, max_mblocks,
                                     sizeof(std::uint64_t),
                                     "forward FC1 done tiles");
    layout.forward_scheduler = append(
        cursor, 4 + config.max_world_size * 32ULL,
        sizeof(std::uint64_t), "forward scheduler");
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
      layout.backward_dx_return = append(
          cursor, route_elements, sizeof(transport::BFloat16),
          "backward dX return");
      layout.backward_dx_combined = append(
          cursor, checked_mul(config.max_tokens_per_rank, hidden,
                              "backward combined dX"),
          sizeof(transport::BFloat16), "backward combined dX");
      const auto intermediate_tiles = (intermediate + 255ULL) / 256ULL;
      layout.backward_dgate_partial = append(
          cursor, checked_mul(routes, intermediate_tiles,
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
      layout.backward_dw_timeline = append(
          cursor, max_owned_experts, sizeof(BackwardExpertDwTimeline),
          "backward per-expert dW timeline");
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
  TransportDeviceStatus *host_status = nullptr;
  std::uint64_t *host_backward_progress = nullptr;
  std::uint64_t *device_backward_progress = nullptr;
  std::size_t backward_progress_words = 0;
  std::uint64_t *host_cancel_epoch = nullptr;
  std::uint64_t *device_cancel_epoch = nullptr;
  cudaEvent_t producer_ready = nullptr;
  cudaEvent_t start = nullptr;
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

struct ActivationPoolState {
  ActivationTransportPoolConfig config;
  Layout layout;
  std::vector<std::byte> session_identity;
  std::uint32_t world = 0;
  CudaExecution execution;
  std::uint32_t device = 0;
  std::mutex mutex;
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
      if (slot.host_backward_progress)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_backward_progress)
        (void)cudaFreeHost(slot.host_backward_progress);
      if (slot.host_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::PinnedFree);
      if (slot.host_cancel_epoch)
        (void)cudaFreeHost(slot.host_cancel_epoch);
      if (slot.device_cancel_epoch)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device_cancel_epoch)
        (void)cudaFree(slot.device_cancel_epoch);
      if (slot.device)
        internal::record_cuda_api(internal::CudaApiAuditKind::DeviceFree);
      if (slot.device)
        (void)cudaFree(slot.device);
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

// The storage is owned by the local rank: ingress_published is written by
// remote Dispatch producers and outgoing_sequence is written by remote Return
// producers. Initialize only the local allocation, then publish setup-ready
// to every peer. No peer may issue a put until every owner has published the
// same epoch, which prevents initialization racing a remote release store.
__global__ void initialize_forward_generation_signals_kernel(
    void *local_base, void **peer_bases, std::size_t ingress_published_offset,
    std::size_t outgoing_sequence_offset, std::size_t setup_epoch_offset,
    std::uint32_t rank, std::uint32_t world, std::uint32_t channels,
    unsigned long long epoch) {
  const auto index = static_cast<std::uint32_t>(threadIdx.x);
  auto *incoming = peer_at<unsigned long long>(local_base,
                                                ingress_published_offset);
  auto *outgoing = peer_at<unsigned long long>(local_base,
                                                outgoing_sequence_offset);
  const auto initial = epoch - 1ULL;
  for (std::uint32_t link = index; link < world * channels;
       link += blockDim.x) {
    incoming[link] = initial;
    outgoing[link] = initial;
  }
  __syncthreads();
  __threadfence_system();
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x) {
    auto *ready = peer_at<unsigned long long>(peer_bases[peer],
                                               setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ready(ready[rank]);
    peer_ready.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  auto *local_ready = peer_at<unsigned long long>(local_base,
                                                   setup_epoch_offset);
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x) {
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
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x) {
    auto *ack = peer_at<unsigned long long>(peer_bases[peer],
                                             setup_epoch_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system>
        peer_ack(ack[world + rank]);
    peer_ack.store(epoch, ::cuda::memory_order_release);
  }
  __syncthreads();
  for (std::uint32_t peer = index; peer < world; peer += blockDim.x) {
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
  std::size_t terminal_epoch_offset;
  const unsigned long long *cancel_epoch;
  bool cache_fc1;
};

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
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
__device__ std::uint32_t forward_mblock_expert(const ForwardKernelArgs &a,
                                               std::uint64_t mblock,
                                               std::uint64_t &local_mblock) {
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const auto blocks = (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
    if (mblock < blocks) {
      local_mblock = mblock;
      return expert;
    }
    mblock -= blocks;
  }
  local_mblock = 0;
  return a.dispatch.sizes.owned_expert_count;
}

#endif

__device__ bool forward_cancelled(const ForwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
             a.epoch;
}

__global__ __launch_bounds__(384, 1) void
local_p2p_forward_megakernel(
    ForwardKernelArgs a, const __grid_constant__ CUtensorMap x_map,
    const __grid_constant__ CUtensorMap s_map,
    const __grid_constant__ CUtensorMap w1_map,
    const __grid_constant__ CUtensorMap w2_map) {
  namespace cg = cooperative_groups;
  auto grid = cg::this_grid();
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
  grid.sync();
#if defined(CUTE_ARCH_MMA_SM90A_ENABLED) && !MEGAMOE_DEBUG_VALIDATE
  // Release reuses the otherwise debug-only claim arena as an immutable
  // mblock lookup.  This removes the repeated expert-prefix scan from every
  // FC1/FC2 N tile and enables chunked task acquisition below.
  for (std::uint64_t mblock =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       mblock < scheduler[3];
       mblock += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
    std::uint64_t local_mblock = 0;
    const auto expert = forward_mblock_expert(a, mblock, local_mblock);
    claims[mblock] = (static_cast<unsigned long long>(expert) << 32U) |
                     static_cast<std::uint32_t>(local_mblock);
  }
  grid.sync();
#endif

  // DispatchPut: all resident CTAs copy independent canonical rows.
  unsigned long long dispatch_peer_rows = 0ULL;
  for (std::uint64_t route = blockIdx.x; route < a.dispatch.sizes.route_count;
       route += gridDim.x) {
    const auto destination = a.dispatch.route_dst_rank[route];
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
  if (threadIdx.x == 0 && dispatch_peer_rows != 0ULL)
    atomicAdd(&status->actual_dispatch_peer_bytes,
              dispatch_peer_rows * a.hidden * sizeof(transport::BFloat16));
  // Every participating lane orders its peer payload stores before the team
  // rendezvous. A fence issued only by the signal lane would not cover stores
  // performed by the other lanes in the CTA.
  __threadfence_system();
  grid.sync();
  if (blockIdx.x == 0) {
    if (threadIdx.x < a.world) {
      auto *signal = peer_at<unsigned long long>(
          a.peer_bases[threadIdx.x], a.ingress_published_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> published(
          signal[a.rank * a.channels]);
      published.store(a.epoch, ::cuda::memory_order_release);
    }
    for (std::uint32_t source = threadIdx.x; source < a.world;
         source += blockDim.x) {
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
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->dispatch_complete_clock = device_globaltimer_ns();
    status->fc1_start_clock = status->dispatch_complete_clock;
  }

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)
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
  const auto fc1_ntiles =
      static_cast<std::uint32_t>((a.intermediate + 127ULL) / 128ULL);
  const auto forward_mblocks = scheduler[3];
  const auto fc1_tasks = forward_mblocks * fc1_ntiles;
  constexpr unsigned int kTaskChunk = MEGAMOE_DEBUG_VALIDATE ? 1U : 4U;
  __shared__ unsigned long long task_shared;
#if MEGAMOE_DEBUG_VALIDATE
  __shared__ std::uint32_t expert_shared;
  __shared__ std::uint64_t local_mblock_shared;
  __shared__ unsigned int claim_shared;
#endif
  unsigned int task_in_chunk = kTaskChunk;
  while (!forward_cancelled(a)) {
    if (task_in_chunk == kTaskChunk) {
      if (threadIdx.x == 0)
        task_shared = atomicAdd(&scheduler[0],
                                static_cast<unsigned long long>(kTaskChunk));
      __syncthreads();
      task_in_chunk = 0U;
    }
    const auto task = task_shared + task_in_chunk++;
    if (task >= fc1_tasks)
      break;
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0) {
      const auto old = atomicAdd(&claims[task], 0ULL);
      claim_shared = old != a.epoch &&
                             atomicCAS(&claims[task], old, a.epoch) == old
                         ? 1U
                         : 0U;
    }
    __syncthreads();
    if (claim_shared == 0U)
      continue;
#else
    (void)claims;
#endif
    const auto mblock = task / fc1_ntiles;
    const auto n_tile = static_cast<std::uint32_t>(task % fc1_ntiles);
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0)
      expert_shared = forward_mblock_expert(
          a, mblock, local_mblock_shared);
    __syncthreads();
    const auto expert = expert_shared;
    const auto local_mblock = local_mblock_shared;
#else
    const auto mapping = claims[mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
#endif
    if (expert >= a.dispatch.sizes.owned_expert_count)
      break;
    const auto row =
        a.dispatch.expert_row_prefix[expert] + local_mblock * 128ULL;
    const auto remaining_rows =
        a.dispatch.expert_row_count[expert] - local_mblock * 128ULL;
    const auto valid_rows = static_cast<std::uint32_t>(
        remaining_rows < 128ULL ? remaining_rows : 128ULL);
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
            const float av[2] = {av0, av1};
            const float bv[2] = {bv0, bv1};
#pragma unroll
            for (std::uint32_t value = 0; value < 2U; ++value) {
              const auto output_column =
                  static_cast<std::uint64_t>(n) + column + value;
              if (block_m < valid_rows && output_column < a.intermediate) {
                const auto output_index =
                    (row + block_m) * a.intermediate + output_column;
                // Hopper's native exponential is the forward specialization's
                // documented SiLU approximation.  Keeping it at this fixed
                // point makes both cache modes deterministic while avoiding a
                // scalar libdevice call for every FC1 output element.
                const auto sigmoid = 1.0F / (1.0F + __expf(-av[value]));
                s[output_index] = transport::BFloat16{
                    float_to_bf16_bits((av[value] * sigmoid) * bv[value])};
                if (a.cache_fc1 && ab_cache) {
                  ab_cache[output_index] = transport::BFloat16{
                      float_to_bf16_bits(av[value])};
                  ab_cache[a.max_destination_rows * a.intermediate +
                           output_index] = transport::BFloat16{
                      float_to_bf16_bits(bv[value])};
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
  }
  if (threadIdx.x == 0)
    atomicMax(&status->fc1_complete_clock, device_globaltimer_ns());

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

  const auto fc2_ntiles =
      static_cast<std::uint32_t>((a.hidden + 255ULL) / 256ULL);
  const auto fc2_tasks = forward_mblocks * fc2_ntiles;
  task_in_chunk = kTaskChunk;
  while (!forward_cancelled(a)) {
    if (task_in_chunk == kTaskChunk) {
      if (threadIdx.x == 0)
        task_shared = atomicAdd(&scheduler[1],
                                static_cast<unsigned long long>(kTaskChunk));
      __syncthreads();
      task_in_chunk = 0U;
    }
    const auto task = task_shared + task_in_chunk++;
    if (task >= fc2_tasks)
      break;
    const auto claim_index = fc1_tasks + task;
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0) {
      const auto old = atomicAdd(&claims[claim_index], 0ULL);
      claim_shared = old != a.epoch &&
                             atomicCAS(&claims[claim_index], old, a.epoch) == old
                         ? 1U
                         : 0U;
    }
    __syncthreads();
    if (claim_shared == 0U)
      continue;
#else
    (void)claim_index;
#endif
    const auto mblock = task / fc2_ntiles;
    const auto n_tile = static_cast<std::uint32_t>(task % fc2_ntiles);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device>
        fc1_ready(fc1_done[mblock]);
    while ((fc1_ready.load(::cuda::memory_order_acquire) & 0xffffffffULL) <
           fc1_ntiles) {
    }
    if (threadIdx.x == 0)
      atomicMin(&status->dispatch_blocks_done, device_globaltimer_ns());
#if MEGAMOE_DEBUG_VALIDATE
    if (threadIdx.x == 0)
      expert_shared = forward_mblock_expert(
          a, mblock, local_mblock_shared);
    __syncthreads();
    const auto expert = expert_shared;
    const auto local_mblock = local_mblock_shared;
#else
    const auto mapping = claims[mblock];
    const auto expert = static_cast<std::uint32_t>(mapping >> 32U);
    const auto local_mblock = static_cast<std::uint32_t>(mapping);
#endif
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
      if (return_mblock_shared != 0U) {
        const auto finished = atomicAdd(&scheduler[4U + a.world], 1ULL) + 1ULL;
        if (finished == forward_mblocks)
          status->fc2_complete_clock = device_globaltimer_ns();
      }
    }
    __syncthreads();
    if (return_mblock_shared != 0U) {
      __shared__ unsigned int return_peer_rows_shared;
      if (threadIdx.x == 0)
        atomicMin(&status->relay_blocks_done, device_globaltimer_ns());
      for (std::uint32_t source = threadIdx.x; source < a.world;
           source += blockDim.x)
        return_rows_by_source[source] = 0U;
      if (threadIdx.x == 0)
        return_peer_rows_shared = 0U;
      __syncthreads();
      for (std::uint32_t local_row = 0; local_row < valid_rows; ++local_row) {
        const auto return_row = row + local_row;
        const auto header = headers[return_row];
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
            ++return_peer_rows_shared;
        }
      }
      if (threadIdx.x == 0 && return_peer_rows_shared != 0U)
        atomicAdd(&status->actual_return_peer_bytes,
                  static_cast<unsigned long long>(return_peer_rows_shared) *
                      a.hidden * sizeof(transport::BFloat16));
      __threadfence_system();
      __syncthreads();
      for (std::uint32_t source_rank = threadIdx.x; source_rank < a.world;
           source_rank += blockDim.x) {
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
  }
#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->error = 1U;
    status->error_field = 90U;
  }
#endif
  if (threadIdx.x == 0)
    atomicMax(&status->return_complete_clock, device_globaltimer_ns());

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
  const auto token_warp = warp % kWarpsPerToken;
  for (std::uint64_t token =
           static_cast<std::uint64_t>(blockIdx.x) * kTokensPerCta +
           warp / kWarpsPerToken;
       token < a.dispatch.identity.num_tokens && !forward_cancelled(a);
       token += static_cast<std::uint64_t>(gridDim.x) * kTokensPerCta) {
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
  }
  if (lane == 0U)
    atomicMax(&status->reduce_blocks_done, device_globaltimer_ns());
  __syncthreads();
  grid.sync();
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
}

struct BackwardKernelArgs {
  DeviceDispatchKernelView dispatch;
  const transport::BFloat16 *dy_input;
  const float *gates;
  const transport::BFloat16 *w1;
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
  std::size_t canonical_x_offset;
  std::size_t direct_headers_offset;
  std::size_t dy_canonical_offset;
  std::size_t canonical_gate_offset;
  std::size_t dab_offset;
  std::size_t recompute_ab_offset;
  std::size_t backward_s_offset;
  std::size_t forward_ab_cache_offset;
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
  std::size_t status_offset;
  const unsigned long long *cancel_epoch;
  unsigned long long *progress;
  std::uint32_t progress_experts;
  bool use_forward_cache;
};

__device__ bool backward_cancelled(const BackwardKernelArgs &a);

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

__device__ std::uint32_t backward_mblock_expert(
    const BackwardKernelArgs &a, std::uint64_t mblock,
    std::uint64_t &local_mblock) {
  for (std::uint32_t expert = 0;
       expert < a.dispatch.sizes.owned_expert_count; ++expert) {
    const auto blocks = (a.dispatch.expert_row_count[expert] + 127ULL) / 128ULL;
    if (mblock < blocks) {
      local_mblock = mblock;
      return expert;
    }
    mblock -= blocks;
  }
  local_mblock = 0;
  return a.dispatch.sizes.owned_expert_count;
}

#endif

__device__ bool backward_cancelled(const BackwardKernelArgs &a) {
  return a.cancel_epoch &&
         atomicAdd_system(
             const_cast<unsigned long long *>(a.cancel_epoch), 0ULL) ==
             a.epoch;
}

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
  extern __shared__ __align__(1024) unsigned char storage[];
  auto *status = peer_at<BackwardDeviceStatus>(a.local_base, a.status_offset);
  auto *scheduler = peer_at<unsigned long long>(a.local_base,
                                                a.scheduler_offset);
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
  auto *ab = a.use_forward_cache
                 ? peer_at<transport::BFloat16>(a.local_base,
                                                 a.forward_ab_cache_offset)
                       : peer_at<transport::BFloat16>(a.local_base,
                                                 a.recompute_ab_offset);
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
  }
  grid.sync();
  for (std::uint32_t expert =
           static_cast<std::uint32_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       expert < a.dispatch.sizes.owned_expert_count;
       expert += static_cast<std::uint32_t>(gridDim.x) * blockDim.x) {
    dw_timeline[expert] = BackwardExpertDwTimeline{};
    if (a.dispatch.expert_row_count[expert] == 0ULL) {
      dw_timeline[expert].ready_clock = status->start_clock;
      atomicMin(&status->dw1_first_ready_clock, status->start_clock);
      atomicMin(&status->dw2_first_ready_clock, status->start_clock);
    }
    if (a.progress != nullptr && expert < a.progress_experts) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> rows(
          a.progress[kBackwardProgressExpertBase + expert * 8U]);
      rows.store(a.dispatch.expert_row_count[expert],
                 ::cuda::memory_order_release);
    }
  }
  grid.sync();
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
    const auto expert = backward_mblock_expert(a, mblock, local);
    claims[a.dispatch.sizes.owned_expert_count + mblock] =
        (static_cast<unsigned long long>(expert) << 32U) |
        static_cast<std::uint32_t>(local);
#endif
  }
  grid.sync();

  // DYDispatchPut: dY and its route-local source gate share the exact Forward
  // canonical row mapping. Gate stays out of the activation payload contract.
  unsigned long long dy_peer_rows = 0ULL;
  for (std::uint64_t route = blockIdx.x; route < a.dispatch.sizes.route_count;
       route += gridDim.x) {
    const auto destination = a.dispatch.route_dst_rank[route];
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
    if (threadIdx.x == 0) {
      destination_gate[row] = a.gates[route];
      if (destination != a.rank)
        ++dy_peer_rows;
    }
  }
  if (threadIdx.x == 0 && dy_peer_rows != 0ULL)
    atomicAdd(&status->actual_dy_peer_bytes,
              dy_peer_rows * a.hidden * sizeof(transport::BFloat16));
  __threadfence_system();
  grid.sync();
  if (blockIdx.x == 0) {
    if (threadIdx.x < a.world) {
      auto *signal = peer_at<unsigned long long>(
          a.peer_bases[threadIdx.x], a.incoming_signal_offset);
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          signal[a.rank * a.channels]);
      ready.store(a.epoch, ::cuda::memory_order_release);
    }
    auto *signal = peer_at<unsigned long long>(a.local_base,
                                                a.incoming_signal_offset);
    for (std::uint32_t source = threadIdx.x; source < a.world;
         source += blockDim.x) {
      ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
          signal[source * a.channels]);
      while (ready.load(::cuda::memory_order_acquire) < a.epoch &&
             !backward_cancelled(a)) {
      }
    }
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dy_dispatch_complete_clock = device_globaltimer_ns();
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
  backward_publish_progress(a, kBackwardDvStarted, dv_tasks);
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
    __syncthreads();
  }
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dv_complete_clock = device_globaltimer_ns();
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
        auto *source_dx = peer_at<transport::BFloat16>(
            a.peer_bases[header.source_rank], a.dx_return_offset);
        reinterpret_cast<uint4 *>(
            source_dx + header.route_index * a.hidden + output_column)[0] =
            packed;
        if (ntile == 0U && local_vector == 0U) {
          auto *source_dgate = peer_at<float>(
              a.peer_bases[header.source_rank], a.dgate_offset);
          source_dgate[header.route_index] = dgate_canonical[row + local_row];
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
  backward_publish_progress(a, kBackwardDxComplete, dx_tasks);

  // Every dX contribution and dGate scalar is now globally visible. Publish
  // the per-source completion before dW so peer acquire latency is hidden by
  // the independent weight-gradient work instead of extending the tail.
  if (blockIdx.x == 0 && threadIdx.x < a.world) {
    auto *signal = peer_at<unsigned long long>(
        a.peer_bases[threadIdx.x], a.outgoing_signal_offset);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_system> ready(
        signal[a.rank * a.channels]);
    ready.store(a.epoch, ::cuda::memory_order_release);
  }

  // dW2 and dW1 use split_k=1. Each task owns one output tile and traverses
  // that expert's padded canonical-M segment in ascending order.
  // dW2 is evaluated as dW2^T = U^T @ dY.  Its epilogue transposes each
  // 128x128 accumulator tile directly into the parameter layout [H,I].
  backward_publish_progress(a, kBackwardDw2Started, dw2_tasks);
  while (true) {
    if (threadIdx.x == 0)
      task_shared = atomicAdd(&scheduler[3], 1ULL);
    __syncthreads();
    if (task_shared >= dw2_tasks)
      break;
    auto task = task_shared;
    const auto per_expert = dw2_mtiles * dw2_ntiles;
    const auto expert = static_cast<std::uint32_t>(task / per_expert);
    task %= per_expert;
    const auto ntile = task / dw2_mtiles;
    const auto mtile = task % dw2_mtiles;
    const auto reduction_start = claims[expert];
    const auto reduction =
        (a.dispatch.expert_row_count[expert] + 63ULL) & ~63ULL;
    if (threadIdx.x == 0U) {
      const auto now = device_globaltimer_ns();
      if (atomicCAS(&dw_timeline[expert].dw2_first_start_clock, 0ULL, now) ==
          0ULL)
        atomicMin(&status->dw2_first_start_clock, now);
    }
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
  // Release timing needs only the true global completion boundary. One
  // timestamp per CTA replaces the per-tile audit atomics retained in debug.
  if (threadIdx.x == 0U)
    atomicMax(&status->dw2_last_complete_clock, device_globaltimer_ns());
#endif
  backward_publish_progress(a, kBackwardDw2Complete, dw2_tasks);

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
    task_shared = blockIdx.x;
  __syncthreads();
  backward_publish_progress(a, kBackwardDw1Started, dw1_tasks);
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
      if (threadIdx.x == 0) {
        const auto now = device_globaltimer_ns();
        if (atomicCAS(&dw_timeline[expert].dw1_first_start_clock, 0ULL, now) ==
            0ULL)
          atomicMin(&status->dw1_first_start_clock, now);
      }
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
        task_shared += gridDim.x;
    }
  }
  backward_publish_progress(a, kBackwardDw1Complete, dw1_tasks);
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->dw2_complete_clock = status->dw2_last_complete_clock;
#if !MEGAMOE_DEBUG_VALIDATE
    status->dw1_last_complete_clock = device_globaltimer_ns();
#endif
    status->dw1_complete_clock = status->dw1_last_complete_clock;
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
  }
  backward_reduce_token_slice(a, dx_return, dx_combined, blockIdx.x,
                              gridDim.x);
  grid.sync();
  if (blockIdx.x == 0 && threadIdx.x == 0)
    status->dx_reduce_complete_clock = device_globaltimer_ns();
  backward_publish_progress(a, kBackwardReduceComplete,
                            a.dispatch.identity.num_tokens);

#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    status->error = 1U;
    status->error_field = 90U;
  }
  grid.sync();
#endif

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
    backward_publish_progress(a, kBackwardTerminal, a.epoch);
  }
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
};

struct LocalP2pBackendState;

struct DeviceActivationTransportHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  ActivationTransportExpectation identity;
  ActivationTransportDeviceAudit audit;
  ActivationTransportTimings timings;
  std::uint64_t route_count = 0;
  std::uint64_t destination_rows = 0;
  std::shared_ptr<LocalP2pBackendState> local_p2p_backend;
};

struct DeviceActivationTransport::Impl {
  CollectiveSession session;
  CudaExecution execution;
  ActivationTransportResourcePool pool;
};

struct LocalP2pBackendState {
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<internal::NcclSessionState> nccl;
  NcclSession session;
  CudaExecution execution;
  std::vector<void *> peer_slot_bases;
  std::vector<std::uint8_t> owns_mapping;

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
  ~ForwardAllocation() noexcept {
    (void)cudaSetDevice(static_cast<int>(device));
    if (ready) {
      internal::record_cuda_api(internal::CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(ready);
    }
    if (w2) {
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
    if (w1) {
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
  std::shared_ptr<LocalP2pBackendState> backend;
  DeviceDispatchHandle dispatch;
  BFloat16ActivationLease activations;
  ForwardWeightLease weights;
  ActivationTransportExpectation identity;
  ForwardMegaKernelAudit audit;
  ForwardMegaKernelTimings timings;
  ForwardCacheMode cache_mode = ForwardCacheMode::NoFc1OutputCache;
  std::uint64_t incoming_rows = 0;
  std::uint64_t intermediate = 0;
};

struct LocalCudaP2pForwardMegaKernel::Impl {
  std::shared_ptr<LocalP2pBackendState> backend;
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
};

struct DeviceBackwardHandle::Impl {
  std::shared_ptr<SlotLease> slot;
  std::shared_ptr<ActivationPoolState> pool;
  std::shared_ptr<LocalP2pBackendState> backend;
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
  std::shared_ptr<LocalP2pBackendState> backend;
};

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
  impl->w1_map = make_bf16_tma_map(
      impl->allocation->w1, hidden_size,
      static_cast<std::uint64_t>(owned_experts) * 2ULL * intermediate_size,
      128, "W1");
  impl->w2_map = make_bf16_tma_map(
      impl->allocation->w2, intermediate_size,
      static_cast<std::uint64_t>(owned_experts) * hidden_size, 256, "W2");
  return ForwardWeightLease(std::move(impl));
}

void testing::prepare_forward_weights_for_backward(
    const ForwardWeightLease &weights, const CudaExecution &execution) {
  if (!weights || !execution)
    transport_error(NotifyErrorCode::InvalidHandle,
                    "backward-weight-prepare", "weights/execution", 0,
                    "valid", "empty");
  auto &impl = ForwardBackwardWeightAccess::get(weights);
  const auto state = internal::checked_execution_state(execution);
  if (state->device != impl.device ||
      !execution.token().same_instance(impl.execution))
    transport_error(NotifyErrorCode::WorkspaceMismatch,
                    "backward-weight-prepare", "device/execution", 0,
                    "exact weight producer execution", "foreign");
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
    impl.w1_transpose_map = make_bf16_tma_map(
        impl.allocation->w1_transpose, 2ULL * impl.intermediate,
        static_cast<std::uint64_t>(impl.owned_experts) * impl.hidden, 256,
        "W1 transpose for dX");
    impl.w2_transpose_map = make_bf16_tma_map(
        impl.allocation->w2_transpose, impl.hidden,
        static_cast<std::uint64_t>(impl.owned_experts) * impl.intermediate,
        128, "W2 transpose for dV");
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

void testing::flip_backward_dx_byte_for_validation(
    const DeviceBackwardHandle &handle, std::uint64_t byte_offset,
    const CudaExecution &execution) {
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
  const auto dx_bytes = checked_mul(
      checked_mul(impl.identity.tokens, impl.hidden,
                  "backward byte flip dX elements"),
      sizeof(transport::BFloat16), "backward byte flip dX bytes");
  if (byte_offset >= dx_bytes)
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-byte-flip", "byte_offset", impl.identity.rank,
                    "<" + std::to_string(dx_bytes),
                    std::to_string(byte_offset));
  auto &slot = impl.pool->slots[impl.slot->index];
  flip_validation_byte_kernel<<<1, 1, 0, state->stream>>>(
      at<unsigned char>(slot.device,
                        impl.pool->layout.backward_dx_combined),
      byte_offset);
  internal::throw_cuda(
      "backward-byte-flip", cudaGetLastError(), impl.identity.rank,
      impl.identity.world, "backward-validation", impl.identity.generation,
      ExchangeKind::ExpertCounts, "flip actual dX byte");
  internal::throw_cuda(
      "backward-byte-flip", cudaStreamSynchronize(state->stream),
      impl.identity.rank, impl.identity.world, "backward-validation",
      impl.identity.generation, ExchangeKind::ExpertCounts,
      "wait actual dX byte flip");
  impl.hashes_captured = false;
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
  state->session_identity = session.stable_identity_bytes();
  state->world = session.world_size();
  state->execution = execution;
  state->device = execution.device_ordinal();
  internal::throw_cuda("transport-pool-create",
                       cudaSetDevice(static_cast<int>(state->device)), 0,
                       session.world_size(), session.identity(),
                       session.generation(), ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  state->slots.resize(config.max_inflight_generations);
  try {
    internal::throw_cuda(
        "transport-pool-create",
        cudaStreamCreateWithFlags(&state->cancel_stream, cudaStreamNonBlocking),
        0, session.world_size(), session.identity(), session.generation(),
        ExchangeKind::ExpertCounts, "cudaStreamCreate(cancel)");
    for (auto &slot : state->slots) {
      internal::throw_cuda("transport-pool-create",
                           cudaMalloc(&slot.device, state->layout.total), 0,
                           session.world_size(), session.identity(),
                           session.generation(), ExchangeKind::ExpertCounts,
                           "cudaMalloc(transport slot)");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
      internal::throw_cuda(
          "transport-pool-create",
          cudaMemsetAsync(slot.device, 0, state->layout.total,
                          internal::checked_execution_state(execution)->stream),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts,
          "one-time initialize complete transport slot storage");
      if (config.max_intermediate_size != 0) {
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
      }
      internal::throw_cuda(
          "transport-pool-create",
          cudaHostAlloc(reinterpret_cast<void **>(&slot.host_status),
                        sizeof(TransportDeviceStatus), cudaHostAllocPortable),
          0, session.world_size(), session.identity(), session.generation(),
          ExchangeKind::ExpertCounts, "cudaHostAlloc(transport status)");
      internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
      slot.backward_progress_words =
          32ULL + static_cast<std::size_t>(config.max_owned_experts) * 8ULL;
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
      state->layout.total + sizeof(std::uint64_t), state->slots.size(),
      "pool device bytes");
  result.pinned_bytes = checked_mul(
      sizeof(TransportDeviceStatus) + sizeof(std::uint64_t),
      state->slots.size(), "pool pinned bytes");
  result.event_count = checked_mul(4, state->slots.size(), "pool events");
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

LocalCudaP2pForwardMegaKernel::LocalCudaP2pForwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-create", "pool",
                    0, "valid", "empty");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
}

LocalCudaP2pBackwardMegaKernel::LocalCudaP2pBackwardMegaKernel(
    const NcclSession &session, const CudaExecution &execution,
    const ActivationTransportResourcePool &pool) {
  if (!pool.impl_ || !pool.impl_->state)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-create", "pool",
                    0, "valid", "empty");
  if (!pool.impl_->state->config.enable_backward)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-create",
                    "pool.enable_backward", 0, "true", "false");
  impl_ = std::make_shared<Impl>();
  impl_->backend =
      create_local_p2p_backend(session, execution, pool.impl_->state);
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
  result->audit.credit_stalls = slot.host_status->credit_stalls;
  result->audit.raw_cross_node_payload_bytes =
      slot.host_status->raw_cross_node_payload_bytes;
  result->audit.deduplicated_cross_node_payload_bytes =
      slot.host_status->deduplicated_cross_node_payload_bytes;
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

DeviceActivationTransportHandle LocalCudaP2pActivationTransport::run_synthetic(
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
  const auto state = backend->pool;
  const auto &id = dispatch.identity();
  const auto &activation = *activations.impl_;
  const auto &weight = *weights.impl_;
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=topology, rank=" +
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
      !activation.producer_execution.same_instance(backend->execution.token()) ||
      !activation.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "activation provenance", id.calling_rank,
                    "exact pool/session/generation/execution/shape/ready",
                    "mismatch");
  if (weight.session_identity !=
          backend->session.session().stable_identity_bytes() ||
      !weight.execution.same_instance(backend->execution.token()) ||
      weight.device != backend->execution.device_ordinal() ||
      weight.hidden != config.hidden_size ||
      weight.intermediate != config.intermediate_size || !weight.allocation ||
      !weight.allocation->ready)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "forward-run",
                    "weight provenance", id.calling_rank,
                    "exact session/execution/device/shape/ready", "mismatch");
  if (config.hidden_size != 7168 || config.intermediate_size != 3072)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=forward-run, field=specialization, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=sm90 WGMMA/TMA hidden=7168 intermediate=3072, actual=" +
            std::to_string(config.hidden_size) + "x" +
            std::to_string(config.intermediate_size));
  if (state->config.max_hidden_size != config.hidden_size ||
      state->config.max_intermediate_size != config.intermediate_size ||
      !state->slots[activation.slot->index].forward_maps_valid ||
      (config.cache_mode == ForwardCacheMode::CacheFc1InputOutput &&
       !state->config.enable_forward_cache))
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded, "forward-run",
                    "forward pool geometry/cache", id.calling_rank,
                    "exact main-shape TMA maps and requested cache storage",
                    "mismatch/missing");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
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
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.start, exec->stream),
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

  // Notify has already completed the rank-wide generation agreement on this
  // communicator.  The IPC team and per-slot peer mappings are session-static,
  // while payload visibility is ordered by the device release/acquire signals
  // below, so a second host-enqueued AllGather here only serialized E2E.

  if (slot.transport_epoch >= (1ULL << 63U) - 1ULL)
    transport_error(NotifyErrorCode::ArithmeticOverflow, "forward-run",
                    "transport_epoch", id.calling_rank, "<2^63-1",
                    std::to_string(slot.transport_epoch));
  const auto generation_epoch = ++slot.transport_epoch;
  initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
      slot.device, at<void *>(slot.device, layout.peer_bases),
      layout.ingress_published_sequence, layout.outgoing_route_sequence,
      layout.p2p_barrier_receive, id.calling_rank, id.world_size,
      state->config.channels, generation_epoch);
  internal::throw_cuda(
      "forward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize owner completion signals and wait peer setup epoch");
  ForwardKernelArgs args{};
  args.dispatch = view;
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
  args.forward_scheduler_offset = layout.forward_scheduler;
  args.terminal_epoch_offset = layout.p2p_barrier_receive;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  args.cache_fc1 =
      config.cache_mode == ForwardCacheMode::CacheFc1InputOutput;

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
  internal::throw_cuda(
      "forward-run",
      cudaFuncSetAttribute(local_p2p_forward_megakernel,
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
  int cluster_supported = 0;
  internal::throw_cuda(
      "forward-run",
      cudaDeviceGetAttribute(&cluster_supported, cudaDevAttrClusterLaunch,
                             static_cast<int>(exec->device)),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaDevAttrClusterLaunch");
  int max_active_clusters = 0;
  internal::throw_cuda(
      "forward-run",
      cudaOccupancyMaxActiveClusters(
          &max_active_clusters,
          reinterpret_cast<const void *>(local_p2p_forward_megakernel),
          &launch),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaOccupancyMaxActiveClusters");
  internal::throw_cuda(
      "forward-run",
      cudaLaunchKernelEx(&launch, local_p2p_forward_megakernel, args,
                         slot.forward_x_map, slot.forward_s_map,
                         weight.w1_map, weight.w2_map),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "single persistent WGMMA/TMA forward launch");
  // The persistent kernel includes the peer-visible terminal epoch barrier;
  // no host-enqueued terminal collective is needed here.
  internal::throw_cuda(
      "forward-run",
      cudaMemcpyAsync(slot.host_status, at<void>(slot.device, layout.status),
                      sizeof(TransportDeviceStatus), cudaMemcpyDeviceToHost,
                      exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "D2H forward status");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                            sizeof(TransportDeviceStatus));
  internal::throw_cuda(
      "forward-run", cudaEventRecord(slot.complete, exec->stream),
      id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts, "cudaEventRecord(forward complete)");
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
          "forward-run", query, id.calling_rank, id.world_size,
          backend->session.session().identity(), id.generation,
          ExchangeKind::ExpertCounts, "cudaEventQuery(forward complete)");
    const auto expired = options.deadline_after.count() > 0 &&
                         std::chrono::steady_clock::now() - wait_start >=
                             options.deadline_after;
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
  if (cancelled || slot.host_status->error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=forward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed forward, actual=cancelled");
  if (slot.host_status->error != 0)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "forward-kernel", "device status", id.calling_rank,
                    "error=0", "error=" +
                                     std::to_string(slot.host_status->error) +
                                     ", field=" +
                                     std::to_string(slot.host_status->error_field));

  float total_ms = 0.0F;
  internal::throw_cuda(
      "forward-timing",
      cudaEventElapsedTime(&total_ms, slot.start, slot.complete),
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
      ForwardKernelSpecialization::Sm90WgmmaTma7168x3072;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions = slot.host_status->wgmma_instructions;
  result->audit.tma_loads = slot.host_status->tma_loads;
  result->audit.tma_multicast_loads =
      slot.host_status->tma_multicast_loads;
  result->audit.runtime_cluster_supported = cluster_supported != 0;
  result->audit.requested_cluster_size = 2;
  result->audit.runtime_cluster_size =
      slot.host_status->runtime_cluster_size;
  result->audit.observed_cluster_rank_mask =
      slot.host_status->cluster_rank_mask;
  result->audit.max_active_clusters =
      static_cast<std::uint32_t>(std::max(max_active_clusters, 0));
  result->audit.fc1_tasks = slot.host_status->fc1_tasks_done;
  result->audit.fc2_tasks = slot.host_status->fc2_tasks_done;
  result->audit.actual_dispatch_peer_bytes =
      slot.host_status->actual_dispatch_peer_bytes;
  result->audit.actual_return_peer_bytes =
      slot.host_status->actual_return_peer_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
  const auto &status = *slot.host_status;
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
  return impl_->audit;
}

const ForwardMegaKernelTimings &DeviceForwardHandle::timings() const {
  if (!impl_)
    transport_error(NotifyErrorCode::InvalidHandle, "forward-handle", "handle",
                    0, "valid", "empty");
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

DeviceBackwardHandle LocalCudaP2pBackwardMegaKernel::run(
    const DeviceForwardHandle &forward,
    const SelectedOutputGradientLease &output_gradient,
    const BackwardMegaKernelConfig &config,
    const ActivationTransportRunOptions &options) {
  if (!impl_ || !impl_->backend || !forward || !output_gradient)
    transport_error(NotifyErrorCode::InvalidHandle, "backward-run",
                    "backend/forward/dY", 0, "valid", "empty");
  const auto backend = impl_->backend;
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
      !dy_lease.producer_execution.same_instance(backend->execution.token()) ||
      !dy_lease.ready_recorded)
    transport_error(NotifyErrorCode::WorkspaceMismatch, "backward-run",
                    "Forward/dY provenance", id.calling_rank,
                    "same pool/slot/session/generation/execution/shape/ready",
                    "mismatch");
  if (id.num_nodes != 1 || id.world_size != backend->nccl->world ||
      id.calling_rank != backend->nccl->rank)
    throw NotifyDispatchError(
        NotifyErrorCode::BackendUnavailable,
        "phase=backward-run, field=topology, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=one node exact NCCL team, actual=nodes=" +
            std::to_string(id.num_nodes) + ", world=" +
            std::to_string(id.world_size));
  if (config.hidden_size != 7168 || config.intermediate_size != 3072 ||
      f.weights.hidden_size() != config.hidden_size ||
      f.weights.intermediate_size() != config.intermediate_size ||
      !f.weights.impl_->allocation->backward_prepared ||
      !f.weights.impl_->w1_transpose || !f.weights.impl_->w2_transpose ||
      !state->config.enable_backward ||
      state->config.max_owned_experts < f.weights.owned_experts())
    transport_error(NotifyErrorCode::WorkspaceCapacityExceeded,
                    "backward-run", "specialization/pool/weights",
                    id.calling_rank,
                    "sm90 7168x3072 and sufficient Backward pool",
                    "mismatch");
  if (config.weight_gradient_dtype != BackwardWeightGradientDType::BFloat16)
    transport_error(NotifyErrorCode::InvalidRuntimeConfig, "backward-run",
                    "dW dtype", id.calling_rank, "BF16", "unsupported");

  const auto expectation = DeviceDispatchViewExpectation{
      id.generation, id.calling_rank, id.world_size, id.num_tokens, id.top_k,
      id.config_digest_word0, id.config_digest_word1};
  auto consumer = f.dispatch.acquire_consumer(backend->execution, expectation);
  const auto view = ActivationTransportBridge::view(consumer);
  auto exec = internal::checked_execution_state(backend->execution);
  auto &slot = state->slots[f.slot->index];
  const auto &layout = state->layout;
  SlotFailureGuard failure_guard{f.slot, false};
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
  initialize_forward_generation_signals_kernel<<<1, 256, 0, exec->stream>>>(
      slot.device, at<void *>(slot.device, layout.peer_bases),
      layout.ingress_published_sequence, layout.outgoing_route_sequence,
      layout.p2p_barrier_receive, id.calling_rank, id.world_size,
      state->config.channels, generation_epoch);
  internal::throw_cuda(
      "backward-run", cudaGetLastError(), id.calling_rank, id.world_size,
      backend->session.session().identity(), id.generation,
      ExchangeKind::ExpertCounts,
      "initialize Backward owner signals and peer setup epoch");

  BackwardKernelArgs args{};
  args.dispatch = view;
  args.dy_input = dy_lease.pointer;
  args.gates = ActivationTransportBridge::gate_pointer(f.dispatch);
  args.w1 = f.weights.impl_->w1;
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
  args.canonical_x_offset = layout.canonical_rows;
  args.direct_headers_offset = layout.direct_route_headers;
  args.dy_canonical_offset = layout.backward_dy_canonical;
  args.canonical_gate_offset = layout.backward_canonical_gate;
  args.dab_offset = layout.backward_dab;
  args.recompute_ab_offset = layout.backward_recompute_ab;
  args.backward_s_offset = layout.forward_s;
  args.forward_ab_cache_offset = layout.forward_ab_cache;
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
  args.dw_timeline_offset = layout.backward_dw_timeline;
  args.claim_offset = layout.forward_task_claim;
  args.incoming_signal_offset = layout.ingress_published_sequence;
  args.outgoing_signal_offset = layout.outgoing_route_sequence;
  args.status_offset = layout.backward_status;
  args.cancel_epoch = reinterpret_cast<const unsigned long long *>(
      slot.device_cancel_epoch);
  std::fill_n(slot.host_backward_progress, slot.backward_progress_words,
              0ULL);
  args.progress = std::getenv("MEGAMOE_BACKWARD_PROGRESS_DUMP") != nullptr
                      ? reinterpret_cast<unsigned long long *>(
                            slot.device_backward_progress)
                      : nullptr;
  args.progress_experts = state->config.max_owned_experts;
  args.use_forward_cache =
      f.cache_mode == ForwardCacheMode::CacheFc1InputOutput;

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
  internal::throw_cuda(
      "backward-run",
      cudaFuncSetAttribute(local_p2p_backward_megakernel,
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
  internal::throw_cuda(
      "backward-run",
      cudaLaunchKernelEx(
          &launch, local_p2p_backward_megakernel, args, slot.forward_x_map,
          slot.backward_dy_map, slot.backward_dab_map, f.weights.impl_->w1_map,
          f.weights.impl_->w2_transpose_map,
          f.weights.impl_->w1_transpose_map,
          slot.backward_dy_mn_map, slot.backward_x_mn_map,
          slot.backward_u_transpose_map, slot.backward_dab_transpose_map,
          slot.backward_dw1_store_map),
      id.calling_rank, id.world_size, backend->session.session().identity(),
      id.generation, ExchangeKind::ExpertCounts,
      "single persistent WGMMA/TMA Backward launch");
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
      std::cerr << "BACKWARD_PROGRESS rank=" << id.calling_rank
                << " generation=" << id.generation
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
  consumer.finish();
  const auto &status = *reinterpret_cast<const BackwardDeviceStatus *>(
      slot.host_status);
  if (cancelled || status.error == 2U)
    throw NotifyDispatchError(
        NotifyErrorCode::TransportTimeout,
        "phase=backward-run, field=cancel/deadline, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=completed Backward, actual=cancelled");
  if (status.error != 0U)
    transport_error(NotifyErrorCode::TransportProtocolFailure,
                    "backward-kernel", "device status", id.calling_rank,
                    "error=0", "error=" + std::to_string(status.error) +
                                     ", field=" +
                                     std::to_string(status.error_field));
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
      BackwardKernelSpecialization::Sm90WgmmaTma7168x3072;
  result->audit.persistent_kernel_launches = 1;
  result->audit.wgmma_instructions = status.wgmma_instructions;
  result->audit.tma_loads = status.tma_loads;
  result->audit.dv_tasks = status.dv_tasks;
  result->audit.dx_tasks = status.dx_tasks;
  result->audit.dw1_tasks = status.dw1_tasks;
  result->audit.dw2_tasks = status.dw2_tasks;
  result->audit.actual_dy_peer_bytes = status.actual_dy_peer_bytes;
  result->audit.actual_dx_peer_bytes = status.actual_dx_peer_bytes;
  result->audit.workspace_bytes = state->layout.total;
  result->audit.recomputed_fc1 = !args.use_forward_cache;
  result->audit.weight_gradient_dtype = config.weight_gradient_dtype;
  result->timings.total_us = static_cast<double>(total_ms) * 1000.0;
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
  copy(result.contribution_by_route.data(), impl.pool->layout.contribution,
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
      "ncclCommWindowRegister",
      "ncclCommWindowDeregister",
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
  result.gin_backend_implemented = false;

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
          const auto cuda_result = cudaMalloc(&allocation, 4096);
          if (cuda_result != cudaSuccess) {
            init_error = std::string("cudaMalloc(window): ") +
                         cudaGetErrorString(cuda_result);
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
      (void)cudaFree(allocation);
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
bool run_world1_forward_backward_sanitizer_harness(bool run_backward) {
  constexpr std::uint64_t hidden = 7168;
  constexpr std::uint64_t intermediate = 3072;
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
        layout.p2p_barrier_receive, 0, 1, 1, 1);
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
    args.terminal_epoch_offset = layout.p2p_barrier_receive;
    args.cancel_epoch =
        reinterpret_cast<const unsigned long long *>(cancel_epoch);
    args.cache_fc1 = true;
    const auto x_map = make_bf16_tma_map(
        at<void>(arena, layout.canonical_rows), hidden, 1, 128,
        "sanitizer canonical X");
    const auto s_map = make_bf16_tma_map(
        at<void>(arena, layout.forward_s), intermediate, 1, 128,
        "sanitizer S");
    const auto w1_map = make_bf16_tma_map(w1, hidden, 2 * intermediate, 128,
                                          "sanitizer W1");
    const auto w2_map = make_bf16_tma_map(w2, intermediate, hidden, 256,
                                          "sanitizer W2");
    const std::size_t smem =
        (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(*input) +
        8U * sizeof(std::uint64_t) + sizeof(std::uint32_t);
    internal::throw_cuda(
        "forward-sanitizer",
        cudaFuncSetAttribute(local_p2p_forward_megakernel,
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
    internal::throw_cuda(
        "forward-sanitizer",
        cudaLaunchKernelEx(&launch, local_p2p_forward_megakernel, args, x_map,
                           s_map, w1_map, w2_map),
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
    passed = status.error == 0U && status.wgmma_instructions != 0ULL &&
             status.tma_loads != 0ULL && status.runtime_cluster_size == 2U;
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
          layout.p2p_barrier_receive, 0, 1, 1, 2);
      BackwardKernelArgs backward{};
      backward.dispatch = view;
      backward.dy_input = dy_input;
      backward.gates = gates;
      backward.w1 = w1;
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
      backward.status_offset = layout.backward_status;
      backward.cancel_epoch =
          reinterpret_cast<const unsigned long long *>(cancel_epoch);
      backward.use_forward_cache = true;
      const auto dy_map = make_bf16_tma_map(
          at<void>(arena, layout.backward_dy_canonical), hidden, 1, 128,
          "sanitizer backward dY");
      const auto dab_map = make_bf16_tma_map(
          at<void>(arena, layout.backward_dab), 2 * intermediate, 1, 128,
          "sanitizer backward dAB");
      const auto w2_transpose_map = make_bf16_tma_map(
          w2_transpose, hidden, intermediate, 128,
          "sanitizer W2 transpose");
      const auto w1_transpose_map = make_bf16_tma_map(
          w1_transpose, 2 * intermediate, hidden, 256,
          "sanitizer W1 transpose");
      const auto canonical_dy_mn_map = make_bf16_tma_map_mn(
          at<void>(arena, layout.backward_dy_canonical), hidden, 1,
          "sanitizer canonical dY MN");
      const auto canonical_x_mn_map = make_bf16_tma_map_mn(
          at<void>(arena, layout.canonical_rows), hidden, 1,
          "sanitizer canonical X MN");
      const auto u_mn_map = make_bf16_tma_map(
          at<void>(arena, layout.backward_u_transpose), 64, intermediate, 128,
          "sanitizer padded U");
      const auto dab_mn_map = make_bf16_tma_map(
          at<void>(arena, layout.backward_dab_transpose), 64,
          2 * intermediate, 128, "sanitizer padded dAB");
      const auto dw1_store_map = make_bf16_tma_store_map(
          at<void>(arena, layout.backward_dw1), hidden, 2 * intermediate,
          "sanitizer dW1 output");
      const std::size_t backward_smem =
          (4U * 128U * 64U + 4U * 256U * 64U) * sizeof(*input) +
          8U * sizeof(std::uint64_t);
      internal::throw_cuda(
          "backward-sanitizer",
          cudaFuncSetAttribute(local_p2p_backward_megakernel,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               static_cast<int>(backward_smem)),
          0, 1, "world1-no-nccl", 1, ExchangeKind::ExpertCounts,
          "set Backward dynamic shared memory");
      launch.blockDim = dim3(384, 1, 1);
      launch.dynamicSmemBytes = backward_smem;
      internal::throw_cuda(
          "backward-sanitizer",
          cudaLaunchKernelEx(
              &launch, local_p2p_backward_megakernel, backward, x_map, dy_map,
              dab_map, w1_map, w2_transpose_map, w1_transpose_map,
              canonical_dy_mn_map, canonical_x_mn_map, u_mn_map,
              dab_mn_map, dw1_store_map),
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
               backward_status.wgmma_instructions != 0ULL &&
               backward_status.tma_loads != 0ULL;
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

} // namespace megamoe::cuda
