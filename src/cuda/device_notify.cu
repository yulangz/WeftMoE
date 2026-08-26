#include "cuda/device_notify_internal.hpp"
#include "cuda/activation_transport_internal.hpp"

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "cuda/cuda_internal.hpp"
#include "cuda/device_notify_bridge.hpp"
#include "megamoe/api/errors.hpp"

namespace megamoe::cuda {

namespace {
std::atomic<std::uint64_t> g_cuda_device_allocations{0};
std::atomic<std::uint64_t> g_cuda_device_frees{0};
std::atomic<std::uint64_t> g_cuda_pinned_allocations{0};
std::atomic<std::uint64_t> g_cuda_pinned_frees{0};
std::atomic<std::uint64_t> g_cuda_event_creates{0};
std::atomic<std::uint64_t> g_cuda_event_destroys{0};
std::atomic<std::uint64_t> g_cuda_device_synchronizes{0};
std::atomic<std::uint64_t> g_cuda_h2d_bytes{0};
std::atomic<std::uint64_t> g_cuda_d2h_bytes{0};
std::atomic<std::uint64_t> g_cuda_d2d_bytes{0};
#if MEGAMOE_BUILD_TESTING
std::atomic<std::uint8_t> g_consumer_fault{0};

bool take_consumer_fault(testing::ConsumerFaultPoint point) noexcept {
  auto expected = static_cast<std::uint8_t>(point);
  return g_consumer_fault.compare_exchange_strong(
      expected,
      static_cast<std::uint8_t>(testing::ConsumerFaultPoint::None));
}
#else
template <typename Point>
constexpr bool take_consumer_fault(Point) noexcept {
  return false;
}
#endif
} // namespace

#if MEGAMOE_BUILD_TESTING
void testing::inject_next_consumer_fault(ConsumerFaultPoint point) noexcept {
  g_consumer_fault.store(static_cast<std::uint8_t>(point));
}
#endif

namespace internal {
void record_cuda_api(CudaApiAuditKind kind, std::uint64_t bytes) noexcept {
  switch (kind) {
  case CudaApiAuditKind::DeviceAllocate: ++g_cuda_device_allocations; break;
  case CudaApiAuditKind::DeviceFree: ++g_cuda_device_frees; break;
  case CudaApiAuditKind::PinnedAllocate: ++g_cuda_pinned_allocations; break;
  case CudaApiAuditKind::PinnedFree: ++g_cuda_pinned_frees; break;
  case CudaApiAuditKind::EventCreate: ++g_cuda_event_creates; break;
  case CudaApiAuditKind::EventDestroy: ++g_cuda_event_destroys; break;
  case CudaApiAuditKind::DeviceSynchronize:
    ++g_cuda_device_synchronizes;
    break;
  case CudaApiAuditKind::HostToDevice: g_cuda_h2d_bytes += bytes; break;
  case CudaApiAuditKind::DeviceToHost: g_cuda_d2h_bytes += bytes; break;
  case CudaApiAuditKind::DeviceToDevice: g_cuda_d2d_bytes += bytes; break;
  }
}
} // namespace internal

#if MEGAMOE_BUILD_TESTING
testing::CudaApiAuditSnapshot testing::cuda_api_audit_snapshot() noexcept {
  return {g_cuda_device_allocations.load(), g_cuda_device_frees.load(),
          g_cuda_pinned_allocations.load(), g_cuda_pinned_frees.load(),
          g_cuda_event_creates.load(), g_cuda_event_destroys.load(),
          g_cuda_device_synchronizes.load(), g_cuda_h2d_bytes.load(),
          g_cuda_d2h_bytes.load(), g_cuda_d2d_bytes.load()};
}
#endif

namespace {

using internal::CudaAllocation;
using internal::CudaBufferState;
using internal::CudaExecutionState;

constexpr std::size_t kAlignment = 256;

[[noreturn]] void invalid(const std::string &message) {
  throw NotifyDispatchError(NotifyErrorCode::CudaFailure, message);
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs,
                          const char *field) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    invalid(std::string("phase=cuda-workspace-query, field=") + field +
            ", rank=n/a, expected=uint64 product, actual=overflow");
  }
  return lhs * rhs;
}

std::size_t checked_size(std::uint64_t value, const char *field) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    invalid(std::string("phase=cuda-workspace-query, field=") + field +
            ", rank=n/a, expected=within size_t, actual=" +
            std::to_string(value));
  }
  return static_cast<std::size_t>(value);
}

void validate_external_cuda_range(const void *pointer,
                                  std::uint64_t required_elements,
                                  std::uint64_t capacity_elements,
                                  std::size_t element_bytes,
                                  std::uint32_t expected_device,
                                  const char *field) {
  if (required_elements > capacity_elements)
    invalid(std::string("phase=external-cuda-producer, field=") + field +
            ", rank=n/a, expected=capacity>=" +
            std::to_string(required_elements) + ", actual=" +
            std::to_string(capacity_elements));
  const auto required_bytes = checked_size(
      checked_mul(required_elements, element_bytes, field), field);
  const auto capacity_bytes = checked_size(
      checked_mul(capacity_elements, element_bytes, field), field);
  if (required_bytes == 0)
    return;
  if (!pointer)
    invalid(std::string("phase=external-cuda-producer, field=") + field +
            ", rank=n/a, expected=non-null device allocation, actual=null");
  cudaPointerAttributes attributes{};
  internal::throw_cuda("external-cuda-producer",
                       cudaPointerGetAttributes(&attributes, pointer), 0, 1,
                       "external", 1, ExchangeKind::ExpertCounts,
                       "cudaPointerGetAttributes");
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != static_cast<int>(expected_device))
    invalid(std::string("phase=external-cuda-producer, field=") + field +
            ", rank=n/a, expected=device allocation on ordinal=" +
            std::to_string(expected_device) + ", actual=wrong memory/device");
  CUdeviceptr base = 0;
  std::size_t allocation_bytes = 0;
  if (cuMemGetAddressRange(&base, &allocation_bytes,
                           reinterpret_cast<CUdeviceptr>(pointer)) !=
      CUDA_SUCCESS)
    invalid(std::string("phase=external-cuda-producer, field=") + field +
            ", rank=n/a, expected=queryable CUDA allocation, actual=cuMemGetAddressRange failure");
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto allocation_begin = static_cast<std::uintptr_t>(base);
  if (begin < allocation_begin ||
      begin - allocation_begin > allocation_bytes ||
      capacity_bytes > allocation_bytes - (begin - allocation_begin))
    invalid(std::string("phase=external-cuda-producer, field=") + field +
            ", rank=n/a, expected=declared capacity within allocation, actual=short range");
}

struct ExternalTensorOwners {
  std::shared_ptr<const void> allocation;
  std::shared_ptr<const internal::ExternalCudaProducerState> producer;
};

std::size_t append_region(std::size_t &cursor, std::uint64_t elements,
                          std::size_t element_size, const char *field) {
  if (cursor > std::numeric_limits<std::size_t>::max() - (kAlignment - 1)) {
    invalid(std::string("phase=cuda-workspace-query, field=") + field +
            ", rank=n/a, expected=alignable workspace offset, actual=overflow");
  }
  const auto aligned = (cursor + kAlignment - 1) & ~(kAlignment - 1);
  if (elements > std::numeric_limits<std::size_t>::max() / element_size) {
    invalid(std::string("phase=cuda-workspace-query, field=") + field +
            ", rank=n/a, expected=byte size within size_t, actual=overflow");
  }
  const auto bytes = static_cast<std::size_t>(elements) * element_size;
  if (bytes > std::numeric_limits<std::size_t>::max() - aligned) {
    invalid(std::string("phase=cuda-workspace-query, field=") + field +
            ", rank=n/a, expected=workspace within size_t, actual=overflow");
  }
  cursor = aligned + bytes;
  return aligned;
}

struct DeviceStatus {
  unsigned long long out_of_range_packed;
  unsigned long long duplicate_packed;
};

enum class MaterializationError : unsigned long long {
  None = 0,
  ArithmeticOverflow = 1,
  CountRowMismatch = 2,
  SummaryRowMismatch = 3,
  ExpertCapacity = 4,
  WorkspaceCapacity = 5,
  OutputCapacity = 6,
  InvalidCollectiveValue = 7,
};

struct DeviceMaterializationStatus {
  unsigned long long error;
  unsigned long long field;
  unsigned long long expected;
  unsigned long long actual;
  unsigned long long owned_experts;
  unsigned long long incoming_rows;
  unsigned long long mtile_count;
  unsigned long long packet_count;
  unsigned long long fanout_count;
};

struct Layout {
  std::size_t keys_in = 0;
  std::size_t keys_out = 0;
  std::size_t indices_in = 0;
  std::size_t indices_out = 0;
  std::size_t sorted_experts = 0;
  std::size_t ones = 0;
  std::size_t inclusive = 0;
  std::size_t expert_count = 0;
  std::size_t packet_count = 0;
  std::size_t expert_id = 0;
  std::size_t local_offset = 0;
  std::size_t route_dst_row = 0;
  std::size_t route_dst_rank = 0;
  std::size_t route_dst_node = 0;
  std::size_t route_dst_local_rank = 0;
  std::size_t combine_order = 0;
  std::size_t masks = 0;
  std::size_t expert_to_rank = 0;
  std::size_t rank_to_node = 0;
  std::size_t rank_to_local = 0;
  std::size_t status = 0;
  std::size_t materialization_status = 0;
  std::size_t expert_total = 0;
  std::size_t expert_base = 0;
  std::size_t expert_capacity = 0;
  std::size_t workspace_capacity = 0;
  std::size_t owned_expert_ids = 0;
  std::size_t owned_expert_count = 0;
  std::size_t owned_expert_prefix = 0;
  std::size_t mtile_offsets = 0;
  std::size_t mtile_rows = 0;
  std::size_t incoming_rows = 0;
  std::size_t reverse_routes = 0;
  std::size_t packet_headers = 0;
  std::size_t packet_fanout_offsets = 0;
  std::size_t packet_fanout = 0;
  std::size_t outgoing_signals = 0;
  std::size_t incoming_signals = 0;
  std::size_t temp = 0;
  std::size_t total = 0;
};

Layout make_device_layout(const CudaNotifyWorkspacePlan &plan) {
  Layout layout;
  std::size_t cursor = 0;
  const auto routes = std::max<std::uint64_t>(plan.route_count, 1);
  layout.keys_in = append_region(cursor, routes, sizeof(std::uint64_t), "keys_in");
  layout.keys_out = append_region(cursor, routes, sizeof(std::uint64_t), "keys_out");
  layout.indices_in = append_region(cursor, routes, sizeof(std::uint64_t), "indices_in");
  layout.indices_out = append_region(cursor, routes, sizeof(std::uint64_t), "indices_out");
  layout.sorted_experts = append_region(cursor, routes, sizeof(std::uint32_t), "sorted_experts");
  layout.ones = append_region(cursor, routes, sizeof(std::uint64_t), "ones");
  layout.inclusive = append_region(cursor, routes, sizeof(std::uint64_t), "inclusive");
  layout.expert_count = append_region(cursor, plan.num_experts, sizeof(std::uint64_t), "expert_count");
  layout.packet_count = append_region(cursor, plan.num_nodes, sizeof(std::uint64_t), "packet_count");
  layout.expert_id = append_region(cursor, routes, sizeof(std::uint32_t), "expert_id");
  layout.local_offset = append_region(cursor, routes, sizeof(std::uint64_t), "local_offset");
  layout.route_dst_row = append_region(cursor, routes, sizeof(std::uint64_t), "route_dst_row");
  layout.route_dst_rank = append_region(cursor, routes, sizeof(std::uint32_t), "route_dst_rank");
  layout.route_dst_node = append_region(cursor, routes, sizeof(std::uint32_t), "route_dst_node");
  layout.route_dst_local_rank = append_region(cursor, routes, sizeof(std::uint32_t), "route_dst_local_rank");
  layout.combine_order = append_region(cursor, routes, sizeof(std::uint32_t), "combine_order");
  layout.masks = append_region(cursor, checked_mul(plan.num_tokens, plan.node_mask_words, "mask_elements"), sizeof(std::uint64_t), "masks");
  layout.expert_to_rank = append_region(cursor, plan.num_experts, sizeof(std::uint32_t), "expert_to_rank");
  layout.rank_to_node = append_region(cursor, plan.world_size, sizeof(std::uint32_t), "rank_to_node");
  layout.rank_to_local = append_region(cursor, plan.world_size, sizeof(std::uint32_t), "rank_to_local");
  layout.status = append_region(cursor, 1, sizeof(DeviceStatus), "status");
  layout.materialization_status = append_region(
      cursor, 1, sizeof(DeviceMaterializationStatus),
      "materialization_status");
  layout.expert_total = append_region(cursor, plan.num_experts,
                                      sizeof(std::uint64_t), "expert_total");
  layout.expert_base = append_region(cursor, plan.num_experts,
                                     sizeof(std::uint64_t), "expert_base");
  layout.expert_capacity = append_region(cursor, plan.num_experts,
                                         sizeof(std::uint64_t),
                                         "expert_capacity");
  layout.workspace_capacity = append_region(cursor, plan.world_size,
                                             sizeof(std::uint64_t),
                                             "workspace_capacity");
  layout.owned_expert_ids = append_region(
      cursor, plan.num_experts, sizeof(std::uint32_t), "owned_expert_ids");
  layout.owned_expert_count = append_region(
      cursor, plan.num_experts, sizeof(std::uint64_t), "owned_expert_count");
  layout.owned_expert_prefix = append_region(
      cursor, static_cast<std::uint64_t>(plan.num_experts) + 1,
      sizeof(std::uint64_t), "owned_expert_prefix");
  layout.mtile_offsets = append_region(
      cursor, static_cast<std::uint64_t>(plan.num_experts) + 1,
      sizeof(std::uint64_t), "mtile_offsets");
  layout.mtile_rows = append_region(cursor, plan.max_incoming_rows,
                                    sizeof(std::uint32_t), "mtile_rows");
  layout.incoming_rows = append_region(cursor, plan.max_incoming_rows,
                                       sizeof(DeviceIncomingRowRecord),
                                       "incoming_rows");
  layout.reverse_routes = append_region(cursor, plan.route_count,
                                        sizeof(DeviceReverseRouteRecord),
                                        "reverse_routes");
  layout.packet_headers = append_region(cursor, plan.route_count,
                                        sizeof(DevicePacketHeaderRecord),
                                        "packet_headers");
  layout.packet_fanout_offsets = append_region(
      cursor, plan.route_count + 1, sizeof(std::uint64_t),
      "packet_fanout_offsets");
  layout.packet_fanout = append_region(cursor, plan.route_count,
                                       sizeof(DeviceFanoutRecord),
                                       "packet_fanout");
  layout.outgoing_signals = append_region(cursor, plan.num_nodes,
                                          sizeof(DeviceSignalRecord),
                                          "outgoing_signals");
  layout.incoming_signals = append_region(cursor, plan.num_nodes,
                                          sizeof(DeviceSignalRecord),
                                          "incoming_signals");
  layout.temp = append_region(cursor, std::max(plan.sort_temp_bytes, plan.scan_temp_bytes), 1, "cub_temp");
  layout.total = cursor;
  return layout;
}

struct HostLayout {
  std::size_t status = 0;
  std::size_t materialization_status = 0;
  std::size_t materialization_ready = 0;
  std::size_t total = 0;
};

HostLayout make_host_layout(const CudaNotifyWorkspacePlan &plan) {
  (void)plan;
  HostLayout layout;
  std::size_t cursor = 0;
  layout.status = append_region(cursor, 1, sizeof(DeviceStatus), "host_status");
  layout.materialization_status = append_region(
      cursor, 1, sizeof(DeviceMaterializationStatus),
      "host_materialization_status");
  layout.materialization_ready = append_region(
      cursor, 1, sizeof(std::uint32_t), "host_materialization_ready");
  layout.total = cursor;
  return layout;
}

template <typename T> T *at(void *base, std::size_t offset) {
  return reinterpret_cast<T *>(static_cast<unsigned char *>(base) + offset);
}

template <typename T> const T *at(const void *base, std::size_t offset) {
  return reinterpret_cast<const T *>(static_cast<const unsigned char *>(base) + offset);
}

__global__ void prepare_routes_kernel(
    const std::int32_t *selected, std::uint64_t routes,
    std::uint64_t route_count_key_stride, std::uint32_t num_experts,
    const std::uint32_t *expert_to_rank, const std::uint32_t *rank_to_node,
    const std::uint32_t *rank_to_local, std::uint64_t *keys,
    std::uint64_t *indices, std::uint32_t *expert_id,
    std::uint32_t *dst_rank, std::uint32_t *dst_node,
    std::uint32_t *dst_local, std::uint64_t *ones, DeviceStatus *status) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (route >= routes)
    return;
  const auto raw = selected[route];
  indices[route] = route;
  ones[route] = 1;
  if (raw < 0 || static_cast<std::uint32_t>(raw) >= num_experts) {
    const auto packed =
        (static_cast<unsigned long long>(route) << 32U) |
        static_cast<std::uint32_t>(raw);
    atomicMin(&status->out_of_range_packed, packed);
    keys[route] = ~std::uint64_t{0};
    expert_id[route] = num_experts;
    dst_rank[route] = 0;
    dst_node[route] = 0;
    dst_local[route] = 0;
    return;
  }
  const auto expert = static_cast<std::uint32_t>(raw);
  const auto destination = expert_to_rank[expert];
  keys[route] = static_cast<std::uint64_t>(expert) * route_count_key_stride +
                route;
  expert_id[route] = expert;
  dst_rank[route] = destination;
  dst_node[route] = rank_to_node[destination];
  dst_local[route] = rank_to_local[destination];
}

__global__ void validate_and_describe_tokens_kernel(
    const std::int32_t *selected, std::uint64_t num_tokens,
    std::uint32_t top_k, std::uint32_t num_experts,
    const std::uint32_t *expert_to_rank, const std::uint32_t *rank_to_node,
    std::uint32_t num_nodes, std::uint32_t mask_words,
    std::uint32_t *combine_order, std::uint64_t *masks,
    std::uint64_t *packet_count, DeviceStatus *status) {
  const auto token = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (token >= num_tokens)
    return;
  const auto base = token * top_k;
  for (std::uint32_t word = 0; word < mask_words; ++word)
    masks[token * mask_words + word] = 0;
  for (std::uint32_t slot = 0; slot < top_k; ++slot) {
    const auto raw = selected[base + slot];
    if (raw < 0 || static_cast<std::uint32_t>(raw) >= num_experts)
      continue;
    for (std::uint32_t prior = 0; prior < slot; ++prior) {
      if (selected[base + prior] == raw) {
        const auto flat = base + slot;
        const auto packed =
            (static_cast<unsigned long long>(flat) << 32U) |
            static_cast<std::uint32_t>(raw);
        atomicMin(&status->duplicate_packed, packed);
      }
    }
    const auto destination = expert_to_rank[static_cast<std::uint32_t>(raw)];
    const auto node = rank_to_node[destination];
    masks[token * mask_words + node / 64U] |=
        std::uint64_t{1} << (node % 64U);
  }
  for (std::uint32_t position = 0; position < top_k; ++position) {
    std::uint32_t best = top_k;
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      bool used = false;
      for (std::uint32_t prior = 0; prior < position; ++prior)
        used = used || combine_order[base + prior] == slot;
      if (used)
        continue;
      if (best == top_k || selected[base + slot] < selected[base + best] ||
          (selected[base + slot] == selected[base + best] && slot < best))
        best = slot;
    }
    combine_order[base + position] = best;
  }
  for (std::uint32_t node = 0; node < num_nodes; ++node) {
    if ((masks[token * mask_words + node / 64U] >> (node % 64U)) & 1U)
      atomicAdd(reinterpret_cast<unsigned long long *>(&packet_count[node]),
                1ULL);
  }
}

__global__ void extract_sorted_experts_kernel(const std::uint64_t *keys,
                                               std::uint32_t *experts,
                                               std::uint64_t routes,
                                               std::uint64_t stride) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < routes)
    experts[index] = static_cast<std::uint32_t>(keys[index] / stride);
}

__global__ void histogram_and_scatter_kernel(
    const std::uint32_t *sorted_experts, const std::uint64_t *sorted_indices,
    const std::uint64_t *inclusive, std::uint64_t routes,
    std::uint32_t num_experts, std::uint64_t *histogram,
    std::uint64_t *local_offset) {
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index >= routes)
    return;
  const auto expert = sorted_experts[index];
  if (expert >= num_experts)
    return;
  local_offset[sorted_indices[index]] = inclusive[index] - 1;
  if (index + 1 == routes || sorted_experts[index + 1] != expert)
    histogram[expert] = inclusive[index];
}

struct DeviceMaterializeArgs {
  std::uint32_t rank;
  std::uint32_t world;
  std::uint32_t experts;
  std::uint32_t nodes;
  std::uint32_t top_k;
  std::uint32_t mask_words;
  std::uint32_t m_tile;
  std::uint64_t tokens;
  std::uint64_t routes;
  std::uint64_t max_tokens;
  std::uint64_t max_incoming_rows;
  bool has_expert_capacity;
  bool has_workspace_capacity;
  const std::uint64_t *count_matrix;
  const std::uint64_t *summary_matrix;
  const std::uint64_t *local_histogram;
  const std::uint64_t *local_summary;
  const std::uint32_t *expert_to_rank;
  const std::uint32_t *rank_to_node;
  const std::uint32_t *rank_to_local;
  const std::uint64_t *expert_capacity;
  const std::uint64_t *workspace_capacity;
  const std::uint32_t *expert_id;
  const std::uint64_t *local_offset;
  const std::uint32_t *route_dst_rank;
  const std::uint32_t *route_dst_node;
  const std::uint32_t *route_dst_local;
  const std::uint32_t *combine_order;
  const std::uint64_t *masks;
  std::uint64_t *expert_total;
  std::uint64_t *expert_base;
  std::uint64_t *route_dst_row;
  std::uint32_t *owned_expert_ids;
  std::uint64_t *owned_expert_count;
  std::uint64_t *owned_expert_prefix;
  std::uint64_t *mtile_offsets;
  std::uint32_t *mtile_rows;
  DeviceIncomingRowRecord *incoming_rows;
  DeviceReverseRouteRecord *reverse_routes;
  DevicePacketHeaderRecord *packet_headers;
  std::uint64_t *packet_fanout_offsets;
  DeviceFanoutRecord *packet_fanout;
  DeviceSignalRecord *outgoing_signals;
  DeviceSignalRecord *incoming_signals;
  DeviceMaterializationStatus *status;
};

__device__ bool device_add(std::uint64_t lhs, std::uint64_t rhs,
                           std::uint64_t &out) {
  if (rhs > ~std::uint64_t{0} - lhs)
    return false;
  out = lhs + rhs;
  return true;
}

__device__ void materialization_fail(DeviceMaterializationStatus *status,
                                     MaterializationError error,
                                     std::uint64_t field,
                                     std::uint64_t expected,
                                     std::uint64_t actual) {
  status->error = static_cast<unsigned long long>(error);
  status->field = field;
  status->expected = expected;
  status->actual = actual;
}

__device__ std::uint32_t rank_at_coordinate(const DeviceMaterializeArgs &a,
                                            std::uint32_t node,
                                            std::uint32_t local_rank) {
  for (std::uint32_t rank = 0; rank < a.world; ++rank) {
    if (a.rank_to_node[rank] == node &&
        a.rank_to_local[rank] == local_rank)
      return rank;
  }
  return a.world;
}

// Small checked planning phase. It establishes all variable output sizes and
// canonical prefixes before any parallel writer uses them as indices.
[[maybe_unused]] __global__ void
materialize_prefix_plan_kernel(DeviceMaterializeArgs a) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *a.status = DeviceMaterializationStatus{};
  std::uint64_t max_routes_per_rank = 0;
  if (a.top_k != 0 && a.max_tokens > ~std::uint64_t{0} / a.top_k) {
    materialization_fail(a.status, MaterializationError::ArithmeticOverflow, 1,
                         ~std::uint64_t{0}, a.max_tokens);
    return;
  }
  max_routes_per_rank = a.max_tokens * a.top_k;

  for (std::uint32_t source = 0; source < a.world; ++source) {
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      const auto value = a.count_matrix[
          static_cast<std::uint64_t>(source) * a.experts + expert];
      if (value > max_routes_per_rank) {
        materialization_fail(a.status,
                             MaterializationError::InvalidCollectiveValue, 2,
                             max_routes_per_rank, value);
        return;
      }
      if (source == a.rank && value != a.local_histogram[expert]) {
        materialization_fail(a.status,
                             MaterializationError::CountRowMismatch, expert,
                             a.local_histogram[expert], value);
        return;
      }
    }
    for (std::uint32_t node = 0; node < a.nodes; ++node) {
      const auto value = a.summary_matrix[
          static_cast<std::uint64_t>(source) * a.nodes + node];
      if (value > a.max_tokens) {
        materialization_fail(a.status,
                             MaterializationError::InvalidCollectiveValue, 3,
                             a.max_tokens, value);
        return;
      }
      if (source == a.rank && value != a.local_summary[node]) {
        materialization_fail(a.status,
                             MaterializationError::SummaryRowMismatch, node,
                             a.local_summary[node], value);
        return;
      }
    }
  }

  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    std::uint64_t total = 0;
    for (std::uint32_t source = 0; source < a.world; ++source) {
      const auto value = a.count_matrix[
          static_cast<std::uint64_t>(source) * a.experts + expert];
      if (!device_add(total, value, total)) {
        materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                             4, ~std::uint64_t{0}, value);
        return;
      }
    }
    a.expert_total[expert] = total;
    if (a.has_expert_capacity && total > a.expert_capacity[expert]) {
      materialization_fail(a.status, MaterializationError::ExpertCapacity,
                           expert, a.expert_capacity[expert], total);
      return;
    }
  }

  for (std::uint32_t destination = 0; destination < a.world; ++destination) {
    std::uint64_t prefix = 0;
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      if (a.expert_to_rank[expert] != destination)
        continue;
      a.expert_base[expert] = prefix;
      if (!device_add(prefix, a.expert_total[expert], prefix)) {
        materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                             5, ~std::uint64_t{0}, a.expert_total[expert]);
        return;
      }
    }
    if (a.has_workspace_capacity &&
        prefix > a.workspace_capacity[destination]) {
      materialization_fail(a.status, MaterializationError::WorkspaceCapacity,
                           destination, a.workspace_capacity[destination],
                           prefix);
      return;
    }
    if (destination == a.rank && prefix > a.max_incoming_rows) {
      materialization_fail(a.status, MaterializationError::OutputCapacity, 6,
                           a.max_incoming_rows, prefix);
      return;
    }
  }

  std::uint64_t owned = 0;
  std::uint64_t incoming = 0;
  std::uint64_t tiles = 0;
  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    ++owned;
    if (!device_add(incoming, a.expert_total[expert], incoming)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           7, ~std::uint64_t{0}, a.expert_total[expert]);
      return;
    }
    const auto count = a.expert_total[expert];
    const auto expert_tiles = count / a.m_tile + (count % a.m_tile != 0);
    if (!device_add(tiles, expert_tiles, tiles)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           8, ~std::uint64_t{0}, expert_tiles);
      return;
    }
  }
  std::uint64_t packets = 0;
  for (std::uint32_t node = 0; node < a.nodes; ++node) {
    if (!device_add(packets, a.local_summary[node], packets)) {
      materialization_fail(a.status, MaterializationError::ArithmeticOverflow,
                           9, ~std::uint64_t{0}, a.local_summary[node]);
      return;
    }
  }
  if (owned > a.experts || incoming > a.max_incoming_rows ||
      tiles > a.max_incoming_rows || packets > a.routes) {
    materialization_fail(a.status, MaterializationError::OutputCapacity, 10,
                         a.max_incoming_rows, incoming);
    return;
  }
  a.status->owned_experts = owned;
  a.status->incoming_rows = incoming;
  a.status->mtile_count = tiles;
  a.status->packet_count = packets;
  a.status->fanout_count = a.routes;

  std::uint64_t owned_index = 0;
  std::uint64_t incoming_prefix = 0;
  std::uint64_t tile_index = 0;
  a.owned_expert_prefix[0] = 0;
  a.mtile_offsets[0] = 0;
  for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    a.owned_expert_ids[owned_index] = expert;
    a.owned_expert_count[owned_index] = a.expert_total[expert];
    incoming_prefix += a.expert_total[expert];
    a.owned_expert_prefix[owned_index + 1] = incoming_prefix;
    const auto count = a.expert_total[expert];
    tile_index += count / a.m_tile + (count % a.m_tile != 0);
    a.mtile_offsets[owned_index + 1] = tile_index;
    ++owned_index;
  }

}

// Packet order is node-major and then token-major. One block owns a complete
// destination node, so a deterministic block scan can compact token flags and
// fan-out counts without atomics or a serial route walk. The per-node bases are
// derived from immutable histogram/summary rows and therefore do not depend on
// block scheduling.
__global__ void materialize_packet_plan_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockScan = cub::BlockScan<unsigned long long, kThreads>;
  __shared__ typename BlockScan::TempStorage scan_storage;
  __shared__ unsigned long long packet_carry;
  __shared__ unsigned long long fanout_carry;
  __shared__ unsigned long long chunk_packet_base;
  __shared__ unsigned long long chunk_fanout_base;
  __shared__ unsigned long long packet_base;
  __shared__ unsigned long long fanout_base;
  __shared__ std::uint32_t proxy_rank;

  const auto node = static_cast<std::uint32_t>(blockIdx.x);
  if (node >= a.nodes || a.status->error != 0)
    return;
  if (threadIdx.x == 0) {
    packet_base = 0;
    fanout_base = 0;
    packet_carry = 0;
    fanout_carry = 0;
    for (std::uint32_t prior = 0; prior < node; ++prior)
      packet_base += a.local_summary[prior];
    for (std::uint32_t expert = 0; expert < a.experts; ++expert) {
      const auto destination = a.expert_to_rank[expert];
      if (a.rank_to_node[destination] < node)
        fanout_base += a.local_histogram[expert];
    }
    proxy_rank = rank_at_coordinate(a, node, a.rank_to_local[a.rank]);
  }
  __syncthreads();

  for (std::uint64_t chunk = 0; chunk < a.tokens; chunk += kThreads) {
    const auto token = chunk + threadIdx.x;
    unsigned long long present = 0;
    unsigned long long fanouts = 0;
    if (token < a.tokens) {
      present = static_cast<unsigned long long>(
          (a.masks[token * a.mask_words + node / 64U] >> (node % 64U)) & 1U);
      const auto route_base = token * a.top_k;
      for (std::uint32_t slot = 0; slot < a.top_k; ++slot)
        fanouts += a.route_dst_node[route_base + slot] == node ? 1ULL : 0ULL;
    }

    unsigned long long packet_prefix = 0;
    unsigned long long packet_aggregate = 0;
    BlockScan(scan_storage).ExclusiveSum(present, packet_prefix,
                                         packet_aggregate);
    __syncthreads();
    if (threadIdx.x == 0) {
      chunk_packet_base = packet_carry;
      packet_carry += packet_aggregate;
    }
    __syncthreads();

    unsigned long long fanout_prefix = 0;
    unsigned long long fanout_aggregate = 0;
    BlockScan(scan_storage).ExclusiveSum(fanouts, fanout_prefix,
                                         fanout_aggregate);
    __syncthreads();
    if (threadIdx.x == 0) {
      chunk_fanout_base = fanout_carry;
      fanout_carry += fanout_aggregate;
    }
    __syncthreads();

    if (present != 0) {
      const auto packet = packet_base + chunk_packet_base + packet_prefix;
      const auto fanout = fanout_base + chunk_fanout_base + fanout_prefix;
      a.packet_headers[packet] = DevicePacketHeaderRecord{
          chunk_packet_base + packet_prefix, token, a.rank, node, proxy_rank,
          0};
      a.packet_fanout_offsets[packet] = fanout;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0 && node + 1U == a.nodes)
    a.packet_fanout_offsets[packet_base + packet_carry] =
        fanout_base + fanout_carry;
}

__device__ void materialization_fail_once(DeviceMaterializationStatus *status,
                                          MaterializationError error,
                                          std::uint64_t field,
                                          std::uint64_t expected,
                                          std::uint64_t actual) {
  if (atomicCAS(&status->error, 0ULL,
                static_cast<unsigned long long>(error)) == 0ULL) {
    status->field = field;
    status->expected = expected;
    status->actual = actual;
  }
}

[[maybe_unused]] __global__ void
materialize_plan_init_kernel(DeviceMaterializeArgs a) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *a.status = DeviceMaterializationStatus{};
    if (a.top_k == 0 || a.max_tokens > ~std::uint64_t{0} / a.top_k) {
      materialization_fail_once(a.status,
                                MaterializationError::ArithmeticOverflow, 1,
                                ~std::uint64_t{0}, a.max_tokens);
      return;
    }
    const auto max_routes = a.max_tokens * a.top_k;
    if (a.world != 0 && max_routes > ~std::uint64_t{0} / a.world)
      materialization_fail_once(a.status,
                                MaterializationError::ArithmeticOverflow, 2,
                                ~std::uint64_t{0}, max_routes);
  }
}

[[maybe_unused]] __global__ void
materialize_expert_totals_kernel(DeviceMaterializeArgs a) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (expert >= a.experts || a.status->error != 0)
    return;
  if (a.top_k == 0 || a.max_tokens > ~std::uint64_t{0} / a.top_k) {
    materialization_fail_once(a.status,
                              MaterializationError::ArithmeticOverflow, 1,
                              ~std::uint64_t{0}, a.max_tokens);
    return;
  }
  const auto max_routes = a.max_tokens * a.top_k;
  std::uint64_t total = 0;
  for (std::uint32_t source = 0; source < a.world; ++source) {
    const auto value = a.count_matrix[
        static_cast<std::uint64_t>(source) * a.experts + expert];
    if (value > max_routes || value > ~std::uint64_t{0} - total) {
      materialization_fail_once(
          a.status, value > max_routes
                        ? MaterializationError::InvalidCollectiveValue
                        : MaterializationError::ArithmeticOverflow,
          expert, max_routes, value);
      return;
    }
    if (source == a.rank && value != a.local_histogram[expert]) {
      materialization_fail_once(a.status,
                                MaterializationError::CountRowMismatch,
                                expert, a.local_histogram[expert], value);
      return;
    }
    total += value;
  }
  if (a.has_expert_capacity && total > a.expert_capacity[expert]) {
    materialization_fail_once(a.status, MaterializationError::ExpertCapacity,
                              expert, a.expert_capacity[expert], total);
    return;
  }
  a.expert_total[expert] = total;
}

[[maybe_unused]] __global__ void
materialize_expert_prefix_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (expert >= a.experts || a.status->error != 0)
    return;
  const auto owner = a.expert_to_rank[expert];
  unsigned long long destination_partial = 0;
  unsigned long long owned_prefix_partial = 0;
  unsigned long long tile_prefix_partial = 0;
  unsigned long long owned_index_partial = 0;
  for (std::uint32_t prior = threadIdx.x; prior < expert;
       prior += kThreads) {
    if (a.expert_to_rank[prior] == owner)
      destination_partial += a.expert_total[prior];
    if (a.expert_to_rank[prior] == a.rank) {
      ++owned_index_partial;
      owned_prefix_partial += a.expert_total[prior];
      const auto count = a.expert_total[prior];
      tile_prefix_partial += count / a.m_tile + (count % a.m_tile != 0);
    }
  }

  const auto destination_prefix =
      BlockReduce(reduce_storage).Sum(destination_partial);
  __syncthreads();
  const auto owned_prefix =
      BlockReduce(reduce_storage).Sum(owned_prefix_partial);
  __syncthreads();
  const auto tile_prefix = BlockReduce(reduce_storage).Sum(tile_prefix_partial);
  __syncthreads();
  const auto owned_index = BlockReduce(reduce_storage).Sum(owned_index_partial);
  __syncthreads();
  if (threadIdx.x == 0) {
    a.expert_base[expert] = destination_prefix;
    if (owner == a.rank) {
      a.owned_expert_ids[owned_index] = expert;
      a.owned_expert_count[owned_index] = a.expert_total[expert];
      a.owned_expert_prefix[owned_index] = owned_prefix;
      a.mtile_offsets[owned_index] = tile_prefix;
    }
  }
}

[[maybe_unused]] __global__ void materialize_destination_capacity_kernel(
    DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  const auto destination = static_cast<std::uint32_t>(blockIdx.x);
  if (destination >= a.world || a.status->error != 0)
    return;
  unsigned long long rows_partial = 0;
  for (std::uint32_t expert = threadIdx.x; expert < a.experts;
       expert += kThreads)
    if (a.expert_to_rank[expert] == destination)
      rows_partial += a.expert_total[expert];
  const auto rows = BlockReduce(reduce_storage).Sum(rows_partial);
  if (threadIdx.x == 0) {
    if (a.has_workspace_capacity && rows > a.workspace_capacity[destination])
      materialization_fail_once(
          a.status, MaterializationError::WorkspaceCapacity, destination,
          a.workspace_capacity[destination], rows);
    if (destination == a.rank && rows > a.max_incoming_rows)
      materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                                destination, a.max_incoming_rows, rows);
  }
}

[[maybe_unused]] __global__ void
materialize_plan_finalize_kernel(DeviceMaterializeArgs a) {
  constexpr std::uint32_t kThreads = 256;
  using BlockReduce = cub::BlockReduce<unsigned long long, kThreads>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  if (blockIdx.x != 0 || a.status->error != 0)
    return;
  unsigned long long owned_partial = 0;
  unsigned long long incoming_partial = 0;
  unsigned long long tiles_partial = 0;
  for (std::uint32_t expert = threadIdx.x; expert < a.experts;
       expert += kThreads) {
    if (a.expert_to_rank[expert] != a.rank)
      continue;
    ++owned_partial;
    incoming_partial += a.expert_total[expert];
    const auto count = a.expert_total[expert];
    tiles_partial += count / a.m_tile + (count % a.m_tile != 0);
  }
  unsigned long long packets_partial = 0;
  for (std::uint32_t node = threadIdx.x; node < a.nodes;
       node += kThreads) {
    const auto value = a.summary_matrix[
        static_cast<std::uint64_t>(a.rank) * a.nodes + node];
    if (value != a.local_summary[node] || value > a.max_tokens) {
      materialization_fail_once(a.status,
                                MaterializationError::SummaryRowMismatch,
                                node, a.local_summary[node], value);
    } else {
      packets_partial += value;
    }
  }
  const auto owned = BlockReduce(reduce_storage).Sum(owned_partial);
  __syncthreads();
  const auto incoming = BlockReduce(reduce_storage).Sum(incoming_partial);
  __syncthreads();
  const auto tiles = BlockReduce(reduce_storage).Sum(tiles_partial);
  __syncthreads();
  const auto packets = BlockReduce(reduce_storage).Sum(packets_partial);
  __syncthreads();
  if (threadIdx.x == 0 && a.status->error == 0) {
    if (incoming > a.max_incoming_rows || tiles > a.max_incoming_rows ||
        packets > a.routes) {
      materialization_fail_once(a.status, MaterializationError::OutputCapacity,
                                10, a.max_incoming_rows, incoming);
      return;
    }
    a.owned_expert_prefix[owned] = incoming;
    a.mtile_offsets[owned] = tiles;
    a.status->owned_experts = owned;
    a.status->incoming_rows = incoming;
    a.status->mtile_count = tiles;
    a.status->packet_count = packets;
    a.status->fanout_count = a.routes;
  }
}

__global__ void materialize_routes_kernel(DeviceMaterializeArgs a) {
  const auto route = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (a.status->error != 0 || route >= a.routes)
    return;
  const auto expert = a.expert_id[route];
  std::uint64_t source_prefix = 0;
  for (std::uint32_t source = 0; source < a.rank; ++source)
    source_prefix += a.count_matrix[
        static_cast<std::uint64_t>(source) * a.experts + expert];
  const auto row =
      a.expert_base[expert] + source_prefix + a.local_offset[route];
  a.route_dst_row[route] = row;
  a.reverse_routes[route] = DeviceReverseRouteRecord{
      route / a.top_k, row, a.rank,
      static_cast<std::uint32_t>(route % a.top_k), expert,
      a.route_dst_rank[route]};
}

__global__ void materialize_incoming_kernel(DeviceMaterializeArgs a) {
  const auto pair = static_cast<std::uint64_t>(blockIdx.x);
  const auto expert = static_cast<std::uint32_t>(pair / a.world);
  const auto source = static_cast<std::uint32_t>(pair % a.world);
  if (a.status->error != 0 || expert >= a.experts ||
      a.expert_to_rank[expert] != a.rank)
    return;
  std::uint64_t source_prefix = 0;
  for (std::uint32_t prior = 0; prior < source; ++prior)
    source_prefix += a.count_matrix[
        static_cast<std::uint64_t>(prior) * a.experts + expert];
  const auto count = a.count_matrix[
      static_cast<std::uint64_t>(source) * a.experts + expert];
  for (std::uint64_t offset = threadIdx.x; offset < count;
       offset += blockDim.x) {
    const auto row = a.expert_base[expert] + source_prefix + offset;
    a.incoming_rows[row] =
        DeviceIncomingRowRecord{offset, row, source, expert};
  }
}

__global__ void materialize_mtiles_kernel(DeviceMaterializeArgs a) {
  const auto expert = static_cast<std::uint32_t>(blockIdx.x);
  if (a.status->error != 0 || expert >= a.experts ||
      a.expert_to_rank[expert] != a.rank)
    return;
  std::uint64_t owned_index = 0;
  for (std::uint32_t prior = 0; prior < expert; ++prior)
    owned_index += a.expert_to_rank[prior] == a.rank;
  const auto count = a.expert_total[expert];
  const auto tiles = count / a.m_tile + (count % a.m_tile != 0);
  for (std::uint64_t tile = threadIdx.x; tile < tiles;
       tile += blockDim.x) {
    const auto first = tile * a.m_tile;
    const auto remaining = count - first;
    a.mtile_rows[a.mtile_offsets[owned_index] + tile] =
        static_cast<std::uint32_t>(remaining < a.m_tile ? remaining
                                                        : a.m_tile);
  }
}

__global__ void materialize_fanout_kernel(DeviceMaterializeArgs a) {
  const auto packet = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                      threadIdx.x;
  if (a.status->error != 0 || packet >= a.status->packet_count)
    return;
  const auto header = a.packet_headers[packet];
  auto output = a.packet_fanout_offsets[packet];
  const auto base = header.local_token_id * a.top_k;
  for (std::uint32_t position = 0; position < a.top_k; ++position) {
    const auto slot = a.combine_order[base + position];
    const auto route = base + slot;
    if (a.route_dst_node[route] != header.destination_node)
      continue;
    a.packet_fanout[output++] = DeviceFanoutRecord{
        a.route_dst_row[route], slot, a.route_dst_local[route],
        a.expert_id[route], 0};
  }
}

__global__ void materialize_signals_kernel(DeviceMaterializeArgs a) {
  const auto node = static_cast<std::uint32_t>(blockIdx.x) * blockDim.x +
                    threadIdx.x;
  if (a.status->error != 0 || node >= a.nodes)
    return;
  const auto source_node = a.rank_to_node[a.rank];
  const auto rail = a.rank_to_local[a.rank];
  const auto proxy = rank_at_coordinate(a, node, rail);
  a.outgoing_signals[node] = DeviceSignalRecord{
      a.local_summary[node], a.local_summary[node], source_node, node, a.rank,
      proxy, rail, 0};
  const auto source_rank = rank_at_coordinate(a, node, rail);
  const auto expected = a.summary_matrix[
      static_cast<std::uint64_t>(source_rank) * a.nodes + source_node];
  a.incoming_signals[node] = DeviceSignalRecord{
      expected, expected, node, source_node, source_rank, a.rank, rail, 0};
}

#if !MEGAMOE_DEBUG_VALIDATE
__global__ void publish_materialization_status_kernel(
    const DeviceMaterializationStatus *source,
    DeviceMaterializationStatus *mapped_status,
    std::uint32_t *mapped_ready) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  *mapped_status = *source;
  __threadfence_system();
  ::cuda::atomic_ref<std::uint32_t, ::cuda::thread_scope_system> ready(
      *mapped_ready);
  ready.store(1U, ::cuda::memory_order_release);
}
#endif

__global__ void dispatch_consumer_probe_delay_kernel(
    const std::uint64_t *source, std::uint64_t delay_cycles) {
  const auto start = clock64();
  while (clock64() - start < delay_cycles) {
  }
  if (source && threadIdx.x == 0 && blockIdx.x == 0)
    asm volatile("" : : "l"(*source));
}

__global__ void delayed_route_fill_kernel(std::int32_t *routes,
                                          std::uint64_t count,
                                          std::int32_t first_expert,
                                          std::uint64_t delay_cycles) {
  const auto start = clock64();
  while (clock64() - start < delay_cycles) {
  }
  const auto index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
  if (index < count)
    routes[index] = first_expert + static_cast<std::int32_t>(index);
}

std::string route_coordinates(std::uint64_t flat, std::uint32_t top_k) {
  return "token=" + std::to_string(flat / top_k) +
         ", slot=" + std::to_string(flat % top_k);
}

struct WorkspaceState {
  CudaNotifyWorkspacePlan plan;
  Layout device_layout;
  HostLayout host_layout;
  RankCollectiveContext context;
  CudaExecution execution;
  std::shared_ptr<CudaAllocation> allocation;
  std::shared_ptr<const void> slot_lease;
  std::uint64_t epoch = 0;
};

DeviceBufferLease make_lease(const std::shared_ptr<WorkspaceState> &state,
                             std::size_t offset, std::uint64_t width) {
  if (width == 0)
    return {};
  auto buffer = std::make_shared<CudaBufferState>();
  buffer->owner = state;
  buffer->pointer = at<void>(state->allocation->device_ptr, offset);
  buffer->device = state->execution.device_ordinal();
  return make_device_buffer_lease(1, width, width, std::move(buffer));
}

} // namespace

struct CudaNotifyPoolAllocation {
  CudaNotifyWorkspacePlan max_plan;
  std::shared_ptr<internal::CudaAllocation> route;
  std::shared_ptr<internal::CudaAllocation> workspace;
  ~CudaNotifyPoolAllocation() noexcept {
    if (!workspace || !workspace->dispatch_slot_quarantined.load(
                          std::memory_order_acquire))
      return;
    (void)cudaSetDevice(static_cast<int>(workspace->device));
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceSynchronize);
    if (cudaDeviceSynchronize() != cudaSuccess) {
      // A broken context cannot prove that consumer work ended. Fail closed by
      // abandoning native resources to the process instead of freeing storage
      // that a device may still reference.
      if (route)
        route->abandon_on_teardown = true;
      workspace->abandon_on_teardown = true;
    }
  }
};

namespace internal {

std::unique_ptr<BackendCountExchangeOperation>
CudaDeviceLeaseAccess::start(const CountExchangeRequest &) {
  invalid("phase=cuda-device-lease-access, field=start, rank=n/a, "
          "expected=never called, actual=called");
}

std::shared_ptr<const CudaBufferState>
CudaDeviceLeaseAccess::checked(const DeviceBufferLease &lease,
                               std::uint32_t expected_device) {
  if (!lease)
    invalid("phase=cuda-device-lease-access, field=lease, rank=n/a, "
            "expected=valid, actual=empty");
  auto state = std::dynamic_pointer_cast<const CudaBufferState>(
      backend_device_state(lease));
  if (!state || !state->pointer || state->device != expected_device)
    invalid("phase=cuda-device-lease-access, field=backend, rank=n/a, "
            "expected=CUDA lease on execution device, actual=foreign/null");
  return state;
}

CudaExecutionState::~CudaExecutionState() noexcept {
  if (!stream)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  (void)cudaStreamSynchronize(stream);
  (void)cudaStreamDestroy(stream);
}

CudaAllocation::~CudaAllocation() noexcept {
  if (abandon_on_teardown)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  if (completion)
    (void)cudaEventSynchronize(completion);
  if (mapping_start)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (mapping_start)
    (void)cudaEventDestroy(mapping_start);
  if (mapping_complete)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (mapping_complete)
    (void)cudaEventDestroy(mapping_complete);
  if (planning_start)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (planning_start)
    (void)cudaEventDestroy(planning_start);
  if (planning_complete)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (planning_complete)
    (void)cudaEventDestroy(planning_complete);
  if (producer_ready && owns_producer_event)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (producer_ready && owns_producer_event)
    (void)cudaEventDestroy(producer_ready);
  for (auto event : consumer_events) {
    if (event) {
      record_cuda_api(CudaApiAuditKind::EventDestroy);
      (void)cudaEventDestroy(event);
    }
  }
  if (completion)
    record_cuda_api(CudaApiAuditKind::EventDestroy);
  if (completion)
    (void)cudaEventDestroy(completion);
  if (host_ptr)
    record_cuda_api(CudaApiAuditKind::PinnedFree);
  if (host_ptr)
    (void)cudaFreeHost(host_ptr);
  if (device_ptr && owns_device_memory)
    record_cuda_api(CudaApiAuditKind::DeviceFree);
  if (device_ptr && owns_device_memory)
    (void)cudaFree(device_ptr);
}

void throw_cuda(const char *phase, cudaError_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation) {
  if (error == cudaSuccess)
    return;
  std::ostringstream stream;
  const char *kind_text = kind == ExchangeKind::Control
                              ? "control"
                              : (kind == ExchangeKind::ExpertCounts
                                     ? "expert_counts"
                                     : "notify_packet_summary");
  stream << "phase=" << phase << ", kind=" << kind_text
         << ", rank=" << rank << ", world=" << world << ", session='"
         << session << "', generation=" << generation << ", operation="
         << operation << ", cuda_error=" << static_cast<int>(error) << " ("
         << cudaGetErrorName(error) << ": " << cudaGetErrorString(error)
         << ")";
  throw NotifyDispatchError(NotifyErrorCode::CudaFailure, stream.str());
}

} // namespace internal

struct CudaExecution::Impl {
  std::shared_ptr<CudaExecutionState> state;
  ExecutionToken token;
};

namespace internal {
std::unique_ptr<BackendCountExchangeOperation>
CudaExecutionStateAccess::start(const CountExchangeRequest &) {
  invalid("phase=cuda-execution-state-access, field=start, rank=n/a, "
          "expected=never called, actual=called");
}

std::shared_ptr<const CudaExecutionState>
CudaExecutionStateAccess::checked(const CudaExecution &execution) {
  if (!execution)
    invalid("phase=cuda-context, field=execution, rank=n/a, "
            "expected=valid, actual=empty");
  auto state = std::dynamic_pointer_cast<const CudaExecutionState>(
      backend_execution_state(execution.token()));
  if (!state || !state->stream)
    invalid("phase=cuda-context, field=execution_backend, rank=n/a, "
            "expected=live CUDA execution, actual=foreign/null");
  return state;
}

std::shared_ptr<const CudaExecutionState>
checked_execution_state(const CudaExecution &execution) {
  return CudaExecutionStateAccess::checked(execution);
}
} // namespace internal

CudaExecution::CudaExecution(const CudaExecution &) = default;
CudaExecution::CudaExecution(CudaExecution &&) noexcept = default;
CudaExecution &CudaExecution::operator=(const CudaExecution &) = default;
CudaExecution &CudaExecution::operator=(CudaExecution &&) noexcept = default;
CudaExecution::~CudaExecution() = default;

CudaExecution make_cuda_execution(std::string identity,
                                  std::uint32_t device_ordinal) {
  int count = 0;
  internal::throw_cuda("execution-create", cudaGetDeviceCount(&count), 0, 1,
                       identity, 1, ExchangeKind::ExpertCounts,
                       "cudaGetDeviceCount");
  if (device_ordinal >= static_cast<std::uint32_t>(count))
    invalid("phase=execution-create, field=device_ordinal, rank=n/a, expected=[0," +
            std::to_string(count) + "), actual=" +
            std::to_string(device_ordinal));
  internal::throw_cuda("execution-create", cudaSetDevice(static_cast<int>(device_ordinal)),
                       0, 1, identity, 1, ExchangeKind::ExpertCounts,
                       "cudaSetDevice");
  auto state = std::make_shared<CudaExecutionState>();
  state->device = device_ordinal;
  internal::throw_cuda("execution-create",
                       cudaStreamCreateWithFlags(&state->stream,
                                                 cudaStreamNonBlocking),
                       0, 1, identity, 1, ExchangeKind::ExpertCounts,
                       "cudaStreamCreateWithFlags");
  auto impl = std::make_shared<CudaExecution::Impl>();
  impl->state = state;
  impl->token = make_execution_token(std::move(identity), state);
  return CudaExecution(std::move(impl));
}

const ExecutionToken &CudaExecution::token() const {
  if (!impl_)
    invalid("attempted to access empty CUDA execution");
  return impl_->token;
}

std::uint32_t CudaExecution::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA execution");
  return impl_->state->device;
}

struct SelectedExpertRouteLease<std::int32_t>::Impl {
  std::shared_ptr<CudaAllocation> allocation;
  std::shared_ptr<const void> slot_lease;
  const void *device_ptr = nullptr;
  CollectiveSession producer_session;
  ExecutionToken producer_execution;
  ExecutionToken consumer_execution;
  std::uint32_t layout = kSelectedExpertRouteLayoutVersion;
  std::uint32_t device = 0;
  std::uint64_t generation = 0;
  bool ready_recorded = false;
  std::uint64_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t routes = 0;
};

struct SelectedExpertGateLease<float>::Impl {
  std::shared_ptr<CudaAllocation> allocation;
  const float *device_ptr = nullptr;
  CudaRouteInput bound_routes;
  CollectiveSession producer_session;
  ExecutionToken producer_execution;
  ExecutionToken consumer_execution;
  std::uint32_t layout = kSelectedExpertGateLayoutVersion;
  std::uint32_t device = 0;
  std::uint64_t generation = 0;
  bool ready_recorded = false;
  std::uint64_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t routes = 0;
};

ExternalCudaProducer make_external_cuda_producer(
    RankCollectiveContext context, std::uint32_t device_ordinal,
    CUstream_st *producer_stream, std::shared_ptr<const void> stream_owner,
    std::string identity) {
  if (!context || !producer_stream || !stream_owner || identity.empty())
    invalid("phase=external-cuda-producer-create, field=context/stream/owner/identity, rank=n/a, expected=all valid, actual=empty");
  int device_count = 0;
  internal::throw_cuda("external-cuda-producer-create",
                       cudaGetDeviceCount(&device_count),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaGetDeviceCount");
  if (device_ordinal >= static_cast<std::uint32_t>(device_count))
    invalid("phase=external-cuda-producer-create, field=device, rank=" +
            std::to_string(context.calling_rank()) + ", expected=[0," +
            std::to_string(device_count) + "), actual=" +
            std::to_string(device_ordinal));
  auto backend = std::make_shared<internal::ExternalCudaProducerState>();
  backend->context = std::move(context);
  backend->device = device_ordinal;
  backend->stream = reinterpret_cast<cudaStream_t>(producer_stream);
  backend->stream_owner = std::move(stream_owner);
  backend->execution = make_execution_token(std::move(identity), backend);
  auto state = std::make_shared<ExternalCudaProducer::State>();
  state->backend = std::move(backend);
  return ExternalCudaProducer(std::move(state));
}

ExternalCudaProducer::ExternalCudaProducer(const ExternalCudaProducer &) =
    default;
ExternalCudaProducer::ExternalCudaProducer(ExternalCudaProducer &&) noexcept =
    default;
ExternalCudaProducer &ExternalCudaProducer::operator=(
    const ExternalCudaProducer &) = default;
ExternalCudaProducer &ExternalCudaProducer::operator=(
    ExternalCudaProducer &&) noexcept = default;
ExternalCudaProducer::~ExternalCudaProducer() = default;

const RankCollectiveContext &ExternalCudaProducer::context() const {
  if (!state_)
    invalid("phase=external-cuda-producer, field=handle, rank=n/a, expected=valid, actual=empty");
  return state_->backend->context;
}

std::uint32_t ExternalCudaProducer::device_ordinal() const {
  if (!state_)
    invalid("phase=external-cuda-producer, field=handle, rank=n/a, expected=valid, actual=empty");
  return state_->backend->device;
}

CudaRouteInput ExternalCudaProducer::publish_selected_experts(
    const std::int32_t *device_data, std::uint64_t capacity_elements,
    std::uint64_t num_tokens, std::uint32_t top_k,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || top_k == 0)
    invalid("phase=external-route-publish, field=producer/owner/top_k, rank=n/a, expected=valid/nonzero, actual=invalid");
  const auto routes = checked_mul(num_tokens, top_k, "external route count");
  validate_external_cuda_range(device_data, routes, capacity_elements,
                               sizeof(std::int32_t), state_->backend->device,
                               "selected_experts");
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state_->backend->device;
  allocation->device_ptr = const_cast<std::int32_t *>(device_data);
  allocation->device_bytes = checked_size(
      checked_mul(capacity_elements, sizeof(std::int32_t),
                  "external route capacity"),
      "external route capacity");
  allocation->owns_device_memory = false;
  allocation->external_owner = std::make_shared<ExternalTensorOwners>(
      ExternalTensorOwners{std::move(allocation_owner), state_->backend});
  internal::throw_cuda(
      "external-route-publish",
      cudaEventCreateWithFlags(&allocation->producer_ready,
                               cudaEventDisableTiming),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventCreate(producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "external-route-publish",
      cudaEventRecord(allocation->producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = device_data;
  impl->producer_session = state_->backend->context.session();
  impl->producer_execution = state_->backend->execution;
  impl->consumer_execution = state_->backend->execution;
  impl->device = state_->backend->device;
  impl->generation = state_->backend->context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = num_tokens;
  impl->top_k = top_k;
  impl->routes = routes;
  return CudaRouteInput(std::move(impl));
}

CudaGateInput ExternalCudaProducer::publish_gates(
    const float *device_data, std::uint64_t capacity_elements,
    const CudaRouteInput &exact_routes,
    std::shared_ptr<const void> allocation_owner) const {
  if (!state_ || !allocation_owner || !exact_routes.impl_)
    invalid("phase=external-gate-publish, field=producer/owner/routes, rank=n/a, expected=valid, actual=empty");
  const auto &route = *exact_routes.impl_;
  if (route.producer_session.stable_identity_bytes() !=
          state_->backend->context.session().stable_identity_bytes() ||
      route.generation != state_->backend->context.session().generation() ||
      route.device != state_->backend->device ||
      !route.producer_execution.same_instance(state_->backend->execution))
    invalid("phase=external-gate-publish, field=route_provenance, rank=" +
            std::to_string(state_->backend->context.calling_rank()) +
            ", expected=exact producer route lease, actual=mismatch");
  validate_external_cuda_range(device_data, route.routes, capacity_elements,
                               sizeof(float), state_->backend->device, "gates");
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state_->backend->device;
  allocation->device_ptr = const_cast<float *>(device_data);
  allocation->device_bytes = checked_size(
      checked_mul(capacity_elements, sizeof(float), "external gate capacity"),
      "external gate capacity");
  allocation->owns_device_memory = false;
  allocation->external_owner = std::make_shared<ExternalTensorOwners>(
      ExternalTensorOwners{std::move(allocation_owner), state_->backend});
  // Gate and route readiness share the immutable route provenance event. A
  // second record extends it to cover all producer work enqueued so far.
  internal::throw_cuda(
      "external-gate-publish",
      cudaEventRecord(route.allocation->producer_ready, state_->backend->stream),
      state_->backend->context.calling_rank(), state_->backend->context.session().world_size(),
      state_->backend->context.session().identity(),
      state_->backend->context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(route+gate ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = device_data;
  impl->bound_routes = exact_routes;
  impl->producer_session = state_->backend->context.session();
  impl->producer_execution = state_->backend->execution;
  impl->consumer_execution = state_->backend->execution;
  impl->device = state_->backend->device;
  impl->generation = state_->backend->context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = route.tokens;
  impl->top_k = route.top_k;
  impl->routes = route.routes;
  return CudaGateInput(std::move(impl));
}

CudaRouteInput upload_cuda_routes(const RankRoutingInput &routes,
                                  std::uint32_t top_k,
                                  const RankCollectiveContext &context,
                                  const CudaExecution &execution) {
  if (!context || !execution ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=cuda-route-upload, field=producer_context, rank=n/a, "
            "expected=matching session/execution, actual=mismatch/empty");
  const auto state = internal::checked_execution_state(execution);
  const auto count = checked_mul(routes.num_tokens, top_k, "route_count");
  if (routes.selected_experts.size() != checked_size(count, "route_count"))
    invalid("phase=cuda-route-upload, field=selected_experts.size, rank=n/a, expected=" +
            std::to_string(count) + ", actual=" +
            std::to_string(routes.selected_experts.size()));
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state->device;
  const auto route_bytes = checked_size(
      checked_mul(count, sizeof(std::int32_t), "route_input_bytes"),
      "route_input_bytes");
  allocation->device_bytes = std::max<std::size_t>(route_bytes, 1);
  internal::throw_cuda("route-upload", cudaSetDevice(static_cast<int>(state->device)),
                       0, 1, execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("route-upload",
                       cudaMalloc(&allocation->device_ptr,
                                  allocation->device_bytes),
                       0, 1, execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaMalloc(route input)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda("route-upload",
                       cudaEventCreateWithFlags(&allocation->producer_ready,
                                                cudaEventDisableTiming),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  if (count != 0) {
    internal::throw_cuda(
        "route-upload",
        cudaMemcpyAsync(allocation->device_ptr, routes.selected_experts.data(),
                        route_bytes,
                        cudaMemcpyHostToDevice, state->stream),
        0, 1, execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(route input)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              route_bytes);
  }
  internal::throw_cuda("route-upload",
                       cudaEventRecord(allocation->producer_ready,
                                       state->stream),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = impl->allocation->device_ptr;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = state->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens;
  impl->top_k = top_k;
  impl->routes = count;
  return CudaRouteInput(std::move(impl));
}

std::uint64_t CudaRouteInput::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->tokens;
}
std::uint32_t CudaRouteInput::top_k() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->top_k;
}
std::uint64_t CudaRouteInput::route_count() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->routes;
}
std::uint64_t CudaRouteInput::device_bytes() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return checked_mul(impl_->routes, sizeof(std::int32_t),
                     "route input bytes");
}

std::uint32_t CudaRouteInput::layout_version() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->layout;
}
std::uint32_t CudaRouteInput::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->device;
}
std::uint64_t CudaRouteInput::route_generation() const {
  if (!impl_)
    invalid("attempted to access empty CUDA route input");
  return impl_->generation;
}

std::uint64_t CudaGateInput::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->tokens;
}
std::uint32_t CudaGateInput::top_k() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->top_k;
}
std::uint64_t CudaGateInput::route_count() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->routes;
}
std::uint64_t CudaGateInput::device_bytes() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return checked_mul(impl_->routes, sizeof(float), "gate input bytes");
}
std::uint32_t CudaGateInput::layout_version() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->layout;
}
std::uint32_t CudaGateInput::device_ordinal() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->device;
}
std::uint64_t CudaGateInput::gate_generation() const {
  if (!impl_)
    invalid("attempted to access empty CUDA gate input");
  return impl_->generation;
}

CudaGateInput DeviceGateProvenanceBridge::upload_internal(
    const std::vector<float> &host, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  if (host.size() != checked_size(routes.route_count(), "gate route count"))
    invalid("phase=cuda-gate-upload, field=shape, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(routes.route_count()) + ", actual=" +
            std::to_string(host.size()));
  for (std::size_t index = 0; index < host.size(); ++index)
    if (!std::isfinite(host[index]) || host[index] < 0.0F)
      invalid("phase=cuda-gate-upload, field=value, rank=" +
              std::to_string(context.calling_rank()) + ", expected=finite>=0, actual=invalid, route=" +
              std::to_string(index));
  const auto state = internal::checked_execution_state(execution);
  auto allocation = std::make_shared<CudaAllocation>();
  allocation->device = state->device;
  const auto bytes = checked_size(
      checked_mul(routes.route_count(), sizeof(float), "gate input bytes"),
      "gate input bytes");
  allocation->device_bytes = std::max<std::size_t>(bytes, 1);
  internal::throw_cuda(
      "gate-upload", cudaSetDevice(static_cast<int>(state->device)),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "gate-upload", cudaMalloc(&allocation->device_ptr,
                                  allocation->device_bytes),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaMalloc(gate input)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  if (bytes != 0) {
    internal::throw_cuda(
        "gate-upload",
        cudaMemcpyAsync(allocation->device_ptr, host.data(), bytes,
                        cudaMemcpyHostToDevice, state->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemcpyAsync(gate input)");
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  }
  internal::throw_cuda(
      "gate-upload",
      cudaEventRecord(routes.impl_->allocation->producer_ready, state->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(gate producer ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->device_ptr = static_cast<const float *>(impl->allocation->device_ptr);
  impl->bound_routes = routes;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = state->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens();
  impl->top_k = routes.top_k();
  impl->routes = routes.route_count();
  return CudaGateInput(std::move(impl));
}

CudaGateInput testing::upload_fp32_gates_for_testing(
    const std::vector<float> &host, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  return DeviceGateProvenanceBridge::upload_internal(
      host, routes, context, execution);
}

CudaRouteInput import_selected_expert_routes(
    const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!routes.impl_ || !consumer_context || !consumer_execution)
    invalid("phase=route-stream-handoff, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &source = *routes.impl_;
  if (!consumer_context.execution().same_instance(consumer_execution.token()) ||
      consumer_context.calling_rank() >= consumer_context.session().world_size() ||
      source.producer_session.stable_identity_bytes() !=
          consumer_context.session().stable_identity_bytes() ||
      source.generation != consumer_context.session().generation())
    invalid("phase=route-stream-handoff, field=session/generation/execution, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=matching immutable producer provenance, actual=mismatch");
  const auto exec = internal::checked_execution_state(consumer_execution);
  if (exec->device != source.device || !source.ready_recorded ||
      !source.allocation || !source.allocation->producer_ready)
    invalid("phase=route-stream-handoff, field=device/ready_event, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=recorded producer event on matching device, actual=mismatch");
  internal::throw_cuda("route-stream-handoff",
                       cudaStreamWaitEvent(exec->stream,
                                           source.allocation->producer_ready),
                       consumer_context.calling_rank(),
                       consumer_context.session().world_size(),
                       consumer_context.session().identity(),
                       consumer_context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaStreamWaitEvent");
  auto impl = std::make_shared<CudaRouteInput::Impl>(source);
  impl->consumer_execution = consumer_execution.token();
  return CudaRouteInput(std::move(impl));
}

CudaGateInput import_selected_expert_gates(
    const CudaGateInput &gates, const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!gates.impl_ || !routes.impl_ || !consumer_context ||
      !consumer_execution)
    invalid("phase=gate-stream-handoff, field=input, rank=n/a, expected=valid, actual=empty");
  const auto &source = *gates.impl_;
  if (!consumer_context.execution().same_instance(consumer_execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          consumer_context.session().stable_identity_bytes() ||
      source.generation != consumer_context.session().generation() ||
      source.device != consumer_execution.device_ordinal() ||
      source.tokens != routes.num_tokens() || source.top_k != routes.top_k() ||
      !source.ready_recorded || !source.bound_routes.impl_ ||
      source.bound_routes.impl_->allocation != routes.impl_->allocation ||
      !routes.impl_->allocation || !routes.impl_->allocation->producer_ready)
    invalid("phase=gate-stream-handoff, field=provenance/shape/ready, rank=" +
            std::to_string(consumer_context.calling_rank()) +
            ", expected=matching exact routes/session/generation/device, actual=mismatch");
  const auto exec = internal::checked_execution_state(consumer_execution);
  internal::throw_cuda(
      "gate-stream-handoff",
      cudaStreamWaitEvent(exec->stream,
                          routes.impl_->allocation->producer_ready),
      consumer_context.calling_rank(), consumer_context.session().world_size(),
      consumer_context.session().identity(),
      consumer_context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(route+gate ready)");
  auto impl = std::make_shared<CudaGateInput::Impl>(source);
  impl->bound_routes = routes;
  impl->consumer_execution = consumer_execution.token();
  return CudaGateInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::reissue_for_testing(
    const CudaRouteInput &storage, const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!storage.impl_ || !context || !execution)
    invalid("phase=test-route-reissue, field=input, rank=n/a, "
            "expected=valid private storage/context/execution, actual=empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.allocation || !source.allocation->producer_ready)
    invalid("phase=test-route-reissue, field=provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=same immutable storage pipeline/execution/device, "
            "actual=mismatch");
  auto impl = std::make_shared<CudaRouteInput::Impl>(source);
  impl->producer_session = context.session();
  impl->generation = context.session().generation();
  return CudaRouteInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::schedule_delayed_for_testing(
    const CudaRouteInput &storage, std::int32_t first_expert,
    std::uint64_t delay_cycles, const RankCollectiveContext &context,
    const CudaExecution &producer_execution, bool record_ready) {
  auto exact = reissue_for_testing(storage, context, producer_execution);
  auto impl = std::make_shared<CudaRouteInput::Impl>(*exact.impl_);
  auto exec = internal::checked_execution_state(producer_execution);
  const auto bytes = checked_size(
      checked_mul(impl->routes, sizeof(std::int32_t), "test delayed routes"),
      "test delayed routes");
  if (bytes != 0) {
    internal::throw_cuda(
        "test-route-producer", cudaMemset(
            const_cast<void *>(impl->device_ptr), 0xff, bytes),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaMemset(route poison)");
    const auto blocks = static_cast<unsigned int>((impl->routes + 127U) / 128U);
    delayed_route_fill_kernel<<<blocks, 128, 0, exec->stream>>>(
        static_cast<std::int32_t *>(const_cast<void *>(impl->device_ptr)),
        impl->routes, first_expert, delay_cycles);
    internal::throw_cuda(
        "test-route-producer", cudaGetLastError(), context.calling_rank(),
        context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "delayed route kernel launch");
  }
  impl->ready_recorded = false;
  if (record_ready) {
    internal::throw_cuda(
        "test-route-producer",
        cudaEventRecord(impl->allocation->producer_ready, exec->stream),
        context.calling_rank(), context.session().world_size(),
        context.session().identity(), context.session().generation(),
        ExchangeKind::ExpertCounts, "cudaEventRecord(producer ready)");
    impl->ready_recorded = true;
  }
  return CudaRouteInput(std::move(impl));
}

CudaRouteInput DeviceRouteProvenanceBridge::unsafe_import_without_wait_for_testing(
    const CudaRouteInput &routes,
    const RankCollectiveContext &consumer_context,
    const CudaExecution &consumer_execution) {
  if (!routes.impl_ || !consumer_context || !consumer_execution)
    invalid("phase=test-route-import-no-wait, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  auto impl = std::make_shared<CudaRouteInput::Impl>(*routes.impl_);
  impl->consumer_execution = consumer_execution.token();
  return CudaRouteInput(std::move(impl));
}

void DeviceRouteProvenanceBridge::validate(
    const CudaRouteInput &routes, const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!routes.impl_ || !context || !execution)
    invalid("phase=device-route-provenance, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &route = *routes.impl_;
  const auto required_bytes = checked_mul(
      route.routes, sizeof(std::int32_t), "route provenance bytes");
  if (route.layout != kSelectedExpertRouteLayoutVersion ||
      route.device != execution.device_ordinal() ||
      !route.consumer_execution.same_instance(execution.token()) ||
      route.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      route.generation != context.session().generation() ||
      !route.ready_recorded || !route.allocation ||
      !route.allocation->producer_ready ||
      required_bytes > route.allocation->device_bytes)
    invalid("phase=device-route-provenance, field=dtype/layout/shape/device/session/generation/execution/ready/capacity, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact immutable producer provenance, actual=mismatch");
}

bool DeviceGateProvenanceBridge::same_route_storage(
    const CudaGateInput &gates, const CudaRouteInput &routes) {
  return gates.impl_ && routes.impl_ && gates.impl_->bound_routes.impl_ &&
         gates.impl_->bound_routes.impl_->allocation == routes.impl_->allocation;
}

void DeviceGateProvenanceBridge::validate(
    const CudaGateInput &gates, const CudaRouteInput &routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  if (!gates.impl_)
    invalid("phase=device-gate-provenance, field=input, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=valid FP32 gate lease, actual=empty");
  const auto &gate = *gates.impl_;
  const auto required_bytes = checked_mul(
      gate.routes, sizeof(float), "gate provenance bytes");
  if (gate.layout != kSelectedExpertGateLayoutVersion ||
      gate.device != execution.device_ordinal() ||
      !gate.consumer_execution.same_instance(execution.token()) ||
      gate.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      gate.generation != context.session().generation() ||
      gate.tokens != routes.num_tokens() || gate.top_k != routes.top_k() ||
      gate.routes != routes.route_count() || !gate.ready_recorded ||
      !gate.allocation || !gate.bound_routes.impl_ ||
      !gate.bound_routes.impl_->allocation ||
      !gate.bound_routes.impl_->allocation->producer_ready ||
      required_bytes > gate.allocation->device_bytes ||
      !same_route_storage(gates, routes))
    invalid("phase=device-gate-provenance, field=dtype/layout/shape/device/session/generation/execution/ready/capacity/route_binding, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact immutable route+gate provenance, actual=mismatch");
}

CudaGateInput DeviceGateProvenanceBridge::reissue_for_testing(
    const CudaGateInput &storage, const CudaRouteInput &exact_routes,
    const RankCollectiveContext &context,
    const CudaExecution &execution) {
  if (!storage.impl_ || !exact_routes || !context || !execution)
    invalid("phase=test-gate-reissue, field=input, rank=n/a, expected=valid, actual=empty");
  const auto &source = *storage.impl_;
  if (!context.execution().same_instance(execution.token()) ||
      !source.producer_execution.same_instance(execution.token()) ||
      source.producer_session.stable_identity_bytes() !=
          context.session().stable_identity_bytes() ||
      source.device != execution.device_ordinal() || !source.ready_recorded ||
      !source.allocation || !source.bound_routes.impl_ ||
      !source.bound_routes.impl_->allocation ||
      !source.bound_routes.impl_->allocation->producer_ready ||
      !same_route_storage(storage, exact_routes) ||
      source.tokens != exact_routes.num_tokens() ||
      source.top_k != exact_routes.top_k())
    invalid("phase=test-gate-reissue, field=provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=same immutable route storage/execution/device, actual=mismatch");
  auto impl = std::make_shared<CudaGateInput::Impl>(source);
  impl->bound_routes = exact_routes;
  impl->producer_session = context.session();
  impl->generation = context.session().generation();
  return CudaGateInput(std::move(impl));
}

const float *
DeviceGateProvenanceBridge::device_pointer(const CudaGateInput &gates) {
  if (!gates.impl_ || !gates.impl_->device_ptr)
    invalid("phase=device-gate-view, field=gate, rank=n/a, expected=valid, actual=empty");
  return gates.impl_->device_ptr;
}

CudaNotifyWorkspacePlan query_cuda_notify_workspace(
    const RuntimeConfig &config, std::uint32_t calling_rank,
    std::uint64_t num_tokens, std::uint64_t max_tokens_per_rank) {
  validate_runtime_config(config);
  if (calling_rank >= config.topology.world_size)
    invalid("phase=cuda-workspace-query, field=calling_rank, rank=" +
            std::to_string(calling_rank) + ", expected=[0," +
            std::to_string(config.topology.world_size) + "), actual=" +
            std::to_string(calling_rank));
  if (max_tokens_per_rank < num_tokens)
    invalid("phase=cuda-workspace-query, field=max_tokens_per_rank, rank=" +
            std::to_string(calling_rank) + ", expected>=" +
            std::to_string(num_tokens) + ", actual=" +
            std::to_string(max_tokens_per_rank));
  CudaNotifyWorkspacePlan plan;
  plan.calling_rank = calling_rank;
  plan.num_tokens = num_tokens;
  plan.max_tokens_per_rank = max_tokens_per_rank;
  plan.top_k = config.top_k;
  plan.num_experts = config.num_experts;
  plan.world_size = config.topology.world_size;
  plan.num_nodes = config.topology.num_nodes;
  plan.node_mask_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(plan.num_nodes) + 63U) / 64U);
  plan.route_count = checked_mul(num_tokens, config.top_k, "route_count");
  plan.max_incoming_rows = checked_mul(
      checked_mul(config.topology.world_size, max_tokens_per_rank,
                  "max_global_tokens"),
      config.top_k, "max_incoming_rows");
  if (plan.route_count > static_cast<std::uint64_t>(INT_MAX))
    invalid("phase=cuda-workspace-query, field=route_count, rank=n/a, expected<=INT_MAX, actual=" +
            std::to_string(plan.route_count));
  if (plan.route_count != 0 && config.num_experts >
                                   std::numeric_limits<std::uint64_t>::max() /
                                       plan.route_count)
    invalid("phase=cuda-workspace-query, field=sort_key, rank=n/a, expected=expert*route_count within uint64, actual=overflow");
  const auto cub_items = static_cast<int>(std::max<std::uint64_t>(plan.route_count, 1));
  std::size_t sort_bytes = 0;
  std::size_t scan_bytes = 0;
  (void)cub::DeviceRadixSort::SortPairs(
      nullptr, sort_bytes, static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr), cub_items);
  (void)cub::DeviceScan::InclusiveSumByKey(
      nullptr, scan_bytes, static_cast<std::uint32_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr),
      static_cast<std::uint64_t *>(nullptr), cub_items);
  plan.sort_temp_bytes = sort_bytes;
  plan.scan_temp_bytes = scan_bytes;
  const auto device = make_device_layout(plan);
  const auto host = make_host_layout(plan);
  plan.device_bytes = device.total;
  plan.host_staging_bytes = host.total;
  return plan;
}

struct CudaNotifyWorkspace::Impl {
  std::shared_ptr<WorkspaceState> state;
};

CudaNotifyWorkspace::CudaNotifyWorkspace() = default;
CudaNotifyWorkspace::CudaNotifyWorkspace(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaNotifyWorkspace::~CudaNotifyWorkspace() noexcept = default;
CudaNotifyWorkspace::CudaNotifyWorkspace(CudaNotifyWorkspace &&) noexcept = default;
CudaNotifyWorkspace &CudaNotifyWorkspace::operator=(CudaNotifyWorkspace &&) noexcept = default;

const CudaNotifyWorkspacePlan &CudaNotifyWorkspace::plan() const {
  if (!impl_ || !impl_->state)
    invalid("attempted to access empty CUDA notify workspace");
  return impl_->state->plan;
}

CudaNotifyWorkspace allocate_cuda_notify_workspace(
    const CudaNotifyWorkspacePlan &plan,
    const RankCollectiveContext &context, const CudaExecution &execution) {
  if (!context || !execution)
    invalid("phase=cuda-workspace-allocate, field=context, rank=n/a, expected=valid, actual=empty");
  if (plan.calling_rank != context.calling_rank())
    invalid("phase=cuda-workspace-allocate, field=calling_rank, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(context.calling_rank()) + ", actual=" +
            std::to_string(plan.calling_rank));
  if (plan.world_size != context.session().world_size())
    invalid("phase=cuda-workspace-allocate, field=world_size, rank=" +
            std::to_string(context.calling_rank()) + ", expected=" +
            std::to_string(context.session().world_size()) + ", actual=" +
            std::to_string(plan.world_size));
  if (!context.execution().same_instance(execution.token()))
    invalid("phase=cuda-workspace-allocate, field=execution, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=context execution instance, actual=different");
  const auto state_exec = internal::checked_execution_state(execution);
  auto state = std::make_shared<WorkspaceState>();
  state->plan = plan;
  state->device_layout = make_device_layout(plan);
  state->host_layout = make_host_layout(plan);
  if (state->device_layout.total != plan.device_bytes ||
      state->host_layout.total != plan.host_staging_bytes)
    invalid("phase=cuda-workspace-allocate, field=plan.bytes, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=recomputed layout, actual=mismatch");
  state->context = context;
  state->epoch = context.session().generation();
  state->execution = execution;
  state->allocation = std::make_shared<CudaAllocation>();
  auto &allocation = *state->allocation;
  allocation.device = state_exec->device;
  allocation.device_bytes = checked_size(plan.device_bytes, "device_bytes");
  allocation.host_bytes = checked_size(plan.host_staging_bytes, "host_bytes");
  internal::throw_cuda("workspace-allocate", cudaSetDevice(static_cast<int>(allocation.device)),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("workspace-allocate",
                       cudaMalloc(&allocation.device_ptr,
                                  std::max<std::size_t>(allocation.device_bytes, 1)),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaMalloc(workspace)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda("workspace-allocate",
                       cudaHostAlloc(&allocation.host_ptr,
                                     std::max<std::size_t>(allocation.host_bytes, 1),
                                     cudaHostAllocPortable |
                                         cudaHostAllocMapped),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaHostAlloc(staging)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  internal::throw_cuda(
      "workspace-allocate",
      cudaHostGetDevicePointer(&allocation.host_device_ptr,
                               allocation.host_ptr, 0),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaHostGetDevicePointer(staging)");
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.mapping_start),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate(mapping start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.mapping_complete),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate(mapping complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreateWithFlags(&allocation.completion,
                                                cudaEventDisableTiming),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.planning_start),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(planning start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreate(&allocation.planning_complete),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "cudaEventCreate(planning complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  allocation.consumer_events.reserve(4);
  for (std::uint32_t i = 0; i < 4; ++i) {
    cudaEvent_t event = nullptr;
    internal::throw_cuda("workspace-allocate",
                         cudaEventCreateWithFlags(&event,
                                                  cudaEventDisableTiming),
                         context.calling_rank(), context.session().world_size(),
                         context.session().identity(), context.session().generation(),
                         ExchangeKind::ExpertCounts,
                         "cudaEventCreate(consumer completion)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    allocation.consumer_events.push_back(event);
    allocation.consumer_event_states.push_back(
        internal::ConsumerEventSlotState::Available);
  }
  auto impl = std::make_shared<CudaNotifyWorkspace::Impl>();
  impl->state = std::move(state);
  return CudaNotifyWorkspace(std::move(impl));
}

std::shared_ptr<CudaNotifyPoolAllocation> DeviceNotifyPoolBridge::allocate(
    const CudaNotifyWorkspacePlan &max_plan,
    const CudaExecution &execution,
    std::uint32_t max_inflight_consumers) {
  if (!execution)
    invalid("phase=device-pool-allocate, field=execution, rank=n/a, "
            "expected=valid, actual=empty");
  const auto exec = internal::checked_execution_state(execution);
  auto result = std::make_shared<CudaNotifyPoolAllocation>();
  result->max_plan = max_plan;
  result->route = std::make_shared<CudaAllocation>();
  result->workspace = std::make_shared<CudaAllocation>();
  result->route->device = exec->device;
  result->workspace->device = exec->device;
  result->route->device_bytes = checked_size(
      checked_mul(max_plan.route_count, sizeof(std::int32_t),
                  "pool route bytes"),
      "pool route bytes");
  result->workspace->device_bytes =
      checked_size(max_plan.device_bytes, "pool workspace device bytes");
  result->workspace->host_bytes = checked_size(
      max_plan.host_staging_bytes, "pool workspace pinned bytes");
  internal::throw_cuda("device-pool-allocate",
                       cudaSetDevice(static_cast<int>(exec->device)),
                       max_plan.calling_rank, max_plan.world_size,
                       execution.token().identity(), 1,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda(
      "device-pool-allocate",
      cudaMalloc(&result->route->device_ptr,
                 std::max<std::size_t>(result->route->device_bytes, 1)),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaMalloc(pool route slot)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaMalloc(&result->workspace->device_ptr,
                 std::max<std::size_t>(result->workspace->device_bytes, 1)),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaMalloc(pool notify arena)");
  internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaHostAlloc(&result->workspace->host_ptr,
                    std::max<std::size_t>(result->workspace->host_bytes, 1),
                    cudaHostAllocPortable | cudaHostAllocMapped),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaHostAlloc(pool status staging)");
  internal::record_cuda_api(internal::CudaApiAuditKind::PinnedAllocate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaHostGetDevicePointer(&result->workspace->host_device_ptr,
                               result->workspace->host_ptr, 0),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaHostGetDevicePointer(pool status staging)");
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->mapping_start),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool mapping start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->mapping_complete),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool mapping complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreateWithFlags(&result->workspace->completion,
                               cudaEventDisableTiming),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool completion)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->planning_start),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool planning start)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreate(&result->workspace->planning_complete),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool planning complete)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  internal::throw_cuda(
      "device-pool-allocate",
      cudaEventCreateWithFlags(&result->route->producer_ready,
                               cudaEventDisableTiming),
      max_plan.calling_rank, max_plan.world_size,
      execution.token().identity(), 1, ExchangeKind::ExpertCounts,
      "cudaEventCreate(pool producer ready)");
  internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
  result->workspace->consumer_events.reserve(max_inflight_consumers);
  for (std::uint32_t i = 0; i < max_inflight_consumers; ++i) {
    cudaEvent_t event = nullptr;
    internal::throw_cuda(
        "device-pool-allocate",
        cudaEventCreateWithFlags(&event, cudaEventDisableTiming),
        max_plan.calling_rank, max_plan.world_size,
        execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaEventCreate(pool consumer completion)");
    internal::record_cuda_api(internal::CudaApiAuditKind::EventCreate);
    result->workspace->consumer_events.push_back(event);
    result->workspace->consumer_event_states.push_back(
        internal::ConsumerEventSlotState::Available);
  }
  return result;
}

CudaRouteInput DeviceNotifyPoolBridge::upload(
    const RankRoutingInput &routes, std::uint32_t top_k,
    const RankCollectiveContext &context,
    const CudaExecution &execution,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    std::shared_ptr<const void> slot_lease) {
  if (!allocation || !slot_lease)
    invalid("phase=device-pool-route-upload, field=pool_slot, rank=n/a, "
            "expected=valid, actual=empty");
  const auto count = checked_mul(routes.num_tokens, top_k, "route_count");
  if (routes.selected_experts.size() != checked_size(count, "route_count") ||
      count > allocation->max_plan.route_count)
    invalid("phase=device-pool-route-upload, field=shape, rank=n/a, "
            "expected=valid route count within pool, actual=" +
            std::to_string(routes.selected_experts.size()));
  const auto exec = internal::checked_execution_state(execution);
  const auto bytes = checked_size(
      checked_mul(count, sizeof(std::int32_t), "route bytes"),
      "route bytes");
  if (bytes != 0)
    internal::throw_cuda(
        "device-pool-route-upload",
        cudaMemcpyAsync(allocation->route->device_ptr,
                        routes.selected_experts.data(), bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        allocation->max_plan.calling_rank, allocation->max_plan.world_size,
        execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(host convenience route upload)");
  if (bytes != 0)
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice, bytes);
  internal::throw_cuda(
      "device-pool-route-upload",
      cudaEventRecord(allocation->route->producer_ready, exec->stream),
      context.calling_rank(), context.session().world_size(),
      context.session().identity(), context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventRecord(producer ready)");
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = allocation->route;
  impl->slot_lease = std::move(slot_lease);
  impl->device_ptr = allocation->route->device_ptr;
  impl->producer_session = context.session();
  impl->producer_execution = execution.token();
  impl->consumer_execution = execution.token();
  impl->device = exec->device;
  impl->generation = context.session().generation();
  impl->ready_recorded = true;
  impl->tokens = routes.num_tokens;
  impl->top_k = top_k;
  impl->routes = count;
  return CudaRouteInput(std::move(impl));
}

CudaNotifyWorkspace DeviceNotifyPoolBridge::workspace(
    const CudaNotifyWorkspacePlan &actual_plan,
    const RankCollectiveContext &context,
    const CudaExecution &execution,
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    std::shared_ptr<const void> slot_lease) {
  if (!allocation || !slot_lease || !context || !execution)
    invalid("phase=device-pool-workspace, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto device_layout = make_device_layout(actual_plan);
  const auto host_layout = make_host_layout(actual_plan);
  if (device_layout.total != actual_plan.device_bytes ||
      host_layout.total != actual_plan.host_staging_bytes ||
      device_layout.total > allocation->workspace->device_bytes ||
      host_layout.total > allocation->workspace->host_bytes ||
      actual_plan.route_count > allocation->max_plan.route_count ||
      actual_plan.max_incoming_rows >
          allocation->max_plan.max_incoming_rows)
    invalid("phase=device-pool-workspace, field=capacity, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=actual layout within pool slot, actual=short");
  auto state = std::make_shared<WorkspaceState>();
  state->plan = actual_plan;
  state->device_layout = device_layout;
  state->host_layout = host_layout;
  state->context = context;
  state->epoch = context.session().generation();
  state->execution = execution;
  state->allocation = allocation->workspace;
  state->slot_lease = std::move(slot_lease);
  auto impl = std::make_shared<CudaNotifyWorkspace::Impl>();
  impl->state = std::move(state);
  return CudaNotifyWorkspace(std::move(impl));
}

bool DeviceNotifyPoolBridge::drain(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation,
    const CudaExecution &execution) noexcept {
  if (!allocation || !execution)
    return false;
  try {
    const auto exec = internal::checked_execution_state(execution);
    (void)cudaSetDevice(static_cast<int>(exec->device));
    if (allocation->workspace && allocation->workspace->completion) {
      if (cudaEventRecord(allocation->workspace->completion, exec->stream) !=
          cudaSuccess)
        return false;
      if (cudaEventSynchronize(allocation->workspace->completion) !=
          cudaSuccess)
        return false;
    } else {
      if (cudaStreamSynchronize(exec->stream) != cudaSuccess)
        return false;
    }
    return true;
  } catch (...) {
    // Slot lease destruction is fail-closed and cannot throw. The pool owns
    // the allocation until teardown, whose allocation destructor drains again.
    return false;
  }
}

std::uint64_t DeviceNotifyPoolBridge::device_bytes(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation)
    return 0;
  return static_cast<std::uint64_t>(allocation->route->device_bytes) +
         static_cast<std::uint64_t>(allocation->workspace->device_bytes);
}

std::uint64_t DeviceNotifyPoolBridge::pinned_bytes(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  return allocation
             ? static_cast<std::uint64_t>(allocation->workspace->host_bytes)
             : 0;
}

std::uint64_t DeviceNotifyPoolBridge::event_count(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation)
    return 0;
  return 6U + allocation->workspace->consumer_events.size();
}

bool DeviceNotifyPoolBridge::quarantined(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  if (!allocation || !allocation->workspace)
    return true;
  std::lock_guard<std::mutex> lock(allocation->workspace->consumer_mutex);
  return allocation->workspace->dispatch_slot_quarantined.load(
      std::memory_order_acquire);
}

std::array<std::uint32_t, 3> DeviceNotifyPoolBridge::consumer_event_usage(
    const std::shared_ptr<CudaNotifyPoolAllocation> &allocation) noexcept {
  std::array<std::uint32_t, 3> usage{};
  if (!allocation || !allocation->workspace)
    return usage;
  std::lock_guard<std::mutex> lock(allocation->workspace->consumer_mutex);
  for (const auto state : allocation->workspace->consumer_event_states) {
    if (state == internal::ConsumerEventSlotState::Available)
      ++usage[0];
    else if (state == internal::ConsumerEventSlotState::InUse)
      ++usage[1];
    else
      ++usage[2];
  }
  return usage;
}

struct CudaLocalNotifyHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  std::uint64_t num_tokens = 0;
  DeviceDispatchMetadata device;
  DeviceBufferLease expert_count;
  DeviceBufferLease packet_count;
  CudaRouteInput routes;
  CudaGateInput gates;
  std::shared_ptr<WorkspaceState> workspace;
};

std::uint64_t CudaLocalNotifyHandle::num_tokens() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->num_tokens;
}
const RuntimeConfig &CudaLocalNotifyHandle::config() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->config;
}
const RankCollectiveContext &CudaLocalNotifyHandle::context() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->context;
}
const DeviceBufferLease &
CudaLocalNotifyHandle::device_expert_count_for_backend() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local expert count");
  return impl_->expert_count;
}
const DeviceBufferLease &
CudaLocalNotifyHandle::device_packet_count_for_backend() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local packet count");
  return impl_->packet_count;
}

struct CudaNotifyPreflight::Impl {
  LocalPreflightStatus status;
  CudaLocalNotifyHandle local;
  double device_mapping_us = 0;
};

const LocalPreflightStatus &CudaNotifyPreflight::status() const {
  if (!impl_)
    invalid("phase=cuda-preflight, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->status;
}

double CudaNotifyPreflight::device_mapping_us() const {
  if (!impl_)
    invalid("phase=cuda-preflight, field=handle, rank=n/a, expected=valid, actual=empty");
  return impl_->device_mapping_us;
}

bool CudaNotifyPreflight::valid() const {
  return impl_ && impl_->status.error == ControlLocalError::None &&
         static_cast<bool>(impl_->local);
}

const CudaLocalNotifyHandle &CudaNotifyPreflight::local() const {
  if (!valid())
    invalid("phase=cuda-preflight, field=local, rank=n/a, expected=successful preflight, actual=failed/empty");
  return impl_->local;
}

struct CudaNotifyOperation::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  CudaRouteInput routes;
  CudaGateInput gates;
  std::shared_ptr<WorkspaceState> workspace;
  bool terminal = false;
  bool waited = false;
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    const CudaGateInput &gates, CudaNotifyWorkspace workspace) {
  validate_runtime_config(config);
  if (!context || !execution || !routes || !gates || !workspace)
    invalid("phase=cuda-notify-start, field=input, rank=n/a, expected=all handles valid, actual=empty");
  const auto rank = context.calling_rank();
  if (context.session().world_size() != config.topology.world_size ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=cuda-notify-start, field=context, rank=" +
            std::to_string(rank) +
            ", expected=config world and exact execution, actual=mismatch");
  DeviceRouteProvenanceBridge::validate(routes, context, execution);
  DeviceGateProvenanceBridge::validate(gates, routes, context, execution);
  const auto expected = query_cuda_notify_workspace(
      config, context.calling_rank(), routes.num_tokens(),
      workspace.plan().max_tokens_per_rank);
  const auto &actual = workspace.plan();
  if (expected.calling_rank != actual.calling_rank ||
      expected.num_tokens != actual.num_tokens ||
      expected.max_tokens_per_rank != actual.max_tokens_per_rank ||
      expected.top_k != actual.top_k ||
      expected.num_experts != actual.num_experts ||
      expected.world_size != actual.world_size ||
      expected.num_nodes != actual.num_nodes ||
      expected.node_mask_words != actual.node_mask_words ||
      expected.route_count != actual.route_count ||
      expected.max_incoming_rows != actual.max_incoming_rows ||
      expected.device_bytes != actual.device_bytes ||
      expected.host_staging_bytes != actual.host_staging_bytes ||
      expected.sort_temp_bytes != actual.sort_temp_bytes ||
      expected.scan_temp_bytes != actual.scan_temp_bytes)
    invalid("phase=cuda-notify-start, field=workspace.plan, rank=" +
            std::to_string(rank) + ", expected=exact queried plan, actual=mismatch");
  if (!workspace.impl_->state->context.same_instance(context))
    invalid("phase=cuda-notify-start, field=workspace.context, rank=" +
            std::to_string(rank) + ", expected=exact context, actual=different");
  auto state = workspace.impl_->state;
  auto exec = internal::checked_execution_state(execution);
  const auto &dl = state->device_layout;
  void *base = state->allocation->device_ptr;
  internal::throw_cuda("notify-start", cudaSetDevice(static_cast<int>(exec->device)),
                       rank, context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.expert_to_rank),
                                       config.topology.expert_to_rank.data(),
                                       config.num_experts * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(expert_to_rank)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.num_experts) * sizeof(std::uint32_t));
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_node),
                                       config.topology.rank_to_node.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_node)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.topology.world_size) *
          sizeof(std::uint32_t));
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_local),
                                       config.topology.rank_to_local_rank.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_local)");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::HostToDevice,
      static_cast<std::uint64_t>(config.topology.world_size) *
          sizeof(std::uint32_t));
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.expert_count), 0,
                                                        config.num_experts * sizeof(std::uint64_t), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemsetAsync(expert_count)");
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.packet_count), 0,
                                                        config.topology.num_nodes * sizeof(std::uint64_t), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::NotifyPacketSummary,
                       "cudaMemsetAsync(packet_count)");
  internal::throw_cuda("notify-start", cudaMemsetAsync(at<void>(base, dl.status), 0xff,
                                                        sizeof(DeviceStatus), exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemsetAsync(status)");
  const auto route_count = state->plan.route_count;
  internal::throw_cuda(
      "notify-start",
      cudaEventRecord(state->allocation->mapping_start, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapping start)");
  internal::throw_cuda(
      "notify-start",
      cudaMemsetAsync(
          at<void>(base, dl.local_offset), 0,
          checked_size(checked_mul(route_count, sizeof(std::uint64_t),
                                   "local_offset_bytes"),
                       "local_offset_bytes"),
          exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaMemsetAsync(local_offset)");
  const auto route_blocks = static_cast<unsigned>((route_count + 255U) / 256U);
  const auto token_blocks = static_cast<unsigned>((state->plan.num_tokens + 255U) / 256U);
  if (route_count != 0) {
    prepare_routes_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        static_cast<const std::int32_t *>(routes.impl_->device_ptr),
        route_count, route_count, config.num_experts,
        at<const std::uint32_t>(base, dl.expert_to_rank),
        at<const std::uint32_t>(base, dl.rank_to_node),
        at<const std::uint32_t>(base, dl.rank_to_local),
        at<std::uint64_t>(base, dl.keys_in),
        at<std::uint64_t>(base, dl.indices_in),
        at<std::uint32_t>(base, dl.expert_id),
        at<std::uint32_t>(base, dl.route_dst_rank),
        at<std::uint32_t>(base, dl.route_dst_node),
        at<std::uint32_t>(base, dl.route_dst_local_rank),
        at<std::uint64_t>(base, dl.ones), at<DeviceStatus>(base, dl.status));
    validate_and_describe_tokens_kernel<<<token_blocks, 256, 0, exec->stream>>>(
        static_cast<const std::int32_t *>(routes.impl_->device_ptr),
        state->plan.num_tokens, config.top_k, config.num_experts,
        at<const std::uint32_t>(base, dl.expert_to_rank),
        at<const std::uint32_t>(base, dl.rank_to_node), config.topology.num_nodes,
        state->plan.node_mask_words, at<std::uint32_t>(base, dl.combine_order),
        at<std::uint64_t>(base, dl.masks),
        at<std::uint64_t>(base, dl.packet_count),
        at<DeviceStatus>(base, dl.status));
    internal::throw_cuda("notify-start", cudaGetLastError(), rank,
                         context.session().world_size(), context.session().identity(),
                         context.session().generation(), ExchangeKind::ExpertCounts,
                         "route kernels launch");
    std::size_t sort_bytes = checked_size(state->plan.sort_temp_bytes, "sort_temp_bytes");
    internal::throw_cuda(
        "notify-start",
        cub::DeviceRadixSort::SortPairs(
            at<void>(base, dl.temp), sort_bytes,
            at<std::uint64_t>(base, dl.keys_in),
            at<std::uint64_t>(base, dl.keys_out),
            at<std::uint64_t>(base, dl.indices_in),
            at<std::uint64_t>(base, dl.indices_out),
            static_cast<int>(route_count), 0, 64, exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cub::DeviceRadixSort::SortPairs");
    extract_sorted_experts_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        at<const std::uint64_t>(base, dl.keys_out),
        at<std::uint32_t>(base, dl.sorted_experts), route_count, route_count);
    std::size_t scan_bytes = checked_size(state->plan.scan_temp_bytes, "scan_temp_bytes");
    internal::throw_cuda(
        "notify-start",
        cub::DeviceScan::InclusiveSumByKey(
            at<void>(base, dl.temp), scan_bytes,
            at<std::uint32_t>(base, dl.sorted_experts),
            at<std::uint64_t>(base, dl.ones),
            at<std::uint64_t>(base, dl.inclusive),
            static_cast<int>(route_count), std::equal_to<std::uint32_t>(),
            exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cub::DeviceScan::InclusiveSumByKey");
    histogram_and_scatter_kernel<<<route_blocks, 256, 0, exec->stream>>>(
        at<const std::uint32_t>(base, dl.sorted_experts),
        at<const std::uint64_t>(base, dl.indices_out),
        at<const std::uint64_t>(base, dl.inclusive), route_count,
        config.num_experts, at<std::uint64_t>(base, dl.expert_count),
        at<std::uint64_t>(base, dl.local_offset));
    internal::throw_cuda("notify-start", cudaGetLastError(), rank,
                         context.session().world_size(), context.session().identity(),
                         context.session().generation(), ExchangeKind::ExpertCounts,
                         "mapping kernels launch");
  }
  internal::throw_cuda(
      "notify-start",
      cudaEventRecord(state->allocation->mapping_complete, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapping complete)");
  const auto &hl = state->host_layout;
  void *host = state->allocation->host_ptr;
  const auto copy = [&](std::size_t host_offset, std::size_t device_offset,
                        std::size_t bytes, const char *name) {
    if (bytes == 0)
      return;
    internal::throw_cuda("notify-start",
                         cudaMemcpyAsync(at<void>(host, host_offset),
                                         at<void>(base, device_offset), bytes,
                                         cudaMemcpyDeviceToHost, exec->stream),
                         rank, context.session().world_size(),
                         context.session().identity(),
                         context.session().generation(),
                         ExchangeKind::ExpertCounts, name);
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                              bytes);
  };
  copy(hl.status, dl.status, sizeof(DeviceStatus), "D2H status");
  internal::throw_cuda("notify-start",
                       cudaEventRecord(state->allocation->completion, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaEventRecord");
  auto impl = std::make_shared<CudaNotifyOperation::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->routes = routes;
  impl->gates = gates;
  impl->workspace = std::move(state);
  return CudaNotifyOperation(std::move(impl));
}

CudaNotifyOperation::CudaNotifyOperation() = default;
CudaNotifyOperation::CudaNotifyOperation(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CudaNotifyOperation::~CudaNotifyOperation() noexcept {
  if (impl_ && !impl_->terminal && impl_->workspace) {
    (void)cudaSetDevice(static_cast<int>(impl_->execution.device_ordinal()));
    (void)cudaEventSynchronize(impl_->workspace->allocation->completion);
    impl_->terminal = true;
  }
}
CudaNotifyOperation::CudaNotifyOperation(CudaNotifyOperation &&) noexcept = default;
CudaNotifyOperation &CudaNotifyOperation::operator=(CudaNotifyOperation &&other) noexcept {
  if (this == &other)
    return *this;
  if (impl_ && !impl_->terminal && impl_->workspace) {
    (void)cudaSetDevice(static_cast<int>(impl_->execution.device_ordinal()));
    (void)cudaEventSynchronize(impl_->workspace->allocation->completion);
  }
  impl_ = std::move(other.impl_);
  return *this;
}

CudaNotifyPreflight CudaNotifyOperation::wait_preflight() {
  if (!impl_)
    invalid("phase=cuda-notify-wait, field=operation, rank=n/a, expected=valid, actual=empty");
  if (impl_->waited || impl_->terminal)
    invalid("phase=cuda-notify-wait, field=state, rank=" +
            std::to_string(impl_->context.calling_rank()) +
            ", expected=pending, actual=terminal");
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda("notify-wait",
                       cudaEventSynchronize(impl_->workspace->allocation->completion),
                       rank, impl_->context.session().world_size(),
                       impl_->context.session().identity(),
                       impl_->context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventSynchronize");
  impl_->terminal = true;
  impl_->waited = true;
  const auto &state = impl_->workspace;
  const auto *status = at<const DeviceStatus>(state->allocation->host_ptr,
                                              state->host_layout.status);
  auto preflight_impl = std::make_shared<CudaNotifyPreflight::Impl>();
  float mapping_ms = 0;
  internal::throw_cuda(
      "notify-wait",
      cudaEventElapsedTime(&mapping_ms,
                           impl_->workspace->allocation->mapping_start,
                           impl_->workspace->allocation->mapping_complete),
      rank, impl_->context.session().world_size(),
      impl_->context.session().identity(), impl_->context.session().generation(),
      ExchangeKind::ExpertCounts, "cudaEventElapsedTime(mapping)");
  preflight_impl->device_mapping_us =
      static_cast<double>(mapping_ms) * 1000.0;
  const auto unpack_status = [&](unsigned long long packed,
                                 ControlLocalError error,
                                 std::uint64_t expected,
                                 std::uint64_t actual) {
    const auto flat = static_cast<std::uint64_t>(packed >> 32U);
    const auto expert_bits = static_cast<std::uint32_t>(packed);
    std::int32_t raw_expert = 0;
    std::memcpy(&raw_expert, &expert_bits, sizeof(raw_expert));
    preflight_impl->status = LocalPreflightStatus{
        error, ControlField::LocalValidation, flat / impl_->config.top_k,
        flat % impl_->config.top_k, raw_expert, expected, actual};
  };
  if (status->out_of_range_packed != ~0ULL) {
    const auto expert_bits =
        static_cast<std::uint32_t>(status->out_of_range_packed);
    unpack_status(status->out_of_range_packed,
                  ControlLocalError::ExpertOutOfRange,
                  impl_->config.num_experts, expert_bits);
    return CudaNotifyPreflight(std::move(preflight_impl));
  }
  if (status->duplicate_packed != ~0ULL) {
    unpack_status(status->duplicate_packed, ControlLocalError::DuplicateExpert,
                  0, 1);
    return CudaNotifyPreflight(std::move(preflight_impl));
  }
  auto result = std::make_shared<CudaLocalNotifyHandle::Impl>();
  result->config = impl_->config;
  result->context = impl_->context;
  result->num_tokens = state->plan.num_tokens;
  result->routes = impl_->routes;
  result->gates = impl_->gates;
  result->workspace = state;
  const auto route_width = state->plan.route_count;
  result->device.expert_id = make_lease(state, state->device_layout.expert_id, route_width);
  result->device.local_offset = make_lease(state, state->device_layout.local_offset, route_width);
  result->device.route_dst_row = make_lease(state, state->device_layout.route_dst_row, route_width);
  result->device.route_dst_rank = make_lease(state, state->device_layout.route_dst_rank, route_width);
  result->device.route_dst_node = make_lease(state, state->device_layout.route_dst_node, route_width);
  result->device.route_dst_local_rank = make_lease(state, state->device_layout.route_dst_local_rank, route_width);
  result->device.combine_order = make_lease(state, state->device_layout.combine_order, route_width);
  result->device.destination_node_mask = make_lease(
      state, state->device_layout.masks,
      checked_mul(state->plan.num_tokens, state->plan.node_mask_words, "mask lease"));
  result->expert_count = make_lease(
      state, state->device_layout.expert_count, state->plan.num_experts);
  result->packet_count = make_lease(
      state, state->device_layout.packet_count, state->plan.num_nodes);
  preflight_impl->local = CudaLocalNotifyHandle(std::move(result));
  return CudaNotifyPreflight(std::move(preflight_impl));
}

CudaLocalNotifyHandle CudaNotifyOperation::wait() {
  auto preflight = wait_preflight();
  if (preflight.valid())
    return preflight.local();
  const auto &status = preflight.status();
  const auto rank = impl_->context.calling_rank();
  const auto flat = status.token * impl_->config.top_k + status.slot;
  if (status.error == ControlLocalError::ExpertOutOfRange) {
    throw NotifyDispatchError(
        NotifyErrorCode::ExpertOutOfRange,
        "phase=cuda-device-validation, field=selected_experts, rank=" +
            std::to_string(rank) + ", expected=[0," +
            std::to_string(impl_->config.num_experts) + "), actual=" +
            std::to_string(status.expert) + ", " +
            route_coordinates(flat, impl_->config.top_k));
  }
  throw NotifyDispatchError(
      NotifyErrorCode::DuplicateExpert,
      "phase=cuda-device-validation, field=selected_experts, rank=" +
          std::to_string(rank) +
          ", expected=unique within token, actual=duplicate expert=" +
          std::to_string(status.expert) + ", " +
          route_coordinates(flat, impl_->config.top_k));
}

void CudaNotifyOperation::cancel() {
  if (!impl_ || impl_->terminal)
    return;
  const auto rank = impl_->context.calling_rank();
  internal::throw_cuda("notify-cancel",
                       cudaEventSynchronize(impl_->workspace->allocation->completion),
                       rank, impl_->context.session().world_size(),
                       impl_->context.session().identity(),
                       impl_->context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventSynchronize");
  impl_->terminal = true;
}


struct DeviceDispatchHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  DeviceDispatchMetadata device;
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  DeviceMaterializationAudit audit;
  CudaRouteInput source_routes;
  CudaGateInput source_gates;
  std::shared_ptr<WorkspaceState> workspace;
  BackendCompletion expert_completion;
  BackendCompletion summary_completion;
};

struct DeviceDispatchConsumerLease::Impl {
  DeviceDispatchHandle owner;
  std::shared_ptr<WorkspaceState> workspace;
  DeviceDispatchMetadata device;
  DeviceDispatchIdentity identity;
  DeviceDispatchSizes sizes;
  RankCollectiveContext context;
  std::uint64_t epoch = 0;
  CudaExecution execution;
  std::size_t event_index = SIZE_MAX;
  DeviceDispatchConsumerState state = DeviceDispatchConsumerState::Acquired;

  DeviceDispatchKernelView kernel_view() const {
    if (!owner || state == DeviceDispatchConsumerState::Released ||
        !workspace || workspace->epoch != epoch ||
        identity.generation != epoch ||
        workspace->allocation->dispatch_slot_quarantined.load(
            std::memory_order_acquire))
      invalid("phase=dispatch-consumer-view, field=state, rank=n/a, "
              "expected=acquired/submitted, actual=empty/released");
    const auto &workspace_ref = *workspace;
    const auto &dl = workspace_ref.device_layout;
    auto *base = workspace_ref.allocation->device_ptr;
    auto exec = internal::checked_execution_state(execution);
    auto count = internal::CudaDeviceLeaseAccess::checked(
        device.source_expert_count, exec->device);
    DeviceDispatchKernelView view{};
    view.identity = identity;
    view.sizes = sizes;
    view.expert_id = at<const std::uint32_t>(base, dl.expert_id);
    view.local_offset = at<const std::uint64_t>(base, dl.local_offset);
    view.route_dst_row = at<const std::uint64_t>(base, dl.route_dst_row);
    view.route_dst_rank = at<const std::uint32_t>(base, dl.route_dst_rank);
    view.route_dst_node = at<const std::uint32_t>(base, dl.route_dst_node);
    view.route_dst_local_rank =
        at<const std::uint32_t>(base, dl.route_dst_local_rank);
    view.combine_order = at<const std::uint32_t>(base, dl.combine_order);
    view.destination_node_mask = at<const std::uint64_t>(base, dl.masks);
    view.source_expert_count = static_cast<const std::uint64_t *>(count->pointer);
    view.owned_expert_ids = at<const std::uint32_t>(base, dl.owned_expert_ids);
    view.expert_row_count = at<const std::uint64_t>(base, dl.owned_expert_count);
    view.expert_row_prefix = at<const std::uint64_t>(base, dl.owned_expert_prefix);
    view.mtile_row_offsets = at<const std::uint64_t>(base, dl.mtile_offsets);
    view.mtile_expected_rows = at<const std::uint32_t>(base, dl.mtile_rows);
    view.incoming_rows = at<const DeviceIncomingRowRecord>(base, dl.incoming_rows);
    view.reverse_routes = at<const DeviceReverseRouteRecord>(base, dl.reverse_routes);
    view.packet_headers = at<const DevicePacketHeaderRecord>(base, dl.packet_headers);
    view.packet_fanout_offsets =
        at<const std::uint64_t>(base, dl.packet_fanout_offsets);
    view.packet_fanout = at<const DeviceFanoutRecord>(base, dl.packet_fanout);
    view.outgoing_signals = at<const DeviceSignalRecord>(base, dl.outgoing_signals);
    view.incoming_proxy_signals =
        at<const DeviceSignalRecord>(base, dl.incoming_signals);
    return view;
  }

  void release_owner_noexcept(bool event_usable) noexcept {
    if (!owner || state == DeviceDispatchConsumerState::Released)
      return;
    auto workspace_owner = workspace;
    {
      std::lock_guard<std::mutex> lock(
          workspace_owner->allocation->consumer_mutex);
      if (event_index <
          workspace_owner->allocation->consumer_event_states.size())
        workspace_owner->allocation->consumer_event_states[event_index] =
            event_usable ? internal::ConsumerEventSlotState::Available
                         : internal::ConsumerEventSlotState::Quarantined;
    }
    state = DeviceDispatchConsumerState::Released;
    owner = {};
    device = {};
    workspace.reset();
  }

  bool drain_submitted_noexcept() noexcept {
    auto workspace_owner = workspace;
    (void)cudaSetDevice(static_cast<int>(execution.device_ordinal()));
    const bool injected_both = take_consumer_fault(
        testing::ConsumerFaultPoint::EventAndStreamSynchronize);
    const bool injected_event = injected_both || take_consumer_fault(
        testing::ConsumerFaultPoint::EventSynchronize);
    const bool event_ok = !injected_event &&
        cudaEventSynchronize(
            workspace_owner->allocation->consumer_events[event_index]) ==
        cudaSuccess;
    bool stream_ok = true;
    if (!event_ok) {
      stream_ok = false;
      try {
        auto exec = internal::checked_execution_state(execution);
        stream_ok = !injected_both &&
                    cudaStreamSynchronize(exec->stream) == cudaSuccess;
      } catch (...) {
        stream_ok = false;
      }
    }
    if (!event_ok && !stream_ok) {
      std::lock_guard<std::mutex> lock(
          workspace_owner->allocation->consumer_mutex);
      workspace_owner->allocation->dispatch_slot_quarantined.store(
          true, std::memory_order_release);
    }
    state = DeviceDispatchConsumerState::Completed;
    release_owner_noexcept(event_ok);
    return event_ok;
  }

  void record_completion() {
    if (state != DeviceDispatchConsumerState::Acquired)
      invalid("phase=dispatch-consumer-submit, field=state, rank=" +
              std::to_string(identity.calling_rank) +
              ", expected=Acquired, actual=invalid");
    try {
      if (take_consumer_fault(testing::ConsumerFaultPoint::EventRecord))
        throw NotifyDispatchError(
            NotifyErrorCode::CudaFailure,
            "phase=dispatch-consumer-submit, field=event_record, rank=" +
                std::to_string(identity.calling_rank) +
                ", expected=success, actual=injected failure");
      internal::throw_cuda(
          "dispatch-consumer-submit",
          cudaEventRecord(workspace->allocation->consumer_events[event_index],
                          internal::checked_execution_state(execution)->stream),
          identity.calling_rank, identity.world_size,
          context.session().identity(), identity.generation,
          ExchangeKind::ExpertCounts, "cudaEventRecord(consumer completion)");
      state = DeviceDispatchConsumerState::Submitted;
    } catch (...) {
      bool stream_ok = false;
      try {
        stream_ok = cudaStreamSynchronize(
                        internal::checked_execution_state(execution)->stream) ==
                    cudaSuccess;
      } catch (...) {
        stream_ok = false;
      }
      if (!stream_ok) {
        std::lock_guard<std::mutex> lock(
            workspace->allocation->consumer_mutex);
        workspace->allocation->dispatch_slot_quarantined.store(
            true, std::memory_order_release);
      }
      release_owner_noexcept(false);
      throw;
    }
  }

  void release_noexcept() noexcept {
    if (!owner || state == DeviceDispatchConsumerState::Released)
      return;
    if (state == DeviceDispatchConsumerState::Submitted) {
      (void)drain_submitted_noexcept();
      return;
    }
    release_owner_noexcept(true);
  }
};

namespace {

void validate_device_completion(const BackendCompletion &completion,
                                const RankCollectiveContext &context,
                                ExchangeKind kind, std::uint64_t width) {
  if (!completion.has_device_result() || completion.has_host_result() ||
      completion.kind() != kind ||
      !completion.context().same_instance(context))
    invalid("phase=device-materialize, field=completion.provenance, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=device-only exact context/kind, actual=mismatch");
  const auto &lease = completion.device_result();
  if (lease.rows() != context.session().world_size() ||
      lease.row_width() != width)
    invalid("phase=device-materialize, field=completion.shape, rank=" +
            std::to_string(context.calling_rank()) + ", expected=[" +
            std::to_string(context.session().world_size()) + "," +
            std::to_string(width) + "], actual=[" +
            std::to_string(lease.rows()) + "," +
            std::to_string(lease.row_width()) + "]");
}

template <typename T>
std::vector<T> download_vector(const std::shared_ptr<WorkspaceState> &state,
                               const CudaExecutionState &exec,
                               const void *source, std::uint64_t count,
                               const char *field) {
  std::vector<T> result(checked_size(count, field));
  if (count == 0)
    return result;
  internal::throw_cuda(
      "device-validation-download",
      cudaMemcpy(result.data(), source,
                 checked_size(checked_mul(count, sizeof(T), field), field),
                 cudaMemcpyDeviceToHost),
      state->context.calling_rank(), state->context.session().world_size(),
      state->context.session().identity(),
      state->context.session().generation(), ExchangeKind::ExpertCounts, field);
  (void)exec;
  return result;
}

} // namespace

DeviceDispatchHandle DeviceMetadataBridge::materialize(
    const CudaLocalNotifyHandle &local, BackendCompletion expert_counts,
    BackendCompletion packet_summary, const CudaExecution &execution,
    Timings *timings) {
  if (!local || !execution)
    invalid("phase=device-materialize, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &context = local.impl_->context;
  const auto &config = local.impl_->config;
  if (!context.execution().same_instance(execution.token()))
    invalid("phase=device-materialize, field=execution, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=local execution, actual=different");
  validate_device_completion(expert_counts, context,
                             ExchangeKind::ExpertCounts,
                             config.num_experts);
  validate_device_completion(packet_summary, context,
                             ExchangeKind::NotifyPacketSummary,
                             config.topology.num_nodes);
  auto exec = internal::checked_execution_state(execution);
  auto count_state = internal::CudaDeviceLeaseAccess::checked(
      expert_counts.device_result(), exec->device);
  auto summary_state = internal::CudaDeviceLeaseAccess::checked(
      packet_summary.device_result(), exec->device);
  auto state = local.impl_->workspace;
  const auto &plan = state->plan;
  const auto &dl = state->device_layout;
  auto *base = state->allocation->device_ptr;
  const auto rank = context.calling_rank();
#if !MEGAMOE_DEBUG_VALIDATE
  auto *host_ready = at<std::uint32_t>(
      state->allocation->host_ptr,
      state->host_layout.materialization_ready);
  __atomic_store_n(host_ready, 0U, __ATOMIC_RELEASE);
#endif
  internal::throw_cuda("device-materialize",
                       cudaSetDevice(static_cast<int>(exec->device)), rank,
                       context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaSetDevice");

  const auto h2d = [&](std::size_t offset, const void *source,
                       std::uint64_t count, std::size_t element_bytes,
                       const char *field) {
    if (count == 0)
      return;
    const auto bytes =
        checked_size(checked_mul(count, element_bytes, field), field);
    internal::throw_cuda(
        "device-materialize",
        cudaMemcpyAsync(at<void>(base, offset), source, bytes,
                        cudaMemcpyHostToDevice, exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts, field);
    internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                              bytes);
  };
  h2d(dl.expert_capacity, config.capacity.expert_rows.data(),
      config.capacity.expert_rows.size(), sizeof(std::uint64_t),
      "expert capacity");
  h2d(dl.workspace_capacity,
      config.capacity.workspace_rows_per_rank.data(),
      config.capacity.workspace_rows_per_rank.size(), sizeof(std::uint64_t),
      "workspace capacity");

  const auto zero = [&](std::size_t offset, std::uint64_t count,
                        std::size_t element_bytes, const char *field) {
    if (count == 0)
      return;
    internal::throw_cuda(
        "device-materialize",
        cudaMemsetAsync(at<void>(base, offset), 0,
                        checked_size(checked_mul(count, element_bytes, field),
                                     field),
                        exec->stream),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts, field);
  };
  zero(dl.materialization_status, 1, sizeof(DeviceMaterializationStatus),
       "materialization status");
  zero(dl.expert_total, plan.num_experts, sizeof(std::uint64_t),
       "expert total");
  zero(dl.expert_base, plan.num_experts, sizeof(std::uint64_t), "expert base");
  zero(dl.route_dst_row, plan.route_count, sizeof(std::uint64_t),
       "route dst row");
  zero(dl.owned_expert_ids, plan.num_experts, sizeof(std::uint32_t),
       "owned experts");
  zero(dl.owned_expert_count, plan.num_experts, sizeof(std::uint64_t),
       "owned counts");
  zero(dl.owned_expert_prefix,
       static_cast<std::uint64_t>(plan.num_experts) + 1,
       sizeof(std::uint64_t), "owned prefix");
  zero(dl.mtile_offsets, static_cast<std::uint64_t>(plan.num_experts) + 1,
       sizeof(std::uint64_t), "mtile offsets");
  zero(dl.mtile_rows, plan.max_incoming_rows, sizeof(std::uint32_t),
       "mtile rows");
  zero(dl.incoming_rows, plan.max_incoming_rows,
       sizeof(DeviceIncomingRowRecord), "incoming rows");
  zero(dl.reverse_routes, plan.route_count,
       sizeof(DeviceReverseRouteRecord), "reverse routes");
  zero(dl.packet_headers, plan.route_count,
       sizeof(DevicePacketHeaderRecord), "packet headers");
  zero(dl.packet_fanout_offsets, plan.route_count + 1,
       sizeof(std::uint64_t), "fanout offsets");
  zero(dl.packet_fanout, plan.route_count, sizeof(DeviceFanoutRecord),
       "fanout");
  zero(dl.outgoing_signals, plan.num_nodes, sizeof(DeviceSignalRecord),
       "outgoing signals");
  zero(dl.incoming_signals, plan.num_nodes, sizeof(DeviceSignalRecord),
       "incoming signals");

  DeviceMaterializeArgs args{};
  args.rank = rank;
  args.world = plan.world_size;
  args.experts = plan.num_experts;
  args.nodes = plan.num_nodes;
  args.top_k = plan.top_k;
  args.mask_words = plan.node_mask_words;
  args.m_tile = config.m_tile_rows;
  args.tokens = plan.num_tokens;
  args.routes = plan.route_count;
  args.max_tokens = plan.max_tokens_per_rank;
  args.max_incoming_rows = plan.max_incoming_rows;
  args.has_expert_capacity = !config.capacity.expert_rows.empty();
  args.has_workspace_capacity =
      !config.capacity.workspace_rows_per_rank.empty();
  args.count_matrix = static_cast<const std::uint64_t *>(count_state->pointer);
  args.summary_matrix =
      static_cast<const std::uint64_t *>(summary_state->pointer);
  args.local_histogram = at<const std::uint64_t>(base, dl.expert_count);
  args.local_summary = at<const std::uint64_t>(base, dl.packet_count);
  args.expert_to_rank = at<const std::uint32_t>(base, dl.expert_to_rank);
  args.rank_to_node = at<const std::uint32_t>(base, dl.rank_to_node);
  args.rank_to_local = at<const std::uint32_t>(base, dl.rank_to_local);
  args.expert_capacity = at<const std::uint64_t>(base, dl.expert_capacity);
  args.workspace_capacity =
      at<const std::uint64_t>(base, dl.workspace_capacity);
  args.expert_id = at<const std::uint32_t>(base, dl.expert_id);
  args.local_offset = at<const std::uint64_t>(base, dl.local_offset);
  args.route_dst_rank = at<const std::uint32_t>(base, dl.route_dst_rank);
  args.route_dst_node = at<const std::uint32_t>(base, dl.route_dst_node);
  args.route_dst_local =
      at<const std::uint32_t>(base, dl.route_dst_local_rank);
  args.combine_order = at<const std::uint32_t>(base, dl.combine_order);
  args.masks = at<const std::uint64_t>(base, dl.masks);
  args.expert_total = at<std::uint64_t>(base, dl.expert_total);
  args.expert_base = at<std::uint64_t>(base, dl.expert_base);
  args.route_dst_row = at<std::uint64_t>(base, dl.route_dst_row);
  args.owned_expert_ids = at<std::uint32_t>(base, dl.owned_expert_ids);
  args.owned_expert_count = at<std::uint64_t>(base, dl.owned_expert_count);
  args.owned_expert_prefix = at<std::uint64_t>(base, dl.owned_expert_prefix);
  args.mtile_offsets = at<std::uint64_t>(base, dl.mtile_offsets);
  args.mtile_rows = at<std::uint32_t>(base, dl.mtile_rows);
  args.incoming_rows =
      at<DeviceIncomingRowRecord>(base, dl.incoming_rows);
  args.reverse_routes =
      at<DeviceReverseRouteRecord>(base, dl.reverse_routes);
  args.packet_headers =
      at<DevicePacketHeaderRecord>(base, dl.packet_headers);
  args.packet_fanout_offsets =
      at<std::uint64_t>(base, dl.packet_fanout_offsets);
  args.packet_fanout = at<DeviceFanoutRecord>(base, dl.packet_fanout);
  args.outgoing_signals =
      at<DeviceSignalRecord>(base, dl.outgoing_signals);
  args.incoming_signals =
      at<DeviceSignalRecord>(base, dl.incoming_signals);
  args.status =
      at<DeviceMaterializationStatus>(base, dl.materialization_status);

  constexpr std::uint32_t threads = 256;
  const auto blocks_for = [&](std::uint64_t count, const char *field) {
    const auto blocks = (count + threads - 1U) / threads;
    if (blocks > std::numeric_limits<unsigned int>::max())
      invalid(std::string("phase=device-materialize, field=") + field +
              ", rank=n/a, expected=CUDA grid.x capacity, actual=" +
              std::to_string(blocks));
    return static_cast<unsigned int>(std::max<std::uint64_t>(blocks, 1));
  };
  const auto incoming_pairs = checked_mul(plan.num_experts, plan.world_size,
                                          "incoming expert/source pairs");
  if (incoming_pairs > std::numeric_limits<unsigned int>::max())
    invalid("phase=device-materialize, field=incoming_grid, rank=" +
            std::to_string(rank) + ", expected<=UINT_MAX, actual=" +
            std::to_string(incoming_pairs));

  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_start, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(prefix planning start)");
#if MEGAMOE_DEBUG_VALIDATE
  materialize_prefix_plan_kernel<<<1, 1, 0, exec->stream>>>(args);
#else
  materialize_plan_init_kernel<<<1, 1, 0, exec->stream>>>(args);
  materialize_expert_totals_kernel<<<
      blocks_for(plan.num_experts, "expert_total_grid"), threads, 0,
      exec->stream>>>(args);
  materialize_expert_prefix_kernel<<<
      plan.num_experts, threads, 0,
      exec->stream>>>(args);
  materialize_destination_capacity_kernel<<<
      plan.world_size, threads, 0,
      exec->stream>>>(args);
  materialize_plan_finalize_kernel<<<1, threads, 0, exec->stream>>>(args);
#endif
  materialize_packet_plan_kernel<<<plan.num_nodes, threads, 0, exec->stream>>>(
      args);
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->planning_complete, exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(prefix planning complete)");
  materialize_routes_kernel<<<blocks_for(plan.route_count, "route_grid"),
                              threads, 0, exec->stream>>>(args);
  materialize_incoming_kernel<<<
      static_cast<unsigned int>(std::max<std::uint64_t>(incoming_pairs, 1)),
      threads, 0, exec->stream>>>(args);
  materialize_mtiles_kernel<<<plan.num_experts, threads, 0, exec->stream>>>(
      args);
  materialize_fanout_kernel<<<blocks_for(plan.route_count, "packet_grid"),
                              threads, 0, exec->stream>>>(args);
  materialize_signals_kernel<<<blocks_for(plan.num_nodes, "signal_grid"),
                               threads, 0, exec->stream>>>(args);
  internal::throw_cuda("device-materialize", cudaGetLastError(), rank,
                       context.session().world_size(),
                       context.session().identity(),
                       context.session().generation(),
                       ExchangeKind::ExpertCounts,
                       "parallel materialization kernel launch");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->mapping_complete, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(materialization complete)");
#if MEGAMOE_DEBUG_VALIDATE
  internal::throw_cuda(
      "device-materialize",
      cudaMemcpyAsync(at<void>(state->allocation->host_ptr,
                               state->host_layout.materialization_status),
                      args.status, sizeof(DeviceMaterializationStatus),
                      cudaMemcpyDeviceToHost, exec->stream),
      rank, context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "D2H bounded materialization status");
  internal::record_cuda_api(
      internal::CudaApiAuditKind::DeviceToHost,
      sizeof(DeviceMaterializationStatus));
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->completion, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord");
  internal::throw_cuda(
      "device-materialize",
      cudaEventSynchronize(state->allocation->completion), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventSynchronize");
#else
  auto *mapped_status = at<DeviceMaterializationStatus>(
      state->allocation->host_device_ptr,
      state->host_layout.materialization_status);
  auto *mapped_ready = at<std::uint32_t>(
      state->allocation->host_device_ptr,
      state->host_layout.materialization_ready);
  publish_materialization_status_kernel<<<1, 1, 0, exec->stream>>>(
      args.status, mapped_status, mapped_ready);
  internal::throw_cuda(
      "device-materialize", cudaGetLastError(), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "publish_materialization_status_kernel");
  internal::throw_cuda(
      "device-materialize",
      cudaEventRecord(state->allocation->completion, exec->stream), rank,
      context.session().world_size(), context.session().identity(),
      context.session().generation(), ExchangeKind::ExpertCounts,
      "cudaEventRecord(mapped materialization status)");
  std::uint64_t polls = 0;
  while (__atomic_load_n(host_ready, __ATOMIC_ACQUIRE) != 1U) {
    if ((++polls & 0x3ffU) == 0U) {
      const auto event = cudaEventQuery(state->allocation->completion);
      if (event != cudaSuccess && event != cudaErrorNotReady)
        internal::throw_cuda(
            "device-materialize", event, rank,
            context.session().world_size(), context.session().identity(),
            context.session().generation(), ExchangeKind::ExpertCounts,
            "cudaEventQuery(mapped materialization status)");
      if (event == cudaSuccess &&
          __atomic_load_n(host_ready, __ATOMIC_ACQUIRE) != 1U)
        invalid("phase=device-materialize, field=mapped_status_ready, rank=" +
                std::to_string(rank) +
                ", expected=1 before completion, actual=0");
    }
  }
#endif
  if (timings) {
    float planning_ms = 0;
    float remaining_ms = 0;
    internal::throw_cuda(
        "device-materialize",
        cudaEventElapsedTime(&planning_ms, state->allocation->planning_start,
                             state->allocation->planning_complete),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(prefix planning)");
    internal::throw_cuda(
        "device-materialize",
        cudaEventElapsedTime(&remaining_ms,
                             state->allocation->planning_complete,
                             state->allocation->mapping_complete),
        rank, context.session().world_size(), context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaEventElapsedTime(remaining materialization)");
    timings->prefix_planning_us = static_cast<double>(planning_ms) * 1000.0;
    timings->remaining_parallel_us =
        static_cast<double>(remaining_ms) * 1000.0;
  }
  const auto *status = at<const DeviceMaterializationStatus>(
      state->allocation->host_ptr,
      state->host_layout.materialization_status);
  if (status->error != 0) {
    auto code = NotifyErrorCode::CountExchangeFailure;
    if (status->error ==
        static_cast<unsigned long long>(MaterializationError::ExpertCapacity))
      code = NotifyErrorCode::ExpertCapacityExceeded;
    if (status->error == static_cast<unsigned long long>(
                             MaterializationError::WorkspaceCapacity))
      code = NotifyErrorCode::WorkspaceCapacityExceeded;
    if (status->error == static_cast<unsigned long long>(
                             MaterializationError::ArithmeticOverflow))
      code = NotifyErrorCode::ArithmeticOverflow;
    throw NotifyDispatchError(
        code, "phase=device-materialize, field=" +
                  std::to_string(status->field) + ", rank=" +
                  std::to_string(rank) + ", expected=" +
                  std::to_string(status->expected) + ", actual=" +
                  std::to_string(status->actual));
  }

  auto impl = std::make_shared<DeviceDispatchHandle::Impl>();
  impl->config = config;
  impl->context = context;
  impl->execution = execution;
  impl->workspace = state;
  impl->expert_completion = std::move(expert_counts);
  impl->summary_completion = std::move(packet_summary);
  impl->device = local.impl_->device;
  impl->source_routes = local.impl_->routes;
  impl->source_gates = local.impl_->gates;
  impl->device.source_expert_count =
      impl->expert_completion.device_result();
  impl->device.packet_summary_count =
      impl->summary_completion.device_result();
  impl->device.owned_expert_ids =
      make_lease(state, dl.owned_expert_ids, status->owned_experts);
  impl->device.expert_row_count =
      make_lease(state, dl.owned_expert_count, status->owned_experts);
  impl->device.expert_row_prefix =
      make_lease(state, dl.owned_expert_prefix, status->owned_experts + 1);
  impl->device.mtile_row_offsets =
      make_lease(state, dl.mtile_offsets, status->owned_experts + 1);
  impl->device.mtile_expected_rows =
      make_lease(state, dl.mtile_rows, status->mtile_count);
  impl->device.incoming_rows =
      make_lease(state, dl.incoming_rows, status->incoming_rows);
  impl->device.reverse_routes =
      make_lease(state, dl.reverse_routes, plan.route_count);
  impl->device.packet_headers =
      make_lease(state, dl.packet_headers, status->packet_count);
  impl->device.packet_fanout_offsets =
      make_lease(state, dl.packet_fanout_offsets, status->packet_count + 1);
  impl->device.packet_fanout =
      make_lease(state, dl.packet_fanout, status->fanout_count);
  impl->device.outgoing_signals =
      make_lease(state, dl.outgoing_signals, plan.num_nodes);
  impl->device.incoming_proxy_signals =
      make_lease(state, dl.incoming_signals, plan.num_nodes);
  impl->identity.generation = context.session().generation();
  impl->identity.calling_rank = rank;
  impl->identity.world_size = plan.world_size;
  impl->identity.num_tokens = plan.num_tokens;
  impl->identity.top_k = plan.top_k;
  impl->identity.num_experts = plan.num_experts;
  impl->identity.num_nodes = plan.num_nodes;
  const auto wire = encode_canonical_config(
      config, plan.max_tokens_per_rank,
      context.session().stable_identity_bytes());
  const auto config_identity = stable_session_id(wire);
  impl->identity.config_digest_word0 = config_identity.word0;
  impl->identity.config_digest_word1 = config_identity.word1;
  impl->sizes.route_count = plan.route_count;
  impl->sizes.node_mask_words = plan.node_mask_words;
  impl->sizes.owned_expert_count =
      static_cast<std::uint32_t>(status->owned_experts);
  impl->sizes.incoming_row_count = status->incoming_rows;
  impl->sizes.mtile_count = status->mtile_count;
  impl->sizes.packet_count = status->packet_count;
  impl->sizes.fanout_count = status->fanout_count;
  impl->sizes.outgoing_signal_count = plan.num_nodes;
  impl->sizes.incoming_signal_count = plan.num_nodes;
  return DeviceDispatchHandle(std::move(impl));
}

DeviceDispatchHandle testing::materialize_fake_device_collectives(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution,
    const RankRoutingInput &local_routes,
    std::uint64_t max_tokens_per_rank,
    const std::vector<std::uint64_t> &expert_count_matrix,
    const std::vector<std::uint64_t> &packet_summary_matrix) {
  if (!context || !execution ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=fake-device-materializer, field=context, rank=n/a, "
            "expected=valid exact execution, actual=mismatch");
  const auto world = config.topology.world_size;
  const auto count_elements = checked_mul(world, config.num_experts,
                                          "fake count matrix elements");
  const auto summary_elements = checked_mul(
      world, config.topology.num_nodes, "fake summary matrix elements");
  if (expert_count_matrix.size() != checked_size(count_elements, "counts") ||
      packet_summary_matrix.size() !=
          checked_size(summary_elements, "summaries"))
    invalid("phase=fake-device-materializer, field=matrix_shape, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=exact initialized matrices, actual=mismatch");

  auto routes = upload_cuda_routes(local_routes, config.top_k, context,
                                   execution);
  std::vector<float> unit_gates(
      checked_size(routes.route_count(), "fake materializer gate count"),
      1.0F);
  auto gates = testing::upload_fp32_gates_for_testing(
      unit_gates, routes, context, execution);
  const auto plan = query_cuda_notify_workspace(
      config, context.calling_rank(), local_routes.num_tokens,
      max_tokens_per_rank);
  auto workspace = allocate_cuda_notify_workspace(plan, context, execution);
  auto operation = start_cuda_rank_local_notify(
      config, context, execution, routes, gates, std::move(workspace));
  auto preflight = operation.wait_preflight();
  if (!preflight.valid())
    invalid("phase=fake-device-materializer, field=preflight, rank=" +
            std::to_string(context.calling_rank()) +
            ", expected=valid, actual=failed");
  const auto exec = internal::checked_execution_state(execution);

  const auto make_matrix = [&](const std::vector<std::uint64_t> &host,
                               std::uint64_t rows, std::uint64_t width,
                               const char *field) {
    auto allocation = std::make_shared<internal::CudaAllocation>();
    allocation->device = exec->device;
    allocation->device_bytes = checked_size(
        checked_mul(host.size(), sizeof(std::uint64_t), field), field);
    internal::throw_cuda(
        "fake-device-materializer",
        cudaMalloc(&allocation->device_ptr,
                   std::max<std::size_t>(allocation->device_bytes, 1)),
        context.calling_rank(), world, context.session().identity(),
        context.session().generation(), ExchangeKind::ExpertCounts,
        "cudaMalloc(fake initialized matrix)");
    internal::record_cuda_api(internal::CudaApiAuditKind::DeviceAllocate);
    if (!host.empty()) {
      internal::throw_cuda(
          "fake-device-materializer",
          cudaMemcpyAsync(allocation->device_ptr, host.data(),
                          allocation->device_bytes, cudaMemcpyHostToDevice,
                          exec->stream),
          context.calling_rank(), world, context.session().identity(),
          context.session().generation(), ExchangeKind::ExpertCounts,
          "cudaMemcpyAsync(fake initialized matrix)");
      internal::record_cuda_api(internal::CudaApiAuditKind::HostToDevice,
                                allocation->device_bytes);
    }
    auto buffer = std::make_shared<internal::CudaBufferState>();
    buffer->owner = allocation;
    buffer->pointer = allocation->device_ptr;
    buffer->device = exec->device;
    auto lease = std::make_shared<internal::CudaCompletionLease>();
    lease->owner = allocation;
    return std::make_pair(
        make_device_buffer_lease(rows, width, host.size(), std::move(buffer)),
        std::static_pointer_cast<const BackendCompletionLeaseState>(lease));
  };

  auto count = make_matrix(expert_count_matrix, world, config.num_experts,
                           "fake count matrix bytes");
  auto summary = make_matrix(packet_summary_matrix, world,
                             config.topology.num_nodes,
                             "fake summary matrix bytes");
  auto count_completion = make_device_backend_completion(
      context, ExchangeKind::ExpertCounts, std::move(count.first),
      std::move(count.second));
  auto summary_completion = make_device_backend_completion(
      context, ExchangeKind::NotifyPacketSummary, std::move(summary.first),
      std::move(summary.second));
  return DeviceMetadataBridge::materialize(
      preflight.local(), std::move(count_completion),
      std::move(summary_completion), execution);
}

const DeviceDispatchIdentity &DeviceDispatchHandle::identity() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle identity");
  return impl_->identity;
}

const DeviceDispatchSizes &DeviceDispatchHandle::sizes() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle sizes");
  return impl_->sizes;
}

const DeviceMaterializationAudit &DeviceDispatchHandle::audit() const {
  if (!impl_)
    invalid("attempted to access empty device dispatch handle audit");
  return impl_->audit;
}

DeviceDispatchConsumerLease DeviceDispatchHandle::acquire_consumer(
    const CudaExecution &execution,
    const DeviceDispatchViewExpectation &expected) const {
  if (!impl_ || !execution)
    invalid("phase=device-view, field=handle/execution, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &id = impl_->identity;
  if (id.generation != expected.generation ||
      id.calling_rank != expected.calling_rank ||
      id.world_size != expected.world_size ||
      id.num_tokens != expected.num_tokens || id.top_k != expected.top_k)
    invalid("phase=device-view, field=identity/shape/epoch, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=exact handle provenance, actual=mismatch");
  if (id.config_digest_word0 != expected.config_digest_word0 ||
      id.config_digest_word1 != expected.config_digest_word1)
    invalid("phase=device-view, field=config_identity, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=exact handle config digest, actual=mismatch");
  auto exec = internal::checked_execution_state(execution);
  if (exec->device != impl_->execution.device_ordinal())
    invalid("phase=dispatch-consumer-acquire, field=device, rank=" +
            std::to_string(id.calling_rank) +
            ", expected=handle device, actual=foreign device");
  auto workspace = impl_->workspace;
  std::size_t event_index = SIZE_MAX;
  {
    std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
    if (workspace->allocation->dispatch_slot_quarantined.load(
            std::memory_order_acquire))
      invalid("phase=dispatch-consumer-acquire, field=slot_state, rank=" +
              std::to_string(id.calling_rank) +
              ", expected=usable, actual=Quarantined");
    for (std::size_t i = 0;
         i < workspace->allocation->consumer_event_states.size(); ++i) {
      if (workspace->allocation->consumer_event_states[i] ==
          internal::ConsumerEventSlotState::Available) {
        workspace->allocation->consumer_event_states[i] =
            internal::ConsumerEventSlotState::InUse;
        event_index = i;
        break;
      }
    }
  }
  if (event_index == SIZE_MAX)
    throw NotifyDispatchError(
        NotifyErrorCode::ResourceExhausted,
        "phase=dispatch-consumer-acquire, field=consumer_events, rank=" +
            std::to_string(id.calling_rank) +
            ", expected>=1 available pooled event, actual=0");
  try {
    internal::throw_cuda(
        "dispatch-consumer-acquire",
        cudaStreamWaitEvent(exec->stream, workspace->allocation->completion),
        id.calling_rank, id.world_size, impl_->context.session().identity(),
        id.generation, ExchangeKind::ExpertCounts,
        "cudaStreamWaitEvent(notify completion)");
  } catch (...) {
    std::lock_guard<std::mutex> lock(workspace->allocation->consumer_mutex);
    workspace->allocation->consumer_event_states[event_index] =
        internal::ConsumerEventSlotState::Available;
    throw;
  }
  auto consumer = std::make_shared<DeviceDispatchConsumerLease::Impl>();
  consumer->owner = *this;
  consumer->workspace = impl_->workspace;
  consumer->device = impl_->device;
  consumer->identity = impl_->identity;
  consumer->sizes = impl_->sizes;
  consumer->context = impl_->context;
  consumer->epoch = impl_->workspace->epoch;
  consumer->execution = execution;
  consumer->event_index = event_index;
  return DeviceDispatchConsumerLease(std::move(consumer));
}

DeviceDispatchConsumerLease::~DeviceDispatchConsumerLease() noexcept {
  if (impl_)
    impl_->release_noexcept();
}
DeviceDispatchConsumerLease::DeviceDispatchConsumerLease(
    DeviceDispatchConsumerLease &&other) noexcept = default;
DeviceDispatchConsumerLease &DeviceDispatchConsumerLease::operator=(
    DeviceDispatchConsumerLease &&other) noexcept {
  if (this != &other) {
    if (impl_)
      impl_->release_noexcept();
    impl_ = std::move(other.impl_);
  }
  return *this;
}
DeviceDispatchConsumerState DeviceDispatchConsumerLease::state() const {
  if (!impl_)
    invalid("phase=dispatch-consumer-state, field=lease, rank=n/a, "
            "expected=valid, actual=empty/moved-from");
  return impl_->state;
}
void DeviceDispatchConsumerLease::finish() {
  if (!impl_ || impl_->state == DeviceDispatchConsumerState::Released)
    invalid("phase=dispatch-consumer-finish, field=state, rank=n/a, "
            "expected=acquired/submitted, actual=empty/released");
  if (impl_->state == DeviceDispatchConsumerState::Submitted) {
    const auto &id = impl_->identity;
    if (!impl_->drain_submitted_noexcept())
      throw NotifyDispatchError(
          NotifyErrorCode::CudaFailure,
          "phase=dispatch-consumer-finish, field=consumer_event, rank=" +
              std::to_string(id.calling_rank) +
              ", expected=synchronized, actual=CUDA event failure; event quarantined");
    return;
  }
  impl_->release_noexcept();
}

DeviceDispatchKernelView
ActivationTransportBridge::view(DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_ ||
      consumer.impl_->state != DeviceDispatchConsumerState::Acquired)
    invalid("phase=activation-transport-view, field=consumer, rank=n/a, "
            "expected=Acquired, actual=invalid");
  return consumer.impl_->kernel_view();
}

const float *ActivationTransportBridge::gate_pointer(
    const DeviceDispatchHandle &dispatch) {
  if (!dispatch.impl_ || !dispatch.impl_->source_gates)
    invalid("phase=activation-gate-view, field=dispatch/gates, rank=n/a, "
            "expected=valid owning handle, actual=empty");
  DeviceGateProvenanceBridge::validate(
      dispatch.impl_->source_gates, dispatch.impl_->source_routes,
      dispatch.impl_->context, dispatch.impl_->execution);
  return DeviceGateProvenanceBridge::device_pointer(
      dispatch.impl_->source_gates);
}

void ActivationTransportBridge::wait_for_gates(
    const DeviceDispatchHandle &dispatch, const CudaExecution &execution) {
  if (!dispatch.impl_ || !dispatch.impl_->source_gates.impl_)
    invalid("phase=activation-gate-wait, field=dispatch/gates, rank=n/a, "
            "expected=valid owning handle, actual=empty");
  const auto &gate = *dispatch.impl_->source_gates.impl_;
  const auto exec = internal::checked_execution_state(execution);
  if (gate.device != exec->device || !gate.ready_recorded ||
      !gate.allocation || !gate.bound_routes.impl_ ||
      !gate.bound_routes.impl_->allocation ||
      !gate.bound_routes.impl_->allocation->producer_ready)
    invalid("phase=activation-gate-wait, field=device/ready, rank=" +
            std::to_string(dispatch.impl_->identity.calling_rank) +
            ", expected=matching recorded producer event, actual=mismatch");
  internal::throw_cuda(
      "activation-gate-wait",
      cudaStreamWaitEvent(exec->stream,
                          gate.bound_routes.impl_->allocation->producer_ready),
      dispatch.impl_->identity.calling_rank,
      dispatch.impl_->identity.world_size,
      dispatch.impl_->context.session().identity(),
      dispatch.impl_->identity.generation, ExchangeKind::ExpertCounts,
      "cudaStreamWaitEvent(gate producer ready)");
}

void ActivationTransportBridge::submit(
    DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_)
    invalid("phase=activation-transport-submit, field=consumer, rank=n/a, "
            "expected=valid, actual=empty");
  consumer.impl_->record_completion();
}

const RankCollectiveContext &ActivationTransportBridge::context(
    const DeviceDispatchConsumerLease &consumer) {
  if (!consumer.impl_)
    invalid("phase=activation-transport-context, field=consumer, rank=n/a, "
            "expected=valid, actual=empty");
  return consumer.impl_->context;
}

class DeviceDispatchTestBridge {
public:
  static void submit(DeviceDispatchConsumerLease &consumer,
                     std::uint64_t delay_cycles) {
    if (!consumer.impl_ ||
        consumer.impl_->state != DeviceDispatchConsumerState::Acquired)
      invalid("phase=dispatch-test-submit, field=consumer_state, rank=n/a, "
              "expected=Acquired, actual=invalid");
    auto &impl = *consumer.impl_;
    const auto view = impl.kernel_view();
    if (view.sizes.route_count == 0)
      invalid("phase=dispatch-test-submit, field=route_count, rank=" +
              std::to_string(view.identity.calling_rank) +
              ", expected>=1, actual=0");
    auto exec = internal::checked_execution_state(impl.execution);
    auto workspace = impl.workspace;
    auto *destination = at<void>(workspace->allocation->host_ptr,
                                 workspace->host_layout.materialization_status +
                                     impl.event_index * sizeof(std::uint64_t));
    if (workspace->host_layout.materialization_status +
            (impl.event_index + 1U) * sizeof(std::uint64_t) >
        workspace->host_layout.total)
      invalid("phase=dispatch-test-submit, field=probe_staging, rank=" +
              std::to_string(view.identity.calling_rank) +
              ", expected=pooled consumer staging, actual=short");
    try {
      dispatch_consumer_probe_delay_kernel<<<1, 1, 0, exec->stream>>>(
          view.route_dst_row, delay_cycles);
      internal::throw_cuda(
          "dispatch-test-submit", cudaGetLastError(),
          view.identity.calling_rank, view.identity.world_size,
          impl.context.session().identity(), view.identity.generation,
          ExchangeKind::ExpertCounts, "consumer probe kernel launch");
      internal::throw_cuda(
          "dispatch-test-submit",
          cudaMemcpyAsync(destination, view.route_dst_row,
                          sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                          exec->stream),
          view.identity.calling_rank, view.identity.world_size,
          impl.context.session().identity(), view.identity.generation,
          ExchangeKind::ExpertCounts, "D2H consumer probe");
      internal::record_cuda_api(internal::CudaApiAuditKind::DeviceToHost,
                                sizeof(std::uint64_t));
      impl.record_completion();
    } catch (...) {
      if (impl.owner) {
        const bool stream_ok = cudaStreamSynchronize(exec->stream) == cudaSuccess;
        if (!stream_ok) {
          std::lock_guard<std::mutex> lock(
              workspace->allocation->consumer_mutex);
          workspace->allocation->dispatch_slot_quarantined.store(
              true, std::memory_order_release);
        }
        impl.release_owner_noexcept(false);
      }
      throw;
    }
  }

  static std::uint64_t finish(DeviceDispatchConsumerLease &consumer) {
    if (!consumer.impl_ ||
        consumer.impl_->state != DeviceDispatchConsumerState::Submitted)
      invalid("phase=dispatch-test-finish, field=consumer_state, rank=n/a, "
              "expected=Submitted, actual=invalid");
    auto workspace = consumer.impl_->workspace;
    consumer.finish();
    return *at<const std::uint64_t>(
        workspace->allocation->host_ptr,
        workspace->host_layout.materialization_status +
            consumer.impl_->event_index * sizeof(std::uint64_t));
  }
};

namespace testing {
void submit_dispatch_probe(DeviceDispatchConsumerLease &consumer,
                           std::uint64_t delay_cycles) {
  DeviceDispatchTestBridge::submit(consumer, delay_cycles);
}
std::uint64_t finish_dispatch_probe(DeviceDispatchConsumerLease &consumer) {
  return DeviceDispatchTestBridge::finish(consumer);
}
} // namespace testing

DispatchHandle download_device_metadata_for_validation(
    const DeviceDispatchHandle &handle, const CudaExecution &execution) {
  if (!handle || !execution)
    invalid("phase=device-validation-download, field=input, rank=n/a, "
            "expected=valid, actual=empty");
  const auto &impl = *handle.impl_;
  const auto expectation = DeviceDispatchViewExpectation{
      impl.identity.generation, impl.identity.calling_rank,
      impl.identity.world_size, impl.identity.num_tokens, impl.identity.top_k,
      impl.identity.config_digest_word0, impl.identity.config_digest_word1};
  auto consumer = handle.acquire_consumer(execution, expectation);
  const auto view = consumer.impl_->kernel_view();
  auto exec = internal::checked_execution_state(execution);
  const auto state = impl.workspace;
  auto summary_state = internal::CudaDeviceLeaseAccess::checked(
      impl.device.packet_summary_count, exec->device);
  const auto routes = impl.sizes.route_count;
  SourceRouteMetadata outgoing;
  outgoing.num_tokens = impl.identity.num_tokens;
  outgoing.top_k = impl.identity.top_k;
  outgoing.node_mask_words = impl.sizes.node_mask_words;
  outgoing.expert_id = download_vector<std::uint32_t>(
      state, *exec, view.expert_id, routes, "expert_id");
  outgoing.local_offset = download_vector<std::uint64_t>(
      state, *exec, view.local_offset, routes, "local_offset");
  outgoing.route_dst_row = download_vector<std::uint64_t>(
      state, *exec, view.route_dst_row, routes, "route_dst_row");
  outgoing.route_dst_rank = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_rank, routes, "route_dst_rank");
  outgoing.route_dst_node = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_node, routes, "route_dst_node");
  outgoing.route_dst_local_rank = download_vector<std::uint32_t>(
      state, *exec, view.route_dst_local_rank, routes,
      "route_dst_local_rank");
  outgoing.combine_order = download_vector<std::uint32_t>(
      state, *exec, view.combine_order, routes, "combine_order");
  outgoing.destination_node_mask = download_vector<std::uint64_t>(
      state, *exec, view.destination_node_mask,
      impl.identity.num_tokens * impl.sizes.node_mask_words,
      "destination_node_mask");
  const auto reverse = download_vector<DeviceReverseRouteRecord>(
      state, *exec, view.reverse_routes, routes, "reverse_routes");
  outgoing.reverse_routes.reserve(reverse.size());
  for (const auto &record : reverse)
    outgoing.reverse_routes.push_back(ReverseRouteMetadata{
        record.source_rank, record.local_token_id, record.topk_slot,
        record.expert_id, record.destination_rank, record.dst_row});

  const auto flat_counts = download_vector<std::uint64_t>(
      state, *exec, view.source_expert_count,
      static_cast<std::uint64_t>(impl.identity.world_size) *
          impl.identity.num_experts,
      "source_expert_count");
  const auto flat_summary = download_vector<std::uint64_t>(
      state, *exec, summary_state->pointer,
      static_cast<std::uint64_t>(impl.identity.world_size) *
          impl.identity.num_nodes,
      "packet_summary_count");
  CountMatrix counts(impl.identity.world_size,
                     std::vector<std::uint64_t>(impl.identity.num_experts));
  CountMatrix summaries(impl.identity.world_size,
                        std::vector<std::uint64_t>(impl.identity.num_nodes));
  for (std::uint32_t rank = 0; rank < impl.identity.world_size; ++rank) {
    std::copy_n(flat_counts.begin() +
                    static_cast<std::size_t>(rank) * impl.identity.num_experts,
                impl.identity.num_experts, counts[rank].begin());
    std::copy_n(flat_summary.begin() +
                    static_cast<std::size_t>(rank) * impl.identity.num_nodes,
                impl.identity.num_nodes, summaries[rank].begin());
  }

  DispatchMetadata metadata;
  metadata.config = impl.config;
  metadata.calling_rank = impl.identity.calling_rank;
  metadata.communicator_id = impl.context.session().identity();
  metadata.identity.generation = impl.identity.generation;
  metadata.source_expert_count = counts;
  metadata.outgoing = std::move(outgoing);
  auto &incoming = metadata.incoming;
  incoming.rank = impl.identity.calling_rank;
  incoming.expert_ids = download_vector<std::uint32_t>(
      state, *exec, view.owned_expert_ids, impl.sizes.owned_expert_count,
      "owned_expert_ids");
  incoming.expert_row_count = download_vector<std::uint64_t>(
      state, *exec, view.expert_row_count, impl.sizes.owned_expert_count,
      "expert_row_count");
  incoming.expert_row_prefix = download_vector<std::uint64_t>(
      state, *exec, view.expert_row_prefix,
      static_cast<std::uint64_t>(impl.sizes.owned_expert_count) + 1,
      "expert_row_prefix");
  const auto tile_offsets = download_vector<std::uint64_t>(
      state, *exec, view.mtile_row_offsets,
      static_cast<std::uint64_t>(impl.sizes.owned_expert_count) + 1,
      "mtile_row_offsets");
  const auto tiles = download_vector<std::uint32_t>(
      state, *exec, view.mtile_expected_rows, impl.sizes.mtile_count,
      "mtile_expected_rows");
  incoming.mtile_expected_rows.resize(impl.sizes.owned_expert_count);
  for (std::uint32_t expert = 0; expert < impl.sizes.owned_expert_count;
       ++expert) {
    incoming.mtile_expected_rows[expert].assign(
        tiles.begin() + static_cast<std::ptrdiff_t>(tile_offsets[expert]),
        tiles.begin() + static_cast<std::ptrdiff_t>(tile_offsets[expert + 1]));
  }
  const auto rows = download_vector<DeviceIncomingRowRecord>(
      state, *exec, view.incoming_rows, impl.sizes.incoming_row_count,
      "incoming_rows");
  incoming.incoming_rows.reserve(rows.size());
  for (const auto &row : rows)
    incoming.incoming_rows.push_back(IncomingRowMetadata{
        row.source_rank, row.expert_id, row.source_expert_offset, row.dst_row});

  const auto headers = download_vector<DevicePacketHeaderRecord>(
      state, *exec, view.packet_headers, impl.sizes.packet_count,
      "packet_headers");
  const auto fanout_offsets = download_vector<std::uint64_t>(
      state, *exec, view.packet_fanout_offsets, impl.sizes.packet_count + 1,
      "packet_fanout_offsets");
  const auto fanout = download_vector<DeviceFanoutRecord>(
      state, *exec, view.packet_fanout, impl.sizes.fanout_count,
      "packet_fanout");
  for (std::uint64_t packet = 0; packet < impl.sizes.packet_count; ++packet) {
    const auto &header = headers[static_cast<std::size_t>(packet)];
    OutgoingPacketDescriptor descriptor{
        PacketIdentity{header.source_rank, header.destination_node,
                       header.sequence_on_link},
        header.local_token_id, header.proxy_rank, header.channel, {}};
    for (auto index = fanout_offsets[static_cast<std::size_t>(packet)];
         index < fanout_offsets[static_cast<std::size_t>(packet + 1)]; ++index) {
      const auto &target = fanout[static_cast<std::size_t>(index)];
      descriptor.fanout.push_back(FanoutTarget{
          target.topk_slot, target.destination_local_rank, target.expert_id,
          target.dst_row});
    }
    metadata.transport.outgoing_packets.push_back(std::move(descriptor));
  }
  const auto outgoing_signals = download_vector<DeviceSignalRecord>(
      state, *exec, view.outgoing_signals, impl.sizes.outgoing_signal_count,
      "outgoing_signals");
  const auto incoming_signals = download_vector<DeviceSignalRecord>(
      state, *exec, view.incoming_proxy_signals,
      impl.sizes.incoming_signal_count, "incoming_signals");
  const auto append_signals = [](const auto &source, auto &destination) {
    for (const auto &signal : source)
      destination.push_back(SignalCardinality{
          signal.source_node, signal.destination_node, signal.source_rank,
          signal.destination_proxy_rank, signal.rail_local_rank,
          signal.channel, signal.expected_packets,
          signal.expected_signals});
  };
  append_signals(outgoing_signals, metadata.transport.outgoing_signals);
  append_signals(incoming_signals,
                 metadata.transport.incoming_proxy_signals);
  auto result = DeviceNotifyBridge::make_validation_dispatch(
      std::move(metadata));
  validate_dispatch_metadata(result);
  consumer.impl_->record_completion();
  consumer.finish();
  return result;
}

void validate_cuda_dispatch_metadata(const DeviceDispatchHandle &handle,
                                      const CudaExecution &execution) {
  (void)download_device_metadata_for_validation(handle, execution);
}

} // namespace megamoe::cuda
