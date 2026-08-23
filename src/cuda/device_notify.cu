#include "megamoe/cuda/device_notify.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
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
  unsigned long long out_of_range_route;
  unsigned long long duplicate_route;
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
  std::size_t source_expert_count = 0;
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
  layout.source_expert_count = append_region(
      cursor, checked_mul(plan.world_size, plan.num_experts,
                          "source_expert_count_elements"),
      sizeof(std::uint64_t), "source_expert_count");
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
  std::size_t expert_count = 0;
  std::size_t packet_count = 0;
  std::size_t expert_id = 0;
  std::size_t local_offset = 0;
  std::size_t route_dst_rank = 0;
  std::size_t route_dst_node = 0;
  std::size_t route_dst_local_rank = 0;
  std::size_t combine_order = 0;
  std::size_t masks = 0;
  std::size_t status = 0;
  std::size_t total = 0;
};

HostLayout make_host_layout(const CudaNotifyWorkspacePlan &plan) {
  HostLayout layout;
  std::size_t cursor = 0;
  layout.expert_count = append_region(cursor, plan.num_experts, sizeof(std::uint64_t), "host_expert_count");
  layout.packet_count = append_region(cursor, plan.num_nodes, sizeof(std::uint64_t), "host_packet_count");
  layout.expert_id = append_region(cursor, plan.route_count, sizeof(std::uint32_t), "host_expert_id");
  layout.local_offset = append_region(cursor, plan.route_count, sizeof(std::uint64_t), "host_local_offset");
  layout.route_dst_rank = append_region(cursor, plan.route_count, sizeof(std::uint32_t), "host_route_dst_rank");
  layout.route_dst_node = append_region(cursor, plan.route_count, sizeof(std::uint32_t), "host_route_dst_node");
  layout.route_dst_local_rank = append_region(cursor, plan.route_count, sizeof(std::uint32_t), "host_route_dst_local_rank");
  layout.combine_order = append_region(cursor, plan.route_count, sizeof(std::uint32_t), "host_combine_order");
  layout.masks = append_region(cursor, checked_mul(plan.num_tokens, plan.node_mask_words, "host_mask_elements"), sizeof(std::uint64_t), "host_masks");
  layout.status = append_region(cursor, 1, sizeof(DeviceStatus), "host_status");
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
    atomicMin(&status->out_of_range_route,
              static_cast<unsigned long long>(route));
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
        atomicMin(&status->duplicate_route,
                  static_cast<unsigned long long>(base + slot));
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

DeviceBufferLease make_matrix_lease(
    const std::shared_ptr<WorkspaceState> &state, std::size_t offset,
    std::uint32_t rows, std::uint64_t row_width) {
  const auto elements = checked_mul(rows, row_width, "device lease elements");
  if (elements == 0)
    return {};
  auto buffer = std::make_shared<CudaBufferState>();
  buffer->owner = state;
  buffer->pointer = at<void>(state->allocation->device_ptr, offset);
  buffer->device = state->execution.device_ordinal();
  return make_device_buffer_lease(rows, row_width, elements,
                                  std::move(buffer));
}

} // namespace

namespace internal {

CudaExecutionState::~CudaExecutionState() noexcept {
  if (!stream)
    return;
  (void)cudaSetDevice(static_cast<int>(device));
  (void)cudaStreamSynchronize(stream);
  (void)cudaStreamDestroy(stream);
}

CudaAllocation::~CudaAllocation() noexcept {
  (void)cudaSetDevice(static_cast<int>(device));
  if (completion)
    (void)cudaEventSynchronize(completion);
  if (completion)
    (void)cudaEventDestroy(completion);
  if (host_ptr)
    (void)cudaFreeHost(host_ptr);
  if (device_ptr)
    (void)cudaFree(device_ptr);
}

void throw_cuda(const char *phase, cudaError_t error, std::uint32_t rank,
                std::uint32_t world, const std::string &session,
                std::uint64_t generation, ExchangeKind kind,
                const char *operation) {
  if (error == cudaSuccess)
    return;
  std::ostringstream stream;
  stream << "phase=" << phase << ", kind="
         << (kind == ExchangeKind::ExpertCounts ? "expert_counts"
                                                : "notify_packet_summary")
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
std::shared_ptr<const CudaExecutionState>
checked_execution_state(const CudaExecution &execution) {
  if (!execution)
    invalid("phase=cuda-context, field=execution, rank=n/a, expected=valid, actual=empty");
  return execution.impl_->state;
}
} // namespace internal

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

struct CudaRouteInput::Impl {
  std::shared_ptr<CudaAllocation> allocation;
  ExecutionToken execution;
  std::uint64_t tokens = 0;
  std::uint32_t top_k = 0;
  std::uint64_t routes = 0;
};

CudaRouteInput upload_cuda_routes(const RankRoutingInput &routes,
                                  std::uint32_t top_k,
                                  const CudaExecution &execution) {
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
  if (count != 0) {
    internal::throw_cuda(
        "route-upload",
        cudaMemcpyAsync(allocation->device_ptr, routes.selected_experts.data(),
                        route_bytes,
                        cudaMemcpyHostToDevice, state->stream),
        0, 1, execution.token().identity(), 1, ExchangeKind::ExpertCounts,
        "cudaMemcpyAsync(route input)");
  }
  auto impl = std::make_shared<CudaRouteInput::Impl>();
  impl->allocation = std::move(allocation);
  impl->execution = execution.token();
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
  internal::throw_cuda("workspace-allocate",
                       cudaHostAlloc(&allocation.host_ptr,
                                     std::max<std::size_t>(allocation.host_bytes, 1),
                                     cudaHostAllocPortable),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaHostAlloc(staging)");
  internal::throw_cuda("workspace-allocate",
                       cudaEventCreateWithFlags(&allocation.completion,
                                                cudaEventDisableTiming),
                       context.calling_rank(), context.session().world_size(),
                       context.session().identity(), context.session().generation(),
                       ExchangeKind::ExpertCounts, "cudaEventCreate");
  auto impl = std::make_shared<CudaNotifyWorkspace::Impl>();
  impl->state = std::move(state);
  return CudaNotifyWorkspace(std::move(impl));
}

struct CudaLocalNotifyHandle::Impl {
  LocalHistogramHandle local;
  DeviceDispatchMetadata device;
  std::shared_ptr<WorkspaceState> workspace;
};

const LocalHistogramHandle &CudaLocalNotifyHandle::local() const {
  if (!impl_)
    invalid("attempted to access empty CUDA local notify handle");
  return impl_->local;
}

struct CudaNotifyOperation::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  CudaExecution execution;
  CudaRouteInput routes;
  std::shared_ptr<WorkspaceState> workspace;
  bool terminal = false;
  bool waited = false;
};

CudaNotifyOperation start_cuda_rank_local_notify(
    const RuntimeConfig &config, const RankCollectiveContext &context,
    const CudaExecution &execution, const CudaRouteInput &routes,
    CudaNotifyWorkspace workspace) {
  validate_runtime_config(config);
  if (!context || !execution || !routes || !workspace)
    invalid("phase=cuda-notify-start, field=input, rank=n/a, expected=all handles valid, actual=empty");
  const auto rank = context.calling_rank();
  if (context.session().world_size() != config.topology.world_size ||
      !context.execution().same_instance(execution.token()))
    invalid("phase=cuda-notify-start, field=context, rank=" +
            std::to_string(rank) +
            ", expected=config world and exact execution, actual=mismatch");
  if (!routes.impl_->execution.same_instance(execution.token()))
    invalid("phase=cuda-notify-start, field=route.execution, rank=" +
            std::to_string(rank) + ", expected=exact execution, actual=different");
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
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_node),
                                       config.topology.rank_to_node.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_node)");
  internal::throw_cuda("notify-start",
                       cudaMemcpyAsync(at<void>(base, dl.rank_to_local),
                                       config.topology.rank_to_local_rank.data(),
                                       config.topology.world_size * sizeof(std::uint32_t),
                                       cudaMemcpyHostToDevice, exec->stream),
                       rank, context.session().world_size(), context.session().identity(),
                       context.session().generation(), ExchangeKind::ExpertCounts,
                       "cudaMemcpyAsync(rank_to_local)");
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
        at<const std::int32_t>(routes.impl_->allocation->device_ptr, 0),
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
        at<const std::int32_t>(routes.impl_->allocation->device_ptr, 0),
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
  };
  copy(hl.expert_count, dl.expert_count, config.num_experts * sizeof(std::uint64_t), "D2H expert_count");
  copy(hl.packet_count, dl.packet_count, config.topology.num_nodes * sizeof(std::uint64_t), "D2H packet_count");
  copy(hl.expert_id, dl.expert_id, checked_size(route_count, "routes") * sizeof(std::uint32_t), "D2H expert_id");
  copy(hl.local_offset, dl.local_offset, checked_size(route_count, "routes") * sizeof(std::uint64_t), "D2H local_offset");
  copy(hl.route_dst_rank, dl.route_dst_rank, checked_size(route_count, "routes") * sizeof(std::uint32_t), "D2H dst_rank");
  copy(hl.route_dst_node, dl.route_dst_node, checked_size(route_count, "routes") * sizeof(std::uint32_t), "D2H dst_node");
  copy(hl.route_dst_local_rank, dl.route_dst_local_rank, checked_size(route_count, "routes") * sizeof(std::uint32_t), "D2H dst_local");
  copy(hl.combine_order, dl.combine_order, checked_size(route_count, "routes") * sizeof(std::uint32_t), "D2H combine_order");
  copy(hl.masks, dl.masks, checked_size(checked_mul(state->plan.num_tokens, state->plan.node_mask_words, "masks"), "masks") * sizeof(std::uint64_t), "D2H masks");
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

CudaLocalNotifyHandle CudaNotifyOperation::wait() {
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
  if (status->out_of_range_route != ~0ULL) {
    const auto flat = static_cast<std::uint64_t>(status->out_of_range_route);
    throw NotifyDispatchError(
        NotifyErrorCode::ExpertOutOfRange,
        "phase=cuda-device-validation, field=selected_experts, rank=" +
            std::to_string(rank) + ", expected=[0," +
            std::to_string(impl_->config.num_experts) + "), actual=out-of-range, " +
            route_coordinates(flat, impl_->config.top_k));
  }
  if (status->duplicate_route != ~0ULL) {
    const auto flat = static_cast<std::uint64_t>(status->duplicate_route);
    throw NotifyDispatchError(
        NotifyErrorCode::DuplicateExpert,
        "phase=cuda-device-validation, field=selected_experts, rank=" +
            std::to_string(rank) + ", expected=unique within token, actual=duplicate, " +
            route_coordinates(flat, impl_->config.top_k));
  }
  const auto routes = checked_size(state->plan.route_count, "routes");
  const auto masks = checked_size(
      checked_mul(state->plan.num_tokens, state->plan.node_mask_words, "masks"),
      "masks");
  const auto *host = state->allocation->host_ptr;
  const auto vector_u32 = [&](std::size_t offset, std::size_t count) {
    const auto *first = at<const std::uint32_t>(host, offset);
    return std::vector<std::uint32_t>(first, first + count);
  };
  const auto vector_u64 = [&](std::size_t offset, std::size_t count) {
    const auto *first = at<const std::uint64_t>(host, offset);
    return std::vector<std::uint64_t>(first, first + count);
  };
  SourceRouteMetadata outgoing;
  outgoing.num_tokens = state->plan.num_tokens;
  outgoing.top_k = state->plan.top_k;
  outgoing.node_mask_words = state->plan.node_mask_words;
  outgoing.expert_id = vector_u32(state->host_layout.expert_id, routes);
  outgoing.local_offset = vector_u64(state->host_layout.local_offset, routes);
  outgoing.route_dst_row.assign(routes, 0);
  outgoing.route_dst_rank = vector_u32(state->host_layout.route_dst_rank, routes);
  outgoing.route_dst_node = vector_u32(state->host_layout.route_dst_node, routes);
  outgoing.route_dst_local_rank = vector_u32(state->host_layout.route_dst_local_rank, routes);
  outgoing.combine_order = vector_u32(state->host_layout.combine_order, routes);
  outgoing.destination_node_mask = vector_u64(state->host_layout.masks, masks);
  auto expert_counts = vector_u64(state->host_layout.expert_count,
                                  state->plan.num_experts);
  auto packet_counts = vector_u64(state->host_layout.packet_count,
                                  state->plan.num_nodes);
  auto device_expert = make_lease(state, state->device_layout.expert_count,
                                  state->plan.num_experts);
  auto device_packet = make_lease(state, state->device_layout.packet_count,
                                  state->plan.num_nodes);
  auto local = DeviceNotifyBridge::make_local(
      impl_->config, impl_->context, state->plan.num_tokens,
      std::move(outgoing), std::move(expert_counts), std::move(packet_counts),
      device_expert, device_packet);
  auto result = std::make_shared<CudaLocalNotifyHandle::Impl>();
  result->local = std::move(local);
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
  return CudaLocalNotifyHandle(std::move(result));
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

struct CudaDispatchHandle::Impl {
  DispatchHandle host;
  DeviceDispatchMetadata device;
  std::shared_ptr<WorkspaceState> workspace;
};

const DispatchHandle &CudaDispatchHandle::host() const {
  if (!impl_)
    invalid("attempted to access empty CUDA dispatch handle");
  return impl_->host;
}
const DeviceDispatchMetadata &CudaDispatchHandle::device() const {
  if (!impl_)
    invalid("attempted to access empty CUDA dispatch handle");
  return impl_->device;
}

CudaDispatchHandle stage_cuda_dispatch_metadata(
    DispatchHandle host_handle, const CudaLocalNotifyHandle &local,
    const CudaExecution &execution) {
  if (!host_handle || !local || !execution)
    invalid("phase=cuda-dispatch-stage, field=input, rank=n/a, expected=valid, actual=empty");
  validate_dispatch_metadata(host_handle);
  const auto &local_context = local.impl_->local.context();
  if (!local_context.execution().same_instance(execution.token()))
    invalid("phase=cuda-dispatch-stage, field=execution, rank=" +
            std::to_string(host_handle->calling_rank) +
            ", expected=local execution, actual=different");
  if (host_handle->calling_rank != local_context.calling_rank())
    invalid("phase=cuda-dispatch-stage, field=calling_rank, rank=" +
            std::to_string(local_context.calling_rank()) + ", expected=" +
            std::to_string(local_context.calling_rank()) + ", actual=" +
            std::to_string(host_handle->calling_rank));
  if (host_handle->communicator_id != local_context.session().identity())
    invalid("phase=cuda-dispatch-stage, field=session, rank=" +
            std::to_string(host_handle->calling_rank) + ", expected='" +
            local_context.session().identity() + "', actual='" +
            host_handle->communicator_id + "'");
  if (host_handle->identity.generation !=
      local_context.session().generation())
    invalid("phase=cuda-dispatch-stage, field=generation, rank=" +
            std::to_string(host_handle->calling_rank) + ", expected=" +
            std::to_string(local_context.session().generation()) +
            ", actual=" + std::to_string(host_handle->identity.generation));
  const auto &local_config = local.impl_->local.config();
  const auto &host_config = host_handle->config;
  if (host_config.num_experts != local_config.num_experts ||
      host_config.top_k != local_config.top_k ||
      host_config.m_tile_rows != local_config.m_tile_rows ||
      host_config.topology.world_size != local_config.topology.world_size ||
      host_config.topology.num_nodes != local_config.topology.num_nodes ||
      host_config.topology.rank_to_node !=
          local_config.topology.rank_to_node ||
      host_config.topology.rank_to_local_rank !=
          local_config.topology.rank_to_local_rank ||
      host_config.topology.expert_to_rank !=
          local_config.topology.expert_to_rank ||
      host_config.capacity.expert_rows != local_config.capacity.expert_rows ||
      host_config.capacity.workspace_rows_per_rank !=
          local_config.capacity.workspace_rows_per_rank)
    invalid("phase=cuda-dispatch-stage, field=config, rank=" +
            std::to_string(host_handle->calling_rank) +
            ", expected=exact local runtime config, actual=mismatch");
  const auto &local_out = local.impl_->local.outgoing();
  const auto &host_out = host_handle->outgoing;
  if (host_out.num_tokens != local_out.num_tokens ||
      host_out.top_k != local_out.top_k ||
      host_out.node_mask_words != local_out.node_mask_words ||
      host_out.expert_id != local_out.expert_id ||
      host_out.local_offset != local_out.local_offset ||
      host_out.route_dst_rank != local_out.route_dst_rank ||
      host_out.route_dst_node != local_out.route_dst_node ||
      host_out.route_dst_local_rank != local_out.route_dst_local_rank ||
      host_out.combine_order != local_out.combine_order ||
      host_out.destination_node_mask != local_out.destination_node_mask)
    invalid("phase=cuda-dispatch-stage, field=local_provenance, rank=" +
            std::to_string(host_handle->calling_rank) +
            ", expected=metadata materialized from exact local handle, "
            "actual=route content mismatch");
  const auto &rows = host_handle->outgoing.route_dst_row;
  auto state = local.impl_->workspace;
  if (rows.size() != state->plan.route_count)
    invalid("phase=cuda-dispatch-stage, field=route_dst_row.size, rank=" +
            std::to_string(host_handle->calling_rank) + ", expected=" +
            std::to_string(state->plan.route_count) + ", actual=" +
            std::to_string(rows.size()));
  const auto &metadata = host_handle.metadata();
  const auto &incoming = metadata.incoming;
  const auto &transport = metadata.transport;
  if (incoming.incoming_rows.size() > state->plan.max_incoming_rows)
    invalid("phase=cuda-dispatch-stage, field=incoming_rows.capacity, rank=" +
            std::to_string(metadata.calling_rank) + ", expected<=" +
            std::to_string(state->plan.max_incoming_rows) + ", actual=" +
            std::to_string(incoming.incoming_rows.size()));
  std::size_t tile_count = 0;
  for (const auto &tiles : incoming.mtile_expected_rows) {
    if (tiles.size() > std::numeric_limits<std::size_t>::max() - tile_count)
      invalid("phase=cuda-dispatch-stage, field=mtile_expected_rows.size, rank=" +
              std::to_string(metadata.calling_rank) +
              ", expected=size_t sum, actual=overflow");
    tile_count += tiles.size();
  }
  if (tile_count > state->plan.max_incoming_rows)
    invalid("phase=cuda-dispatch-stage, field=mtile_expected_rows.capacity, rank=" +
            std::to_string(metadata.calling_rank) + ", expected<=" +
            std::to_string(state->plan.max_incoming_rows) + ", actual=" +
            std::to_string(tile_count));
  if (transport.outgoing_packets.size() > state->plan.route_count)
    invalid("phase=cuda-dispatch-stage, field=outgoing_packets.capacity, rank=" +
            std::to_string(metadata.calling_rank) + ", expected<=" +
            std::to_string(state->plan.route_count) + ", actual=" +
            std::to_string(transport.outgoing_packets.size()));

  std::vector<std::uint64_t> count_matrix;
  count_matrix.reserve(checked_size(
      checked_mul(state->plan.world_size, state->plan.num_experts,
                  "source count elements"),
      "source count elements"));
  for (const auto &count_row : metadata.source_expert_count)
    count_matrix.insert(count_matrix.end(), count_row.begin(), count_row.end());

  std::vector<std::uint64_t> mtile_offsets;
  std::vector<std::uint32_t> mtile_rows;
  mtile_offsets.reserve(incoming.mtile_expected_rows.size() + 1);
  mtile_rows.reserve(tile_count);
  mtile_offsets.push_back(0);
  for (const auto &tiles : incoming.mtile_expected_rows) {
    mtile_rows.insert(mtile_rows.end(), tiles.begin(), tiles.end());
    mtile_offsets.push_back(mtile_rows.size());
  }

  std::vector<DeviceIncomingRowRecord> incoming_rows;
  incoming_rows.reserve(incoming.incoming_rows.size());
  for (const auto &row : incoming.incoming_rows)
    incoming_rows.push_back(DeviceIncomingRowRecord{
        row.source_rank, row.expert_id, row.source_expert_offset, row.dst_row});

  std::vector<DeviceReverseRouteRecord> reverse_routes;
  reverse_routes.reserve(host_out.reverse_routes.size());
  for (const auto &route : host_out.reverse_routes)
    reverse_routes.push_back(DeviceReverseRouteRecord{
        route.source_rank, route.local_token_id, route.topk_slot,
        route.expert_id, route.destination_rank, route.dst_row});

  std::vector<DevicePacketHeaderRecord> packet_headers;
  std::vector<std::uint64_t> packet_fanout_offsets;
  std::vector<DeviceFanoutRecord> packet_fanout;
  packet_headers.reserve(transport.outgoing_packets.size());
  packet_fanout_offsets.reserve(transport.outgoing_packets.size() + 1);
  packet_fanout.reserve(checked_size(state->plan.route_count, "route_count"));
  packet_fanout_offsets.push_back(0);
  for (const auto &packet : transport.outgoing_packets) {
    packet_headers.push_back(DevicePacketHeaderRecord{
        packet.identity.source_rank, packet.identity.destination_node,
        packet.identity.sequence_on_link, packet.local_token_id,
        packet.proxy_rank, packet.channel});
    for (const auto &target : packet.fanout)
      packet_fanout.push_back(DeviceFanoutRecord{
          target.topk_slot, target.destination_local_rank, target.expert_id,
          target.dst_row});
    packet_fanout_offsets.push_back(packet_fanout.size());
  }
  if (packet_fanout.size() > state->plan.route_count)
    invalid("phase=cuda-dispatch-stage, field=packet_fanout.capacity, rank=" +
            std::to_string(metadata.calling_rank) + ", expected<=" +
            std::to_string(state->plan.route_count) + ", actual=" +
            std::to_string(packet_fanout.size()));

  const auto convert_signals = [](const auto &source) {
    std::vector<DeviceSignalRecord> result;
    result.reserve(source.size());
    for (const auto &signal : source)
      result.push_back(DeviceSignalRecord{
          signal.source_node, signal.destination_node, signal.source_rank,
          signal.destination_proxy_rank, signal.rail_local_rank,
          signal.channel, signal.expected_packets, signal.expected_signals});
    return result;
  };
  auto outgoing_signals = convert_signals(transport.outgoing_signals);
  auto incoming_signals = convert_signals(transport.incoming_proxy_signals);
  if (outgoing_signals.size() > state->plan.num_nodes ||
      incoming_signals.size() > state->plan.num_nodes)
    invalid("phase=cuda-dispatch-stage, field=signal.capacity, rank=" +
            std::to_string(metadata.calling_rank) + ", expected<=" +
            std::to_string(state->plan.num_nodes) + ", actual=outgoing:" +
            std::to_string(outgoing_signals.size()) + "/incoming:" +
            std::to_string(incoming_signals.size()));

  auto exec = internal::checked_execution_state(execution);
  internal::throw_cuda("dispatch-stage", cudaSetDevice(static_cast<int>(exec->device)),
                       host_handle->calling_rank, host_handle->config.topology.world_size,
                       host_handle->communicator_id, host_handle->identity.generation,
                       ExchangeKind::ExpertCounts, "cudaSetDevice");
  const auto copy_to_device = [&](std::size_t offset, const void *data,
                                  std::uint64_t count,
                                  std::uint64_t element_bytes,
                                  const char *field) {
    if (count == 0)
      return;
    const auto bytes = checked_size(
        checked_mul(count, element_bytes, field), field);
    internal::throw_cuda(
        "dispatch-stage",
        cudaMemcpyAsync(at<void>(state->allocation->device_ptr, offset), data,
                        bytes, cudaMemcpyHostToDevice, exec->stream),
        metadata.calling_rank, metadata.config.topology.world_size,
        metadata.communicator_id, metadata.identity.generation,
        ExchangeKind::ExpertCounts, field);
  };
  const auto &dl = state->device_layout;
  copy_to_device(dl.route_dst_row, rows.data(), rows.size(),
                 sizeof(std::uint64_t), "H2D route_dst_row");
  copy_to_device(dl.source_expert_count, count_matrix.data(),
                 count_matrix.size(), sizeof(std::uint64_t),
                 "H2D source_expert_count");
  copy_to_device(dl.owned_expert_ids, incoming.expert_ids.data(),
                 incoming.expert_ids.size(), sizeof(std::uint32_t),
                 "H2D owned_expert_ids");
  copy_to_device(dl.owned_expert_count, incoming.expert_row_count.data(),
                 incoming.expert_row_count.size(), sizeof(std::uint64_t),
                 "H2D expert_row_count");
  copy_to_device(dl.owned_expert_prefix, incoming.expert_row_prefix.data(),
                 incoming.expert_row_prefix.size(), sizeof(std::uint64_t),
                 "H2D expert_row_prefix");
  copy_to_device(dl.mtile_offsets, mtile_offsets.data(), mtile_offsets.size(),
                 sizeof(std::uint64_t), "H2D mtile_row_offsets");
  copy_to_device(dl.mtile_rows, mtile_rows.data(), mtile_rows.size(),
                 sizeof(std::uint32_t), "H2D mtile_expected_rows");
  copy_to_device(dl.incoming_rows, incoming_rows.data(), incoming_rows.size(),
                 sizeof(DeviceIncomingRowRecord), "H2D incoming_rows");
  copy_to_device(dl.reverse_routes, reverse_routes.data(),
                 reverse_routes.size(), sizeof(DeviceReverseRouteRecord),
                 "H2D reverse_routes");
  copy_to_device(dl.packet_headers, packet_headers.data(),
                 packet_headers.size(), sizeof(DevicePacketHeaderRecord),
                 "H2D packet_headers");
  copy_to_device(dl.packet_fanout_offsets, packet_fanout_offsets.data(),
                 packet_fanout_offsets.size(), sizeof(std::uint64_t),
                 "H2D packet_fanout_offsets");
  copy_to_device(dl.packet_fanout, packet_fanout.data(), packet_fanout.size(),
                 sizeof(DeviceFanoutRecord), "H2D packet_fanout");
  copy_to_device(dl.outgoing_signals, outgoing_signals.data(),
                 outgoing_signals.size(), sizeof(DeviceSignalRecord),
                 "H2D outgoing_signals");
  copy_to_device(dl.incoming_signals, incoming_signals.data(),
                 incoming_signals.size(), sizeof(DeviceSignalRecord),
                 "H2D incoming_proxy_signals");
  internal::throw_cuda("dispatch-stage",
                       cudaEventRecord(state->allocation->completion, exec->stream),
                       host_handle->calling_rank, host_handle->config.topology.world_size,
                       host_handle->communicator_id, host_handle->identity.generation,
                       ExchangeKind::ExpertCounts, "cudaEventRecord");
  internal::throw_cuda("dispatch-stage",
                       cudaEventSynchronize(state->allocation->completion),
                       host_handle->calling_rank, host_handle->config.topology.world_size,
                       host_handle->communicator_id, host_handle->identity.generation,
                       ExchangeKind::ExpertCounts, "cudaEventSynchronize");
  auto impl = std::make_shared<CudaDispatchHandle::Impl>();
  impl->host = std::move(host_handle);
  impl->device = local.impl_->device;
  impl->device.source_expert_count = make_matrix_lease(
      state, dl.source_expert_count, state->plan.world_size,
      state->plan.num_experts);
  impl->device.owned_expert_ids = make_lease(
      state, dl.owned_expert_ids, incoming.expert_ids.size());
  impl->device.expert_row_count = make_lease(
      state, dl.owned_expert_count, incoming.expert_row_count.size());
  impl->device.expert_row_prefix = make_lease(
      state, dl.owned_expert_prefix, incoming.expert_row_prefix.size());
  impl->device.mtile_row_offsets = make_lease(
      state, dl.mtile_offsets, mtile_offsets.size());
  impl->device.mtile_expected_rows = make_lease(
      state, dl.mtile_rows, mtile_rows.size());
  impl->device.incoming_rows = make_lease(
      state, dl.incoming_rows, incoming_rows.size());
  impl->device.reverse_routes = make_lease(
      state, dl.reverse_routes, reverse_routes.size());
  impl->device.packet_headers = make_lease(
      state, dl.packet_headers, packet_headers.size());
  impl->device.packet_fanout_offsets = make_lease(
      state, dl.packet_fanout_offsets, packet_fanout_offsets.size());
  impl->device.packet_fanout = make_lease(
      state, dl.packet_fanout, packet_fanout.size());
  impl->device.outgoing_signals = make_lease(
      state, dl.outgoing_signals, outgoing_signals.size());
  impl->device.incoming_proxy_signals = make_lease(
      state, dl.incoming_signals, incoming_signals.size());
  impl->workspace = std::move(state);
  return CudaDispatchHandle(std::move(impl));
}

void validate_cuda_dispatch_metadata(const CudaDispatchHandle &handle,
                                     const CudaExecution &execution) {
  if (!handle || !execution)
    invalid("phase=cuda-device-cross-check, field=input, rank=n/a, "
            "expected=valid handles, actual=empty");
  const auto state = handle.impl_->workspace;
  const auto &metadata = handle.impl_->host.metadata();
  if (!state ||
      !state->context.execution().same_instance(execution.token()))
    invalid("phase=cuda-device-cross-check, field=execution, rank=" +
            std::to_string(metadata.calling_rank) +
            ", expected=dispatch execution instance, actual=different");
  auto exec = internal::checked_execution_state(execution);
  internal::throw_cuda(
      "device-cross-check", cudaSetDevice(static_cast<int>(exec->device)),
      metadata.calling_rank, metadata.config.topology.world_size,
      metadata.communicator_id, metadata.identity.generation,
      ExchangeKind::ExpertCounts, "cudaSetDevice");
  const auto download = [&](auto tag, std::size_t offset,
                            std::uint64_t count, const char *field) {
    using T = decltype(tag);
    std::vector<T> result(checked_size(count, field));
    if (count == 0)
      return result;
    const auto bytes = checked_size(checked_mul(count, sizeof(T), field), field);
    internal::throw_cuda(
        "device-cross-check",
        cudaMemcpy(result.data(), at<const void>(state->allocation->device_ptr,
                                                offset),
                   bytes, cudaMemcpyDeviceToHost),
        metadata.calling_rank, metadata.config.topology.world_size,
        metadata.communicator_id, metadata.identity.generation,
        ExchangeKind::ExpertCounts, field);
    return result;
  };
  const auto mismatch = [&](const char *field) {
    invalid("phase=cuda-device-cross-check, field=" + std::string(field) +
            ", rank=" + std::to_string(metadata.calling_rank) +
            ", expected=immutable host metadata, actual=device mismatch");
  };
  const auto &dl = state->device_layout;
  const auto &out = metadata.outgoing;
  const auto routes = state->plan.route_count;
  const auto check_lease = [&](const DeviceBufferLease &lease,
                               std::uint32_t rows, std::uint64_t width,
                               const char *field) {
    const auto elements = checked_mul(rows, width, field);
    if (elements == 0) {
      if (lease)
        mismatch(field);
      return;
    }
    if (!lease || lease.rows() != rows || lease.row_width() != width ||
        lease.element_count() != elements ||
        lease.capacity_elements() < elements)
      mismatch(field);
  };
  const auto &leases = handle.impl_->device;
  check_lease(leases.expert_id, 1, routes, "lease.expert_id");
  check_lease(leases.local_offset, 1, routes, "lease.local_offset");
  check_lease(leases.route_dst_row, 1, routes, "lease.route_dst_row");
  check_lease(leases.route_dst_rank, 1, routes, "lease.route_dst_rank");
  check_lease(leases.route_dst_node, 1, routes, "lease.route_dst_node");
  check_lease(leases.route_dst_local_rank, 1, routes,
              "lease.route_dst_local_rank");
  check_lease(leases.combine_order, 1, routes, "lease.combine_order");
  check_lease(leases.destination_node_mask, 1,
              checked_mul(out.num_tokens, out.node_mask_words,
                          "lease.destination_node_mask"),
              "lease.destination_node_mask");
  check_lease(leases.source_expert_count, state->plan.world_size,
              state->plan.num_experts, "lease.source_expert_count");
  check_lease(leases.owned_expert_ids, 1,
              metadata.incoming.expert_ids.size(),
              "lease.owned_expert_ids");
  check_lease(leases.expert_row_count, 1,
              metadata.incoming.expert_row_count.size(),
              "lease.expert_row_count");
  check_lease(leases.expert_row_prefix, 1,
              metadata.incoming.expert_row_prefix.size(),
              "lease.expert_row_prefix");
  check_lease(leases.mtile_row_offsets, 1,
              metadata.incoming.mtile_expected_rows.size() + 1,
              "lease.mtile_row_offsets");
  std::size_t lease_tile_count = 0;
  for (const auto &tiles : metadata.incoming.mtile_expected_rows)
    lease_tile_count += tiles.size();
  check_lease(leases.mtile_expected_rows, 1, lease_tile_count,
              "lease.mtile_expected_rows");
  check_lease(leases.incoming_rows, 1,
              metadata.incoming.incoming_rows.size(),
              "lease.incoming_rows");
  check_lease(leases.reverse_routes, 1, out.reverse_routes.size(),
              "lease.reverse_routes");
  check_lease(leases.packet_headers, 1,
              metadata.transport.outgoing_packets.size(),
              "lease.packet_headers");
  check_lease(leases.packet_fanout_offsets, 1,
              metadata.transport.outgoing_packets.size() + 1,
              "lease.packet_fanout_offsets");
  check_lease(leases.packet_fanout, 1, routes, "lease.packet_fanout");
  check_lease(leases.outgoing_signals, 1,
              metadata.transport.outgoing_signals.size(),
              "lease.outgoing_signals");
  check_lease(leases.incoming_proxy_signals, 1,
              metadata.transport.incoming_proxy_signals.size(),
              "lease.incoming_proxy_signals");
  if (download(std::uint32_t{}, dl.expert_id, routes, "expert_id") !=
      out.expert_id)
    mismatch("expert_id");
  if (download(std::uint64_t{}, dl.local_offset, routes, "local_offset") !=
      out.local_offset)
    mismatch("local_offset");
  if (download(std::uint64_t{}, dl.route_dst_row, routes, "route_dst_row") !=
      out.route_dst_row)
    mismatch("route_dst_row");
  if (download(std::uint32_t{}, dl.route_dst_rank, routes,
               "route_dst_rank") != out.route_dst_rank)
    mismatch("route_dst_rank");
  if (download(std::uint32_t{}, dl.route_dst_node, routes,
               "route_dst_node") != out.route_dst_node)
    mismatch("route_dst_node");
  if (download(std::uint32_t{}, dl.route_dst_local_rank, routes,
               "route_dst_local_rank") != out.route_dst_local_rank)
    mismatch("route_dst_local_rank");
  if (download(std::uint32_t{}, dl.combine_order, routes, "combine_order") !=
      out.combine_order)
    mismatch("combine_order");
  const auto mask_count = checked_mul(out.num_tokens, out.node_mask_words,
                                      "destination_node_mask");
  if (download(std::uint64_t{}, dl.masks, mask_count,
               "destination_node_mask") != out.destination_node_mask)
    mismatch("destination_node_mask");

  std::vector<std::uint64_t> expected_counts;
  expected_counts.reserve(checked_size(
      checked_mul(state->plan.world_size, state->plan.num_experts,
                  "source_expert_count"),
      "source_expert_count"));
  for (const auto &row : metadata.source_expert_count)
    expected_counts.insert(expected_counts.end(), row.begin(), row.end());
  if (download(std::uint64_t{}, dl.source_expert_count,
               expected_counts.size(), "source_expert_count") !=
      expected_counts)
    mismatch("source_expert_count");

  const auto &incoming = metadata.incoming;
  if (download(std::uint32_t{}, dl.owned_expert_ids,
               incoming.expert_ids.size(), "owned_expert_ids") !=
      incoming.expert_ids)
    mismatch("owned_expert_ids");
  if (download(std::uint64_t{}, dl.owned_expert_count,
               incoming.expert_row_count.size(), "expert_row_count") !=
      incoming.expert_row_count)
    mismatch("expert_row_count");
  if (download(std::uint64_t{}, dl.owned_expert_prefix,
               incoming.expert_row_prefix.size(), "expert_row_prefix") !=
      incoming.expert_row_prefix)
    mismatch("expert_row_prefix");
  std::vector<std::uint64_t> expected_tile_offsets{0};
  std::vector<std::uint32_t> expected_tiles;
  for (const auto &tiles : incoming.mtile_expected_rows) {
    expected_tiles.insert(expected_tiles.end(), tiles.begin(), tiles.end());
    expected_tile_offsets.push_back(expected_tiles.size());
  }
  if (download(std::uint64_t{}, dl.mtile_offsets,
               expected_tile_offsets.size(), "mtile_row_offsets") !=
      expected_tile_offsets)
    mismatch("mtile_row_offsets");
  if (download(std::uint32_t{}, dl.mtile_rows, expected_tiles.size(),
               "mtile_expected_rows") != expected_tiles)
    mismatch("mtile_expected_rows");

  const auto device_incoming = download(
      DeviceIncomingRowRecord{}, dl.incoming_rows,
      incoming.incoming_rows.size(), "incoming_rows");
  for (std::size_t index = 0; index < device_incoming.size(); ++index) {
    const auto &actual = device_incoming[index];
    const auto &expected = incoming.incoming_rows[index];
    if (actual.source_rank != expected.source_rank ||
        actual.expert_id != expected.expert_id ||
        actual.source_expert_offset != expected.source_expert_offset ||
        actual.dst_row != expected.dst_row)
      mismatch("incoming_rows");
  }
  const auto device_reverse = download(
      DeviceReverseRouteRecord{}, dl.reverse_routes, out.reverse_routes.size(),
      "reverse_routes");
  for (std::size_t index = 0; index < device_reverse.size(); ++index) {
    const auto &actual = device_reverse[index];
    const auto &expected = out.reverse_routes[index];
    if (actual.source_rank != expected.source_rank ||
        actual.local_token_id != expected.local_token_id ||
        actual.topk_slot != expected.topk_slot ||
        actual.expert_id != expected.expert_id ||
        actual.destination_rank != expected.destination_rank ||
        actual.dst_row != expected.dst_row)
      mismatch("reverse_routes");
  }

  const auto &packets = metadata.transport.outgoing_packets;
  const auto device_headers = download(
      DevicePacketHeaderRecord{}, dl.packet_headers, packets.size(),
      "packet_headers");
  std::vector<std::uint64_t> expected_fanout_offsets{0};
  std::vector<DeviceFanoutRecord> expected_fanout;
  for (std::size_t index = 0; index < packets.size(); ++index) {
    const auto &actual = device_headers[index];
    const auto &expected = packets[index];
    if (actual.source_rank != expected.identity.source_rank ||
        actual.destination_node != expected.identity.destination_node ||
        actual.sequence_on_link != expected.identity.sequence_on_link ||
        actual.local_token_id != expected.local_token_id ||
        actual.proxy_rank != expected.proxy_rank ||
        actual.channel != expected.channel)
      mismatch("packet_headers");
    for (const auto &target : expected.fanout)
      expected_fanout.push_back(DeviceFanoutRecord{
          target.topk_slot, target.destination_local_rank, target.expert_id,
          target.dst_row});
    expected_fanout_offsets.push_back(expected_fanout.size());
  }
  if (download(std::uint64_t{}, dl.packet_fanout_offsets,
               expected_fanout_offsets.size(), "packet_fanout_offsets") !=
      expected_fanout_offsets)
    mismatch("packet_fanout_offsets");
  const auto device_fanout =
      download(DeviceFanoutRecord{}, dl.packet_fanout,
               expected_fanout.size(), "packet_fanout");
  for (std::size_t index = 0; index < device_fanout.size(); ++index) {
    const auto &actual = device_fanout[index];
    const auto &expected = expected_fanout[index];
    if (actual.topk_slot != expected.topk_slot ||
        actual.destination_local_rank != expected.destination_local_rank ||
        actual.expert_id != expected.expert_id ||
        actual.dst_row != expected.dst_row)
      mismatch("packet_fanout");
  }

  const auto compare_signals = [&](const auto &expected, std::size_t offset,
                                   const char *field) {
    const auto actual = download(DeviceSignalRecord{}, offset, expected.size(),
                                 field);
    for (std::size_t index = 0; index < actual.size(); ++index) {
      const auto &lhs = actual[index];
      const auto &rhs = expected[index];
      if (lhs.source_node != rhs.source_node ||
          lhs.destination_node != rhs.destination_node ||
          lhs.source_rank != rhs.source_rank ||
          lhs.destination_proxy_rank != rhs.destination_proxy_rank ||
          lhs.rail_local_rank != rhs.rail_local_rank ||
          lhs.channel != rhs.channel ||
          lhs.expected_packets != rhs.expected_packets ||
          lhs.expected_signals != rhs.expected_signals)
        mismatch(field);
    }
  };
  compare_signals(metadata.transport.outgoing_signals, dl.outgoing_signals,
                  "outgoing_signals");
  compare_signals(metadata.transport.incoming_proxy_signals,
                  dl.incoming_signals, "incoming_proxy_signals");
}

} // namespace megamoe::cuda
