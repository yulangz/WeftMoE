
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
  unsigned long long wave_count;
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
  std::size_t route_dst_mtile = 0;
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
  std::size_t expert_tile_base = 0;
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
  std::size_t packet_primary_tile = 0;
  std::size_t packet_primary_expert = 0;
  std::size_t packet_primary_rank = 0;
  std::size_t local_route_order = 0;
  std::size_t dispatch_wave_offsets = 0;
  std::size_t dispatch_wave_destination_node = 0;
  std::size_t dispatch_wave_channel = 0;
  std::size_t dispatch_wave_first_primary_tile = 0;
  std::size_t dispatch_wave_last_primary_tile = 0;
  std::size_t dispatch_wave_packet_count = 0;
  std::size_t dispatch_wave_payload_bytes = 0;
  std::size_t outgoing_signals = 0;
  std::size_t incoming_signals = 0;
  std::size_t temp = 0;
  std::size_t total = 0;
};

Layout make_device_layout(const CudaNotifyWorkspacePlan &plan) {
  Layout layout;
  std::size_t cursor = 0;
  const auto routes = std::max<std::uint64_t>(plan.route_count, 1);
  const auto sort_keys = std::max<std::uint64_t>(routes, plan.num_nodes);
  layout.keys_in = append_region(cursor, sort_keys, sizeof(std::uint64_t), "keys_in");
  layout.keys_out = append_region(cursor, sort_keys, sizeof(std::uint64_t), "keys_out");
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
  layout.route_dst_mtile = append_region(cursor, routes, sizeof(std::uint32_t),
                                         "route_dst_mtile");
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
  layout.expert_tile_base = append_region(
      cursor, plan.num_experts, sizeof(std::uint64_t), "expert_tile_base");
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
  layout.packet_primary_tile = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t), "packet_primary_tile");
  layout.packet_primary_expert = append_region(
      cursor, plan.route_count, sizeof(std::uint32_t), "packet_primary_expert");
  layout.packet_primary_rank = append_region(
      cursor, plan.route_count, sizeof(std::uint32_t), "packet_primary_rank");
  layout.local_route_order = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t), "local_route_order");
  layout.dispatch_wave_offsets = append_region(
      cursor, plan.route_count + 1, sizeof(std::uint64_t),
      "dispatch_wave_offsets");
  layout.dispatch_wave_destination_node = append_region(
      cursor, plan.route_count, sizeof(std::uint32_t),
      "dispatch_wave_destination_node");
  layout.dispatch_wave_channel = append_region(
      cursor, plan.route_count, sizeof(std::uint32_t),
      "dispatch_wave_channel");
  layout.dispatch_wave_first_primary_tile = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t),
      "dispatch_wave_first_primary_tile");
  layout.dispatch_wave_last_primary_tile = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t),
      "dispatch_wave_last_primary_tile");
  layout.dispatch_wave_packet_count = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t),
      "dispatch_wave_packet_count");
  layout.dispatch_wave_payload_bytes = append_region(
      cursor, plan.route_count, sizeof(std::uint64_t),
      "dispatch_wave_payload_bytes");
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

