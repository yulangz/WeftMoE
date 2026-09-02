#include <cuda_runtime_api.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/cuda/activation_transport.hpp"
#include "megamoe/version.hpp"

namespace {

void check_cuda(cudaError_t result, const char *operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
}

template <class T> struct DeviceTensor {
  T *data = nullptr;
  std::uint64_t elements = 0;
  std::shared_ptr<const void> owner;
};

template <class T>
DeviceTensor<T> upload(cudaStream_t stream, const std::vector<T> &host) {
  T *pointer = nullptr;
  const auto bytes = host.size() * sizeof(T);
  check_cuda(cudaMalloc(reinterpret_cast<void **>(&pointer),
                        bytes == 0 ? sizeof(T) : bytes),
             "cudaMalloc");
  std::shared_ptr<const void> owner(
      pointer, [](const void *value) noexcept {
        (void)cudaFree(const_cast<void *>(value));
      });
  if (bytes != 0)
    check_cuda(cudaMemcpyAsync(pointer, host.data(), bytes,
                               cudaMemcpyHostToDevice, stream),
               "cudaMemcpyAsync");
  return {pointer, static_cast<std::uint64_t>(host.size()), std::move(owner)};
}

std::vector<megamoe::transport::BFloat16>
bf16_values(std::size_t count, float scale) {
  std::vector<megamoe::transport::BFloat16> result(count);
  for (std::size_t index = 0; index < count; ++index) {
    const auto value = scale * static_cast<float>((index % 13U) + 1U);
    result[index] = megamoe::transport::float_to_bfloat16(value);
  }
  return result;
}

int run_world1_forward() {
  constexpr std::uint64_t tokens = 1;
  constexpr std::uint32_t top_k = 1;
  constexpr std::uint32_t experts = 1;
  constexpr std::uint64_t hidden = 7168;
  constexpr std::uint64_t intermediate = 3072;
  constexpr std::uint64_t generation = 1;

  auto bootstrap = megamoe::cuda::make_nccl_bootstrap_id();
  auto session = megamoe::cuda::make_nccl_session(
      "minimal-world1", 1, 0, generation, 0, bootstrap);
  auto execution = megamoe::cuda::make_cuda_execution("consumer", 0);
  auto context = megamoe::make_rank_collective_context(
      session.session(), 0, execution.token());

  cudaStream_t producer_stream = nullptr;
  check_cuda(cudaStreamCreateWithFlags(&producer_stream,
                                       cudaStreamNonBlocking),
             "cudaStreamCreateWithFlags");
  std::shared_ptr<const void> stream_owner(
      producer_stream, [](const void *value) noexcept {
        (void)cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(const_cast<void *>(value)));
      });
  auto producer = megamoe::cuda::make_external_cuda_producer(
      context, 0, producer_stream, stream_owner, "external-router-autograd");

  auto route_storage = upload<std::int32_t>(producer_stream, {0});
  auto gate_storage = upload<float>(producer_stream, {1.0F});
  auto activation_storage = upload(
      producer_stream, bf16_values(static_cast<std::size_t>(tokens * hidden),
                                   0.01F));
  auto w1_storage = upload(
      producer_stream,
      bf16_values(static_cast<std::size_t>(experts * 2ULL * intermediate * hidden),
                  0.0005F));
  auto w2_storage = upload(
      producer_stream,
      bf16_values(static_cast<std::size_t>(experts * hidden * intermediate),
                  0.0007F));
  auto producer_routes = producer.publish_selected_experts(
      route_storage.data, route_storage.elements, tokens, top_k,
      route_storage.owner);
  auto producer_gates = producer.publish_gates(
      gate_storage.data, gate_storage.elements, producer_routes,
      gate_storage.owner);
  auto routes = megamoe::cuda::import_selected_expert_routes(
      producer_routes, context, execution);
  auto gates = megamoe::cuda::import_selected_expert_gates(
      producer_gates, routes, context, execution);

  megamoe::RuntimeConfig config;
  config.num_experts = experts;
  config.top_k = top_k;
  config.m_tile_rows = 128;
  config.topology.world_size = 1;
  config.topology.num_nodes = 1;
  config.topology.rank_to_node = {0};
  config.topology.rank_to_local_rank = {0};
  config.topology.expert_to_rank = {0};
  config.capacity.expert_rows = {tokens};
  config.capacity.workspace_rows_per_rank = {tokens};

  megamoe::cuda::DeviceNotifyResourcePoolConfig notify_pool_config;
  notify_pool_config.capacity_config = config;
  notify_pool_config.max_tokens_per_rank = tokens;
  notify_pool_config.max_config_wire_bytes = 1U << 20U;
  notify_pool_config.max_inflight_generations = 1;
  auto notify_pool = megamoe::cuda::make_device_notify_resource_pool(
      session, execution, notify_pool_config);
  megamoe::cuda::CudaNotifyGenerationCoordinator notify(session, execution,
                                                         notify_pool);

  megamoe::cuda::ActivationTransportPoolConfig data_pool_config;
  data_pool_config.max_tokens_per_rank = tokens;
  data_pool_config.max_hidden_size = hidden;
  data_pool_config.max_intermediate_size = intermediate;
  data_pool_config.max_top_k = top_k;
  data_pool_config.max_world_size = 1;
  data_pool_config.max_nodes = 1;
  data_pool_config.max_destination_rows_per_rank = tokens;
  data_pool_config.max_owned_experts = experts;
  data_pool_config.ring_slots_per_source_channel = 2;
  data_pool_config.max_inflight_generations = 1;
  auto data_pool = megamoe::cuda::make_activation_transport_pool(
      session.session(), execution, data_pool_config);

  auto activations = producer.publish_activations(
      activation_storage.data, activation_storage.elements, tokens, hidden,
      data_pool, activation_storage.owner);
  auto weights = producer.publish_weights(
      w1_storage.data, w1_storage.elements, w1_storage.owner, w2_storage.data,
      w2_storage.elements, w2_storage.owner, experts, 0, hidden, intermediate);
  megamoe::cuda::DeviceNotifyGenerationRequest notify_request;
  notify_request.config = config;
  notify_request.local_routes = std::move(routes);
  notify_request.local_gates = std::move(gates);
  notify_request.max_tokens_per_rank = tokens;
  auto dispatch = notify.run_device(std::move(notify_request));

  megamoe::cuda::LocalCudaP2pForwardMegaKernel forward_kernel(
      session, execution, data_pool);
  const megamoe::cuda::ForwardMegaKernelConfig forward_config{
      hidden, intermediate};
  auto forward = forward_kernel.run(dispatch.dispatch(), activations, weights,
                                    forward_config);
  if (!forward)
    throw std::runtime_error("Forward returned an empty result");
  std::cout << "MegaMOE API v" << megamoe::kPublicApiVersionMajor
            << " world1 Forward complete\n";
  return 0;
}

} // namespace

int main() {
  try {
    return run_world1_forward();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
