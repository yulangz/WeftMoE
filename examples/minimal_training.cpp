#include <cstdint>
#include <utility>

#include "megamoe/cuda/activation_transport.hpp"
#include "megamoe/version.hpp"

namespace {

using OutputGradientProducer = megamoe::cuda::SelectedOutputGradientLease (*)(
    const megamoe::cuda::DeviceForwardHandle &,
    const megamoe::cuda::CudaExecution &);

// Router/autograd integrations create the typed input leases. This example
// starts at that zero-copy boundary and shows the complete owning lifetime:
// session -> pools -> Notify -> Forward -> Backward -> result destruction.
[[maybe_unused]] void run_training_generation(
    const megamoe::cuda::NcclSession &session,
    const megamoe::cuda::CudaExecution &execution,
    megamoe::RuntimeConfig config,
    megamoe::cuda::CudaRouteInput routes,
    megamoe::cuda::CudaGateInput gates,
    const megamoe::cuda::BFloat16ActivationLease &activations,
    const megamoe::cuda::ForwardWeightLease &weights,
    std::uint64_t max_tokens_per_rank,
    std::uint64_t hidden,
    std::uint64_t intermediate,
    OutputGradientProducer make_output_gradient) {
  megamoe::cuda::DeviceNotifyResourcePoolConfig notify_pool_config;
  notify_pool_config.capacity_config = config;
  notify_pool_config.max_tokens_per_rank = max_tokens_per_rank;
  notify_pool_config.max_config_wire_bytes = 64U * 1024U * 1024U;
  auto notify_pool = megamoe::cuda::make_device_notify_resource_pool(
      session, execution, notify_pool_config);
  megamoe::cuda::CudaNotifyGenerationCoordinator notify(session, execution,
                                                         notify_pool);

  megamoe::cuda::DeviceNotifyGenerationRequest notify_request;
  notify_request.config = config;
  notify_request.local_routes = std::move(routes);
  notify_request.local_gates = std::move(gates);
  notify_request.max_tokens_per_rank = max_tokens_per_rank;
  auto dispatch_generation = notify.run_device(std::move(notify_request));

  megamoe::cuda::ActivationTransportPoolConfig data_pool_config;
  data_pool_config.max_tokens_per_rank = max_tokens_per_rank;
  data_pool_config.max_hidden_size = hidden;
  data_pool_config.max_intermediate_size = intermediate;
  data_pool_config.max_top_k = config.top_k;
  data_pool_config.max_world_size = config.topology.world_size;
  data_pool_config.max_nodes = config.topology.num_nodes;
  data_pool_config.max_destination_rows_per_rank = max_tokens_per_rank;
  data_pool_config.max_owned_experts = config.num_experts;
  data_pool_config.ring_slots_per_source_channel = 2;
  data_pool_config.enable_forward_cache = true;
  data_pool_config.enable_backward = true;
  const auto capability = megamoe::cuda::probe_activation_transport_capability(
      session, execution);
  data_pool_config.enable_gin_rdma = capability.gin_usable;
  auto data_pool = megamoe::cuda::make_activation_transport_pool(
      session.session(), execution, data_pool_config);

  megamoe::cuda::prepare_forward_weights_for_backward(weights, execution);
  const megamoe::cuda::ForwardMegaKernelConfig forward_config{
      hidden, intermediate,
      megamoe::cuda::ForwardCacheMode::CacheFc1InputOutput};
  const megamoe::cuda::BackwardMegaKernelConfig backward_config{
      hidden, intermediate,
      megamoe::cuda::BackwardWeightGradientDType::BFloat16};

  const auto execute = [&](auto &forward_kernel, auto &backward_kernel) {
    auto forward = forward_kernel.run(dispatch_generation.dispatch(),
                                      activations, weights, forward_config);
    auto output_gradient = make_output_gradient(forward, execution);
    auto backward = backward_kernel.run(forward, output_gradient,
                                        backward_config);
    (void)backward.audit();
  };

  if (capability.gin_usable) {
    megamoe::cuda::GinRdmaForwardMegaKernel forward(session, execution,
                                                    data_pool);
    megamoe::cuda::GinRdmaBackwardMegaKernel backward(session, execution,
                                                      data_pool);
    execute(forward, backward);
  } else {
    megamoe::cuda::LocalCudaP2pForwardMegaKernel forward(session, execution,
                                                         data_pool);
    megamoe::cuda::LocalCudaP2pBackwardMegaKernel backward(session, execution,
                                                           data_pool);
    execute(forward, backward);
  }
}

static_assert(megamoe::kPublicApiVersionMajor == 2U);

} // namespace

int main() { return 0; }
