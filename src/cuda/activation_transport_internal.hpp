#pragma once

#include "cuda/device_notify_internal.hpp"
#include "megamoe/cuda/activation_transport.hpp"

namespace megamoe::cuda {

class ActivationTransportBridge {
public:
  static DeviceDispatchKernelView
  view(DeviceDispatchConsumerLease &consumer);
  static const float *gate_pointer(const DeviceDispatchHandle &dispatch);
  static void wait_for_gates(const DeviceDispatchHandle &dispatch,
                             const CudaExecution &execution);
  static void submit(DeviceDispatchConsumerLease &consumer);
  static const RankCollectiveContext &
  context(const DeviceDispatchConsumerLease &consumer);
  static BFloat16ActivationLease upload_for_testing(
      const std::vector<transport::BFloat16> &host,
      std::uint64_t num_tokens, std::uint64_t hidden,
      const RankCollectiveContext &context, const CudaExecution &execution,
      const ActivationTransportResourcePool &pool);
  static BFloat16ActivationLease reissue_for_benchmark(
      const BFloat16ActivationLease &storage,
      const RankCollectiveContext &context, const CudaExecution &execution);
  static ActivationTransportValidationDownload download_for_validation(
      const DeviceActivationTransportHandle &handle,
      const CudaExecution &execution);
};

} // namespace megamoe::cuda
