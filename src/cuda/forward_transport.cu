#include "cuda/communication/chunk_ring.cuh"
#include "cuda/communication/local_notification.cuh"
#include "cuda/primitives/hopper.cuh"
// Dedicated transport grid compilation; compute keeps its own register budget.
#define WEFTMOE_FORWARD_TRANSPORT_TU 1

#include "cuda/activation_transport_internal.hpp"
#include "cuda/forward/transport_launch.hpp"

#include <cuda.h>
#include <cuda/atomic>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
#include <nccl.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cute/arch/cluster_sm90.hpp>
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
#include <string_view>
#include <thread>

#include "cuda/cuda_internal.hpp"
#include "cuda/backward/tuning_config.hpp"
#include "cuda/persistent_v3/backward/pipeline_state.cuh"
#include "cuda/persistent_v3/backward/scheduler.cuh"
#include "cuda/persistent_v3/contracts.hpp"
#include "cuda/persistent_v3/forward/pipeline_state.cuh"
#include "cuda/persistent_v3/queues.cuh"
#include "cuda/persistent_v3/roles/common.cuh"
#include "cuda/profiling/device_timeline.cuh"
#include "cuda/transport_generation_completion.cuh"
#include "weftmoe/api/errors.hpp"

#ifndef WEFTMOE_FORWARD_LIGHT_PROGRESS
#define WEFTMOE_FORWARD_LIGHT_PROGRESS 0
#endif

namespace weftmoe::cuda {
struct ActivationMegaKernelBackendState;
namespace {

#include "cuda/forward/device_kernels.cuh"

} // namespace

namespace forward_transport_launch {

cudaError_t set_max_dynamic_shared_memory(int bytes) {
  return cudaFuncSetAttribute(
      local_p2p_forward_megakernel<false, ForwardKernelRole::TransportOnly>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

cudaError_t set_local_max_dynamic_shared_memory(int bytes) {
  return cudaFuncSetAttribute(
      local_p2p_forward_megakernel<true, ForwardKernelRole::TransportOnly>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

cudaError_t launch(const cudaLaunchConfig_t *config, const void *opaque_args,
                   std::size_t args_bytes, const CUtensorMap &x_map,
                   const CUtensorMap &s_map, const CUtensorMap &w1_map,
                   const CUtensorMap &w2_map) {
  if (opaque_args == nullptr || args_bytes != sizeof(ForwardKernelArgs))
    return cudaErrorInvalidValue;
  static_assert(__is_trivially_copyable(ForwardKernelArgs));
  ForwardKernelArgs args{};
  std::memcpy(&args, opaque_args, sizeof(args));
  return cudaLaunchKernelEx(
      config,
      local_p2p_forward_megakernel<false, ForwardKernelRole::TransportOnly>,
      args, x_map, s_map, w1_map, w2_map);
}

cudaError_t launch_local(const cudaLaunchConfig_t *config,
                         const void *opaque_args, std::size_t args_bytes,
                         const CUtensorMap &x_map, const CUtensorMap &s_map,
                         const CUtensorMap &w1_map,
                         const CUtensorMap &w2_map) {
  if (opaque_args == nullptr || args_bytes != sizeof(ForwardKernelArgs))
    return cudaErrorInvalidValue;
  static_assert(__is_trivially_copyable(ForwardKernelArgs));
  ForwardKernelArgs args{};
  std::memcpy(&args, opaque_args, sizeof(args));
  return cudaLaunchKernelEx(
      config,
      local_p2p_forward_megakernel<true, ForwardKernelRole::TransportOnly>,
      args, x_map, s_map, w1_map, w2_map);
}

} // namespace forward_transport_launch
} // namespace weftmoe::cuda
