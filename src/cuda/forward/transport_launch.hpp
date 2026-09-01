#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

// Included after activation_transport.cu's device-only anonymous namespace,
// while already inside namespace megamoe::cuda.
namespace forward_transport_launch {

cudaError_t set_max_dynamic_shared_memory(int bytes);

cudaError_t launch(const cudaLaunchConfig_t *config, const void *args,
                   std::size_t args_bytes, const CUtensorMap &x_map,
                   const CUtensorMap &s_map, const CUtensorMap &w1_map,
                   const CUtensorMap &w2_map);

} // namespace forward_transport_launch
