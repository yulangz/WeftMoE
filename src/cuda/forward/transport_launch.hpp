#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

// Host interface to the separately compiled transport grid.
namespace weftmoe::cuda::forward_transport_launch {

cudaError_t set_max_dynamic_shared_memory(int bytes);
cudaError_t set_local_max_dynamic_shared_memory(int bytes);

cudaError_t launch(const cudaLaunchConfig_t *config, const void *args,
                   std::size_t args_bytes, const CUtensorMap &x_map,
                   const CUtensorMap &s_map, const CUtensorMap &w1_map,
                   const CUtensorMap &w2_map);

cudaError_t launch_local(const cudaLaunchConfig_t *config, const void *args,
                         std::size_t args_bytes, const CUtensorMap &x_map,
                         const CUtensorMap &s_map, const CUtensorMap &w1_map,
                         const CUtensorMap &w2_map);

} // namespace weftmoe::cuda::forward_transport_launch
