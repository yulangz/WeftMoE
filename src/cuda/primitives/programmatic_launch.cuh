#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

namespace weftmoe::cuda::programmatic_launch {

// Experimental opt-in. The forward backend already requires SM90; retain
// ordinary stream serialization unless explicitly enabled for the process.
inline bool enabled() {
  static const bool value = [] {
    const char *setting = std::getenv("WEFTMOE_ENABLE_PDL");
    return setting && setting[0] == '1' && setting[1] == '\0';
  }();
  return value;
}

inline cudaLaunchAttribute attribute() {
  cudaLaunchAttribute result{};
  result.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  result.val.programmaticStreamSerializationAllowed = 1;
  return result;
}

template <class Kernel, class... Args>
cudaError_t launch(dim3 grid, dim3 block, std::size_t shared,
                   cudaStream_t stream, Kernel kernel, Args... args) {
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = shared;
  config.stream = stream;
  auto dependency = attribute();
  if (enabled()) {
    config.attrs = &dependency;
    config.numAttrs = 1;
  }
  return cudaLaunchKernelEx(&config, kernel, args...);
}

__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // This is also valid for an ordinarily serialized launch: it has no
  // outstanding programmatic dependency in that case.
  cudaGridDependencySynchronize();
#endif
}

__device__ __forceinline__ void release() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // A scheduling hint, not a data-ready signal. The successor must wait
  // before touching any memory produced by this grid.
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

__device__ __forceinline__ void begin() {
  wait();
  release();
}

} // namespace weftmoe::cuda::programmatic_launch
