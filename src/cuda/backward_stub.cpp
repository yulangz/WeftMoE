#include "weftmoe/cuda/backward.hpp"

namespace weftmoe::cuda {

// CPU-only control-plane binaries retain the public query but do not link the
// CUDA implementation. The CUDA translation unit provides the strong,
// qualified definition.
__attribute__((weak)) bool backward_pipeline_available() noexcept {
  return false;
}

} // namespace weftmoe::cuda
