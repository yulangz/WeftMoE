#include <cstdint>
#include <iostream>

#include "weftmoe/cuda/activation_transport.hpp"
#include "weftmoe/cuda/backward.hpp"
#include "weftmoe/version.hpp"

int main() {
  constexpr weftmoe::cuda::ForwardMegaKernelConfig forward{7168, 3072};

  // A real Forward launch is collective across eight or sixteen processes and
  // binds caller-owned route, activation, gate, and weight leases. Keep this
  // installed example runnable as a lightweight loader/capability check; the
  // repository launchers demonstrate the collective setup.
  const auto capability =
      weftmoe::cuda::probe_activation_transport_capability(1);
  std::cout << "WeftMoE API v" << weftmoe::kPublicApiVersionMajor
            << " Forward=" << forward.hidden_size << 'x'
            << forward.intermediate_size
            << " GIN=" << static_cast<unsigned int>(capability.gin_usable)
            << " Backward="
            << static_cast<unsigned int>(
                   weftmoe::cuda::backward_pipeline_available())
            << '\n';
  return 0;
}
