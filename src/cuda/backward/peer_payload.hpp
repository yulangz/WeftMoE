#pragma once

#include "weftmoe/transport/activation_protocol.hpp"

namespace weftmoe::cuda::backward_detail {

// Addresses imported into the consuming process, not another process's
// original CUDA pointers. Payload geometry is rank-local; scheduler metadata
// continues to use the shared transport-pool layout.
struct PeerPayload {
  transport::BFloat16 *dr;
  transport::BFloat16 *contribution;
  float *dgate;
};

} // namespace weftmoe::cuda::backward_detail
