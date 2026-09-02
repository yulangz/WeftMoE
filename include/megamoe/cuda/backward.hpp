#pragma once

#include "megamoe/version.hpp"

MEGAMOE_PUBLIC_API_BEGIN

namespace megamoe::cuda {

// Backward is intentionally unavailable while its pipeline is redesigned to
// use the Forward transport/compute overlap architecture.
bool backward_pipeline_available() noexcept;

} // namespace megamoe::cuda

MEGAMOE_PUBLIC_API_END
