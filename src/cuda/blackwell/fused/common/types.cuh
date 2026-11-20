// Adapted from DeepSeek MegaMoE (MIT), see LICENSE-DeepSeek.
#pragma once

#include <cute/arch/mma_sm100_desc.hpp>

namespace weftmoe::sm100 {

static constexpr uint32_t kNumDeviceLocalityDomains = 2;

} // namespace weftmoe::sm100
