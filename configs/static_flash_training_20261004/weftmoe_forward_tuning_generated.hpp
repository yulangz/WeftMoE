#pragma once

// Flash-size training benchmark: retain the validated PF profile, changing
// only the compiled expert dimensions. This is not a Flash-specific tune.
#include "../static_inference_ring_20260924/weftmoe_forward_tuning_generated.hpp"

#undef WEFTMOE_FWD_MAIN_HIDDEN
#undef WEFTMOE_FWD_MAIN_INTERMEDIATE
#define WEFTMOE_FWD_MAIN_HIDDEN 4096
#define WEFTMOE_FWD_MAIN_INTERMEDIATE 2048
