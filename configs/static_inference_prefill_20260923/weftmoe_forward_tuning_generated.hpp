#pragma once

// Preserve the narrow-N decode entry; select its N128/N256 K128 2/2-stage
// companion for >=8192 source tokens/rank. No additional weight allocation.
#include "../static_inference_20260922/weftmoe_forward_tuning_generated.hpp"
#define WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS 8192
