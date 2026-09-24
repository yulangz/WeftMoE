#pragma once

// H2048/I512 packed gate8/up8 profiles, measured at 1620 MHz with CW rings.
// Communication changes the crossover: EP8 uses the wide entry at 512,
// while EP16 keeps the narrow entry until 2048 tokens per source rank.
#include "../static_inference_20260922/weftmoe_forward_tuning_generated.hpp"
#define WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS 512
#define WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS 2048
