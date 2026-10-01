#pragma once

// H2048/I512 packed gate8/up8 profiles, measured at 1620 MHz with CW rings.
// Communication changes the crossover: EP8 uses the wide entry at 512,
// while EP16 keeps the narrow entry until 2048 tokens per source rank.
#include "../static_inference_20260922/weftmoe_forward_tuning_generated.hpp"
#define WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS 512
#define WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS 2048

// PF uses static scheduling with the shared unified Backward arithmetic.
// Forward-only profiles otherwise select the standalone Backward defaults.
#define WEFTMOE_BWD_UNIFIED_WORKERS 1
#define WEFTMOE_BWD_RELAY_CTAS 16
#define WEFTMOE_BWD_UNIFIED_RDMA_CTAS 16
// Long-sequence static Backward: keep the 64512-register CTA budget while
// reducing the GIN control path's stack footprint. Forward quotas stay intact.
#define WEFTMOE_BWD_UNIFIED_MATH_REGISTERS 208
#define WEFTMOE_BWD_UNIFIED_CONTROL_REGISTERS 88
