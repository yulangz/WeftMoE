#pragma once

// Same stable DT arithmetic/epilogues; specialize only the expert dimensions.
#include "../dt_flash_probe_20261009/weftmoe_forward_tuning_generated.hpp"
#undef WEFTMOE_FWD_MAIN_HIDDEN
#undef WEFTMOE_FWD_MAIN_INTERMEDIATE
#undef WEFTMOE_BWD_MAIN_HIDDEN
#undef WEFTMOE_BWD_MAIN_INTERMEDIATE
#undef WEFTMOE_FORWARD_TUNING_ID
#define WEFTMOE_FWD_MAIN_HIDDEN 2048
#define WEFTMOE_FWD_MAIN_INTERMEDIATE 512
#define WEFTMOE_BWD_MAIN_HIDDEN 2048
#define WEFTMOE_BWD_MAIN_INTERMEDIATE 512
#define WEFTMOE_FORWARD_TUNING_ID "dt-qwen35-probe-20261016"
