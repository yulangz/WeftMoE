#pragma once

// Same GEMM as the measured single-node Flash PF. Only network packet lanes
// change; worker placement is selected by WEFTMOE_PF_NETWORK_WORKERS (1/2).
#include "../pf_inference_flash_20261008/weftmoe_forward_tuning_generated.hpp"
#undef WEFTMOE_FWD_STATIC_V4_NETWORK_LANES
#define WEFTMOE_FWD_STATIC_V4_NETWORK_LANES 16
#undef WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS
#define WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS 16
#undef WEFTMOE_FORWARD_TUNING_ID
#define WEFTMOE_FORWARD_TUNING_ID "pf-network-flash-20261009"
