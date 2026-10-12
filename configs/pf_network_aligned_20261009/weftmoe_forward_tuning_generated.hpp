#pragma once

// Cross-check both warp layouts at the same D32/R128 packet geometry.
#include "../pf_network_flash_20261009/weftmoe_forward_tuning_generated.hpp"
#undef WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS
#define WEFTMOE_FWD_STATIC_V4_RETURN_PACKET_ROWS 128
#undef WEFTMOE_FORWARD_TUNING_ID
#define WEFTMOE_FORWARD_TUNING_ID "pf-network-aligned-20261009"
