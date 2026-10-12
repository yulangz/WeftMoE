#pragma once
#include "../pf_network_l132_20261009/weftmoe_forward_tuning_generated.hpp"
#undef WEFTMOE_BWD_ACTIVATION_TILE_K
#define WEFTMOE_BWD_ACTIVATION_TILE_K 128
#undef WEFTMOE_BWD_DAB_STAGES
#define WEFTMOE_BWD_DAB_STAGES 2
#undef WEFTMOE_BWD_DX_STAGES
#define WEFTMOE_BWD_DX_STAGES 2
