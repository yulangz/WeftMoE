#pragma once

// Reuse the validated multi-node V3 Qwen GEMM/communication profile.
// H/I match the V4 matrix; E=64 and top-k=4 are supplied by the fixture.
#include "../communication_spsc_20260916/qwen.hpp"
