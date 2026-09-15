#pragma once

// The wire layout is shared by Forward, Backward and both worker layouts.
// This header can precede either phase's tuning header in a translation unit.
#if defined(WEFTMOE_FORWARD_USE_GENERATED_TUNING)
#include <weftmoe_forward_tuning_generated.hpp>
#elif defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING)
#include <weftmoe_backward_tuning_generated.hpp>
#endif

#ifndef WEFTMOE_GIN_RING_DEPTH
// Frozen generated headers, including those passed through -include, predate
// this control and used four physical slots. Preserve that layout when they
// are rebuilt. New generated configurations always define the value explicitly.
// Include this header before either phase supplies default model dimensions.
#if defined(WEFTMOE_FORWARD_USE_GENERATED_TUNING) || \
    defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING) || \
    defined(WEFTMOE_FWD_MAIN_HIDDEN) || defined(WEFTMOE_BWD_MAIN_HIDDEN)
#define WEFTMOE_GIN_RING_DEPTH 4
#else
#define WEFTMOE_GIN_RING_DEPTH 8
#endif
#endif

static_assert(WEFTMOE_GIN_RING_DEPTH == 4 || WEFTMOE_GIN_RING_DEPTH == 8 ||
                  WEFTMOE_GIN_RING_DEPTH == 16,
              "GIN physical ring depth must be 4, 8 or 16");
