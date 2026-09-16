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

#ifndef WEFTMOE_GIN_DISPATCH_CHANNELS
#define WEFTMOE_GIN_DISPATCH_CHANNELS 64
#endif
static_assert(WEFTMOE_GIN_DISPATCH_CHANNELS > 0,
              "GIN Dispatch channels must be positive");

#ifndef WEFTMOE_GIN_CHUNK_ROWS
#define WEFTMOE_GIN_CHUNK_ROWS 64
#endif
static_assert(WEFTMOE_GIN_CHUNK_ROWS > 0 && WEFTMOE_GIN_CHUNK_ROWS <= 128,
              "GIN packets must contain 1 to 128 rows");

#ifndef WEFTMOE_GIN_CONTEXT_COUNT
#define WEFTMOE_GIN_CONTEXT_COUNT 0
#endif
static_assert(WEFTMOE_GIN_CONTEXT_COUNT >= 0, "GIN context count must be nonnegative");

#ifndef WEFTMOE_GIN_RETURN_PORTS
#define WEFTMOE_GIN_RETURN_PORTS 8
#endif
static_assert(WEFTMOE_GIN_RETURN_PORTS > 0, "GIN Return streams must be positive");
