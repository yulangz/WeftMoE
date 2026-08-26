#pragma once

// Repository-internal declarations for validation downloads, synthetic
// transports, fault seams and fixture factories.  Installed clients never add
// src/ to their include path and therefore cannot enable this API surface.
#ifndef MEGAMOE_INTERNAL_TESTING_API
#define MEGAMOE_INTERNAL_TESTING_API 1
#endif

#include "megamoe/cuda/activation_transport.hpp"
#include "megamoe/cuda/device_notify.hpp"
#include "megamoe/cuda/nccl_count_exchange.hpp"
