#pragma once

// Device communicators explicitly request GDAKI. Keep only that backend in
// device code, including calls made inside NCCL's barrier helpers.
// Leave host parsing defaults intact: NCCL disables GDAKI in the host pass
// and needs a backend to infer return types in its non-template wrappers.
#if defined(__CUDA_ARCH__)
#define NCCL_GIN_PROXY_ENABLE 0
#define NCCL_GIN_GPI_ENABLE 0
#define NCCL_GIN_EFA_GDA_ENABLE 0
#endif

#include <nccl_device.h>
