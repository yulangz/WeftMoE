#include "cuda/persistent_v3/forward/dispatch/preamble.cuh"

#if __has_include(<nccl_device.h>)

#include "cuda/persistent_v3/forward/dispatch/local_publication.cuh"
#include "cuda/persistent_v3/forward/dispatch/pack_and_relay.cuh"
#include "cuda/persistent_v3/forward/dispatch/inbound_publication.cuh"
#include "cuda/persistent_v3/forward/dispatch/local_relay_role.inc"
#include "cuda/persistent_v3/forward/dispatch/rdma_roles.inc"

#endif // __has_include(<nccl_device.h>)
