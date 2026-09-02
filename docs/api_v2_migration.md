# API v2 migration

API v2 uses owning, typed CUDA leases rather than raw pointers or public kernel
PODs. Callers should:

1. create one Notify pool and one activation pool for the maximum shape;
2. publish router routes, gates, activations, and weights through
   `ExternalCudaProducer`;
3. run `CudaNotifyGenerationCoordinator::run_device` for the exact generation;
4. select `LocalCudaP2pForwardMegaKernel` or `GinRdmaForwardMegaKernel`;
5. retain the returned `DeviceForwardHandle` until its output consumer has
   completed.

Generic device buffers cannot be rebound as route or activation inputs.
Session, generation, device, execution stream, shape, layout, and allocation
owner are checked at every handoff. Test upload/download helpers are private and
are not installed.

The old single-node fallback, runtime-shape entry points, Forward cache modes,
and Backward handles/configuration have been removed. Code that conditionally
supports training should include `megamoe/cuda/backward.hpp` and require
`backward_pipeline_available()` before constructing a training path; it returns
false in this revision.
