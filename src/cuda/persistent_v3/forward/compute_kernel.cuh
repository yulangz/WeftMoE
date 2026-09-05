#include "cuda/persistent_v3/forward/compute/preamble.cuh"

#if defined(CUTE_ARCH_MMA_SM90A_ENABLED)

#include "cuda/persistent_v3/forward/compute/task_selection.cuh"
#include "cuda/persistent_v3/forward/compute/pipeline_setup.inc"
#include "cuda/persistent_v3/forward/compute/fc1_pipeline.inc"
#include "cuda/persistent_v3/forward/compute/fc2_return_pipeline.inc"

#endif // CUTE_ARCH_MMA_SM90A_ENABLED

#include "cuda/persistent_v3/forward/compute/kernel_launchers.cuh"
