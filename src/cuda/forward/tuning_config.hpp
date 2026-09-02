#pragma once

#include <cstddef>
#include <cstdint>

// A generated autotune header defines the MEGAMOE_FWD_* macros before the
// defaults below.  The header is selected by the build system, not through the
// process environment, so every tuned binary contains exactly one auditable
// Forward configuration.
#if defined(MEGAMOE_FORWARD_USE_GENERATED_TUNING)
#include <megamoe_forward_tuning_generated.hpp>
#endif

#ifndef MEGAMOE_FWD_COMPILED_TUNING
#define MEGAMOE_FWD_COMPILED_TUNING 0
#endif
#ifndef MEGAMOE_FWD_ENABLE_LEGACY_ENV
#define MEGAMOE_FWD_ENABLE_LEGACY_ENV 1
#endif
#ifndef MEGAMOE_FWD_MAIN_HIDDEN
#define MEGAMOE_FWD_MAIN_HIDDEN 7168
#endif
#ifndef MEGAMOE_FWD_MAIN_INTERMEDIATE
#define MEGAMOE_FWD_MAIN_INTERMEDIATE 3072
#endif
#ifndef MEGAMOE_FWD_TILE_M
#define MEGAMOE_FWD_TILE_M 128
#endif
#ifndef MEGAMOE_FWD_FC1_TILE_N
#define MEGAMOE_FWD_FC1_TILE_N 128
#endif
#ifndef MEGAMOE_FWD_FC2_TILE_N
#define MEGAMOE_FWD_FC2_TILE_N 256
#endif
#ifndef MEGAMOE_FWD_TILE_K
#define MEGAMOE_FWD_TILE_K 64
#endif
#ifndef MEGAMOE_FWD_FC1_STAGES
#define MEGAMOE_FWD_FC1_STAGES 4
#endif
#ifndef MEGAMOE_FWD_FC2_STAGES
#define MEGAMOE_FWD_FC2_STAGES 4
#endif
#ifndef MEGAMOE_FWD_COMPUTE_THREADS
#define MEGAMOE_FWD_COMPUTE_THREADS 384
#endif
#ifndef MEGAMOE_FWD_MATH_WARPGROUPS
#define MEGAMOE_FWD_MATH_WARPGROUPS 2
#endif
#ifndef MEGAMOE_FWD_PRODUCER_WARPGROUPS
#define MEGAMOE_FWD_PRODUCER_WARPGROUPS 1
#endif
#ifndef MEGAMOE_FWD_MATH_REGISTERS
#define MEGAMOE_FWD_MATH_REGISTERS 232
#endif
#ifndef MEGAMOE_FWD_PRODUCER_REGISTERS
#define MEGAMOE_FWD_PRODUCER_REGISTERS 40
#endif
#ifndef MEGAMOE_FWD_REDUCE_REGISTERS
#define MEGAMOE_FWD_REDUCE_REGISTERS 168
#endif
#ifndef MEGAMOE_FWD_WGMMA_WAIT_GROUP
#define MEGAMOE_FWD_WGMMA_WAIT_GROUP 0
#endif
#ifndef MEGAMOE_FWD_FC2_STORE_ATOM_N
#define MEGAMOE_FWD_FC2_STORE_ATOM_N 64
#endif
#ifndef MEGAMOE_FWD_CLUSTER_SIZE
#define MEGAMOE_FWD_CLUSTER_SIZE 2
#endif
#ifndef MEGAMOE_FWD_RDMA_CTAS
#define MEGAMOE_FWD_RDMA_CTAS 8
#endif
#ifndef MEGAMOE_FWD_RELAY_CTAS
#define MEGAMOE_FWD_RELAY_CTAS 8
#endif
#ifndef MEGAMOE_FWD_RDMA_CONTROL_WARPS
#define MEGAMOE_FWD_RDMA_CONTROL_WARPS 2
#endif
#ifndef MEGAMOE_FWD_RELAY_TMA_WARPS
#define MEGAMOE_FWD_RELAY_TMA_WARPS 8
#endif
#ifndef MEGAMOE_FWD_LOCAL_DISPATCH_TMA_WARPS
#define MEGAMOE_FWD_LOCAL_DISPATCH_TMA_WARPS 8
#endif
#ifndef MEGAMOE_FWD_RELAY_SCATTER_WARPS
#define MEGAMOE_FWD_RELAY_SCATTER_WARPS 12
#endif
#ifndef MEGAMOE_FWD_LOCAL_RETURN_BATCH_MBLOCKS
#define MEGAMOE_FWD_LOCAL_RETURN_BATCH_MBLOCKS 4
#endif
#ifndef MEGAMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS
#define MEGAMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS 32
#endif
#ifndef MEGAMOE_FWD_RETURN_CHUNK_ROWS
#define MEGAMOE_FWD_RETURN_CHUNK_ROWS 256
#endif
#ifndef MEGAMOE_FWD_RETURN_MAX_INFLIGHT
#define MEGAMOE_FWD_RETURN_MAX_INFLIGHT 16
#endif
#ifndef MEGAMOE_FWD_LOCAL_ROUTE_BATCH_ROWS
#define MEGAMOE_FWD_LOCAL_ROUTE_BATCH_ROWS 128
#endif
#ifndef MEGAMOE_FWD_STARTUP_MBLOCKS
#define MEGAMOE_FWD_STARTUP_MBLOCKS 2
#endif
#ifndef MEGAMOE_FWD_FC1_LOCALITY_GROUP
#define MEGAMOE_FWD_FC1_LOCALITY_GROUP 2
#endif
#ifndef MEGAMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN
#define MEGAMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN 4
#endif
#ifndef MEGAMOE_FWD_FC1_LOCALITY_GROUP4_END
#define MEGAMOE_FWD_FC1_LOCALITY_GROUP4_END 12
#endif
#ifndef MEGAMOE_FWD_FC2_PRIORITY_BURST
#define MEGAMOE_FWD_FC2_PRIORITY_BURST 0
#endif
#ifndef MEGAMOE_FWD_TASK_CLAIM_CHUNK
#define MEGAMOE_FWD_TASK_CLAIM_CHUNK 1
#endif
#ifndef MEGAMOE_FWD_COMPLETION_CHUNKS_PER_PASS
#define MEGAMOE_FWD_COMPLETION_CHUNKS_PER_PASS 1
#endif
#ifndef MEGAMOE_FWD_COMPLETION_MARKER_WARPS
#define MEGAMOE_FWD_COMPLETION_MARKER_WARPS 1
#endif
#ifndef MEGAMOE_FWD_REDUCE_CTA_DIVISOR
#define MEGAMOE_FWD_REDUCE_CTA_DIVISOR 1
#endif
#ifndef MEGAMOE_FWD_REDUCE_SCHEDULER
// 0=ready queue, 1=dynamic scan, 2=static token stripes.
#define MEGAMOE_FWD_REDUCE_SCHEDULER 0
#endif
#ifndef MEGAMOE_FWD_REDUCE_WARPS
#define MEGAMOE_FWD_REDUCE_WARPS 12
#endif
#ifndef MEGAMOE_FWD_REDUCE_TOKENS_PER_CLAIM
#define MEGAMOE_FWD_REDUCE_TOKENS_PER_CLAIM 12
#endif
#ifndef MEGAMOE_FWD_TASK_IDLE_NANOSLEEP
#define MEGAMOE_FWD_TASK_IDLE_NANOSLEEP 64
#endif
#ifndef MEGAMOE_FWD_TRANSPORT_IDLE_NANOSLEEP
#define MEGAMOE_FWD_TRANSPORT_IDLE_NANOSLEEP 64
#endif
#ifndef MEGAMOE_FWD_RETURN_IDLE_NANOSLEEP
#define MEGAMOE_FWD_RETURN_IDLE_NANOSLEEP 32
#endif
#ifndef MEGAMOE_FWD_REDUCE_QUEUE_NANOSLEEP
#define MEGAMOE_FWD_REDUCE_QUEUE_NANOSLEEP 128
#endif
#ifndef MEGAMOE_FWD_SETUP_NANOSLEEP
#define MEGAMOE_FWD_SETUP_NANOSLEEP 256
#endif
#ifndef MEGAMOE_FWD_FINALIZE_NANOSLEEP
#define MEGAMOE_FWD_FINALIZE_NANOSLEEP 128
#endif
#ifndef MEGAMOE_FWD_FC2_TMA_STORE
#define MEGAMOE_FWD_FC2_TMA_STORE 1
#endif
#ifndef MEGAMOE_FWD_REGISTERED_OUTPUT
#define MEGAMOE_FWD_REGISTERED_OUTPUT 1
#endif
#ifndef MEGAMOE_FWD_RELAY_LOCAL_RETURN
#define MEGAMOE_FWD_RELAY_LOCAL_RETURN 1
#endif

namespace megamoe::cuda::forward_tuning {

inline constexpr bool kCompiledTuning = MEGAMOE_FWD_COMPILED_TUNING != 0;
inline constexpr bool kEnableLegacyEnvironment =
    MEGAMOE_FWD_ENABLE_LEGACY_ENV != 0;
inline constexpr std::uint32_t kMainHidden = MEGAMOE_FWD_MAIN_HIDDEN;
inline constexpr std::uint32_t kMainIntermediate =
    MEGAMOE_FWD_MAIN_INTERMEDIATE;
inline constexpr std::uint32_t kTileM = MEGAMOE_FWD_TILE_M;
inline constexpr std::uint32_t kFc1TileN = MEGAMOE_FWD_FC1_TILE_N;
inline constexpr std::uint32_t kFc2TileN = MEGAMOE_FWD_FC2_TILE_N;
inline constexpr std::uint32_t kTileK = MEGAMOE_FWD_TILE_K;
inline constexpr std::uint32_t kFc1Stages = MEGAMOE_FWD_FC1_STAGES;
inline constexpr std::uint32_t kFc2Stages = MEGAMOE_FWD_FC2_STAGES;
inline constexpr std::uint32_t kComputeThreads =
    MEGAMOE_FWD_COMPUTE_THREADS;
inline constexpr std::uint32_t kMathWarpgroups =
    MEGAMOE_FWD_MATH_WARPGROUPS;
inline constexpr std::uint32_t kProducerWarpgroups =
    MEGAMOE_FWD_PRODUCER_WARPGROUPS;
inline constexpr std::uint32_t kMathRegisters =
    MEGAMOE_FWD_MATH_REGISTERS;
inline constexpr std::uint32_t kProducerRegisters =
    MEGAMOE_FWD_PRODUCER_REGISTERS;
inline constexpr std::uint32_t kReduceRegisters =
    MEGAMOE_FWD_REDUCE_REGISTERS;
inline constexpr std::uint32_t kWgmmaWaitGroup =
    MEGAMOE_FWD_WGMMA_WAIT_GROUP;
inline constexpr std::uint32_t kFc2StoreAtomN =
    MEGAMOE_FWD_FC2_STORE_ATOM_N;
inline constexpr std::uint32_t kClusterSize = MEGAMOE_FWD_CLUSTER_SIZE;
inline constexpr std::uint32_t kRdmaCtas = MEGAMOE_FWD_RDMA_CTAS;
inline constexpr std::uint32_t kRelayCtas = MEGAMOE_FWD_RELAY_CTAS;
inline constexpr std::uint32_t kRdmaControlWarps =
    MEGAMOE_FWD_RDMA_CONTROL_WARPS;
inline constexpr std::uint32_t kRelayTmaWarps =
    MEGAMOE_FWD_RELAY_TMA_WARPS;
inline constexpr std::uint32_t kLocalDispatchTmaWarps =
    MEGAMOE_FWD_LOCAL_DISPATCH_TMA_WARPS;
inline constexpr std::uint32_t kRelayScatterWarps =
    MEGAMOE_FWD_RELAY_SCATTER_WARPS;
inline constexpr std::uint32_t kLocalReturnBatchMblocks =
    MEGAMOE_FWD_LOCAL_RETURN_BATCH_MBLOCKS;
inline constexpr std::uint32_t kDispatchPublishBatchRows =
    MEGAMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS;
inline constexpr std::uint32_t kReturnChunkRows =
    MEGAMOE_FWD_RETURN_CHUNK_ROWS;
inline constexpr std::uint32_t kReturnMaxInflight =
    MEGAMOE_FWD_RETURN_MAX_INFLIGHT;
inline constexpr std::uint32_t kLocalRouteBatchRows =
    MEGAMOE_FWD_LOCAL_ROUTE_BATCH_ROWS;
inline constexpr std::uint32_t kStartupMblocks =
    MEGAMOE_FWD_STARTUP_MBLOCKS;
inline constexpr std::uint32_t kFc1LocalityGroup =
    MEGAMOE_FWD_FC1_LOCALITY_GROUP;
inline constexpr std::uint32_t kFc1LocalityGroup4Begin =
    MEGAMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN;
inline constexpr std::uint32_t kFc1LocalityGroup4End =
    MEGAMOE_FWD_FC1_LOCALITY_GROUP4_END;
// Zero means strict, unbounded FC2-first scheduling.
inline constexpr std::uint32_t kFc2PriorityBurst =
    MEGAMOE_FWD_FC2_PRIORITY_BURST;
inline constexpr std::uint32_t kTaskClaimChunk =
    MEGAMOE_FWD_TASK_CLAIM_CHUNK;
inline constexpr std::uint32_t kCompletionChunksPerPass =
    MEGAMOE_FWD_COMPLETION_CHUNKS_PER_PASS;
inline constexpr std::uint32_t kCompletionMarkerWarps =
    MEGAMOE_FWD_COMPLETION_MARKER_WARPS;
inline constexpr std::uint32_t kReduceCtaDivisor =
    MEGAMOE_FWD_REDUCE_CTA_DIVISOR;
inline constexpr std::uint32_t kReduceScheduler =
    MEGAMOE_FWD_REDUCE_SCHEDULER;
inline constexpr std::uint32_t kReduceWarps = MEGAMOE_FWD_REDUCE_WARPS;
inline constexpr std::uint32_t kReduceTokensPerClaim =
    MEGAMOE_FWD_REDUCE_TOKENS_PER_CLAIM;
inline constexpr std::uint32_t kTaskIdleNanosleep =
    MEGAMOE_FWD_TASK_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kTransportIdleNanosleep =
    MEGAMOE_FWD_TRANSPORT_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReturnIdleNanosleep =
    MEGAMOE_FWD_RETURN_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReduceQueueNanosleep =
    MEGAMOE_FWD_REDUCE_QUEUE_NANOSLEEP;
inline constexpr std::uint32_t kSetupNanosleep =
    MEGAMOE_FWD_SETUP_NANOSLEEP;
inline constexpr std::uint32_t kFinalizeNanosleep =
    MEGAMOE_FWD_FINALIZE_NANOSLEEP;
inline constexpr bool kFc2TmaStore = MEGAMOE_FWD_FC2_TMA_STORE != 0;
inline constexpr bool kRegisteredOutput =
    MEGAMOE_FWD_REGISTERED_OUTPUT != 0;
inline constexpr bool kRelayLocalReturn =
    MEGAMOE_FWD_RELAY_LOCAL_RETURN != 0;

constexpr std::size_t max_size(std::size_t a, std::size_t b) {
  return a > b ? a : b;
}

inline constexpr std::size_t kAStageElements =
    static_cast<std::size_t>(kTileM) * kTileK;
inline constexpr std::size_t kFc1BStageElements =
    static_cast<std::size_t>(kFc1TileN) * kTileK;
inline constexpr std::size_t kFc2BStageElements =
    static_cast<std::size_t>(kFc2TileN) * kTileK;
inline constexpr std::size_t kAStorageElements =
    max_size(kFc1Stages, kFc2Stages) * kAStageElements;
inline constexpr std::size_t kBStorageElements = max_size(
    2U * kFc1Stages * kFc1BStageElements,
    kFc2Stages * kFc2BStageElements);
inline constexpr std::size_t kGemmStorageBytes =
    (kAStorageElements + kBStorageElements) * sizeof(std::uint16_t);
inline constexpr std::size_t kBarrierCount =
    3U * kFc1Stages + 2U * kFc2Stages;

inline constexpr std::size_t compute_smem_bytes(std::uint32_t world) {
  return kGemmStorageBytes + kBarrierCount * sizeof(std::uint64_t) +
         static_cast<std::size_t>(world) * sizeof(std::uint32_t);
}

inline constexpr std::size_t transport_smem_bytes(std::uint32_t hidden,
                                                   std::uint32_t world) {
  const auto streams = max_size(kLocalDispatchTmaWarps, kRelayTmaWarps);
  const auto row_staging = streams * static_cast<std::size_t>(hidden) *
                           sizeof(std::uint16_t);
  return max_size(kGemmStorageBytes, row_staging) +
         kBarrierCount * sizeof(std::uint64_t) +
         static_cast<std::size_t>(world) * sizeof(std::uint32_t);
}

static_assert(kTileM == 64U || kTileM == 128U);
static_assert(kFc1TileN >= 64U && kFc1TileN <= 256U &&
              kFc1TileN % 64U == 0U);
static_assert(kFc2TileN >= 64U && kFc2TileN <= 256U &&
              kFc2TileN % 64U == 0U);
// The Hopper K-major 128-byte swizzle currently fixes one async stage at K64.
// It remains an explicit singleton hyperparameter until another descriptor
// layout is added and validated.
static_assert(kTileK == 64U);
static_assert(kFc1Stages >= 2U && kFc1Stages <= 4U);
static_assert(kFc2Stages >= 2U && kFc2Stages <= 4U);
static_assert(kComputeThreads == 384U);
static_assert(kMathWarpgroups == 2U && kProducerWarpgroups == 1U);
static_assert(kMathRegisters <= 255U && kProducerRegisters <= 255U &&
              kReduceRegisters <= 255U);
static_assert((kMathWarpgroups * kMathRegisters +
               kProducerWarpgroups * kProducerRegisters) *
                  128U <=
              65536U);
static_assert(kComputeThreads * kReduceRegisters <= 65536U);
static_assert(kWgmmaWaitGroup <= 1U);
static_assert(kFc2StoreAtomN == 64U);
static_assert(!kFc2TmaStore || kTileM == 128U);
static_assert(kClusterSize == 1U || kClusterSize == 2U);
static_assert(kRdmaCtas > 0U && kRelayCtas >= kRdmaCtas &&
              kRelayCtas % kRdmaCtas == 0U);
static_assert(kRdmaControlWarps >= 2U &&
              kRdmaControlWarps < kComputeThreads / 32U);
static_assert(kRelayTmaWarps <= kComputeThreads / 32U);
static_assert(kLocalDispatchTmaWarps > 0U &&
              kLocalDispatchTmaWarps <= kComputeThreads / 32U);
static_assert(kRelayScatterWarps > 0U &&
              kRelayScatterWarps <= kComputeThreads / 32U);
static_assert(kLocalReturnBatchMblocks >= 1U &&
              kLocalReturnBatchMblocks <= 8U);
static_assert(kReturnChunkRows > 0U && kReturnMaxInflight <= 64U);
static_assert(kLocalRouteBatchRows > 0U && kStartupMblocks > 0U);
static_assert(kTaskClaimChunk == 1U,
              "only the proven one-task queue claim is currently compiled");
static_assert(kCompletionChunksPerPass > 0U &&
              kCompletionChunksPerPass <= 4U);
static_assert(kCompletionMarkerWarps > 0U &&
              kCompletionMarkerWarps <= kComputeThreads / 32U);
static_assert(kReduceCtaDivisor > 0U);
static_assert(kReduceScheduler <= 2U);
static_assert(kReduceWarps > 0U &&
              kReduceWarps <= kComputeThreads / 32U);
static_assert(kReduceTokensPerClaim > 0U &&
              kReduceTokensPerClaim <= kReduceWarps);

} // namespace megamoe::cuda::forward_tuning
