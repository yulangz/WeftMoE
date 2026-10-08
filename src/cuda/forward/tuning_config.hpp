#pragma once

#ifndef WEFTMOE_FWD_REMOTE_FC1_PRIORITY
#define WEFTMOE_FWD_REMOTE_FC1_PRIORITY 0
#endif

#include <cstddef>
#include <cstdint>
#include "cuda/communication/tuning_config.hpp"

// A generated autotune header defines the WEFTMOE_FWD_* macros before the
// defaults below.  The header is selected by the build system, not through the
// process environment, so every tuned binary contains exactly one auditable
// Forward configuration.
#if defined(WEFTMOE_FORWARD_USE_GENERATED_TUNING)
#include <weftmoe_forward_tuning_generated.hpp>
#elif defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING)
#include <weftmoe_backward_tuning_generated.hpp>
#define WEFTMOE_FWD_MAIN_HIDDEN WEFTMOE_BWD_MAIN_HIDDEN
#define WEFTMOE_FWD_MAIN_INTERMEDIATE WEFTMOE_BWD_MAIN_INTERMEDIATE
#endif

#include "cuda/unified_worker_tuning.hpp"

#ifndef WEFTMOE_FWD_MAIN_HIDDEN
#define WEFTMOE_FWD_MAIN_HIDDEN 7168
#endif
#ifndef WEFTMOE_FWD_MAIN_INTERMEDIATE
#define WEFTMOE_FWD_MAIN_INTERMEDIATE 3072
#endif
#ifndef WEFTMOE_FWD_TILE_M
#define WEFTMOE_FWD_TILE_M 128
#endif
#ifndef WEFTMOE_FWD_FC1_TILE_N
#define WEFTMOE_FWD_FC1_TILE_N 128
#endif
#ifndef WEFTMOE_FWD_FC2_TILE_N
#define WEFTMOE_FWD_FC2_TILE_N 256
#endif
#ifndef WEFTMOE_FWD_TILE_K
#define WEFTMOE_FWD_TILE_K 64
#endif
#ifndef WEFTMOE_FWD_FC1_STAGES
#define WEFTMOE_FWD_FC1_STAGES 4
#endif
#ifndef WEFTMOE_FWD_FC2_STAGES
#define WEFTMOE_FWD_FC2_STAGES 4
#endif
#ifndef WEFTMOE_FWD_FC2_CLAIM_TILES
#define WEFTMOE_FWD_FC2_CLAIM_TILES 1
#endif
#ifndef WEFTMOE_FWD_OPPORTUNISTIC_REDUCE_WARPS
#define WEFTMOE_FWD_OPPORTUNISTIC_REDUCE_WARPS 1
#endif
#ifndef WEFTMOE_FWD_GIN_RECEIVE_ROWS
#define WEFTMOE_FWD_GIN_RECEIVE_ROWS 128
#endif
#ifndef WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT
#define WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT 3
#endif
#ifndef WEFTMOE_FWD_COMPUTE_THREADS
#define WEFTMOE_FWD_COMPUTE_THREADS 384
#endif
#ifndef WEFTMOE_FWD_MATH_WARPGROUPS
#define WEFTMOE_FWD_MATH_WARPGROUPS 2
#endif
#ifndef WEFTMOE_FWD_PRODUCER_WARPGROUPS
#define WEFTMOE_FWD_PRODUCER_WARPGROUPS 1
#endif
#ifndef WEFTMOE_FWD_MATH_REGISTERS
#define WEFTMOE_FWD_MATH_REGISTERS 232
#endif
#ifndef WEFTMOE_FWD_PRODUCER_REGISTERS
#define WEFTMOE_FWD_PRODUCER_REGISTERS 40
#endif
#ifndef WEFTMOE_FWD_REDUCE_REGISTERS
#define WEFTMOE_FWD_REDUCE_REGISTERS 168
#endif
#ifndef WEFTMOE_FWD_WGMMA_WAIT_GROUP
#define WEFTMOE_FWD_WGMMA_WAIT_GROUP 0
#endif
#ifndef WEFTMOE_FWD_FC2_STORE_ATOM_N
#define WEFTMOE_FWD_FC2_STORE_ATOM_N 64
#endif
#ifndef WEFTMOE_FWD_CLUSTER_SIZE
#define WEFTMOE_FWD_CLUSTER_SIZE 2
#endif
#ifndef WEFTMOE_FWD_RDMA_CTAS
#define WEFTMOE_FWD_RDMA_CTAS 8
#endif
#ifndef WEFTMOE_FWD_RELAY_TMA_WARPS
#if WEFTMOE_FWD_RDMA_CTAS == 0
#define WEFTMOE_FWD_RELAY_TMA_WARPS 12
#else
#define WEFTMOE_FWD_RELAY_TMA_WARPS 8
#endif
#endif
#ifndef WEFTMOE_FWD_LOCAL_DISPATCH_TMA_WARPS
#define WEFTMOE_FWD_LOCAL_DISPATCH_TMA_WARPS 12
#endif
#ifndef WEFTMOE_FWD_RELAY_SCATTER_WARPS
#define WEFTMOE_FWD_RELAY_SCATTER_WARPS 12
#endif
#ifndef WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS
#if WEFTMOE_FWD_RDMA_CTAS == 0
#define WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS 256
#else
#define WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS 512
#endif
#endif
#ifndef WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS
#if WEFTMOE_FWD_RDMA_CTAS == 0
#define WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS 16
#else
#define WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS 32
#endif
#endif
#ifndef WEFTMOE_FWD_LOCAL_ROUTE_BATCH_ROWS
#define WEFTMOE_FWD_LOCAL_ROUTE_BATCH_ROWS 128
#endif
#ifndef WEFTMOE_FWD_STARTUP_MBLOCKS
#define WEFTMOE_FWD_STARTUP_MBLOCKS 2
#endif
#ifndef WEFTMOE_FWD_FC1_LOCALITY_GROUP
#define WEFTMOE_FWD_FC1_LOCALITY_GROUP 2
#endif
#ifndef WEFTMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN
#define WEFTMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN 4
#endif
#ifndef WEFTMOE_FWD_FC1_LOCALITY_GROUP4_END
#define WEFTMOE_FWD_FC1_LOCALITY_GROUP4_END 12
#endif
#ifndef WEFTMOE_FWD_FC2_PRIORITY_BURST
#define WEFTMOE_FWD_FC2_PRIORITY_BURST 0
#endif
#ifndef WEFTMOE_FWD_TASK_CLAIM_CHUNK
#define WEFTMOE_FWD_TASK_CLAIM_CHUNK 1
#endif
#ifndef WEFTMOE_FWD_COMPLETION_PEER_PROBES
#define WEFTMOE_FWD_COMPLETION_PEER_PROBES 1
#endif
#ifndef WEFTMOE_FWD_REDUCE_SCHEDULER
// 0=ready queue, 1=dynamic scan, 2=static token stripes.
#define WEFTMOE_FWD_REDUCE_SCHEDULER 0
#endif
#ifndef WEFTMOE_FWD_REDUCE_WARPS
#define WEFTMOE_FWD_REDUCE_WARPS 12
#endif
#ifndef WEFTMOE_FWD_REDUCE_TOKENS_PER_CLAIM
#define WEFTMOE_FWD_REDUCE_TOKENS_PER_CLAIM 12
#endif
#ifndef WEFTMOE_FWD_TASK_IDLE_NANOSLEEP
#define WEFTMOE_FWD_TASK_IDLE_NANOSLEEP 64
#endif
#ifndef WEFTMOE_FWD_TRANSPORT_IDLE_NANOSLEEP
#define WEFTMOE_FWD_TRANSPORT_IDLE_NANOSLEEP 64
#endif
#ifndef WEFTMOE_FWD_RETURN_IDLE_NANOSLEEP
#define WEFTMOE_FWD_RETURN_IDLE_NANOSLEEP 32
#endif
#ifndef WEFTMOE_FWD_REDUCE_QUEUE_NANOSLEEP
#define WEFTMOE_FWD_REDUCE_QUEUE_NANOSLEEP 128
#endif
#ifndef WEFTMOE_FWD_SETUP_NANOSLEEP
#define WEFTMOE_FWD_SETUP_NANOSLEEP 256
#endif
#ifndef WEFTMOE_FWD_FINALIZE_NANOSLEEP
#define WEFTMOE_FWD_FINALIZE_NANOSLEEP 128
#endif
#ifndef WEFTMOE_FWD_FC2_TMA_STORE
#define WEFTMOE_FWD_FC2_TMA_STORE 1
#endif
#ifndef WEFTMOE_FWD_RELAY_LOCAL_RETURN
#define WEFTMOE_FWD_RELAY_LOCAL_RETURN 1
#endif

namespace weftmoe::cuda::forward_tuning {
// Both topologies need enough workers for local copies and received-chunk
// expansion. Explicit tuning overrides apply unchanged.
#ifndef WEFTMOE_FWD_RELAY_CTAS
#define WEFTMOE_FWD_RELAY_CTAS 24
inline constexpr std::uint32_t kRemoteRelayCtas = 24;
#else
inline constexpr std::uint32_t kRemoteRelayCtas = WEFTMOE_FWD_RELAY_CTAS;
#endif

// Unified workers are a compile-time layout choice shared by every shape.
#ifndef WEFTMOE_FWD_UNIFIED_WORKERS
// Bare builds use unified workers. Generated plans select each phase
// explicitly; a missing phase switch preserves historical standalone builds.
#if defined(WEFTMOE_FORWARD_USE_GENERATED_TUNING) || \
    defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING)
#define WEFTMOE_FWD_UNIFIED_WORKERS 0
#else
#define WEFTMOE_FWD_UNIFIED_WORKERS 1
#endif
#endif
// Communication ownership only; every physical CTA retains its GEMM roles.
// Zero keeps the mixed worker layout for controlled comparisons.
#ifndef WEFTMOE_FWD_UNIFIED_RDMA_CTAS
#define WEFTMOE_FWD_UNIFIED_RDMA_CTAS 16
#endif
inline constexpr unsigned kUnifiedRdmaCtas = WEFTMOE_FWD_UNIFIED_RDMA_CTAS;
static_assert(WEFTMOE_FWD_UNIFIED_RDMA_CTAS >= 0);
inline constexpr unsigned unified_rdma_ctas(unsigned total, bool use_gin) {
  return !use_gin || total < 2U ? 0U
         : (kUnifiedRdmaCtas < total ? kUnifiedRdmaCtas : total - 1U);
}

inline constexpr unsigned kUnifiedGinPackSlots = WEFTMOE_UNIFIED_GIN_PACK_SLOTS;
// A sender has 40 bytes of private progress state and each packing slot has a
// 40-byte handoff. Round the allocation to the device structure's alignment.
inline constexpr unsigned kUnifiedGinStateBytes =
    ((40U + 40U * kUnifiedGinPackSlots + 127U) / 128U) * 128U;
static_assert(kUnifiedGinPackSlots >= 1U && kUnifiedGinPackSlots <= 8U);
using UnifiedResources = unified_worker_tuning::Resources<
    WEFTMOE_FWD_UNIFIED_THREADS, WEFTMOE_FWD_UNIFIED_COMM_THREADS,
    WEFTMOE_FWD_UNIFIED_COMM_BASE, WEFTMOE_FWD_UNIFIED_MATH_REGISTERS,
    WEFTMOE_FWD_UNIFIED_CONTROL_REGISTERS, WEFTMOE_FWD_UNIFIED_PRODUCER_REGISTERS,
    WEFTMOE_FWD_UNIFIED_SEGMENT_BYTES, WEFTMOE_FWD_UNIFIED_COPY_BUFFERS>;
inline constexpr unsigned kUnifiedThreads = UnifiedResources::kUnifiedThreads;
inline constexpr unsigned kUnifiedCommThreads = UnifiedResources::kUnifiedCommThreads;
inline constexpr unsigned kUnifiedCommBase = UnifiedResources::kUnifiedCommBase;
inline constexpr unsigned kUnifiedMathRegisters = UnifiedResources::kUnifiedMathRegisters;
inline constexpr unsigned kUnifiedControlRegisters = UnifiedResources::kUnifiedControlRegisters;
inline constexpr unsigned kUnifiedProducerRegisters = UnifiedResources::kUnifiedProducerRegisters;
inline constexpr unsigned kUnifiedSegmentBytes = UnifiedResources::kUnifiedSegmentBytes;
inline constexpr unsigned kUnifiedCopyBuffers = UnifiedResources::kUnifiedCopyBuffers;
inline constexpr unsigned kGinReceiveRows = WEFTMOE_FWD_GIN_RECEIVE_ROWS;
static_assert(kGinReceiveRows >= 1U && kGinReceiveRows <= 128U);
inline constexpr unsigned kGinChannelPackLimit =
    WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT;
static_assert(kGinChannelPackLimit >= 1U && kGinChannelPackLimit <= 8U);
inline constexpr unsigned kUnifiedInitialRegisters = UnifiedResources::kUnifiedInitialRegisters;
inline constexpr unsigned kUnifiedRegisterPool = UnifiedResources::kUnifiedRegisterPool;
inline constexpr bool kUnifiedRowPrefetch = WEFTMOE_FWD_UNIFIED_ROW_PREFETCH != 0;
// Local scheduling and cross-row prefetch remain independent of GIN packing.
inline constexpr bool kUnifiedOptimizeGinLocalTransport =
    WEFTMOE_FWD_UNIFIED_OPTIMIZE_GIN_LOCAL_TRANSPORT != 0;
inline constexpr unsigned kUnifiedReturnDrainRows =
    WEFTMOE_FWD_UNIFIED_RETURN_DRAIN_ROWS;
static_assert(kUnifiedReturnDrainRows >= 32U &&
              kUnifiedReturnDrainRows <= 1024U &&
              kUnifiedReturnDrainRows % 32U == 0U);
inline constexpr bool kUnifiedSplitNvlink = WEFTMOE_FWD_UNIFIED_SPLIT_NVLINK != 0;
inline constexpr bool kUnifiedWorkers = WEFTMOE_FWD_UNIFIED_WORKERS != 0;

// V4 keeps a distinct resource type so later tuning cannot silently change
// the static entry's register contract.  Cold scheduling and transport state
// live in noinline device routines/shared memory. CUDA 12.9 requires the same
// 216/64 entry split as V3 even though the static worker loops carry less
// control state.
#ifndef WEFTMOE_FWD_STATIC_V4_MATH_REGISTERS
#define WEFTMOE_FWD_STATIC_V4_MATH_REGISTERS 216
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_CONTROL_REGISTERS
#define WEFTMOE_FWD_STATIC_V4_CONTROL_REGISTERS 64
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_FC1_STAGES
#define WEFTMOE_FWD_STATIC_V4_FC1_STAGES WEFTMOE_FWD_FC1_STAGES
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_FC2_STAGES
#define WEFTMOE_FWD_STATIC_V4_FC2_STAGES WEFTMOE_FWD_FC2_STAGES
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS
#define WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS 0
#endif
#ifndef WEFTMOE_FWD_STATIC_V4_CTA_COUNT
#define WEFTMOE_FWD_STATIC_V4_CTA_COUNT 0
#endif
#ifndef WEFTMOE_FWD_LOCAL_PREFILL_STREAMING
#define WEFTMOE_FWD_LOCAL_PREFILL_STREAMING 0
#endif
inline constexpr bool kLocalPrefillStreaming = WEFTMOE_FWD_LOCAL_PREFILL_STREAMING;
inline constexpr unsigned kStaticV4CtaCount = WEFTMOE_FWD_STATIC_V4_CTA_COUNT;
static_assert(kStaticV4CtaCount % 2U == 0U);
#ifndef WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS
#define WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS \
  WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS
#endif
inline constexpr unsigned kStaticV4PrefillMinTokens =
    WEFTMOE_FWD_STATIC_V4_PREFILL_MIN_TOKENS;
inline constexpr unsigned kStaticV4NetworkPrefillMinTokens =
    WEFTMOE_FWD_STATIC_V4_NETWORK_PREFILL_MIN_TOKENS;
inline constexpr bool kStaticV4PrefillEnabled =
    kStaticV4PrefillMinTokens || kStaticV4NetworkPrefillMinTokens;
inline constexpr unsigned kStaticV4MathRegisters =
    WEFTMOE_FWD_STATIC_V4_MATH_REGISTERS;
inline constexpr unsigned kStaticV4ControlRegisters =
    WEFTMOE_FWD_STATIC_V4_CONTROL_REGISTERS;
inline constexpr unsigned kStaticV4TileK = WEFTMOE_FWD_TILE_K;
inline constexpr unsigned kStaticV4Fc1Stages =
    WEFTMOE_FWD_STATIC_V4_FC1_STAGES;
inline constexpr unsigned kStaticV4Fc2Stages =
    WEFTMOE_FWD_STATIC_V4_FC2_STAGES;
using StaticV4Resources = unified_worker_tuning::Resources<
    WEFTMOE_FWD_UNIFIED_THREADS, WEFTMOE_FWD_UNIFIED_COMM_THREADS,
    WEFTMOE_FWD_UNIFIED_COMM_BASE, WEFTMOE_FWD_STATIC_V4_MATH_REGISTERS,
    WEFTMOE_FWD_STATIC_V4_CONTROL_REGISTERS,
    WEFTMOE_FWD_STATIC_V4_CONTROL_REGISTERS,
    WEFTMOE_FWD_UNIFIED_SEGMENT_BYTES,
    WEFTMOE_FWD_UNIFIED_COPY_BUFFERS>;

inline constexpr std::uint32_t kMainHidden = WEFTMOE_FWD_MAIN_HIDDEN;
inline constexpr std::uint32_t kMainIntermediate =
    WEFTMOE_FWD_MAIN_INTERMEDIATE;
inline constexpr std::uint32_t kTileM = WEFTMOE_FWD_TILE_M;
inline constexpr std::uint32_t kFc1TileN = WEFTMOE_FWD_FC1_TILE_N;
inline constexpr std::uint32_t kFc2TileN = WEFTMOE_FWD_FC2_TILE_N;
// Training and the general multi-node path retain their proven K64 layout.
// The public tile_k axis belongs to the V4 single-node inference entry, whose
// packed W1 path can independently tune one or two K64 swizzle atoms/stage.
inline constexpr std::uint32_t kTileK = 64U;
inline constexpr std::uint32_t kFc1Stages = WEFTMOE_FWD_FC1_STAGES;
inline constexpr std::uint32_t kFc2Stages = WEFTMOE_FWD_FC2_STAGES;
inline constexpr std::uint32_t kFc2ClaimTiles = WEFTMOE_FWD_FC2_CLAIM_TILES;
inline constexpr std::uint32_t kOpportunisticReduceWarps =
    WEFTMOE_FWD_OPPORTUNISTIC_REDUCE_WARPS;
static_assert(kOpportunisticReduceWarps >= 1U &&
              kOpportunisticReduceWarps <= 8U);
inline constexpr std::uint32_t kComputeThreads =
    WEFTMOE_FWD_COMPUTE_THREADS;
// Local transport has no WGMMA register partition. Four warpgroups expose
// more independent peer-copy streams while fitting one CTA per Hopper SM.
inline constexpr std::uint32_t kLocalTransportThreads = 512U;
inline constexpr std::uint32_t kMathWarpgroups =
    WEFTMOE_FWD_MATH_WARPGROUPS;
inline constexpr std::uint32_t kProducerWarpgroups =
    WEFTMOE_FWD_PRODUCER_WARPGROUPS;
inline constexpr std::uint32_t kMathRegisters =
    WEFTMOE_FWD_MATH_REGISTERS;
inline constexpr std::uint32_t kProducerRegisters =
    WEFTMOE_FWD_PRODUCER_REGISTERS;
inline constexpr std::uint32_t kReduceRegisters =
    WEFTMOE_FWD_REDUCE_REGISTERS;
inline constexpr std::uint32_t kWgmmaWaitGroup =
    WEFTMOE_FWD_WGMMA_WAIT_GROUP;
inline constexpr std::uint32_t kFc2StoreAtomN =
    WEFTMOE_FWD_FC2_STORE_ATOM_N;
inline constexpr std::uint32_t kClusterSize = WEFTMOE_FWD_CLUSTER_SIZE;
inline constexpr std::uint32_t kRdmaCtas = WEFTMOE_FWD_RDMA_CTAS;
inline constexpr std::uint32_t kRelayCtas = WEFTMOE_FWD_RELAY_CTAS;
inline constexpr std::uint32_t kRelayTmaWarps = WEFTMOE_FWD_RELAY_TMA_WARPS;
inline constexpr std::uint32_t kLocalDispatchTmaWarps =
    WEFTMOE_FWD_LOCAL_DISPATCH_TMA_WARPS;
inline constexpr std::uint32_t kRelayScatterWarps =
    WEFTMOE_FWD_RELAY_SCATTER_WARPS;
inline constexpr std::uint32_t kLocalReturnBatchRows =
    WEFTMOE_FWD_LOCAL_RETURN_BATCH_ROWS;
inline constexpr std::uint32_t kDispatchPublishBatchRows =
    WEFTMOE_FWD_DISPATCH_PUBLISH_BATCH_ROWS;
inline constexpr std::uint32_t kLocalRouteBatchRows =
    WEFTMOE_FWD_LOCAL_ROUTE_BATCH_ROWS;
inline constexpr std::uint32_t kStartupMblocks =
    WEFTMOE_FWD_STARTUP_MBLOCKS;
inline constexpr std::uint32_t kFc1LocalityGroup =
    WEFTMOE_FWD_FC1_LOCALITY_GROUP;
inline constexpr std::uint32_t kFc1LocalityGroup4Begin =
    WEFTMOE_FWD_FC1_LOCALITY_GROUP4_BEGIN;
inline constexpr std::uint32_t kFc1LocalityGroup4End =
    WEFTMOE_FWD_FC1_LOCALITY_GROUP4_END;
// Zero means strict, unbounded FC2-first scheduling.
inline constexpr std::uint32_t kFc2PriorityBurst =
    WEFTMOE_FWD_FC2_PRIORITY_BURST;
inline constexpr std::uint32_t kTaskClaimChunk =
    WEFTMOE_FWD_TASK_CLAIM_CHUNK;
inline constexpr std::uint32_t kCompletionPeerProbes =
    WEFTMOE_FWD_COMPLETION_PEER_PROBES;
inline constexpr std::uint32_t kReduceScheduler =
    WEFTMOE_FWD_REDUCE_SCHEDULER;
inline constexpr std::uint32_t kReduceWarps = WEFTMOE_FWD_REDUCE_WARPS;
inline constexpr std::uint32_t kReduceTokensPerClaim =
    WEFTMOE_FWD_REDUCE_TOKENS_PER_CLAIM;
inline constexpr std::uint32_t kTaskIdleNanosleep =
    WEFTMOE_FWD_TASK_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kTransportIdleNanosleep =
    WEFTMOE_FWD_TRANSPORT_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReturnIdleNanosleep =
    WEFTMOE_FWD_RETURN_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReduceQueueNanosleep =
    WEFTMOE_FWD_REDUCE_QUEUE_NANOSLEEP;
inline constexpr std::uint32_t kSetupNanosleep =
    WEFTMOE_FWD_SETUP_NANOSLEEP;
inline constexpr std::uint32_t kFinalizeNanosleep =
    WEFTMOE_FWD_FINALIZE_NANOSLEEP;
inline constexpr bool kFc2TmaStore = WEFTMOE_FWD_FC2_TMA_STORE != 0;
inline constexpr bool kRelayLocalReturn = WEFTMOE_FWD_RELAY_LOCAL_RETURN != 0;

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
    max_size(kAStorageElements + kBStorageElements,
             kFc2Stages * (kAStageElements + kFc2BStageElements) +
                 (kFc2ClaimTiles > 1U ? kTileM * kFc2TileN : 0U)) *
    sizeof(std::uint16_t);
inline constexpr std::size_t kBarrierCount =
    3U * kFc1Stages + 2U * kFc2Stages;

inline constexpr std::size_t kStaticV4AStageElements =
    static_cast<std::size_t>(kTileM) * kStaticV4TileK;
inline constexpr std::size_t kStaticV4Fc1BStageElements =
    static_cast<std::size_t>(kFc1TileN) * kStaticV4TileK;
inline constexpr std::size_t kStaticV4Fc2BStageElements =
    static_cast<std::size_t>(kFc2TileN) * kStaticV4TileK;
inline constexpr std::size_t kStaticV4AStorageElements =
    max_size(kStaticV4Fc1Stages, kStaticV4Fc2Stages) *
    kStaticV4AStageElements;
// Packed [gate8, up8] W1 needs one B arena; the general path needs two.
// FC2 has an independent A/B offset. Its arena is accounted for below;
// its B does not coexist with FC1's maximum A extent.
inline constexpr std::size_t kStaticV4BStorageElements =
    kStaticV4Fc1Stages * kStaticV4Fc1BStageElements;
inline constexpr std::size_t kStaticV4GemmStorageBytes =
    max_size(kStaticV4AStorageElements + kStaticV4BStorageElements,
             kStaticV4Fc2Stages *
                     (kStaticV4AStageElements +
                      kStaticV4Fc2BStageElements) +
                 (kFc2ClaimTiles > 1U ? kTileM * kFc2TileN : 0U)) *
    sizeof(std::uint16_t);
inline constexpr std::size_t kStaticV4BarrierCount =
    2U * kStaticV4Fc1Stages + 2U * kStaticV4Fc2Stages;

inline constexpr std::size_t compute_smem_bytes(std::uint32_t world) {
  return kGemmStorageBytes + kBarrierCount * sizeof(std::uint64_t) +
         static_cast<std::size_t>(world) * sizeof(std::uint32_t);
}

inline constexpr std::size_t
static_v4_compute_smem_bytes(std::uint32_t world) {
  return kStaticV4GemmStorageBytes +
         kStaticV4BarrierCount * sizeof(std::uint64_t) +
         static_cast<std::size_t>(world) * sizeof(std::uint32_t);
}

// Optional prefill companion to the measured narrow-N decode profile. Both
// entries need 192 KiB of operands; the prefill entry has fewer barriers, so the
// same allocation is a valid upper bound. Descriptors share the weight data.
template <bool Prefill> struct StaticV4Gemm {
  static constexpr unsigned kFc1N = Prefill ? 128U : kFc1TileN;
  static constexpr unsigned kFc2N = Prefill ? 256U : kFc2TileN;
  static constexpr unsigned kFc1Stages = Prefill ? 3U : kStaticV4Fc1Stages;
  static constexpr unsigned kFc2Stages = Prefill ? 2U : kStaticV4Fc2Stages;
};
static_assert(!kStaticV4PrefillEnabled ||
                  (kTileM == 128U && kFc1TileN == 64U && kFc2TileN == 128U &&
                   kStaticV4TileK == 128U && kStaticV4Fc1Stages == 3U &&
                   kStaticV4Fc2Stages == 3U && kFc2ClaimTiles == 1U),
              "Wide prefill companion requires the N64/N128 K128 3/3-stage "
              "claim1 decode profile");

inline constexpr std::size_t transport_smem_bytes(std::uint32_t hidden,
                                                   std::uint32_t world) {
  const auto streams = max_size(
      max_size(kLocalDispatchTmaWarps, kRelayTmaWarps), kRelayScatterWarps);
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
static_assert(kTileK == 64U);
static_assert(kStaticV4TileK == 64U || kStaticV4TileK == 128U);
static_assert(kFc1Stages >= 2U && kFc1Stages <= 4U);
static_assert(kFc2Stages >= 2U && kFc2Stages <= 4U);
static_assert(kStaticV4Fc1Stages >= 2U && kStaticV4Fc1Stages <= 6U);
static_assert(kStaticV4Fc2Stages >= 2U && kStaticV4Fc2Stages <= 6U);
static_assert(kFc2ClaimTiles == 1U || kFc2ClaimTiles == 2U ||
              kFc2ClaimTiles == 4U || kFc2ClaimTiles == 8U);
static_assert(kGemmStorageBytes <= 227U * 1024U,
              "FC2 input/output staging exceeds Hopper shared memory");
static_assert(kStaticV4GemmStorageBytes <= 227U * 1024U,
              "V4 inference staging exceeds Hopper shared memory");
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
static_assert(kFc2TmaStore && kTileM == 128U,
              "Forward requires M128 with the TMA FC2 epilogue");
static_assert(kClusterSize == 2U,
              "Forward requires two-CTA TMA multicast clusters");
// GIN owns eight physical contexts.  A one-node specialization has no RDMA
// role at all and may tune Relay independently; multi-node keeps Relay in
// whole context-sized groups.
constexpr bool valid_transport_role_counts() {
  if constexpr (kRdmaCtas == 0U)
    return kRelayCtas > 0U;
  else
    return kRdmaCtas == 8U && kRelayCtas >= kRdmaCtas &&
           kRelayCtas % (kRdmaCtas == 0U ? 1U : kRdmaCtas) == 0U;
}
static_assert(valid_transport_role_counts());
static_assert(kRdmaCtas != 0U ||
              kLocalDispatchTmaWarps < kLocalTransportThreads / 32U,
              "local split Relay reserves one warp for FC1-ready publication");
static_assert(kRelayTmaWarps <= kComputeThreads / 32U);
static_assert(kLocalDispatchTmaWarps > 0U &&
              kLocalDispatchTmaWarps <= kComputeThreads / 32U);
static_assert(kRelayScatterWarps > 0U &&
              kRelayScatterWarps <= kComputeThreads / 32U);
static_assert(kLocalReturnBatchRows >= 32U &&
              kLocalReturnBatchRows <= 1024U &&
              kLocalReturnBatchRows % 32U == 0U);
static_assert(kLocalRouteBatchRows > 0U && kStartupMblocks > 0U);
static_assert(kRelayLocalReturn, "local Return requires Relay ownership");
static_assert(kTaskClaimChunk == 1U,
              "only the proven one-task queue claim is currently compiled");
static_assert(kCompletionPeerProbes > 0U &&
              kCompletionPeerProbes <= 4U);
static_assert(kReduceScheduler <= 2U);
static_assert(kReduceWarps > 0U &&
              kReduceWarps <= kComputeThreads / 32U);
static_assert(kReduceTokensPerClaim > 0U &&
              kReduceTokensPerClaim <= kReduceWarps);

} // namespace weftmoe::cuda::forward_tuning
