#pragma once

#include <cstddef>
#include <cstdint>

// Dynamic builds generate this header from an explicit JSON configuration.
// The checked-in defaults are one auditable Hopper specialization, not a set
// of runtime environment switches.
#if defined(MEGAMOE_BACKWARD_USE_GENERATED_TUNING)
#include <megamoe_backward_tuning_generated.hpp>
#endif

#ifndef MEGAMOE_BWD_MAIN_HIDDEN
#define MEGAMOE_BWD_MAIN_HIDDEN 7168
#endif
#ifndef MEGAMOE_BWD_MAIN_INTERMEDIATE
#define MEGAMOE_BWD_MAIN_INTERMEDIATE 3072
#endif
#ifndef MEGAMOE_BWD_TILE_M
#define MEGAMOE_BWD_TILE_M 128
#endif
#ifndef MEGAMOE_BWD_DAB_TILE_N
#define MEGAMOE_BWD_DAB_TILE_N 256
#endif
#ifndef MEGAMOE_BWD_DX_TILE_N
#define MEGAMOE_BWD_DX_TILE_N 256
#endif
#ifndef MEGAMOE_BWD_DW_TILE_M
#define MEGAMOE_BWD_DW_TILE_M 128
#endif
#ifndef MEGAMOE_BWD_DW_TILE_N
#define MEGAMOE_BWD_DW_TILE_N 256
#endif
#ifndef MEGAMOE_BWD_TILE_K
#define MEGAMOE_BWD_TILE_K 64
#endif
#ifndef MEGAMOE_BWD_DAB_STAGES
#define MEGAMOE_BWD_DAB_STAGES 4
#endif
#ifndef MEGAMOE_BWD_DX_STAGES
#define MEGAMOE_BWD_DX_STAGES 4
#endif
#ifndef MEGAMOE_BWD_DW1_STAGES
#define MEGAMOE_BWD_DW1_STAGES 3
#endif
#ifndef MEGAMOE_BWD_DW2_STAGES
#define MEGAMOE_BWD_DW2_STAGES 3
#endif
#ifndef MEGAMOE_BWD_COMPUTE_THREADS
#define MEGAMOE_BWD_COMPUTE_THREADS 384
#endif
#ifndef MEGAMOE_BWD_MATH_WARPGROUPS
#define MEGAMOE_BWD_MATH_WARPGROUPS 2
#endif
#ifndef MEGAMOE_BWD_PRODUCER_WARPGROUPS
#define MEGAMOE_BWD_PRODUCER_WARPGROUPS 1
#endif
#ifndef MEGAMOE_BWD_MATH_REGISTERS
#define MEGAMOE_BWD_MATH_REGISTERS 224
#endif
#ifndef MEGAMOE_BWD_PRODUCER_REGISTERS
#define MEGAMOE_BWD_PRODUCER_REGISTERS 48
#endif
#ifndef MEGAMOE_BWD_RDMA_CTAS
#define MEGAMOE_BWD_RDMA_CTAS 8
#endif
#ifndef MEGAMOE_BWD_RELAY_CTAS
#define MEGAMOE_BWD_RELAY_CTAS 12
#endif
#ifndef MEGAMOE_BWD_QUEUE_IDLE_NANOSLEEP
#define MEGAMOE_BWD_QUEUE_IDLE_NANOSLEEP 64
#endif
#ifndef MEGAMOE_BWD_DISPATCH_BATCH_ROWS
#define MEGAMOE_BWD_DISPATCH_BATCH_ROWS 32
#endif
#ifndef MEGAMOE_BWD_EPILOGUE_ROWS_PER_TASK
#define MEGAMOE_BWD_EPILOGUE_ROWS_PER_TASK 12
#endif
#ifndef MEGAMOE_BWD_DW_CLAIM_TILES
#define MEGAMOE_BWD_DW_CLAIM_TILES 8
#endif
#ifndef MEGAMOE_BWD_GIN_MAX_INFLIGHT
#define MEGAMOE_BWD_GIN_MAX_INFLIGHT 24
#endif
#ifndef MEGAMOE_BWD_SETUP_THREADS
#define MEGAMOE_BWD_SETUP_THREADS 256
#endif
#ifndef MEGAMOE_BWD_SETUP_IDLE_NANOSLEEP
#define MEGAMOE_BWD_SETUP_IDLE_NANOSLEEP 128
#endif
#ifndef MEGAMOE_BWD_CACHE_THREADS
#define MEGAMOE_BWD_CACHE_THREADS 256
#endif
#ifndef MEGAMOE_BWD_CACHE_PACK_CHUNKS
#define MEGAMOE_BWD_CACHE_PACK_CHUNKS 4
#endif
#ifndef MEGAMOE_BWD_REDUCE_THREADS
#define MEGAMOE_BWD_REDUCE_THREADS 256
#endif
#ifndef MEGAMOE_BWD_REDUCE_IDLE_NANOSLEEP
#define MEGAMOE_BWD_REDUCE_IDLE_NANOSLEEP 256
#endif

namespace megamoe::cuda::backward_tuning {

inline constexpr std::uint32_t kMainHidden = MEGAMOE_BWD_MAIN_HIDDEN;
inline constexpr std::uint32_t kMainIntermediate =
    MEGAMOE_BWD_MAIN_INTERMEDIATE;
inline constexpr std::uint32_t kTileM = MEGAMOE_BWD_TILE_M;
inline constexpr std::uint32_t kDabTileN = MEGAMOE_BWD_DAB_TILE_N;
inline constexpr std::uint32_t kDxTileN = MEGAMOE_BWD_DX_TILE_N;
inline constexpr std::uint32_t kDwTileM = MEGAMOE_BWD_DW_TILE_M;
inline constexpr std::uint32_t kDwTileN = MEGAMOE_BWD_DW_TILE_N;
inline constexpr std::uint32_t kTileK = MEGAMOE_BWD_TILE_K;
inline constexpr std::uint32_t kDabStages = MEGAMOE_BWD_DAB_STAGES;
inline constexpr std::uint32_t kDxStages = MEGAMOE_BWD_DX_STAGES;
inline constexpr std::uint32_t kDw1Stages = MEGAMOE_BWD_DW1_STAGES;
inline constexpr std::uint32_t kDw2Stages = MEGAMOE_BWD_DW2_STAGES;
inline constexpr std::uint32_t kComputeThreads =
    MEGAMOE_BWD_COMPUTE_THREADS;
inline constexpr std::uint32_t kMathWarpgroups =
    MEGAMOE_BWD_MATH_WARPGROUPS;
inline constexpr std::uint32_t kProducerWarpgroups =
    MEGAMOE_BWD_PRODUCER_WARPGROUPS;
inline constexpr std::uint32_t kMathRegisters =
    MEGAMOE_BWD_MATH_REGISTERS;
inline constexpr std::uint32_t kProducerRegisters =
    MEGAMOE_BWD_PRODUCER_REGISTERS;
inline constexpr std::uint32_t kRdmaCtas = MEGAMOE_BWD_RDMA_CTAS;
inline constexpr std::uint32_t kRelayCtas = MEGAMOE_BWD_RELAY_CTAS;
inline constexpr std::uint32_t kQueueIdleNanosleep =
    MEGAMOE_BWD_QUEUE_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kDispatchBatchRows =
    MEGAMOE_BWD_DISPATCH_BATCH_ROWS;
inline constexpr std::uint32_t kEpilogueRowsPerTask =
    MEGAMOE_BWD_EPILOGUE_ROWS_PER_TASK;
inline constexpr std::uint32_t kDwClaimTiles = MEGAMOE_BWD_DW_CLAIM_TILES;
inline constexpr std::uint32_t kGinMaxInflight =
    MEGAMOE_BWD_GIN_MAX_INFLIGHT;
inline constexpr std::uint32_t kSetupThreads = MEGAMOE_BWD_SETUP_THREADS;
inline constexpr std::uint32_t kSetupIdleNanosleep =
    MEGAMOE_BWD_SETUP_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kCacheThreads = MEGAMOE_BWD_CACHE_THREADS;
inline constexpr std::uint32_t kCachePackChunks =
    MEGAMOE_BWD_CACHE_PACK_CHUNKS;
inline constexpr std::uint32_t kReduceThreads = MEGAMOE_BWD_REDUCE_THREADS;
inline constexpr std::uint32_t kReduceIdleNanosleep =
    MEGAMOE_BWD_REDUCE_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kDw2TilesPerExpert =
    (kMainHidden / kDwTileM) * (kMainIntermediate / kDwTileN);
inline constexpr std::uint32_t kDw1TilesPerExpert =
    ((2U * kMainIntermediate) / kDwTileM) *
    (kMainHidden / kDwTileN);

static_assert(kMainHidden == 7168U && kMainIntermediate == 3072U);
static_assert(kTileM == 128U);
static_assert(kDabTileN == 128U || kDabTileN == 256U);
static_assert(kDxTileN == 128U || kDxTileN == 256U);
static_assert(kDwTileM == 128U || kDwTileM == 256U);
static_assert(kDwTileN == 128U || kDwTileN == 256U);
static_assert(kMainHidden % kDwTileM == 0U);
static_assert(kMainIntermediate % kDwTileN == 0U);
static_assert((2U * kMainIntermediate) % kDwTileM == 0U);
static_assert(kMainHidden % kDwTileN == 0U);
static_assert(kTileK == 64U);
static_assert(kDabStages >= 2U && kDabStages <= 4U);
static_assert(kDxStages >= 2U && kDxStages <= 4U);
static_assert(kDw1Stages >= 2U && kDw1Stages <= 4U);
static_assert(kDw2Stages >= 2U && kDw2Stages <= 4U);
static_assert(kComputeThreads == 384U);
static_assert(kMathWarpgroups == 2U && kProducerWarpgroups == 1U);
static_assert((kMathWarpgroups * kMathRegisters +
               kProducerWarpgroups * kProducerRegisters) *
                  128U <=
              65536U);
static_assert(kRdmaCtas == 8U);
static_assert(kRelayCtas >= 8U && kRelayCtas % 4U == 0U);
static_assert(kDispatchBatchRows >= kComputeThreads / 32U &&
              kDispatchBatchRows <= 128U);
static_assert(kEpilogueRowsPerTask >= kComputeThreads / 32U &&
              kEpilogueRowsPerTask <= kTileM);
static_assert(kDwClaimTiles == 1U || kDwClaimTiles == 4U ||
              kDwClaimTiles == 8U || kDwClaimTiles == 16U);
static_assert(kGinMaxInflight >= 1U && kGinMaxInflight <= 64U);
static_assert(kSetupThreads >= 32U && kSetupThreads <= 1024U &&
              kSetupThreads % 32U == 0U);
static_assert(kCacheThreads >= 32U && kCacheThreads <= 1024U &&
              kCacheThreads % 32U == 0U);
static_assert(kCachePackChunks >= 1U && kCachePackChunks <= 16U);
static_assert(kReduceThreads >= 32U && kReduceThreads <= 1024U &&
              kReduceThreads % 32U == 0U);

} // namespace megamoe::cuda::backward_tuning
