#pragma once

#include <cstddef>
#include <cstdint>

// Dynamic builds generate this header from an explicit JSON configuration.
// The checked-in defaults are one auditable Hopper specialization, not a set
// of runtime environment switches.
#if defined(WEFTMOE_BACKWARD_USE_GENERATED_TUNING)
#include <weftmoe_backward_tuning_generated.hpp>
#endif

#ifndef WEFTMOE_BWD_MAIN_HIDDEN
#define WEFTMOE_BWD_MAIN_HIDDEN 7168
#endif
#ifndef WEFTMOE_BWD_MAIN_INTERMEDIATE
#define WEFTMOE_BWD_MAIN_INTERMEDIATE 3072
#endif
#ifndef WEFTMOE_BWD_TILE_M
#define WEFTMOE_BWD_TILE_M 128
#endif
#ifndef WEFTMOE_BWD_DAB_TILE_N
#define WEFTMOE_BWD_DAB_TILE_N 256
#endif
#ifndef WEFTMOE_BWD_DX_TILE_N
#define WEFTMOE_BWD_DX_TILE_N 256
#endif
#ifndef WEFTMOE_BWD_DW_TILE_M
#define WEFTMOE_BWD_DW_TILE_M 128
#endif
#ifndef WEFTMOE_BWD_DW_TILE_N
#define WEFTMOE_BWD_DW_TILE_N 256
#endif
#ifndef WEFTMOE_BWD_TILE_K
#define WEFTMOE_BWD_TILE_K 64
#endif
#ifndef WEFTMOE_BWD_DAB_STAGES
#define WEFTMOE_BWD_DAB_STAGES 4
#endif
#ifndef WEFTMOE_BWD_DX_STAGES
#define WEFTMOE_BWD_DX_STAGES 4
#endif
#ifndef WEFTMOE_BWD_DW1_STAGES
#define WEFTMOE_BWD_DW1_STAGES 3
#endif
#ifndef WEFTMOE_BWD_DW2_STAGES
#define WEFTMOE_BWD_DW2_STAGES 3
#endif
#ifndef WEFTMOE_BWD_COMPUTE_THREADS
#define WEFTMOE_BWD_COMPUTE_THREADS 384
#endif
#ifndef WEFTMOE_BWD_MATH_WARPGROUPS
#define WEFTMOE_BWD_MATH_WARPGROUPS 2
#endif
#ifndef WEFTMOE_BWD_PRODUCER_WARPGROUPS
#define WEFTMOE_BWD_PRODUCER_WARPGROUPS 1
#endif
#ifndef WEFTMOE_BWD_MATH_REGISTERS
#define WEFTMOE_BWD_MATH_REGISTERS 224
#endif
#ifndef WEFTMOE_BWD_PRODUCER_REGISTERS
#define WEFTMOE_BWD_PRODUCER_REGISTERS 48
#endif
#ifndef WEFTMOE_BWD_RDMA_CTAS
#define WEFTMOE_BWD_RDMA_CTAS 8
#endif
#ifndef WEFTMOE_BWD_QUEUE_IDLE_NANOSLEEP
#define WEFTMOE_BWD_QUEUE_IDLE_NANOSLEEP 64
#endif
#ifndef WEFTMOE_BWD_READY_WORD_STAGGER
#define WEFTMOE_BWD_READY_WORD_STAGGER 0
#endif
#ifndef WEFTMOE_BWD_ACTIVATION_OWNER_GROUP
#define WEFTMOE_BWD_ACTIVATION_OWNER_GROUP 4
#endif
#ifndef WEFTMOE_BWD_DISPATCH_BATCH_ROWS
#if WEFTMOE_BWD_RDMA_CTAS == 0
#define WEFTMOE_BWD_DISPATCH_BATCH_ROWS 32
#else
#define WEFTMOE_BWD_DISPATCH_BATCH_ROWS 64
#endif
#endif
#ifndef WEFTMOE_BWD_EPILOGUE_ROWS_PER_TASK
#define WEFTMOE_BWD_EPILOGUE_ROWS_PER_TASK 12
#endif
#ifndef WEFTMOE_BWD_DW2_CLAIM_TILES
#define WEFTMOE_BWD_DW2_CLAIM_TILES 4
#endif
#ifndef WEFTMOE_BWD_DW1_CLAIM_TILES
#define WEFTMOE_BWD_DW1_CLAIM_TILES 4
#endif
#ifndef WEFTMOE_BWD_SETUP_THREADS
#define WEFTMOE_BWD_SETUP_THREADS 256
#endif
#ifndef WEFTMOE_BWD_SETUP_IDLE_NANOSLEEP
#define WEFTMOE_BWD_SETUP_IDLE_NANOSLEEP 128
#endif
#ifndef WEFTMOE_BWD_REDUCE_THREADS
#define WEFTMOE_BWD_REDUCE_THREADS 256
#endif
#ifndef WEFTMOE_BWD_REDUCE_IDLE_NANOSLEEP
#define WEFTMOE_BWD_REDUCE_IDLE_NANOSLEEP 256
#endif

namespace weftmoe::cuda::backward_tuning {
// Untuned single-node defaults favor enough transport workers for Ling-sized
// workloads. Explicit tuning overrides apply to either topology unchanged.
#ifndef WEFTMOE_BWD_RELAY_CTAS
#define WEFTMOE_BWD_RELAY_CTAS 32
inline constexpr std::uint32_t kRemoteRelayCtas = 16;
#else
inline constexpr std::uint32_t kRemoteRelayCtas = WEFTMOE_BWD_RELAY_CTAS;
#endif

inline constexpr std::uint32_t kMainHidden = WEFTMOE_BWD_MAIN_HIDDEN;
inline constexpr std::uint32_t kMainIntermediate =
    WEFTMOE_BWD_MAIN_INTERMEDIATE;
inline constexpr std::uint32_t kTileM = WEFTMOE_BWD_TILE_M;
inline constexpr std::uint32_t kDabTileN = WEFTMOE_BWD_DAB_TILE_N;
inline constexpr std::uint32_t kDxTileN = WEFTMOE_BWD_DX_TILE_N;
inline constexpr std::uint32_t kDwTileM = WEFTMOE_BWD_DW_TILE_M;
inline constexpr std::uint32_t kDwTileN = WEFTMOE_BWD_DW_TILE_N;
inline constexpr std::uint32_t kTileK = WEFTMOE_BWD_TILE_K;
inline constexpr std::uint32_t kDabStages = WEFTMOE_BWD_DAB_STAGES;
inline constexpr std::uint32_t kDxStages = WEFTMOE_BWD_DX_STAGES;
inline constexpr std::uint32_t kDw1Stages = WEFTMOE_BWD_DW1_STAGES;
inline constexpr std::uint32_t kDw2Stages = WEFTMOE_BWD_DW2_STAGES;
inline constexpr std::uint32_t kComputeThreads =
    WEFTMOE_BWD_COMPUTE_THREADS;
inline constexpr std::uint32_t kMathWarpgroups =
    WEFTMOE_BWD_MATH_WARPGROUPS;
inline constexpr std::uint32_t kProducerWarpgroups =
    WEFTMOE_BWD_PRODUCER_WARPGROUPS;
inline constexpr std::uint32_t kMathRegisters =
    WEFTMOE_BWD_MATH_REGISTERS;
inline constexpr std::uint32_t kProducerRegisters =
    WEFTMOE_BWD_PRODUCER_REGISTERS;
inline constexpr std::uint32_t kRdmaCtas = WEFTMOE_BWD_RDMA_CTAS;
inline constexpr std::uint32_t kRelayCtas = WEFTMOE_BWD_RELAY_CTAS;
inline constexpr std::uint32_t kQueueIdleNanosleep =
    WEFTMOE_BWD_QUEUE_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReadyWordStagger =
    WEFTMOE_BWD_READY_WORD_STAGGER;
inline constexpr std::uint32_t kActivationOwnerGroup =
    WEFTMOE_BWD_ACTIVATION_OWNER_GROUP;
inline constexpr std::uint32_t kDispatchBatchRows =
    WEFTMOE_BWD_DISPATCH_BATCH_ROWS;
inline constexpr std::uint32_t kEpilogueRowsPerTask =
    WEFTMOE_BWD_EPILOGUE_ROWS_PER_TASK;
inline constexpr std::uint32_t kDw2ClaimTiles =
    WEFTMOE_BWD_DW2_CLAIM_TILES;
inline constexpr std::uint32_t kDw1ClaimTiles =
    WEFTMOE_BWD_DW1_CLAIM_TILES;
inline constexpr std::uint32_t kSetupThreads = WEFTMOE_BWD_SETUP_THREADS;
inline constexpr std::uint32_t kSetupIdleNanosleep =
    WEFTMOE_BWD_SETUP_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kReduceThreads = WEFTMOE_BWD_REDUCE_THREADS;
inline constexpr std::uint32_t kReduceIdleNanosleep =
    WEFTMOE_BWD_REDUCE_IDLE_NANOSLEEP;
inline constexpr std::uint32_t kDw2TilesPerExpert =
    (kMainHidden / kDwTileM) * (kMainIntermediate / kDwTileN);
inline constexpr std::uint32_t kDw1TilesPerExpert =
    ((2U * kMainIntermediate) / kDwTileM) *
    (kMainHidden / kDwTileN);

static_assert(kMainHidden > 0U && kMainHidden % 256U == 0U);
static_assert(kMainIntermediate > 0U && kMainIntermediate % 256U == 0U);
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
static_assert(kRdmaCtas == 0U || kRdmaCtas == 8U);
static_assert(kRelayCtas >= 8U && kRelayCtas % 4U == 0U);
static_assert(kReadyWordStagger <= 1U);
static_assert(kActivationOwnerGroup >= 1U &&
              kActivationOwnerGroup <= 32U &&
              (kActivationOwnerGroup & (kActivationOwnerGroup - 1U)) == 0U);
static_assert(kDispatchBatchRows >= kComputeThreads / 32U &&
              kDispatchBatchRows <= 128U);
static_assert(kEpilogueRowsPerTask >= kComputeThreads / 32U &&
              kEpilogueRowsPerTask <= kTileM);
static_assert(kDw2ClaimTiles == 1U || kDw2ClaimTiles == 4U ||
              kDw2ClaimTiles == 8U || kDw2ClaimTiles == 16U);
static_assert(kDw1ClaimTiles == 1U || kDw1ClaimTiles == 4U ||
              kDw1ClaimTiles == 8U || kDw1ClaimTiles == 16U);
static_assert(kSetupThreads >= 32U && kSetupThreads <= 1024U &&
              kSetupThreads % 32U == 0U);
static_assert(kReduceThreads >= 32U && kReduceThreads <= 1024U &&
              kReduceThreads % 32U == 0U);

} // namespace weftmoe::cuda::backward_tuning
