#pragma once

namespace backward_pipeline_detail {

using weftmoe::cuda::pipeline::PaddedCounter;
using weftmoe::cuda::pipeline::backward::TaskKind;
using weftmoe::cuda::pipeline::backward::TileTask;

constexpr std::uint32_t kTerminalTask = ~1U;
constexpr std::uint32_t kDw2MTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileM;
constexpr std::uint32_t kDw2NTiles =
    backward_tuning::kMainIntermediate / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw1MTiles =
    (2U * backward_tuning::kMainIntermediate) / backward_tuning::kDwTileM;
constexpr std::uint32_t kDw1NTiles =
    backward_tuning::kMainHidden / backward_tuning::kDwTileN;
constexpr std::uint32_t kDw2TilesPerExpert =
    backward_tuning::kDw2TilesPerExpert;
constexpr std::uint32_t kDw1TilesPerExpert =
    backward_tuning::kDw1TilesPerExpert;

static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileM == 0U);
static_assert(backward_tuning::kMainIntermediate % backward_tuning::kDwTileN ==
              0U);
static_assert((2U * backward_tuning::kMainIntermediate) %
                  backward_tuning::kDwTileM ==
              0U);
static_assert(backward_tuning::kMainHidden % backward_tuning::kDwTileN == 0U);

__device__ __forceinline__ unsigned long long *
scheduler_words(const BackwardKernelArgs &a) {
  return peer_at<unsigned long long>(a.local_base, a.scheduler_offset);
}

__device__ __forceinline__ void initialize_status(const BackwardKernelArgs &a) {
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  status->start_clock = ~0ULL;
  status->dispatch_complete_clock = 0ULL;
  status->fc1_start_clock = ~0ULL;
  status->fc1_complete_clock = 0ULL;
  status->relay_blocks_done = ~0ULL; // First dX tile start.
  status->fc2_complete_clock = 0ULL;
  status->reduce_blocks_done = ~0ULL; // First returned M-block.
  status->return_complete_clock = 0ULL;
  status->backward_dw2_start_clock = ~0ULL;
  status->backward_dw2_complete_clock = 0ULL;
  status->backward_dw1_start_clock = ~0ULL;
  status->backward_dw1_complete_clock = 0ULL;
  status->backward_reduce_start_clock = ~0ULL;
  status->backward_reduce_complete_clock = 0ULL;
  status->backward_sender_complete_clock = 0ULL;
  status->backward_transport_complete_clock = 0ULL;
  status->backward_return_send_complete_clock = 0ULL;
  status->backward_return_receive_complete_clock = 0ULL;
  status->gin_return_put_count = 0ULL;
  status->gin_return_payload_bytes = 0ULL;
}

} // namespace backward_pipeline_detail
