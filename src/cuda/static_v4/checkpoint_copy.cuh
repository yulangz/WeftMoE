#pragma once

// Both local and network Pull use one G2S read. Training adds a second S2G
// checkpoint; drain covers both writes before publishing any slot credit.
struct RingCheckpointDestination {
  const ForwardKernelArgs &a;
  __device__ __forceinline__ void *operator()(std::uint64_t index,
                                              unsigned checkpoint) const {
    if (checkpoint)
      return a.saved_x + a.checkpoint.rows[index] * a.hidden;
    const auto row = static_v4::ring::row(a, index / forward_tuning::kTileM) +
                     index % forward_tuning::kTileM;
    return peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset) +
           row * a.hidden;
  }
};

template <class Storage, bool Training> class RingPullCopy {
  using Pipeline = std::conditional_t<
      Training, FanoutRowPrefetchPipeline<Storage, std::uint64_t,
                                          RingCheckpointDestination>,
      RowPrefetchPipeline<Storage>>;
  Pipeline pipeline_;
  __device__ static Pipeline make(const ForwardKernelArgs &a, Storage &s) {
    if constexpr (Training)
      return Pipeline(s, 0U, RingCheckpointDestination{a});
    else
      return Pipeline(s, 0U);
  }

public:
  __device__ RingPullCopy(const ForwardKernelArgs &a, Storage &s)
      : pipeline_(make(a, s)) {}
  __device__ __forceinline__ void copy(const void *source, void *target,
                                       unsigned bytes, std::uint64_t index) {
    if constexpr (Training)
      pipeline_.copy(source, index, bytes, 2U);
    else
      pipeline_.copy(source, target, bytes);
  }
  __device__ __forceinline__ void drain() { pipeline_.drain(); }
};
