#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuda/forward/tuning_config.hpp"

#if defined(__CUDACC__)
#define MEGAMOE_V3_HOST_DEVICE __host__ __device__
#define MEGAMOE_V3_DEVICE __device__
#else
#define MEGAMOE_V3_HOST_DEVICE
#define MEGAMOE_V3_DEVICE
#endif

namespace megamoe::cuda::persistent_v3 {

// V3 has one scheduling quantum everywhere: one 128-row canonical M block.
// The final block of every expert is physically padded, not merely predicated,
// so dW can traverse a uniform, deterministic K extent in 64-row WGMMA steps.
inline constexpr std::uint32_t kMBlockRows = 128U;
inline constexpr std::uint32_t kRdmaCtaCount = forward_tuning::kRdmaCtas;
// Backward and the generic persistent-v3 entry retain their fixed eight-CTA
// relay contract. Forward carries its independently tuned relay count in
// ForwardKernelArgs and must not silently alter Backward's role layout.
inline constexpr std::uint32_t kDefaultRelayCtaCount = 8U;
inline constexpr std::uint32_t kThreadsPerCta =
    forward_tuning::kComputeThreads;
inline constexpr std::uint32_t kWarpsPerCta = kThreadsPerCta / 32U;
inline constexpr std::uint32_t kInvalidTask = ~0U;

template <class T>
MEGAMOE_V3_HOST_DEVICE constexpr T align_up(T value, T alignment) {
  static_assert(std::is_unsigned_v<T>);
  return value == 0 ? 0 : ((value - 1) / alignment + 1) * alignment;
}

MEGAMOE_V3_HOST_DEVICE constexpr std::uint64_t
padded_expert_rows(std::uint64_t real_rows) {
  return align_up(real_rows, static_cast<std::uint64_t>(kMBlockRows));
}

MEGAMOE_V3_HOST_DEVICE constexpr std::uint64_t
expert_mblocks(std::uint64_t real_rows) {
  return padded_expert_rows(real_rows) / kMBlockRows;
}

struct ExpertRowSpan {
  std::uint64_t canonical_begin = 0;
  std::uint64_t padded_begin = 0;
  std::uint64_t real_rows = 0;
  std::uint64_t padded_rows = 0;

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint64_t mblocks() const {
    return padded_rows / kMBlockRows;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr bool valid() const {
    return padded_rows == padded_expert_rows(real_rows) &&
           padded_begin % kMBlockRows == 0;
  }
};

enum class WorkerRole : std::uint8_t { Rdma, Relay, Compute, Invalid };

// RDMA CTA i owns GIN context i for the entire kernel lifetime. Relay CTA i
// is paired with the same context. All remaining resident CTAs are compute
// only; none may issue GIN operations or perform NVLink fan-out.
struct RoleLayout {
  std::uint32_t total_ctas = 0;
  std::uint32_t relay_ctas = kDefaultRelayCtaCount;

  MEGAMOE_V3_HOST_DEVICE constexpr bool valid() const {
    return relay_ctas >= kRdmaCtaCount &&
           total_ctas > kRdmaCtaCount + relay_ctas;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint32_t relay_begin() const {
    return kRdmaCtaCount;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint32_t compute_begin() const {
    return kRdmaCtaCount + relay_ctas;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint32_t compute_ctas() const {
    return valid() ? total_ctas - compute_begin() : 0U;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr WorkerRole
  role(std::uint32_t cta) const {
    if (!valid() || cta >= total_ctas)
      return WorkerRole::Invalid;
    if (cta < kRdmaCtaCount)
      return WorkerRole::Rdma;
    if (cta < compute_begin())
      return WorkerRole::Relay;
    return WorkerRole::Compute;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint32_t
  role_ordinal(std::uint32_t cta) const {
    const auto worker = role(cta);
    if (worker == WorkerRole::Rdma)
      return cta;
    if (worker == WorkerRole::Relay)
      return cta - relay_begin();
    if (worker == WorkerRole::Compute)
      return cta - compute_begin();
    return kInvalidTask;
  }

  MEGAMOE_V3_HOST_DEVICE constexpr std::uint32_t
  context_for_relay(std::uint32_t relay_ordinal) const {
    return relay_ordinal % kRdmaCtaCount;
  }
};

enum class TaskKind : std::uint8_t {
  DispatchRelay,
  Fc1,
  Fc2,
  ReturnRelay,
  DyRelay,
  Dv,
  Dx,
  Dw1,
  Dw2,
  Reduce,
  Count
};

inline constexpr std::uint32_t kTaskKindCount =
    static_cast<std::uint32_t>(TaskKind::Count);

// Fixed-size handoff descriptor. Payload addresses stay in immutable runtime
// tables; queues carry indices, epochs and extents only.
struct alignas(16) PipelineTask {
  std::uint64_t epoch = 0;
  std::uint32_t index = 0;
  std::uint16_t expert = 0;
  std::uint8_t kind = 0;
  std::uint8_t context = 0;
  std::uint32_t row_begin = 0;
  std::uint16_t row_count = 0;
  std::uint16_t flags = 0;
};

static_assert(sizeof(PipelineTask) == 32U);
static_assert(alignof(PipelineTask) == 16U);
static_assert(padded_expert_rows(0) == 0);
static_assert(padded_expert_rows(1) == 128);
static_assert(padded_expert_rows(127) == 128);
static_assert(padded_expert_rows(128) == 128);
static_assert(padded_expert_rows(129) == 256);
static_assert(RoleLayout{132, 8}.role(0) == WorkerRole::Rdma);
static_assert(RoleLayout{132, 8}.role(7) == WorkerRole::Rdma);
static_assert(RoleLayout{132, 8}.role(8) == WorkerRole::Relay);
static_assert(RoleLayout{132, 8}.role(15) == WorkerRole::Relay);
static_assert(RoleLayout{132, 8}.role(16) == WorkerRole::Compute);
static_assert(RoleLayout{132, 8}.compute_ctas() == 116);

} // namespace megamoe::cuda::persistent_v3
