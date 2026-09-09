#pragma once

#include <cstdint>

#include "cuda/backward/tuning_config.hpp"
#include "cuda/persistent_v3/queues.cuh"

namespace weftmoe::cuda::persistent_v3::backward {

enum class TaskKind : std::uint8_t {
  Dx = 0U,
  Dab = 1U,
  Dw2 = 2U,
  Dw1 = 3U,
  None = 4U,
};

inline constexpr std::uint32_t kTaskKindCount = 4U;

// A descriptor table is immutable for one retained Forward handle. Runtime
// publication changes only an owner-ready summary and fixed per-owner claim
// bits; there is no wrapping task ring and no enqueue payload traffic.
struct TileTask {
  std::uint32_t owner = 0U;
  std::uint32_t tile = 0U;
  TaskKind kind = TaskKind::None;
  std::uint32_t tile_count = 1U;
};

struct TileDomain {
  unsigned long long *ready_words = nullptr;
  unsigned long long *claim_words = nullptr;
  std::uint32_t owner_count = 0U;
  std::uint32_t tiles_per_owner = 0U;
  TaskKind kind = TaskKind::None;
  bool linear_claim = false;
  std::uint32_t claim_granularity = 1U;

  WEFTMOE_V3_HOST_DEVICE constexpr std::uint32_t word_count() const {
    return (owner_count + 63U) / 64U;
  }

  WEFTMOE_V3_HOST_DEVICE constexpr std::uint32_t
  claim_words_per_owner() const {
    if (linear_claim)
      return 1U;
    return (tiles_per_owner + 63U) / 64U;
  }

  WEFTMOE_V3_HOST_DEVICE constexpr unsigned long long
  valid_claim_mask(std::uint32_t word) const {
    const auto begin = word * 64U;
    const auto remaining = tiles_per_owner - begin;
    return remaining >= 64U ? ~0ULL : ((1ULL << remaining) - 1ULL);
  }
};

struct WorkQueues {
  TileDomain dx;
  TileDomain dab;
  TileDomain dw2;
  TileDomain dw1;
};

#if defined(__CUDACC__)
WEFTMOE_V3_DEVICE inline void publish_owner(TileDomain domain,
                                             std::uint32_t owner) {
  if (owner >= domain.owner_count || domain.tiles_per_owner == 0U)
    return;
  const auto claim_word_begin = owner * domain.claim_words_per_owner();
  for (std::uint32_t word = 0U; word < domain.claim_words_per_owner();
       ++word) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[claim_word_begin + word]);
    claim.store(0ULL, ::cuda::memory_order_relaxed);
  }
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
      domain.ready_words[owner / 64U]);
  ready.fetch_or(1ULL << (owner % 64U), ::cuda::memory_order_release);
}

// Publish only the leading tile prefix of an otherwise fixed-size domain.
// Bits outside the prefix start in the claimed state, so the ordinary claim
// path removes the owner-ready bit as soon as the exposed phase is drained.
// A later phase can reset and republish the same immutable owner descriptor.
WEFTMOE_V3_DEVICE inline void publish_owner_prefix(
    TileDomain domain, std::uint32_t owner, std::uint32_t prefix_tiles) {
  if (owner >= domain.owner_count || prefix_tiles == 0U ||
      prefix_tiles > domain.tiles_per_owner)
    return;
  const auto claim_word_begin = owner * domain.claim_words_per_owner();
  for (std::uint32_t word = 0U; word < domain.claim_words_per_owner();
       ++word) {
    const auto begin = word * 64U;
    const auto remaining = prefix_tiles <= begin ? 0U : prefix_tiles - begin;
    const auto available = remaining < 64U ? remaining : 64U;
    const auto available_mask =
        available == 64U ? ~0ULL : ((1ULL << available) - 1ULL);
    const auto initially_claimed =
        domain.valid_claim_mask(word) & ~available_mask;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[claim_word_begin + word]);
    // A delayed retirement may observe this reset before the ready summary
    // is republished. Its acquire probe must also observe the new phase.
    claim.store(initially_claimed, ::cuda::memory_order_release);
  }
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
      domain.ready_words[owner / 64U]);
  ready.fetch_or(1ULL << (owner % 64U), ::cuda::memory_order_release);
}

WEFTMOE_V3_DEVICE inline bool owner_fully_claimed(TileDomain domain,
                                                   std::uint32_t owner) {
  if (domain.linear_claim) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[owner]);
    return claim.load(::cuda::memory_order_relaxed) >=
           domain.tiles_per_owner;
  }
  const auto words = domain.claim_words_per_owner();
  const auto begin = owner * words;
  for (std::uint32_t word = 0U; word < words; ++word) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[begin + word]);
    if ((claim.load(::cuda::memory_order_acquire) &
         domain.valid_claim_mask(word)) != domain.valid_claim_mask(word))
      return false;
  }
  return true;
}

// The dU and SwiGLU epilogue phases reuse the DAB owner. A worker can observe
// the old phase fully claimed, then retire its ready bit after another worker
// has already published the next phase. Repair that stale retirement when the
// new phase still has unclaimed work. Single-publication domains need no probe.
WEFTMOE_V3_DEVICE inline void retire_claimed_owner(TileDomain domain,
                                                   std::uint32_t owner) {
  ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
      domain.ready_words[owner / 64U]);
  const auto mask = 1ULL << (owner % 64U);
  if (domain.kind == TaskKind::Dab) {
    ready.fetch_and(~mask, ::cuda::memory_order_acq_rel);
    if (!owner_fully_claimed(domain, owner))
      ready.fetch_or(mask, ::cuda::memory_order_release);
  } else {
    ready.fetch_and(~mask, ::cuda::memory_order_relaxed);
  }
}

WEFTMOE_V3_DEVICE inline bool try_claim_owner_tile(TileDomain domain,
                                                    std::uint32_t owner,
                                                    TileTask &task) {
  if (domain.linear_claim) {
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[owner]);
    const auto tile = claim.fetch_add(domain.claim_granularity,
                                      ::cuda::memory_order_relaxed);
    if (tile >= domain.tiles_per_owner)
      return false;
    const auto remaining = domain.tiles_per_owner -
                           static_cast<std::uint32_t>(tile);
    task = TileTask{owner, static_cast<std::uint32_t>(tile), domain.kind,
                    remaining < domain.claim_granularity
                        ? remaining
                        : domain.claim_granularity};
    return true;
  }
  const auto words = domain.claim_words_per_owner();
  const auto begin = owner * words;
  for (std::uint32_t word = 0U; word < words; ++word) {
    const auto valid = domain.valid_claim_mask(word);
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> claim(
        domain.claim_words[begin + word]);
    auto observed = claim.load(::cuda::memory_order_relaxed);
    while ((observed & valid) != valid) {
      const auto available = valid & ~observed;
      const auto bit = static_cast<std::uint32_t>(__ffsll(available) - 1);
      const auto desired = observed | (1ULL << bit);
      if (claim.compare_exchange_weak(observed, desired,
                                      ::cuda::memory_order_relaxed,
                                      ::cuda::memory_order_relaxed)) {
        task = TileTask{owner, word * 64U + bit, domain.kind, 1U};
        return true;
      }
    }
  }
  return false;
}

WEFTMOE_V3_DEVICE inline bool try_claim_domain_once(TileDomain domain,
                                                     TileTask &task) {
  const auto words = domain.word_count();
  if (words == 0U || domain.tiles_per_owner == 0U)
    return false;
  // M-block owners are contiguous by expert. Starting from the same ready
  // word lets CTAs consume neighboring M-blocks while their W1/W2 tiles are
  // still resident in L2; the complete bitmap is still visited on a miss.
  std::uint32_t first = 0U;
  if constexpr (backward_tuning::kReadyWordStagger != 0U)
    first = static_cast<std::uint32_t>(blockIdx.x) % words;
  for (std::uint32_t offset = 0U; offset < words; ++offset) {
    const auto word_index = (first + offset) % words;
    ::cuda::atomic_ref<unsigned long long, ::cuda::thread_scope_device> ready(
        domain.ready_words[word_index]);
    auto bits = ready.load(::cuda::memory_order_acquire);
    while (bits != 0ULL) {
      // Spread CTAs over the first compact owner group without giving up the
      // expert-major ready-word order.  Since each owner claims N tiles from
      // zero upward, neighboring M-blocks then request the same W1/W2 tile
      // close together and can reuse it from L2.
      auto bit = static_cast<std::uint32_t>(__ffsll(bits) - 1);
      if constexpr (backward_tuning::kActivationOwnerGroup > 1U) {
        if (domain.kind == TaskKind::Dx || domain.kind == TaskKind::Dab) {
          constexpr std::uint32_t kOwnerGroup =
              backward_tuning::kActivationOwnerGroup;
          const auto group_begin = bit & ~(kOwnerGroup - 1U);
          const auto group_bits =
              (bits >> group_begin) & ((1ULL << kOwnerGroup) - 1ULL);
          const auto rotation =
              static_cast<std::uint32_t>(blockIdx.x) & (kOwnerGroup - 1U);
          const auto after_rotation = group_bits & (~0ULL << rotation);
          const auto selected = after_rotation != 0ULL ? after_rotation
                                                        : group_bits;
          bit = group_begin +
                static_cast<std::uint32_t>(__ffsll(selected) - 1);
        }
      }
      const auto owner = word_index * 64U + bit;
      const auto mask = 1ULL << bit;
      if (owner >= domain.owner_count) {
        ready.fetch_and(~mask, ::cuda::memory_order_relaxed);
        bits &= ~mask;
        continue;
      }
      if (try_claim_owner_tile(domain, owner, task)) {
        if (owner_fully_claimed(domain, owner))
          retire_claimed_owner(domain, owner);
        return true;
      }
      if (owner_fully_claimed(domain, owner))
        retire_claimed_owner(domain, owner);
      bits &= ~mask;
    }
  }
  return false;
}

// One scheduling boundary calls each logical queue at most once. Keep this
// straight-line order explicit: there is deliberately no cohort, burst/age
// state, or second Q_DX check before dW.
WEFTMOE_V3_DEVICE inline bool try_claim_work(WorkQueues queues,
                                             TileTask &task) {
  if (try_claim_domain_once(queues.dx, task))
    return true;
  if (try_claim_domain_once(queues.dab, task))
    return true;
  if (try_claim_domain_once(queues.dw2, task))
    return true;
  return try_claim_domain_once(queues.dw1, task);
}
#endif

} // namespace weftmoe::cuda::persistent_v3::backward
