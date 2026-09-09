#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "cuda/transport_generation_completion.cuh"

namespace weftmoe::cuda::testing {

// CPU model of the production counter protocol.  It deliberately models no
// work queue: a randomized driver chooses among ready WAITING tasks, and the
// terminal condition depends only on immutable expected counts plus restored
// credits.  This keeps scheduler interleaving tests independent of CUDA/NCCL.
class GenerationSchedulerModel {
 public:
  enum class State : std::uint8_t { Waiting, Claimed, Completed };

  explicit GenerationSchedulerModel(
      const std::array<std::uint32_t,
                       transport_state::kTaskClassCount> &expected,
      std::uint32_t contexts, std::uint32_t credits_per_context)
      : credits_per_context_(credits_per_context), contexts_(contexts) {
    for (std::size_t kind = 0; kind < tasks_.size(); ++kind) {
      tasks_[kind].assign(expected[kind], State::Waiting);
      counters_[kind].expected = expected[kind];
    }
    context_credit_.assign(contexts, credits_per_context);
    context_inflight_.assign(contexts, 0U);
    context_directory_ready_.assign(contexts, false);
  }

  void publish_directory(std::uint32_t context) {
    checked_context(context);
    context_directory_ready_[context] = true;
  }

  bool claim(transport_state::TaskClass kind, std::uint32_t task,
             std::uint32_t context) {
    checked_context(context);
    auto &states = tasks_[index(kind)];
    if (task >= states.size() || states[task] != State::Waiting)
      return false;
    if (context_credit_[context] == 0U)
      return false;
    states[task] = State::Claimed;
    --context_credit_[context];
    ++context_inflight_[context];
    ++counters_[index(kind)].claimed;
    ++counters_[index(kind)].inflight;
    return true;
  }

  bool complete(transport_state::TaskClass kind, std::uint32_t task,
                std::uint32_t context) {
    checked_context(context);
    auto &states = tasks_[index(kind)];
    if (task >= states.size() || states[task] != State::Claimed)
      return false;
    states[task] = State::Completed;
    ++context_credit_[context];
    --context_inflight_[context];
    ++counters_[index(kind)].completed;
    --counters_[index(kind)].inflight;
    return true;
  }

  bool terminal_ready() const {
    for (const auto ready : context_directory_ready_)
      if (!ready)
        return false;
    for (const auto &counter : counters_)
      if (counter.expected != counter.claimed ||
          counter.expected != counter.completed || counter.inflight != 0U)
        return false;
    for (std::size_t context = 0; context < contexts_; ++context)
      if (context_inflight_[context] != 0U ||
          context_credit_[context] != credits_per_context_)
        return false;
    return true;
  }

  bool publish_terminal_once() {
    if (!terminal_ready() || terminal_publish_count_ != 0U)
      return false;
    ++terminal_publish_count_;
    return true;
  }

  const auto &counters() const noexcept { return counters_; }
  std::uint32_t terminal_publish_count() const noexcept {
    return terminal_publish_count_;
  }
  std::uint32_t context_credit(std::uint32_t context) const {
    checked_context(context);
    return context_credit_[context];
  }

 private:
  static constexpr std::size_t index(transport_state::TaskClass kind) {
    return static_cast<std::size_t>(kind);
  }
  void checked_context(std::uint32_t context) const {
    if (context >= contexts_)
      throw std::out_of_range("generation scheduler context");
  }

  std::array<std::vector<State>, transport_state::kTaskClassCount> tasks_;
  std::array<transport_state::TaskCompletionRecord,
             transport_state::kTaskClassCount>
      counters_{};
  std::uint32_t credits_per_context_ = 0;
  std::uint32_t contexts_ = 0;
  std::vector<std::uint32_t> context_credit_;
  std::vector<std::uint32_t> context_inflight_;
  std::vector<bool> context_directory_ready_;
  std::uint32_t terminal_publish_count_ = 0;
};

} // namespace weftmoe::cuda::testing
