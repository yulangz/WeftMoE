#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "megamoe/cuda/device_notify.hpp"

namespace megamoe::cuda {

class NcclSession;

class NcclBootstrapId {
public:
  NcclBootstrapId() = default;
  std::string hex() const;
  explicit operator bool() const noexcept { return !bytes_.empty(); }

private:
  explicit NcclBootstrapId(std::vector<std::uint8_t> bytes)
      : bytes_(std::move(bytes)) {}
  std::vector<std::uint8_t> bytes_;
  friend NcclBootstrapId make_nccl_bootstrap_id();
  friend NcclBootstrapId parse_nccl_bootstrap_id(const std::string &);
  friend class NcclSession;
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
};

NcclBootstrapId make_nccl_bootstrap_id();
NcclBootstrapId parse_nccl_bootstrap_id(const std::string &hex);

class NcclSession {
public:
  NcclSession() = default;
  const CollectiveSession &session() const;
  std::uint32_t rank() const;
  std::uint32_t device_ordinal() const;
  explicit operator bool() const noexcept { return static_cast<bool>(impl_); }

private:
  struct Impl;
  explicit NcclSession(std::shared_ptr<const Impl> impl)
      : impl_(std::move(impl)) {}
  std::shared_ptr<const Impl> impl_;
  friend NcclSession make_nccl_session(std::string, std::uint32_t,
                                       std::uint32_t, std::uint64_t,
                                       std::uint32_t,
                                       const NcclBootstrapId &);
  friend class NcclAllGatherCountExchange;
};

NcclSession make_nccl_session(std::string identity, std::uint32_t world_size,
                              std::uint32_t rank, std::uint64_t generation,
                              std::uint32_t device_ordinal,
                              const NcclBootstrapId &bootstrap_id);

class NcclAllGatherCountExchange final : public CountExchange {
public:
  struct Impl;
  NcclAllGatherCountExchange(const NcclSession &session,
                             const CudaExecution &execution,
                             std::uint64_t max_expert_row_width,
                             std::uint64_t max_summary_row_width);
  ~NcclAllGatherCountExchange() noexcept override;
  NcclAllGatherCountExchange(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange &
  operator=(const NcclAllGatherCountExchange &) = delete;
  NcclAllGatherCountExchange(NcclAllGatherCountExchange &&) noexcept;
  NcclAllGatherCountExchange &
  operator=(NcclAllGatherCountExchange &&) noexcept;

  std::unique_ptr<BackendCountExchangeOperation>
  start(const CountExchangeRequest &request) override;

private:
  std::shared_ptr<Impl> impl_;
};

const char *compiled_nccl_version() noexcept;

} // namespace megamoe::cuda
