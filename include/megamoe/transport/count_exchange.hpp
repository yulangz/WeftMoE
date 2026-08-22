#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace megamoe {

using CountMatrix = std::vector<std::vector<std::uint64_t>>;

enum class MemoryLocation { Host, Device };
enum class ExchangeKind { ExpertCounts, NotifyPacketSummary };
enum class ExchangeState { Pending, Complete, Failed };

// Opaque handles deliberately avoid leaking CUDA, NCCL, or GIN types through
// the transport-independent metadata layer.
struct ExecutionContext {
    std::uintptr_t stream = 0;
    std::uintptr_t backend_context = 0;
};

struct CollectiveContext {
    std::uint32_t calling_rank = 0;
    std::uint32_t world_size = 0;
    std::string communicator_id;
    std::uint64_t generation = 0;
    ExecutionContext execution;
};

// Type-erased integer buffer descriptor. `owner` keeps the allocation alive;
// `address` is interpreted only by a backend supporting `location`.
struct IntegerBuffer {
    MemoryLocation location = MemoryLocation::Host;
    std::uint64_t elements = 0;
    std::uintptr_t address = 0;
    std::shared_ptr<const void> owner;
};

struct CountExchangeRequest {
    CollectiveContext collective;
    ExchangeKind kind = ExchangeKind::ExpertCounts;
    std::uint64_t row_width = 0;
    // Caller-owned immutable input. Backends retain `owner` while pending.
    IntegerBuffer input;
};

struct CountExchangeResult {
    CollectiveContext collective;
    ExchangeKind kind = ExchangeKind::ExpertCounts;
    std::uint32_t rows = 0;
    std::uint64_t row_width = 0;

    // Operation-owned until wait(); shared result ownership afterwards.
    IntegerBuffer output;

    // Host control-plane helper. Device output fails explicitly rather than
    // being dereferenced by the generic metadata layer.
    CountMatrix copy_host_rows() const;
};

IntegerBuffer make_host_integer_buffer(
    std::shared_ptr<const std::vector<std::uint64_t>> values);
CountExchangeResult make_host_exchange_result(
    CollectiveContext context, ExchangeKind kind, const CountMatrix& rows);

class CountExchangeOperation {
public:
    virtual ~CountExchangeOperation() = default;
    virtual ExchangeState state() const noexcept = 0;
    virtual std::string error_message() const = 0;
    virtual CountExchangeResult wait() = 0;
};

class CountExchange {
public:
    virtual ~CountExchange() = default;
    virtual std::unique_ptr<CountExchangeOperation> start(
        const CountExchangeRequest& request) = 0;
};

// Host-only fake collective. It is a deterministic test backend, not an NCCL,
// GIN, LSA, or RDMA implementation. All ranks share one instance and start the
// same (communicator, generation, kind) round before any wait.
class HostFakeCountExchange final : public CountExchange {
public:
    struct SharedState;

    HostFakeCountExchange(std::uint32_t world_size, std::string communicator_id);
    ~HostFakeCountExchange() override;

    std::unique_ptr<CountExchangeOperation> start(
        const CountExchangeRequest& request) override;

    void fail_next_start(std::string message);
    void fail_next_completion(std::string message);

private:
    std::shared_ptr<SharedState> state_;
};

}  // namespace megamoe
