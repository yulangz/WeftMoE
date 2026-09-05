void validate_runtime_config(const RuntimeConfig &config) {
  (void)validate_config(config);
}

class DispatchHandleFactory {
public:
  static DispatchHandle make(std::shared_ptr<const DispatchMetadata> metadata) {
    if (!metadata) {
      fail(NotifyErrorCode::InvalidHandle,
           "cannot construct DispatchHandle from null metadata");
    }
    return DispatchHandle(
        std::move(metadata),
        std::make_shared<const DispatchHandle::ProvenanceToken>());
  }
};

const DispatchMetadata &DispatchHandle::metadata() const {
  if (!metadata_) {
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty/default DispatchHandle");
  }
  return *metadata_;
}

void validate_dispatch_metadata(const DispatchHandle &handle) {
  validate_dispatch_metadata_object(handle.metadata(), "dispatch-validation");
}

const RuntimeConfig &LocalHistogramHandle::config() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->config;
}

const RankCollectiveContext &LocalHistogramHandle::context() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->context;
}

const SourceRouteMetadata &LocalHistogramHandle::outgoing() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->outgoing;
}

const HostIntegerRow &LocalHistogramHandle::local_expert_count() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->local_expert_count;
}

const HostIntegerRow &
LocalHistogramHandle::packet_count_by_destination_node() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->packet_count_by_destination_node;
}

const DeviceBufferLease &LocalHistogramHandle::device_expert_count() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->device_expert_count;
}

const DeviceBufferLease &LocalHistogramHandle::device_packet_count() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty local histogram handle");
  return impl_->device_packet_count;
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_expert_id(
    const LocalHistogramHandle &source, std::size_t route_index,
    std::uint32_t expert_id) {
  if (!source.impl_) {
    fail(NotifyErrorCode::InvalidHandle,
         "corruption hook source local histogram is empty");
  }
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  if (route_index >= impl->outgoing.expert_id.size()) {
    fail(NotifyErrorCode::InvalidRouteShape,
         "corruption hook route index out of range: actual=" +
             std::to_string(route_index) + ", expected=[0," +
             std::to_string(impl->outgoing.expert_id.size()) + ")");
  }
  impl->outgoing.expert_id[route_index] = expert_id;
  return LocalHistogramHandle(std::move(impl));
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_truncated_expert_ids(
    const LocalHistogramHandle &source) {
  if (!source.impl_) {
    fail(NotifyErrorCode::InvalidHandle,
         "corruption hook source local histogram is empty");
  }
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  if (!impl->outgoing.expert_id.empty())
    impl->outgoing.expert_id.pop_back();
  return LocalHistogramHandle(std::move(impl));
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_local_offset(
    const LocalHistogramHandle &source, std::size_t route_index,
    std::uint64_t value) {
  if (!source.impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "phase=corruption-hook, field=local, rank=unknown, expected=valid, "
         "actual=empty");
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  impl->outgoing.local_offset.at(route_index) = value;
  return LocalHistogramHandle(std::move(impl));
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_combine_order(
    const LocalHistogramHandle &source, std::size_t route_index,
    std::uint32_t value) {
  if (!source.impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "phase=corruption-hook, field=local, rank=unknown, expected=valid, "
         "actual=empty");
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  impl->outgoing.combine_order.at(route_index) = value;
  return LocalHistogramHandle(std::move(impl));
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_node_mask(
    const LocalHistogramHandle &source, std::size_t word_index,
    std::uint64_t value) {
  if (!source.impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "phase=corruption-hook, field=local, rank=unknown, expected=valid, "
         "actual=empty");
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  impl->outgoing.destination_node_mask.at(word_index) = value;
  return LocalHistogramHandle(std::move(impl));
}

LocalHistogramHandle testing::NotifyCorruptionHook::with_local_expert_count(
    const LocalHistogramHandle &source, std::size_t expert_index,
    std::uint64_t value) {
  if (!source.impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "phase=corruption-hook, field=local, rank=unknown, expected=valid, "
         "actual=empty");
  auto impl = std::make_shared<LocalHistogramHandle::Impl>(*source.impl_);
  auto counts = impl->local_expert_count.values();
  counts.at(expert_index) = value;
  impl->local_expert_count = make_host_integer_row(std::move(counts));
  return LocalHistogramHandle(std::move(impl));
}

ValidatedCollectiveHandle testing::NotifyCorruptionHook::with_collective_matrix(
    const ValidatedCollectiveHandle &source, const CountMatrix &matrix) {
  if (!source.impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "phase=corruption-hook, field=completion, rank=unknown, "
         "expected=valid, "
         "actual=empty");
  auto impl = std::make_shared<ValidatedCollectiveHandle::Impl>(*source.impl_);
  impl->matrix = make_host_count_matrix(matrix);
  return ValidatedCollectiveHandle(std::move(impl));
}

DispatchHandle testing::NotifyCorruptionHook::with_expert_row_count(
    const DispatchHandle &source, std::size_t expert_index,
    std::uint64_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->incoming.expert_row_count.at(expert_index) = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_expert_row_prefix(
    const DispatchHandle &source, std::size_t prefix_index,
    std::uint64_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->incoming.expert_row_prefix.at(prefix_index) = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle
testing::NotifyCorruptionHook::with_packet_identity_destination_node(
    const DispatchHandle &source, std::size_t packet_index,
    std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .identity.destination_node = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle
testing::NotifyCorruptionHook::with_packet_proxy(const DispatchHandle &source,
                                                 std::size_t packet_index,
                                                 std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index).proxy_rank = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_packet_sequence(
    const DispatchHandle &source, std::size_t packet_index,
    std::uint64_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .identity.sequence_on_link = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_fanout_expert(
    const DispatchHandle &source, std::size_t packet_index,
    std::size_t target_index, std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .fanout.at(target_index)
      .expert_id = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_fanout_local_rank(
    const DispatchHandle &source, std::size_t packet_index,
    std::size_t target_index, std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .fanout.at(target_index)
      .destination_local_rank = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_fanout_row(
    const DispatchHandle &source, std::size_t packet_index,
    std::size_t target_index, std::uint64_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .fanout.at(target_index)
      .dst_row = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_fanout_slot(
    const DispatchHandle &source, std::size_t packet_index,
    std::size_t target_index, std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->transport.outgoing_packets.at(packet_index)
      .fanout.at(target_index)
      .topk_slot = value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

DispatchHandle testing::NotifyCorruptionHook::with_mtile_row(
    const DispatchHandle &source, std::size_t expert_index,
    std::size_t tile_index, std::uint32_t value) {
  auto metadata = std::make_shared<DispatchMetadata>(source.metadata());
  metadata->incoming.mtile_expected_rows.at(expert_index).at(tile_index) =
      value;
  return DispatchHandle(std::move(metadata), source.provenance_);
}

const RankCollectiveContext &ValidatedCollectiveHandle::context() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty validated collective handle");
  return impl_->context;
}

ExchangeKind ValidatedCollectiveHandle::kind() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty validated collective handle");
  return impl_->kind;
}

std::uint32_t ValidatedCollectiveHandle::rows() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty validated collective handle");
  return impl_->matrix.rows();
}

std::uint64_t ValidatedCollectiveHandle::row_width() const {
  if (!impl_)
    fail(NotifyErrorCode::InvalidHandle,
         "attempted to access an empty validated collective handle");
  return impl_->matrix.row_width();
}

LocalHistogramHandle
build_rank_local_histogram(const RuntimeConfig &config,
                           const RankCollectiveContext &context,
                           const RankRoutingInput &local_routes) {
  validate_config(config);
  validate_rank_context_for_config(context, config, "histogram",
                                   ExchangeKind::ExpertCounts);
  const auto calling_rank = context.calling_rank();
  if (calling_rank >= config.topology.world_size) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "calling rank out of range: rank=" + std::to_string(calling_rank) +
             ", world_size=" + std::to_string(config.topology.world_size));
  }
  const auto num_routes = checked_mul(local_routes.num_tokens, config.top_k,
                                      "rank-local route count for rank=" +
                                          std::to_string(calling_rank));
  if (local_routes.selected_experts.size() !=
      checked_size(num_routes, "route array")) {
    fail(NotifyErrorCode::InvalidRouteShape,
         "selected_experts shape mismatch: rank=" +
             std::to_string(calling_rank) + ", actual=" +
             std::to_string(local_routes.selected_experts.size()) +
             ", expected=" + std::to_string(num_routes));
  }

  auto local = std::make_shared<LocalHistogramHandle::Impl>();
  local->config = config;
  local->context = context;
  local->provenance = std::make_shared<const std::uint8_t>(0);
  std::vector<std::uint64_t> counts(config.num_experts, 0);
  std::vector<std::uint64_t> packet_counts(config.topology.num_nodes, 0);
  auto &outgoing = local->outgoing;
  outgoing.num_tokens = local_routes.num_tokens;
  outgoing.top_k = config.top_k;
  outgoing.node_mask_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(config.topology.num_nodes) + 63U) / 64U);
  outgoing.expert_id.resize(local_routes.selected_experts.size());
  outgoing.local_offset.resize(local_routes.selected_experts.size());
  outgoing.route_dst_row.resize(local_routes.selected_experts.size());
  outgoing.route_dst_rank.resize(local_routes.selected_experts.size());
  outgoing.route_dst_node.resize(local_routes.selected_experts.size());
  outgoing.route_dst_local_rank.resize(local_routes.selected_experts.size());
  outgoing.combine_order.resize(local_routes.selected_experts.size());
  outgoing.destination_node_mask.assign(
      checked_size(checked_mul(local_routes.num_tokens,
                               outgoing.node_mask_words,
                               "destination node masks for rank=" +
                                   std::to_string(calling_rank)),
                   "destination node masks"),
      0);

  for (std::uint64_t token = 0; token < local_routes.num_tokens; ++token) {
    std::unordered_set<std::uint32_t> selected;
    std::vector<std::uint32_t> slots(config.top_k);
    std::iota(slots.begin(), slots.end(), 0U);
    std::vector<bool> token_nodes(config.topology.num_nodes, false);
    const auto base = checked_size(
        checked_mul(token, config.top_k, "route base"), "route base");
    for (std::uint32_t slot = 0; slot < config.top_k; ++slot) {
      const auto flat = base + slot;
      const auto raw_expert = local_routes.selected_experts[flat];
      if (raw_expert < 0 ||
          static_cast<std::uint32_t>(raw_expert) >= config.num_experts) {
        fail(NotifyErrorCode::ExpertOutOfRange,
             "expert ID out of range at " +
                 route_context(calling_rank, token, slot) +
                 ": actual=" + std::to_string(raw_expert) + ", expected [0," +
                 std::to_string(config.num_experts) + ")");
      }
      const auto expert = static_cast<std::uint32_t>(raw_expert);
      if (!selected.insert(expert).second) {
        fail(NotifyErrorCode::DuplicateExpert,
             "duplicate expert at " + route_context(calling_rank, token, slot) +
                 ": expert=" + std::to_string(expert));
      }
      const auto destination_rank = config.topology.expert_to_rank[expert];
      const auto destination_node =
          config.topology.rank_to_node[destination_rank];
      outgoing.expert_id[flat] = expert;
      outgoing.local_offset[flat] = counts[expert];
      counts[expert] = checked_add(
          counts[expert], 1,
          "local expert count for rank=" + std::to_string(calling_rank) +
              ", expert=" + std::to_string(expert));
      outgoing.route_dst_rank[flat] = destination_rank;
      outgoing.route_dst_node[flat] = destination_node;
      outgoing.route_dst_local_rank[flat] =
          config.topology.rank_to_local_rank[destination_rank];
      const auto mask_base = checked_size(
          checked_mul(token, outgoing.node_mask_words, "node mask base"),
          "node mask base");
      outgoing.destination_node_mask[mask_base + destination_node / 64U] |=
          std::uint64_t{1} << (destination_node % 64U);
      token_nodes[destination_node] = true;
    }
    std::stable_sort(slots.begin(), slots.end(), [&](auto lhs, auto rhs) {
      return outgoing.expert_id[base + lhs] < outgoing.expert_id[base + rhs];
    });
    for (std::uint32_t position = 0; position < config.top_k; ++position) {
      outgoing.combine_order[base + position] = slots[position];
    }
    for (std::uint32_t node = 0; node < config.topology.num_nodes; ++node) {
      if (token_nodes[node]) {
        packet_counts[node] = checked_add(
            packet_counts[node], 1,
            "packet summary for rank=" + std::to_string(calling_rank) +
                ", destination_node=" + std::to_string(node));
      }
    }
  }
  local->local_expert_count = make_host_integer_row(std::move(counts));
  local->packet_count_by_destination_node =
      make_host_integer_row(std::move(packet_counts));
  return LocalHistogramHandle(std::move(local));
}

DispatchHandle cuda::DeviceNotifyBridge::make_validation_dispatch(
    DispatchMetadata metadata) {
  auto value = std::make_shared<DispatchMetadata>(std::move(metadata));
  value->identity.content_hash = hash_metadata(*value);
  return DispatchHandle(std::move(value),
                        std::make_shared<DispatchHandle::ProvenanceToken>());
}

CollectiveOperation start_count_exchange(const LocalHistogramHandle &local,
                                         CountExchange &backend) {
  if (!local) {
    fail(NotifyErrorCode::CountExchangeFailure,
         "phase=start, kind=expert_counts, context=empty, local handle "
         "mismatch: expected=valid, actual=empty/default");
  }
  const auto &context = local.context();
  auto impl = std::make_unique<CollectiveOperation::Impl>();
  auto resources = std::make_shared<OperationResourceLease>();
  resources->local = local;
  resources->context = context;
  resources->kind = ExchangeKind::ExpertCounts;
  resources->row_width = local.config().num_experts;
  resources->input = local.local_expert_count();
  try {
    resources->backend = backend.start(CountExchangeRequest{
        context, resources->kind, resources->row_width, resources->input,
        local.device_expert_count()});
    if (!resources->backend)
      throw std::runtime_error("backend returned null operation");
  } catch (const NotifyDispatchError &error) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (const std::exception &error) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (...) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='unknown non-standard exception'");
  }
  impl->resources = std::move(resources);
  return CollectiveOperation(std::move(impl));
}

CollectiveOperation
start_notify_summary_exchange(const LocalHistogramHandle &local,
                              CountExchange &backend) {
  if (!local) {
    fail(NotifyErrorCode::CountExchangeFailure,
         "phase=start, kind=notify_packet_summary, context=empty, local handle "
         "mismatch: expected=valid, actual=empty/default");
  }
  const auto &context = local.context();
  auto impl = std::make_unique<CollectiveOperation::Impl>();
  auto resources = std::make_shared<OperationResourceLease>();
  resources->local = local;
  resources->context = context;
  resources->kind = ExchangeKind::NotifyPacketSummary;
  resources->row_width = local.config().topology.num_nodes;
  resources->input = local.packet_count_by_destination_node();
  try {
    resources->backend = backend.start(CountExchangeRequest{
        context, resources->kind, resources->row_width, resources->input,
        local.device_packet_count()});
    if (!resources->backend)
      throw std::runtime_error("backend returned null operation");
  } catch (const NotifyDispatchError &error) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (const std::exception &error) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (...) {
    exchange_failure("start", resources->kind, context,
                     "backend_error='unknown non-standard exception'");
  }
  impl->resources = std::move(resources);
  return CollectiveOperation(std::move(impl));
}

CollectiveOperation::CollectiveOperation() = default;

CollectiveOperation::CollectiveOperation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

CollectiveOperation::~CollectiveOperation() noexcept {
  if (!impl_ || !impl_->resources || impl_->resources->terminal ||
      !impl_->resources->backend)
    return;
  try {
    impl_->resources->backend->cancel();
  } catch (...) {
    // Destructors are a no-throw cleanup boundary. Explicit cancel() exposes
    // the same backend error as CountExchangeFailure when the caller cares.
  }
  impl_->resources->terminal = true;
  impl_->resources->terminal_state = ExchangeState::Cancelled;
}

CollectiveOperation::CollectiveOperation(CollectiveOperation &&other) noexcept =
    default;

CollectiveOperation &
CollectiveOperation::operator=(CollectiveOperation &&other) noexcept {
  if (this == &other)
    return *this;
  if (impl_ && impl_->resources && !impl_->resources->terminal &&
      impl_->resources->backend) {
    try {
      impl_->resources->backend->cancel();
    } catch (...) {
    }
    impl_->resources->terminal = true;
    impl_->resources->terminal_state = ExchangeState::Cancelled;
  }
  impl_ = std::move(other.impl_);
  return *this;
}

ExchangeState CollectiveOperation::state() const noexcept {
  if (!impl_)
    return ExchangeState::Failed;
  const auto &resources = impl_->resources;
  if (!resources)
    return ExchangeState::Failed;
  if (resources->terminal)
    return resources->terminal_state;
  if (!resources->backend)
    return ExchangeState::Failed;
  return resources->backend->state();
}

ValidatedCollectiveHandle CollectiveOperation::wait() {
  if (!impl_) {
    fail(
        NotifyErrorCode::CountExchangeFailure,
        "phase=wait, operation mismatch: expected=valid, actual=empty/default");
  }
  auto resources = impl_->resources;
  if (!resources) {
    fail(NotifyErrorCode::CountExchangeFailure,
         "phase=wait, field=operation.resources, rank=unknown, expected=valid, "
         "actual=empty");
  }
  if (resources->terminal || resources->waited) {
    exchange_failure(
        "wait", resources->kind, resources->context,
        "operation state mismatch: expected=pending, actual=terminal");
  }
  resources->waited = true;
  try {
    auto completion = resources->backend->wait();
    validate_backend_completion(completion, resources->context, resources->kind,
                                resources->row_width);
    auto completion_resources = std::make_shared<CompletionResourceLease>();
    completion_resources->operation = resources;
    completion_resources->completion = std::move(completion);
    auto validated = std::make_shared<ValidatedCollectiveHandle::Impl>();
    validated->context = resources->context;
    validated->kind = resources->kind;
    validated->matrix = completion_resources->completion.host_result();
    validated->local_provenance = resources->local.impl_->provenance;
    validated->resources = std::move(completion_resources);
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Complete;
    return ValidatedCollectiveHandle(std::move(validated));
  } catch (const NotifyDispatchError &error) {
    std::string cleanup;
    try {
      resources->backend->cancel();
    } catch (const std::exception &cancel_error) {
      cleanup = ", cancel_error='" + exception_text(cancel_error) + "'";
    } catch (...) {
      cleanup = ", cancel_error='unknown non-standard exception'";
    }
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("wait", resources->kind, resources->context,
                     "backend_error='" + exception_text(error) + "'" + cleanup);
  } catch (const std::exception &error) {
    std::string cleanup;
    try {
      resources->backend->cancel();
    } catch (const std::exception &cancel_error) {
      cleanup = ", cancel_error='" + exception_text(cancel_error) + "'";
    } catch (...) {
      cleanup = ", cancel_error='unknown non-standard exception'";
    }
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("wait", resources->kind, resources->context,
                     "backend_error='" + exception_text(error) + "'" + cleanup);
  } catch (...) {
    std::string cleanup;
    try {
      resources->backend->cancel();
    } catch (...) {
      cleanup = ", cancel_error='exception during cleanup'";
    }
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("wait", resources->kind, resources->context,
                     "backend_error='unknown non-standard exception'" +
                         cleanup);
  }
}

void CollectiveOperation::cancel() {
  if (!impl_) {
    fail(NotifyErrorCode::CountExchangeFailure,
         "phase=cancel, operation mismatch: expected=valid, "
         "actual=empty/default");
  }
  auto resources = impl_->resources;
  if (!resources) {
    fail(NotifyErrorCode::CountExchangeFailure,
         "phase=cancel, field=operation.resources, rank=unknown, "
         "expected=valid, "
         "actual=empty");
  }
  if (resources->terminal)
    return;
  try {
    resources->backend->cancel();
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Cancelled;
  } catch (const NotifyDispatchError &error) {
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("cancel", resources->kind, resources->context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (const std::exception &error) {
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("cancel", resources->kind, resources->context,
                     "backend_error='" + exception_text(error) + "'");
  } catch (...) {
    resources->terminal = true;
    resources->terminal_state = ExchangeState::Failed;
    exchange_failure("cancel", resources->kind, resources->context,
                     "backend_error='unknown non-standard exception'");
  }
}

