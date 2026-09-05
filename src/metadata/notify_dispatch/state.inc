struct LocalHistogramHandle::Impl {
  RuntimeConfig config;
  RankCollectiveContext context;
  SourceRouteMetadata outgoing;
  HostIntegerRow local_expert_count;
  HostIntegerRow packet_count_by_destination_node;
  DeviceBufferLease device_expert_count;
  DeviceBufferLease device_packet_count;
  std::shared_ptr<const std::uint8_t> provenance;
};

struct OperationResourceLease {
  LocalHistogramHandle local;
  RankCollectiveContext context;
  ExchangeKind kind = ExchangeKind::ExpertCounts;
  std::uint64_t row_width = 0;
  HostIntegerRow input;
  std::unique_ptr<BackendCountExchangeOperation> backend;
  bool terminal = false;
  bool waited = false;
  ExchangeState terminal_state = ExchangeState::Pending;
};

struct CompletionResourceLease {
  std::shared_ptr<OperationResourceLease> operation;
  BackendCompletion completion;
};

struct ValidatedCollectiveHandle::Impl {
  RankCollectiveContext context;
  ExchangeKind kind = ExchangeKind::ExpertCounts;
  HostCountMatrix matrix;
  std::shared_ptr<const std::uint8_t> local_provenance;
  std::shared_ptr<const CompletionResourceLease> resources;
};

struct CollectiveOperation::Impl {
  std::shared_ptr<OperationResourceLease> resources;
};

