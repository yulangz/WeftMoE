namespace {

[[noreturn]] void fail(NotifyErrorCode code, const std::string &message) {
  throw NotifyDispatchError(code, message);
}

std::uint64_t checked_add(std::uint64_t lhs, std::uint64_t rhs,
                          const std::string &context) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    fail(NotifyErrorCode::ArithmeticOverflow,
         "uint64 overflow while computing " + context +
             ": lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs));
  }
  return lhs + rhs;
}

std::uint64_t checked_mul(std::uint64_t lhs, std::uint64_t rhs,
                          const std::string &context) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    fail(NotifyErrorCode::ArithmeticOverflow,
         "uint64 overflow while computing " + context +
             ": lhs=" + std::to_string(lhs) + ", rhs=" + std::to_string(rhs));
  }
  return lhs * rhs;
}

std::size_t checked_size(std::uint64_t value, const std::string &context) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    fail(NotifyErrorCode::ArithmeticOverflow,
         "size_t overflow while computing " + context +
             ": value=" + std::to_string(value));
  }
  return static_cast<std::size_t>(value);
}

struct DerivedTopology {
  std::uint32_t gpus_per_node = 0;
  // [node, local_rank] -> global rank.
  std::vector<std::uint32_t> rank_at;
};

DerivedTopology validate_config(const RuntimeConfig &config) {
  const auto &topology = config.topology;
  if (config.num_experts == 0 || config.top_k == 0 || config.m_tile_rows == 0) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "num_experts, top_k, and m_tile_rows must be positive: experts=" +
             std::to_string(config.num_experts) +
             ", top_k=" + std::to_string(config.top_k) +
             ", m_tile_rows=" + std::to_string(config.m_tile_rows));
  }
  if (config.top_k > config.num_experts) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "top_k exceeds num_experts even for zero-token input: top_k=" +
             std::to_string(config.top_k) +
             ", num_experts=" + std::to_string(config.num_experts));
  }
  if (topology.world_size == 0 || topology.num_nodes == 0) {
    fail(NotifyErrorCode::InvalidTopology,
         "world_size and num_nodes must be positive: world_size=" +
             std::to_string(topology.world_size) +
             ", num_nodes=" + std::to_string(topology.num_nodes));
  }
  if (topology.world_size % topology.num_nodes != 0) {
    fail(NotifyErrorCode::InvalidTopology,
         "heterogeneous topology is unsupported: world_size=" +
             std::to_string(topology.world_size) +
             " is not divisible by num_nodes=" +
             std::to_string(topology.num_nodes));
  }
  if (topology.rank_to_node.size() != topology.world_size ||
      topology.rank_to_local_rank.size() != topology.world_size) {
    fail(NotifyErrorCode::InvalidTopology,
         "rank coordinate arrays must contain every rank exactly once: "
         "node_entries=" +
             std::to_string(topology.rank_to_node.size()) + ", local_entries=" +
             std::to_string(topology.rank_to_local_rank.size()) +
             ", expected=" + std::to_string(topology.world_size));
  }
  if (topology.expert_to_rank.size() != config.num_experts) {
    fail(NotifyErrorCode::InvalidTopology,
         "expert ownership size mismatch: actual=" +
             std::to_string(topology.expert_to_rank.size()) +
             ", expected=" + std::to_string(config.num_experts));
  }
  if (!config.capacity.expert_rows.empty() &&
      config.capacity.expert_rows.size() != config.num_experts) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "expert capacity size mismatch: actual=" +
             std::to_string(config.capacity.expert_rows.size()) +
             ", expected=" + std::to_string(config.num_experts));
  }
  if (!config.capacity.workspace_rows_per_rank.empty() &&
      config.capacity.workspace_rows_per_rank.size() != topology.world_size) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "workspace capacity size mismatch: actual=" +
             std::to_string(config.capacity.workspace_rows_per_rank.size()) +
             ", expected=" + std::to_string(topology.world_size));
  }

  DerivedTopology derived;
  derived.gpus_per_node = topology.world_size / topology.num_nodes;
  derived.rank_at.assign(topology.world_size,
                         std::numeric_limits<std::uint32_t>::max());
  std::vector<std::uint32_t> ranks_per_node(topology.num_nodes, 0);
  for (std::uint32_t rank = 0; rank < topology.world_size; ++rank) {
    const auto node = topology.rank_to_node[rank];
    const auto local_rank = topology.rank_to_local_rank[rank];
    if (node >= topology.num_nodes) {
      fail(NotifyErrorCode::InvalidTopology,
           "rank has out-of-range node coordinate: rank=" +
               std::to_string(rank) + ", node=" + std::to_string(node) +
               ", expected node in [0," + std::to_string(topology.num_nodes) +
               ")");
    }
    if (local_rank >= derived.gpus_per_node) {
      fail(NotifyErrorCode::InvalidTopology,
           "rank has sparse/out-of-range local coordinate: rank=" +
               std::to_string(rank) + ", node=" + std::to_string(node) +
               ", local_rank=" + std::to_string(local_rank) + ", expected [0," +
               std::to_string(derived.gpus_per_node) + ")");
    }
    const auto coordinate =
        static_cast<std::size_t>(node) * derived.gpus_per_node + local_rank;
    if (derived.rank_at[coordinate] !=
        std::numeric_limits<std::uint32_t>::max()) {
      fail(NotifyErrorCode::InvalidTopology,
           "duplicate topology coordinate: node=" + std::to_string(node) +
               ", local_rank=" + std::to_string(local_rank) +
               ", ranks=" + std::to_string(derived.rank_at[coordinate]) +
               " and " + std::to_string(rank));
    }
    derived.rank_at[coordinate] = rank;
    ++ranks_per_node[node];
  }
  for (std::uint32_t node = 0; node < topology.num_nodes; ++node) {
    if (ranks_per_node[node] != derived.gpus_per_node) {
      fail(NotifyErrorCode::InvalidTopology,
           "homogeneous topology violation at node=" + std::to_string(node) +
               ": ranks=" + std::to_string(ranks_per_node[node]) +
               ", expected=" + std::to_string(derived.gpus_per_node));
    }
    for (std::uint32_t local = 0; local < derived.gpus_per_node; ++local) {
      const auto coordinate =
          static_cast<std::size_t>(node) * derived.gpus_per_node + local;
      if (derived.rank_at[coordinate] ==
          std::numeric_limits<std::uint32_t>::max()) {
        fail(NotifyErrorCode::InvalidTopology,
             "missing same-rail coordinate: node=" + std::to_string(node) +
                 ", local_rank=" + std::to_string(local));
      }
    }
  }
  for (std::uint32_t expert = 0; expert < config.num_experts; ++expert) {
    const auto owner = topology.expert_to_rank[expert];
    if (owner >= topology.world_size) {
      fail(NotifyErrorCode::InvalidTopology,
           "expert owner out of range: expert=" + std::to_string(expert) +
               ", owner=" + std::to_string(owner) +
               ", world_size=" + std::to_string(topology.world_size));
    }
  }
  return derived;
}

std::string route_context(std::uint32_t rank, std::uint64_t token,
                          std::uint32_t slot) {
  return "rank=" + std::to_string(rank) + ", token=" + std::to_string(token) +
         ", topk_slot=" + std::to_string(slot);
}

const char *kind_name(ExchangeKind kind) noexcept {
  switch (kind) {
  case ExchangeKind::Control: return "control";
  case ExchangeKind::ExpertCounts: return "expert_counts";
  case ExchangeKind::NotifyPacketSummary: return "notify_packet_summary";
  }
  return "unknown";
}

std::string exchange_context(const char *phase, ExchangeKind kind,
                             const RankCollectiveContext &context) {
  if (!context) {
    return std::string("phase=") + phase + ", kind=" + kind_name(kind) +
           ", context=empty";
  }
  return std::string("phase=") + phase + ", kind=" + kind_name(kind) +
         ", rank=" + std::to_string(context.calling_rank()) +
         ", world_size=" + std::to_string(context.session().world_size()) +
         ", session='" + context.session().identity() +
         "', generation=" + std::to_string(context.session().generation()) +
         ", execution='" + context.execution().identity() + "'";
}

[[noreturn]] void exchange_failure(const char *phase, ExchangeKind kind,
                                   const RankCollectiveContext &context,
                                   const std::string &detail) {
  fail(NotifyErrorCode::CountExchangeFailure,
       exchange_context(phase, kind, context) + ", " + detail);
}

void validate_rank_context_for_config(const RankCollectiveContext &context,
                                      const RuntimeConfig &config,
                                      const char *phase, ExchangeKind kind) {
  if (!context) {
    exchange_failure(phase, kind, context,
                     "rank context mismatch: expected=valid, actual=empty");
  }
  if (context.session().world_size() != config.topology.world_size) {
    exchange_failure(
        phase, kind, context,
        "world_size mismatch: expected=" +
            std::to_string(config.topology.world_size) +
            ", actual=" + std::to_string(context.session().world_size()));
  }
  if (context.calling_rank() >= config.topology.world_size) {
    exchange_failure(phase, kind, context,
                     "calling_rank mismatch: expected=[0," +
                         std::to_string(config.topology.world_size) +
                         "), actual=" + std::to_string(context.calling_rank()));
  }
}

void validate_backend_completion(const BackendCompletion &completion,
                                 const RankCollectiveContext &expected,
                                 ExchangeKind expected_kind,
                                 std::uint64_t expected_width) {
  const char *phase = "wait";
  const auto &actual = completion.context();
  if (actual.calling_rank() != expected.calling_rank()) {
    exchange_failure(phase, expected_kind, expected,
                     "calling_rank mismatch: expected=" +
                         std::to_string(expected.calling_rank()) +
                         ", actual=" + std::to_string(actual.calling_rank()));
  }
  if (actual.session().world_size() != expected.session().world_size()) {
    exchange_failure(
        phase, expected_kind, expected,
        "world_size mismatch: expected=" +
            std::to_string(expected.session().world_size()) +
            ", actual=" + std::to_string(actual.session().world_size()));
  }
  if (actual.session().identity() != expected.session().identity()) {
    exchange_failure(phase, expected_kind, expected,
                     "session identity mismatch: expected='" +
                         expected.session().identity() + "', actual='" +
                         actual.session().identity() + "'");
  }
  if (actual.session().generation() != expected.session().generation()) {
    exchange_failure(
        phase, expected_kind, expected,
        "generation mismatch: expected=" +
            std::to_string(expected.session().generation()) +
            ", actual=" + std::to_string(actual.session().generation()));
  }
  if (!actual.session().same_instance(expected.session())) {
    exchange_failure(
        phase, expected_kind, expected,
        "session provenance mismatch: expected=same session instance, "
        "actual=different session instance");
  }
  if (!actual.execution().same_instance(expected.execution())) {
    exchange_failure(phase, expected_kind, expected,
                     "execution token mismatch: expected='" +
                         expected.execution().identity() + "', actual='" +
                         actual.execution().identity() + "'");
  }
  if (completion.kind() != expected_kind) {
    exchange_failure(phase, expected_kind, expected,
                     "collective kind mismatch: expected=" +
                         std::string(kind_name(expected_kind)) +
                         ", actual=" + kind_name(completion.kind()));
  }
  if (!completion.has_host_result()) {
    exchange_failure(
        phase, expected_kind, expected,
        "host result mismatch: expected=typed host matrix, actual=" +
            std::string(completion.has_device_result() ? "opaque device lease"
                                                       : "empty"));
  }
  const auto &matrix = completion.host_result();
  if (matrix.rows() != expected.session().world_size()) {
    exchange_failure(phase, expected_kind, expected,
                     "field=matrix.rows, expected=" +
                         std::to_string(expected.session().world_size()) +
                         ", actual=" + std::to_string(matrix.rows()));
  }
  if (matrix.row_width() != expected_width) {
    exchange_failure(
        phase, expected_kind, expected,
        "field=matrix.row_width, expected=" + std::to_string(expected_width) +
            ", actual=" + std::to_string(matrix.row_width()));
  }
  (void)checked_mul(matrix.rows(), matrix.row_width(),
                    "validated collective matrix elements");
}

std::string exception_text(const NotifyDispatchError &error) {
  return std::string("NotifyDispatchError: ") + error.what();
}

std::string exception_text(const std::exception &error) {
  return std::string("std::exception: ") + error.what();
}

[[noreturn]] void metadata_failure(NotifyErrorCode code, const char *phase,
                                   const std::string &field, std::uint32_t rank,
                                   const std::string &expected,
                                   const std::string &actual,
                                   const std::string &coordinates = {}) {
  fail(code, std::string("phase=") + phase + ", field=" + field +
                 ", rank=" + std::to_string(rank) + ", expected=" + expected +
                 ", actual=" + actual +
                 (coordinates.empty() ? "" : ", " + coordinates));
}

void validate_local_structure(const LocalHistogramHandle &local) {
  const auto &config = local.config();
  (void)validate_config(config);
  validate_rank_context_for_config(local.context(), config, "materialize-local",
                                   ExchangeKind::ExpertCounts);
  const auto rank = local.context().calling_rank();
  const auto &outgoing = local.outgoing();
  if (outgoing.top_k != config.top_k) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "outgoing.top_k", rank, std::to_string(config.top_k),
                     std::to_string(outgoing.top_k));
  }
  const auto routes_u64 = checked_mul(outgoing.num_tokens, outgoing.top_k,
                                      "local structural route count");
  const auto routes = checked_size(routes_u64, "local structural route count");
  const auto require_routes = [&](std::size_t actual, const char *field) {
    if (actual != routes) {
      metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                       field, rank, std::to_string(routes),
                       std::to_string(actual));
    }
  };
  require_routes(outgoing.expert_id.size(), "expert_id");
  require_routes(outgoing.local_offset.size(), "local_offset");
  require_routes(outgoing.route_dst_row.size(), "route_dst_row");
  require_routes(outgoing.route_dst_rank.size(), "route_dst_rank");
  require_routes(outgoing.route_dst_node.size(), "route_dst_node");
  require_routes(outgoing.route_dst_local_rank.size(), "route_dst_local_rank");
  require_routes(outgoing.combine_order.size(), "combine_order");
  if (!outgoing.reverse_routes.empty()) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "outgoing.reverse_routes", rank, "empty",
                     std::to_string(outgoing.reverse_routes.size()));
  }
  const auto expected_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(config.topology.num_nodes) + 63U) / 64U);
  if (outgoing.node_mask_words != expected_words) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "outgoing.node_mask_words", rank,
                     std::to_string(expected_words),
                     std::to_string(outgoing.node_mask_words));
  }
  const auto mask_elements =
      checked_size(checked_mul(outgoing.num_tokens, expected_words,
                               "local structural mask elements"),
                   "local structural mask elements");
  if (outgoing.destination_node_mask.size() != mask_elements) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "outgoing.destination_node_mask.size", rank,
                     std::to_string(mask_elements),
                     std::to_string(outgoing.destination_node_mask.size()));
  }
  if (local.local_expert_count().size() != config.num_experts) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "local_expert_count.size", rank,
                     std::to_string(config.num_experts),
                     std::to_string(local.local_expert_count().size()));
  }
  if (local.packet_count_by_destination_node().size() !=
      config.topology.num_nodes) {
    metadata_failure(
        NotifyErrorCode::InvalidRouteShape, "materialize-local",
        "packet_count_by_destination_node.size", rank,
        std::to_string(config.topology.num_nodes),
        std::to_string(local.packet_count_by_destination_node().size()));
  }

  std::vector<std::uint64_t> counts(config.num_experts, 0);
  std::vector<std::uint64_t> packets(config.topology.num_nodes, 0);
  std::vector<std::uint64_t> masks(mask_elements, 0);
  for (std::uint64_t token = 0; token < outgoing.num_tokens; ++token) {
    const auto base = checked_size(
        checked_mul(token, outgoing.top_k, "local validation route base"),
        "local validation route base");
    std::vector<bool> nodes(config.topology.num_nodes, false);
    std::vector<bool> slots(config.top_k, false);
    std::vector<bool> selected_experts(config.num_experts, false);
    std::uint32_t prior_expert = 0;
    for (std::uint32_t slot = 0; slot < outgoing.top_k; ++slot) {
      const auto flat = base + slot;
      const auto expert = outgoing.expert_id[flat];
      if (expert >= config.num_experts) {
        metadata_failure(NotifyErrorCode::ExpertOutOfRange, "materialize-local",
                         "outgoing.expert_id", rank,
                         "[0," + std::to_string(config.num_experts) + ")",
                         std::to_string(expert),
                         "token=" + std::to_string(token) +
                             ", slot=" + std::to_string(slot));
      }
      if (selected_experts[expert]) {
        metadata_failure(NotifyErrorCode::DuplicateExpert, "materialize-local",
                         "outgoing.expert_id", rank, "unique within token",
                         std::to_string(expert),
                         "token=" + std::to_string(token) +
                             ", slot=" + std::to_string(slot));
      }
      selected_experts[expert] = true;
      const auto destination_rank = config.topology.expert_to_rank[expert];
      const auto destination_node =
          config.topology.rank_to_node[destination_rank];
      const auto destination_local =
          config.topology.rank_to_local_rank[destination_rank];
      if (outgoing.local_offset[flat] != counts[expert]) {
        metadata_failure(NotifyErrorCode::InvalidRouteShape,
                         "materialize-local", "outgoing.local_offset", rank,
                         std::to_string(counts[expert]),
                         std::to_string(outgoing.local_offset[flat]),
                         "token=" + std::to_string(token) +
                             ", slot=" + std::to_string(slot));
      }
      counts[expert] =
          checked_add(counts[expert], 1, "validated local expert count");
      if (outgoing.route_dst_rank[flat] != destination_rank ||
          outgoing.route_dst_node[flat] != destination_node ||
          outgoing.route_dst_local_rank[flat] != destination_local) {
        metadata_failure(NotifyErrorCode::InvalidRouteShape,
                         "materialize-local", "outgoing.destination", rank,
                         "topology-derived coordinates", "mismatch",
                         "token=" + std::to_string(token) +
                             ", slot=" + std::to_string(slot) +
                             ", expert=" + std::to_string(expert));
      }
      nodes[destination_node] = true;
      const auto mask_base = checked_size(
          checked_mul(token, expected_words, "validated mask base"),
          "validated mask base");
      masks[mask_base + destination_node / 64U] |= std::uint64_t{1}
                                                   << (destination_node % 64U);
    }
    for (std::uint32_t position = 0; position < outgoing.top_k; ++position) {
      const auto slot = outgoing.combine_order[base + position];
      if (slot >= outgoing.top_k || slots[slot]) {
        metadata_failure(NotifyErrorCode::InvalidRouteShape,
                         "materialize-local", "outgoing.combine_order", rank,
                         "slot permutation [0," +
                             std::to_string(outgoing.top_k) + ")",
                         std::to_string(slot),
                         "token=" + std::to_string(token) +
                             ", position=" + std::to_string(position));
      }
      slots[slot] = true;
      const auto expert = outgoing.expert_id[base + slot];
      if (position != 0 && expert < prior_expert) {
        metadata_failure(NotifyErrorCode::InvalidRouteShape,
                         "materialize-local", "outgoing.combine_order", rank,
                         "nondecreasing expert order", "decreasing order",
                         "token=" + std::to_string(token));
      }
      prior_expert = expert;
    }
    for (std::uint32_t node = 0; node < config.topology.num_nodes; ++node) {
      if (nodes[node])
        packets[node] =
            checked_add(packets[node], 1, "validated local packet summary");
    }
  }
  if (counts != local.local_expert_count().values()) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "local_expert_count", rank, "recomputed stable counts",
                     "content mismatch");
  }
  if (packets != local.packet_count_by_destination_node().values()) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "packet_count_by_destination_node", rank,
                     "recomputed token-node counts", "content mismatch");
  }
  if (masks != outgoing.destination_node_mask) {
    metadata_failure(NotifyErrorCode::InvalidRouteShape, "materialize-local",
                     "outgoing.destination_node_mask", rank,
                     "recomputed node mask", "content mismatch");
  }
}

class Hash64 {
public:
  void add(std::uint64_t value) {
    for (unsigned shift = 0; shift < 64; shift += 8) {
      value_ ^= (value >> shift) & 0xffU;
      value_ *= 1099511628211ULL;
    }
  }
  void add(const std::string &value) {
    add(value.size());
    for (const auto ch : value) {
      value_ ^= static_cast<unsigned char>(ch);
      value_ *= 1099511628211ULL;
    }
  }
  template <typename T> void add_vector(const std::vector<T> &values) {
    add(values.size());
    for (const auto value : values)
      add(static_cast<std::uint64_t>(value));
  }
  std::uint64_t value() const { return value_; }

private:
  std::uint64_t value_ = 1469598103934665603ULL;
};

std::uint64_t hash_metadata(const DispatchMetadata &metadata) {
  Hash64 hash;
  const auto &config = metadata.config;
  hash.add(config.num_experts);
  hash.add(config.top_k);
  hash.add(config.m_tile_rows);
  hash.add(config.topology.world_size);
  hash.add(config.topology.num_nodes);
  hash.add_vector(config.topology.rank_to_node);
  hash.add_vector(config.topology.rank_to_local_rank);
  hash.add_vector(config.topology.expert_to_rank);
  hash.add_vector(config.capacity.expert_rows);
  hash.add_vector(config.capacity.workspace_rows_per_rank);
  hash.add(metadata.calling_rank);
  hash.add(metadata.communicator_id);
  for (const auto &row : metadata.source_expert_count)
    hash.add_vector(row);
  const auto &outgoing = metadata.outgoing;
  hash.add(outgoing.num_tokens);
  hash.add(outgoing.top_k);
  hash.add(outgoing.node_mask_words);
  hash.add_vector(outgoing.expert_id);
  hash.add_vector(outgoing.local_offset);
  hash.add_vector(outgoing.route_dst_row);
  hash.add_vector(outgoing.route_dst_rank);
  hash.add_vector(outgoing.route_dst_node);
  hash.add_vector(outgoing.route_dst_local_rank);
  hash.add_vector(outgoing.combine_order);
  hash.add_vector(outgoing.destination_node_mask);
  for (const auto &reverse : outgoing.reverse_routes) {
    hash.add(reverse.source_rank);
    hash.add(reverse.local_token_id);
    hash.add(reverse.topk_slot);
    hash.add(reverse.expert_id);
    hash.add(reverse.destination_rank);
    hash.add(reverse.dst_row);
  }
  const auto &incoming = metadata.incoming;
  hash.add(incoming.rank);
  hash.add_vector(incoming.expert_ids);
  hash.add_vector(incoming.expert_row_count);
  hash.add_vector(incoming.expert_row_prefix);
  for (const auto &tiles : incoming.mtile_expected_rows)
    hash.add_vector(tiles);
  for (const auto &row : incoming.incoming_rows) {
    hash.add(row.source_rank);
    hash.add(row.expert_id);
    hash.add(row.source_expert_offset);
    hash.add(row.dst_row);
  }
  for (const auto &packet : metadata.transport.outgoing_packets) {
    hash.add(packet.identity.source_rank);
    hash.add(packet.identity.destination_node);
    hash.add(packet.identity.sequence_on_link);
    hash.add(packet.local_token_id);
    hash.add(packet.proxy_rank);
    hash.add(packet.channel);
    for (const auto &target : packet.fanout) {
      hash.add(target.topk_slot);
      hash.add(target.destination_local_rank);
      hash.add(target.expert_id);
      hash.add(target.dst_row);
    }
  }
  hash.add_vector(metadata.transport.dispatch_wave_packet_order);
  hash.add_vector(metadata.transport.dispatch_wave_offsets);
  hash.add_vector(metadata.transport.dispatch_wave_destination_node);
  hash.add_vector(metadata.transport.dispatch_wave_channel);
  hash.add_vector(metadata.transport.dispatch_wave_first_primary_tile);
  hash.add_vector(metadata.transport.dispatch_wave_last_primary_tile);
  hash.add_vector(metadata.transport.dispatch_wave_packet_count);
  hash.add_vector(metadata.transport.dispatch_wave_payload_bytes);
  const auto add_signals = [&hash](const auto &signals) {
    for (const auto &signal : signals) {
      hash.add(signal.source_node);
      hash.add(signal.destination_node);
      hash.add(signal.source_rank);
      hash.add(signal.destination_proxy_rank);
      hash.add(signal.rail_local_rank);
      hash.add(signal.channel);
      hash.add(signal.expected_packets);
      hash.add(signal.expected_signals);
    }
  };
  add_signals(metadata.transport.outgoing_signals);
  add_signals(metadata.transport.incoming_proxy_signals);
  return hash.value();
}

void materialize_dispatch_waves(TransportMetadata &transport,
                                const RuntimeConfig &config,
                                const std::vector<std::uint64_t> &expert_base) {
  struct PacketPriority {
    std::uint64_t packet = 0;
    std::uint64_t tile = 0;
    std::uint32_t expert = 0;
    std::uint32_t destination_rank = 0;
  };
  std::vector<PacketPriority> priorities;
  priorities.reserve(transport.outgoing_packets.size());
  for (std::size_t packet = 0; packet < transport.outgoing_packets.size();
       ++packet) {
    const auto &descriptor = transport.outgoing_packets[packet];
    if (descriptor.fanout.empty())
      fail(NotifyErrorCode::InvalidHandle,
           "wave packet has no fan-out: packet=" + std::to_string(packet));
    PacketPriority primary{static_cast<std::uint64_t>(packet),
                           std::numeric_limits<std::uint64_t>::max(),
                           std::numeric_limits<std::uint32_t>::max(),
                           std::numeric_limits<std::uint32_t>::max()};
    for (const auto &target : descriptor.fanout) {
      if (target.expert_id >= config.num_experts ||
          target.dst_row < expert_base[target.expert_id])
        fail(NotifyErrorCode::InvalidHandle,
             "wave fan-out has invalid canonical row: packet=" +
                 std::to_string(packet));
      const auto tile =
          (target.dst_row - expert_base[target.expert_id]) /
          config.m_tile_rows;
      const auto destination_rank =
          config.topology.expert_to_rank[target.expert_id];
      if (std::tie(tile, target.expert_id, destination_rank) <
          std::tie(primary.tile, primary.expert, primary.destination_rank)) {
        primary.tile = tile;
        primary.expert = target.expert_id;
        primary.destination_rank = destination_rank;
      }
    }
    priorities.push_back(primary);
  }
  const auto scheduled_expert = [&](const PacketPriority &priority) {
    auto key = static_cast<std::uint64_t>(priority.expert);
    if (config.topology.num_nodes != 0U &&
        config.topology.world_size == config.topology.num_nodes * 8U) {
      const auto local =
          config.topology.rank_to_local_rank[priority.destination_rank];
      const auto local_priority = ((local & 1U) << 2U) | (local & 2U) |
                                  ((local & 4U) >> 2U);
      key += static_cast<std::uint64_t>(local_priority) * config.num_experts;
    }
    return key;
  };
  std::sort(priorities.begin(), priorities.end(),
            [&](const auto &lhs, const auto &rhs) {
              const auto &left = transport.outgoing_packets[lhs.packet];
              const auto &right = transport.outgoing_packets[rhs.packet];
              return std::make_tuple(left.identity.destination_node, lhs.tile,
                                     scheduled_expert(lhs),
                                     lhs.destination_rank,
                                     left.local_token_id) <
                     std::make_tuple(right.identity.destination_node, rhs.tile,
                                     scheduled_expert(rhs),
                                     rhs.destination_rank,
                                     right.local_token_id);
            });
  auto &order = transport.dispatch_wave_packet_order;
  auto &offsets = transport.dispatch_wave_offsets;
  order.reserve(priorities.size());
  offsets.push_back(0);
  std::uint32_t previous_node = std::numeric_limits<std::uint32_t>::max();
  std::uint64_t previous_tile = std::numeric_limits<std::uint64_t>::max();
  for (const auto &priority : priorities) {
    const auto &packet = transport.outgoing_packets[priority.packet];
    if (packet.identity.destination_node != previous_node ||
        priority.tile != previous_tile ||
        (!transport.dispatch_wave_packet_count.empty() &&
         transport.dispatch_wave_packet_count.back() >=
             kDispatchMaxPacketsPerWave)) {
      if (!order.empty())
        offsets.push_back(order.size());
      transport.dispatch_wave_destination_node.push_back(
          packet.identity.destination_node);
      transport.dispatch_wave_channel.push_back(packet.channel);
      transport.dispatch_wave_first_primary_tile.push_back(priority.tile);
      transport.dispatch_wave_last_primary_tile.push_back(priority.tile);
      transport.dispatch_wave_packet_count.push_back(0);
      transport.dispatch_wave_payload_bytes.push_back(0);
      previous_node = packet.identity.destination_node;
      previous_tile = priority.tile;
    }
    order.push_back(priority.packet);
    transport.dispatch_wave_packet_count.back() += 1;
    transport.dispatch_wave_payload_bytes.back() +=
        kDispatchPacketHeaderDescriptorBytes +
        packet.fanout.size() * kDispatchFanoutDescriptorBytes;
  }
  if (!priorities.empty())
    offsets.push_back(order.size());
}

void validate_dispatch_metadata_object(const DispatchMetadata &metadata,
                                       const char *phase) {
  const auto derived = validate_config(metadata.config);
  const auto &config = metadata.config;
  const auto world = config.topology.world_size;
  const auto experts = config.num_experts;
  const auto rank = metadata.calling_rank;
  if (rank >= world) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase, "calling_rank",
                     rank, "[0," + std::to_string(world) + ")",
                     std::to_string(rank));
  }
  if (metadata.source_expert_count.size() != world) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "source_expert_count.rows", rank, std::to_string(world),
                     std::to_string(metadata.source_expert_count.size()));
  }
  for (std::uint32_t source = 0; source < world; ++source) {
    if (metadata.source_expert_count[source].size() != experts) {
      metadata_failure(
          NotifyErrorCode::InvalidHandle, phase,
          "source_expert_count.row_width", rank, std::to_string(experts),
          std::to_string(metadata.source_expert_count[source].size()),
          "source_rank=" + std::to_string(source));
    }
  }

  const auto &out = metadata.outgoing;
  if (out.top_k != config.top_k) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase, "outgoing.top_k",
                     rank, std::to_string(config.top_k),
                     std::to_string(out.top_k));
  }
  const auto route_count = checked_size(
      checked_mul(out.num_tokens, out.top_k, "dispatch route count"),
      "dispatch route count");
  const auto require_routes = [&](std::size_t actual, const char *field) {
    if (actual != route_count) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase, field, rank,
                       std::to_string(route_count), std::to_string(actual));
    }
  };
  require_routes(out.expert_id.size(), "outgoing.expert_id.size");
  require_routes(out.local_offset.size(), "outgoing.local_offset.size");
  require_routes(out.route_dst_row.size(), "outgoing.route_dst_row.size");
  require_routes(out.route_dst_rank.size(), "outgoing.route_dst_rank.size");
  require_routes(out.route_dst_node.size(), "outgoing.route_dst_node.size");
  require_routes(out.route_dst_local_rank.size(),
                 "outgoing.route_dst_local_rank.size");
  require_routes(out.combine_order.size(), "outgoing.combine_order.size");
  require_routes(out.reverse_routes.size(), "outgoing.reverse_routes.size");
  const auto expected_mask_words = static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(config.topology.num_nodes) + 63U) / 64U);
  if (out.node_mask_words != expected_mask_words) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "outgoing.node_mask_words", rank,
                     std::to_string(expected_mask_words),
                     std::to_string(out.node_mask_words));
  }
  const auto mask_count =
      checked_size(checked_mul(out.num_tokens, expected_mask_words,
                               "dispatch destination masks"),
                   "dispatch destination masks");
  if (out.destination_node_mask.size() != mask_count) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "outgoing.destination_node_mask.size", rank,
                     std::to_string(mask_count),
                     std::to_string(out.destination_node_mask.size()));
  }

  CountMatrix source_prefix(static_cast<std::size_t>(world) + 1,
                            std::vector<std::uint64_t>(experts, 0));
  for (std::uint32_t source = 0; source < world; ++source) {
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      source_prefix[source + 1][expert] =
          checked_add(source_prefix[source][expert],
                      metadata.source_expert_count[source][expert],
                      "dispatch source prefix");
    }
  }
  const auto &expert_total = source_prefix.back();
  std::vector<std::uint64_t> expert_base(experts, 0);
  for (std::uint32_t destination = 0; destination < world; ++destination) {
    std::uint64_t prefix = 0;
    for (std::uint32_t expert = 0; expert < experts; ++expert) {
      if (config.topology.expert_to_rank[expert] != destination)
        continue;
      expert_base[expert] = prefix;
      prefix =
          checked_add(prefix, expert_total[expert], "dispatch expert base");
    }
  }

  std::vector<std::uint64_t> local_counts(experts, 0);
  std::vector<std::uint64_t> expected_masks(mask_count, 0);
  for (std::uint64_t token = 0; token < out.num_tokens; ++token) {
    const auto base = checked_size(
        checked_mul(token, out.top_k, "dispatch validation route base"),
        "dispatch validation route base");
    std::vector<std::uint32_t> expected_order(out.top_k);
    std::iota(expected_order.begin(), expected_order.end(), 0U);
    for (std::uint32_t slot = 0; slot < out.top_k; ++slot) {
      const auto flat = base + slot;
      const auto expert = out.expert_id[flat];
      if (expert >= experts) {
        metadata_failure(
            NotifyErrorCode::ExpertOutOfRange, phase, "outgoing.expert_id",
            rank, "[0," + std::to_string(experts) + ")", std::to_string(expert),
            "token=" + std::to_string(token) +
                ", slot=" + std::to_string(slot));
      }
      if (out.local_offset[flat] != local_counts[expert]) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "outgoing.local_offset", rank,
                         std::to_string(local_counts[expert]),
                         std::to_string(out.local_offset[flat]),
                         "token=" + std::to_string(token) +
                             ", slot=" + std::to_string(slot));
      }
      ++local_counts[expert];
      const auto destination_rank = config.topology.expert_to_rank[expert];
      const auto destination_node =
          config.topology.rank_to_node[destination_rank];
      const auto destination_local =
          config.topology.rank_to_local_rank[destination_rank];
      if (out.route_dst_rank[flat] != destination_rank ||
          out.route_dst_node[flat] != destination_node ||
          out.route_dst_local_rank[flat] != destination_local) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "outgoing.destination", rank,
                         "topology-derived coordinates", "mismatch",
                         "route=" + std::to_string(flat));
      }
      const auto expected_row = checked_add(
          expert_base[expert],
          checked_add(source_prefix[rank][expert], out.local_offset[flat],
                      "dispatch canonical expert row"),
          "dispatch canonical row");
      if (out.route_dst_row[flat] != expected_row) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "outgoing.route_dst_row", rank,
                         std::to_string(expected_row),
                         std::to_string(out.route_dst_row[flat]),
                         "route=" + std::to_string(flat));
      }
      const auto &reverse = out.reverse_routes[flat];
      if (reverse.source_rank != rank || reverse.local_token_id != token ||
          reverse.topk_slot != slot || reverse.expert_id != expert ||
          reverse.destination_rank != destination_rank ||
          reverse.dst_row != expected_row) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "outgoing.reverse_routes", rank,
                         "canonical route identity", "mismatch",
                         "route=" + std::to_string(flat) +
                             ", actual_source=" +
                             std::to_string(reverse.source_rank) +
                             ", actual_token=" +
                             std::to_string(reverse.local_token_id) +
                             ", actual_slot=" +
                             std::to_string(reverse.topk_slot) +
                             ", actual_expert=" +
                             std::to_string(reverse.expert_id) +
                             ", actual_destination=" +
                             std::to_string(reverse.destination_rank) +
                             ", actual_row=" +
                             std::to_string(reverse.dst_row));
      }
      const auto mask_base = checked_size(
          checked_mul(token, expected_mask_words, "dispatch mask base"),
          "dispatch mask base");
      expected_masks[mask_base + destination_node / 64U] |=
          std::uint64_t{1} << (destination_node % 64U);
    }
    std::stable_sort(
        expected_order.begin(), expected_order.end(), [&](auto lhs, auto rhs) {
          return out.expert_id[base + lhs] < out.expert_id[base + rhs];
        });
    for (std::uint32_t position = 0; position < out.top_k; ++position) {
      if (out.combine_order[base + position] != expected_order[position]) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "outgoing.combine_order", rank,
                         std::to_string(expected_order[position]),
                         std::to_string(out.combine_order[base + position]),
                         "token=" + std::to_string(token) +
                             ", position=" + std::to_string(position));
      }
    }
  }
  if (local_counts != metadata.source_expert_count[rank]) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "source_expert_count.calling_rank_row", rank,
                     "recomputed local counts", "content mismatch");
  }
  if (expected_masks != out.destination_node_mask) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "outgoing.destination_node_mask", rank,
                     "recomputed node mask", "content mismatch");
  }

  const auto &incoming = metadata.incoming;
  if (incoming.rank != rank) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase, "incoming.rank",
                     rank, std::to_string(rank), std::to_string(incoming.rank));
  }
  std::vector<std::uint32_t> owned_experts;
  for (std::uint32_t expert = 0; expert < experts; ++expert) {
    if (config.topology.expert_to_rank[expert] == rank)
      owned_experts.push_back(expert);
  }
  const auto owned = owned_experts.size();
  const auto require_owned = [&](std::size_t actual, const char *field) {
    if (actual != owned) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase, field, rank,
                       std::to_string(owned), std::to_string(actual));
    }
  };
  require_owned(incoming.expert_ids.size(), "incoming.expert_ids.size");
  require_owned(incoming.expert_row_count.size(),
                "incoming.expert_row_count.size");
  require_owned(incoming.mtile_expected_rows.size(),
                "incoming.mtile_expected_rows.size");
  if (incoming.expert_row_prefix.size() != owned + 1) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "incoming.expert_row_prefix.size", rank,
                     std::to_string(owned + 1),
                     std::to_string(incoming.expert_row_prefix.size()));
  }
  std::uint64_t incoming_total = 0;
  for (std::size_t index = 0; index < owned; ++index) {
    const auto expert = owned_experts[index];
    if (incoming.expert_ids[index] != expert) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "incoming.expert_ids", rank, std::to_string(expert),
                       std::to_string(incoming.expert_ids[index]),
                       "index=" + std::to_string(index));
    }
    if (incoming.expert_row_count[index] != expert_total[expert]) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "incoming.expert_row_count", rank,
                       std::to_string(expert_total[expert]),
                       std::to_string(incoming.expert_row_count[index]),
                       "expert=" + std::to_string(expert));
    }
    if (incoming.expert_row_prefix[index] != incoming_total) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "incoming.expert_row_prefix", rank,
                       std::to_string(incoming_total),
                       std::to_string(incoming.expert_row_prefix[index]),
                       "index=" + std::to_string(index));
    }
    incoming_total = checked_add(incoming_total, expert_total[expert],
                                 "validated incoming prefix");
    std::vector<std::uint32_t> expected_tiles;
    for (std::uint64_t first = 0; first < expert_total[expert];) {
      const auto remaining = expert_total[expert] - first;
      expected_tiles.push_back(static_cast<std::uint32_t>(
          std::min<std::uint64_t>(remaining, config.m_tile_rows)));
      if (remaining <= config.m_tile_rows)
        break;
      first = checked_add(first, config.m_tile_rows, "validated M-tile offset");
    }
    if (incoming.mtile_expected_rows[index] != expected_tiles) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "incoming.mtile_expected_rows", rank,
                       "canonical tile rows", "content mismatch",
                       "expert=" + std::to_string(expert));
    }
  }
  if (incoming.expert_row_prefix.back() != incoming_total) {
    metadata_failure(
        NotifyErrorCode::InvalidHandle, phase, "incoming.expert_row_prefix",
        rank, std::to_string(incoming_total),
        std::to_string(incoming.expert_row_prefix.back()), "index=terminal");
  }

  std::vector<OutgoingPacketDescriptor> expected_packets;
  const auto source_local = config.topology.rank_to_local_rank[rank];
  std::vector<std::uint64_t> sequence(config.topology.num_nodes, 0);
  for (std::uint32_t node = 0; node < config.topology.num_nodes; ++node) {
    const auto proxy =
        derived.rank_at[static_cast<std::size_t>(node) * derived.gpus_per_node +
                        source_local];
    for (std::uint64_t token = 0; token < out.num_tokens; ++token) {
      const auto base =
          checked_size(checked_mul(token, out.top_k, "validated packet base"),
                       "validated packet base");
      std::vector<FanoutTarget> fanout;
      for (std::uint32_t slot = 0; slot < out.top_k; ++slot) {
        const auto flat = base + slot;
        if (out.route_dst_node[flat] == node) {
          fanout.push_back(FanoutTarget{slot, out.route_dst_local_rank[flat],
                                        out.expert_id[flat],
                                        out.route_dst_row[flat]});
        }
      }
      if (fanout.empty())
        continue;
      std::stable_sort(fanout.begin(), fanout.end(),
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.expert_id < rhs.expert_id;
                       });
      expected_packets.push_back(
          OutgoingPacketDescriptor{PacketIdentity{rank, node, sequence[node]++},
                                   token, proxy, 0, std::move(fanout)});
    }
  }
  const auto &actual_packets = metadata.transport.outgoing_packets;
  if (actual_packets.size() != expected_packets.size()) {
    metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                     "transport.outgoing_packets.size", rank,
                     std::to_string(expected_packets.size()),
                     std::to_string(actual_packets.size()));
  }
  for (std::size_t packet = 0; packet < expected_packets.size(); ++packet) {
    const auto &expected = expected_packets[packet];
    const auto &actual = actual_packets[packet];
    const auto packet_coord = "packet=" + std::to_string(packet);
    if (actual.identity.source_rank != expected.identity.source_rank ||
        actual.identity.destination_node !=
            expected.identity.destination_node ||
        actual.identity.sequence_on_link !=
            expected.identity.sequence_on_link) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "transport.packet.identity", rank,
                       "canonical packet identity", "mismatch", packet_coord);
    }
    if (actual.local_token_id != expected.local_token_id) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "transport.packet.local_token_id", rank,
                       std::to_string(expected.local_token_id),
                       std::to_string(actual.local_token_id), packet_coord);
    }
    if (actual.proxy_rank != expected.proxy_rank) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "transport.packet.proxy_rank", rank,
                       std::to_string(expected.proxy_rank),
                       std::to_string(actual.proxy_rank), packet_coord);
    }
    if (actual.channel != expected.channel) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "transport.packet.channel", rank,
                       std::to_string(expected.channel),
                       std::to_string(actual.channel), packet_coord);
    }
    if (actual.fanout.size() != expected.fanout.size()) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "transport.packet.fanout.size", rank,
                       std::to_string(expected.fanout.size()),
                       std::to_string(actual.fanout.size()), packet_coord);
    }
    for (std::size_t target = 0; target < expected.fanout.size(); ++target) {
      const auto &expected_target = expected.fanout[target];
      const auto &actual_target = actual.fanout[target];
      const auto target_coord =
          packet_coord + ", target=" + std::to_string(target);
      if (actual_target.topk_slot != expected_target.topk_slot) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "transport.fanout.topk_slot", rank,
                         std::to_string(expected_target.topk_slot),
                         std::to_string(actual_target.topk_slot), target_coord);
      }
      if (actual_target.destination_local_rank !=
          expected_target.destination_local_rank) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "transport.fanout.destination_local_rank", rank,
                         std::to_string(expected_target.destination_local_rank),
                         std::to_string(actual_target.destination_local_rank),
                         target_coord);
      }
      if (actual_target.expert_id != expected_target.expert_id) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "transport.fanout.expert_id", rank,
                         std::to_string(expected_target.expert_id),
                         std::to_string(actual_target.expert_id), target_coord);
      }
      if (actual_target.dst_row != expected_target.dst_row) {
        metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                         "transport.fanout.dst_row", rank,
                         std::to_string(expected_target.dst_row),
                         std::to_string(actual_target.dst_row), target_coord);
      }
    }
  }
  TransportMetadata expected_transport;
  expected_transport.outgoing_packets = expected_packets;
  materialize_dispatch_waves(expected_transport, config, expert_base);
  const auto require_wave_field = [&](const auto &actual,
                                      const auto &expected,
                                      const char *field) {
    if (actual != expected) {
      std::size_t mismatch = 0;
      while (mismatch < actual.size() && mismatch < expected.size() &&
             actual[mismatch] == expected[mismatch])
        ++mismatch;
      const auto actual_value =
          mismatch < actual.size() ? std::to_string(actual[mismatch]) : "end";
      const auto expected_value = mismatch < expected.size()
                                      ? std::to_string(expected[mismatch])
                                      : "end";
      metadata_failure(NotifyErrorCode::InvalidHandle, phase, field, rank,
                       "size=" + std::to_string(expected.size()) +
                           ", index=" + std::to_string(mismatch) +
                           ", value=" + expected_value,
                       "size=" + std::to_string(actual.size()) +
                           ", index=" + std::to_string(mismatch) +
                           ", value=" + actual_value);
    }
  };
  require_wave_field(metadata.transport.dispatch_wave_packet_order,
                     expected_transport.dispatch_wave_packet_order,
                     "transport.dispatch_wave_packet_order");
  require_wave_field(metadata.transport.dispatch_wave_offsets,
                     expected_transport.dispatch_wave_offsets,
                     "transport.dispatch_wave_offsets");
  require_wave_field(metadata.transport.dispatch_wave_destination_node,
                     expected_transport.dispatch_wave_destination_node,
                     "transport.dispatch_wave_destination_node");
  require_wave_field(metadata.transport.dispatch_wave_channel,
                     expected_transport.dispatch_wave_channel,
                     "transport.dispatch_wave_channel");
  require_wave_field(metadata.transport.dispatch_wave_first_primary_tile,
                     expected_transport.dispatch_wave_first_primary_tile,
                     "transport.dispatch_wave_first_primary_tile");
  require_wave_field(metadata.transport.dispatch_wave_last_primary_tile,
                     expected_transport.dispatch_wave_last_primary_tile,
                     "transport.dispatch_wave_last_primary_tile");
  require_wave_field(metadata.transport.dispatch_wave_packet_count,
                     expected_transport.dispatch_wave_packet_count,
                     "transport.dispatch_wave_packet_count");
  require_wave_field(metadata.transport.dispatch_wave_payload_bytes,
                     expected_transport.dispatch_wave_payload_bytes,
                     "transport.dispatch_wave_payload_bytes");
  if (metadata.identity.content_hash != 0) {
    const auto expected_hash = hash_metadata(metadata);
    if (metadata.identity.content_hash != expected_hash) {
      metadata_failure(NotifyErrorCode::InvalidHandle, phase,
                       "identity.content_hash", rank,
                       std::to_string(expected_hash),
                       std::to_string(metadata.identity.content_hash));
    }
  }
}

} // namespace

