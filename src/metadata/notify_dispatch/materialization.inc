DispatchHandle
materialize_rank_metadata(const LocalHistogramHandle &local,
                          const ValidatedCollectiveHandle &expert_counts,
                          const ValidatedCollectiveHandle &packet_summary) {
  if (!local)
    fail(NotifyErrorCode::InvalidHandle,
         "materialize local handle mismatch: expected=valid, actual=empty");
  if (!expert_counts || !packet_summary) {
    fail(
        NotifyErrorCode::CountExchangeFailure,
        "materialize completion mismatch: expected=two valid handles, actual=" +
            std::string(!expert_counts ? "empty expert_counts"
                                       : "empty packet_summary"));
  }
  const auto derived = validate_config(local.impl_->config);
  const auto &config = local.impl_->config;
  const auto world_size = config.topology.world_size;
  const auto num_experts = config.num_experts;
  const auto rank = local.impl_->context.calling_rank();
  validate_local_structure(local);
  if (rank >= world_size) {
    fail(NotifyErrorCode::InvalidRuntimeConfig,
         "materialization calling rank out of range: rank=" +
             std::to_string(rank));
  }
  const auto validate_phase = [&](const ValidatedCollectiveHandle &completion,
                                  ExchangeKind expected_kind,
                                  std::uint64_t expected_width) {
    const auto &actual = completion.impl_->context;
    const auto &expected = local.impl_->context;
    if (completion.impl_->kind != expected_kind) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=collective.kind, expected=" +
                           std::string(kind_name(expected_kind)) +
                           ", actual=" + kind_name(completion.impl_->kind));
    }
    if (actual.calling_rank() != expected.calling_rank()) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=calling_rank, expected=" +
                           std::to_string(expected.calling_rank()) +
                           ", actual=" + std::to_string(actual.calling_rank()));
    }
    if (actual.session().world_size() != expected.session().world_size()) {
      exchange_failure(
          "materialize", expected_kind, expected,
          "field=world_size, expected=" +
              std::to_string(expected.session().world_size()) +
              ", actual=" + std::to_string(actual.session().world_size()));
    }
    if (actual.session().identity() != expected.session().identity()) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=session.identity, expected='" +
                           expected.session().identity() + "', actual='" +
                           actual.session().identity() + "'");
    }
    if (actual.session().generation() != expected.session().generation()) {
      exchange_failure(
          "materialize", expected_kind, expected,
          "field=session.generation, expected=" +
              std::to_string(expected.session().generation()) +
              ", actual=" + std::to_string(actual.session().generation()));
    }
    if (!actual.session().same_instance(expected.session())) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=session.provenance, expected=same session "
                       "instance, actual=different session instance");
    }
    if (!actual.execution().same_instance(expected.execution())) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=execution.provenance, expected='" +
                           expected.execution().identity() + "', actual='" +
                           actual.execution().identity() + "'");
    }
    if (completion.impl_->local_provenance != local.impl_->provenance) {
      exchange_failure("materialize", expected_kind, expected,
                       "field=local.provenance, expected=same "
                       "local handle, actual=different local handle");
    }
    if (completion.impl_->matrix.rows() != world_size) {
      exchange_failure(
          "materialize", expected_kind, expected,
          "field=matrix.rows, expected=" + std::to_string(world_size) +
              ", actual=" + std::to_string(completion.impl_->matrix.rows()));
    }
    if (completion.impl_->matrix.row_width() != expected_width) {
      exchange_failure(
          "materialize", expected_kind, expected,
          "field=matrix.row_width, expected=" + std::to_string(expected_width) +
              ", actual=" +
              std::to_string(completion.impl_->matrix.row_width()));
    }
    (void)checked_mul(world_size, expected_width,
                      "materialization collective matrix elements");
  };
  validate_phase(expert_counts, ExchangeKind::ExpertCounts, num_experts);
  validate_phase(packet_summary, ExchangeKind::NotifyPacketSummary,
                 config.topology.num_nodes);
  CountMatrix counts;
  CountMatrix summaries;
  try {
    counts = expert_counts.impl_->matrix.copy_rows();
    summaries = packet_summary.impl_->matrix.copy_rows();
  } catch (const NotifyDispatchError &error) {
    exchange_failure("materialize", ExchangeKind::ExpertCounts,
                     local.impl_->context,
                     "typed matrix read failed: original_error='" +
                         exception_text(error) + "'");
  } catch (const std::exception &error) {
    exchange_failure("materialize", ExchangeKind::ExpertCounts,
                     local.impl_->context,
                     "typed matrix read failed: original_error='" +
                         exception_text(error) + "'");
  } catch (...) {
    exchange_failure("materialize", ExchangeKind::ExpertCounts,
                     local.impl_->context,
                     "typed matrix read failed: original_error='unknown "
                     "non-standard exception'");
  }
  if (counts.at(rank) != local.impl_->local_expert_count.values()) {
    metadata_failure(NotifyErrorCode::CountExchangeFailure, "materialize",
                     "expert_counts.calling_rank_row", rank,
                     "local histogram count row", "content mismatch");
  }
  if (summaries.at(rank) !=
      local.impl_->packet_count_by_destination_node.values()) {
    metadata_failure(NotifyErrorCode::CountExchangeFailure, "materialize",
                     "packet_summary.calling_rank_row", rank,
                     "local histogram summary row", "content mismatch");
  }

  CountMatrix source_prefix(static_cast<std::size_t>(world_size) + 1,
                            std::vector<std::uint64_t>(num_experts, 0));
  std::vector<std::uint64_t> expert_total(num_experts, 0);
  for (std::uint32_t source = 0; source < world_size; ++source) {
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
      source_prefix[source + 1][expert] = checked_add(
          source_prefix[source][expert], counts[source][expert],
          "source prefix for source_rank=" + std::to_string(source) +
              ", expert=" + std::to_string(expert));
    }
  }
  expert_total = source_prefix.back();

  std::vector<std::uint64_t> expert_base(num_experts, 0);
  std::vector<std::uint64_t> rank_total_rows(world_size, 0);
  for (std::uint32_t destination_rank = 0; destination_rank < world_size;
       ++destination_rank) {
    std::uint64_t prefix = 0;
    for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
      if (config.topology.expert_to_rank[expert] != destination_rank)
        continue;
      expert_base[expert] = prefix;
      if (!config.capacity.expert_rows.empty() &&
          expert_total[expert] > config.capacity.expert_rows[expert]) {
        fail(NotifyErrorCode::ExpertCapacityExceeded,
             "expert capacity exceeded: expert=" + std::to_string(expert) +
                 ", owner_rank=" + std::to_string(destination_rank) +
                 ", actual_rows=" + std::to_string(expert_total[expert]) +
                 ", capacity=" +
                 std::to_string(config.capacity.expert_rows[expert]));
      }
      prefix = checked_add(prefix, expert_total[expert],
                           "destination expert prefix for rank=" +
                               std::to_string(destination_rank));
    }
    rank_total_rows[destination_rank] = prefix;
    if (!config.capacity.workspace_rows_per_rank.empty() &&
        prefix > config.capacity.workspace_rows_per_rank[destination_rank]) {
      fail(NotifyErrorCode::WorkspaceCapacityExceeded,
           "workspace row capacity exceeded: destination_rank=" +
               std::to_string(destination_rank) +
               ", actual_rows=" + std::to_string(prefix) + ", capacity=" +
               std::to_string(
                   config.capacity.workspace_rows_per_rank[destination_rank]));
    }
  }

  auto metadata = std::make_shared<DispatchMetadata>();
  metadata->config = config;
  metadata->calling_rank = rank;
  metadata->communicator_id = local.impl_->context.session().identity();
  metadata->identity.generation = local.impl_->context.session().generation();
  metadata->source_expert_count = counts;
  metadata->outgoing = local.impl_->outgoing;
  auto &outgoing = metadata->outgoing;
  outgoing.reverse_routes.resize(outgoing.expert_id.size());
  for (std::uint64_t token = 0; token < outgoing.num_tokens; ++token) {
    const auto base = checked_size(
        checked_mul(token, outgoing.top_k, "route base"), "route base");
    for (std::uint32_t slot = 0; slot < outgoing.top_k; ++slot) {
      const auto flat = base + slot;
      const auto expert = outgoing.expert_id[flat];
      const auto destination_rank = outgoing.route_dst_rank[flat];
      const auto within_expert =
          checked_add(source_prefix[rank][expert], outgoing.local_offset[flat],
                      "canonical row within expert for " +
                          route_context(rank, token, slot));
      const auto dst_row = checked_add(expert_base[expert], within_expert,
                                       "canonical destination row for " +
                                           route_context(rank, token, slot));
      if (dst_row >= rank_total_rows[destination_rank]) {
        fail(NotifyErrorCode::CountExchangeFailure,
             "canonical row exceeds destination layout for " +
                 route_context(rank, token, slot));
      }
      outgoing.route_dst_row[flat] = dst_row;
      outgoing.reverse_routes[flat] = ReverseRouteMetadata{
          rank, token, slot, expert, destination_rank, dst_row};
    }
  }

  auto &incoming = metadata->incoming;
  incoming.rank = rank;
  incoming.expert_row_prefix.push_back(0);
  for (std::uint32_t expert = 0; expert < num_experts; ++expert) {
    if (config.topology.expert_to_rank[expert] != rank)
      continue;
    incoming.expert_ids.push_back(expert);
    incoming.expert_row_count.push_back(expert_total[expert]);
    incoming.expert_row_prefix.push_back(checked_add(
        incoming.expert_row_prefix.back(), expert_total[expert],
        "local incoming expert prefix for rank=" + std::to_string(rank)));
    std::vector<std::uint32_t> tiles;
    for (std::uint64_t first = 0; first < expert_total[expert];) {
      const auto remaining = expert_total[expert] - first;
      tiles.push_back(static_cast<std::uint32_t>(
          std::min<std::uint64_t>(remaining, config.m_tile_rows)));
      if (remaining <= config.m_tile_rows)
        break;
      first = checked_add(first, config.m_tile_rows,
                          "M-tile offset for expert=" + std::to_string(expert));
    }
    incoming.mtile_expected_rows.push_back(std::move(tiles));
    for (std::uint32_t source = 0; source < world_size; ++source) {
      for (std::uint64_t offset = 0; offset < counts[source][expert];
           ++offset) {
        const auto dst_row =
            checked_add(expert_base[expert],
                        checked_add(source_prefix[source][expert], offset,
                                    "incoming source offset"),
                        "incoming canonical row");
        incoming.incoming_rows.push_back(
            IncomingRowMetadata{source, expert, offset, dst_row});
      }
    }
  }
  std::sort(incoming.incoming_rows.begin(), incoming.incoming_rows.end(),
            [](const auto &lhs, const auto &rhs) {
              return lhs.dst_row < rhs.dst_row;
            });

  const auto source_node = config.topology.rank_to_node[rank];
  const auto source_local_rank = config.topology.rank_to_local_rank[rank];
  std::vector<std::uint64_t> sequence_by_node(config.topology.num_nodes, 0);
  for (std::uint32_t destination_node = 0;
       destination_node < config.topology.num_nodes; ++destination_node) {
    const auto proxy_rank =
        derived.rank_at[static_cast<std::size_t>(destination_node) *
                            derived.gpus_per_node +
                        source_local_rank];
    for (std::uint64_t token = 0; token < outgoing.num_tokens; ++token) {
      const auto route_base =
          checked_size(checked_mul(token, outgoing.top_k, "packet route base"),
                       "packet route base");
      std::vector<FanoutTarget> fanout;
      for (std::uint32_t slot = 0; slot < outgoing.top_k; ++slot) {
        const auto flat = route_base + slot;
        if (outgoing.route_dst_node[flat] != destination_node)
          continue;
        fanout.push_back(FanoutTarget{slot, outgoing.route_dst_local_rank[flat],
                                      outgoing.expert_id[flat],
                                      outgoing.route_dst_row[flat]});
      }
      if (fanout.empty())
        continue;
      std::stable_sort(fanout.begin(), fanout.end(),
                       [](const auto &lhs, const auto &rhs) {
                         return lhs.expert_id < rhs.expert_id;
                       });
      const auto sequence = sequence_by_node[destination_node]++;
      metadata->transport.outgoing_packets.push_back(OutgoingPacketDescriptor{
          PacketIdentity{rank, destination_node, sequence}, token, proxy_rank,
          0, std::move(fanout)});
    }
    const auto expected = summaries[rank][destination_node];
    if (sequence_by_node[destination_node] != expected) {
      fail(NotifyErrorCode::CountExchangeFailure,
           "packet summary disagrees with materialized descriptors: "
           "source_rank=" +
               std::to_string(rank) + ", destination_node=" +
               std::to_string(destination_node) + ", descriptors=" +
               std::to_string(sequence_by_node[destination_node]) +
               ", summary=" + std::to_string(expected));
    }
    metadata->transport.outgoing_signals.push_back(
        SignalCardinality{source_node, destination_node, rank, proxy_rank,
                          source_local_rank, 0, expected, expected});
  }

  materialize_dispatch_waves(metadata->transport, config, expert_base);

  const auto destination_node = config.topology.rank_to_node[rank];
  const auto rail_local_rank = config.topology.rank_to_local_rank[rank];
  for (std::uint32_t remote_source_node = 0;
       remote_source_node < config.topology.num_nodes; ++remote_source_node) {
    const auto source_rank =
        derived.rank_at[static_cast<std::size_t>(remote_source_node) *
                            derived.gpus_per_node +
                        rail_local_rank];
    const auto expected = summaries[source_rank][destination_node];
    metadata->transport.incoming_proxy_signals.push_back(
        SignalCardinality{remote_source_node, destination_node, source_rank,
                          rank, rail_local_rank, 0, expected, expected});
  }

  validate_dispatch_metadata_object(*metadata, "materialize-output");
  metadata->identity.content_hash = hash_metadata(*metadata);
  return DispatchHandleFactory::make(std::move(metadata));
}

std::vector<DispatchHandle> build_notify_dispatch_for_testing(
    const RuntimeConfig &config,
    const std::vector<RankRoutingInput> &routes_by_rank,
    const std::string &communicator_id, std::uint64_t generation) {
  validate_config(config);
  if (routes_by_rank.size() != config.topology.world_size) {
    fail(NotifyErrorCode::InvalidRouteShape,
         "test wrapper requires one rank-local input per rank: actual=" +
             std::to_string(routes_by_rank.size()) +
             ", expected=" + std::to_string(config.topology.world_size));
  }
  const auto session = make_collective_session(
      communicator_id, config.topology.world_size, generation);
  std::vector<LocalHistogramHandle> locals;
  locals.reserve(routes_by_rank.size());
  for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
    const auto execution =
        make_execution_token("host-fake-rank-" + std::to_string(rank));
    const auto context = make_rank_collective_context(session, rank, execution);
    locals.push_back(
        build_rank_local_histogram(config, context, routes_by_rank[rank]));
  }

  HostFakeCountExchange backend(config.topology.world_size, communicator_id);
  std::vector<CollectiveOperation> count_operations;
  count_operations.reserve(config.topology.world_size);
  for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
    count_operations.push_back(start_count_exchange(locals[rank], backend));
  }
  std::vector<ValidatedCollectiveHandle> count_results;
  count_results.reserve(config.topology.world_size);
  for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
    count_results.push_back(count_operations[rank].wait());
  }

  std::vector<CollectiveOperation> summary_operations;
  summary_operations.reserve(config.topology.world_size);
  for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
    summary_operations.push_back(
        start_notify_summary_exchange(locals[rank], backend));
  }
  std::vector<DispatchHandle> handles;
  handles.reserve(config.topology.world_size);
  for (std::uint32_t rank = 0; rank < config.topology.world_size; ++rank) {
    auto summary = summary_operations[rank].wait();
    handles.push_back(
        materialize_rank_metadata(locals[rank], count_results[rank], summary));
  }
  return handles;
}

