#pragma once

// Included in forward_unified_transport after the common copy pipeline.
namespace network_ring_transport {

namespace ring = static_v4::ring;
using Endpoint = static_v4::netring::Endpoint;

struct ChannelPlan {
  unsigned source;
  std::uint64_t begin;
  std::uint64_t count;
};

template <bool Returning, bool Receive>
__device__ __forceinline__ ChannelPlan channel_plan(const ForwardKernelArgs &a,
                                                    unsigned channel) {
  const auto &network = a.static_network.rings;
  const auto peer = network.peer(a.node, channel);
  constexpr bool PeerSource = Returning != Receive;
  const auto source =
      PeerSource ? peer * a.local_world_size + a.rank % a.local_world_size
                 : a.rank;
  const auto node = PeerSource ? a.node : peer;
  const auto *offsets =
      ring::source_metadata(a, source,
                            Returning ? a.ring.returning.channel_offsets
                                      : a.ring.dispatch.channel_offsets);
  const auto index = node * network.lanes + channel % network.lanes;
  return {source, offsets[index], offsets[index + 1U] - offsets[index]};
}

__device__ __forceinline__ static_forward::ComputeSlot &
c_slot(const ForwardKernelArgs &a, unsigned destination,
       const static_forward::RouteLocation &location) {
  return ring::peer(a, destination,
                    a.ring.slots)[location.row / forward_tuning::kTileM];
}

// Each worker multiplexes its fixed channel residue without waiting on a
// software dependency. Once ready, a packet's finite copy/Put cannot block RX.
template <bool Returning, class Storage>
__device__ __noinline__ void transmit(const ForwardKernelArgs &a,
                                      Storage &storage, unsigned worker) {
  if ((threadIdx.x & 31U) != 0U)
    return;
  const auto &r = a.ring;
  const auto workers = a.static_network.ring_workers;
  Endpoint endpoint(a, (Returning ? workers : 0U) + worker);
  RowPrefetchPipeline<Storage> prefetch(storage, Returning ? 1U : 0U);
  const auto payload_offset =
      static_forward::PacketLayout::planned(0U, a.hidden).payload;
  const auto row_bytes =
      static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16));
  const auto owned = a.dispatch.sizes.owned_expert_count;
  std::uint64_t sent_rows = 0U, sent_packets = 0U;
  bool done;
  do {
    done = true;
    for (unsigned channel = worker; channel < endpoint.layout().channels();
         channel += workers) {
      auto &state = endpoint.state(Returning, channel);
      const auto plan = channel_plan<Returning, false>(a, channel);
      const auto terminal = state.tx_base + plan.count;
      if (state.tx_cursor == terminal) {
        done &= endpoint.drained(Returning, false, channel, terminal);
        continue;
      }
      done = false;
      const auto sequence = state.tx_cursor;
      if (!endpoint.can_send(Returning, channel, sequence))
        continue;
      const auto packet = plan.begin + sequence - state.tx_base;
      std::uint64_t window;
      unsigned rows, expected;
      const std::uint64_t *routes = nullptr;
      const static_forward::NodeToken *tokens = nullptr;
      const std::int32_t *experts = nullptr;
      const static_forward::RouteLocation *locations = nullptr;
      if constexpr (Returning) {
        const auto desc =
            ring::source_metadata(a, plan.source, r.returning.packets)[packet];
        window = desc.window;
        rows = expected = desc.rows;
        routes = ring::source_metadata(a, plan.source, r.source.route_order) +
                 desc.route_begin;
        experts =
            ring::source_metadata(a, plan.source, r.source.selected_experts);
        locations = ring::source_metadata(a, plan.source, r.locations);
        bool ready = true;
        const static_forward::ComputeSlot *previous = nullptr;
        for (unsigned row = 0U; row < rows; ++row) {
          const auto route = routes[row];
          const auto &location = locations[route];
          const auto *slot = &c_slot(
              a, static_cast<unsigned>(experts[route]) / owned, location);
          if (slot == previous)
            continue;
          if (ring::acquire<::cuda::thread_scope_system>(slot->generation) !=
                  location.generation ||
              ring::acquire<::cuda::thread_scope_system>(slot->fc2_done) <
                  slot->fc2_target) {
            ready = false;
            break;
          }
          previous = slot;
        }
        if (!ready)
          continue;
      } else {
        const auto desc = r.dispatch.packets[packet];
        window = desc.window;
        rows = desc.tokens;
        expected = desc.fanouts;
        tokens = r.source.node_tokens + desc.token_begin;
        const auto source_sequence =
            r.state->source_base + window / r.source.shape.groups();
        if (ring::acquire(
                r.contributions[source_sequence % r.contribution_depth]
                    .generation) != source_sequence / r.contribution_depth + 1U)
          continue;
      }
      const auto offset = endpoint.payload(Returning, false, channel, sequence);
      auto *payload = peer_at<char>(a.gin_window_base, offset + payload_offset);
      if constexpr (Returning)
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnPackStart, GinReturn,
                                   channel, sequence, rows);
      else
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchPackStart,
                                   GinDispatch, channel, sequence, rows);
      asm volatile("fence.proxy.async.global;" ::: "memory");
      for (unsigned row = 0U; row < rows; ++row) {
        const transport::BFloat16 *source;
        if constexpr (Returning) {
          const auto route = routes[row];
          const auto destination =
              static_cast<unsigned>(experts[route]) / owned;
          source =
              peer_at<const transport::BFloat16>(a.peer_bases[destination],
                                                 a.forward_output_rows_offset) +
              std::uint64_t(locations[route].row) * a.hidden;
        } else {
          source = peer_at<const transport::BFloat16>(a.local_base,
                                                      a.source_input_offset) +
                   tokens[row].token * a.hidden;
        }
        prefetch.copy(source, payload + std::uint64_t(row) * row_bytes,
                      row_bytes);
      }
      prefetch.drain();
      finish_copies<false>();
      if constexpr (Returning)
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnPackEnd, GinReturn,
                                   channel, sequence, rows);
      else
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, DispatchPackEnd,
                                   GinDispatch, channel, sequence, rows);
      if constexpr (Returning) {
        // A packet is one admitted window: a physical C slot cannot alias
        // another block in it. Collapse consecutive rows of that block into
        // one retirement, after every payload copy has actually completed.
        static_forward::ComputeSlot *previous = nullptr;
        unsigned count = 0U;
        for (unsigned row = 0U; row < rows; ++row) {
          const auto route = routes[row];
          const auto destination =
              static_cast<unsigned>(experts[route]) / owned;
          auto *slot = &c_slot(a, destination, locations[route]);
          if (previous != slot) {
            if (count)
              static_v4::reduce_add_release_system(
                  reinterpret_cast<unsigned long long *>(&previous->returned),
                  count);
            previous = slot;
            count = 0U;
          }
          ++count;
        }
        static_v4::reduce_add_release_system(
            reinterpret_cast<unsigned long long *>(&previous->returned), count);
      }
      auto &control = endpoint.slot(Returning, channel, sequence);
      control.retire_target += expected;
      *peer_at<static_forward::PacketHeader>(a.gin_window_base, offset) = {
          sequence, window, control.retire_target, rows,
          Returning ? 0U : expected};
      endpoint.send(Returning, channel, sequence,
                    payload_offset + std::uint64_t(rows) * row_bytes);
      if constexpr (Returning)
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, ReturnGinPutIssue,
                                   GinReturn, channel, sequence, rows);
      else
        WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, GinPutIssue, GinDispatch,
                                   channel, sequence, rows);
      ++state.tx_cursor;
      sent_rows += rows;
      ++sent_packets;
    }
  } while (!done);
  auto *status = peer_at<TransportDeviceStatus>(a.local_base, a.status_offset);
  atomicAdd(Returning ? &status->gin_return_put_count
                      : &status->gin_dispatch_put_count,
            sent_packets);
  if constexpr (Returning)
    atomicAdd(&status->gin_return_payload_bytes, sent_rows * row_bytes);
  else
    atomicAdd(&status->deduplicated_cross_node_payload_bytes,
              sent_rows * row_bytes);
}

__device__ __noinline__ void receive_progress(const ForwardKernelArgs &a,
                                              unsigned worker) {
  if ((threadIdx.x & 31U) != 0U)
    return;
  const auto workers = a.static_network.ring_workers;
  Endpoint endpoint(a, 2U * workers + worker);
  bool done;
  do {
    done = true;
    for (unsigned channel = worker; channel < endpoint.layout().channels();
         channel += workers) {
      for (unsigned direction = 0U; direction < 2U; ++direction) {
        const bool returning = direction != 0U;
        auto &state = endpoint.state(returning, channel);
        const auto plan = returning ? channel_plan<true, true>(a, channel)
                                    : channel_plan<false, true>(a, channel);
        const auto terminal = state.rx_base + plan.count;
        if (state.rx_cursor < terminal &&
            endpoint.import_ready(returning, channel, state.rx_cursor)) {
          if (returning)
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, SourceSignalObserved,
                                       GinReturn, channel, state.rx_cursor, 0U);
          else
            WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, RemoteSignalObserved,
                                       GinDispatch, channel, state.rx_cursor,
                                       0U);
          ++state.rx_cursor;
        }
        endpoint.acknowledge(returning, channel);
        done &= state.rx_cursor == terminal &&
                endpoint.drained(returning, true, channel, terminal);
      }
    }
  } while (!done);
}

template <class Storage>
__device__ __noinline__ void receive_return(const ForwardKernelArgs &a,
                                            Storage &storage, unsigned worker) {
  if ((threadIdx.x & 31U) != 0U)
    return;
  const auto &r = a.ring;
  const auto workers = a.static_network.ring_workers;
  // This role only computes addresses; all GIN calls remain with
  // receive_progress.
  Endpoint endpoint(a, 0U);
  RowPrefetchPipeline<Storage> prefetch(storage, 1U);
  const auto payload_offset =
      static_forward::PacketLayout::planned(0U, a.hidden).payload;
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.contribution_offset);
  const auto row_bytes =
      static_cast<unsigned>(a.hidden * sizeof(transport::BFloat16));
  bool done;
  do {
    done = true;
    for (unsigned channel = worker; channel < endpoint.layout().channels();
         channel += workers) {
      auto &state = endpoint.state(true, channel);
      const auto plan = channel_plan<true, true>(a, channel);
      if (state.consume_cursor == state.rx_base + plan.count)
        continue;
      done = false;
      const auto sequence = state.consume_cursor;
      auto &control = endpoint.slot(true, channel, sequence);
      if (ring::acquire(control.visible_generation) !=
          sequence / endpoint.layout().depth + 1U)
        continue;
      const auto desc =
          r.returning.packets[plan.begin + sequence - state.rx_base];
      const auto &address = ring::window_source(a, desc.window, a.rank);
      const auto route_begin =
          r.source.shape.token_begin(desc.window) * r.source.shape.top_k;
      const auto *routes = r.source.route_order + desc.route_begin;
      const auto *payload = peer_at<const char>(
          a.gin_window_base,
          endpoint.payload(true, true, channel, sequence) + payload_offset);
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, SourceRelayStart,
                                 SourceRelay, channel, sequence, desc.rows);
      asm volatile("fence.proxy.async.global;" ::: "memory");
      for (unsigned row = 0U; row < desc.rows; ++row)
        prefetch.copy(payload + std::uint64_t(row) * row_bytes,
                      output + (address.row_base + routes[row] - route_begin) *
                                   a.hidden,
                      row_bytes);
      prefetch.drain();
      finish_copies<false>();
      for (unsigned row = 0U; row < desc.rows; ++row)
        ring::publish<::cuda::thread_scope_system>(
            r.contribution_ready[address.row_base + routes[row] - route_begin],
            address.generation);
      ring::arrive(control.consumed_total, desc.rows);
      WEFTMOE_TIMELINE_RECORD_IF(a.timeline, true, SourceRelayEnd, SourceRelay,
                                 channel, sequence, desc.rows);
      ++state.consume_cursor;
    }
  } while (!done);
}

// Packet residues are immutable. Non-consumers skip without touching RX;
// consumers traverse source/lane/sequence order and retire the whole local
// fanout.
template <class Storage>
__device__ __noinline__ void
pull_window(const ForwardKernelArgs &a, Storage &storage, unsigned worker,
            unsigned workers, std::uint64_t window, unsigned &peer_rows) {
  const auto &r = a.ring;
  const auto &shape = r.source.shape;
  const auto &network = a.static_network;
  RowPrefetchPipeline<Storage> prefetch(storage, 0U);
  auto *output =
      peer_at<transport::BFloat16>(a.local_base, a.canonical_rows_offset);
  auto *gates = peer_at<float>(a.local_base, a.canonical_gates_offset);
  const auto owned = a.dispatch.sizes.owned_expert_count;
  const auto rail = a.rank % a.local_world_size;
  const auto payload_offset =
      static_forward::PacketLayout::planned(0U, a.hidden).payload;
  for (auto source = shape.source_begin(window);
       source < shape.source_end(window); ++source) {
    if (source / a.local_world_size == a.node)
      continue;
    const auto source_windows =
        static_forward::ceil_div(r.source_epochs[source].tokens, shape.tokens);
    const auto local_window = window / shape.groups();
    if (local_window >= source_windows)
      continue;
    const auto *prefixes =
        ring::source_metadata(a, source, r.source.packet_prefix);
    const auto *offsets =
        ring::source_metadata(a, source, r.dispatch.channel_offsets);
    const auto *packets = ring::source_metadata(a, source, r.dispatch.packets);
    const auto *consumers =
        ring::source_metadata(a, source, r.dispatch.consumer_rows);
    const auto *tokens = ring::source_metadata(a, source, r.source.node_tokens);
    const auto *experts =
        ring::source_metadata(a, source, r.source.selected_experts);
    const auto *source_gates = ring::source_metadata(a, source, r.source_gates);
    const auto *locations = ring::source_metadata(a, source, r.locations);
    for (unsigned lane = 0U; lane < network.rings.lanes; ++lane) {
      const auto directory = a.node * network.rings.lanes + lane;
      const auto channel =
          network.rings.channel(a.node, source / a.local_world_size, lane);
      const auto *state =
          static_cast<const static_forward::NetworkChannel *>(ncclGetLsaPointer(
              a.gin_window,
              network.ring_region + network.rings.state(false, channel),
              source % a.local_world_size));
      const auto *prefix = prefixes + directory * (source_windows + 1U);
      const auto begin = offsets[directory] + prefix[local_window];
      const auto end = offsets[directory] + prefix[local_window + 1U];
      const auto first =
          begin + (worker + workers - (begin + source) % workers) % workers;
      for (auto packet = first; packet < end; packet += workers) {
        const auto count = consumers[packet * a.local_world_size + rail];
        if (!count)
          continue;
        const auto sequence = state->rx_base + packet - offsets[directory];
        auto &control =
            *static_cast<static_forward::NetworkSlot *>(ncclGetLsaPointer(
                a.gin_window,
                network.ring_region +
                    network.rings.control(false, channel, sequence),
                source % a.local_world_size));
        while (ring::acquire<::cuda::thread_scope_system>(
                   control.visible_generation) !=
               sequence / network.rings.depth + 1U) {
        }
        const auto desc = packets[packet];
        const auto *payload =
            static_cast<const transport::BFloat16 *>(ncclGetLsaPointer(
                a.gin_window,
                network.ring_region +
                    network.rings.payload(false, true, channel, sequence) +
                    payload_offset,
                source % a.local_world_size));
        for (unsigned pass = 0U; pass < 2U; ++pass) {
          for (unsigned token = 0U; token < desc.tokens; ++token) {
            const auto route_begin =
                tokens[desc.token_begin + token].token * shape.top_k;
            for (unsigned k = 0U; k < shape.top_k; ++k) {
              const auto route = route_begin + k;
              if (static_cast<unsigned>(experts[route]) / owned != a.rank)
                continue;
              const auto location = locations[route];
              if (!pass) {
                while (!ring::admitted(a, location.block)) {
                }
                gates[location.row] = source_gates[route];
                asm volatile("fence.proxy.async.global;" ::: "memory");
                prefetch.copy(payload + std::uint64_t(token) * a.hidden,
                              output + std::uint64_t(location.row) * a.hidden,
                              static_cast<unsigned>(
                                  a.hidden * sizeof(transport::BFloat16)));
              } else {
                const auto valid = r.blocks[location.block].tile.rows;
                const auto m = location.row % forward_tuning::kTileM;
                const auto credit =
                    m + 1U == valid ? forward_tuning::kTileM - valid + 1U : 1U;
                static_v4::reduce_add_release_device(
                    reinterpret_cast<unsigned long long *>(
                        &ring::slot(a, location.block).x_full),
                    credit);
              }
            }
          }
          if (!pass) {
            prefetch.drain();
            finish_copies<false>();
          }
        }
        ring::arrive<::cuda::thread_scope_system>(control.consumed_total,
                                                  count);
        peer_rows += source % a.local_world_size != rail ? count : 0U;
      }
    }
  }
}

} // namespace network_ring_transport
