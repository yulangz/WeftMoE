#pragma once

#ifndef MEGAMOE_INTERNAL_TESTING_API
#define MEGAMOE_INTERNAL_TESTING_API 1
#endif

#include <cstddef>
#include <cstdint>
#include <vector>

#include "megamoe/api/notify_dispatch.hpp"

namespace megamoe::testing {

// Internal sanitizer-only/test-only hook. This header is intentionally outside
// the public include tree.
class NotifyCorruptionHook {
public:
  static LocalHistogramHandle with_expert_id(const LocalHistogramHandle &source,
                                             std::size_t route_index,
                                             std::uint32_t expert_id);
  static LocalHistogramHandle
  with_truncated_expert_ids(const LocalHistogramHandle &source);
  static LocalHistogramHandle
  with_local_offset(const LocalHistogramHandle &source, std::size_t route_index,
                    std::uint64_t value);
  static LocalHistogramHandle
  with_combine_order(const LocalHistogramHandle &source,
                     std::size_t route_index, std::uint32_t value);
  static LocalHistogramHandle with_node_mask(const LocalHistogramHandle &source,
                                             std::size_t word_index,
                                             std::uint64_t value);
  static LocalHistogramHandle
  with_local_expert_count(const LocalHistogramHandle &source,
                          std::size_t expert_index, std::uint64_t value);
  static ValidatedCollectiveHandle
  with_collective_matrix(const ValidatedCollectiveHandle &source,
                         const CountMatrix &matrix);

  static DispatchHandle with_expert_row_count(const DispatchHandle &source,
                                              std::size_t expert_index,
                                              std::uint64_t value);
  static DispatchHandle with_expert_row_prefix(const DispatchHandle &source,
                                               std::size_t prefix_index,
                                               std::uint64_t value);
  static DispatchHandle
  with_packet_identity_destination_node(const DispatchHandle &source,
                                        std::size_t packet_index,
                                        std::uint32_t value);
  static DispatchHandle with_packet_proxy(const DispatchHandle &source,
                                          std::size_t packet_index,
                                          std::uint32_t value);
  static DispatchHandle with_packet_sequence(const DispatchHandle &source,
                                             std::size_t packet_index,
                                             std::uint64_t value);
  static DispatchHandle with_fanout_expert(const DispatchHandle &source,
                                           std::size_t packet_index,
                                           std::size_t target_index,
                                           std::uint32_t value);
  static DispatchHandle with_fanout_local_rank(const DispatchHandle &source,
                                               std::size_t packet_index,
                                               std::size_t target_index,
                                               std::uint32_t value);
  static DispatchHandle with_fanout_row(const DispatchHandle &source,
                                        std::size_t packet_index,
                                        std::size_t target_index,
                                        std::uint64_t value);
  static DispatchHandle with_fanout_slot(const DispatchHandle &source,
                                         std::size_t packet_index,
                                         std::size_t target_index,
                                         std::uint32_t value);
  static DispatchHandle with_mtile_row(const DispatchHandle &source,
                                       std::size_t expert_index,
                                       std::size_t tile_index,
                                       std::uint32_t value);
};

} // namespace megamoe::testing
