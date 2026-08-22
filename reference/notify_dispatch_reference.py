"""Independent, clarity-first reference for deterministic Notify Dispatch.

This module intentionally does not import or bind the C++ implementation. It is
the oracle intended for later CUDA result comparisons.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence


class ReferenceNotifyError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceTopology:
    rank_to_node: Sequence[int]
    rank_to_local_rank: Sequence[int]
    expert_to_rank: Sequence[int]
    num_nodes: int


@dataclass(frozen=True)
class ReferenceResult:
    source_expert_count: List[List[int]]
    local_offset: List[List[int]]
    route_dst_row: List[List[int]]
    route_dst_rank: List[List[int]]
    route_dst_node: List[List[int]]
    route_dst_local_rank: List[List[int]]
    destination_node_mask: List[List[int]]
    expert_ids: List[List[int]]
    expert_row_count: List[List[int]]
    expert_row_prefix: List[List[int]]
    combine_order: List[List[int]]
    reverse_routes: List[List[tuple]]
    mtile_expected_rows: List[List[List[int]]]


def build_reference(
    selected_experts_by_rank: Sequence[Sequence[Sequence[int]]],
    topology: ReferenceTopology,
    num_experts: int,
    top_k: int,
    m_tile_rows: int,
    expert_capacity: Optional[Sequence[int]] = None,
    arrival_order: Optional[Sequence[int]] = None,
) -> ReferenceResult:
    """Return canonical metadata using only serial scans and source-rank prefix."""
    world_size = len(selected_experts_by_rank)
    if not (world_size and num_experts and top_k and m_tile_rows):
        raise ReferenceNotifyError("runtime dimensions must be positive")
    if len(topology.rank_to_node) != world_size or len(topology.rank_to_local_rank) != world_size:
        raise ReferenceNotifyError("rank topology shape mismatch")
    if len(topology.expert_to_rank) != num_experts:
        raise ReferenceNotifyError("expert ownership shape mismatch")
    if expert_capacity is not None and len(expert_capacity) != num_experts:
        raise ReferenceNotifyError("expert capacity shape mismatch")
    if topology.num_nodes <= 0:
        raise ReferenceNotifyError("num_nodes must be positive")
    coordinates = set()
    for rank, (node, local_rank) in enumerate(zip(topology.rank_to_node,
                                                  topology.rank_to_local_rank)):
        if node < 0 or node >= topology.num_nodes or local_rank < 0:
            raise ReferenceNotifyError(f"invalid topology coordinate for rank {rank}")
        if (node, local_rank) in coordinates:
            raise ReferenceNotifyError("duplicate (node, local_rank) coordinate")
        coordinates.add((node, local_rank))
    if any(owner < 0 or owner >= world_size for owner in topology.expert_to_rank):
        raise ReferenceNotifyError("expert owner is out of range")

    local_offset: List[List[int]] = []
    counts = [[0 for _ in range(num_experts)] for _ in range(world_size)]
    combine_order: List[List[int]] = []
    node_mask_words = (topology.num_nodes + 63) // 64
    destination_node_mask: List[List[int]] = []
    for rank, tokens in enumerate(selected_experts_by_rank):
        rank_offsets: List[int] = []
        rank_order: List[int] = []
        rank_node_masks: List[int] = []
        for token_id, experts in enumerate(tokens):
            if len(experts) != top_k:
                raise ReferenceNotifyError(f"top-k shape mismatch at rank={rank}, token={token_id}")
            if len(set(experts)) != top_k:
                raise ReferenceNotifyError(f"duplicate expert at rank={rank}, token={token_id}")
            token_node_mask = [0] * node_mask_words
            for expert in experts:
                if expert < 0 or expert >= num_experts:
                    raise ReferenceNotifyError(f"expert out of range: {expert}")
                rank_offsets.append(counts[rank][expert])
                counts[rank][expert] += 1
                destination_rank = topology.expert_to_rank[expert]
                destination_node = topology.rank_to_node[destination_rank]
                token_node_mask[destination_node // 64] |= 1 << (destination_node % 64)
            rank_order.extend(sorted(range(top_k), key=lambda slot: experts[slot]))
            rank_node_masks.extend(token_node_mask)
        local_offset.append(rank_offsets)
        combine_order.append(rank_order)
        destination_node_mask.append(rank_node_masks)

    delivery = list(range(world_size)) if arrival_order is None else list(arrival_order)
    if sorted(delivery) != list(range(world_size)):
        raise ReferenceNotifyError("arrival order must be a rank permutation")
    exchanged = [None] * world_size
    for source_rank in delivery:
        exchanged[source_rank] = list(counts[source_rank])

    owned = [[e for e in range(num_experts) if topology.expert_to_rank[e] == rank]
             for rank in range(world_size)]
    expert_row_count: List[List[int]] = []
    expert_row_prefix: List[List[int]] = []
    mtile_expected_rows: List[List[List[int]]] = []
    local_expert_index = [0] * num_experts
    for rank, experts in enumerate(owned):
        rank_counts = []
        prefix = [0]
        rank_tiles = []
        for local_index, expert in enumerate(experts):
            local_expert_index[expert] = local_index
            count = sum(exchanged[source][expert] for source in range(world_size))
            if expert_capacity is not None and count > expert_capacity[expert]:
                raise ReferenceNotifyError(f"expert capacity exceeded: {expert}")
            rank_counts.append(count)
            prefix.append(prefix[-1] + count)
            rank_tiles.append([min(m_tile_rows, count - first)
                               for first in range(0, count, m_tile_rows)])
        expert_row_count.append(rank_counts)
        expert_row_prefix.append(prefix)
        mtile_expected_rows.append(rank_tiles)

    route_dst_row: List[List[int]] = []
    route_dst_rank: List[List[int]] = []
    route_dst_node: List[List[int]] = []
    route_dst_local_rank: List[List[int]] = []
    reverse_routes: List[List[tuple]] = [
        [None] * expert_row_prefix[rank][-1] for rank in range(world_size)
    ]
    for source_rank, tokens in enumerate(selected_experts_by_rank):
        rank_rows = []
        rank_destinations = []
        rank_nodes = []
        rank_local_ranks = []
        flat = 0
        for token_id, experts in enumerate(tokens):
            for slot, expert in enumerate(experts):
                destination_rank = topology.expert_to_rank[expert]
                expert_base = expert_row_prefix[destination_rank][local_expert_index[expert]]
                source_base = sum(exchanged[prior][expert] for prior in range(source_rank))
                dst_row = expert_base + source_base + local_offset[source_rank][flat]
                rank_rows.append(dst_row)
                rank_destinations.append(destination_rank)
                rank_nodes.append(topology.rank_to_node[destination_rank])
                rank_local_ranks.append(topology.rank_to_local_rank[destination_rank])
                if reverse_routes[destination_rank][dst_row] is not None:
                    raise AssertionError("canonical row collision")
                reverse_routes[destination_rank][dst_row] = (
                    source_rank, token_id, slot, expert, destination_rank, dst_row
                )
                flat += 1
        route_dst_row.append(rank_rows)
        route_dst_rank.append(rank_destinations)
        route_dst_node.append(rank_nodes)
        route_dst_local_rank.append(rank_local_ranks)

    if any(route is None for rank_routes in reverse_routes for route in rank_routes):
        raise AssertionError("reverse-route hole")
    return ReferenceResult(counts, local_offset, route_dst_row, route_dst_rank,
                           route_dst_node, route_dst_local_rank,
                           destination_node_mask, owned, expert_row_count,
                           expert_row_prefix, combine_order, reverse_routes,
                           mtile_expected_rows)
