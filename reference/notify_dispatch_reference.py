"""Independent rank-local Notify Dispatch reference.

This clarity-first implementation has no binding to the C++ library. It emits
the same machine-readable deterministic fields for CUDA/backend comparison.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence


class ReferenceNotifyError(ValueError):
    pass


@dataclass(frozen=True)
class ReferenceTopology:
    world_size: int
    num_nodes: int
    rank_to_node: Sequence[int]
    rank_to_local_rank: Sequence[int]
    expert_to_rank: Sequence[int]


@dataclass(frozen=True)
class ReferenceConfig:
    num_experts: int
    top_k: int
    m_tile_rows: int
    topology: ReferenceTopology
    expert_capacity: Sequence[int] = ()
    workspace_capacity: Sequence[int] = ()


@dataclass(frozen=True)
class LocalHistogram:
    config: ReferenceConfig
    calling_rank: int
    outgoing: dict
    local_expert_count: List[int]
    packet_count_by_destination_node: List[int]


def validate_config(config: ReferenceConfig) -> tuple[int, List[int]]:
    topology = config.topology
    if config.num_experts <= 0 or config.top_k <= 0 or config.m_tile_rows <= 0:
        raise ReferenceNotifyError("runtime dimensions must be positive")
    if config.top_k > config.num_experts:
        raise ReferenceNotifyError("top_k exceeds num_experts")
    if topology.world_size <= 0 or topology.num_nodes <= 0:
        raise ReferenceNotifyError("world_size and num_nodes must be positive")
    if topology.world_size % topology.num_nodes:
        raise ReferenceNotifyError("heterogeneous topology is unsupported")
    if len(topology.rank_to_node) != topology.world_size or \
            len(topology.rank_to_local_rank) != topology.world_size:
        raise ReferenceNotifyError("rank coordinate arrays must contain every rank")
    if len(topology.expert_to_rank) != config.num_experts:
        raise ReferenceNotifyError("expert ownership size mismatch")
    if config.expert_capacity and len(config.expert_capacity) != config.num_experts:
        raise ReferenceNotifyError("expert capacity size mismatch")
    if config.workspace_capacity and len(config.workspace_capacity) != topology.world_size:
        raise ReferenceNotifyError("workspace capacity size mismatch")

    gpus_per_node = topology.world_size // topology.num_nodes
    rank_at = [-1] * topology.world_size
    ranks_per_node = [0] * topology.num_nodes
    for rank, (node, local_rank) in enumerate(zip(topology.rank_to_node,
                                                  topology.rank_to_local_rank)):
        if node < 0 or node >= topology.num_nodes:
            raise ReferenceNotifyError(f"rank={rank} has out-of-range node={node}")
        if local_rank < 0 or local_rank >= gpus_per_node:
            raise ReferenceNotifyError(
                f"rank={rank}, node={node} has sparse local_rank={local_rank}")
        coordinate = node * gpus_per_node + local_rank
        if rank_at[coordinate] != -1:
            raise ReferenceNotifyError(
                f"duplicate coordinate node={node}, local_rank={local_rank}")
        rank_at[coordinate] = rank
        ranks_per_node[node] += 1
    for node in range(topology.num_nodes):
        if ranks_per_node[node] != gpus_per_node:
            raise ReferenceNotifyError(f"node={node} is empty/non-homogeneous")
        for local_rank in range(gpus_per_node):
            if rank_at[node * gpus_per_node + local_rank] < 0:
                raise ReferenceNotifyError(
                    f"missing rail node={node}, local_rank={local_rank}")
    for expert, owner in enumerate(topology.expert_to_rank):
        if owner < 0 or owner >= topology.world_size:
            raise ReferenceNotifyError(f"expert={expert} owner={owner} out of range")
    return gpus_per_node, rank_at


def build_rank_local_histogram(config: ReferenceConfig, calling_rank: int,
                               local_routes: Sequence[Sequence[int]]) -> LocalHistogram:
    validate_config(config)
    if calling_rank < 0 or calling_rank >= config.topology.world_size:
        raise ReferenceNotifyError("calling rank out of range")
    counts = [0] * config.num_experts
    packet_counts = [0] * config.topology.num_nodes
    node_mask_words = (config.topology.num_nodes + 63) // 64
    outgoing = {
        "num_tokens": len(local_routes),
        "top_k": config.top_k,
        "node_mask_words": node_mask_words,
        "expert_id": [],
        "local_offset": [],
        "route_dst_row": [0] * (len(local_routes) * config.top_k),
        "route_dst_rank": [],
        "route_dst_node": [],
        "route_dst_local_rank": [],
        "combine_order": [],
        "destination_node_mask": [],
        "reverse_routes": [],
    }
    for token, experts in enumerate(local_routes):
        if len(experts) != config.top_k:
            raise ReferenceNotifyError(
                f"route shape mismatch rank={calling_rank}, token={token}")
        if len(set(experts)) != config.top_k:
            raise ReferenceNotifyError(
                f"duplicate expert rank={calling_rank}, token={token}")
        token_masks = [0] * node_mask_words
        token_nodes = set()
        for slot, expert in enumerate(experts):
            if expert < 0 or expert >= config.num_experts:
                raise ReferenceNotifyError(
                    f"expert out of range rank={calling_rank}, token={token}, slot={slot}")
            destination_rank = config.topology.expert_to_rank[expert]
            destination_node = config.topology.rank_to_node[destination_rank]
            outgoing["expert_id"].append(expert)
            outgoing["local_offset"].append(counts[expert])
            counts[expert] += 1
            outgoing["route_dst_rank"].append(destination_rank)
            outgoing["route_dst_node"].append(destination_node)
            outgoing["route_dst_local_rank"].append(
                config.topology.rank_to_local_rank[destination_rank])
            token_masks[destination_node // 64] |= 1 << (destination_node % 64)
            token_nodes.add(destination_node)
        outgoing["combine_order"].extend(
            sorted(range(config.top_k), key=lambda slot: experts[slot]))
        outgoing["destination_node_mask"].extend(token_masks)
        for node in token_nodes:
            packet_counts[node] += 1
    return LocalHistogram(config, calling_rank, outgoing, counts, packet_counts)


def _signal(source_node: int, destination_node: int, source_rank: int,
            destination_proxy_rank: int, rail_local_rank: int,
            expected: int) -> dict:
    return {
        "source_node": source_node,
        "destination_node": destination_node,
        "source_rank": source_rank,
        "destination_proxy_rank": destination_proxy_rank,
        "rail_local_rank": rail_local_rank,
        "channel": 0,
        "expected_packets": expected,
        "expected_signals": expected,
    }


class _Hash64:
    def __init__(self):
        self.value = 1469598103934665603

    def add(self, value):
        if isinstance(value, str):
            encoded = value.encode("utf-8")
            self.add(len(encoded))
            for byte in encoded:
                self.value ^= byte
                self.value = (self.value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
            return
        value = int(value) & 0xFFFFFFFFFFFFFFFF
        for shift in range(0, 64, 8):
            self.value ^= (value >> shift) & 0xFF
            self.value = (self.value * 1099511628211) & 0xFFFFFFFFFFFFFFFF

    def vector(self, values):
        self.add(len(values))
        for value in values:
            self.add(value)


def _content_hash(metadata: dict) -> int:
    h = _Hash64()
    config = metadata["config"]
    for key in ("num_experts", "top_k", "m_tile_rows", "world_size", "num_nodes"):
        h.add(config[key])
    for key in ("rank_to_node", "rank_to_local_rank", "expert_to_rank",
                "expert_capacity", "workspace_capacity"):
        h.vector(config[key])
    h.add(metadata["calling_rank"])
    h.add(metadata["communicator_id"])
    for row in metadata["source_expert_count"]:
        h.vector(row)
    outgoing = metadata["outgoing"]
    for key in ("num_tokens", "top_k", "node_mask_words"):
        h.add(outgoing[key])
    for key in ("expert_id", "local_offset", "route_dst_row", "route_dst_rank",
                "route_dst_node", "route_dst_local_rank", "combine_order",
                "destination_node_mask"):
        h.vector(outgoing[key])
    for route in outgoing["reverse_routes"]:
        for key in ("source_rank", "local_token_id", "topk_slot", "expert_id",
                    "destination_rank", "dst_row"):
            h.add(route[key])
    incoming = metadata["incoming"]
    h.add(incoming["rank"])
    for key in ("expert_ids", "expert_row_count", "expert_row_prefix"):
        h.vector(incoming[key])
    for tiles in incoming["mtile_expected_rows"]:
        h.vector(tiles)
    for row in incoming["incoming_rows"]:
        for key in ("source_rank", "expert_id", "source_expert_offset", "dst_row"):
            h.add(row[key])
    transport = metadata["transport"]
    for packet in transport["outgoing_packets"]:
        for key in ("source_rank", "destination_node", "sequence_on_link"):
            h.add(packet["identity"][key])
        for key in ("local_token_id", "proxy_rank", "channel"):
            h.add(packet[key])
        for target in packet["fanout"]:
            for key in ("topk_slot", "destination_local_rank", "expert_id", "dst_row"):
                h.add(target[key])
    for key in ("dispatch_wave_packet_order", "dispatch_wave_offsets",
                "dispatch_wave_destination_node", "dispatch_wave_channel",
                "dispatch_wave_first_primary_tile",
                "dispatch_wave_last_primary_tile",
                "dispatch_wave_packet_count",
                "dispatch_wave_payload_bytes"):
        h.vector(transport[key])
    for group in ("outgoing_signals", "incoming_proxy_signals"):
        for signal in transport[group]:
            for key in ("source_node", "destination_node", "source_rank",
                        "destination_proxy_rank", "rail_local_rank", "channel",
                        "expected_packets", "expected_signals"):
                h.add(signal[key])
    return h.value


def materialize_rank_metadata(local: LocalHistogram, count_matrix: Sequence[Sequence[int]],
                              summary_matrix: Sequence[Sequence[int]],
                              communicator_id: str, generation: int) -> dict:
    config = local.config
    gpus_per_node, rank_at = validate_config(config)
    world_size = config.topology.world_size
    if not communicator_id or generation <= 0:
        raise ReferenceNotifyError("invalid collective identity/generation")
    if len(count_matrix) != world_size or any(
            len(row) != config.num_experts for row in count_matrix):
        raise ReferenceNotifyError("count matrix shape mismatch")
    if len(summary_matrix) != world_size or any(
            len(row) != config.topology.num_nodes for row in summary_matrix):
        raise ReferenceNotifyError("summary matrix shape mismatch")
    if list(count_matrix[local.calling_rank]) != local.local_expert_count:
        raise ReferenceNotifyError("calling rank count row was corrupted")
    if list(summary_matrix[local.calling_rank]) != local.packet_count_by_destination_node:
        raise ReferenceNotifyError("calling rank summary row was corrupted")

    source_prefix = [[0] * config.num_experts for _ in range(world_size + 1)]
    for source in range(world_size):
        for expert in range(config.num_experts):
            source_prefix[source + 1][expert] = (
                source_prefix[source][expert] + count_matrix[source][expert])
    expert_total = source_prefix[-1]
    expert_base = [0] * config.num_experts
    rank_total = [0] * world_size
    for destination_rank in range(world_size):
        prefix = 0
        for expert in range(config.num_experts):
            if config.topology.expert_to_rank[expert] != destination_rank:
                continue
            expert_base[expert] = prefix
            if config.expert_capacity and expert_total[expert] > config.expert_capacity[expert]:
                raise ReferenceNotifyError(f"expert capacity exceeded expert={expert}")
            prefix += expert_total[expert]
        rank_total[destination_rank] = prefix
        if config.workspace_capacity and prefix > config.workspace_capacity[destination_rank]:
            raise ReferenceNotifyError(
                f"workspace capacity exceeded rank={destination_rank}")

    outgoing = {key: (list(value) if isinstance(value, list) else value)
                for key, value in local.outgoing.items()}
    outgoing["route_dst_row"] = list(outgoing["route_dst_row"])
    outgoing["reverse_routes"] = []
    for flat, expert in enumerate(outgoing["expert_id"]):
        token, slot = divmod(flat, config.top_k)
        destination_rank = outgoing["route_dst_rank"][flat]
        dst_row = (expert_base[expert] + source_prefix[local.calling_rank][expert] +
                   outgoing["local_offset"][flat])
        if dst_row >= rank_total[destination_rank]:
            raise AssertionError("canonical row outside destination")
        outgoing["route_dst_row"][flat] = dst_row
        outgoing["reverse_routes"].append({
            "source_rank": local.calling_rank,
            "local_token_id": token,
            "topk_slot": slot,
            "expert_id": expert,
            "destination_rank": destination_rank,
            "dst_row": dst_row,
        })

    owned = [expert for expert in range(config.num_experts)
             if config.topology.expert_to_rank[expert] == local.calling_rank]
    incoming = {
        "rank": local.calling_rank,
        "expert_ids": owned,
        "expert_row_count": [expert_total[e] for e in owned],
        "expert_row_prefix": [0],
        "mtile_expected_rows": [],
        "incoming_rows": [],
    }
    for expert in owned:
        count = expert_total[expert]
        incoming["expert_row_prefix"].append(
            incoming["expert_row_prefix"][-1] + count)
        incoming["mtile_expected_rows"].append(
            [min(config.m_tile_rows, count - first)
             for first in range(0, count, config.m_tile_rows)])
        for source in range(world_size):
            for offset in range(count_matrix[source][expert]):
                incoming["incoming_rows"].append({
                    "source_rank": source,
                    "expert_id": expert,
                    "source_expert_offset": offset,
                    "dst_row": expert_base[expert] + source_prefix[source][expert] + offset,
                })
    incoming["incoming_rows"].sort(key=lambda row: row["dst_row"])

    source_rank = local.calling_rank
    source_node = config.topology.rank_to_node[source_rank]
    source_local_rank = config.topology.rank_to_local_rank[source_rank]
    packets = []
    outgoing_signals = []
    for destination_node in range(config.topology.num_nodes):
        proxy_rank = rank_at[destination_node * gpus_per_node + source_local_rank]
        sequence = 0
        for token in range(outgoing["num_tokens"]):
            fanout = []
            for slot in range(config.top_k):
                flat = token * config.top_k + slot
                if outgoing["route_dst_node"][flat] != destination_node:
                    continue
                fanout.append({
                    "topk_slot": slot,
                    "destination_local_rank": outgoing["route_dst_local_rank"][flat],
                    "expert_id": outgoing["expert_id"][flat],
                    "dst_row": outgoing["route_dst_row"][flat],
                })
            if not fanout:
                continue
            fanout.sort(key=lambda target: target["expert_id"])
            packets.append({
                "identity": {"source_rank": source_rank,
                             "destination_node": destination_node,
                             "sequence_on_link": sequence},
                "local_token_id": token,
                "proxy_rank": proxy_rank,
                "channel": 0,
                "fanout": fanout,
            })
            sequence += 1
        expected = summary_matrix[source_rank][destination_node]
        if sequence != expected:
            raise AssertionError("packet summary mismatch")
        outgoing_signals.append(_signal(source_node, destination_node, source_rank,
                                        proxy_rank, source_local_rank, expected))

    destination_node = config.topology.rank_to_node[source_rank]
    rail_local_rank = config.topology.rank_to_local_rank[source_rank]
    incoming_signals = []
    for remote_source_node in range(config.topology.num_nodes):
        remote_source_rank = rank_at[
            remote_source_node * gpus_per_node + rail_local_rank]
        expected = summary_matrix[remote_source_rank][destination_node]
        incoming_signals.append(_signal(remote_source_node, destination_node,
                                        remote_source_rank, source_rank,
                                        rail_local_rank, expected))

    priorities = []
    for packet_index, packet in enumerate(packets):
        primary = min(
            ((target["dst_row"] - expert_base[target["expert_id"]]) //
             config.m_tile_rows,
             target["expert_id"],
             config.topology.expert_to_rank[target["expert_id"]])
            for target in packet["fanout"])
        tile, expert, destination_rank = primary
        scheduled_expert = expert
        if gpus_per_node == 8:
            local = config.topology.rank_to_local_rank[destination_rank]
            local_priority = ((local & 1) << 2) | (local & 2) | ((local & 4) >> 2)
            scheduled_expert += local_priority * config.num_experts
        priorities.append((packet["identity"]["destination_node"], tile,
                           scheduled_expert, destination_rank,
                           packet["local_token_id"], packet_index))
    priorities.sort()
    wave_order = [entry[-1] for entry in priorities]
    wave_offsets = [0]
    wave_nodes = []
    wave_channels = []
    wave_first_tiles = []
    wave_last_tiles = []
    wave_packet_counts = []
    wave_payload_bytes = []
    previous = None
    for position, entry in enumerate(priorities):
        node, tile, _expert, _rank, _token, packet_index = entry
        key = (node, tile)
        if key != previous:
            if position:
                wave_offsets.append(position)
            wave_nodes.append(node)
            wave_channels.append(packets[packet_index]["channel"])
            wave_first_tiles.append(tile)
            wave_last_tiles.append(tile)
            wave_packet_counts.append(0)
            wave_payload_bytes.append(0)
            previous = key
        wave_packet_counts[-1] += 1
        wave_payload_bytes[-1] += 32 + 32 * len(packets[packet_index]["fanout"])
    if priorities:
        wave_offsets.append(len(priorities))

    metadata = {
        "schema": 1,
        "config": {
            "num_experts": config.num_experts,
            "top_k": config.top_k,
            "m_tile_rows": config.m_tile_rows,
            "world_size": world_size,
            "num_nodes": config.topology.num_nodes,
            "rank_to_node": list(config.topology.rank_to_node),
            "rank_to_local_rank": list(config.topology.rank_to_local_rank),
            "expert_to_rank": list(config.topology.expert_to_rank),
            "expert_capacity": list(config.expert_capacity),
            "workspace_capacity": list(config.workspace_capacity),
        },
        "calling_rank": source_rank,
        "communicator_id": communicator_id,
        "identity": {"content_hash": 0, "generation": generation},
        "source_expert_count": [list(row) for row in count_matrix],
        "outgoing": outgoing,
        "incoming": incoming,
        "transport": {
            "outgoing_packets": packets,
            "dispatch_wave_packet_order": wave_order,
            "dispatch_wave_offsets": wave_offsets,
            "dispatch_wave_destination_node": wave_nodes,
            "dispatch_wave_channel": wave_channels,
            "dispatch_wave_first_primary_tile": wave_first_tiles,
            "dispatch_wave_last_primary_tile": wave_last_tiles,
            "dispatch_wave_packet_count": wave_packet_counts,
            "dispatch_wave_payload_bytes": wave_payload_bytes,
            "outgoing_signals": outgoing_signals,
            "incoming_proxy_signals": incoming_signals,
        },
    }
    metadata["identity"]["content_hash"] = _content_hash(metadata)
    return metadata


def build_reference_all(config: ReferenceConfig,
                        routes_by_rank: Sequence[Sequence[Sequence[int]]],
                        communicator_id: str = "host-fake",
                        generation: int = 1) -> List[dict]:
    validate_config(config)
    if len(routes_by_rank) != config.topology.world_size:
        raise ReferenceNotifyError("one local route input is required per rank")
    locals_ = [build_rank_local_histogram(config, rank, routes_by_rank[rank])
               for rank in range(config.topology.world_size)]
    counts = [local.local_expert_count for local in locals_]
    summaries = [local.packet_count_by_destination_node for local in locals_]
    return [materialize_rank_metadata(local, counts, summaries,
                                      communicator_id, generation)
            for local in locals_]
