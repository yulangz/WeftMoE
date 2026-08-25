import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

from benchmarks.nccl_library import configure_nccl_library
from benchmarks.run_integrated_comparison import verify_fixture_agreement

from reference.notify_dispatch_reference import (
    ReferenceConfig,
    ReferenceNotifyError,
    ReferenceTopology,
    build_rank_local_histogram,
    build_reference_all,
)


def config(nodes=2, gpus=2, experts=8, top_k=2, mtile=4):
    world = nodes * gpus
    topology = ReferenceTopology(
        world, nodes,
        [node for node in range(nodes) for _ in range(gpus)],
        [local for _ in range(nodes) for local in range(gpus)],
        [expert % world for expert in range(experts)],
    )
    return ReferenceConfig(experts, top_k, mtile, topology)


class RankLocalReferenceTest(unittest.TestCase):
    def test_nccl_library_path_list_resolution(self):
        with tempfile.TemporaryDirectory() as nccl_dir, \
             tempfile.TemporaryDirectory() as compat_dir:
            library = Path(nccl_dir) / "libnccl.so.2"
            library.touch()
            environment = {"LD_LIBRARY_PATH": "/existing"}
            resolved = configure_nccl_library(
                environment, os.pathsep.join((compat_dir, nccl_dir)))
            self.assertEqual(resolved, str(library))
            self.assertEqual(environment["MEGAMOE_NCCL_LIBRARY"],
                             str(library))
            self.assertEqual(environment["LD_LIBRARY_PATH"].split(
                os.pathsep), [compat_dir, nccl_dir, "/existing"])

    def test_actual_device_weight_hash_mismatch_is_fatal(self):
        base = {
            "weight_generator_version": 3,
            "per_rank_activation_hash": [1],
            "per_rank_route_hash": [2],
            "per_rank_gate_hash": [3],
            "per_rank_weight_sample_hash": [4],
            "per_rank_actual_w1_device_hash": [[5, 6]],
            "per_rank_actual_w2_device_hash": [[7, 8]],
        }
        foreign = dict(base)
        foreign["per_rank_actual_w1_device_hash"] = [[5, 9]]
        with self.assertRaisesRegex(RuntimeError,
                                    "actual_w1_device_hash"):
            verify_fixture_agreement([base], [foreign])

    def test_dynamic_gate_fixture_hashes(self):
        fixture = json.loads((Path(__file__).parents[1] / "benchmarks" /
                              "dynamic_gate_fixture.json").read_text())
        seed = int(fixture["seed_hex"], 16)
        tokens, top_k = fixture["shape"]
        actual = []
        for rank in range(8):
            digest = 1469598103934665603
            for token in range(tokens):
                scores = [1 + ((seed + token * 1315423911 +
                                slot * 2654435761 + rank * 97531) % 65521)
                          for slot in range(top_k)]
                total = sum(scores)
                for score in scores:
                    for byte in struct.pack("<f", score / total):
                        digest = ((digest ^ byte) * 1099511628211) & (
                            (1 << 64) - 1)
            actual.append(str(digest))
        self.assertEqual(actual,
                         fixture["fnv1a64_fp32_little_endian_by_rank"])

    def test_rank_local_determinism_and_partial_tile(self):
        cfg = config()
        routes = [
            [[0, 1], [0, 2], [0, 3]],
            [[0, 4], [0, 5]],
            [],
            [[0, 6]],
        ]
        first = build_reference_all(cfg, routes, "ref", 2)
        second = build_reference_all(cfg, routes, "ref", 2)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["incoming"]["expert_row_count"][0], 6)
        self.assertEqual(first[0]["incoming"]["mtile_expected_rows"][0], [4, 2])

    def test_node_dedup_and_multiword_mask(self):
        cfg = config(nodes=73, gpus=1, experts=73, top_k=2)
        routes = [[] for _ in range(73)]
        routes[0] = [[0, 72]]
        result = build_reference_all(cfg, routes, "nodes", 1)[0]
        self.assertEqual(result["outgoing"]["node_mask_words"], 2)
        self.assertEqual(result["outgoing"]["destination_node_mask"], [1, 1 << 8])
        self.assertEqual(len(result["transport"]["outgoing_packets"]), 2)

    def test_topology_and_topk_rejections(self):
        cfg = config()
        sparse = ReferenceConfig(
            cfg.num_experts, cfg.top_k, cfg.m_tile_rows,
            ReferenceTopology(4, 2, [0, 0, 1, 1], [0, 2, 0, 1],
                              cfg.topology.expert_to_rank),
        )
        with self.assertRaises(ReferenceNotifyError):
            build_rank_local_histogram(sparse, 0, [])
        bad_topk = ReferenceConfig(2, 3, 4,
                                   ReferenceTopology(1, 1, [0], [0], [0, 0]))
        with self.assertRaises(ReferenceNotifyError):
            build_rank_local_histogram(bad_topk, 0, [])


if __name__ == "__main__":
    unittest.main()
