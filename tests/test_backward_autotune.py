#!/usr/bin/env python3

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import autotune_backward as tune  # noqa: E402
import forward_jit  # noqa: E402


class BackwardAutotuneTest(unittest.TestCase):
    def setUp(self):
        self.plan = tune.load_plan(
            ROOT / "configs" / "backward_h200_autotune.json")
        self.candidates = tune.expand_candidates(self.plan)

    def test_multinode_plan_is_narrow_complete_and_unique(self):
        self.assertEqual(len(self.candidates), 8)
        self.assertLessEqual(len(self.candidates),
                             self.plan["max_candidates"])
        self.assertEqual(self.plan["invariants"], tune.EXPECTED_INVARIANTS)
        for candidate in self.candidates:
            self.assertEqual(set(candidate["config"]),
                             set(tune.PARAMETER_MACROS))

    def test_world8_plan_reuses_split_architecture(self):
        plan = tune.load_plan(
            ROOT / "configs" / "backward_h200_world8_autotune.json")
        candidates = tune.expand_candidates(plan)
        self.assertEqual(len(candidates), 6)
        self.assertEqual(plan["benchmark"]["launcher"], "local")
        self.assertEqual(candidates[0]["config"]["relay_ctas"], 12)
        self.assertEqual(candidates[0]["config"]["rdma_ctas"], 8)
        self.assertEqual(candidates[0]["config"]["dispatch_batch_rows"], 32)
        self.assertEqual(candidates[0]["config"]["dw_claim_tiles"], 8)

    def test_cpp_defaults_and_generator_have_same_parameter_surface(self):
        header = (ROOT / "src" / "cuda" / "backward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        cpp_macros = set(re.findall(
            r"^#ifndef (MEGAMOE_BWD_[A-Z0-9_]+)$", header,
            flags=re.MULTILINE))
        self.assertEqual(cpp_macros, set(tune.PARAMETER_MACROS.values()))

    def test_cpp_defaults_match_incumbent(self):
        header = (ROOT / "src" / "cuda" / "backward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        defaults = dict(re.findall(
            r"^#define (MEGAMOE_BWD_[A-Z0-9_]+) ([0-9]+)$", header,
            flags=re.MULTILINE))
        incumbent = self.candidates[0]["config"]
        for key, macro in tune.PARAMETER_MACROS.items():
            self.assertEqual(int(defaults[macro]), incumbent[key], key)

    def test_generated_header_is_complete_and_stable(self):
        candidate = self.candidates[0]
        first = tune.generated_header(candidate["name"], candidate["config"])
        second = tune.generated_header(candidate["name"], candidate["config"])
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            tune.generated_header("renamed_candidate", candidate["config"]),
        )
        for macro in tune.PARAMETER_MACROS.values():
            self.assertEqual(first.count(f"#define {macro} "), 1)

    def test_jit_targets_have_disjoint_cache_keys(self):
        signature = {"architecture": "sm_90a"}
        header = "#define TILE_M 128\n"
        forward = forward_jit.content_cache_key(signature, header)
        backward = forward_jit.content_cache_key(
            signature, header,
            artifact_kind=forward_jit.BACKWARD_TARGET.artifact_kind,
        )
        self.assertNotEqual(forward, backward)
        self.assertNotEqual(forward_jit.FORWARD_TARGET.generated_header_name,
                            forward_jit.BACKWARD_TARGET.generated_header_name)

    def test_invalid_scheduler_invariants_are_rejected(self):
        invalid = copy.deepcopy(self.plan)
        invalid["invariants"]["second_dx_probe"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.json"
            path.write_text(json.dumps(invalid), encoding="utf-8")
            with self.assertRaisesRegex(tune.PlanError,
                                        "scheduler contract"):
                tune.load_plan(path)

    def test_materialized_header_uses_content_digest(self):
        candidate = self.candidates[0]
        with tempfile.TemporaryDirectory() as directory:
            paths = tune.materialize_candidate(candidate, Path(directory))
            self.assertIn(tune.candidate_digest(candidate["config"]),
                          paths["build_dir"].name)
            self.assertEqual(paths["header"].name,
                             "megamoe_backward_tuning_generated.hpp")

    def test_shared_memory_model_matches_default(self):
        self.assertEqual(
            tune.dynamic_shared_memory(self.candidates[0]["config"]),
            213104,
        )

    def test_launcher_uses_cached_backward_scenario(self):
        command = tune.launcher_command(
            Path("/tmp/test_cuda_notify"), self.plan["benchmark"], rounds=30)
        self.assertEqual(command[command.index("--scenario") + 1],
                         "gin_backward_bench")
        self.assertEqual(command[command.index("--gpus-per-node") + 1], "8")

    def test_correctness_fingerprint_covers_all_outputs(self):
        record = {
            "shape": {"tokens": 8},
            "per_rank_dx_hash": [1],
            "per_rank_dgate_hash": [2],
            "per_rank_dw1_hash": [3],
            "per_rank_dw2_hash": [4],
            "repeat_equal": True,
        }
        first = tune.correctness_fingerprint(record)
        record["per_rank_dw2_hash"] = [5]
        self.assertNotEqual(first, tune.correctness_fingerprint(record))


if __name__ == "__main__":
    unittest.main()
