#!/usr/bin/env python3

import copy
import re
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import autotune_forward as tune  # noqa: E402


class ForwardAutotuneTest(unittest.TestCase):
    def setUp(self):
        self.plan = tune.load_plan(
            ROOT / "configs" / "forward_h200_autotune.json")
        self.candidates = tune.expand_candidates(self.plan)

    def test_default_plan_is_narrow_and_complete(self):
        self.assertEqual(len(self.candidates), 7)
        self.assertLessEqual(len(self.candidates), self.plan["max_candidates"])
        for candidate in self.candidates:
            self.assertEqual(set(candidate["config"]),
                             set(tune.PARAMETER_MACROS))

    def test_cpp_defaults_and_generator_have_same_parameter_surface(self):
        header = (ROOT / "src" / "cuda" / "forward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        cpp_macros = set(re.findall(
            r"^#ifndef (MEGAMOE_FWD_[A-Z0-9_]+)$", header,
            flags=re.MULTILINE))
        controls = {
            "MEGAMOE_FWD_COMPILED_TUNING",
            "MEGAMOE_FWD_ENABLE_LEGACY_ENV",
        }
        self.assertEqual(cpp_macros - controls,
                         set(tune.PARAMETER_MACROS.values()))

    def test_generated_header_disables_runtime_tuning(self):
        candidate = self.candidates[0]
        header = tune.generated_header(candidate["name"],
                                       candidate["config"])
        self.assertIn("#define MEGAMOE_FWD_COMPILED_TUNING 1", header)
        self.assertIn("#define MEGAMOE_FWD_ENABLE_LEGACY_ENV 0", header)
        for macro in tune.PARAMETER_MACROS.values():
            self.assertEqual(header.count(f"#define {macro} "), 1)

    def test_generated_header_is_stable(self):
        candidate = self.candidates[0]
        first = tune.generated_header(candidate["name"], candidate["config"])
        second = tune.generated_header(candidate["name"], candidate["config"])
        self.assertEqual(first, second)
        self.assertEqual(tune.candidate_digest(candidate["config"]),
                         tune.candidate_digest(candidate["config"]))

    def test_register_file_overflow_is_rejected(self):
        invalid = copy.deepcopy(self.candidates[0]["config"])
        invalid["math_registers"] = 255
        with self.assertRaisesRegex(tune.PlanError,
                                    "exceeds 65536"):
            tune.validate_tuning(invalid)

    def test_missing_fixed_parameter_is_rejected(self):
        invalid_plan = copy.deepcopy(self.plan)
        del invalid_plan["fixed"]["1_gemm"]["tile_k"]
        with self.assertRaisesRegex(tune.PlanError, "omits parameters"):
            tune.flatten_groups(invalid_plan["fixed"], require_all=True)

    def test_materialization_uses_content_addressed_build_directory(self):
        candidate = self.candidates[0]
        with tempfile.TemporaryDirectory() as directory:
            paths = tune.materialize_candidate(candidate, Path(directory))
            self.assertIn(tune.candidate_digest(candidate["config"]),
                          paths["build_dir"].name)
            self.assertEqual(paths["header"].read_text(encoding="utf-8"),
                             tune.generated_header(candidate["name"],
                                                   candidate["config"]))


if __name__ == "__main__":
    unittest.main()
