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

    def test_world8_plan_is_narrow_and_unique(self):
        plan = tune.load_plan(
            ROOT / "configs" / "forward_h200_world8_autotune.json")
        candidates = tune.expand_candidates(plan)
        self.assertEqual(len(candidates), 4)
        selected = candidates[0]["config"]
        self.assertEqual(selected["rdma_ctas"], 0)
        self.assertEqual(selected["relay_ctas"], 16)
        self.assertEqual(selected["local_dispatch_tma_warps"], 4)

    def test_benchmark_workload_is_explicit(self):
        benchmark = self.plan["benchmark"]
        self.assertEqual(benchmark["seed"], 0x4D554C54)
        self.assertEqual(benchmark["top_k"], 6)
        self.assertEqual(benchmark["tokens_per_rank"], 8192)
        command = tune.launcher_command(
            Path("/tmp/test_cuda_notify"), benchmark,
            rounds=benchmark["tune_rounds"])
        self.assertEqual(command[command.index("--seed") + 1],
                         str(benchmark["seed"]))

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

    def test_cpp_defaults_match_explicit_production_point(self):
        header = (ROOT / "src" / "cuda" / "forward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        cpp_defaults = dict(re.findall(
            r"^#define (MEGAMOE_FWD_[A-Z0-9_]+) ([0-9]+)$", header,
            flags=re.MULTILINE))
        production = self.candidates[0]["config"]
        for key, macro in tune.PARAMETER_MACROS.items():
            expected = production[key]
            if key in tune.BOOLEAN_PARAMETERS:
                expected = int(expected)
            elif key in tune.ENUM_PARAMETERS:
                expected = tune.ENUM_PARAMETERS[key][expected]
            self.assertEqual(int(cpp_defaults[macro]), expected,
                             f"C++ default differs for {key}")

    def test_generated_header_disables_runtime_tuning(self):
        candidate = self.candidates[0]
        header = tune.generated_header(candidate["name"],
                                       candidate["config"])
        self.assertIn("#define MEGAMOE_FWD_COMPILED_TUNING 1", header)
        self.assertIn("#define MEGAMOE_FWD_ENABLE_LEGACY_ENV 0", header)
        self.assertIn("#define MEGAMOE_FWD_REDUCE_SCHEDULER 0", header)
        for macro in tune.PARAMETER_MACROS.values():
            self.assertEqual(header.count(f"#define {macro} "), 1)

    def test_default_source_is_compile_time_specialized(self):
        header = (ROOT / "src" / "cuda" / "forward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        self.assertRegex(header,
                         r"(?m)^#define MEGAMOE_FWD_COMPILED_TUNING 1$")
        self.assertRegex(
            header,
            r"(?m)^#define MEGAMOE_FWD_ENABLE_LEGACY_ENV 0$")

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

    def test_shared_memory_model_matches_current_m128_shape(self):
        memory = tune.dynamic_shared_memory(self.candidates[0]["config"], 16)
        self.assertEqual(memory["compute_bytes"], 196832)
        self.assertEqual(memory["transport_bytes"], 196832)


if __name__ == "__main__":
    unittest.main()
