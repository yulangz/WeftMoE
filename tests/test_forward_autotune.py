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
import forward_jit  # noqa: E402


def makefile_list(variable: str) -> tuple[str, ...]:
    lines = (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    prefix = f"{variable} := "
    for index, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        values = []
        remainder = line[len(prefix):].strip()
        while True:
            continued = remainder.endswith("\\")
            if continued:
                remainder = remainder[:-1].strip()
            values.extend(remainder.split())
            if not continued:
                return tuple(values)
            index += 1
            remainder = lines[index].strip()
    raise AssertionError(f"Makefile variable {variable} is absent")


class ForwardAutotuneTest(unittest.TestCase):
    def setUp(self):
        self.plan = tune.load_plan(
            ROOT / "configs" / "forward_h200_autotune.json")
        self.candidates = tune.expand_candidates(self.plan)

    def test_default_plan_is_narrow_and_complete(self):
        self.assertEqual(self.plan["schema"], 2)
        self.assertIn("jit", self.plan)
        self.assertNotIn("build", self.plan)
        self.assertEqual(self.plan["jit"]["architecture"], "sm_90a")
        self.assertEqual(self.plan["selection"], {
            "confirmation_candidates": 2,
            "confirmation_repeats": 2,
            "confirmation_rounds": 110,
            "minimum_improvement_percent": 0.5,
        })
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

    def test_jit_cache_key_covers_signature_and_generated_source(self):
        signature = {"nvcc": "12.9", "architecture": "sm_90a"}
        first = forward_jit.content_cache_key(signature, "#define TILE_M 128\n")
        self.assertEqual(
            first,
            forward_jit.content_cache_key(
                {"architecture": "sm_90a", "nvcc": "12.9"},
                "#define TILE_M 128\n"))
        self.assertNotEqual(
            first,
            forward_jit.content_cache_key(signature, "#define TILE_M 64\n"))
        self.assertNotEqual(
            first,
            forward_jit.content_cache_key(
                {"nvcc": "12.9", "architecture": "sm_100a"},
                "#define TILE_M 128\n"))

    def test_jit_settings_are_explicit_and_closed(self):
        forward_jit.validate_jit_config(self.plan["jit"])
        invalid = copy.deepcopy(self.plan["jit"])
        invalid["shell_command"] = "make"
        with self.assertRaisesRegex(forward_jit.JitError,
                                    "unknown JIT settings"):
            forward_jit.validate_jit_config(invalid)

    def test_jit_source_graph_matches_conventional_testing_build(self):
        self.assertEqual(makefile_list("LIB_SOURCES"),
                         forward_jit.HOST_SOURCES)
        self.assertEqual(makefile_list("CUDA_LIB_SOURCES"),
                         forward_jit.CUDA_SOURCES)
        self.assertEqual(makefile_list("CUDA_TEST_SOURCES"),
                         (forward_jit.TEST_SOURCE,))

    def test_selection_settings_are_explicit_and_closed(self):
        invalid = copy.deepcopy(self.plan["selection"])
        invalid["shell_command"] = "benchmark-once"
        with self.assertRaisesRegex(tune.PlanError,
                                    "selection must contain exactly"):
            tune.validate_selection(invalid, len(self.candidates))

    def test_weighted_metric_uses_measured_iterations(self):
        records = [
            {"measured": 30, "forward": {"mean": 10.0}},
            {"measured": 100, "forward": {"mean": 20.0}},
        ]
        self.assertAlmostEqual(
            tune.weighted_metric(records, "forward.mean"),
            (30 * 10.0 + 100 * 20.0) / 130)

    def test_confirmation_threshold_rejects_noise(self):
        incumbent = {
            "name": "baseline",
            "selection_metric_us": 100.1,
            "aggregate_metric_us": 100.0,
        }
        challenger = {
            "name": "challenger",
            "selection_metric_us": 99.8,
            "aggregate_metric_us": 99.8,
        }
        winner, decision = tune.choose_confirmed_candidate(
            [incumbent, challenger], "baseline", 0.5)
        self.assertIs(winner, incumbent)
        self.assertEqual(decision["reason"], "challenger_below_threshold")
        challenger["aggregate_metric_us"] = 99.0
        winner, decision = tune.choose_confirmed_candidate(
            [incumbent, challenger], "baseline", 0.5)
        self.assertIs(winner, challenger)
        self.assertEqual(decision["reason"],
                         "challenger_exceeds_threshold")

    def test_timing_summary_uses_parallel_phase_wall_time(self):
        result = {
            "phase_wall_seconds": {"compile": 7.0},
            "candidates": [
                {"build": {"elapsed_seconds": 6.0}},
                {"build": {"elapsed_seconds": 6.5}},
            ],
        }
        timing = tune._timing_summary(result, 8.0)
        self.assertEqual(timing["build_seconds"], 7.0)
        self.assertEqual(timing["total_seconds"], 8.0)

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
