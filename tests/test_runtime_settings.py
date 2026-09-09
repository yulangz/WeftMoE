"""Preparation policy shared by CLI and future framework adapters."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from weftmoe import ModelSpec, RuntimeSettings, token_bucket
from autotune_common import configure_preparation, PlanError


class PreparationTests(unittest.TestCase):
    def test_bucket_boundaries_and_overflow(self):
        for tokens, representative in [(0, 64), (1, 64), (64, 64), (65, 8192),
                                       (8192, 8192), (8193, 65536), (65536, 65536)]:
            self.assertEqual(token_bucket(tokens).tune_tokens, representative)
        for tokens in (-1, 65537, True, 1.5):
            with self.assertRaises(ValueError):
                token_bucket(tokens)

    def test_environment_and_phase_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = RuntimeSettings.from_environment({"WEFTMOE_ROOT": directory,
                "WEFTMOE_AUTOTUNE_CONFIG": str(Path(directory) / "space.json")})
            self.assertEqual(settings.compile_directory("forward"), Path(directory) / "compile/forward")
            self.assertEqual(settings.tune_directory("backward"), Path(directory) / "tune/backward")
            self.assertEqual(settings.autotune_config, Path(directory) / "space.json")
            self.assertFalse(settings.compile_directory("forward").exists())

    def test_workload_keeps_requested_count(self):
        for exact, expected in [(False, 8192), (True, 1000)]:
            plan = {"jit": {}, "benchmark": {"tokens_per_rank": 1000}}
            with patch.dict(os.environ, {"WEFTMOE_ROOT": "/tmp/weft-policy-test"}):
                configure_preparation(plan, "forward", exact_tokens=exact)
            self.assertEqual(plan["benchmark"]["tokens_per_rank"], expected)
            self.assertEqual(plan["benchmark"]["requested_tokens_per_rank"], 1000)
            self.assertEqual(plan["jit"]["cache_root"], "/tmp/weft-policy-test/compile/forward")
        plan = {"jit": {}, "benchmark": {"tokens_per_rank": 65537}}
        with self.assertRaises(PlanError):
            configure_preparation(plan, "forward")
        configure_preparation(plan, "forward", exact_tokens=True)
        self.assertEqual(plan["benchmark"]["tokens_per_rank"], 65537)

    def test_model_is_separate_from_tuning(self):
        self.assertEqual(ModelSpec(2560, 768, 512, 8).top_k, 8)
        with self.assertRaises(ValueError):
            ModelSpec(2560, 768, 1, 8)


if __name__ == "__main__":
    unittest.main()
