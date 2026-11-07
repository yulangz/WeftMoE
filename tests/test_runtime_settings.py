"""Model dimensions and token bucket boundaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from weftmoe import ModelSpec, token_bucket


class PreparationTests(unittest.TestCase):
    def test_bucket_boundaries_and_overflow(self):
        for tokens, representative in [(0, 64), (1, 64), (64, 64), (65, 8192),
                                       (8192, 8192), (8193, 20480), (20480, 20480), (20481, 65536), (65536, 65536),
                                       (65537, 131072), (262144, 262144)]:
            self.assertEqual(token_bucket(tokens).tune_tokens, representative)
        for tokens in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                token_bucket(tokens)

    def test_model_is_separate_from_tuning(self):
        self.assertEqual(ModelSpec(2560, 768, 512, 8).top_k, 8)
        with self.assertRaises(ValueError):
            ModelSpec(2560, 768, 1, 8)


if __name__ == "__main__":
    unittest.main()
