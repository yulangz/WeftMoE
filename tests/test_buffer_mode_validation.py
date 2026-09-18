"""Require the native inference Graph and retained-gradient evidence per rank."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_local_activation_p2p as local


def evidence():
    return [dict(rank=str(rank), tokens="129", retained_capacity="1",
                 inference_graph_replays="6", skew="1", gradient_bits="all")
            for rank in range(8)]


class BufferModeValidationTest(unittest.TestCase):
    def test_current_native_evidence_accepts_any_rank_order(self):
        local.validate_buffer_mode_results(evidence()[::-1], 8)

    def test_incomplete_rank_coverage_is_rejected(self):
        for records in (evidence()[:-1], evidence() + [evidence()[0]],
                        evidence()[:-1] + [evidence()[0]]):
            with self.assertRaisesRegex(AssertionError, "incomplete"):
                local.validate_buffer_mode_results(records, 8)

    def test_partial_or_stale_evidence_is_rejected(self):
        for key in ("inference_graph_replays", "retained_capacity", "skew",
                    "gradient_bits"):
            for value in (None, "0"):
                with self.subTest(key=key, value=value):
                    records = evidence()
                    if value is None:
                        del records[3][key]
                    else:
                        records[3][key] = value
                    records[3]["graph_replays"] = "6"
                    with self.assertRaisesRegex(AssertionError, "incomplete"):
                        local.validate_buffer_mode_results(records, 8)


if __name__ == "__main__":
    unittest.main()
