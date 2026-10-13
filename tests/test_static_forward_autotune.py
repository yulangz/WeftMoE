"""V4 tuning contracts: actual parameter wiring and safe result publication."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune_forward as tune
import static_forward_tuning as v4
from weftmoe.tuning import select_profile


class StaticForwardAutotuneTest(unittest.TestCase):
    def plan(self, kind="decode", nodes=1):
        return tune.load_plan(ROOT / f"configs/forward_v4_{kind}_autotune.json",
                              model=dict(hidden=2048, intermediate=512, experts=256, top_k=8),
                              benchmark=dict(tokens_per_rank=64),
                              system=dict(nodes=nodes, gpus_per_node=4 if nodes == 1 else 8))

    def test_joint_axes_reach_generated_code(self):
        for kind, nodes in (("decode", 1), ("ring", 2), ("ring", 4)):
            plan = self.plan(kind, nodes)
            candidates = tune.expand_candidates(plan)
            self.assertGreater(len(candidates), 1)
            self.assertEqual(plan["selection"]["confirmation_candidates"], 2)
            for candidate in candidates:
                config = candidate["config"]
                header = tune.generated_header(candidate["name"], config)
                for key, macro in v4.MACROS.items():
                    self.assertIn(f"#define {macro} {config['v4'][key]}\n", header)
                self.assertIn("#define WEFTMOE_FWD_TILE_M 128\n", header)
            if kind == "decode":
                self.assertEqual({c["config"]["v4"]["decode_math_m"] for c in candidates}, {0, 16, 64})
            else:
                self.assertEqual({c["config"]["v4"]["decode_math_m"] for c in candidates}, {0})

    def test_unsafe_network_layouts_are_rejected(self):
        with self.assertRaisesRegex(tune.PlanError, "network ring implements M128"):
            tune.expand_candidates(self.plan(nodes=2))
        config = tune.expand_candidates(self.plan(kind="ring", nodes=2))[0]["config"]
        config["v4"].update(backward_striped_signals=1, network_lanes=32)
        with self.assertRaisesRegex(tune.PlanError, "one channel lane per Put worker"):
            v4.validate(config)
        config["v4"]["network_lanes"] = 64
        v4.validate(config)

    def test_route_suite_rejects_memory_and_batch_dependence(self):
        spec = dict(memory_limit_bytes=1024, cases=[dict(route="balanced", seed=0),
                    dict(route="balanced", seed=0, validate_only=True)])
        record = dict(static_schedule=1, ring_window_tokens=256, measured=20,
                      slowest_rank_us=dict(mean=10.0), per_rank_device_bytes=[100],
                      slowest_rank_samples_us=[10.0] * 20, per_rank_prefix_hash=["abc"])
        runs = [dict(returncode=0, record=copy.deepcopy(record), log="fixture") for _ in range(2)]
        self.assertEqual(v4.aggregate(runs, spec)["forward_slowest_rank_us"]["mean"], 10)
        runs[1]["record"]["per_rank_prefix_hash"] = ["different"]
        with self.assertRaisesRegex(tune.PlanError, "batch invariance"):
            v4.aggregate(runs, spec)
        runs[1]["record"] = dict(record, per_rank_device_bytes=[1025])
        with self.assertRaisesRegex(tune.PlanError, "memory budget"):
            v4.aggregate(runs, spec)

    def test_table_requires_exact_shape_and_compiled_protocol(self):
        plan = self.plan()
        candidate = tune.expand_candidates(plan)[0]
        result = dict(benchmark=plan["benchmark"], selected=dict(
            config=candidate["config"], final_metric_us=10))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "table.json"
            v4.export_table(path, result, Path(tmp) / "profile.hpp")
            entry = json.loads(path.read_text())["entries"][0]
            key, digest = entry["key"], entry["compiled_digest"]
            self.assertEqual(select_profile(path, key, digest)["window_tokens"], 256)
            with self.assertRaisesRegex(ValueError, "requires a native rebuild"):
                select_profile(path, key, "stale")
            with self.assertRaisesRegex(ValueError, "no validated"):
                select_profile(path, dict(key, max_tokens=8192), digest)

    def test_search_confirms_and_exports_only_matching_outputs(self):
        def builds(candidates, *_):
            return {c["name"]: dict(returncode=0, binary="/test-only/binary", log="build")
                    for c in candidates}

        def measure(candidate, *_, **kwargs):
            name = candidate["name"]
            # The fastest configuration is wrong; it must never be published.
            latency = {"incumbent": 10, "v4_0001": 8, "v4_0002": 1}[name]
            case = dict(shape={}, per_rank_input_hash=["input"], per_rank_tokens=[64],
                        per_rank_output_hash=["wrong" if name == "v4_0002" else "correct"])
            return dict(returncode=0, log="test-only", elapsed_seconds=0, record=dict(
                benchmark="static-v4-route-suite", measured=kwargs["rounds"] - 10,
                cases=[case], forward_slowest_rank_us=dict(mean=latency, p95=latency)))

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(tune, "ForwardJitCompiler"), \
                mock.patch.object(tune, "compile_candidates", side_effect=builds), \
                mock.patch.object(tune, "benchmark_candidate", side_effect=measure):
            result, table = Path(tmp) / "result.json", Path(tmp) / "table.json"
            code = tune.main([
                "--config", str(ROOT / "configs/forward_v4_decode_autotune.json"),
                "--nodes", "1", "--gpus-per-node", "4", "--hidden", "2048",
                "--intermediate", "512", "--experts", "256", "--top-k", "8",
                "--tokens", "64", "--exact-tokens", "--candidate", "incumbent",
                "--candidate", "v4_0001", "--candidate", "v4_0002",
                "--output", str(result), "--export-table", str(table)])
            self.assertEqual(code, 0)
            selected = json.loads(result.read_text())["selected"]
            self.assertEqual(selected["name"], "v4_0001")
            self.assertTrue(Path(selected["header"]).is_file())
            self.assertEqual(json.loads(table.read_text())["entries"][0]["metric_us"], 8)


if __name__ == "__main__":
    unittest.main()
