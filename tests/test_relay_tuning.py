"""Relay tuning must not be restricted to previously benchmarked counts."""

import importlib
import json
import sys
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


class RelayTuningTest(unittest.TestCase):
    def config(self, phase, topology):
        space = json.loads((ROOT / "configs" /
                            f"{phase}_h200_{topology}_autotune.json").read_text())
        tune = importlib.import_module(f"autotune_{phase}")
        config = {key: values[0] for key, values in
                  tune.flatten_groups(space, require_all=False).items()}
        config.update(main_hidden=2560, main_intermediate=768)
        return config

    def test_larger_relay_counts_generate_for_both_topologies(self):
        for phase in ("forward", "backward"):
            tune = importlib.import_module(f"autotune_{phase}")
            for topology in ("single_node", "multi_node"):
                for relay in (32, 40, 48, 64, 96):
                    with self.subTest(phase=phase, topology=topology, relay=relay):
                        config = self.config(phase, topology)
                        config["relay_ctas"] = relay
                        tune.validate_tuning(config)
                        header = tune.generated_header("large_relay", config)
                        macro = tune.PARAMETER_MACROS["relay_ctas"]
                        self.assertIn(f"#define {macro} {relay}", header)

    def test_return_cursors_cover_all_local_world_sizes(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        source = r'''#include "cuda/persistent_v3/forward/pipeline_state.cuh"
using namespace weftmoe::cuda;
constexpr bool cursors_fit() {
  constexpr auto capacity = sizeof(persistent_v3::ForwardPipelineControl::local_return_head)
      / sizeof(persistent_v3::PaddedCounter);
  for (unsigned int world = 1; world <= persistent_v3::kRdmaCtaCount; ++world) {
    const auto stripes = (forward_tuning::kRelayCtas + world - 1) / world;
    if (world * stripes > capacity) return false;
  }
  return true;
}
static_assert(cursors_fit(), "Return cursor capacity must follow Relay count");
'''
        for relay in (40, 48, 96):
            with self.subTest(relay=relay), tempfile.TemporaryDirectory() as tmp:
                result = subprocess.run(
                    [compiler, "-std=c++17", "-x", "c++", "-c", "-",
                     "-I", str(ROOT / "src"), "-I", str(ROOT / "include"),
                     f"-DWEFTMOE_FWD_RELAY_CTAS={relay}",
                     "-o", str(Path(tmp) / "layout.o")],
                    input=source, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_topology_alignment_is_still_required(self):
        for phase, topology, relay in (
            ("forward", "single_node", 0),
            ("forward", "single_node", 6),
            ("forward", "multi_node", 4),
            ("forward", "multi_node", 36),
            ("backward", "single_node", 4),
            ("backward", "multi_node", 34),
        ):
            with self.subTest(phase=phase, topology=topology, relay=relay):
                tune = importlib.import_module(f"autotune_{phase}")
                config = self.config(phase, topology)
                config["relay_ctas"] = relay
                with self.assertRaises(tune.PlanError):
                    tune.validate_tuning(config)


class ForwardTransportNamingTest(unittest.TestCase):
    def test_new_name_preserves_candidates_headers_and_cache_digests(self):
        tune = importlib.import_module("autotune_forward")
        for topology in ("single_node", "multi_node"):
            space = json.loads((ROOT / "configs" /
                f"forward_h200_{topology}_autotune.json").read_text())
            space["2_transport"]["rdma_ctas"] = [
                0 if topology == "single_node" else 8]
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "config.json"
                path.write_text(json.dumps(space))
                model = dict(hidden=2560, intermediate=768, experts=512, top_k=8)
                new = tune.expand_candidates(tune.load_plan(path, model=model, system={"nodes": 1 if topology == "single_node" else 2}))
                path.write_text(json.dumps(space).replace(
                    '"local_return_tma_warps"', '"relay_tma_warps"'))
                old = tune.expand_candidates(tune.load_plan(path, model=model, system={"nodes": 1 if topology == "single_node" else 2}))
                self.assertEqual(new, old)
                for a, b in zip(new, old):
                    self.assertEqual(tune.generated_header(a["name"], a["config"]),
                                     tune.generated_header(b["name"], b["config"]))
                    self.assertEqual(tune.candidate_digest(a["config"]),
                                     tune.candidate_digest(b["config"]))

    def test_mixed_aliases_are_rejected(self):
        tune = importlib.import_module("autotune_forward")
        with self.assertRaises(tune.PlanError):
            tune.flatten_groups({"2_transport": {
                "local_return_tma_warps": [8], "relay_tma_warps": [0]}},
                require_all=False)


if __name__ == "__main__":
    unittest.main()
