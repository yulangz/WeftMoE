"""Generated grids prune coupled resource conflicts; frozen plans remain strict."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import autotune_forward as fw
import autotune_backward as bw
import unified_tuning
from autotune_common import PlanError


class UnifiedSearchSpaceTest(unittest.TestCase):
    model = dict(hidden=2560, intermediate=1280, experts=32, top_k=8)

    def source(self, phase, topology="multi_node"):
        return json.loads((ROOT / f"configs/{phase}_h200_{topology}_autotune.json").read_text())

    def plan(self, phase, document=None, *, nodes=2):
        module = fw if phase == "forward" else bw
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(document if document is not None else self.source(phase)))
            return module.load_plan(path, model=self.model, system=dict(nodes=nodes))

    def frozen(self, phase):
        plan = self.plan(phase)
        plan["schema"] = 2 if phase == "forward" else 1
        plan["candidates"] = [dict(name="incumbent", override={})]
        plan["max_candidates"] = 1
        plan["selection"]["confirmation_candidates"] = 1
        plan.pop("search_space", None)
        plan.pop("candidate_filter", None)
        return plan

    def test_public_search_size_and_resource_profiles(self):
        for phase, module, raw_count, accepted in (("forward", fw, 144, 144),
                                                   ("backward", bw, 48, 36)):
            for topology in ("single_node", "multi_node"):
                with self.subTest(phase=phase, topology=topology):
                    source = self.source(phase, topology)
                    plan = self.plan(phase, source,
                                     nodes=1 if topology == "single_node" else 2)
                    candidates = module.expand_candidates(plan)
                    local = topology == "single_node"
                    expected_raw = (8 if phase == "forward" else 16) if local else raw_count
                    expected_accepted = (8 if phase == "forward" else 12) if local else accepted
                    self.assertEqual(plan["candidate_filter"]["generated"], expected_raw)
                    self.assertEqual(len(candidates), expected_accepted)
                    self.assertTrue(all(c["config"]["unified_workers"] == 1
                                        for c in candidates))
                    self.assertEqual({c["config"]["gin_ring_depth"] for c in candidates},
                                     {8} if local else {4, 8, 16})
                    if phase == "forward":
                        limits = [3] if topology == "single_node" else [3, 1, 2]
                        self.assertEqual(source["1_gemm"]["gin_channel_pack_limit"],
                                         limits)
                        self.assertEqual({c["config"]["gin_channel_pack_limit"]
                                          for c in candidates}, set(limits))
                    if phase == "backward":
                        self.assertNotIn("gin_channel_pack_limit", module.PARAMETER_MACROS)
                        self.assertEqual({(c["config"]["unified_math_registers"],
                                           c["config"]["unified_control_registers"])
                                          for c in candidates},
                                         {(224, 56), (216, 56), (216, 64)})
                        self.assertTrue(all("initial CTA pool 64512" in c["reason"]
                                            for c in plan["candidate_filter"]["rejected"]))

    def test_public_local_forward_gin_only_axes_are_singletons(self):
        gemm = self.source("forward", "single_node")["1_gemm"]
        for key in ("gin_channel_pack_limit",
                    "gin_ring_depth",
                    "unified_gin_context_probes", "unified_gin_pack_slots",
                    "unified_optimize_gin_local_transport"):
            with self.subTest(key=key):
                self.assertEqual(len(gemm[key]), 1)

    def test_custom_local_grid_can_still_override_gin_policies(self):
        source = self.source("forward", "single_node")
        source["1_gemm"].update(gin_channel_pack_limit=[8, 1])
        candidates = fw.expand_candidates(self.plan("forward", source, nodes=1))
        self.assertEqual(len(candidates), 16)
        self.assertEqual({c["config"]["gin_channel_pack_limit"]
                          for c in candidates}, {8, 1})

    def test_invalid_axis_value_is_not_filtered(self):
        source = self.source("backward")
        source["1_gemm"]["unified_control_registers"] = [63, 64]
        with self.assertRaisesRegex(PlanError, "multiple of eight"):
            self.plan("backward", source)


    def test_empty_legal_grid_fails_before_any_build(self):
        source = self.source("backward")
        source["1_gemm"]["unified_math_registers"] = [224]
        source["1_gemm"]["unified_control_registers"] = [64]
        with self.assertRaisesRegex(PlanError, "no legal resource combination"):
            self.plan("backward", source)

    def test_shared_memory_conflict_is_filtered_and_first_survivor_is_usable(self):
        source = self.source("backward")
        source["1_gemm"]["unified_segment_bytes"] = [65536, 4096]
        plan = self.plan("backward", source)
        candidates = bw.expand_candidates(plan)
        self.assertEqual(len(candidates), 36)
        self.assertEqual(candidates[0]["config"]["unified_segment_bytes"], 4096)
        self.assertTrue(any("shared memory" in c["reason"]
                            for c in plan["candidate_filter"]["rejected"]))

    def test_explicit_frozen_resource_error_remains_fatal(self):
        for phase, module in (("forward", fw), ("backward", bw)):
            with self.subTest(phase=phase):
                plan = self.frozen(phase)
                plan["fixed"]["1_gemm"].update(unified_math_registers=224,
                                                unified_control_registers=64)
                with self.assertRaisesRegex(PlanError, "initial CTA pool"):
                    module.expand_candidates(self.plan(phase, plan))

    def test_old_frozen_plan_upgrade_preserves_legacy_workers(self):
        for phase, module in (("forward", fw), ("backward", bw)):
            with self.subTest(phase=phase):
                plan = self.frozen(phase)
                for key in unified_tuning.DEFAULTS:
                    plan["fixed"]["1_gemm"].pop(key, None)
                plan["fixed"]["1_gemm"].pop("gin_channel_pack_limit", None)
                expanded = module.expand_candidates(self.plan(phase, plan))
                self.assertEqual(len(expanded), 1)
                for key, value in unified_tuning.DEFAULTS.items():
                    self.assertEqual(expanded[0]["config"][key], value)
                self.assertEqual(expanded[0]["config"]["unified_workers"], 0)
                if phase == "forward":
                    self.assertEqual(expanded[0]["config"]["gin_channel_pack_limit"], 3)

    def test_older_forward_grid_defaults_channel_pack_limit(self):
        source = self.source("forward")
        del source["1_gemm"]["gin_channel_pack_limit"]
        candidates = fw.expand_candidates(self.plan("forward", source))
        self.assertEqual(len(candidates), 48)
        self.assertEqual({c["config"]["gin_receive_rows"] for c in candidates},
                         {32, 128})
        self.assertTrue(all(c["config"]["gin_channel_pack_limit"] == 3
                            for c in candidates))

    def test_older_grids_keep_four_physical_ring_slots(self):
        for phase, module in (("forward", fw), ("backward", bw)):
            with self.subTest(phase=phase):
                source = self.source(phase)
                del source["1_gemm"]["gin_ring_depth"]
                candidates = module.expand_candidates(self.plan(phase, source))
                self.assertTrue(all(c["config"]["gin_ring_depth"] == 4
                                    for c in candidates))

    def test_physical_ring_depth_is_independent_of_pending_slots(self):
        for phase, module in (("forward", fw), ("backward", bw)):
            config = module.expand_candidates(self.plan(phase))[0]["config"]
            for depth in (4, 8, 16):
                for slots in (1, 3, 8):
                    with self.subTest(phase=phase, depth=depth, slots=slots):
                        module.validate_tuning(dict(config, gin_ring_depth=depth,
                                                    unified_gin_pack_slots=slots))

    def test_physical_ring_invalid_values_are_not_resource_filters(self):
        for phase in ("forward", "backward"):
            source = self.source(phase)
            for value in (0, 3, 5, 32, True):
                with self.subTest(phase=phase, value=value), self.assertRaisesRegex(
                        PlanError, "gin_ring_depth"):
                    source["1_gemm"]["gin_ring_depth"] = [value, 8]
                    self.plan(phase, source)

    def test_channel_pack_limit_does_not_require_matching_slot_capacity(self):
        config = fw.expand_candidates(self.plan("forward"))[0]["config"]
        for slots, limit in ((1, 8), (8, 1), (8, 2), (3, 3), (3, 8)):
            with self.subTest(slots=slots, limit=limit):
                fw.validate_tuning(dict(config, unified_gin_pack_slots=slots,
                                        gin_channel_pack_limit=limit))

    def test_channel_pack_limit_invalid_values_are_not_filtered(self):
        source = self.source("forward")
        for value in (0, 9, True):
            with self.subTest(value=value), self.assertRaisesRegex(
                    PlanError, "gin_channel_pack_limit"):
                source["1_gemm"]["gin_channel_pack_limit"] = [value, 3]
                self.plan("forward", source)


    def test_legacy_grid_is_not_pruned(self):
        source = self.source("backward")
        source["1_gemm"]["unified_workers"] = [0]
        # Even a legacy grid containing invalid inactive unified resource
        # fields keeps its former strict validation behavior.
        plan = self.plan("backward", source)
        self.assertNotIn("candidate_filter", plan)
        with self.assertRaisesRegex(PlanError, "initial CTA pool"):
            bw.expand_candidates(plan)


if __name__ == "__main__":
    unittest.main()
