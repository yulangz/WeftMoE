#!/usr/bin/env python3

import copy
import json
import math
import re
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import autotune_backward as tune  # noqa: E402
import forward_jit  # noqa: E402


def grid_count(topology):
    source = json.loads((ROOT / f"configs/backward_h200_{topology}_autotune.json").read_text())
    return math.prod(len(values) for group in source.values() for values in group.values())


class BackwardAutotuneTest(unittest.TestCase):
    def test_resource_inspection_selects_active_kernel_family(self):
        unified_symbols = ["_unified_backward_kernel_local0",
                           "_unified_backward_kernel_local1"]
        split_symbols = ["_backward_local_compute_kernel",
                         "_backward_local_transport_kernel"]
        for unified, symbols, valid, expected in (
            (1, unified_symbols + split_symbols, True, {"unified": 2}),
            (1, split_symbols, False, {"unified": 0}),
            (0, unified_symbols + split_symbols, True, {"compute": 1, "transport": 1}),
            (0, unified_symbols + split_symbols[:1], False, {"compute": 1, "transport": 0}),
        ):
            fixture = "".join(
                f" Function {symbol}:\n REG:168 STACK:504 SHARED:2048 LOCAL:0\n"
                for symbol in symbols)
            with self.subTest(unified=unified, symbols=symbols), \
                    tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(tune.shutil, "which", return_value="cuobjdump"), \
                    mock.patch.object(tune.subprocess, "run", return_value=mock.Mock(
                        returncode=0, stdout=fixture)):
                result = tune.inspect_backward_kernels(
                    Path("binary"), Path(tmp) / "resources.log",
                    dict(self.candidates[0]["config"], unified_workers=unified))
            self.assertEqual(result["valid_compiled_kernel_set"], valid)
            self.assertEqual({key: len(entries) for key, entries in result["kernels"].items()},
                             expected)

    def setUp(self):
        self.plan = tune.load_plan(
            ROOT / "configs" / "backward_h200_multi_node_autotune.json",
            model=dict(hidden=7168, intermediate=3072, experts=384, top_k=6),
            system={"ep": 16, "remote": "root@example"})
        self.candidates = tune.expand_candidates(self.plan)

    def test_multinode_plan_is_narrow_complete_and_unique(self):
        self.assertEqual(len(self.candidates) + len(self.plan.get("candidate_filter", {}).get("rejected", [])),
                         grid_count("multi_node"))
        self.assertLessEqual(len(self.candidates),
                             self.plan["max_candidates"])
        self.assertEqual(self.plan["invariants"], tune.EXPECTED_INVARIANTS)
        for candidate in self.candidates:
            self.assertEqual(set(candidate["config"]),
                             set(tune.PARAMETER_MACROS))

    def test_world8_plan_uses_unified_workers(self):
        plan = tune.load_plan(
            ROOT / "configs" / "backward_h200_single_node_autotune.json",
            model=dict(hidden=7168, intermediate=3072, experts=384, top_k=6))
        candidates = tune.expand_candidates(plan)
        self.assertEqual(len(candidates) + len(plan.get("candidate_filter", {}).get("rejected", [])),
                         grid_count("single_node"))
        self.assertEqual(plan["benchmark"]["launcher"], "local")
        self.assertEqual(candidates[0]["config"]["relay_ctas"], 32)
        self.assertEqual(candidates[0]["config"]["rdma_ctas"], 0)
        self.assertEqual(candidates[0]["config"]["unified_workers"], 1)
        self.assertEqual(candidates[0]["config"]["dispatch_batch_rows"], 64)
        self.assertEqual(candidates[0]["config"]["dw2_claim_tiles"], 4)
        self.assertEqual(candidates[0]["config"]["dw1_claim_tiles"], 4)
        self.assertEqual(candidates[0]["config"]["ready_word_stagger"], 0)
        self.assertEqual(candidates[0]["config"]["activation_owner_group"], 4)
        self.assertEqual({c["config"]["dispatch_batch_rows"] for c in candidates}, {32, 64})

    def test_dynamic_model_and_system_defaults(self):
        path = ROOT / "configs/backward_h200_single_node_autotune.json"
        model = dict(hidden=2560, intermediate=768, experts=512, top_k=8)
        with self.assertRaises(tune.PlanError):
            tune.load_plan(path)
        plan = tune.load_plan(path, model=model, benchmark={"tokens_per_rank": 65536})
        self.assertEqual(plan["fixed"]["1_gemm"]["main_hidden"], 2560)
        self.assertEqual(plan["benchmark"]["experts"], 512)
        self.assertTrue(plan["benchmark"]["disable_ib"])
        self.assertNotIn("socket_ifname", plan["benchmark"])
        with self.assertRaises(tune.PlanError):
            tune.load_plan(path, model=model, system={"world": 16})
        with mock.patch.object(tune, "_run", return_value={"returncode": 1, "output": ""}) as run:
            tune.benchmark_candidate(tune.expand_candidates(plan)[0], Path("/tmp/binary"),
                                     plan["benchmark"], Path("/tmp"), {}, rounds=30)
        self.assertEqual(run.call_args.args[1]["WEFTMOE_BENCH_EXPERTS"], "512")

    def test_grid_rejects_model_fields_duplicates_and_missing_parameters(self):
        source = json.loads((ROOT / "configs/backward_h200_single_node_autotune.json").read_text())
        invalid = []
        model_axis = copy.deepcopy(source)
        model_axis["1_gemm"]["main_hidden"] = [2560, 7168]
        invalid.append(model_axis)
        duplicate = copy.deepcopy(source)
        duplicate["2_transport"]["relay_ctas"] = [12, 12]
        invalid.append(duplicate)
        missing = copy.deepcopy(source)
        del missing["3_scheduler"]["dw1_claim_tiles"]
        invalid.append(missing)
        for space in invalid:
            with self.subTest(space=space), self.assertRaises(tune.PlanError):
                tune.make_runtime_plan(space, model=dict(hidden=2560, intermediate=768, experts=512, top_k=8))

    def test_cpp_defaults_and_generator_have_same_parameter_surface(self):
        header = (ROOT / "src" / "cuda" / "backward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        shared = (ROOT / "src/cuda/unified_worker_tuning.hpp").read_text()
        atom = (ROOT / "src/cuda/primitives/hopper.cuh").read_text()
        ring = (ROOT / "src/cuda/communication/tuning_config.hpp").read_text()
        common_macros = set(re.findall(
            r"^#ifndef (WEFTMOE_UNIFIED_GIN_(?:CONTEXT_PROBES|PACK_SLOTS)|WEFTMOE_WGMMA_N256_ATOM_N|WEFTMOE_GIN_[A-Z0-9_]+)$",
            shared + "\n" + atom + "\n" + ring, flags=re.MULTILINE))
        cpp_macros = set(re.findall(
            r"^#ifndef (WEFTMOE_BWD_[A-Z0-9_]+)$", header + shared,
            flags=re.MULTILINE))
        self.assertEqual(cpp_macros | common_macros, set(tune.PARAMETER_MACROS.values()))

    def test_cpp_defaults_are_valid_independently_of_search_incumbent(self):
        header = (ROOT / "src" / "cuda" / "backward" /
                  "tuning_config.hpp").read_text(encoding="utf-8")
        header += (ROOT / "src/cuda/forward/tuning_config.hpp").read_text()
        header += (ROOT / "src/cuda/primitives/hopper.cuh").read_text()
        header += (ROOT / "src/cuda/unified_worker_tuning.hpp").read_text()
        header += (ROOT / "src/cuda/communication/tuning_config.hpp").read_text()
        defaults = dict(re.findall(
            r"^#define (WEFTMOE_(?:BWD_[A-Z0-9_]+|UNIFIED_[A-Z0-9_]+|WGMMA_N256_ATOM_N|GIN_[A-Z0-9_]+)) ([A-Za-z0-9_]+)$", header,
            flags=re.MULTILINE))
        # Search order may change without changing the untuned C++ fallback.
        config = {}
        for key, macro in tune.PARAMETER_MACROS.items():
            value = defaults[macro]
            while not value.isdecimal():
                value = defaults[value]
            config[key] = int(value)
        tune.validate_tuning(config)

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
                             "weftmoe_backward_tuning_generated.hpp")

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
