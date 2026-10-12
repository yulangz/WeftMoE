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

import autotune_forward as tune  # noqa: E402
import forward_jit  # noqa: E402


def grid_count(topology):
    source = json.loads((ROOT / f"configs/forward_h200_{topology}_autotune.json").read_text())
    return math.prod(len(values) for group in source.values() for values in group.values())


class ForwardAutotuneTest(unittest.TestCase):


    def test_runtime_does_not_select_checkout_nccl_artifacts(self):
        from autotune_common import resolve_runtime_settings

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            private = root / "artifacts/gin_port_fix/build/lib"
            private.mkdir(parents=True)
            (private / "libnccl.so").touch()
            model = dict(hidden=2560, intermediate=768, experts=512, top_k=8)
            settings, _ = resolve_runtime_settings(
                model=model, system={"nodes": 2}, repo_root=root)
            self.assertNotIn("nccl_lib_dir", settings)
            self.assertNotIn(str(private), settings.get("runtime_library_path", ""))
            settings, _ = resolve_runtime_settings(
                model=model, system={"nodes": 2, "nccl_lib_dir": "/explicit/nccl"},
                repo_root=root)
            self.assertEqual(settings["nccl_lib_dir"], "/explicit/nccl")






    def test_local_runtime_library_path_precedes_inherited_libraries(self):
        benchmark = dict(self.plan["benchmark"], launcher="local", world=8,
                         nccl_lib_dir="/link/nccl",
                         runtime_library_path="/cuda/compat:/runtime/nccl")
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                tune.subprocess, "run", return_value=mock.Mock(
                    returncode=1, stdout="", stderr="")) as run:
            tune.benchmark_candidate(self.candidates[0], Path("/tmp/binary"),
                                     benchmark, Path(tmp),
                                     {"LD_LIBRARY_PATH": "/inherited"}, rounds=30)
            self.assertEqual(run.call_args.kwargs["env"]["LD_LIBRARY_PATH"],
                             "/cuda/compat:/runtime/nccl:/inherited")


    def test_grid_rejects_missing_duplicate_invalid_and_oversized_axes(self):
        source = json.loads((ROOT / "configs" /
                             "forward_h200_single_node_autotune.json").read_text())
        source = tune.make_runtime_plan(source, model=dict(
            hidden=7168, intermediate=3072, experts=384, top_k=6))
        cases = []
        model_in_search = copy.deepcopy(source)
        model_in_search["search_space"]["1_gemm"]["main_hidden"] = [2560, 7168]
        cases.append(model_in_search)
        missing_model = copy.deepcopy(source)
        del missing_model["model"]
        cases.append(missing_model)
        missing = copy.deepcopy(source)
        del missing["search_space"]["2_transport"]["rdma_ctas"]
        cases.append(missing)
        duplicate = copy.deepcopy(source)
        duplicate["search_space"]["2_transport"]["relay_ctas"] = [16, 16]
        cases.append(duplicate)
        invalid = copy.deepcopy(source)
        invalid["search_space"]["2_transport"]["local_dispatch_tma_warps"] = [4, 16]
        cases.append(invalid)
        oversized = copy.deepcopy(source)
        oversized["max_candidates"] = 1
        cases.append(oversized)
        for document in cases:
            with self.subTest(document=document), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "plan.json"
                path.write_text(json.dumps(document))
                with self.assertRaises(tune.PlanError):
                    tune.expand_candidates(tune.load_plan(path))

    def test_dynamic_model_and_optional_system_defaults(self):
        path = ROOT / "configs/forward_h200_single_node_autotune.json"
        with self.assertRaises(tune.PlanError):
            tune.load_plan(path)
        plan = tune.load_plan(path, model=dict(hidden=2560, intermediate=768,
                                              experts=512, top_k=8),
                              benchmark=dict(tokens_per_rank=65536))
        self.assertEqual(plan["fixed"]["1_gemm"]["main_hidden"], 2560)
        self.assertEqual(plan["benchmark"]["experts"], 512)
        self.assertTrue(plan["benchmark"]["disable_ib"])
        self.assertNotIn("socket_ifname", plan["benchmark"])
        self.assertEqual(plan["benchmark"]["tokens_per_rank"], 65536)
        with self.assertRaises(tune.PlanError):
            tune.load_plan(path, model=dict(hidden=2560, intermediate=768,
                                            experts=512, top_k=8), system={"world": 16})
        multi = tune.load_plan(ROOT / "configs/forward_h200_multi_node_autotune.json",
                              model=dict(hidden=2560, intermediate=768, experts=512, top_k=8),
                              system={"ep": 16, "remote": "root@example", "socket_ifname": "eth1"})
        self.assertTrue(multi["benchmark"]["enable_ib"])
        self.assertEqual(multi["benchmark"]["socket_ifname"], "eth1")

    def test_dynamic_experts_reach_benchmark_environment(self):
        benchmark = dict(self.plan["benchmark"], launcher="local", world=8, experts=512)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                tune.subprocess, "run", return_value=mock.Mock(returncode=1, stdout="")) as run:
            tune.benchmark_candidate(self.candidates[0], Path("/tmp/binary"),
                                     benchmark, Path(tmp), {}, rounds=30)
            self.assertEqual(run.call_args.kwargs["env"]["WEFTMOE_BENCH_EXPERTS"], "512")

    def setUp(self):
        self.plan = tune.load_plan(
            ROOT / "configs" / "forward_h200_multi_node_autotune.json",
            model=dict(hidden=7168, intermediate=3072, experts=384, top_k=6),
            system={"ep": 16, "remote": "root@example"})
        self.candidates = tune.expand_candidates(self.plan)

    def test_removed_reduce_divisor_legacy_migration(self):
        document = copy.deepcopy(self.plan)
        document["schema"] = 2
        document["max_candidates"] = 2
        document["candidates"] = document["candidates"][:2]
        document["fixed"]["4_completion_reduce"]["reduce_cta_divisor"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.json"
            path.write_text(json.dumps(document))
            migrated = tune.expand_candidates(tune.load_plan(path))
        self.assertEqual(migrated[0]["config"], self.candidates[0]["config"])
        config = dict(self.candidates[0]["config"], reduce_cta_divisor=2)
        with self.assertRaises(tune.PlanError):
            tune.validate_tuning(config)

    def test_removed_compute_return_path_is_rejected(self):
        config = dict(self.candidates[0]["config"], relay_local_return=False)
        with self.assertRaisesRegex(tune.PlanError, "relay_local_return=true"):
            tune.validate_tuning(config)


    def test_standalone_cutlass_path_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "cutlass/include"
            for relative in ("cute/arch/mma_sm90_gmma.hpp", "cutlass/arch/barrier.h"):
                file = include / relative
                file.parent.mkdir(parents=True, exist_ok=True)
                file.touch()
            self.assertEqual(forward_jit._cutlass_include(
                {}, {"CUTLASS_INCLUDE_DIR": str(include)}, root), include)
            self.assertEqual(forward_jit._cutlass_include(
                {"cutlass_include_dir": str(include)},
                {"CUTLASS_INCLUDE_DIR": str(root / "missing")}, root), include)
            with self.assertRaises(forward_jit.JitError):
                forward_jit._cutlass_include(
                    {"cutlass_include_dir": str(root / "missing")}, {}, root)

    def test_world8_plan_is_narrow_and_unique(self):
        plan = tune.load_plan(
            ROOT / "configs" / "forward_h200_single_node_autotune.json",
            model=dict(hidden=7168, intermediate=3072, experts=384, top_k=6))
        candidates = tune.expand_candidates(plan)
        self.assertEqual(len(candidates), grid_count("single_node"))
        selected = candidates[0]["config"]
        self.assertEqual(selected["rdma_ctas"], 0)
        self.assertEqual(selected["relay_ctas"], 24)
        source = json.loads((ROOT / "configs/forward_h200_single_node_autotune.json").read_text())
        self.assertEqual(selected["local_dispatch_tma_warps"],
                         source["2_transport"]["local_dispatch_tma_warps"][0])

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



    def test_generated_header_has_one_definition_per_parameter(self):
        candidate = self.candidates[0]
        header = tune.generated_header(candidate["name"],
                                       candidate["config"])
        self.assertIn("#define WEFTMOE_FWD_REDUCE_SCHEDULER 0", header)
        for macro in tune.PARAMETER_MACROS.values():
            self.assertEqual(header.count(f"#define {macro} "), 1)

    def test_generated_header_is_stable(self):
        candidate = self.candidates[0]
        first = tune.generated_header(candidate["name"], candidate["config"])
        second = tune.generated_header(candidate["name"], candidate["config"])
        self.assertEqual(first, second)
        self.assertEqual(
            first,
            tune.generated_header("renamed_candidate", candidate["config"]),
        )
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

    def test_jit_source_graph_is_complete(self):
        for source in (*forward_jit.HOST_SOURCES, *forward_jit.CUDA_SOURCES,
                       forward_jit.TEST_SOURCE):
            self.assertTrue((ROOT / source).is_file(), source)

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



if __name__ == "__main__":
    unittest.main()
