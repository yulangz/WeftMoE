"""Combined builds must preserve each phase's chosen worker resources."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "benchmarks"))
import autotune_forward as fw
import autotune_backward as bw
import unified_tuning
from run_performance_matrix import combined_generated_header


class UnifiedDefaultMacroTest(unittest.TestCase):
    def test_rdma_partition_keeps_a_relay_on_small_grids(self):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("a host C++ compiler is required")
        for requested in (0, 1, 3, 16, 2048):
            source = f"#define WEFTMOE_FWD_UNIFIED_RDMA_CTAS {requested}\n"
            source += '#include "cuda/forward/tuning_config.hpp"\n'
            source += "using weftmoe::cuda::forward_tuning::unified_rdma_ctas;\n"
            for total in (2, 8, 9, 16, 31, 132, 256):
                source += f"static_assert(unified_rdma_ctas({total}, false) == 0);\n"
                source += f"static_assert(unified_rdma_ctas({total}, true) == {min(requested, total - 1)});\n"
            result = subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++",
                 "-I", str(ROOT / "src"), "-"], input=source, text=True,
                capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)

    def preprocess(self, defines="", *, generated=None, generated_defines=""):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("a host C++ preprocessor is required for macro checks")
        keys = [f"WEFTMOE_{phase}_UNIFIED_{field}"
                for phase in ("FWD", "BWD")
                for field in ("WORKERS", "MATH_REGISTERS", "CONTROL_REGISTERS",
                              "SEGMENT_BYTES", "COPY_BUFFERS")]
        keys.append("WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT")
        keys.append("WEFTMOE_GIN_RING_DEPTH")
        results = []
        with tempfile.TemporaryDirectory() as temporary:
            prefix = defines
            if generated:
                phase = "FWD" if generated == "forward" else "BWD"
                header = (f"#define WEFTMOE_{phase}_MAIN_HIDDEN 7168\n"
                          f"#define WEFTMOE_{phase}_MAIN_INTERMEDIATE 3072\n"
                          + generated_defines)
                Path(temporary, f"weftmoe_{generated}_tuning_generated.hpp").write_text(header)
                prefix += f"#define WEFTMOE_{generated.upper()}_USE_GENERATED_TUNING 1\n"
            for order in (("forward", "backward"), ("backward", "forward")):
                source = prefix + "".join(
                    f'#include "cuda/{phase}/tuning_config.hpp"\n' for phase in order)
                # Expand real production macros without CUDA compilation,
                # host code generation, or an executable. String literals
                # preserve the labels while the second token is expanded.
                source += "".join(f'WEFTMOE_MACRO_VALUE "{key}" {key}\n' for key in keys)
                completed = subprocess.run(
                    [compiler, "-E", "-P", "-x", "c++", "-I", str(ROOT / "src"),
                     "-I", temporary, "-"], input=source, text=True,
                    capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                values = {}
                for line in completed.stdout.splitlines():
                    if line.startswith("WEFTMOE_MACRO_VALUE "):
                        _, key, value = line.split()
                        values[key.strip('"')] = int(value)
                self.assertEqual(set(values), set(keys))
                results.append(values)
        self.assertEqual(results[0], results[1], "phase include order changed defaults")
        return results[0]

    def test_bare_build_enables_both_phases_with_legal_default_resources(self):
        values = self.preprocess()
        self.assertEqual(values["WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT"], 3)
        for phase, buffers in (("FWD", 2), ("BWD", 1)):
            for field, expected in (("WORKERS", 1), ("MATH_REGISTERS", 216),
                                    ("CONTROL_REGISTERS", 64), ("SEGMENT_BYTES", 4096),
                                    ("COPY_BUFFERS", buffers)):
                self.assertEqual(values[f"WEFTMOE_{phase}_UNIFIED_{field}"], expected)

    def test_generated_headers_without_worker_switches_stay_legacy(self):
        for generated in ("forward", "backward"):
            with self.subTest(generated=generated):
                values = self.preprocess(generated=generated)
                self.assertEqual(values["WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT"], 3)
                for phase in ("FWD", "BWD"):
                    self.assertEqual(values[f"WEFTMOE_{phase}_UNIFIED_WORKERS"], 0)

    def test_standalone_generated_header_only_enables_its_explicit_phase(self):
        for generated, enabled in (("forward", "FWD"), ("backward", "BWD")):
            with self.subTest(generated=generated):
                values = self.preprocess(generated=generated, generated_defines=
                    f"#define WEFTMOE_{enabled}_UNIFIED_WORKERS 1\n")
                for phase in ("FWD", "BWD"):
                    self.assertEqual(values[f"WEFTMOE_{phase}_UNIFIED_WORKERS"],
                                     int(phase == enabled))

    def test_explicit_switches_override_bare_and_generated_fallbacks(self):
        for generated in (None, "forward", "backward"):
            for forward, backward in ((0, 0), (1, 1), (0, 1)):
                with self.subTest(generated=generated, workers=(forward, backward)):
                    values = self.preprocess(
                        f"#define WEFTMOE_FWD_UNIFIED_WORKERS {forward}\n"
                        f"#define WEFTMOE_BWD_UNIFIED_WORKERS {backward}\n",
                        generated=generated)
                    self.assertEqual(values["WEFTMOE_FWD_UNIFIED_WORKERS"], forward)
                    self.assertEqual(values["WEFTMOE_BWD_UNIFIED_WORKERS"], backward)


    def test_physical_ring_default_and_historical_headers(self):
        self.assertEqual(self.preprocess()["WEFTMOE_GIN_RING_DEPTH"], 8)
        for phase in ("forward", "backward"):
            with self.subTest(phase=phase):
                self.assertEqual(self.preprocess(generated=phase)[
                    "WEFTMOE_GIN_RING_DEPTH"], 4)
                macro = "FWD" if phase == "forward" else "BWD"
                self.assertEqual(self.preprocess(
                    f"#define WEFTMOE_{macro}_MAIN_HIDDEN 7168\n")[
                        "WEFTMOE_GIN_RING_DEPTH"], 4)
                for depth in (4, 8, 16):
                    defines = f"#define WEFTMOE_GIN_RING_DEPTH {depth}\n"
                    self.assertEqual(self.preprocess(defines, generated=phase)[
                        "WEFTMOE_GIN_RING_DEPTH"], depth)
                    self.assertEqual(self.preprocess(generated=phase,
                        generated_defines=defines)["WEFTMOE_GIN_RING_DEPTH"], depth)

    def test_channel_pack_limit_explicit_and_generated_overrides(self):
        for value in (1, 2, 8):
            for generated in (None, "forward", "backward"):
                with self.subTest(value=value, generated=generated):
                    values = self.preprocess(
                        f"#define WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT {value}\n",
                        generated=generated)
                    self.assertEqual(values["WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT"], value)
            values = self.preprocess(generated="forward", generated_defines=
                f"#define WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT {value}\n")
            self.assertEqual(values["WEFTMOE_FWD_GIN_CHANNEL_PACK_LIMIT"], value)

    def test_shared_buffer_override_precedes_fallback_but_phase_override_wins(self):
        cases = [
            ({"WEFTMOE_UNIFIED_COPY_BUFFERS": 1}, (1, 1)),
            ({"WEFTMOE_UNIFIED_COPY_BUFFERS": 2}, (2, 2)),
            ({"WEFTMOE_UNIFIED_COPY_BUFFERS": 2,
              "WEFTMOE_BWD_UNIFIED_COPY_BUFFERS": 1}, (2, 1)),
            ({"WEFTMOE_UNIFIED_COPY_BUFFERS": 1,
              "WEFTMOE_FWD_UNIFIED_COPY_BUFFERS": 2}, (2, 1)),
            ({"WEFTMOE_FWD_UNIFIED_COPY_BUFFERS": 1,
              "WEFTMOE_BWD_UNIFIED_COPY_BUFFERS": 2}, (1, 2)),
        ]
        for defines, expected in cases:
            with self.subTest(defines=defines):
                values = self.preprocess("".join(
                    f"#define {key} {value}\n" for key, value in defines.items()))
                self.assertEqual((values["WEFTMOE_FWD_UNIFIED_COPY_BUFFERS"],
                                  values["WEFTMOE_BWD_UNIFIED_COPY_BUFFERS"]), expected)


class UnifiedPhaseResourcesTest(unittest.TestCase):
    def configs(self):
        result = []
        for phase, module in (("forward", fw), ("backward", bw)):
            plan = module.load_plan(
                ROOT / f"configs/{phase}_h200_multi_node_autotune.json",
                model=dict(hidden=7168, intermediate=3072, experts=384, top_k=6),
                system=dict(nodes=2, remote="root@example"))
            result.append(dict(module.expand_candidates(plan)[0]["config"],
                               unified_workers=1, wgmma_n256_atom_n=128,
                               unified_gin_pack_slots=3))
        forward, backward = result
        forward.update(fc1_stages=2, fc2_stages=2, fc1_tile_n=128, fc2_tile_n=128,
                       unified_threads=512, unified_comm_threads=128,
                       unified_comm_base=384, unified_math_registers=208,
                       unified_control_registers=64, unified_producer_registers=32,
                       unified_segment_bytes=4096, unified_copy_buffers=2)
        backward.update(unified_threads=384, unified_comm_threads=96,
                        unified_comm_base=288, unified_math_registers=216,
                        unified_control_registers=64, unified_producer_registers=0,
                        unified_segment_bytes=2048, unified_copy_buffers=1)
        return forward, backward

    def syntax_check(self, header, expectations, *, generated=True):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("a host C++ compiler is required for header checks")
        # No CUDA compilation or executable: parse only the two pure tuning
        # headers and static_assert their resolved values in both include orders.
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "weftmoe_forward_tuning_generated.hpp").write_text(header)
            for order in (("forward", "backward"), ("backward", "forward")):
                source = ("#define WEFTMOE_FORWARD_USE_GENERATED_TUNING 1\n"
                          if generated else header)
                source += "".join(f'#include "cuda/{phase}/tuning_config.hpp"\n'
                                  for phase in order)
                source += "".join(
                    f"static_assert(weftmoe::cuda::{phase}_tuning::{name} == {value});\n"
                    for phase, name, value in expectations)
                completed = subprocess.run(
                    [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++",
                     "-I", str(ROOT / "src"), "-I", tmp, "-"],
                    input=source, text=True, capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_combined_header_resolves_independent_resources_in_both_orders(self):
        forward, backward = self.configs()
        header = combined_generated_header("different_resources", forward, backward)
        self.syntax_check(header, [
            ("forward", "kUnifiedThreads", 512),
            ("forward", "kUnifiedCommBase", 384),
            ("forward", "kUnifiedCommThreads", 128),
            ("forward", "kUnifiedMathRegisters", 208),
            ("forward", "kUnifiedProducerRegisters", 32),
            ("forward", "kUnifiedSegmentBytes", 4096),
            ("forward", "kUnifiedCopyBuffers", 2),
            ("backward", "kUnifiedThreads", 384),
            ("backward", "kUnifiedCommBase", 288),
            ("backward", "kUnifiedCommThreads", 96),
            ("backward", "kUnifiedMathRegisters", 216),
            ("backward", "kUnifiedProducerRegisters", 64),
            ("backward", "kUnifiedSegmentBytes", 2048),
            ("backward", "kUnifiedCopyBuffers", 1)])

    def test_shared_protocol_or_atom_mismatch_is_rejected(self):
        forward, backward = self.configs()
        for key, other in (("unified_gin_pack_slots", 1),
                           ("gin_ring_depth", 16),
                           ("unified_gin_context_probes", 4),
                           ("wgmma_n256_atom_n", 256)):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                combined_generated_header("mismatch", forward, dict(backward, **{key: other}))


    def test_old_combined_build_records_keep_four_physical_slots(self):
        forward, backward = self.configs()
        forward.pop("gin_ring_depth")
        backward.pop("gin_ring_depth")
        header = combined_generated_header("historical", forward, backward)
        self.assertEqual(header.count("#define WEFTMOE_GIN_RING_DEPTH 4\n"), 2)
        self.assertNotIn("gin_ring_depth", forward)
        self.assertNotIn("gin_ring_depth", backward)
        with self.assertRaisesRegex(ValueError, "gin_ring_depth"):
            combined_generated_header("mixed", forward,
                                      dict(backward, gin_ring_depth=8))


    def test_legacy_shared_macros_remain_defaults_with_phase_override(self):
        header = """#define WEFTMOE_UNIFIED_MATH_REGISTERS 216
#define WEFTMOE_UNIFIED_CONTROL_REGISTERS 64
#define WEFTMOE_UNIFIED_COPY_BUFFERS 2
#define WEFTMOE_BWD_UNIFIED_MATH_REGISTERS 208
#define WEFTMOE_BWD_UNIFIED_CONTROL_REGISTERS 80
#define WEFTMOE_BWD_UNIFIED_COPY_BUFFERS 1
"""
        self.syntax_check(header, [
            ("forward", "kUnifiedMathRegisters", 216),
            ("forward", "kUnifiedProducerRegisters", 64),
            ("forward", "kUnifiedCopyBuffers", 2),
            ("backward", "kUnifiedMathRegisters", 208),
            ("backward", "kUnifiedProducerRegisters", 80),
            ("backward", "kUnifiedCopyBuffers", 1)], generated=False)

    def test_only_declared_shared_keys_have_identical_macro_names(self):
        forward = unified_tuning.parameter_macros("forward")
        backward = unified_tuning.parameter_macros("backward")
        self.assertEqual({key for key in forward if forward[key] == backward[key]},
                         unified_tuning.SHARED_PARAMETERS)

    def test_backward_staging_covers_every_reducer_warp(self):
        _, backward = self.configs()
        with self.assertRaisesRegex(ValueError, "pointer tables"):
            bw.validate_tuning(dict(backward, unified_segment_bytes=256))
        bw.validate_tuning(dict(backward, unified_segment_bytes=384))


if __name__ == "__main__":
    unittest.main()
