"""Physical ring configuration must resolve before host/device layout headers."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GinRingTuningTest(unittest.TestCase):
    def compile_header(self, source, *, headers=None):
        compiler = shutil.which("c++")
        if not compiler:
            self.skipTest("a host C++ compiler is required for ring layout checks")
        with tempfile.TemporaryDirectory() as temporary:
            for name, content in (headers or {}).items():
                Path(temporary, name).write_text(content)
            return subprocess.run(
                [compiler, "-std=c++17", "-fsyntax-only", "-x", "c++",
                 "-I", str(ROOT / "src"), "-I", temporary, "-"],
                input=source, text=True, capture_output=True, timeout=10)

    def test_wire_header_can_precede_phase_tuning(self):
        for phase in ("forward", "backward"):
            for generated in (False, True):
                for depth in (None, 4, 8, 16):
                    expected = depth if depth is not None else (4 if generated else 8)
                    prefix = (f"#define WEFTMOE_GIN_RING_DEPTH {depth}\n"
                              if depth is not None else "")
                    if generated:
                        prefix += f"#define WEFTMOE_{phase.upper()}_USE_GENERATED_TUNING 1\n"
                    macro = "FWD" if phase == "forward" else "BWD"
                    headers = {f"weftmoe_{phase}_tuning_generated.hpp":
                               f"#define WEFTMOE_{macro}_MAIN_HIDDEN 7168\n"
                               f"#define WEFTMOE_{macro}_MAIN_INTERMEDIATE 3072\n"}
                    source = prefix + '#include "cuda/communication/tuning_config.hpp"\n'
                    source += f"constexpr unsigned before_phase = WEFTMOE_GIN_RING_DEPTH;\n"
                    source += f'#include "cuda/{phase}/tuning_config.hpp"\n'
                    source += (f"static_assert(before_phase == {expected});\n"
                               f"static_assert(WEFTMOE_GIN_RING_DEPTH == before_phase);\n")
                    with self.subTest(phase=phase, generated=generated, depth=depth):
                        result = self.compile_header(source, headers=headers)
                        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unsupported_physical_depths_fail_compilation(self):
        for depth in (0, 3, 5, 32):
            with self.subTest(depth=depth):
                result = self.compile_header(
                    f"#define WEFTMOE_GIN_RING_DEPTH {depth}\n"
                    '#include "cuda/communication/tuning_config.hpp"\n')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("GIN physical ring depth must be 4, 8 or 16", result.stderr)



if __name__ == "__main__":
    unittest.main()
