import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import autotune_forward
import autotune_backward


class AutotuneDefaultsTest(unittest.TestCase):
    def test_rounds_and_global_compile_budget(self):
        model = dict(hidden=4096, intermediate=1536, experts=64, top_k=4)
        for phase, module in [('forward', autotune_forward), ('backward', autotune_backward)]:
            for topology, nodes in [('single_node', 1), ('multi_node', 2)]:
                with self.subTest(phase=phase, topology=topology):
                    space = json.loads((ROOT / f'configs/{phase}_h200_{topology}_autotune.json').read_text())
                    plan = module.make_runtime_plan(space, model=model, system=dict(nodes=nodes, gpus_per_node=8))
                    self.assertEqual(plan['jit']['jobs'], 16)
                    self.assertEqual(plan['benchmark']['tune_rounds'] - 10, 15)
                    self.assertEqual(plan['selection']['confirmation_rounds'] - 10, 30)
                    self.assertEqual(plan['benchmark']['final_rounds'] - 10, 30)


if __name__ == '__main__':
    unittest.main()
