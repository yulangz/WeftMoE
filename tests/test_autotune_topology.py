"""Topology must be caller-owned and independent of the tuning grid."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from autotune_common import PlanError, resolve_parallel_settings

MODEL = dict(hidden=2560, intermediate=768, experts=512, top_k=8)


class AutotuneTopologyTest(unittest.TestCase):
    def test_parallel_resolution(self):
        for settings, nodes, ep in [({}, 1, 8), ({'ep': 32}, 4, 32),
                                    ({'nodes': 8}, 8, 64),
                                    ({'nodes': 3, 'gpus_per_node': 8}, 3, 24)]:
            with self.subTest(settings=settings):
                self.assertEqual(resolve_parallel_settings(settings),
                                 dict(nodes=nodes, ep=ep, gpus_per_node=8))
        for settings in ({'ep': 15}, {'nodes': 4, 'ep': 16}, {'nodes': 0},
                         {'nodes': True}, {'gpus_per_node': 0}):
            with self.subTest(settings=settings), self.assertRaises(PlanError):
                resolve_parallel_settings(settings)

    def test_topology_controls_candidates_and_execution(self):
        for phase in ('forward', 'backward'):
            tune = importlib.import_module('autotune_' + phase)
            source = json.loads((ROOT / f'configs/{phase}_h200_single_node_autotune.json').read_text())
            # A small valid grid whose RDMA values deliberately contradict local topology.
            for group in source.values():
                for key, values in group.items():
                    group[key] = values[:1]
            source['2_transport']['relay_ctas'] = [16]
            source['2_transport']['rdma_ctas'] = [8, 0, 12]
            if phase == 'backward':
                source['1_gemm']['dab_tile_n'] = [128]
            original = copy.deepcopy(source)
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'misleading_multi_node_name.json'
                path.write_text(json.dumps(source))
                local = tune.load_plan(path, model=MODEL, system={'nodes': 1})
                candidates = tune.expand_candidates(local)
                self.assertEqual(len(candidates), 1)
                self.assertEqual(candidates[0]['config']['rdma_ctas'], 0)
                self.assertEqual(local['benchmark']['ep'], 8)
                self.assertEqual(json.loads(path.read_text()), original)
                with self.assertRaises(PlanError):
                    tune.load_plan(path, model=MODEL, system={'remote': 'root@example'})
                with self.assertRaises(PlanError):
                    tune.load_plan(path, model=MODEL, system={'nodes': 2})
                source['2_transport']['rdma_ctas'] = [8]
                path.write_text(json.dumps(source))
                for nodes in (2, 4, 8):
                    plan = tune.load_plan(path, model=MODEL, system={'nodes': nodes, 'remote': 'root@example'})
                    self.assertEqual(plan['benchmark']['ep'], nodes * 8)
                    self.assertEqual(tune.expand_candidates(plan)[0]['config']['rdma_ctas'], 8)
                    if nodes == 2:
                        command = tune.launcher_command(Path('/tmp/binary'), plan['benchmark'], rounds=30)
                        self.assertIn('root@example', command)
                    else:
                        with self.assertRaisesRegex(PlanError, 'GIN protocol'):
                            tune.launcher_command(Path('/tmp/binary'), plan['benchmark'], rounds=30)
                # Explicit parallel arguments cannot silently override frozen topology.
                path.write_text(json.dumps(local))
                with self.assertRaisesRegex(PlanError, 'frozen plan'):
                    tune.load_plan(path, system={'ep': 16})

    def test_cli_default_config_and_system_precedence(self):
        for phase in ('forward', 'backward'):
            tune = importlib.import_module('autotune_' + phase)
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                settings = root / 'system.json'
                settings.write_text(json.dumps({'nodes': 2}))
                output = root / 'out.json'
                for ep in (8, 32, 64):
                    # Explicit nodes overrides the system file, EP cross-checks it.
                    tune.main(['--system-config', str(settings), '--nodes', str(ep // 8),
                               '--ep', str(ep), '--hidden', '2560', '--intermediate', '768',
                               '--experts', '512', '--top-k', '8', '--generate-only',
                               '--candidate', 'incumbent', '--cache-root', str(root / 'cache'),
                               '--output', str(output)])
                    manifest = json.loads(output.read_text())
                    self.assertEqual(manifest['benchmark']['ep'], ep)
                    topology = 'single_node' if ep == 8 else 'multi_node'
                    self.assertEqual(Path(manifest['config_path']).name,
                                     f'{phase}_h200_{topology}_autotune.json')


if __name__ == '__main__':
    unittest.main()
