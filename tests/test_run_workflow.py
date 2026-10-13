import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run as workflow
from weftmoe import ModelSpec


class RunWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.model = ModelSpec(4096, 1536, 64, 8)
        self.environment = patch.dict('os.environ', {'WEFTMOE_ROOT': str(self.directory),
                                                    'WEFTMOE_AUTOTUNE_CONFIG': ''})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_simple_defaults_and_exact_tokens(self):
        for phase, relay in [('forward', 24), ('backward', 32)]:
            result = workflow.run(self.model, phase=phase, tokens=1000, generate_only=True)
            self.assertEqual(len(result['candidates']), 1)
            self.assertEqual(result['candidates'][0]['config']['relay_ctas'], relay)
            self.assertEqual(result['candidates'][0]['config']['rdma_ctas'], 0)
            self.assertEqual(result['benchmark']['tokens_per_rank'], 1000)
            self.assertEqual(result['jit']['jobs'], 16)

    def test_tune_uses_bucket_and_full_candidate_set(self):
        for phase in ('forward', 'backward'):
            result = workflow.run(self.model, phase=phase, tokens=1000,
                                  autotune=True, generate_only=True)
            self.assertGreater(len(result['candidates']), 1)
            self.assertEqual(result['benchmark']['tokens_per_rank'], 8192)
            self.assertEqual(result['benchmark']['tune_rounds'], 25)
            self.assertEqual(result['selection_policy']['confirmation_rounds'], 40)

    def test_explicit_config_and_system_topology(self):
        config = ROOT / 'configs/forward_h200_single_node_autotune.json'
        result = workflow.run(self.model, config=config, system=dict(nodes=1, gpus_per_node=2),
                              generate_only=True)
        self.assertEqual(result['topology']['ep'], 2)
        self.assertEqual(result['candidates'][0]['config']['relay_ctas'], 24)

    def test_default_and_autotune_incumbent_agree(self):
        for phase in ('forward', 'backward'):
            for nodes in (1, 2):
                system = dict(nodes=nodes, gpus_per_node=8)
                if nodes > 1:
                    system['remote'] = 'root@example'
                plain = workflow.run(self.model, phase=phase, system=system,
                                     generate_only=True)
                tune = workflow.run(self.model, phase=phase, system=system,
                                    autotune=True, generate_only=True)
                self.assertEqual(plain['candidates'][0]['config'],
                                 tune['candidates'][0]['config'])
                self.assertGreater(len(tune['candidates']),
                                   len(plain['candidates']))

    def test_tuned_winner_benchmarks_requested_tokens(self):
        import autotune_forward as forward
        generated = workflow.run(self.model, tokens=1000, autotune=True, generate_only=True)
        candidate = generated['candidates'][-1]
        def tune(arguments):
            self.assertIn('--skip-final', arguments)
            path = Path(arguments[arguments.index('--output') + 1])
            path.write_text(json.dumps({'selected': dict(candidate, binary='/tmp/test-binary')}))
            return 0
        with patch.object(forward, 'main', side_effect=tune), patch.object(
                forward, 'benchmark_candidate', return_value=dict(returncode=0, record={'measured': 30})) as benchmark:
            result = workflow.run(self.model, tokens=1000, autotune=True)
        self.assertEqual(benchmark.call_args.args[2]['tokens_per_rank'], 1000)
        self.assertEqual(benchmark.call_args.kwargs['rounds'], 40)
        self.assertEqual(result['config'], candidate['config'])
        self.assertEqual(result['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
