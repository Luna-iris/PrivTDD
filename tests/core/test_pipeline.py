"""Development tests outside the inference import graph; no large-data refits."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from privtdd.config import PrivTDDConfig


class PrivTDDRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=2)


    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()


    def test_midpoint_overflow(self):
        from privtdd.base_distribution.degree_posterior import compress_integer_cdf
        W = 1172756665
        output = compress_integer_cdf(lambda k: np.clip((np.asarray(k, float)+1)/(W+1), 0, 1), W, 1e7)
        self.assertTrue(np.all(output['keys'] >= 0))
        self.assertEqual(output['keys'].dtype, np.dtype('int64'))
        self.assertLessEqual(output['bound'], 1e7)


    def test_full_entry_and_cache_isolation(self):
        from unittest.mock import patch
        from privtdd.release.pipeline import estimate_tdd
        rows = [{(i-1)%12, (i+1)%12, (i+6)%12} for i in range(12)]
        config = PrivTDDConfig(workers=2, geometry_draws=32, geometry_replicates=2)
        with tempfile.TemporaryDirectory() as folder:
            with patch('privtdd.release.pipeline.fit_joint7', side_effect=RuntimeError('simulated interruption')):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    estimate_tdd(rows, 12, 11, 4., config=config, workdir=folder)
            self.assertTrue((Path(folder)/'joint_channel.npz').exists())
            with patch('privtdd.release.pipeline.build_family', side_effect=AssertionError('channel must be reused')):
                output, info = estimate_tdd(rows, 12, 11, 4., config=config, workdir=folder, return_debug=True)
            self.assertAlmostEqual(sum(output.values()), 12., places=8)
            self.assertEqual(info['posterior_updates'], 1)
            self.assertLessEqual(info['joint']['convex_gap'], 1e-6)
            again = estimate_tdd(rows, 12, 11, 4., config=config, workdir=folder)
            self.assertEqual(output, again)
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                estimate_tdd(rows, 12, 11, 2., config=config, workdir=folder)


    def test_cli_graph_prediction_then_metrics(self):
        from privtdd.io.runner import main
        from privtdd.reporting.collection import prepare_graph
        rows = [{(i-1)%12, (i+1)%12, (i+6)%12} for i in range(12)]
        ip, ix = prepare_graph(rows, 12, 11)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root/'graph.npz', indptr=ip, indices=ix)
            np.save(root/'truth.npy', np.zeros(12, np.int64))
            main(['--graph', str(root/'graph.npz'), '--public-cap', '11', '--epsilon', '4',
                  '--output', str(root/'output'), '--workers', '2', '--geometry-draws', '32',
                  '--geometry-replicates', '2', '--truth', str(root/'truth.npy')])
            self.assertTrue((root/'output/prediction.npz').exists())
            self.assertTrue((root/'output/run.json').exists())
            timing = json.loads((root/'output/completion.json').read_text())['timing']
            self.assertEqual(timing['scope'], 'fresh_graph_to_metrics')
            self.assertGreater(timing['collection_seconds'], 0.)
            self.assertGreater(timing['postprocessing_seconds'], 0.)
            metrics = json.loads((root/'output/metrics.json').read_text(encoding='utf-8'))
            self.assertTrue(all(np.isfinite(metrics[key]) for key in ('MSE', 'EMD', 'KS')))
            self.assertGreaterEqual((root/'output/metrics.json').stat().st_mtime_ns,
                                    (root/'output/prediction.npz').stat().st_mtime_ns)

