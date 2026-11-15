"""Existing CLI grid orchestration: isolation, resume, failures and no fake seeds."""
import csv
import io
import json
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from privtdd.io.runner import main


class GridRunnerRegression(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(redirect_stderr(io.StringIO()))

    def fixtures(self, root):
        """Two simple undirected rings, stored in the repository cache format."""
        for name in ('alpha', 'beta'):
            folder = root/'cache'/name
            folder.mkdir(parents=True)
            rows = [sorted(((i-1)%6, (i+1)%6)) for i in range(6)]
            np.save(folder/'indptr.npy', np.arange(0, 13, 2, dtype=np.int64))
            np.save(folder/'indices.npy', np.array(rows, dtype=np.int64).ravel())
            np.save(folder/'triangles.npy', np.zeros(6, dtype=np.int64))
            (folder/'graph.json').write_text(json.dumps(dict(public_degree_cap=5, nodes=6)))
        return ['--datasets', 'alpha', 'beta', '--cache-root', str(root/'cache'),
                '--epsilons', '1', '2', '--seeds', '11', '12', '--output', str(root/'out')]

    @staticmethod
    def fake_fit(caps, noisy, public_cap, epsilon, **kwargs):
        return {0: float(len(caps))}, dict(posterior_updates=1)

    def test_grid_isolation_summary_and_completed_resume(self):
        from privtdd.reporting.collection import collect_reports_joint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixtures(root)
            with patch('privtdd.io.runner.collect_reports_joint', wraps=collect_reports_joint) as collect:
                with patch('privtdd.io.runner.estimate_tdd_from_reports', side_effect=self.fake_fit) as fit:
                    main(args)
                    self.assertEqual(collect.call_count, 8)
                    self.assertEqual(fit.call_count, 8)
                    self.assertEqual([c.args[4] for c in collect.call_args_list], [11, 12]*4)
                    main(args)
                    self.assertEqual(collect.call_count, 8)
                    self.assertEqual(fit.call_count, 8)
            cases = list((root/'out').glob('*/case_manifest.json'))
            self.assertEqual(len(cases), 8)
            identities = [json.loads(p.read_text())['identity'] for p in cases]
            self.assertEqual(len(set(identities)), 8)
            for p in cases:
                self.assertTrue((p.parent/'reports.npz').exists())
                self.assertTrue((p.parent/'completion.json').exists())
            with (root/'out/summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row['completed'] == '2' and row['evaluated'] == '2' for row in rows))
            self.assertTrue(all(float(row['MSE_mean']) == 0 and float(row['MSE_std']) == 0 for row in rows))
            # Changed numerical controls must not silently skip an old result.
            previous_summary = (root/'out/summary.csv').read_bytes()
            with self.assertRaisesRegex(ValueError, 'Different inputs/code/settings'):
                main(args+['--geometry-draws', '32'])
            self.assertEqual((root/'out/summary.csv').read_bytes(), previous_summary)

    def test_failure_continues_and_resume_reuses_private_reports(self):
        from privtdd.reporting.collection import collect_reports_joint
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixtures(root)+['--continue-on-error']
            calls = 0
            def fails_once(*values, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError('synthetic fit failure')
                return self.fake_fit(*values, **kwargs)
            with patch('privtdd.io.runner.collect_reports_joint', wraps=collect_reports_joint) as collect:
                with patch('privtdd.io.runner.estimate_tdd_from_reports', side_effect=fails_once):
                    with self.assertRaises(SystemExit) as stopped:
                        main(args)
                    self.assertEqual(stopped.exception.code, 1)
                progress = json.loads((root/'out/batch_progress.json').read_text())
                self.assertEqual((progress['completed'], progress['failed']), (7, 1))
                with patch('privtdd.io.runner.estimate_tdd_from_reports', side_effect=self.fake_fit) as fit:
                    main(args)
                    self.assertEqual(fit.call_count, 1)
                self.assertEqual(collect.call_count, 8)

    def test_dry_run_and_rejecting_fake_multiseed_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixtures(root)
            with patch('privtdd.io.runner.collect_reports_joint', side_effect=AssertionError('must not collect')):
                main(args+['--dry-run'])
            self.assertFalse((root/'out').exists())
            with self.assertRaises(SystemExit):
                main(['--reports', 'not_needed.npz', '--public-cap', '5', '--epsilon', '1',
                      '--seeds', '11', '12', '--output', str(root/'out')])
            self.assertFalse((root/'out').exists())

    def test_evaluation_is_after_prediction(self):
        from privtdd.io.runner import evaluate_saved_prediction
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixtures(root)
            args = ['--graph', str(root/'cache/alpha'), '--public-cap', '5', '--epsilon', '1',
                    '--seed', '11', '--truth', str(root/'cache/alpha/triangles.npy'), '--output', str(root/'single')]
            def evaluate(prediction, truth_path, public_cap):
                self.assertTrue(prediction.is_file())
                return evaluate_saved_prediction(prediction, truth_path, public_cap)
            with patch('privtdd.io.runner.estimate_tdd_from_reports', side_effect=self.fake_fit):
                with patch('privtdd.io.runner.evaluate_saved_prediction', side_effect=evaluate) as evaluation:
                    main(args)
                    evaluation.assert_called_once()

    def test_guard_presets_and_public_degree_policy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.fixtures(root)
            for name in ('alpha', 'beta'):
                (root/'cache'/name/'graph.json').write_text(json.dumps(dict(public_degree_cap=3, nodes=6)))
            for preset in ('exact', 'binned', 'fast'):
                output = io.StringIO()
                with redirect_stdout(output):
                    main(args+['--dry-run', '--acceleration', preset])
                plan = json.loads(output.getvalue())
                self.assertTrue(plan['config']['safeguard'])
                self.assertEqual(plan['config']['safeguard_tau'], 1.)
                self.assertTrue(all(case['public_cap'] == 5 for case in plan['plan']))
                self.assertEqual(plan['config']['degree_grid_size'], 0 if preset == 'exact' else 512)
            output = io.StringIO()
            with redirect_stdout(output):
                main(args+['--dry-run', '--no-safeguard', '--safeguard-tau', '2', '--public-cap-policy', 'declared'])
            plan = json.loads(output.getvalue())
            self.assertFalse(plan['config']['safeguard'])
            self.assertEqual(plan['config']['safeguard_tau'], 2.)
            self.assertTrue(all(case['public_cap'] == 3 for case in plan['plan']))
            self.assertFalse((root/'out').exists())

    def test_report_channel_metadata_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reports = root/'reports.npz'
            arrays = dict(caps=np.array([0, 1, 2, 3]), noisy_triangle_report=np.zeros(4))
            np.savez(reports, **arrays)
            args = ['--reports', str(reports), '--public-cap', '3', '--epsilon', '1']
            with patch('privtdd.io.runner.estimate_tdd_from_reports', side_effect=self.fake_fit) as fit:
                with self.assertRaisesRegex(ValueError, 'lacks cap_offset'):
                    main(args+['--output', str(root/'unknown')])
                fit.assert_not_called()
                main(args+['--output', str(root/'declared'), '--report-cap-offset', '0'])
                self.assertEqual(fit.call_args.kwargs['report_cap_offset'], 0)
                np.savez(reports, **arrays, cap_offset=0)
                main(args+['--output', str(root/'tagged')])
                self.assertEqual(int(fit.call_args.kwargs['report_cap_offset']), 0)
            # The actual estimator rejects an incompatible channel before fitting.
            np.savez(reports, **arrays, cap_offset=150)
            with self.assertRaisesRegex(ValueError, 'cap_offset must match current L=0'):
                main(args+['--output', str(root/'wrong_channel')])
            with self.assertRaisesRegex(ValueError, 'conflicts with report metadata'):
                main(args+['--output', str(root/'conflicting'), '--report-cap-offset', '0'])


if __name__ == '__main__':
    unittest.main()
