"""L0 channel consistency and final S7 integration across production presets."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from privtdd.release import pipeline as a
from privtdd.config import PrivTDDConfig, CAP_OFFSET
from privtdd.experiments.ablation import fast_config
from privtdd.release import transport_guard as s
from tests.core import reference_safeguard as v


class MainlineSafeguardTests(unittest.TestCase):
    def test_default_L0_channel_grid_and_report_contract(self):
        from privtdd.degree.channel import log_cap_probability
        from privtdd.degree.operator import CapDegreeOperator
        from privtdd.degree.canonical import canonicalize_tail
        from privtdd.degree.reduced import collapse_equivalent_tail
        from privtdd.degree.working_set import collapse_equivalent_tail as working_tail
        from privtdd.numerics.quantization import degree_grid
        self.assertEqual(CAP_OFFSET, 0)
        self.assertTrue(PrivTDDConfig().safeguard)
        self.assertTrue(fast_config().safeguard)
        D = 8
        grid = np.arange(D+1)
        kernel = np.exp(log_cap_probability(grid[:, None], grid[None, :], D, .3))
        self.assertEqual(np.linalg.matrix_rank(kernel), D+1)
        prior = np.arange(1., D+2); prior /= prior.sum()
        operator = CapDegreeOperator(D, .3)
        np.testing.assert_allclose(operator.forward(prior), kernel@prior, atol=2e-15)
        np.testing.assert_allclose(operator.adjoint(prior), kernel.T@prior, atol=2e-15)
        for function in (canonicalize_tail, collapse_equivalent_tail, working_tail):
            np.testing.assert_array_equal(function(prior, .3)[0], prior)
        caps = np.arange(0, 4001, 9, dtype=np.int64)
        np.testing.assert_array_equal(degree_grid(caps, 5000, 512), v.shifted_degree_grid(caps, 5000, 512, 0))
        ip, ix, _ = v.synthetic_graph(12, 'tail')
        released = a.collect_reports_joint(ip, ix, 11, 2., 20261028)
        noise = np.random.default_rng(20261028).laplace(0, 1/.3, 12)
        np.testing.assert_array_equal(released['caps'], np.clip(np.floor(np.diff(ip)+noise), 0, 11))
        self.assertEqual(int(released['cap_offset']), 0)
        with self.assertRaisesRegex(ValueError, 'cap_offset'):
            a.estimate_tdd_from_reports(released['caps'], released['noisy_triangle_report'],
                11, 2., report_cap_offset=150)

    def test_reference_envelope_and_degree_margin(self):
        rng = np.random.default_rng(741)
        for D in (2, 9, 100):
            prior = rng.dirichlet(np.ones(D+1))
            noisy = rng.normal(10, 30, 100)
            direct, reference, joint, cap = s.capacity_reference(noisy, prior, D)
            points = np.union1d(direct[0], cap[0])
            np.testing.assert_allclose(s.cdf_at(*s.law(*reference), points),
                np.maximum(s.cdf_at(*s.law(*direct), points), s.cdf_at(*cap, points)), atol=2e-14)
            np.testing.assert_allclose(np.bincount(joint['degree'], weights=joint['probability'], minlength=D+1),
                prior, atol=2e-14)
            self.assertTrue(np.all(joint['triangles'] <= joint['degree']*(joint['degree']-1)//2))
            # The validation runner is deliberately retained as an independent
            # pre-integration implementation of the numerical safeguard.
            expected = v.capacity_reference(noisy, prior, D)
            for actual_pair, expected_pair in zip((direct, reference, cap), (expected[0], expected[1], expected[3])):
                for actual, old in zip(actual_pair, expected_pair):
                    np.testing.assert_array_equal(actual, old)

    def test_trigger_boundary_zero_distance_and_large_sparse_domain(self):
        raw = (np.array([10]), np.array([20.]))
        reference = (np.array([0]), np.array([20.]))
        output, info = s.safeguard(raw, reference, tau=.5, D=5)
        self.assertEqual(info['alpha'], .25)
        self.assertAlmostEqual(s.w1(output, reference), 2.5)
        self.assertAlmostEqual(s.w1(output, raw), 7.5)
        for other, tau in ((reference, 2.), (raw, .5)):
            output, info = s.safeguard(raw, other, tau=tau, D=5)
            self.assertEqual(info['alpha'], 1.)
            for actual, old in zip(output, raw):
                np.testing.assert_array_equal(actual, old)
        D = 200000
        prior = np.zeros(D+1); prior[[0, 300, D]] = [.1, .2, .7]
        _, reference, joint, cap = s.capacity_reference(np.arange(D, dtype=float), prior, D)
        self.assertLessEqual(s.capacity_violation(reference, cap), 2e-12)
        self.assertAlmostEqual(reference[1].sum(), D, places=8)

    def test_final_stage_for_exact_binned_fast_and_legacy(self):
        ip, ix, _ = v.synthetic_graph(8, 'sparse')
        released = a.collect_reports_joint(ip, ix, 7, 2., 20261028)
        exact = PrivTDDConfig(workers=1, blas_threads=1, geometry_draws=64, geometry_replicates=1)
        configs = dict(exact=exact, binned=replace(exact, degree_grid_size=512, report_bin_width=1/16),
            fast=fast_config(workers=1, blas_threads=1, geometry_draws=64, geometry_replicates=1),
            legacy=replace(exact, posterior_backend='legacy'))
        with tempfile.TemporaryDirectory(prefix='joint_L0_guard_') as temporary:
            root = Path(temporary)
            for name, config in configs.items():
                with self.subTest(preset=name):
                    config = replace(config, safeguard_tau=1e-6)
                    work = root/name
                    hist, info = a.estimate_tdd_from_reports(released['caps'], released['noisy_triangle_report'],
                        7, 2., config=config, workdir=work, report_cap_offset=0, return_debug=True)
                    self.assertEqual(info['posterior_updates'], 1)
                    self.assertEqual(info['cap_offset'], 0)
                    self.assertTrue(info['safeguard']['triggered'])
                    raw = v.load_hist(work/'prediction_raw.npz')
                    with np.load(work/'degree.npz') as z:
                        prior = z['prior']
                    _, reference, _, cap = v.capacity_reference(released['noisy_triangle_report'], prior, 7)
                    expected, stats = v.safeguard(raw, reference, tau=1e-6, D=7)
                    actual = v.load_hist(work/'prediction.npz')
                    for x, y in zip(actual, expected):
                        np.testing.assert_array_equal(x, y)
                    self.assertAlmostEqual(sum(hist.values()), 8.)
                    self.assertLessEqual(s.capacity_violation(actual, cap), 2e-9)
                    with patch.object(a, '_run_joint', side_effect=AssertionError('completed case refitted')):
                        again = a.estimate_tdd_from_reports(released['caps'], released['noisy_triangle_report'],
                            7, 2., config=config, workdir=work, report_cap_offset=0)
                    self.assertEqual(hist, again)
                    with self.assertRaisesRegex(ValueError, 'different inputs'):
                        a.estimate_tdd_from_reports(released['caps'], released['noisy_triangle_report'],
                            7, 2., config=replace(config, safeguard_tau=1.), workdir=work)

    def test_no_trigger_and_disabled_guard_preserve_posterior(self):
        ip, ix, _ = v.synthetic_graph(8, 'tail')
        cfg = fast_config(workers=1, blas_threads=1, geometry_draws=64, geometry_replicates=1)
        with tempfile.TemporaryDirectory(prefix='joint_L0_raw_') as temporary:
            root = Path(temporary)
            enabled, info = a.estimate_tdd((ip, ix), 8, 7, 2., 20261028,
                config=replace(cfg, safeguard_tau=100.), workdir=root/'enabled', return_debug=True)
            self.assertEqual(info['safeguard']['alpha'], 1.)
            for x, y in zip(v.load_hist(root/'enabled/prediction_raw.npz'), v.load_hist(root/'enabled/prediction.npz')):
                np.testing.assert_array_equal(x, y)
            with patch.object(a, 'capacity_reference', side_effect=AssertionError('disabled guard ran')):
                disabled, info = a.estimate_tdd((ip, ix), 8, 7, 2., 20261028,
                    config=replace(cfg, safeguard=False), workdir=root/'disabled', return_debug=True)
            self.assertFalse(info['safeguard']['enabled'])
            self.assertEqual(enabled, disabled)

    def test_config_rejects_invalid_guard(self):
        for tau in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                PrivTDDConfig(safeguard_tau=tau)
        with self.assertRaises(ValueError):
            PrivTDDConfig(safeguard=1)


if __name__ == '__main__':
    unittest.main()
