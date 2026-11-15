"""Small accelerated regression checks: population weights, fallback to the original path, cache isolation, and grid solving.

No real large graphs are run, and truth is not used to choose binning parameters; these tests do not establish superiority over any baseline.
"""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from privtdd.config import PrivTDDConfig, CAP_OFFSET
from privtdd.numerics.quantization import bin_reports, degree_grid, fit_degree_grid


class QuantizationRegression(unittest.TestCase):
    def test_triangle_bins_exact_mass_and_transport_bound(self):
        from privtdd.base_distribution.degree_posterior import compress_noise_cdf
        rng = np.random.default_rng(23)
        for W in (0, 64, 65, 66, 600):
            mass = rng.dirichlet(np.ones(W+1))
            cumulative = np.cumsum(mass)
            for step in (.5, 1., 17.4, 1000.):
                out = compress_noise_cdf(lambda keys: cumulative[keys], W, step)
                rebuilt = np.zeros(W+1)
                rebuilt[out['keys']] = out['probability']
                np.testing.assert_allclose(rebuilt[:min(W, 64)+1], mass[:min(W, 64)+1], atol=2e-15)
                self.assertAlmostEqual(rebuilt[W], mass[W])
                self.assertAlmostEqual(rebuilt.sum(), 1.)
                self.assertLessEqual(float(np.abs(np.cumsum(rebuilt-mass)).sum()), out['bound']+1e-10)
                for left, right, prob in zip(out['lower'], out['upper'], out['probability']):
                    self.assertAlmostEqual(prob, mass[left:right+1].sum())

    def test_lazy_structural_sources_preserve_samples(self):
        from privtdd.references.context import structural_context, prepare_compositions
        from privtdd.references.independent_edge import generate, generate_beta
        from privtdd.references.clustering import generate as geo
        from privtdd.references.geometry_model import population_pair_weights, calibrate
        caps = np.repeat([120, 150, 170], [80, 70, 60])
        prior = np.zeros(201)
        prior[[0, 1, 2, 5, 12]] = [.05, .1, .15, .65, .05]
        context, _ = structural_context(caps, prior, .6)
        packet, _ = prepare_compositions(context, draws=32, replicates=2)
        laws, old_draws, _ = generate(packet, float(packet['endpoint_means'][0]))
        beta, draws = generate_beta(packet)
        np.testing.assert_array_equal(draws, old_draws[:, 0])
        for j in range(len(beta.support)):
            for a, b in zip(beta.row(j), laws[0].row(j)):
                np.testing.assert_array_equal(a, b)
        M = population_pair_weights(packet)
        for target in (packet['endpoint_means'][0], np.mean(packet['endpoint_means']), packet['endpoint_means'][1]):
            calibration = calibrate(packet['probability'], M, float(target))
            old, old_samples, _ = geo(packet, float(target), endpoint_draws=old_draws[:, :2])
            endpoints = old_draws[:, :2] if calibration['kappa'] == 1. else np.column_stack((draws, np.zeros_like(draws)))
            new, new_samples, _ = geo(packet, float(target), endpoint_draws=endpoints, calibration=calibration)
            np.testing.assert_array_equal(new_samples, old_samples)
            for j in range(len(new.support)):
                for a, b in zip(new.row(j), old.row(j)):
                    np.testing.assert_array_equal(a, b)

    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    def test_fine_noise_grid_reuses_original_compression(self):
        from privtdd.base_distribution.degree_posterior import compress_noise_cdf, compress_integer_cdf
        W = 100000
        cdf = lambda k: (np.asarray(k, float)+1)/(W+1)
        old = compress_integer_cdf(cdf, W, 30.)
        new = compress_noise_cdf(cdf, W, 1., fallback_tolerance=30.)
        self.assertEqual(new['mode'], 'adaptive_W1_fallback')
        self.assertLessEqual(new['bound'], 30.)
        for key in ('keys', 'probability'):
            np.testing.assert_array_equal(new[key], old[key])
        guarded = compress_noise_cdf(cdf, W, 100., fallback_tolerance=30.)
        unguarded = compress_noise_cdf(cdf, W, 100.)
        self.assertEqual(guarded['mode'], 'noise_scaled')
        np.testing.assert_array_equal(guarded['keys'], unguarded['keys'])
        np.testing.assert_array_equal(guarded['probability'], unguarded['probability'])

    def test_batched_channel_preserves_kernel_and_adjoint(self):
        from privtdd.correction.sparse_channel import batched_integer_channel, use_batched_channel
        from privtdd.base_distribution.count_channel import logpdf_q
        rng = np.random.default_rng(53)
        keys = np.array([0, 1, 64, 120, 190])
        report = np.r_[0., rng.normal(30, 100, 37)]
        scales = np.r_[0., rng.uniform(1, 40, 37)]
        shapes = rng.dirichlet(np.ones(len(keys)), size=3).T
        weights = rng.uniform(.1, 2, (len(report), 3))
        kernel = np.ones((len(report), len(keys)))
        kernel[1:] = np.exp(logpdf_q(report[1:, None], 20, keys[None, :], .7, scales[1:, None]))
        forward = batched_integer_channel(report, scales, 20, keys, .7, shapes, block_pairs=31)
        adjoint = batched_integer_channel(report, scales, 20, keys, .7, weights, adjoint=True, block_pairs=31)
        np.testing.assert_allclose(forward, kernel@shapes, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(adjoint, kernel.T@weights, rtol=2e-14, atol=1e-15)
        np.testing.assert_allclose(forward.T@weights, shapes.T@adjoint, rtol=2e-14, atol=1e-15)
        self.assertTrue(use_batched_channel([(20, np.arange(3), .5)], 20, 100))
        self.assertFalse(use_batched_channel([(20, np.arange(3000), .5)], 20, 100))
        self.assertFalse(use_batched_channel([(20, np.arange(3), .5)], 5, 10))

    def test_fast_pipeline_with_batched_channel_matches_sparse(self):
        from privtdd.release.pipeline import estimate_tdd_from_reports
        caps = np.repeat([120, 150, 170], [80, 70, 60])
        noisy = np.repeat([-20., 5., 35.], [80, 70, 60])
        prior = np.zeros(201)
        prior[[0, 1, 2, 5, 12]] = [.05, .10, .15, .65, .05]
        outputs = []
        with tempfile.TemporaryDirectory() as directory:
            for batch in (False, True):
                hist, info = estimate_tdd_from_reports(caps, noisy, 200, 4., degree_prior=prior,
                    config=PrivTDDConfig(workers=2, geometry_draws=32, geometry_replicates=2,
                        report_bin_width=1/16, triangle_bin_width=1/16,
                        structural_lazy_endpoints=True, batch_small_groups=batch,
                        batch_f1=batch, compiled_structure=batch),
                    workdir=Path(directory)/str(batch), return_debug=True)
                self.assertEqual(info['posterior_updates'], 1)
                self.assertAlmostEqual(sum(hist.values()), len(caps))
                outputs.append(hist)
            keys = sorted(outputs[0].keys() | outputs[1].keys())
            np.testing.assert_allclose([outputs[0].get(k, 0.) for k in keys],
                                       [outputs[1].get(k, 0.) for k in keys], rtol=2e-6, atol=2e-6)

    def test_noise_scaled_bins_preserve_cap_counts_and_means(self):
        caps = np.array([0, 1, 8, 8, 8, 8, 9, 9])
        report = np.array([0., 0., -1.1, -1.2, 1., 1.2, 1., 1.2])
        c, q, w, stats = bin_reports(caps, report, .5, 1/16)
        self.assertEqual(w.sum(), len(report))
        self.assertLess(len(c), len(caps))
        self.assertLessEqual(stats['max_displacement_in_noise_units'], 1/16+1e-14)
        for cap in np.unique(caps):
            self.assertEqual(w[c == cap].sum(), np.sum(caps == cap))
            self.assertAlmostEqual(float(q[c == cap]@w[c == cap]), report[caps == cap].sum())
        self.assertTrue(np.all(q[c <= 1] == 0))
        # Exact duplicate aggregation is reversible without discarding tail reports.
        c, q, w, _ = bin_reports(np.array([2, 2, 3]), np.array([1e10, 1e10, -1e10]), 1., 0.)
        self.assertEqual(list(zip(c, q, w)), [(2, 1e10, 2), (3, -1e10, 1)])

    def test_grid_fixed_low_degrees_and_bounded_size(self):
        caps = np.arange(150, 1000, 3)
        grid = degree_grid(caps, 20127, 256)
        np.testing.assert_array_equal(grid[:65], np.arange(65))
        self.assertLessEqual(len(grid), 256)
        self.assertTrue({max(0, 20127-CAP_OFFSET), 20127}.issubset(set(grid)))
        np.testing.assert_array_equal(degree_grid(caps, 100, 256), np.arange(101))
        with self.assertRaises(ValueError):
            PrivTDDConfig(degree_grid_size=32)

    def test_grid_newton_recovery_keeps_original_simplex_objective(self):
        from privtdd.numerics.quantization import refine_grid_weights
        from privtdd.degree.channel import log_cap_probability
        from privtdd.numerics.likelihood import mixture_objective
        support = np.r_[np.arange(10), 20, 50, 180]
        K = np.exp(log_cap_probability(np.arange(181)[:, None], support[None, :], 180, .3))
        planted = np.zeros(len(support)); planted[[1, 4, 10, 12]] = [.2, .3, .4, .1]
        frequency = K@planted; frequency /= frequency.sum()
        K /= K.max(axis=1)[:, None]
        initial = np.full(len(support), 1/len(support))
        prior, stats = refine_grid_weights(K, frequency, initial, 1e-7)
        value, gradient = mixture_objective(prior, K, frequency)
        self.assertGreaterEqual(prior.min(), 0.)
        self.assertAlmostEqual(prior.sum(), 1.)
        self.assertLessEqual(prior@gradient-gradient.min(), 1e-7)
        self.assertLessEqual(value, mixture_objective(initial, K, frequency)[0])

    def test_degree_grid_covers_rare_high_caps_without_dropping_reports(self):
        caps = np.r_[np.repeat(np.arange(150, 501), 20), 21933, 22045, 26342, 37932, 45396, 51386]
        grid = degree_grid(caps, 51386, 512)
        centers = np.clip(caps-CAP_OFFSET, 0, 51386)
        distance = np.min(abs(centers[:, None]-grid[None, :]), axis=1)
        self.assertLessEqual(distance.max(), 75)
        self.assertLessEqual(len(grid), 512)
        np.testing.assert_array_equal(grid[:65], np.arange(65))

    def test_restricted_degree_mle_and_progress(self):
        rng = np.random.default_rng(19)
        degrees = rng.choice([2, 8, 30, 180], 1000)
        caps = np.clip(np.floor(degrees+CAP_OFFSET+rng.laplace(0, 5, len(degrees))), 0, 400).astype(int)
        progress = []
        prior, info = fit_degree_grid(caps, 400, .2, 96, checkpoint=lambda *row: progress.append(row))
        self.assertAlmostEqual(prior.sum(), 1.)
        self.assertTrue(np.isfinite(prior).all() and np.all(prior >= 0))
        self.assertLessEqual(info['restricted_gap'], 1e-7)
        self.assertLessEqual(np.count_nonzero(prior), 96)
        self.assertTrue(progress)
        self.assertFalse(info['full_integer_optimality_claimed'])

    def test_weighted_pipeline_matches_expanded_identical_reports(self):
        from privtdd.release.pipeline import estimate_tdd_from_reports
        # Three identical-report groups with unequal populations exercise S3, S5,
        # S6 weights. Structural context receives the same full cap transcript.
        caps = np.repeat([120, 150, 170], [80, 70, 60])
        epsilon, D = 4., 200
        noisy = np.repeat([-20., 5., 35.], [80, 70, 60])
        prior = np.zeros(D+1)
        prior[[0, 1, 2, 5, 12]] = [.05, .10, .15, .65, .05]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for width in (0., 1/16):
                hist, info = estimate_tdd_from_reports(caps, noisy, D, epsilon, degree_prior=prior,
                    config=PrivTDDConfig(workers=2, geometry_draws=32, geometry_replicates=2, report_bin_width=width),
                    workdir=root/str(width), return_debug=True)
                outputs.append(hist)
                self.assertAlmostEqual(sum(hist.values()), len(caps), places=8)
                self.assertEqual(info['posterior_updates'], 1)
            keys = sorted(outputs[0].keys() | outputs[1].keys())
            np.testing.assert_allclose([outputs[0].get(k, 0) for k in keys],
                                       [outputs[1].get(k, 0) for k in keys], atol=2e-6, rtol=2e-6)
            self.assertEqual(info['F1']['representative_rows'], 3)
            self.assertEqual(info['F1']['population_nodes'], len(caps))
            # Resume must use the matching approximation, not the exact cache.
            same = estimate_tdd_from_reports(caps, noisy, D, epsilon, degree_prior=prior,
                config=PrivTDDConfig(workers=2, geometry_draws=32, geometry_replicates=2, report_bin_width=1/16),
                workdir=root/str(1/16))
            self.assertEqual(same, outputs[1])
            with self.assertRaisesRegex(ValueError, 'different inputs'):
                estimate_tdd_from_reports(caps, noisy, D, epsilon, degree_prior=prior,
                    config=PrivTDDConfig(workers=2, geometry_draws=32, geometry_replicates=2, report_bin_width=1/8),
                    workdir=root/str(1/16))

    def test_cli_preset_and_explicit_overrides(self):
        from privtdd.io.runner import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez(root/'reports.npz', caps=np.array([2]*4), noisy_triangle_report=np.zeros(4))
            args = ['--reports', str(root/'reports.npz'), '--public-cap', '3', '--epsilon', '4',
                    '--acceleration', 'binned', '--output', str(root/'output'), '--dry-run']
            output = io.StringIO()
            with redirect_stdout(output):
                main(args)
            config = json.loads(output.getvalue())['config']
            self.assertEqual((config['degree_grid_size'], config['report_bin_width']), (512, 1/16))
            output = io.StringIO()
            with redirect_stdout(output):
                main(args+['--degree-grid-size', '0', '--report-bin-width', '0'])
            config = json.loads(output.getvalue())['config']
            self.assertEqual((config['degree_grid_size'], config['report_bin_width']), (0, 0.))
            self.assertFalse((root/'output').exists())
            output = io.StringIO()
            with redirect_stdout(output):
                main(args+['--acceleration', 'fast'])
            config = json.loads(output.getvalue())['config']
            self.assertEqual(config['triangle_bin_width'], 1/16)
            self.assertTrue(config['structural_lazy_endpoints'])


if __name__ == '__main__':
    unittest.main()
