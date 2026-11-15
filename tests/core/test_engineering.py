"""Bounded numerical regressions; never part of estimator execution."""
import unittest
import numpy as np
from threadpoolctl import threadpool_limits


class EngineeringRegression(unittest.TestCase):
    def test_structural_samples_and_rng_states(self):
        from privtdd.references.kernels import endpoint_block, geometry_block
        from privtdd.references.independent_edge import draw_three
        from privtdd.references.clustering import draw_count
        from privtdd.references.geometry_model import arc_mixture
        r = np.random.default_rng(994)
        P = r.uniform(.01, .9, (5, 5)); P = (P+P.T)/2
        counts = np.vstack([np.zeros((1, 5), np.int64), np.eye(5, dtype=np.int64),
                            r.integers(0, 70, (60, 5))])
        for root in (0, 3):
            for hard in (False, True):
                a, b = np.random.default_rng(12), np.random.default_rng(22)
                c, d = np.random.default_rng(12), np.random.default_rng(22)
                expected = np.array([draw_three(x, root, P, 0., a, b, None, beta_only=not hard) for x in counts])
                actual = endpoint_block(counts, root, P, c, d, hard)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(a.bit_generator.state, c.bit_generator.state)
                self.assertEqual(b.bit_generator.state, d.bit_generator.state)
            for kappa in (0., .37, 1.):
                arcs = arc_mixture(P, kappa)
                a, b = np.random.default_rng(32), np.random.default_rng(32)
                expected = np.array([draw_count(x, root, P, arcs, a) for x in counts])
                actual = geometry_block(counts, root, P, *arcs, b)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(a.bit_generator.state, b.bit_generator.state)

    def test_f1_matrix_preserves_actual_scales(self):
        from privtdd.base_distribution.channel import build_matrix
        r = np.random.default_rng(94)
        caps = np.r_[0, 1, np.arange(3, 100), np.full(2050, 140)]
        report = r.normal(1., 50., len(caps)); report[:2] = 0
        prior = np.zeros(201); prior[[0, 1, 2, 5, 12, 40, 100]] = 1/7
        with threadpool_limits(limits=2):
            a = build_matrix(report, caps, prior, .7, .6, .7, workers=2)
            b = build_matrix(report, caps, prior, .7, .6, .7, workers=2, batch_degrees=True)
        np.testing.assert_allclose(a[0], b[0], rtol=3e-14, atol=1e-17)
        np.testing.assert_array_equal(a[1], b[1])
        for x, y in zip(a[2], b[2]):
            for key in x:
                np.testing.assert_array_equal(x[key], y[key])

    def test_f1_small_scale_integrals(self):
        from privtdd.base_distribution.integrated import IntegratedDegreePosterior
        from privtdd.base_distribution.cells import closure_edges
        r = np.random.default_rng(52)
        y = r.normal(30., 200., 100)
        b = np.r_[np.full(30, 5.), np.arange(7., 77.)]
        a = r.uniform(.1, 1., 100); a /= a.sum()
        theta = np.full(65, 1/65)
        args = (40, y, b, a, theta, closure_edges(65), .7)
        old, new = IntegratedDegreePosterior(*args), IntegratedDegreePosterior(*args, batch_small_groups=True)
        np.testing.assert_allclose(new.component_mass, old.component_mass, atol=1e-13, rtol=1e-12)
        keys = np.arange(781)
        np.testing.assert_allclose(new.cdf(keys), old.cdf(keys), atol=1e-12, rtol=1e-12)

    def test_same_packet_with_cached_root_kernels(self):
        from privtdd.references.context import structural_context, prepare_compositions
        prior = np.zeros(201); prior[[0, 1, 2, 5, 12]] = [.05, .10, .15, .65, .05]
        caps = np.repeat([120, 150, 170], [80, 70, 60])
        with threadpool_limits(limits=2):
            context, _ = structural_context(caps, prior, .6)
            a, x = prepare_compositions(context, draws=64, replicates=2)
            b, y = prepare_compositions(context, draws=64, replicates=2, reuse_root_kernels=True)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        self.assertEqual(x, y)
