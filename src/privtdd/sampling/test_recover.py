"""Synthetic numerical checks for the recovery kernel; read no real graphs, original experiment results, or ground truth."""
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from privtdd.base_distribution.cells import closure_edges, integer_cell_weights
from .recover import _checked_grid, _closure_grid, _operator
from .sampled_channel import SampledChannel


class SampledRecoveryTests(unittest.TestCase):
    def test_small_closure_cells_match_main_implementation(self):
        edges = closure_edges(65)
        for degree in (0, 1, 2, 5, 20):
            with self.subTest(degree=degree):
                w = degree * (degree - 1) // 2
                keys, basis = _closure_grid(degree, edges, 1)
                np.testing.assert_array_equal(keys, np.arange(w + 1))
                atom0 = (keys == 0).astype(float)
                atom1 = (keys == w).astype(float)
                expected = np.column_stack((atom0, integer_cell_weights(w, edges), atom1))
                np.testing.assert_allclose(basis.toarray(), expected, atol=2e-14, rtol=0.)

    def test_forward_and_adjoint_use_the_same_one_node_channel(self):
        rng = np.random.default_rng(321)
        for k in (1, 5, 26):
            channel = SampledChannel(52, k, .61, .49)
            # Cover untruncated and truncated degrees, zero capacity, and zero latent degree; k=1 also covers sparse exposures.
            for d, cap in ((6, 3), (7, 20), (40, 35), (3, 0), (0, 10)):
                with self.subTest(k=k, d=d, cap=cap):
                    w = d * (d - 1) // 2
                    keys = np.unique(rng.integers(w + 1, size=15))
                    q = rng.normal(2., 5., size=11) if cap > 1 else np.zeros(11)
                    shapes = rng.random((len(keys), 3))
                    inverse = rng.random((len(q), 3))
                    forward = _operator(channel, d, cap, keys, q, csr_matrix(shapes))
                    adjoint = _operator(channel, d, cap, keys, q, inverse, adjoint=True)
                    kernel = channel.density(d, cap, keys, q)
                    np.testing.assert_allclose(forward, kernel @ shapes, atol=5e-13, rtol=5e-13)
                    np.testing.assert_allclose(adjoint, kernel.T @ inverse, atol=5e-13, rtol=5e-13)
                    self.assertAlmostEqual(float((forward * inverse).sum()),
                                           float((shapes * adjoint).sum()), places=10)

    def test_high_degree_peaks_near_complete_closure(self):
        # Use graph size only as a public design parameter; do not generate a large graph. Choose synthetic Q near full closure
        # and test continuous-cell quadrature both without original-degree truncation and with a high original degree truncated by cap.
        n = 3_072_441
        edges = closure_edges(65)
        for d, cap, rate in ((5_000, 5_000, .5), (300_000, 300_000, .5),
                             (2_500_000, 2_500_000, .5), (1_560_000, 1_250_000, .01)):
            with self.subTest(d=d, cap=cap, rate=rate):
                channel = SampledChannel(n, int(n * rate), .672, .3224)
                maximum = channel.gap * channel.s * cap * (cap - 1) / 2
                q = maximum * np.array([.03, .98704, .9947, .99972, 1.])
                q = np.r_[q, maximum + 2 * channel.scale(cap)]
                (keys, cell), audit = _checked_grid(
                    channel, d, [(cap, np.arange(len(q)), 1.)], q, edges)
                self.assertEqual(keys[-1], d * (d - 1) // 2)
                self.assertGreater(audit['probes'], 0)
                self.assertLessEqual(audit['probe_relative_change'], .01)
                if d >= 300_000:
                    self.assertGreater(audit['refined_report_bands'], 0)
                np.testing.assert_allclose(np.asarray(cell.sum(axis=0)), 1., atol=1e-11, rtol=0.)
                # Independent Gauss-Legendre reference: integrate on both sides of the known synthetic narrow peak to directly test
                # the corresponding continuous cell's absolute normalization; comparing only two grid levels cannot detect a peak missed by both.
                w = d * (d - 1) // 2
                rho = .9947
                j = np.searchsorted(edges, rho) - 1
                center = w * rho
                width = w * channel.scale(cap) / maximum
                lo, hi = w * edges[j:j + 2]
                x, weight = np.polynomial.legendre.leggauss(80)
                cuts = np.unique(np.clip(center + width * np.array(
                    [-128., -32., -8., -2., 0., 2., 8., 32., 128.]), lo, hi))
                cuts = np.unique(np.r_[lo, cuts, hi])
                reference = 0.
                for a, b in zip(cuts[:-1], cuts[1:]):
                    t = np.rint((a + b) / 2 + (b - a) * x / 2).astype(np.int64)
                    density = channel.density(d, cap, t, [maximum * rho])[0]
                    reference += (b - a) / 2 * (weight @ density) / (hi - lo)
                actual = (channel.density(d, cap, keys, [maximum * rho]) @ cell)[0, j + 1]
                self.assertAlmostEqual(actual / reference, 1., delta=.012)


if __name__ == '__main__':
    unittest.main()
