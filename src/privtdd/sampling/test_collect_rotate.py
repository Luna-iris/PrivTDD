"""Test the sampler against a slow implementation that explicitly lists unordered pairs; enumerate all wedges only in small-graph tests."""
from itertools import combinations
import unittest

import numpy as np
from numba import get_num_threads, set_num_threads

from privtdd.experiments import common as C
from privtdd import data
from privtdd.reporting.collection import collect_reports_joint
from privtdd.reporting.primitives import (
    _pair_uniform_numba, lists_to_csr, project_neighbors, rr_params,
)
from .design import _build_rotated_design, _derive_seed
from .collect_rotate import _count_block, collect, matching_parameters


def selected_pairs(neighbors, design, center, k):
    """Construct the reference set from the matching definition, without the production code's cyclic intervals or binary-search enumeration."""
    shift = int(design["center_shift"][center])
    pairs = set()
    neighbors = set(map(int, neighbors))
    for offset in range(k):
        for i, u in enumerate(design["left"]):
            v = design["right"][(i + shift + offset) % design["m"]]
            if u in neighbors and v in neighbors:
                pairs.add(tuple(sorted((int(u), int(v)))))
    return pairs


class FrontendTests(unittest.TestCase):
    def test_every_selected_pair_even_and_odd(self):
        """Let RR return 1 for only one edge at a time; check membership, deduplication, wraparound, and dummy vertices edge by edge."""
        old = get_num_threads()
        set_num_threads(1)
        try:
            for n in (6, 7):
                design = _build_rotated_design(n, 23)
                # The two centers share several neighbors, and the second center starts at a nonzero block offset.
                rows = [sorted(set(range(n)) - {v}) for v in (2, 3)]
                pp, pi = lists_to_csr(rows)
                previous = [set(), set()]
                for k in range(1, design["m"] + 1):
                    sets = [selected_pairs(row, design, i + 2, k) for i, row in enumerate(rows)]
                    for before, after in zip(previous, sets):
                        self.assertTrue(before <= after)
                    previous = sets
                    for u, v in combinations(range(n), 2):
                        ip, ix = data.csr_from_edges(n, [u], [v])
                        hits, ones = _count_block(ip, ix, pp, pi, design["side"], design["pos"], design["right"],
                                                 design["center_shift"], 2, k, 1., 0., 31)
                        np.testing.assert_array_equal(hits, [len(a) for a in sets])
                        np.testing.assert_array_equal(ones, [int((u, v) in a) for a in sets])
        finally:
            set_num_threads(old)

    def test_released_units_pairing_and_diagnostics(self):
        n, seed = 17, 20261028
        edges = {(i, j) for i, j in combinations(range(n), 2) if (3 * i + 7 * j) % 9 < 4}
        ip, ix = data.csr_from_edges(n, *zip(*sorted(edges)))
        alloc = dict(eps_deg=.3, eps_rr=.7, eps_rel=.35)
        rng = np.random.default_rng(seed)
        caps = np.clip(np.floor(np.diff(ip) + rng.laplace(0, 1 / .3, n)), 0, n - 1).astype(np.int64)
        lap = rng.laplace(0, 1, n)
        p, q, gap = rr_params(.7)
        design = _build_rotated_design(n, _derive_seed(seed, 0xC701))
        with C.allocation_context([.3, .35, .35]):
            full = collect_reports_joint(ip, ix, n - 1, 2., seed)
        np.testing.assert_array_equal(caps, full["caps"])
        for rate in (.0001, .1, .25, .5):
            k, s = matching_parameters(n, rate)
            reports, diag = collect(ip, ix, rate, seed, alloc, workers=1, block_size=3)
            visits = occupied = candidates = 0
            for v in range(n):
                kept = project_neighbors(ix[ip[v]:ip[v+1]], caps[v], v, seed + 0x12345)
                pairs = selected_pairs(kept, design, v, k)
                ones = sum(_pair_uniform_numba(seed + 0x5A17, *pair) < (p if pair in edges else q)
                           for pair in pairs)
                centered = ones - q * len(pairs)
                scale = min(k, max(0, int(caps[v]) - 1)) / .35
                expected = (centered + lap[v] * scale) / (s * gap)
                self.assertAlmostEqual(reports["noisy_triangle_report"][v], expected, places=10)
                visits += len(pairs)
                occupied += bool(pairs)
                candidates += len(kept) * (len(kept)-1) // 2
            self.assertEqual((diag["visits"], diag["candidates"]), (visits, candidates))
            self.assertEqual(diag["active_fraction"], occupied/n)
            np.testing.assert_array_equal(reports["caps"], full["caps"])

    def test_conditional_sensitivity_all_neighborhood_edits(self):
        # Fix a center, public cap, and projection order; enumerate every neighborhood and every single adjacency addition.
        # Take the worst-case RR bits for the added and removed sets separately, covering every shared-RR transcript.
        n, center, q = 7, 2, .22
        others = set(range(n)) - {center}
        design = _build_rotated_design(n, 13)
        for k in range(1, design["m"] + 1):
            for cap in range(n):
                sets = {}
                for size in range(n):
                    for row in combinations(sorted(others), size):
                        kept = project_neighbors(row, cap, center, 71)
                        sets[frozenset(row)] = selected_pairs(kept, design, center, k)
                for subset, old in sets.items():
                    for v in others - subset:
                        new = sets[subset | {v}]
                        a, b = len(old-new), len(new-old)
                        worst = max(q*a + (1-q)*b, (1-q)*a + q*b)
                        self.assertLessEqual(worst, min(k, max(cap-1, 0)) + 1e-14)


if __name__ == "__main__":
    unittest.main()
