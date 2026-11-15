"""Data-format semantics that can change the paper experiment's random stream."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from privtdd.data.prepare import build_csr
from privtdd.data.triangles import validate_and_triangles


class DatasetPreparationTests(unittest.TestCase):
    def build(self, text, policy, n, m):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / 'raw.txt'
            raw.write_text(text)
            build_csr(raw, dict(nodes=n, edges=m, node_mapping=policy, index_dtype='<i4'), root / 'out', root)
            return tuple(np.load(root / 'out' / name) for name in ('indptr.npy', 'indices.npy', 'new_to_original.npy'))

    def test_higgs_first_seen_keeps_loop_only_node_and_deduplicates(self):
        ip, ix, ids = self.build('90 10\n10 90\n90 70\n70 10\n5 5\n', 'first_seen', 4, 3)
        np.testing.assert_array_equal(ids, [90, 10, 70, 5])
        np.testing.assert_array_equal(ip, [0, 2, 4, 6, 6])
        np.testing.assert_array_equal(validate_and_triangles(ip, ix), [1, 1, 1, 0])

    def test_sorted_ids_are_distinct_from_first_seen(self):
        ip, ix, ids = self.build('90 10\n90 70\n70 10\n', 'sorted_observed_ids', 3, 3)
        np.testing.assert_array_equal(ids, [10, 70, 90])
        np.testing.assert_array_equal(validate_and_triangles(ip, ix), [1, 1, 1])

    def test_bio_single_percent_matrix_header_retains_declared_isolates(self):
        ip, ix, ids = self.build('%MatrixMarket matrix coordinate pattern symmetric\n4 4 4\n1 2\n2 1\n2 3\n1 3\n', 'matrix_market_declared', 4, 3)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4])
        np.testing.assert_array_equal(validate_and_triangles(ip, ix), [1, 1, 1, 0])

    def test_wrong_expected_graph_fails(self):
        with self.assertRaises(ValueError):
            self.build('0 1\n', 'sorted_observed_ids', 3, 1)


if __name__ == '__main__':
    unittest.main()
