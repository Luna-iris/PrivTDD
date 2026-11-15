"""I/O failure regression tests; no graph experiments or estimator changes."""
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from privtdd.io.storage import atomic_text_write, json_write, _DEFERRED_TEXT_WRITES
from privtdd.io.runner import write_batch_summary


class IORegression(unittest.TestCase):
    def setUp(self):
        _DEFERRED_TEXT_WRITES.clear()
        self.enterContext(patch('privtdd.io.storage.time.sleep'))

    def tearDown(self):
        _DEFERRED_TEXT_WRITES.clear()

    def test_transient_replace_retries_and_preserves_old_file(self):
        original = Path.replace
        attempts = []
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'result.json'
            target.write_text('old')
            def temporarily_busy(source, dest):
                attempts.append(1)
                if len(attempts) < 3:
                    self.assertEqual(target.read_text(), 'old')
                    raise PermissionError('busy')
                return original(source, dest)
            with patch.object(Path, 'replace', temporarily_busy):
                self.assertTrue(json_write(target, {'ok': True}))
            self.assertEqual(json.loads(target.read_text()), {'ok': True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(list(Path(folder).glob('*.tmp')), [])

    def test_persistent_progress_lock_warns_and_throttles_then_recovers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'progress.json'
            with redirect_stderr(io.StringIO()) as log:
                with patch.object(Path, 'replace', side_effect=PermissionError('busy')) as replace:
                    self.assertFalse(json_write(path, {'stage': 'degree_fit'}))
                    self.assertFalse(json_write(path, {'stage': 'degree_fit'}))
                    self.assertEqual(replace.call_count, 6)
            self.assertIn('may be stale', log.getvalue())
            _DEFERRED_TEXT_WRITES.clear()  # Simulate the cooldown expiring.
            self.assertTrue(json_write(path, {'stage': 'degree_fit'}))

    def test_required_metadata_and_completion_still_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Path, 'replace', side_effect=PermissionError('busy')):
                for name, value in [('case_manifest.json', {}), ('metrics.json', {}),
                                    ('joint_warm_start.json', {}), ('progress.json', {'stage': 'complete'})]:
                    with self.assertRaises(PermissionError):
                        json_write(Path(folder)/name, value)

    def test_busy_csv_does_not_prevent_other_summary_writes(self):
        original = Path.replace
        def lock_results(source, dest):
            if Path(dest).name == 'results.csv':
                raise PermissionError('busy spreadsheet')
            return original(source, dest)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cases = [dict(dataset='example', epsilon=2.)]
            rows = {'one': dict(dataset='example', epsilon=2., status='complete', MSE=1., EMD=2., KS=.1)}
            with redirect_stderr(io.StringIO()), patch.object(Path, 'replace', lock_results):
                self.assertFalse(write_batch_summary(root, cases, rows))
            self.assertTrue((root/'summary.csv').exists())
            self.assertEqual(json.loads((root/'batch_progress.json').read_text())['completed'], 1)
            _DEFERRED_TEXT_WRITES.clear()
            self.assertTrue(write_batch_summary(root, cases, rows))
            self.assertTrue((root/'results.csv').exists())

    def test_required_write_creation_failure_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                atomic_text_write(Path(folder)/'missing'/'identity.json', '{}')


if __name__ == '__main__':
    unittest.main()
