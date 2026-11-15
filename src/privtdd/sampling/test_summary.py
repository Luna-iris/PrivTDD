"""Check pairing: frontend timings survive backend failure, and historical and current timings are separate by default."""
from pathlib import Path
import csv
import tempfile
import unittest
from unittest.mock import patch

from privtdd.experiments import common as C
from privtdd import data
from .summarize import summarize


class SummaryTests(unittest.TestCase):
    def test_missing_full_or_incomplete_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controls, jobs = [], []
            for seed in (1, 2, 3, 4):
                controls.append(dict(dataset="fixture", epsilon=2, seed=seed, rate=1.,
                                     status="failed" if seed == 2 else "complete", visits=100, candidates=100,
                                     EMD=None if seed == 2 else 4., collection_seconds=10., timing_comparable=False))
                job = dict(dataset="fixture", epsilon=2, seed=seed, rate=.1, id=str(seed))
                jobs.append(job)
                directory = root / "jobs" / job["id"]
                result = directory / "result.json"
                C.write_json(result, dict(job, status="complete", visits=10, candidates=100,
                                          collection_seconds=2., EMD=seed, timing_comparable=False))
                if seed != 3:
                    C.write_json(directory / "complete.json", dict(result_sha256=data.digest(result)))
            C.write_json(root / "plan.json", dict(jobs=jobs))
            C.write_json(root / "controls.json", controls)
            with patch(__package__ + ".summarize.plot_summary"):
                summarize(root)
            with (root / "summary" / "contrast_10pct.csv").open(encoding="utf-8-sig", newline="") as f:
                row, = csv.DictReader(f)
            self.assertEqual(int(row["sample_minus_full_EMD_n"]), 2)
            self.assertEqual(float(row["sample_minus_full_EMD"]), -1.5)
            self.assertEqual(int(row["sample_over_full_visits_n"]), 4)
            self.assertEqual(float(row["historical_full_over_sample_collection_seconds"]), 5.)
            self.assertEqual(int(row["speedup_collection_seconds_n"]), 0)
