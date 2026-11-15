import argparse
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from privtdd.evaluation.metrics import compare
from privtdd.experiments import forpaper
from privtdd.experiments.reproduce import repository_root, verify_snapshot, render_e4
from privtdd.experiments.fresh_render import sampling_with_e1_controls


class ReproductionChecks(unittest.TestCase):
    def test_sparse_integer_domain_metrics(self):
        result=compare([0,2],[2,2],[0,1,3],[1,2,1],n=4,public_cap=3)
        self.assertAlmostEqual(result['MSE'],2.5)
        self.assertAlmostEqual(result['W1'],.75)
        self.assertAlmostEqual(result['KS'],.25)

    def test_frozen_coverage_and_integrity(self):
        result=verify_snapshot(repository_root())
        self.assertEqual((result['e1_complete'],result['e1_planned']),(1916,1920))
        self.assertEqual((result['e2_matched'],result['e3_complete']),(476,234))

    def test_changed_snapshot_fails_before_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'reproduction').mkdir()
            evidence=root/'results/published/paper-v1/e1/releases.csv'
            evidence.parent.mkdir(parents=True)
            evidence.write_text('changed')
            manifest={'frozen_files':[{'path':evidence.relative_to(root).as_posix(),'sha256':'0'*64}]}
            (root/'reproduction/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Frozen evidence changed'):
                verify_snapshot(root)

    def test_e4_uses_ten_full_controls_and_five_sampled(self):
        root=repository_root()/'results/published/paper-v1'
        e1=pd.read_csv(root/'e1/releases.csv',float_precision='round_trip')
        e4=pd.read_csv(root/'e4/large40_runs.csv',float_precision='round_trip')
        with tempfile.TemporaryDirectory() as tmp:
            summary=render_e4(e1,e4,Path(tmp))
            np.testing.assert_array_equal(summary.loc[summary.rate==1,'n'],10)
            np.testing.assert_array_equal(summary.loc[summary.rate<1,'n'],5)
            value=summary.query("dataset == 'Higgs social' and rate == 1 and metric == 'EMD'")['mean'].iloc[0]
            self.assertEqual(round(value,2),73.65)
            shortened=e1[~((e1.dataset=='Higgs social') & (e1.epsilon==2) & (e1.method=='PrivTDD') & (e1.seed>=20261033))]
            with self.assertRaisesRegex(ValueError,'Unexpected E4 seed set'):
                render_e4(shortened,e4,Path(tmp))

    def test_plan_matches_actual_paper_scope(self):
        parser=forpaper.C.parser('forpaper');forpaper.configure(parser)
        args=parser.parse_args([])
        tasks=forpaper.make_tasks(args)
        stages=args.design_summary['stages']
        self.assertEqual(stages,{'E1':1920,'E2':480,'E2-extra':411,'E3':80})
        self.assertEqual(len(tasks),2891)
        self.assertFalse(any('estimator' in task for task in tasks))
        extra=[t for t in tasks if t['panel']=='ablation']
        self.assertEqual(sum('w/o Beta reference' in t['variants'] for t in extra),411)
        self.assertEqual(sum('w/o Post' in t['variants'] for t in extra),410)

    def test_fresh_sampling_full_row_uses_all_completed_e1_controls(self):
        sample=pd.DataFrame([dict(dataset='higgs',epsilon=2.,seed=s,rate=r,MSE=99.)
                             for r in (.01,1.) for s in range(5)])
        releases=pd.DataFrame([dict(dataset='higgs',epsilon=2.,seed=s,panel='main',
                                    method='PrivTDD',MSE=float(s)) for s in range(10)])
        actual=sampling_with_e1_controls(sample,releases)
        self.assertEqual(len(actual[actual.rate==1]),10)
        self.assertEqual(actual[actual.rate==1].MSE.mean(),4.5)
        self.assertEqual(len(actual[actual.rate<1]),5)
        partial=sampling_with_e1_controls(sample,releases.iloc[:8])
        self.assertEqual(len(partial[partial.rate==1]),8)
        smoke=sampling_with_e1_controls(sample,releases,smoke=True)
        self.assertEqual(len(smoke[smoke.rate==1]),5)
        self.assertEqual(smoke[smoke.rate==1].MSE.mean(),99.)


if __name__=='__main__':unittest.main()
