"""S5 numerical regressions; synthetic data only, no full graph refits."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from privtdd.correction.optimizer import fit_joint
from privtdd.correction.roundoff import refine_joint


class OptimizerRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = threadpool_limits(limits=2)

    @classmethod
    def tearDownClass(cls):
        cls.limits.restore_original_limits()

    @staticmethod
    def cancellation_case():
        n = 10000
        sign = np.r_[np.ones(n//2), -np.ones(n//2)]
        ratios = np.column_stack((99-sign*1000*.03, sign*1000))
        def gradient(t):
            return t-np.sum(ratios[:, 1]/(1+ratios@np.array([1., t])))
        optimum = brentq(gradient, .02, .04, xtol=1e-15)
        family = dict(counts=np.ones(n), predictive_table=np.ones(n), delta_tables=ratios.T,
                      A=np.array([[0., 1.]]), b=np.array([.05]))
        return family, np.array([1., optimum+1.5e-9]), optimum

    def test_cancellation_recovery_matches_independent_scalar_optimum(self):
        family, warm, optimum = self.cancellation_case()
        new, after = refine_joint(family, initial=warm, tolerance=1e-6, quadratic=np.eye(2))
        self.assertGreater(after['history'][0]['convex_gap'], 1e-6)
        self.assertLessEqual(after['convex_gap'], 1e-6)
        self.assertLess(after['history'][0]['stable_objective_change'], 0)
        self.assertAlmostEqual(new[1], optimum, places=12)

    def test_primary_limit_reaches_automatic_recovery(self):
        rng = np.random.default_rng(97)
        family = dict(counts=np.ones(300), predictive_table=np.ones(300),
                      delta_tables=rng.normal(.01, .05, (3, 300)), A=np.ones((1, 3)), b=np.array([2.]))
        cost = np.array([.01, .02, .03])
        expected, _ = fit_joint(family, range(3), max_iterations=1000, linear_cost=cost)
        with patch('privtdd.correction.roundoff.refine_joint', wraps=refine_joint) as recovery:
            actual, info = fit_joint(family, range(3), max_iterations=1, linear_cost=cost)
        recovery.assert_called_once()
        self.assertEqual(info['recovery_trigger'], 'primary_iteration_limit')
        self.assertLessEqual(info['convex_gap'], 1e-6)
        self.assertFalse(info['statistical_ridge_added'])
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)

    def test_recovery_does_not_accept_bad_warm_start_or_unresolved_gap(self):
        family, warm, _ = self.cancellation_case()
        with self.assertRaisesRegex(ArithmeticError, 'objective differs'):
            refine_joint(family, initial=warm, expected_initial_objective=123.)
        with self.assertRaisesRegex(ArithmeticError, 'not feasible'):
            refine_joint(family, initial=np.array([1., .1]))
        with self.assertRaisesRegex(ArithmeticError, 'unresolved after bounded recovery'):
            refine_joint(family, initial=warm, max_iterations=1)

    def test_oblique_active_face_recovers_without_relaxing_constraints(self):
        # The unconstrained Newton direction points out of x+y<=1 at the warm
        # point. A valid tangential step reaches the analytic constrained optimum.
        family = dict(counts=np.ones(1), predictive_table=np.ones(1),
                      delta_tables=np.zeros((2, 1)), A=np.array([[1., 1.]]), b=np.ones(1))
        point, info = refine_joint(family, initial=np.array([.4, .6]),
                                   linear_cost=np.array([-2., -2.]))
        np.testing.assert_allclose(point, [.5, .5], rtol=0, atol=1e-12)
        self.assertLessEqual(float((family['A']@point-family['b']).max()), 2e-12)
        self.assertLessEqual(info['convex_gap'], 1e-6)
        self.assertEqual(info['history'][0]['step_solver'], 'active_face_Newton')
        self.assertLess(info['history'][0]['stable_objective_change'], 0)

    def test_redundant_active_faces_preserve_original_gram(self):
        family = dict(counts=np.ones(1), predictive_table=np.ones(1),
                      delta_tables=np.zeros((3, 1)),
                      A=np.array([[1., 1., 0.], [2., 2., 0.]]), b=np.array([1., 2.]))
        gram = np.diag([2., 1., 3.])
        point, info = refine_joint(family, initial=np.array([.4, .6, 0.]),
                                   quadratic=gram, linear_cost=np.array([-2., -2., 1.]))
        np.testing.assert_allclose(point, [1/3, 2/3, 0.], rtol=0, atol=1e-12)
        self.assertLessEqual(info['convex_gap'], 1e-6)
        self.assertFalse(info['statistical_ridge_added'])

    def test_tangent_construction_roundoff_is_not_a_zero_step_blocker(self):
        from privtdd.correction.roundoff import face_newton_direction
        family = dict(counts=np.ones(1), predictive_table=np.ones(1),
                      delta_tables=np.zeros((3, 1)), A=np.array([[1., 1., 0.]]), b=np.ones(1))
        def rounded_face(*args, **kwargs):
            direction, stats = face_newton_direction(*args, **kwargs)
            # Simulate a normwise SVD backward error: larger than just the dot
            # product error, but much smaller than the original feasibility check.
            direction[0] += 1e-15
            return direction, stats
        with patch('privtdd.correction.roundoff.face_newton_direction', side_effect=rounded_face):
            point, info = refine_joint(family, initial=np.array([.4, .6000000000000001, .2]),
                                       linear_cost=np.array([-2., -2., -.5]))
        np.testing.assert_allclose(point, [.5, .5, .5], rtol=0, atol=1e-12)
        self.assertLessEqual(float((family['A']@point-family['b']).max()), 2e-12)
        self.assertLessEqual(info['convex_gap'], 1e-6)
        self.assertEqual(info['history'][0]['step_solver'], 'active_face_Newton')
        self.assertLess(info['history'][0]['stable_objective_change'], 0)

    def test_aggregate_resumes_matching_seven_coordinate_warm_point(self):
        from privtdd.release.pipeline import fit_joint7
        from privtdd.config import JOINT_COORDINATES
        rng = np.random.default_rng(804)
        family = dict(counts=np.ones(1200), predictive_table=np.ones(1200),
                      delta_tables=np.zeros((16, 1200)), A=np.ones((1, 16)), b=np.ones(1))
        family['delta_tables'][list(JOINT_COORDINATES)] = rng.normal(.002, .04, (7, 1200))
        expected, normalizer, stats = fit_joint7(family)
        with tempfile.TemporaryDirectory() as folder:
            warm = Path(folder)/'joint_warm_start.json'
            record = dict(point=expected[list(JOINT_COORDINATES)].tolist(), objective=stats['objective'])
            warm.write_text(json.dumps(record), encoding='utf-8')
            actual, resumed_normalizer, resumed = fit_joint7(family, directory=folder)
            self.assertEqual(resumed['recovery_trigger'], 'same_workdir_warm_start')
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
            np.testing.assert_allclose(resumed_normalizer, normalizer, rtol=0, atol=1e-12)
            record['objective'] += 1.
            warm.write_text(json.dumps(record), encoding='utf-8')
            with self.assertRaisesRegex(ArithmeticError, 'objective differs'):
                fit_joint7(family, directory=folder)


if __name__ == '__main__':
    unittest.main()
