import unittest
import numpy as np
from core import minimize, objective_gaps

class OptimizationTests(unittest.TestCase):
    def test_identity_solved_in_one_step(self):
        b = np.array([2., -3.])
        np.testing.assert_allclose(minimize(np.eye(2), b, 1)[-1], b)
    def test_known_diagonal_optimum(self):
        a = np.diag([1., 9.]); b = np.array([2., 9.])
        np.testing.assert_allclose(minimize(a, b, 100, True)[-1], [2., 1.], atol=1e-9)
    def test_gradient_descent_gap_is_nonincreasing(self):
        a = np.diag([1., 7.]); b = np.array([1., 2.])
        gaps = objective_gaps(a, b, minimize(a, b, 50))
        self.assertTrue(np.all(np.diff(gaps) <= 1e-13))
    def test_initial_iterate_included(self):
        self.assertEqual(minimize(np.eye(2), np.ones(2), 0).shape, (1, 2))
    def test_nonconvex_quadratic_rejected(self):
        with self.assertRaises(ValueError): minimize(np.diag([-1., 2.]), np.ones(2))
