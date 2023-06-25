import unittest
import numpy as np
from vectors import grad

class TestGrad(unittest.TestCase):

    def _test_grad_value(self, field_func, points: np.ndarray, expectations: np.ndarray) -> None:

        assert points.ndim == 2
        assert points.shape[1] in [2, 3]
        assert expectations.ndim == 2
        assert expectations.shape[0] == points.shape[0]

        for i in range(points.shape[0]):

            point = points[i]
            exp = expectations[i]

            g = grad(field_func, point)

            assert g.ndim == 1
            assert g.shape[0] in [2, 3]

            for j in range(g.shape[0]):
                self.assertAlmostEqual(g[j], exp[j])

    def test_zero(self):

        field_func = lambda r: 0.0

        points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

        self._test_grad_value(field_func, points, np.zeros_like(points))

    def test_constant(self):

        field_func = lambda r: 1.0

        points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

        self._test_grad_value(field_func, points, np.zeros_like(points))

    def test_uniform_unidirectional_x(self):

        field_func = lambda r: r[0]

        points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

        self._test_grad_value(field_func, points, np.tile(
            np.array([1, 0]),
            (points.shape[0], 1)
            ))

    def test_uniform_unidirectional_y(self):

        field_func = lambda r: r[1]

        points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

        self._test_grad_value(field_func, points, np.tile(
            np.array([0, 1]),
            (points.shape[0], 1)
            ))

    def test_uniform_multidirectional(self):

        field_func = lambda r: -r[0] + r[1]

        points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

        self._test_grad_value(field_func, points, np.tile(
            np.array([-1, 1]),
            (points.shape[0], 1)
            ))
