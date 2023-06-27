import unittest
import numpy as np
from vectors import sqr_magnitudes, magnitudes
from _test_util import *

class TestMagnitudes(ArrayComparingTest):

    def do_test(self, inp: np.ndarray, exp: np.ndarray):

        sqr_exp = np.square(exp)

        out = magnitudes(inp)
        sqr_out = sqr_magnitudes(inp)

        self.compare_arrs(sqr_out, sqr_exp)
        self.compare_arrs(out, exp)

    def test_single(self):

        inp = np.array([2.1, 5.7, 7.3])
        exp = np.array([9.49684158])

        self.do_test(inp, exp)

    def test_many(self):

        inp = np.array([
            [56.2, 5.3, 7.3],
            [-43.0, 0, -6.4],
            [1.2, 5.4, 80],
            [1, 2, 3],
            [0, 0, 0],
            [-1, -2, -5.3],
            [-4, 0, -75]
        ])

        exp = np.array([
            56.91941672,
            43.47367019,
            80.19102194,
            3.741657387,
            0,
            5.752390807,
            75.10659092
        ])
