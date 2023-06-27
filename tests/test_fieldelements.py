import unittest
import numpy as np
from element_base import ElementBase
from point_source import PointSource
from _test_util import *

class TestPointSource(ArrayComparingTest):

    def test_field_at_points_positive(self):

        source = PointSource(np.array([3, 4]), 25)

        inps = np.array([
            [0, 0],
            [6, 8],
            [1, 2],
            [np.inf, np.inf],
            [np.inf, -np.inf],
            [-np.inf, np.inf],
            [-np.inf, -np.inf]
        ])

        exps = np.array([
            1.0,
            1.0,
            3.125,
            0.0,
            0.0,
            0.0,
            0.0
        ])

        outs = source.get_field_at(inps)

        self.compare_arrs(outs, exps)

    def test_field_at_points_negative(self):

        source = PointSource(np.array([3, 4]), -25)

        inps = np.array([
            [0, 0],
            [6, 8],
            [1, 2],
            [np.inf, np.inf],
            [np.inf, -np.inf],
            [-np.inf, np.inf],
            [-np.inf, -np.inf]
        ])

        exps = np.array([
            -1.0,
            -1.0,
            -3.125,
            0.0,
            0.0,
            0.0,
            0.0
        ])
