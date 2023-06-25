import unittest
import numpy as np
from element_base import ElementBase
from point_source import PointSource

class TestPointSource(unittest.TestCase):

    def test_field_at_points_positive(self):

        source = PointSource(np.array([3, 4]), 25)

        self.assertAlmostEqual(
            source.get_field_at(np.array([0, 0])),
            1
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([6, 8])),
            1
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([1, 2])),
            3.125
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([np.inf, np.inf])),
            0
        )

    def test_field_at_points_negative(self):

        source = PointSource(np.array([3, 4]), -25)

        self.assertAlmostEqual(
            source.get_field_at(np.array([0, 0])),
            -1
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([6, 8])),
            -1
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([1, 2])),
            -3.125
        )

        self.assertAlmostEqual(
            source.get_field_at(np.array([np.inf, np.inf])),
            0
        )
