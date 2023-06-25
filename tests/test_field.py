import unittest
import numpy as np
from field import Field
from point_source import PointSource

class TestField(unittest.TestCase):

    def test_single_point_positive(self):

        field = Field()

        field.add_element(PointSource(np.array([3, 4]), 25))

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 0])),
            1
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([6, 8])),
            1
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, 2])),
            3.125
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([np.inf, np.inf])),
            0
        )

    def test_single_point_negative(self):

        field = Field()

        field.add_element(PointSource(np.array([3, 4]), -25))

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 0])),
            -1
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([6, 8])),
            -1
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, 2])),
            -3.125
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([np.inf, np.inf])),
            0
        )

    def test_dipole_equal_points(self):

        field = Field()

        field.add_element(PointSource(np.array([-1, 0]), 5))
        field.add_element(PointSource(np.array([1, 0]), -5))

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 0])),
            0
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 1])),
            0
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([2.2, 1.5])),
            -0.9546932939
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([-1, -2])),
            0.625
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, -2])),
            -0.625
        )

    def test_dipole_unequal_points(self):

        field = Field()

        field.add_element(PointSource(np.array([-1, 0]), 5))
        field.add_element(PointSource(np.array([1, 0]), -1))

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 0])),
            4
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 1])),
            2
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([2.2, 1.5])),
            0.1293175462
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([-1, -2])),
            1.125
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, -2])),
            0.375
        )

    def test_tripole_0(self):

        field = Field()

        field.add_element(PointSource(np.array([-1, 0]), 5))
        field.add_element(PointSource(np.array([1, 0]), -1))
        field.add_element(PointSource(np.array([1, -5]), 0.3))

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 0])),
            4.011538462
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([0, 1])),
            02.008108108
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([2.2, 1.5])),
            0.136184106
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([-1, -2])),
            1.148076923
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, -2])),
            0.4083333333333
        )

        self.assertAlmostEqual(
            field.evaluate(np.array([1, -6])),
            0.3972222222222
        )
