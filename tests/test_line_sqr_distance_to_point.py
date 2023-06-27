from _test_util import *
from vectors import line_sqr_distance_to_point
import numpy as np

class TestLineSqrDistanceToPoint(ArrayComparingTest):

    def test_along_line(self):

        starts = np.tile(
            np.array([-1, -3/2, 1]),
            (5, 1)
        )

        ends = np.tile(
            np.array([3/2, -2, -5]),
            (5, 1)
        )

        rs = np.array([
            [-0.75, -1.55, 0.4],
            [ 0.25, -1.75, -2.],
            [ 0.075, -1.715, -1.58],
            [ 0.95, -1.89, -3.68],
            [-0.4725, -1.6055, -0.266]
        ])

        self.compare_arrs(line_sqr_distance_to_point(starts, ends, rs), np.zeros(shape=(starts.shape[0],)))

    def test_closer_to_line(self):

        starts = np.tile(
            np.array([-1, -3/2]),
            (5, 1)
        )

        ends = np.tile(
            np.array([3/2, -2]),
            (5, 1)
        )

        rs = np.array([
            [-0.7, -1.3],
            [-0.2, -1.6],
            [0.3, -1.8],
            [0.9, -2],
            [1.3, -1.8]
        ])

        exps_sqrt = np.array([
            0.2549509757,
            0.0588348405,
            0.039223227,
            0.1176696811,
            0.1568929081
        ])

        exps = np.square(exps_sqrt)

        outs = line_sqr_distance_to_point(starts, ends, rs)

        self.compare_arrs(outs, exps)

    def test_closer_to_points(self):

        starts = np.tile(
            np.array([-1, -3/2]),
            (4, 1)
        )

        ends = np.tile(
            np.array([3/2, -2]),
            (4, 1)
        )

        rs = np.array([
            [-1.9, 0],
            [-1.6, -2.8],
            [2.1, -2.6],
            [2.8, -2]
        ])

        exps_sqrt = np.array([
            1.2943664919,
            1.3924245595,
            0.4706787243,
            0.2549509757
        ])

        exps = np.square(exps_sqrt)

        outs = line_sqr_distance_to_point(starts, ends, rs)

        self.compare_arrs(outs, exps)

    def test_mixed(self):

        starts = np.tile(
            np.array([12.5, -0.5]),
            (5, 1)
        )

        ends = np.tile(
            np.array([0.3, -4.0]),
            (5, 1)
        )

        rs = np.array([
            [14, 5.6],
            [7.8, 0.5],
            [6.5, -2.2],
            [3.2, -4.1],
            [-2.3, -8.1]
        ])

        exps_sqrt = np.array([
            5.4498366465,
            2.2573054781,
            0.0204851457,
            0.8958311793,
            3.2240467771
        ])

        exps = np.square(exps_sqrt)

        outs = line_sqr_distance_to_point(starts, ends, rs)

        self.compare_arrs(outs, exps)
