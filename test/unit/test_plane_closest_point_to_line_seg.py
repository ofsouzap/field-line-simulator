from test._test_util import *
from vectors import plane_closest_point_to_line_seg, many_normalise
import numpy as np


def test_through_plane():

    line_as = np.array([
        [-1.0, -1.0],
        [-5.0, 1.0],
        [4.0, 7.0],
        [3.4, 10.0],
    ])

    line_bs = np.array([
        [1.0, 1.0],
        [5.0, -1.0],
        [-2.0, 0.0],
        [-1.0, -10.0],
    ])

    plane_poss = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    plane_norms = many_normalise(np.array([
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]))

    exps = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [-1.0769230769, 1.0769230769],
        [0.9836065574, -0.9836065574],
    ])

    outs = plane_closest_point_to_line_seg(plane_poss, plane_norms, line_as, line_bs)

    compare_arrs(outs, exps)


def test_closer_to_vertices():

    line_as = np.array([
        [-14, 3.5],  # closer
        [4.0, 5.0],  # closer
        [6.0, -6.0],
        [0.0, 3.5],  # closer
    ])

    line_bs = np.array([
        [-11.2, 11.0],
        [-8.4, 0.5],
        [7.0, -1.8],  # closer
        [-10.0, 1.0],
    ])

    plane_poss = np.array([
        [1.0, 1.5],
        [1.0, 1.5],
        [1.0, 1.5],
        [1.0, 1.5],
    ])

    plane_norms = many_normalise(np.array([
        [-3.0, 4.5],
        [-3.0, 4.5],
        [-3.0, 4.5],
        [-3.0, 4.5],
    ]))

    exps = np.array([
        [-8.4615384615, -4.8076923077],
        [ 4.6923076923,  3.9615384615],
        [ 3.6307692308,  3.2538461538],
        [ 1.2307692308,  1.6538461538],
    ])

    outs = plane_closest_point_to_line_seg(plane_poss, plane_norms, line_as, line_bs)

    compare_arrs(outs, exps)
