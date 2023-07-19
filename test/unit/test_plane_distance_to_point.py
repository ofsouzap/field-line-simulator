import numpy as np
from vectors import plane_distance_to_point, many_normalise
from test._test_util import *

def test_plane_point():

    poss = np.array([
        [1.0, 4.0, 2.0]
    ])

    norms = np.array([
        [5.4, -4.3, -8.7]
    ])

    rs = np.array([
        [1.0, 4.0, 2.0]
    ])

    exps = np.array([
        0.0
    ])

    outs = plane_distance_to_point(poss, norms, rs)

    compare_arrs(outs, exps)

def test_2d():

    poss = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])

    norm_dirs = np.array([
        [1/3, -1/4],
        [1/3, -1/4],
        [1/3, -1/4],
    ])

    norms = many_normalise(norm_dirs)

    rs = np.array([
        [-10.28, -1.36],
        [7.26, 0.02],
        [-2.1, 5.5],
    ])

    exps = np.array([
        7.408,
        5.796,
        4.98
    ])

    outs = plane_distance_to_point(poss, norms, rs)
    print(outs)
    print(exps)

    compare_arrs(outs, exps)
