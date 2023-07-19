from test._test_util import *
import numpy as np
from vectors import line_seg_closest_point

def test_line_vertices():

    starts = np.array([
        [0.0, 0.0],
        [-1.2, 3.4],
        [2.3, 7.8],
        [-2.3, -6.4],
        [0.0, -3.4]
    ])

    ends = np.array([
        [5.8, 4.2],
        [-7.5, 9.6],
        [2, 4],
        [-3.4, -6.5],
        [-0.1, -0.1]
    ])

    compare_arrs(line_seg_closest_point(starts, ends, starts), starts)
    compare_arrs(line_seg_closest_point(starts, ends, ends), ends)

def test_along_line():

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

    compare_arrs(line_seg_closest_point(starts, ends, rs), rs)

def test_closer_to_line():

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

    exps = np.array([
        [-0.75, -1.55],
        [-0.2115384615, -1.6576923077],
        [ 0.3076923077, -1.7615384615],
        [ 0.9230769231, -1.8846153846],
        [ 1.2692307692, -1.9538461538]
    ])

    outs = line_seg_closest_point(starts, ends, rs)

    compare_arrs(outs, exps)

def test_closer_to_vertices():

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

    outs = line_seg_closest_point(starts, ends, rs)

    compare_arrs(outs, np.concatenate((starts[:2], ends[2:])))

def test_mixed():

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

    exps = np.array([
        starts[0],
        [ 8.4224781178, -1.6697808678],
        [ 6.5056490161, -2.219690856],
        [ 2.9529641815, -3.2389037184],
        ends[4]
    ])

    outs = line_seg_closest_point(starts, ends, rs)

    compare_arrs(outs, exps)
