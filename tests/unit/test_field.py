import numpy as np
from field import Field
from field_element import PointSource
from _test_util import *


def test_single_point_positive():

    field = Field()

    field.add_element(PointSource(np.array([3, 4]), 25))

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

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)


def test_single_point_negative():

    field = Field()

    field.add_element(PointSource(np.array([3, 4]), -25))

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

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)


def test_dipole_equal_points():

    field = Field()

    field.add_element(PointSource(np.array([-1, 0]), 5))
    field.add_element(PointSource(np.array([1, 0]), -5))

    inps = np.array([
        [0, 0],
        [0, 1],
        [2.2, 1.5],
        [-1, -2],
        [1, -2],
        [np.inf, np.inf],
        [np.inf, -np.inf],
        [-np.inf, np.inf],
        [-np.inf, -np.inf]
    ])

    exps = np.array([
        0.0,
        0.0,
        -0.9546932939,
        0.625,
        -0.625,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)


def test_dipole_unequal_points():

    field = Field()

    field.add_element(PointSource(np.array([-1, 0]), 5))
    field.add_element(PointSource(np.array([1, 0]), -1))

    inps = np.array([
        [0, 0],
        [0, 1],
        [2.2, 1.5],
        [-1, -2],
        [1, -2],
        [np.inf, np.inf],
        [np.inf, -np.inf],
        [-np.inf, np.inf],
        [-np.inf, -np.inf]
    ])

    exps = np.array([
        4.0,
        2.0,
        0.1293175462,
        1.125,
        0.375,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)


def test_tripole_0():

    field = Field()

    field.add_element(PointSource(np.array([-1, 0]), 5))
    field.add_element(PointSource(np.array([1, 0]), -1))
    field.add_element(PointSource(np.array([1, -5]), 0.3))

    inps = np.array([
        [0, 0],
        [0, 1],
        [2.2, 1.5],
        [-1, -2],
        [1, -2],
        [1, -6],
        [np.inf, np.inf],
        [np.inf, -np.inf],
        [-np.inf, np.inf],
        [-np.inf, -np.inf]
    ])

    exps = np.array([
        4.011538462,
        2.008108108,
        0.136184106,
        1.148076923,
        0.4083333333333,
        0.3972222222222,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)
