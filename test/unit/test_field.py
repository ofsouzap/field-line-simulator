import numpy as np
from field import Field
from field_element import PointSource
from test._test_util import *


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
        5.0,
        5.0,
        8.83883476,
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
        -5.0,
        -5.0,
        -8.83883476,
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
        -1.18811644,
        0.73223305,
        -0.73223305,
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
        2.82843712,
        0.89420038,
        2.14644661,
        1.26776695,
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
        4.05883484,
        2.87774682,
        0.93958725,
        2.22965164,
        1.36776695,
        0.92390275,
        0.0,
        0.0,
        0.0,
        0.0
    ])

    outs = field.evaluate(inps)

    compare_arrs(outs, exps)
