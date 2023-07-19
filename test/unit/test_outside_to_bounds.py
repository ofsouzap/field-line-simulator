import numpy as np
from vectors import outside_bounds
from test._test_util import *

def test_within():

    xs = np.array([
        [0.0, -5.0, -100.0],
        [0.5, -1, 0],
        [3.0, 0.0, np.inf],
        [0.5, -2.0, 100.0]
    ], dtype=float)

    bounds = np.array([
        [0.0, 3.0],
        [-np.inf, 0],
        [-np.inf, np.inf]
    ], dtype=float)

    exp = np.zeros(shape=(xs.shape[0]), dtype=bool)

    out = outside_bounds(xs, bounds)

    compare_arrs(out, exp)

def test_outside():

    xs = np.array([

        [np.inf, 0.0, -1.0],  # First incorrect
        [2.0, 1.2, 5.0],  # Second incorrect
        [1.0, -0.5, 11.0],  # Third incorrect

        [-1.0, -np.inf, -3.5],  # Only third correct
        [-5.0, 1.0, -11.0],  # Only second correct
        [4.9, -0.3, -53.2],  # Only first correct

    ], dtype=float)

    bounds = np.array([
        [0.0, 5.0],
        [-1.0, 1.0],
        [-10.0, 10.0]
    ], dtype=float)

    exp = np.ones(shape=(xs.shape[0]), dtype=bool)

    out = outside_bounds(xs, bounds)

    compare_arrs(out, exp)

def test_mixture():

    xs = np.array([
        [0.5, -4.0],
        [3.0, 4.0],
        [3.5, -10],
        [0.2, -3],
        [0.2, 0.2],
        [-np.inf, 0.5],
    ], dtype=float)

    bounds = np.array([
        [0.0, 1.0],
        [-np.inf, 0],
    ], dtype=float)

    exp = np.array([
        False,
        True,
        True,
        False,
        True,
        True
    ], dtype=bool)

    out = outside_bounds(xs, bounds)

    compare_arrs(out, exp)
