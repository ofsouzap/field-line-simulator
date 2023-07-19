import numpy as np
from vectors import estimate_grad, sqr_magnitudes
from _test_util import *

def _test_grad_value(field_func, points: np.ndarray, expectations: np.ndarray) -> None:

    assert points.ndim == 2
    assert points.shape[1] in [2, 3]
    assert expectations.ndim == 2
    assert expectations.shape[0] == points.shape[0]

    output = estimate_grad(field_func, points)

    compare_arrs(output, expectations)

def test_zero():

    field_func = lambda rs: np.zeros(shape=(rs.shape[0],), dtype=rs.dtype)

    points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

    _test_grad_value(field_func, points, np.zeros_like(points))

def test_constant():

    field_func = lambda rs: np.ones(shape=(rs.shape[0],), dtype=rs.dtype) * 1.5

    points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

    _test_grad_value(field_func, points, np.zeros_like(points))

def test_uniform_unidirectional_x():

    field_func = lambda rs: rs[:, 0]

    # points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T
    points = np.array([[0, 0]])

    _test_grad_value(field_func, points, np.tile(
        np.array([1, 0]),
        (points.shape[0], 1)
        ))

def test_uniform_unidirectional_y():

    field_func = lambda rs: rs[:, 1]

    points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

    _test_grad_value(field_func, points, np.tile(
        np.array([0, 1]),
        (points.shape[0], 1)
        ))

def test_uniform_multidirectional():

    field_func = lambda rs: rs[:, 1] - rs[:, 0]

    points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

    _test_grad_value(field_func, points, np.tile(
        np.array([-1, 1]),
        (points.shape[0], 1)
        ))

def test_inf_field():

    field_func = lambda rs: np.repeat(np.inf, rs.shape[0])

    points = np.mgrid[-20:20:np.pi, -20:20:np.pi].reshape(2,-1).T

    _test_grad_value(field_func, points, np.tile(
        np.array([0, 0]),
        (points.shape[0], 1)
        ))

def test_singularity():

    field_func = lambda rs: np.where(
        sqr_magnitudes(rs) == 0,
        np.repeat(np.inf, rs.shape[0]),
        np.reciprocal(sqr_magnitudes(rs))
    )

    points = np.array([[0, 0]])

    exps = np.array([[0, 0]])

    _test_grad_value(field_func, points, exps)
