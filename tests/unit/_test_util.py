import numpy as np


def compare_arrs(a: np.ndarray, b: np.ndarray):

    assert a.ndim == b.ndim, "Arrays have different dimensionality"
    assert a.shape == b.shape, "Arrays have different shapes"

    assert np.all(np.isclose(a, b)), "Arrays' elements differ"
