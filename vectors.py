from typing import Callable
import numpy as np

EPS = 1e-6

def line_seg_distance_to_point(seg_start: np.ndarray, seg_end: np.ndarray, r: np.ndarray):

    vec_se = seg_end - seg_start  # start -> end
    vec_sr = r - seg_start  # start -> r
    vec_er = r - seg_end  # end -> r

    if np.dot(vec_se, vec_er) > 0:

        return np.linalg.norm(vec_er)

    elif np.dot(vec_se, vec_sr) < 0:

        return np.linalg.norm(vec_sr)

    else:

        return np.linalg.norm(np.cross(vec_se, vec_sr) / np.linalg.norm(vec_se))

def grad(field_func: Callable[[np.ndarray], float], pos: np.ndarray) -> np.ndarray:
    """Approximates the gradient vector of a scalar field

Parameters:

    field - a function that takes a point in the field and returns the fields value at that point

    pos - the position at which to evaluate the gradient
"""

    grad = np.empty_like(pos)

    for i in range(grad.shape[0]):

        eps_vec = np.zeros_like(pos)
        eps_vec[i] += EPS

        right_grad = (field_func(pos + eps_vec) - field_func(pos)) / EPS
        left_grad = (field_func(pos) - field_func(pos - eps_vec)) / EPS

        avg_grad = np.average([right_grad, left_grad])

        grad[i] = avg_grad

    return grad
