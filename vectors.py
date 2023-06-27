from typing import Callable
import numpy as np
import vectors

EPS = 1e-6

def line_sqr_distance_to_point(line_as: np.ndarray, line_bs: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Calculates the minimum distance between multiple points and corresponding infinite lines"""

    assert line_as.ndim == 2, "Invalid input dimensionality"
    assert line_bs.ndim == 2, "Invalid input dimensionality"
    assert rs.ndim == 2, "Invalid input dimensionality"
    assert line_as.shape == line_bs.shape == rs.shape, "Inputs don't have the same shape"

    vecs_se = line_bs - line_as  # start -> end
    line_dirs = vecs_se / vectors.magnitudes(vecs_se)[:, np.newaxis]  # The normalised direction vectors of the lines
    vecs_sr = rs - line_as  # start -> r

    dots = np.diagonal(vecs_sr @ line_dirs.T)

    dists = vectors.sqr_magnitudes(vecs_sr - (dots[:, np.newaxis] * line_dirs))

    return dists

def line_distance_to_point(line_as: np.ndarray, line_bs: np.ndarray, rs: np.ndarray) -> np.ndarray:
    return np.sqrt(line_sqr_distance_to_point(line_as, line_bs, rs))

def line_seg_sqr_distance_to_point(seg_starts: np.ndarray, seg_ends: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Calculates the minimum square distance between multiple points and corresponding line segments.
Note that the ordering of seg_starts, seg_ends and rs must correspond to each other \
so that the point `rs[i]` can be compared to the line segment connecting `seg_starts[i]` and `seg_ends[i]`

Parameters:

    seg_starts - a 2D array of the position vectors of the starts of each of the line segments

    seg_ends - a 2D array of the position vectors of the ends of each of the line segments

    rs - a 2D array of the position vectors of the points to measure to each of the line segments
"""

    assert seg_starts.ndim == 2, "Invalid input dimensionality"
    assert seg_ends.ndim == 2, "Invalid input dimensionality"
    assert rs.ndim == 2, "Invalid input dimensionality"
    assert seg_starts.shape == seg_ends.shape == rs.shape, "Inputs don't have the same shape"

    # Create vectors and calculate possible distances

    vecs_se = seg_ends - seg_starts  # start -> end
    vecs_sr = rs - seg_starts  # start -> r
    vecs_er = rs - seg_ends  # end -> r

    dists_to_e = vectors.sqr_magnitudes(vecs_er)
    dists_to_s = vectors.sqr_magnitudes(vecs_sr)
    dists_to_line = line_sqr_distance_to_point(seg_starts, seg_ends, rs)

    # Decide between the possible distance values

    end_dots = np.diagonal(vecs_se @ vecs_er.T)
    start_dots = np.diagonal(vecs_se @ vecs_sr.T)

    return np.where(
        end_dots > 0,
        dists_to_e,
        np.where(
            start_dots < 0,
            dists_to_s,
            dists_to_line
        )
    )

def line_seg_distance_to_point(seg_starts: np.ndarray, seg_ends: np.ndarray, rs: np.ndarray) -> np.ndarray:
    return np.sqrt(line_seg_sqr_distance_to_point(seg_starts, seg_ends, rs))

def line_seg_closest_point(seg_starts: np.ndarray, seg_ends: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Calculates the closest points on line segments to points.
Note that the ordering of seg_starts, seg_ends and rs must correspond to each other \
so that the point `rs[i]` can be compared to the line segment connecting `seg_starts[i]` and `seg_ends[i]`

Parameters:

    seg_starts - a 2D array of the position vectors of the starts of each of the line segments

    seg_ends - a 2D array of the position vectors of the ends of each of the line segments

    rs - a 2D array of the position vectors of the points to measure to each of the line segments
"""

    assert seg_starts.ndim == 2, "Invalid input dimensionality"
    assert seg_ends.ndim == 2, "Invalid input dimensionality"
    assert rs.ndim == 2, "Invalid input dimensionality"
    assert seg_starts.shape == seg_ends.shape == rs.shape, "Inputs don't have the same shape"

    # Create vectors and calculate possible distances

    vecs_se = seg_ends - seg_starts  # start -> end
    line_dirs = vecs_se / vectors.magnitudes(vecs_se)[:, np.newaxis]  # The normalised direction vectors of the lines
    vecs_sr = rs - seg_starts  # start -> r
    vecs_er = rs - seg_ends  # end -> r

    on_line_dots = np.diagonal(vecs_sr @ line_dirs.T)

    # Decide between the possible distance values

    end_dots = np.diagonal(vecs_se @ vecs_er.T)
    start_dots = np.diagonal(vecs_se @ vecs_sr.T)

    end_dots_mat = np.tile(end_dots, (seg_starts.shape[1], 1)).T
    start_dots_mat = np.tile(start_dots, (seg_starts.shape[1], 1)).T

    return np.where(
        end_dots_mat >= 0,
        seg_ends,  # The end of the line segment
        np.where(
            start_dots_mat <= 0,
            seg_starts,  # The start of the line segment
            seg_starts + (on_line_dots[:, np.newaxis] * line_dirs)  # A position along the line segment
        )
    )

def grad(field_func: Callable[[np.ndarray], np.ndarray], poss: np.ndarray) -> np.ndarray:
    """Approximates the gradient vector of a scalar field at some positions

Parameters:

    field_func - a function that takes a 2D array of position vectors in the field and returns the field's values at those positions

    poss - the positions at which to evaluate the gradient
"""

    assert poss.ndim == 2, "Invalid input dimensionality"

    grad = np.empty_like(poss)

    for i in range(grad.shape[1]):

        # Create the epsilon vector (the small amount to move in each direction)

        eps_vec = np.zeros(shape=poss.shape[1])
        eps_vec[i] += EPS

        right_grads = (field_func(poss + eps_vec) - field_func(poss)) / EPS
        left_grads = (field_func(poss) - field_func(poss - eps_vec)) / EPS

        avg_grads = np.average(np.array([right_grads, left_grads]), axis=0)

        grad[:, i] = avg_grads

    return grad

def sqr_magnitudes(vecs: np.ndarray) -> np.ndarray:
    """Takes a 2D array of vectors or a single vector and returns a 1D array of the squares of the vectors' magnitudes"""

    assert vecs.ndim <= 2, "Invalid input shape"

    if vecs.ndim == 1:
        inp = vecs.reshape((1, vecs.shape[0]))
    else:
        inp = vecs

    sqrs = np.square(inp)
    outs = np.sum(sqrs, axis=1)

    return outs

def magnitudes(vecs: np.ndarray) -> np.ndarray:
    """Takes a 2D array of vectors or a single vector and returns a 1D array of the vectors' magnitudes"""

    return np.sqrt(sqr_magnitudes(vecs))
