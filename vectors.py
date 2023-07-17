from typing import Callable
import numpy as np
import vectors
from settings import EPS


def many_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Takes two arrays of vectors and computes a result array comprising of the dot (aka scalar) products of the corresponding pairs of the vectors

Parameters:

    a - a NxM array of vectors

    b - a NxM array of vectors

Returns:

    dots - a N-element 1D vector such that dots[i] = a[i] . b[i] where '.' represents the dot (aka scalar) product of two vectors
"""

    assert a.ndim == b.ndim == 2, "Incorrect input dimensionality"
    assert a.shape[0] == b.shape[0], "Inputs have different numbers of elements"
    assert a.shape[1] == b.shape[1], "Inputs' vectors have different numbers of components"

    return np.diagonal(a @ b.T)


def line_sqr_distance_to_point(line_as: np.ndarray, line_bs: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Calculates the minimum distance between multiple points and corresponding infinite lines"""

    assert line_as.ndim == 2, "Invalid input dimensionality"
    assert line_bs.ndim == 2, "Invalid input dimensionality"
    assert rs.ndim == 2, "Invalid input dimensionality"
    assert line_as.shape == line_bs.shape == rs.shape, "Inputs don't have the same shape"

    vecs_se = line_bs - line_as  # start -> end
    line_dirs = vecs_se / vectors.magnitudes(vecs_se)[:, np.newaxis]  # The normalised direction vectors of the lines
    vecs_sr = rs - line_as  # start -> r

    dots = many_dot(vecs_sr, line_dirs)

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

    end_dots = many_dot(vecs_se, vecs_er)
    start_dots = many_dot(vecs_se, vecs_sr)

    return np.where(
        np.all(np.isclose(seg_starts, seg_ends), axis=1),
        dists_to_s,  # If start and end are same then just use distance to the point
        np.where(
            end_dots > 0,
            dists_to_e,
            np.where(
                start_dots < 0,
                dists_to_s,
                dists_to_line
            )
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

    on_line_dots = many_dot(vecs_sr, line_dirs)

    # Decide between the possible distance values

    end_dots = many_dot(vecs_se, vecs_er)
    start_dots = many_dot(vecs_se, vecs_sr)
    line_seg_is_point = np.all(np.isclose(seg_starts, seg_ends), axis=1)

    end_dots_mat = np.tile(end_dots, (seg_starts.shape[1], 1)).T
    start_dots_mat = np.tile(start_dots, (seg_starts.shape[1], 1)).T
    line_seg_is_point_mat = np.tile(line_seg_is_point, (seg_starts.shape[1], 1)).T

    return np.where(
        line_seg_is_point_mat,
        seg_starts,
        np.where(
            end_dots_mat >= 0,
            seg_ends,  # The end of the line segment
            np.where(
                start_dots_mat <= 0,
                seg_starts,  # The start of the line segment
                seg_starts + (on_line_dots[:, np.newaxis] * line_dirs)  # A position along the line segment
            )
        )
    )


def plane_distance_to_point(plane_poss: np.ndarray, plane_norms: np.ndarray, rs: np.ndarray) -> np.ndarray:
    """Calculates the minimum distances between points and corresponding infinite planes. Assumes that the plane normals are unit vectors"""

    assert plane_poss.ndim == plane_norms.ndim == rs.ndim == 2
    assert plane_poss.shape[0] == plane_norms.shape[0] == rs.shape[0]
    assert plane_poss.shape[1] == plane_norms.shape[1] == rs.shape[1]

    plane_origin_displacements = plane_poss - rs

    min_displacements = many_dot(
        plane_origin_displacements,
        plane_norms
    )

    distances = np.abs(min_displacements)

    return distances


def grad(field_func: Callable[[np.ndarray], np.ndarray], poss: np.ndarray) -> np.ndarray:
    """Approximates the gradient vector of a scalar field at some positions. At singularities, a grad value of 0 is used

Parameters:

    field_func - a function that takes a 2D array of position vectors in the field and returns the field's values at those positions

    poss - the positions at which to evaluate the gradient
"""

    assert poss.ndim == 2, "Invalid input dimensionality"

    if poss.shape[0] == 0:
        return np.zeros_like(poss)

    grad = np.empty_like(poss)

    for i in range(grad.shape[1]):

        # Create the epsilon vector (the small amount to move in each direction)

        eps_vec = np.zeros(shape=poss.shape[1])
        eps_vec[i] += EPS

        right_grads = (field_func(poss + eps_vec) - field_func(poss)) / EPS
        left_grads = (field_func(poss) - field_func(poss - eps_vec)) / EPS

        paired_grads = np.stack([right_grads, left_grads]).T

        avg_grads = np.average(paired_grads, axis=1)

        grad[:, i] = np.where(
            np.isinf(field_func(poss)),
            np.zeros(shape=(avg_grads.shape[0],)),
            avg_grads
        )

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


def many_normalise(vecs: np.ndarray) -> np.ndarray:
    """Takes a 2D array of vectors and returns a 2D array of the unit vectors in the same directions as the inputs"""

    assert vecs.ndim == 2

    return vecs / np.tile(magnitudes(vecs), (vecs.shape[1],1)).T


def outside_bounds(vecs: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Finds which vectors are outside of the provided bounds

Parameters:

    vecs - a NxM array of vectors to be considered

    bounds - a Mx2 array where each pair is a lower and upper bound (respectively) for values of some component of the vectors

Returns:

    outside_bounds - a M-element 1D array of booleans describing which of the vectors in vecs have any component that is outside of its corresponding bound
"""

    assert vecs.ndim == bounds.ndim == 2, "Invalid input dimensionality"
    assert vecs.shape[1] == bounds.shape[0], "Bounds and vectors don't have same number of components"
    assert np.all(bounds[:, 0] <= bounds[:, 1]), "Bounds must have the first value lesser than or equal to the second value"

    outside_of_lower = np.any(vecs[:, :] < bounds[:, 0], axis=1)
    outside_of_upper = np.any(vecs[:, :] > bounds[:, 1], axis=1)

    outside_bounds = np.logical_or(outside_of_lower, outside_of_upper)

    return outside_bounds


def mat_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """Takes a boolean array representing a mask and returns the 2D array representing the mask working in 2 dimensions

Parameters:

    mask - the original mask to use

    n - the number of values the output should have on its second axis (aka axis 1)

Returns:

    mask_mat - the matrix/2D version of the mask such that mask_mat[i,j] = mask[i] for all 0 <= j <= n
"""

    return np.tile(mask, (n, 1)).T
