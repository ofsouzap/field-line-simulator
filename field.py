from typing import List, Tuple, Optional
from element_base import ElementBase
import numpy as np
import vectors

FIELD_LINE_TRACE_DEFAULT_STEP = 5
FIELD_LINE_TRACE_DEFAULT_ELEMENT_STOP_DISTANCE = 1

class ElementNotInFieldException(Exception): pass

class Field:

    def __init__(self):

        self.__elements: List[ElementBase] = []

    def add_element(self, ele: ElementBase) -> None:
        self.__elements.append(ele)

    def remove_element(self, ele: ElementBase) -> None:

        if ele in self.__elements:
            self.__elements.remove(ele)
        else:
            raise ElementNotInFieldException()

    def evaluate(self, poss: np.ndarray) -> np.ndarray:

        vals = np.zeros(shape=(poss.shape[0]))

        for ele in self.__elements:

            vals += ele.get_field_at(poss)

        return vals

    def grad(self, poss: np.ndarray) -> np.ndarray:
        """Takes an array of position vectors and returns the grad of the field at those positions"""
        return vectors.grad(self.evaluate, poss)

    def line_seg_nearest_element(self,
                                 seg_starts: np.ndarray,
                                 seg_ends: np.ndarray,
                                 use_absorbers: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find the absorbing or emitting field elements that the line segments specified are nearest to

Parameters:

    seg_starts - the starting position vectors of the line segments

    seg_ends - the ending position vectors of the line segments

    use_absorbers - True means the function will look for a nearby emitter, False means it will look for absorbers

Returns:

    out_sqr_distances - array of squares of distances of each point from its returned out_positions value

    out_positions - array of position vectors of found field elements' nearest points for each query
"""

        assert seg_starts.ndim == seg_ends.ndim == 2, "Invalid line segment arrays dimentionality"
        assert seg_starts.shape == seg_ends.shape, "Line segment arrays must be of the same shape"

        assert use_absorbers.dtype == bool, "use_absorbers must be a boolean array"
        assert use_absorbers.ndim == 1, "Invalid use_absorbers dimensionality"

        out_sqr_distances = np.inf * np.ones(shape=(seg_starts.shape[0],), dtype=seg_starts.dtype)
        out_positions = np.zeros_like(seg_starts)

        for ele in self.__elements:  # Iterate through each field element

            closest_points = ele.find_line_seg_nearest_point(seg_starts, seg_ends)

            sqr_distances = vectors.line_seg_sqr_distance_to_point(seg_starts, seg_ends, closest_points)

            closer_mask = sqr_distances < out_sqr_distances  # Which points are closer than the current-closest

            # If the element matches the type being used
            ele_type_mask = np.logical_or(
                np.logical_and(
                    use_absorbers,
                    ele.absorbs
                ),
                np.logical_and(
                    np.logical_not(use_absorbers),
                    ele.emits
                )
            )

            replace_mask = np.logical_and(closer_mask, ele_type_mask)  # AND the masks together
            replace_mask_mat = np.tile(replace_mask, (seg_starts.shape[1], 1)).T

            out_sqr_distances = np.where(
                replace_mask,
                sqr_distances,
                out_sqr_distances
            )

            out_positions = np.where(
                replace_mask_mat,
                closest_points,
                out_positions
            )

        return out_sqr_distances, out_positions

    def trace_field_lines(self,
                          starts: np.ndarray,
                          max_points: int,
                          positives: np.ndarray,
                          step_distance: float = FIELD_LINE_TRACE_DEFAULT_STEP,
                          element_stop_distance: float = FIELD_LINE_TRACE_DEFAULT_ELEMENT_STOP_DISTANCE,
                          clip_ranges: Optional[np.ndarray] = None) -> np.ndarray:
        """Traces field lines starting at some position vectors and following the field for a specified distance or until reaching an absorber/emitter field element

Parameters:

    starts - the positions to start at

    max_points - the maximum number of points to make each field line. Will stop the line after this many points

    positives - whether to trace the lines in the "positive" direction (from positive to negative) instead of the negative direction

    step_distance - how far to step at each point of tracing the field lines

    element_stop_distance - if the lines gets this close to a complementary field element then it will stop at that point

    clip_ranges - a 2D array of shape (N,2) where N is the number of dimensions of the space of the field. \
Each pair describes the range of values outside which the field lines will be clipped

Returns:

    lines - a 3D array where each axis 0 is each field line, axis 1 is the positions of each point of each field line and axis 2 is the components of these positions. \
When a field line is ended early, the final value before clipping is propagated to the end of the array
"""

        assert starts.ndim == 2, "Invalid starting point array dimensionality"
        assert positives.ndim == 1, "Invalid positives array dimensionality"
        assert starts.shape[0] == positives.shape[0], "Starting point and positives arrays are not of matching shapes"

        if clip_ranges is not None:
            assert clip_ranges.ndim == 2, "Invalid clip_ranges dimensionality"
            assert np.all(clip_ranges[:, 0] <= clip_ranges[:, 1]), "clip_ranges lower bounds must not be greater than the upper bounds"
            assert clip_ranges.shape[0] == starts.shape[1], "clip_ranges doesn't have same number of vector components as start positions"

        line_count = starts.shape[0]  # Number of lines being traced
        dim = starts.shape[1]  # Dimensions of the space

        # Initialise output array with the maximum number of possible points needed for each line

        lines = np.zeros(shape=(line_count, max_points, dim))
        # To get the c'th component of the t'th point on the n'th line, we look at:
        #     lines[n, t, c]

        lines[:, 0] = starts

        active_mask = np.ones(shape=(line_count,), dtype=bool)  # Which lines are still being generated

        for t in range(0, max_points-1):

            # Handle active lines

            curr_poss = lines[:, t]  # R^(line_count)x(dim)

            # Calculate directions to move in

            grads = self.grad(curr_poss)  # R^(line_count)x(dim)

            active_mask = np.logical_and(active_mask, np.logical_not(np.all(np.isclose(grads, 0.0), axis=1)))  # Make inactive lines found to be at stationary points

            move_dir = grads / vectors.magnitudes(grads)[:, np.newaxis]  # R^(line_count)x(dim)

            move_dir[positives] *= -1  # Invert the direction of the positive lines' move directions

            # Find and store the next points or propagate old points

            new_poss = curr_poss + (move_dir * step_distance)  # R^(line_count)x(dim)

            active_mask_mat = np.tile(active_mask, (starts.shape[1], 1)).T

            lines[:, t+1] = np.where(
                active_mask_mat,
                new_poss,
                lines[:, t]
            )

            del active_mask_mat

            # Check nearest appropriate field elements of active lines

            nearest_sqr_distances, nearest_poss = self.line_seg_nearest_element(
                lines[:, t],  # The old positions
                lines[:, t+1],  # The new positions
                positives[:]  # Which lines are positive
            )  # TODO - don't calculate these values for all lines, only do them for the active lines. Probably replace the below logic to use np.where instead of arr[mask] syntax

            # Create a mask for lines that are newly-terminated

            point_close_mask = nearest_sqr_distances <= element_stop_distance
            newly_terminated_mask = np.logical_and(point_close_mask, active_mask)

            # Replace the ending positions of the newly-terminated lines and set them as inactive

            lines[newly_terminated_mask, t+1] = nearest_poss[newly_terminated_mask]
            active_mask = np.logical_and(active_mask, np.logical_not(newly_terminated_mask))

            # Clip any lines outside of the allowed range

            if clip_ranges is not None:

                within_lower_mask = np.all(lines[:, t+1] >= clip_ranges[:, 0], axis=1)
                within_upper_mask = np.all(lines[:, t+1] <= clip_ranges[:, 1], axis=1)

                within_clip_bounds_mask = np.logical_and(within_lower_mask, within_upper_mask)

                active_mask = np.logical_and(active_mask, within_clip_bounds_mask)

        # Return the output

        return lines
