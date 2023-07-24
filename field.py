from typing import List, Tuple, Optional, Iterator, TextIO
from field_element import ElementBase, PointSource, ChargePlane
import numpy as np
import vectors
import settings
import re


class ElementNotInFieldException(Exception): pass


class Field:

    def __init__(self):

        self.__elements: List[ElementBase] = []

    def write_to_file(self, stream: TextIO) -> None:

        elements: List[ElementBase] = []

        while stream.readable():

            line = stream.readline()

    @staticmethod
    def read_from_stream(stream: TextIO) -> "Field":

        raise NotImplementedError()  # TODO

    def add_element(self, ele: ElementBase) -> None:
        self.__elements.append(ele)

    def remove_element(self, ele: ElementBase) -> None:

        if ele in self.__elements:
            self.__elements.remove(ele)
        else:
            raise ElementNotInFieldException()

    def iter_elements(self) -> Iterator[ElementBase]:
        for ele in self.__elements:
            yield ele

    def evaluate(self, poss: np.ndarray) -> np.ndarray:

        vals = np.zeros(shape=(poss.shape[0]))

        for ele in self.iter_elements():

            vals += ele.get_field_at(poss)

        return vals

    def grad(self, poss: np.ndarray) -> np.ndarray:
        """Takes an array of position vectors and returns the grad of the field at those positions"""

        grads = np.zeros_like(poss)

        for ele in self.iter_elements():
            grads += ele.get_grad_at(poss)

        return grads

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
            replace_mask_mat = vectors.mat_mask(replace_mask, seg_starts.shape[1])

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

    def __line_trace_next_positions(self,
                                    poss: np.ndarray,
                                    positives: np.ndarray,
                                    step_distance: float) -> np.ndarray:
        """Computes the next point to extend a field line being traced

Parameters:

    poss - a 2D array of the position vectors of the points that the line is currently at

    positives - a 1D array of booleans describing which lines are positive lines

    step_distance - how far the lines should step

Returns:

    nexts - a 2D array of the position vectors of the next points that the lines should go to
"""

        grads = self.grad(poss)  # R^(line_count)x(dim)

        move_dir = grads / vectors.magnitudes(grads)[:, np.newaxis]  # R^(line_count)x(dim)

        move_dir[positives] *= -1  # Invert the direction of the positive lines' move directions

        nexts = poss + (move_dir * step_distance)  # R^(line_count)x(dim)

        return nexts

    def __field_line_trace_single_iteration(self,
                                            t: int,
                                            lines: np.ndarray,
                                            active_mask: np.ndarray,
                                            positives: np.ndarray,
                                            step_distance: float,
                                            element_stop_distance: float,
                                            clip_ranges: np.ndarray) -> None:

        dim = lines.shape[2]

        # Clip any lines outside of the allowed range and deactivate them

        clip_mask = vectors.outside_bounds(lines[active_mask, t, :], clip_ranges)

        # Deactivate lines that went too close to a field element

        nearest_sqr_distances, nearest_poss = self.line_seg_nearest_element(
            lines[active_mask, max(t-1, 0)],  # The old positions
            lines[active_mask, t],  # The new positions
            positives[active_mask]  # Which lines are positive
        )

        point_close_mask = nearest_sqr_distances <= element_stop_distance  # Which of the active lines have been deactivated

        # Calculate next positions for active lines

        active_curr_poss = lines[active_mask, t]  # R^(line_count)x(dim)
        active_positives = positives[active_mask]  # {0,1}^(line_count)

        active_next_poss = self.__line_trace_next_positions(active_curr_poss, active_positives, step_distance=step_distance)

        # Apply effects of computations to active lines, inactive lines and the active mask

        lines[active_mask, t+1, :] = np.where(
            vectors.mat_mask(clip_mask, dim),
            lines[active_mask, t, :],  # When line gets clipped this iteration
            np.where(
                vectors.mat_mask(point_close_mask, dim),
                nearest_poss,  # When line reaches a field element this iteration
                active_next_poss  # Normal behaviour, just continuing the line
            )
        )

        lines[~active_mask, t+1, :] = lines[~active_mask, t, :]

        active_mask[active_mask] = (~clip_mask) & (~point_close_mask)

    def trace_field_lines(self,
                          starts: np.ndarray,
                          max_points: int,
                          positives: np.ndarray,
                          step_distance: Optional[float] = None,
                          element_stop_distance: Optional[float] = None,
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

        if step_distance is None:
            step_distance = settings.field_line_trace_step_distance_screen_space * settings.VIEWPORT_SCALE_FAC

        if element_stop_distance is None:
            element_stop_distance = settings.field_line_trace_element_stop_distance_screen_space * settings.VIEWPORT_SCALE_FAC

        if clip_ranges is not None:
            assert clip_ranges.ndim == 2, "Invalid clip_ranges dimensionality"
            assert np.all(clip_ranges[:, 0] <= clip_ranges[:, 1]), "clip_ranges lower bounds must not be greater than the upper bounds"
            assert clip_ranges.shape[0] == starts.shape[1], "clip_ranges doesn't have same number of vector components as start positions"
        else:
            clip_ranges = np.tile(
                np.array([-np.inf, np.inf]),
                (starts.shape[1], 1)
            )

        line_count = starts.shape[0]  # Number of lines being traced
        dim = starts.shape[1]  # Dimensions of the space

        # Initialise output array with the maximum number of possible points needed for each line

        lines = np.zeros(shape=(line_count, max_points, dim))
        # To get the c'th component of the t'th point on the n'th line, we look at:
        #     lines[n, t, c]

        lines[:, 0] = starts

        active_mask = np.ones(shape=(line_count,), dtype=bool)  # Which lines are still being generated

        for t in range(0, max_points-1):

            # Stop (after writing final points) if no active lines

            if ~np.any(active_mask):
                lines[:, t+1] = lines[:, t]
                break

            # Calculate next points on lines and find lines to become inactive

            self.__field_line_trace_single_iteration(
                t,
                lines,
                active_mask,
                positives,
                step_distance,
                element_stop_distance,
                clip_ranges
            )

        # Return the output

        return lines


class FieldSerialize:

    POINT_SOURCE_REGEX = re.compile(
        r"pointsource (?P<posx>-?\d+.?\d*) (?P<posy>-?\d+.?\d*) (?P<strength>-?\d+.?\d*)",
        re.IGNORECASE
    )

    CHARGE_PLANE_REGEX = re.compile(
        r"chargeplane (?P<posx>-?\d+.?\d*) (?P<posy>-?\d+.?\d*) (?P<strengthdensity>-?\d+.?\d*) (?P<normx>-?\d+.?\d*) (?P<normy>-?\d+.?\d*)",
        re.IGNORECASE
    )

    @staticmethod
    def serialize(field: Field, stream: TextIO) -> None:
        for ele in field.iter_elements():
            FieldSerialize.write_element(stream, ele)

    @staticmethod
    def deserialize(stream: TextIO) -> Field:

        eles: List[ElementBase] = []

        # Read elements from stream

        while True:#stream.readable():

            line = stream.readline()

            # Final line is blank without newline
            if len(line) == 0:
                break

            # Strip whitespace
            line = line.strip()

            # Blank lines
            if len(line) == 0:
                continue

            # Comment lines
            if line[0] == "#":
                continue

            # Regular lines
            try:
                ele = FieldSerialize.parse_element(line)
                eles.append(ele)
            except ValueError:
                print(f"Encountered invalid line when reading file:\n{line}")

        # Create field from elements

        field = Field()

        for ele in eles:
            field.add_element(ele)

        # Return output

        return field

    @staticmethod
    def write_element(stream: TextIO, ele: ElementBase) -> None:
        match ele:
            case PointSource():
                FieldSerialize.write_point_source(stream, ele)
            case ChargePlane():
                FieldSerialize.write_charge_plane(stream, ele)
            case _:
                raise ValueError("Unhandled element subclass")

    @staticmethod
    def write_point_source(stream: TextIO, ps: PointSource) -> None:
        stream.write(f"pointsource {ps.x:.3f} {ps.y:.3f} {ps.strength:.3f}\n")

    @staticmethod
    def write_charge_plane(stream: TextIO, cp: ChargePlane) -> None:
        stream.write(f"chargeplane {cp.x:.3f} {cp.y:.3f} {cp.strength_density:.3f} {cp.normal[0]:.3f} {cp.normal[1]:.3f}")

    @staticmethod
    def parse_element(s: str) -> ElementBase:

        ps = FieldSerialize.try_parse_point_source(s)
        if ps is not None:
            return ps

        cp = FieldSerialize.try_parse_charge_plane(s)
        if cp is not None:
            return cp

        raise ValueError("Invalid input")

    @staticmethod
    def try_parse_point_source(s: str) -> Optional[PointSource]:

        m = FieldSerialize.POINT_SOURCE_REGEX.fullmatch(s)

        if m is None:

            return None

        else:

            posx = float(m.group("posx"))
            posy = float(m.group("posy"))
            strength = float(m.group("strength"))

            pos = np.array([posx, posy], dtype=float)
            ps = PointSource(pos, strength)

            return ps

    @staticmethod
    def try_parse_charge_plane(s: str) -> Optional[ChargePlane]:

        m = FieldSerialize.CHARGE_PLANE_REGEX.fullmatch(s)

        if m is None:

            return None

        else:

            posx = float(m.group("posx"))
            posy = float(m.group("posy"))
            strengthdensity = float(m.group("strengthdensity"))
            normx = float(m.group("normx"))
            normy = float(m.group("normy"))

            pos = np.array([posx, posy], dtype=float)
            norm = np.array([normx, normy], dtype=float)
            cp = ChargePlane(pos, norm, strengthdensity)

            return cp
