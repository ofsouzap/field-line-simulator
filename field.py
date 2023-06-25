from typing import List, Optional
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

    def evaluate(self, pos: np.ndarray) -> float:

        val: float = 0

        for ele in self.__elements:

            val += ele.get_field_at(pos)

        return val

    def grad(self, pos: np.ndarray) -> np.ndarray:
        return vectors.grad(self.evaluate, pos)

    def line_seg_nearby_element(self,
                                seg_start: np.ndarray,
                                seg_end: np.ndarray,
                                range: float,
                                use_absorbers: bool) -> bool:
        """Find if the line segment specified is near to any absorbing or emitting field elements

Parameters:

    seg_start - the starting position vector of the line segment

    seg_end - the ennding position vector of the line segment

    range - how far away the absorbing element can be

    use_absorbers - if True then the function will look for a nearby emitter, if False then it will look for absorbers
"""

        for ele in self.__elements:

            if ele.absorbs != use_absorbers:
                continue

            dist = vectors.line_seg_distance_to_point(
                seg_start,
                seg_end,
                ele.pos
            )

            if dist <= range:
                return True

        return False

    def trace_field_line(self,
                         start: np.ndarray,
                         max_points: int,
                         positive: bool,
                         step_distance: float = FIELD_LINE_TRACE_DEFAULT_STEP,
                         element_stop_distance: float = FIELD_LINE_TRACE_DEFAULT_ELEMENT_STOP_DISTANCE) -> np.ndarray:
        """Traces a field line starting at a point and following the field for a specified distance or until reaching an absorber field element

Parameters:

    field - the field to evaluate in

    start - the position to start at

    max_points - the maximum number of points to make the field line. Will stop the line after this many points

    positive - whether to trace the line in the "positive" direction (from positive to negative) instead

    step_distance - how far to step at each point of tracing the field line

    element_stop_distance - if the line gets this close to a complementary field element then it will stop at that point
"""

        # Initialise output array with the maximum number of possible points needed

        points = np.zeros(shape=(max_points,start.shape[0]), dtype=float)
        points[0] = start
        end_index = 0

        for i in range(0, max_points-1):

            end_index = i
            curr = points[i]

            # Calculate direction to move in

            grad = self.grad(curr)

            move_dir = grad / np.linalg.norm(grad)
            if positive:
                move_dir *= -1

            # Store the next point

            new = curr + (move_dir * step_distance)
            points[i+1] = new

            # Check if passed close enough to an absorber to stop the line

            if self.line_seg_nearby_element(curr, new, element_stop_distance, use_absorbers=positive):
                break

        # Return the output (trim end if didn't get to maximum number of points)

        return points[:end_index]
