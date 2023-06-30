from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np

class ElementBase(ABC):

    def __init__(self, pos: np.ndarray, emitter: bool, absorber: bool):

        self._pos = pos
        self._emitter = emitter
        self._absorber = absorber

    @property
    def pos(self) -> np.ndarray:
        return self._pos

    @property
    def x(self) -> float:
        return self._pos[0]

    @property
    def y(self) -> float:
        return self._pos[1]

    @property
    def z(self) -> float:
        return self._pos[2]

    @property
    def emits(self) -> bool:
        return self._emitter

    @property
    def absorbs(self) -> bool:
        return self._absorber

    @abstractmethod
    def get_field_at(self, poss: np.ndarray) -> np.ndarray:
        """Gets the value of the element's field at the point or points given

Parameters:

    poss - a 2D array of position vectors for the positions to evaluate the field at

Returns:

    values - a 1D array containing the values of the field at the requested positions
"""
        pass

    @abstractmethod
    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
        """Find the nearest points of the element to the provided line segments"""
        pass

    @abstractmethod
    def get_field_line_starts(self, fac: int = 1, dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates and returns the positions at which field lines from this element should be started and which are positive-directed lines taking into account the element's strength

Parameters:

    fac (default 1) - how much to multiply the number of lines decided to create by

    dim (optional) - the number of components each position should have. Usually will be 2 for a 2D space or 3 for a 3D space

Returns:

    line_starts - the positions to start the lines in

    positives - which lines are positive lines

"""
