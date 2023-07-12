from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import vectors
from settings import EPS


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


class PointSource(ElementBase):
    """A 2D/3D point source of a field"""

    def __init__(self,
                 pos: np.ndarray,
                 strength: float
                 ):

        super().__init__(pos, strength > 0, strength < 0)

        self._strength = strength

    @property
    def strength(self) -> float:
        return self._strength

    def get_field_at(self, poss: np.ndarray) -> np.ndarray:

        assert poss.ndim in [2, 3], "Invalid input dimensionality"

        # Using an inverse square law: field_at_point = strength / distance^2

        dists = vectors.magnitudes(poss - self.pos)

        values = np.where(
            np.isclose(dists, 0),
            np.repeat(np.inf, dists.shape[0]),
            np.divide(self.strength, dists*dists)
        )

        return values

    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
        return np.tile(self.pos, (seg_starts.shape[0], 1))

    def __field_line_count_2d(self) -> int:
        return int(abs(np.ceil(self.strength)))

    def __field_line_count_3d(self) -> int:
        raise NotImplementedError()  # TODO

    def __get_field_line_starts_2d(self, fac: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        phi = np.linspace(0, 2*np.pi, self.__field_line_count_2d() * fac, endpoint=False)
        dx = np.cos(phi) * EPS
        dy = np.sin(phi) * EPS

        x = self.pos[0] + dx
        y = self.pos[1] + dy

        line_starts = np.dstack([x, y])[0]
        positives = np.repeat(self.strength > 0, line_starts.shape[0])

        return line_starts, positives

    def __get_field_line_starts_3d(self, fac: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError()  # TODO (make sure equally spaced, not concentrated around any poles)

    def get_field_line_starts(self, fac: int = 1, dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:

        if dim is None:
            dim = self.pos.shape[0]

        if dim == 2:
            return self.__get_field_line_starts_2d(fac)
        elif dim == 3:
            return self.__get_field_line_starts_3d(fac)
        else:
            raise ValueError("Unable to generate field line starts for more than 3-dimensional space for point sources")
