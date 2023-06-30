from typing import Optional, Tuple
from element_base import ElementBase
import numpy as np
import vectors
from settings import EPS

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
