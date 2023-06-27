from typing import Tuple
from element_base import ElementBase
import numpy as np
import vectors

class PointSource(ElementBase):
    """A 2D point source of a field"""

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

        assert poss.ndim == 2, "Invalid input dimensionality"

        # Using an inverse square law: field_at_point = strength / distance^2

        dists = vectors.magnitudes(poss - self.pos)

        values = np.divide(self.strength, dists*dists)

        return values

    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
        return np.tile(self.pos, (seg_starts.shape[0], 1))
