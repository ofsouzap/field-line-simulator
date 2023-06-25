from element_base import ElementBase
import numpy as np

class PointSource(ElementBase):

    def __init__(self,
                 pos: np.ndarray,
                 strength: float
                 ):

        super().__init__(pos, strength > 0, strength < 0)

        self._strength = strength

    @property
    def strength(self) -> float:
        return self._strength

    def get_field_at(self, pos: np.ndarray):

        dist = np.linalg.norm(pos - self.pos)

        return np.divide(self._strength, dist*dist)  # field_at_point = strength / distance^2
