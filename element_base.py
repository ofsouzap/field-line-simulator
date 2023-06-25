from abc import ABC, abstractmethod
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
    def get_field_at(self, pos: np.ndarray) -> float:
        pass
