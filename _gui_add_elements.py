from abc import ABC, abstractmethod
from field_element import ElementBase, PointSource
import numpy as np


class AddElementBase(ABC):

    @abstractmethod
    def get_icon_filename(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def get_display_name(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def create_instance(self, pos: np.ndarray, strength: float) -> ElementBase:
        raise NotImplementedError()


class PointSourceAddElement(AddElementBase):

    def get_icon_filename(self) -> str:
        return "point_source.png"

    def get_display_name(self) -> str:
        return "Point Source"

    def create_instance(self, pos: np.ndarray, strength: float) -> ElementBase:
        return PointSource(pos, strength)
