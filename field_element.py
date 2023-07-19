from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import vectors
from settings import EPS


# Make sure that the line spawn offset isn't so small that the grad calculation goes over the field element
LINE_SPAWN_OFFSET = EPS * 3
"""How far away from a field element to start a line"""


class UnboundedException(Exception): pass


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
    def get_grad_at(self, poss: np.ndarray) -> np.ndarray:
        """Gets the value of the vector gradient of the element's field at the point or points given

Parameters:

    poss - a 2D array of position vectors for the positions to evaluate the field at

Returns:

    values - a 1D array containing the values of the field's vector gradient at the requested positions
"""
        pass

    @abstractmethod
    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
        """Find the nearest points of the element to the provided line segments"""
        pass

    @abstractmethod
    def _get_field_line_starts(self, bounds: np.ndarray, fac: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
        pass


    def get_field_line_starts(self, bounds: np.ndarray, fac: int = 1, dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates and returns the positions at which field lines from this element should be started and which are positive-directed lines taking into account the element's strength

Parameters:

    bounds - an array of pairs for the bounds of the region being rendered

    fac (default 1) - how much to multiply the number of lines decided to create by

    dim (optional) - the number of components each position should have. Usually will be 2 for a 2D space or 3 for a 3D space

Returns:

    line_starts - the positions to start the lines in

    positives - which lines are positive lines

"""

        if dim is None:
            dim = self.pos.shape[0]

        assert (bounds.shape[0] == dim)

        return self._get_field_line_starts(bounds, fac, dim)


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

        dists = vectors.magnitudes(poss - self.pos)

        values = np.where(
            np.isclose(dists, 0),
            np.repeat(np.inf, dists.shape[0]),
            np.divide(self.strength, dists)
        )

        return values

    def get_grad_at(self, poss: np.ndarray) -> np.ndarray:

        displacements = poss - self.pos

        sqr_dists = vectors.sqr_magnitudes(displacements)

        sqr_dists_mat = np.tile(
            sqr_dists,
            (poss.shape[1],1),
        ).T

        unit_grads = -2 * displacements / sqr_dists_mat

        grads = self.strength * unit_grads

        return grads

    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:
        return np.tile(self.pos, (seg_starts.shape[0], 1))

    def __field_line_count_2d(self) -> int:
        return int(abs(np.ceil(np.sqrt(np.abs(self.strength)))))

    def __field_line_count_3d(self) -> int:
        raise NotImplementedError()  # TODO

    def __get_field_line_starts_2d(self, fac: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        phi = np.linspace(0, 2*np.pi, self.__field_line_count_2d() * fac, endpoint=False)
        dx = np.cos(phi) * LINE_SPAWN_OFFSET
        dy = np.sin(phi) * LINE_SPAWN_OFFSET

        x = self.pos[0] + dx
        y = self.pos[1] + dy

        line_starts = np.dstack([x, y])[0]
        positives = np.repeat(self.strength > 0, line_starts.shape[0])

        return line_starts, positives

    def __get_field_line_starts_3d(self, fac: int = 1) -> Tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError()  # TODO (make sure equally spaced, not concentrated around any poles)

    def _get_field_line_starts(self, bounds: np.ndarray, fac: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:

        if dim == 2:
            return self.__get_field_line_starts_2d(fac)
        elif dim == 3:
            return self.__get_field_line_starts_3d(fac)
        else:
            raise ValueError("Unable to generate field line starts for more than 3-dimensional space for point sources")


class ChargePlane(ElementBase):
    """An infinite plane of constant charge"""

    STRENGTH_DENSITY_FACTOR: float = 0.01

    def __init__(self,
                 pos: np.ndarray,
                 normal: np.ndarray,
                 strength_density: float):

        super().__init__(pos, strength_density > 0, strength_density < 0)

        self._normal = normal / vectors.magnitudes(normal)[0]
        self._strength_density = strength_density * ChargePlane.STRENGTH_DENSITY_FACTOR

    @property
    def normal(self) -> np.ndarray:
        return self._normal

    @property
    def d_value(self) -> float:
        return vectors.single_dot(
            self.pos,
            self.normal
        )

    @property
    def strength_density(self) -> float:
        return self._strength_density

    def get_field_at(self, poss: np.ndarray) -> np.ndarray:
        raise UnboundedException("Field from infinite plane is infinite")

    def get_grad_at(self, poss: np.ndarray) -> np.ndarray:

        grad_mag = self.strength_density / 2

        pos_displacements = poss - self.pos  # (N,dim)
        pos_norm_dists = vectors.many_dot(pos_displacements, np.tile(self.normal, (poss.shape[0],1)))  # (N,)
        pos_norm_dist_mags = np.abs(pos_norm_dists)  # (N,)
        signs = -pos_norm_dists / pos_norm_dist_mags  # (N,)

        grad_dirs = signs[np.newaxis, :].T * self.normal
        grads = grad_dirs * grad_mag

        return np.where(
            vectors.mat_mask(np.isclose(pos_norm_dists, 0), poss.shape[1]),
            np.zeros_like(poss),
            grads
        )

    def find_line_seg_nearest_point(self, seg_starts: np.ndarray, seg_ends: np.ndarray) -> np.ndarray:

        N = seg_starts.shape[0]

        return vectors.plane_closest_point_to_line_seg(
            np.tile(self.pos, (N, 1)),
            np.tile(self.normal, (N, 1)),
            seg_starts,
            seg_ends,
        )

    def __get_field_line_spacing(self) -> float:
        """Gets the distance between lines to draw"""
        return round(50+950*(1-np.tanh(abs(self.strength_density))))

    def __get_field_line_starts_2d(self, bounds: np.ndarray, fac: int) -> Tuple[np.ndarray, np.ndarray]:

        corners = np.array([
            [bounds[0,0], bounds[1,0]],  # minumum x, minumum y
            [bounds[0,1], bounds[1,0]],  # maximum x, minimum y
            [bounds[0,0], bounds[1,1]],  # minumum x, maximum y
            [bounds[0,1], bounds[1,1]],  # maximum x, maximum y
        ])
        """Array of positions of the corners of the bounds"""

        corner_plane_points = vectors.plane_closest_point_to_point(
            np.tile(self.pos, (corners.shape[0], 1)),
            np.tile(self.normal, (corners.shape[0], 1)),
            corners
        )

        point_mat1 = np.tile(corner_plane_points[np.newaxis, :], (corner_plane_points.shape[0], 1, 1))
        point_mat2 = np.transpose(point_mat1, (1, 0, 2))

        flat_sqr_distances = vectors.sqr_magnitudes(
            point_mat1.reshape((point_mat1.shape[0]*point_mat1.shape[1],point_mat1.shape[2])) - \
            point_mat2.reshape((point_mat2.shape[0]*point_mat2.shape[1],point_mat2.shape[2]))
        )

        sqr_dist_mat = flat_sqr_distances.reshape((point_mat1.shape[0], point_mat1.shape[1]))

        index_mat1 = np.tile(np.arange(point_mat1.shape[0])[np.newaxis, :], (point_mat1.shape[0], 1))
        index_mat2 = np.transpose(index_mat1, (1, 0))

        max_dist = np.max(sqr_dist_mat)
        max_dist_mask = np.isclose(sqr_dist_mat, max_dist)

        start_point_index = index_mat1[max_dist_mask][0]
        end_point_index = index_mat2[max_dist_mask][0]

        start_point = corner_plane_points[start_point_index]
        end_point = corner_plane_points[end_point_index]

        point_dist: float = vectors.magnitudes(end_point - start_point)[0]

        line_spacing: float = self.__get_field_line_spacing()
        line_count: int = round(fac * point_dist / line_spacing)

        xs = np.linspace(start_point[0], end_point[0], line_count)
        ys = np.linspace(start_point[1], end_point[1], line_count)

        line_start_roots = np.dstack([xs, ys])[0]

        norm_eps = self.normal * LINE_SPAWN_OFFSET
        root_offsets = np.tile(
            np.array([norm_eps, -norm_eps]),
            (line_start_roots.shape[0],1)
        )

        line_starts = np.repeat(line_start_roots, 2, axis=0) + root_offsets

        positives = np.repeat(self.strength_density > 0, line_starts.shape[0])

        return line_starts, positives

    def __get_field_line_starts_3d(self, bounds: np.ndarray, fac: int) -> Tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError()  # TODO

    def _get_field_line_starts(self, bounds: np.ndarray, fac: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:

        if dim == 2:
            return self.__get_field_line_starts_2d(bounds, fac)
        elif dim == 3:
            return self.__get_field_line_starts_3d(bounds, fac)
        else:
            raise ValueError("Unable to generate field line starts for more than 3-dimensional space for charge planes")
