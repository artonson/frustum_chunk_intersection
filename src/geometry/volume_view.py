from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import trimesh.transformations as tt


def zyx_to_xyz(a: np.ndarray) -> np.ndarray:
    assert len(a.shape) == 2 and a.shape[1] == 3
    return a[:, [2, 1, 0]]


@dataclass
class VolumeView(ABC):
    """
    Representation of a volumetric implicit field
    with vector-valued properties. The implicit field is
    expected to come in ZYX order and is always stored in
    this same ZYX order. Getting things in ZYX or XYZ
    order is available using <prop>_zyx and <prop>_xyz
    getters.
    """
    sdf: np.ndarray
    transform: np.ndarray  # in XYZ order
    known: np.ndarray
    colors: np.ndarray
    zyx_grid_: np.ndarray = None
    shape: Tuple = None
    invalid_value: float = np.inf
    sdf_thr: float = np.inf  # set to a finite value for e.g. tsdf

    _zyx_grid: np.ndarray = None
    _zyx_world: np.ndarray = None

    @abstractmethod
    def to_dense(self) -> 'VolumeView': pass

    @abstractmethod
    def to_sparse(self) -> 'VolumeView': pass

    @property
    def grid_to_world(self) -> np.ndarray:
        return np.linalg.inv(self.transform)

    @property
    def world_to_grid(self) -> np.ndarray:
        return self.transform

    @property
    @abstractmethod
    def zyx_grid(self) -> np.ndarray: pass

    @property
    def zyx_world(self):
        if None is not self._zyx_world:
            xyz_world = tt.transform_points(self.xyz_grid, self.grid_to_world)
            self._zyx_world = zyx_to_xyz(xyz_world)
        return self._zyx_world

    @property
    def xyz_grid(self) -> np.ndarray:
        return zyx_to_xyz(self.zyx_grid)

    @property
    def xyz_world(self) -> np.ndarray:
        return zyx_to_xyz(self.zyx_world)

    @property
    def valid_mask(self):
        return self.sdf != self.invalid_value

    @property
    def shape_xyz(self):
        z, y, x = self.shape_zyx
        return x, y, z

    @property
    def shape_zyx(self):
        if None is not self.shape:
            self.shape = self.sdf.shape  # beware: zyx ordering
        return self.shape

    @property
    @abstractmethod
    def sdf_zyx(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def sdf_xyz(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def colors_zyx(self) -> np.ndarray: pass

    @property
    @abstractmethod
    def colors_xyz(self) -> np.ndarray: pass


@dataclass
class DenseVolumeView(VolumeView):
    def to_dense(self) -> VolumeView:
        return self

    def to_sparse(self) -> VolumeView:
        # transform all valid voxels into an array,
        # creating lots of copies
        mask = self.sdf_zyx != self.invalid_value
        return SparseVolumeView(
            self.sdf_zyx[mask].ravel().copy(),
            self.transform.copy(),
            self.known.copy(),
            self.colors_zyx[mask].reshape((-1, 3)).copy()
                if None is not self.colors else None,
            zyx_grid_=self.zyx_grid[mask.ravel()].copy(),
            shape=self.shape_zyx)

    @property
    def zyx_grid(self):
        if None is not self._zyx_grid:
            m, n, p = self.sdf.shape
            x_i, y_i, z_i = np.arange(m), np.arange(n), np.arange(p)
            xx, yy, zz = np.meshgrid(x_i, y_i, z_i, indexing='ij')
            self._zyx_grid = np.stack((xx, yy, zz), axis=3)
        return self._zyx_grid

    @property
    def sdf_zyx(self) -> np.ndarray:
        return self.sdf

    @property
    def sdf_xyz(self) -> np.ndarray:
        return np.transpose(self.sdf, (2, 1, 0))

    @property
    def colors_zyx(self) -> np.ndarray:
        return self.colors

    @property
    def colors_xyz(self) -> np.ndarray:
        return np.transpose(self.colors, (2, 1, 0))

    # def _get_voxels_points_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     m, n, p = self.sdf.shape
    #     x_i, y_i, z_i = np.arange(m), np.arange(n), np.arange(p)
    #     xx, yy, zz = np.meshgrid(x_i, y_i, z_i, indexing='ij')
    #     indexes = np.stack((xx, yy, zz), axis=3)
    #     mask = np.abs(self.sdf) < self.plot_sdf_thr
    #     transform = np.linalg.inv(self.transform)
    #     points = tt.transform_points(swap_xz(indexes[mask]), transform)
    #     return points, mask, transform
    #
    # @property
    # def voxels_xyz(self) -> np.ndarray:
    #     """Returns XYZ coordinates of voxel centers in world frame."""
    #     points, mask, transform = self._get_voxels_points_mask()
    #     return points


@dataclass
class SparseVolumeView(VolumeView):
    def to_dense(self) -> VolumeView:
        # transform all valid voxels into an array,
        # creating lots of copies
        z, y, x = self.zyx_grid_[:, 0], self.zyx_grid_[:, 1], self.zyx_grid_[:, 2]

        sdf_zyx = np.ones(self.shape_zyx, dtype=np.float32) * np.inf
        sdf_zyx[z, y, x] = self.sdf_zyx

        colors_zyx_shape = (self.shape_zyx[0], self.shape_zyx[1], self.shape_zyx[2], 3)
        colors_zyx = np.ones(colors_zyx_shape, dtype=np.float32) * np.inf
        colors_zyx[z, y, x] = self.colors_zyx
        return DenseVolumeView(
            sdf=sdf_zyx,
            transform=self.transform.copy(),
            known=self.known.copy(),
            colors=self.colors_zyx)

    def to_sparse(self) -> VolumeView:
        return self

    @property
    def zyx_grid(self) -> np.ndarray:
        if None is not self._zyx_grid:
            self._zyx_grid = self.zyx_grid_
        return self._zyx_grid

    # def _get_voxels_points_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     mask = np.abs(self.sparse_sdf) < self.plot_sdf_thr
    #     transform = np.linalg.inv(self.transform)
    #     points = tt.transform_points(
    #         swap_xz(self.sparse_indexes[mask]), transform)
    #     return points, mask, transform
    #
    # @property
    # def voxels_xyz(self) -> np.ndarray:
    #     """Returns XYZ coordinates of voxel centers in world frame."""
    #     points, mask, transform = self._get_voxels_points_mask()
    #     return points
    #
    # @property
    # def colors(self) -> np.ndarray:
    #     """Returns XYZ coordinates of voxel centers in world frame."""
    #     points, mask, transform = self._get_voxels_points_mask()
    #     i, j, k = self.sparse_indexes[:, 0], \
    #               self.sparse_indexes[:, 1], \
    #               self.sparse_indexes[:, 2]
    #     colors = self.sparse_colors[i, j, k]
    #     return colors[mask]
    #     # point_colors = rgb_to_packed_colors(
    #     #     colors[mask, 0], colors[mask, 1], colors[mask, 2])
    #     # return point_colors
    #
