from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Mapping

import numpy as np
from PIL import Image as PilImage
import trimesh.transformations as tt

from src.camera_pose import CameraPose
from src.colors import rgb_to_packed_colors
from src.plotting import Plottable, VolumePlottable, PointsPlottable, \
    CameraPlottable, CameraFrustumPlottable
from src.sdf_reader import load_sdf


def swap_xz(a: np.ndarray) -> np.ndarray:
    assert len(a.shape) == 2 and a.shape[1] == 3
    return a[:, [2, 1, 0]]


class PathsLoadable(ABC):
    @classmethod
    @abstractmethod
    def from_paths(self, paths: 'VoxelDataPaths', *args, **kwargs): ...


@dataclass
class CameraView(Plottable, PathsLoadable):
    id: int
    rgb_filename: str
    depth_filename: str
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray

    plot_type = 'axes'  # or 'frustum'

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        camera_id = args[0]

        calib_filename = paths.get_calib_filename(camera_id)
        camera_params = np.loadtxt(calib_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]

        rgb_filename = paths.get_rgb_filename(camera_id)
        rgb_array = np.asarray(PilImage.open(rgb_filename))

        depth_filename = paths.get_depth_filename(camera_id)
        depth_array = np.asarray(PilImage.open(depth_filename), dtype=np.float_)

        return cls(camera_id, rgb_filename, depth_filename,
                   extrinsics, intrinsics, rgb_array, depth_array)

    def plot(self, k3d_plot):
        if self.plot_type == 'axes':
            plottable = CameraPlottable(
                CameraPose(self.extrinsics),
                line_length=.1,
                line_width=.01)
        elif self.plot_type == 'frustum':
            plottable = CameraFrustumPlottable(
                camera_pose=CameraPose(self.extrinsics),
                focal_length=[self.intrinsics[0, 0]],
                image_size=np.array(self.rgb.shape[:2]),
                principal_point=self.intrinsics[[0, 1], 2],
                sensor_size=np.array(self.rgb.shape[:2]),
                line_length=0.01, )
        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)


@dataclass
class ChunkVolume(Plottable, PathsLoadable):
    id: int
    chunk_filename: str
    sdf: np.ndarray
    transform: np.ndarray
    known: np.ndarray
    colors: np.ndarray

    plot_type: str = 'volume'  # or 'points' -- then u have color
    plot_sdf_thr: float = 0.01

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        chunk_id = args[0]

        chunk_filename = paths.get_chunk_filename(chunk_id)
        sdf, sdf_transform, known, colors = load_sdf(
            file=chunk_filename,
            load_sparse=False,
            load_known=False,
            load_colors=True,
            color_file=None)
        sdf[sdf == -np.inf] = np.inf

        return cls(chunk_id, chunk_filename, sdf, sdf_transform, known, colors)

    def plot(self, k3d_plot):
        if self.plot_type == 'volume':
            plottable = VolumePlottable(
                self.sdf < self.plot_sdf_thr,
                color_map='jet_r',
                interpolation=False,
                model_matrix=self.transform,)

        elif self.plot_type == 'points':
            colors = self.colors
            m, n, p = self.sdf.shape
            x_i, y_i, z_i = np.arange(m), np.arange(n), np.arange(p)
            xx, yy, zz = np.meshgrid(x_i, y_i, z_i, indexing='ij')
            indexes = np.stack((xx, yy, zz), axis=3)
            mask = self.sdf < self.plot_sdf_thr
            transform = np.linalg.inv(self.transform)
            voxel_size_in_mm = float(transform[0, 0])
            points = tt.transform_points(swap_xz(indexes[mask]), transform)
            point_colors = rgb_to_packed_colors(
                colors[mask, 0], colors[mask, 1], colors[mask, 2])
            plottable = PointsPlottable(
                points=points,
                point_colors=point_colors,
                point_size=voxel_size_in_mm,)

        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)

    @property
    def voxels_xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of voxel centers in world frame."""
        m, n, p = self.sdf.shape
        x_i, y_i, z_i = np.arange(m), np.arange(n), np.arange(p)
        xx, yy, zz = np.meshgrid(x_i, y_i, z_i, indexing='ij')
        indexes = np.stack((xx, yy, zz), axis=3)
        mask = self.sdf < self.plot_sdf_thr
        transform = np.linalg.inv(self.transform)
        points = tt.transform_points(swap_xz(indexes[mask]), transform)
        return points


@dataclass
class FullVolume(Plottable, PathsLoadable):
    sparse_indexes: np.ndarray
    sparse_sdf: np.ndarray
    shape: Tuple[int, int, int]
    sdf: np.ndarray
    transform: np.ndarray
    known: np.ndarray
    sparse_colors: np.ndarray

    plot_type: str = 'volume'  # or 'points' -- then u have color
    plot_sdf_thr: float = 0.01

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        sdf_filename, rgb_filename = paths.get_full_filenames()
        (indexes, sdf), shape, transform, known, rgb = load_sdf(
            file=sdf_filename,
            load_sparse=True,
            load_known=False,
            load_colors=True,
            color_file=rgb_filename)
        room = np.ones(shape, dtype=np.float32) * np.inf
        room[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = sdf
        return cls(indexes, sdf, shape, room, transform, known, rgb)

    @property
    def voxels_xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of voxel centers in world frame."""
        i, j, k = self.sparse_indexes[:, 0], self.sparse_indexes[:, 1], \
                  self.sparse_indexes[:, 2]
        mask = self.sparse_sdf < self.plot_sdf_thr
        transform = np.linalg.inv(self.transform)
        points = tt.transform_points(
            swap_xz(self.sparse_indexes[mask]), transform)
        return points

    def plot(self, k3d_plot):
        if self.plot_type == 'volume':
            plottable = VolumePlottable(
                self.sdf < self.plot_sdf_thr,
                color_map='jet_r',
                interpolation=False,
                model_matrix=self.transform,)

        elif self.plot_type == 'points':
            i, j, k = self.sparse_indexes[:, 0], self.sparse_indexes[:, 1], \
                      self.sparse_indexes[:, 2]
            mask = self.sparse_sdf < self.plot_sdf_thr
            transform = np.linalg.inv(self.transform)
            voxel_size_in_mm = float(transform[0, 0])
            points = tt.transform_points(
                swap_xz(self.sparse_indexes[mask]), transform)
            colors = self.sparse_colors[i, j, k]
            point_colors = rgb_to_packed_colors(
                colors[mask, 0], colors[mask, 1], colors[mask, 2])
            plottable = PointsPlottable(
                points=points,
                point_colors=point_colors,
                point_size=voxel_size_in_mm,)

        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)


CameraByChunk = Mapping[int, np.ndarray]


@dataclass
class VoxelChunkData(Plottable, PathsLoadable):
    camera_ids: CameraByChunk
    camera_views: Mapping[int, CameraView]
    chunk_volumes: List[ChunkVolume]
    full_volumes: FullVolume

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        pass

    def plot(self, k3d_plot):
        for view in self.camera_views.values():
            view.plot(k3d_plot)
        for volume in self.chunk_volumes:
            volume.plot(k3d_plot)
        self.full_volumes.plot(k3d_plot)


__all__ = [
    'ChunkVolume',
    'CameraView',
    'ChunkVolume',
    'FullVolume',
    'VoxelChunkData',
    'CameraByChunk',
]
