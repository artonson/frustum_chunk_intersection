import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, TYPE_CHECKING

import numpy as np
from skimage.transform import resize

if TYPE_CHECKING:
    from src.datasets.base import DataPaths

from src.geometry.camera_pose import CameraPose
from src.geometry.volume_view import VolumeView
from src.utils.colors import rgb_to_packed_colors
from src.utils.plotting import Plottable, VolumePlottable, PointsPlottable, \
    CameraPlottable, CameraFrustumPlottable


class PathsLoadable(ABC):
    @classmethod
    @abstractmethod
    def from_paths(cls, paths: 'DataPaths', *args, **kwargs): ...


@dataclass
class CameraView(Plottable, PathsLoadable):
    id: int
    extrinsics_filename: str
    intrinsics_filename: str
    rgb_filename: str
    depth_filename: str
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    rgb: np.ndarray
    depth: np.ndarray

    plot_type = 'axes'  # or 'frustum'
    line_length = 0.01
    opacity: float = 1.0

    @classmethod
    def from_paths(cls, paths: 'DataPaths', *args, **kwargs):
        camera_id = args[0]

        extrinsics_filename = paths.get_extrinsics_filename(camera_id)
        try:
            extrinsics = paths.get_extrinsics(extrinsics_filename)
        except FileNotFoundError:
            warnings.warn(f'Could not find extrinrics at {extrinsics_filename}')
            extrinsics = None

        intrinsics_filename = paths.get_intrinsics_filename(camera_id)
        try:
            intrinsics = paths.get_intrinsics(intrinsics_filename)
        except FileNotFoundError:
            warnings.warn(f'Could not find intrinsics at {intrinsics_filename}')
            intrinsics = None

        rgb_filename = paths.get_rgb_filename(camera_id)
        try:
            rgb_array = paths.get_rgb(rgb_filename)
        except FileNotFoundError:
            warnings.warn(f'Could not find RGB image at {rgb_filename}')
            rgb_array = None

        depth_filename = paths.get_depth_filename(camera_id)
        try:
            depth_array = paths.get_depth(depth_filename)
        except FileNotFoundError:
            warnings.warn(f'Could not find depth image at {depth_filename}')
            depth_array = None

        resize_rgb_to_depth = kwargs.get('resize_rgb_to_depth', False)
        if resize_rgb_to_depth and None is not rgb_array \
                and None is not depth_array:
            warnings.warn(f'Resizing RGB from {str(rgb_array.shape)} '
                          f'to {str(depth_array.shape)}')
            rgb_array = resize(
                rgb_array,
                depth_array.shape,
                anti_aliasing=True,
                preserve_range=True).astype(np.uint8)

        return cls(
            camera_id,
            extrinsics_filename, intrinsics_filename,
            rgb_filename, depth_filename,
            extrinsics, intrinsics,
            rgb_array, depth_array)

    def plot(self, k3d_plot):
        if self.plot_type == 'axes':
            plottable = CameraPlottable(
                CameraPose(self.extrinsics),
                line_length=self.line_length,
                line_width=.01,
                name=str(self.id))
        elif self.plot_type == 'frustum':
            plottable = CameraFrustumPlottable(
                camera_pose=CameraPose(self.extrinsics),
                intrinsics=self.intrinsics,
                image_size=np.array(self.depth.shape)[::-1],
                line_length=self.line_length,
                opacity=self.opacity,
                name=str(self.id))
            # plottable = CameraFrustumPlottable(
            #     camera_pose=CameraPose(self.extrinsics),
            #     focal_length=[self.intrinsics[0, 0]],
            #     image_size=np.array(self.rgb.shape[:2]),
            #     principal_point=self.intrinsics[[0, 1], 2],
            #     sensor_size=np.array(self.rgb.shape[:2]),
            #     line_length=self.line_length,
            #     opacity=self.opacity,
            #     name=str(self.id))
        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)


@dataclass
class ChunkVolume(Plottable, PathsLoadable):
    id: int
    chunk_filename: str
    volume: VolumeView

    plot_type: str = 'volume'  # or 'points' -- then u have color
    plot_sdf_thr: float = 0.5
    plot_colors: bool = True

    @classmethod
    def from_paths(cls, paths: 'DataPaths', *args, **kwargs):
        chunk_id = args[0]
        chunk_filename = paths.get_chunk_filename(chunk_id)
        try:
            volume = paths.get_chunk(chunk_filename)
        except FileNotFoundError:
            raise
        return cls(chunk_id, chunk_filename, volume)

    def plot(self, k3d_plot):
        if self.plot_type == 'volume':
            volume = self.volume.to_dense()
            plottable = VolumePlottable(
                volume.sdf_xyz < self.plot_sdf_thr,
                color_map='jet_r',
                interpolation=False,
                model_matrix=volume.world_to_grid,)

        elif self.plot_type == 'points':
            # volume = self.volume.to_dense()
            mask = (np.abs(self.volume.sdf_xyz) < self.plot_sdf_thr).ravel()
            print(mask.shape)
            points = self.volume.xyz_world[mask]
            print(points.shape)
            transform = self.volume.grid_to_world
            voxel_size_in_mm = float(transform[0, 0])
            args = dict(points=points, point_size=voxel_size_in_mm)
            if self.plot_colors:
                if None is not self.volume.colors:
                    c = self.volume.colors_xyz[mask]
                    r, g, b = c[:, 0], c[:, 1], c[:, 2]
                    point_colors = rgb_to_packed_colors(r, g, b)
                    args['point_colors'] = point_colors
                else:
                    print(self.volume.sdf_xyz.shape, )
                    args['attrs'] = np.abs(self.volume.sdf_xyz.ravel()[mask])
            plottable = PointsPlottable(**args)

        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)


@dataclass
class SceneVolume(ChunkVolume):
    # id: int
    # scene_filename: str
    # volume: VolumeView
    #
    # plot_type: str = 'volume'  # or 'points' -- then u have color
    # plot_sdf_thr: float = 0.5
    # plot_colors: bool = True

    @classmethod
    def from_paths(cls, paths: 'DataPaths', *args, **kwargs):
        scene_id = args[0]
        filenames = paths.get_scene_filename(scene_id)
        try:
            volume = paths.get_scene(filenames)
        except FileNotFoundError:
            raise
        return cls(scene_id, filenames, volume)

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
    # def plot(self, k3d_plot):
    #     if self.plot_type == 'volume':
    #         plottable = VolumePlottable(
    #             self.sdf < self.plot_sdf_thr,
    #             color_map='jet_r',
    #             interpolation=False,
    #             model_matrix=self.transform,)
    #
    #     elif self.plot_type == 'points':
    #         points, mask, transform = self._get_voxels_points_mask()
    #         voxel_size_in_mm = float(transform[0, 0])
    #         args = dict(points=points, point_size=voxel_size_in_mm)
    #         if self.plot_colors:
    #             if None is not self.sparse_colors:
    #                 i, j, k = self.sparse_indexes[:, 0], \
    #                     self.sparse_indexes[:, 1], \
    #                     self.sparse_indexes[:, 2]
    #                 colors = self.sparse_colors[i, j, k]
    #                 point_colors = rgb_to_packed_colors(
    #                     colors[mask, 0], colors[mask, 1], colors[mask, 2])
    #                 args['point_colors'] = point_colors
    #         else:
    #             args['attrs'] = np.abs(self.sparse_sdf[mask].ravel())
    #
    #         plottable = PointsPlottable(**args)
    #
    #     else:
    #         raise ValueError(f'{self.plot_type}')
    #
    #     return plottable.plot(k3d_plot)


CameraByChunk = Mapping[str, np.ndarray]


@dataclass
class VoxelChunkData(Plottable, PathsLoadable):
    camera_ids: CameraByChunk = None
    camera_views: Mapping[str, CameraView] = None
    chunk_volumes: List[ChunkVolume] = None
    scene_volume: SceneVolume = None

    @classmethod
    def from_paths(cls, paths: 'DataPaths', *args, **kwargs):
        pass

    def plot(self, k3d_plot):
        for view in self.camera_views.values():
            view.plot(k3d_plot)
        for volume in self.chunk_volumes:
            volume.plot(k3d_plot)
        self.scene_volume.plot(k3d_plot)


__all__ = [
    'ChunkVolume',
    'CameraView',
    'ChunkVolume',
    'SceneVolume',
    'VoxelChunkData',
    'CameraByChunk',
]
