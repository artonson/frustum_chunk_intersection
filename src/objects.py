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
    line_length = 0.01
    opacity: float = 1.0

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        camera_id = args[0]

        calib_filename = paths.get_calib_filename(camera_id)
        camera_params = np.loadtxt(calib_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]

        rgb_filename = paths.get_rgb_filename(camera_id)
        try:
            rgb_array = np.asarray(PilImage.open(rgb_filename))
        except FileNotFoundError:
            rgb_array = None

        depth_filename = paths.get_depth_filename(camera_id)
        try:
            # depth_array = np.asarray(PilImage.open(depth_filename), dtype=np.uint16)
            import imageio
            depth_array = imageio.imread(depth_filename)
        except FileNotFoundError:
            depth_array = None

        return cls(camera_id, rgb_filename, depth_filename,
                   extrinsics, intrinsics, rgb_array, depth_array)

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


def project_rgbd(camera_view, points, colors):
    """Given a scene with 3D points and RGB colors, and
    a camera view with depth, rgb, intrinsics, and extrinsics,
    compute an output camera view with depth, rgb, intrinsics, and extrinsics,
    by projecting the 3D points to camera frame."""
    unprojected_pc = CameraPose(camera_view.extrinsics).world_to_camera(points)
    depth = unprojected_pc[:, 2].copy()
    unprojected_pc /= np.atleast_2d(depth).T
    uv = tt.transform_points(unprojected_pc, camera_view.intrinsics)
    height, width = camera_view.depth.shape
    mask = (uv[:, 0] > 0) & (uv[:, 0] < width) \
           & (uv[:, 1] > 0) & (uv[:, 1] < height) \
           & (depth > 1e-2)
    uv_int = np.floor(uv[mask]).astype(np.int_)
    out_depth = np.zeros_like(camera_view.depth, dtype=np.float32)
    out_depth[uv_int[:, 1], uv_int[:, 0]] = depth[mask]
    out_color = np.zeros_like(camera_view.rgb, dtype=np.int_)
    out_color[uv_int[:, 1], uv_int[:, 0]] = colors[mask]

    from copy import deepcopy
    output_view = deepcopy(camera_view)
    output_view.rgb = out_color
    output_view.depth = out_depth
    return output_view


def unproject_rgbd(camera_view):
    """Given a camera view with depth, rgb, intrinsics, and extrinsics,
    compute a camera view with point cloud defined in global frame,
    rgb, intrinsics, and extrinsics."""
    pixels = camera_view.depth
    height, width = pixels.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    image_integers = np.stack((
        i.ravel(),
        j.ravel(),
        np.ones_like(i).ravel()
    )).T  # [n, 3]
    image_integers = image_integers.astype(np.float32)
    depth_integers = pixels.ravel()
    image_integers = image_integers[depth_integers != 0]
    colors = camera_view.rgb.reshape((-1, 3))[depth_integers != 0]
    depth = depth_integers[depth_integers != 0].astype(np.float32) / 1000
    unprojected_depth = tt.transform_points(
        image_integers,
        np.linalg.inv(camera_view.intrinsics))
    unprojected_depth *= np.atleast_2d(depth).T
    unprojected_pc = CameraPose(camera_view.extrinsics).camera_to_world(unprojected_depth)

    from copy import deepcopy
    output_view = deepcopy(camera_view)
    output_view.rgb = colors
    output_view.depth = unprojected_pc
    return output_view


@dataclass
class ChunkVolume(Plottable, PathsLoadable):
    id: int
    chunk_filename: str
    sdf: np.ndarray
    transform: np.ndarray
    known: np.ndarray
    colors: np.ndarray
    version: str = '2'  # 1 for older 64 chunks, 2 for newer 128 chunks

    plot_type: str = 'volume'  # or 'points' -- then u have color
    plot_sdf_thr: float = 0.5
    plot_colors: bool = True

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        chunk_id = args[0]

        chunk_filename = paths.get_chunk_filename(chunk_id)
        if cls.version == '1':
            load_colors = True
        elif cls.version == '2':
            load_colors = False
        else:
            raise ValueError(
                f'Version {cls.version} of {cls.__name__} is unknown')

        sdf, chunk_transform, known, colors = load_sdf(
            file=chunk_filename,
            load_sparse=False,
            load_known=False,
            load_colors=load_colors,
            color_file=None)
        sdf[sdf == -np.inf] = np.inf

        if cls.version == '2':
            assert None is not paths.full_volume, \
                f'you must first load full volume in version {cls.version}'
            scene_transform = paths.full_volume.transform
            scene_translation = scene_transform[:3, 3]
            chunk_translation = chunk_transform[:3, 3]
            relative_translation = chunk_translation - scene_translation
            corrected_chunk_origin = scene_translation - relative_translation
            chunk_transform[:3, 3] = corrected_chunk_origin

        return cls(chunk_id, chunk_filename, sdf,  chunk_transform, known, colors)

    def plot(self, k3d_plot):
        if self.plot_type == 'volume':
            plottable = VolumePlottable(
                self.sdf < self.plot_sdf_thr,
                color_map='jet_r',
                interpolation=False,
                model_matrix=self.transform,)

        elif self.plot_type == 'points':
            points, mask, transform = self._get_voxels_points_mask()
            voxel_size_in_mm = float(transform[0, 0])
            args = dict(points=points, point_size=voxel_size_in_mm)
            if self.plot_colors:
                if None is not self.colors:
                    point_colors = rgb_to_packed_colors(
                        self.colors[mask, 0],
                        self.colors[mask, 1],
                        self.colors[mask, 2])
                    args['point_colors'] = point_colors
                else:
                    args['attrs'] = np.abs(self.sdf[mask].ravel())
            plottable = PointsPlottable(**args)

        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)

    def _get_voxels_points_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m, n, p = self.sdf.shape
        x_i, y_i, z_i = np.arange(m), np.arange(n), np.arange(p)
        xx, yy, zz = np.meshgrid(x_i, y_i, z_i, indexing='ij')
        indexes = np.stack((xx, yy, zz), axis=3)
        mask = np.abs(self.sdf) < self.plot_sdf_thr
        transform = np.linalg.inv(self.transform)
        points = tt.transform_points(swap_xz(indexes[mask]), transform)
        return points, mask, transform

    @property
    def voxels_xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of voxel centers in world frame."""
        points, mask, transform = self._get_voxels_points_mask()
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
    plot_sdf_thr: float = 0.5
    plot_colors: bool = True

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

    def _get_voxels_points_mask(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.abs(self.sparse_sdf) < self.plot_sdf_thr
        transform = np.linalg.inv(self.transform)
        points = tt.transform_points(
            swap_xz(self.sparse_indexes[mask]), transform)
        return points, mask, transform

    @property
    def voxels_xyz(self) -> np.ndarray:
        """Returns XYZ coordinates of voxel centers in world frame."""
        points, mask, transform = self._get_voxels_points_mask()
        return points

    @property
    def colors(self) -> np.ndarray:
        """Returns XYZ coordinates of voxel centers in world frame."""
        points, mask, transform = self._get_voxels_points_mask()
        i, j, k = self.sparse_indexes[:, 0], \
                  self.sparse_indexes[:, 1], \
                  self.sparse_indexes[:, 2]
        colors = self.sparse_colors[i, j, k]
        return colors[mask]
        # point_colors = rgb_to_packed_colors(
        #     colors[mask, 0], colors[mask, 1], colors[mask, 2])
        # return point_colors

    def plot(self, k3d_plot):
        if self.plot_type == 'volume':
            plottable = VolumePlottable(
                self.sdf < self.plot_sdf_thr,
                color_map='jet_r',
                interpolation=False,
                model_matrix=self.transform,)

        elif self.plot_type == 'points':
            points, mask, transform = self._get_voxels_points_mask()
            voxel_size_in_mm = float(transform[0, 0])
            args = dict(points=points, point_size=voxel_size_in_mm)
            if self.plot_colors:
                if None is not self.sparse_colors:
                    i, j, k = self.sparse_indexes[:, 0], \
                        self.sparse_indexes[:, 1], \
                        self.sparse_indexes[:, 2]
                    colors = self.sparse_colors[i, j, k]
                    point_colors = rgb_to_packed_colors(
                        colors[mask, 0], colors[mask, 1], colors[mask, 2])
                    args['point_colors'] = point_colors
            else:
                args['attrs'] = np.abs(self.sparse_sdf[mask].ravel())

            plottable = PointsPlottable(**args)

        else:
            raise ValueError(f'{self.plot_type}')

        return plottable.plot(k3d_plot)


CameraByChunk = Mapping[str, np.ndarray]


@dataclass
class VoxelChunkData(Plottable, PathsLoadable):
    camera_ids: CameraByChunk = None
    camera_views: Mapping[int, CameraView] = None
    chunk_volumes: List[ChunkVolume] = None
    full_volume: FullVolume = None

    @classmethod
    def from_paths(cls, paths: 'VoxelDataPaths', *args, **kwargs):
        pass

    def plot(self, k3d_plot):
        for view in self.camera_views.values():
            view.plot(k3d_plot)
        for volume in self.chunk_volumes:
            volume.plot(k3d_plot)
        self.full_volume.plot(k3d_plot)


__all__ = [
    'ChunkVolume',
    'CameraView',
    'ChunkVolume',
    'FullVolume',
    'VoxelChunkData',
    'CameraByChunk',
]
