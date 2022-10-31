import glob
import sys
from collections import defaultdict
import os
from typing import Mapping, List, Tuple

import imageio
import numpy as np
from scipy.spatial import cKDTree
import trimesh.transformations as tt
from tqdm import tqdm

from src.camera_pose import CameraPose
from src.datasets import DataPaths
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, FullVolume, unproject_rgbd)
from src.datasets.matterport3d.reader import load_sdf


def split_chunkvolume_filename(s):
    # f'{self.scene_id}_room{self.room_id}__{self.type}__{self.chunk_id}.sdf'
    s = os.path.basename(s)
    s, sdf = os.path.splitext(s)
    scene_room, tp, chunk_id = s.split('__')
    scene_id, room_id = scene_room.split('_')
    room_id = room_id[4:]
    return scene_id, room_id, tp, chunk_id


class Matterport3dDataPaths(DataPaths):
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'images'
    CALIB_DIR = 'camera'
    RGB_DIR = 'color'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'data-geo-color'
    FULL_VOLUMES_DIR = 'mp_sdf_2cm_target'
    CHUNK_VERSION = '2'  # 1 for older 64 chunks, 2 for newer 128 chunks

    def __init__(
            self,
            data_root: str,
            scene_id: str,
            room_id: str,
            chunk_id: str = '*',
            type_id: str = 'cmp',
            load: bool = False,
            fraction: float = 0.8,
            max_distance_thr: float = 0.02,
            verbose: bool = False,
    ):
        self.data_root = data_root
        self.scene_id = scene_id
        self.room_id = room_id
        self.type_id = type_id
        self.fraction = fraction
        self.max_distance_thr = max_distance_thr
        self.verbose = verbose

        if chunk_id == '*':
            wildcard = self.get_chunk_filename('*')
            chunk_filenames = glob.glob(wildcard)
            self.chunk_ids = [split_chunkvolume_filename(fn)[-1] for fn in chunk_filenames]
        else:
            self.chunk_ids = [chunk_id]

        self._data = VoxelChunkData()
        if load:
            self._data = self._load()

    def get_extrinsics_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.CALIB_DIR, f'{camera_id}.txt')
        if self.verbose:
            print(calib_filename)
        return calib_filename

    def get_intrinsics_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.CALIB_DIR, f'{camera_id}.txt')
        if self.verbose:
            print(calib_filename)
        return calib_filename

    def get_rgb_filename(self, camera_id):
        rgb_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.RGB_DIR, f'{camera_id}.jpg')
        if self.verbose:
            print(rgb_filename)
        return rgb_filename

    def get_depth_filename(self, camera_id):
        depth_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.DEPTH_DIR, f'{camera_id}.png')
        if self.verbose:
            print(depth_filename)
        return depth_filename

    def get_extrinsics(self, extrinsics_filename: str) -> np.array:
        camera_params = np.loadtxt(extrinsics_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]
        return extrinsics

    def get_intrinsics(self, intrinsics_filename) -> np.array:
        camera_params = np.loadtxt(intrinsics_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]
        return intrinsics

    def get_depth(self, depth_filename: str) -> np.array:
        depth_array = imageio.imread(depth_filename)
        return depth_array

    def get_rgb(self, rgb_filename: str) -> np.array:
        rgb_array = imageio.imread(rgb_filename)
        return rgb_array

    def get_cameras_dataframe_filename(self, chunk_id):
        df_filename = os.path.join(self.data_root, self.DATA_FRAMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{self.type_id}__{chunk_id}.txt')
        if self.verbose:
            print(df_filename)
        return df_filename

    def get_chunk_filename(self, chunk_id):
        chunk_filename = f'{self.scene_id}_room{self.room_id}__{self.type_id}__{chunk_id}.sdf'
        chunk_filename = os.path.join(
            self.data_root, self.CHUNK_VOLUMES_DIR, chunk_filename)
        if self.verbose:
            print(chunk_filename)
        return chunk_filename

    def get_chunk(self, chunk_filename: str) -> Tuple:
        if self.CHUNK_VERSION == '1':
            load_colors = True
        elif self.CHUNK_VERSION == '2':
            load_colors = False
        else:
            raise ValueError(
                f'Version {self.CHUNK_VERSION} of {self.__name__} is unknown')

        sdf, chunk_transform, known, colors = load_sdf(
            file=chunk_filename,
            load_sparse=False,
            load_known=False,
            load_colors=load_colors,
            color_file=None)
        sdf[sdf == -np.inf] = np.inf

        if self.CHUNK_VERSION == '2':
            assert None is not self.full_volume, \
                f'you must first load full volume in version {self.CHUNK_VERSION}'
            scene_transform = self.full_volume.transform
            scene_translation = scene_transform[:3, 3]
            chunk_translation = chunk_transform[:3, 3]
            relative_translation = chunk_translation - scene_translation
            corrected_chunk_origin = scene_translation - relative_translation
            chunk_transform[:3, 3] = corrected_chunk_origin

        return sdf, chunk_transform, known, colors

    def get_scene_filename(self, scene_id: str) -> str:
        sdf_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.sdf')
        rgb_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.colors')
        if self.verbose:
            print(sdf_filename) ; print((rgb_filename))
        return sdf_filename, rgb_filename

    def get_scene(self, scene_filename) -> Tuple:
        sdf_filename, rgb_filename = scene_filename
        (indexes, sdf), shape, transform, known, colors = load_sdf(
            file=sdf_filename,
            load_sparse=True,
            load_known=False,
            load_colors=True,
            color_file=rgb_filename)
        room = np.ones(shape, dtype=np.float32) * np.inf
        room[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = sdf
        return indexes, sdf, shape, room, transform, known, colors

    def _load(self) -> VoxelChunkData:
        # camera IDs
        if self.verbose:
            print('Loading camera-chunk correspondences')
        camera_ids = None
        camera_ids_by_chunk = None
        if os.path.exists(os.path.join(self.data_root, self.DATA_FRAMES_DIR)):
            try:
                camera_ids_by_chunk = {}
                for chunk_id in self.chunk_ids:
                    ids = np.loadtxt(
                        self.get_cameras_dataframe_filename(chunk_id), dtype=np.int_)
                    camera_ids_by_chunk[chunk_id] = ids[ids != -1]
                camera_ids = np.unique(np.concatenate(list(camera_ids_by_chunk.values())))
            except Exception as e:
                print(f'Cannot read voxel-chunk correspondences: {str(e)}',
                      file=sys.stderr)
                self._data.camera_ids = camera_ids_by_chunk

        if None is camera_ids:
            # load all cameras
            wildcard = self.get_extrinsics_filename('*')
            filenames = glob.glob(wildcard)
            camera_ids = [os.path.splitext(os.path.basename(filename))[0]
                          for filename in filenames]

        # camera views (ext, int, rgb, d)
        if self.verbose:
            camera_ids = tqdm(camera_ids)
            print('Loading camera views')
        camera_views = None
        if os.path.exists(os.path.join(self.data_root, self.IMAGES_DIR)):
            try:
                camera_views = {
                    camera_id: CameraView.from_paths(self, camera_id)
                    for camera_id in camera_ids}
            except Exception as e:
                print(f'Cannot read camera views: {str(e)}', file=sys.stderr)
            self._data.camera_views = camera_views

        # the SDF of the entire room (might be memory intensive)
        # need to read this first, then read chunks for 128^3
        if self.verbose:
            print('Loading full volumes')
        full_volume = None
        if os.path.exists(os.path.join(self.data_root, self.FULL_VOLUMES_DIR)):
            try:
                full_volume = FullVolume.from_paths(self)
            except Exception as e:
                print(f'Cannot read full volumes: {str(e)}', file=sys.stderr)
            self._data.full_volume = full_volume

        # chunks
        if self.verbose:
            print('Loading chunks')
            self.chunk_ids = tqdm(self.chunk_ids)
        chunk_volumes = None
        if os.path.exists(os.path.join(self.data_root, self.CHUNK_VOLUMES_DIR)):
            try:
                chunk_volumes = [
                    ChunkVolume.from_paths(self, chunk_id)
                    for chunk_id in self.chunk_ids]
            except Exception as e:
                print(f'Cannot read chunk volumes: {str(e)}', file=sys.stderr)
            self._data.chunk_volumes = chunk_volumes

        return self._data
