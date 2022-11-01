import glob
import sys
import os
from collections import defaultdict
from typing import Mapping, List

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm

from src.datasets.base import DataPaths
from src.datasets.view_chunk_matching import compute_fraction_of_view_in_chunk, \
    is_visible
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, SceneVolume)


class LargeScaleIndoorDataPaths(DataPaths):
    """
    Shared functions for datasets.
    The general structure that is assumed is as follows:
     - dataset is formed of separate scenes,
       and it should be possible to load only data for a single scene
     - scene is cut into chunks
    """

    # these all are dummy variables
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'images'
    INTRINSICS_DIR = 'intrinsics'
    EXTRINSICS_DIR = 'extrinsics'
    RGB_DIR = 'rgb'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'chunks'
    SCENE_VOLUMES_DIR = 'scenes'

    def __init__(
            self,
            data_root: str,
            scene_id: str,
            chunk_id: str = '*',
            type_id: str = 'cmp',
            load: bool = False,
            verbose: bool = False,
    ):
        super().__init__(verbose)

        self.data_root = data_root
        self.scene_id = scene_id
        self.type_id = type_id
        self.chunk_ids = [chunk_id]

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
            self.EXTRINSICS_DIR, f'{camera_id}.txt')
        if self.verbose:
            print(calib_filename)
        return calib_filename

    def get_intrinsics_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.INTRINSICS_DIR, f'{camera_id}.txt')
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

    def get_depth(self, depth_filename: str) -> np.array:
        depth_array = imageio.imread(depth_filename)
        return depth_array

    def get_rgb(self, rgb_filename: str) -> np.array:
        rgb_array = imageio.imread(rgb_filename)
        return rgb_array

    def _load(self) -> VoxelChunkData:
        # camera IDs
        if self.verbose:
            print('Loading camera-chunk correspondences')
        camera_ids = None

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
        if os.path.exists(os.path.join(self.data_root, self.SCENE_VOLUMES_DIR)):
            try:
                full_volume = SceneVolume.from_paths(self, self.scene_id)
            except Exception as e:
                print(f'Cannot read full volumes: {str(e)}', file=sys.stderr)
            self._data.scene_volume = full_volume

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

    def compute_voxel_visibility(
            self,
            fraction: float = 0.8,
            max_distance_thr: float = 0.02,
    ) -> Mapping[int, List[int]]:

        visibility = defaultdict(list)
        for chunk_volume in self._data.chunk_volumes:
            iterable = self._data.camera_views.values()
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                if is_visible(
                        chunk_volume, camera_view,
                        fraction=fraction,
                        max_distance_thr=max_distance_thr):
                    visibility[chunk_volume.id].append(camera_view.id)
        return visibility

    def compute_fraction_of_view_in_chunk(
            self,
            camera_ids_to_check=None,
            max_distance_thr: float = 0.02,
    ) -> Mapping[int, Mapping[int, float]]:
        visibility = defaultdict(lambda: defaultdict(float))
        for chunk_volume in self._data.chunk_volumes:
            if None is camera_ids_to_check:
                camera_ids_to_check = self.camera_views.keys()
            camera_ids_to_check = set(camera_ids_to_check)
            iterable = [self.camera_views[camera_id]
                        for camera_id in camera_ids_to_check]
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                fraction = compute_fraction_of_view_in_chunk(
                    chunk_volume, camera_view,
                    max_distance_thr=max_distance_thr)
                visibility[chunk_volume.id][camera_view.id] = fraction
        return visibility
