import glob
import sys
import os
from typing import Mapping, List, Tuple

import imageio
import numpy as np
from tqdm import tqdm

from src.datasets.large_scale_indoor import LargeScaleIndoorDataPaths
from src.datasets.scannet.reader import load_scannet_chunk
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, SceneVolume, unproject_rgbd)
from src.datasets.matterport3d.reader import load_sdf
from src.volume_view import SparseVolumeView


class ScannetDataPaths(LargeScaleIndoorDataPaths):
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'scannet_re'
    INTRINSICS_DIR = 'intrinsic'
    EXTRINSICS_DIR = 'pose'
    RGB_DIR = 'color'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'scannet_chunk_128'
    FULL_VOLUMES_DIR = 'scannet_sem3dlabel_nyu40'

    def get_extrinsics(self, extrinsics_filename: str) -> np.array:
        extrinsics = np.loadtxt(extrinsics_filename)
        return extrinsics

    def get_intrinsics(self, intrinsics_filename) -> np.array:
        intrinsics = np.loadtxt(intrinsics_filename)
        return intrinsics



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
        super().__init__(verbose)
        self.data_root = data_root
        self.scene_id = scene_id
        self.room_id = room_id
        self.type_id = type_id
        self.fraction = fraction
        self.max_distance_thr = max_distance_thr

        if chunk_id == '*':
            wildcard = self.get_chunk_filename('*')
            chunk_filenames = glob.glob(wildcard)
            self.chunk_ids = [split_chunkvolume_filename(fn)[-1] for fn in chunk_filenames]
        else:
            self.chunk_ids = [chunk_id]

        self._data = VoxelChunkData()
        if load:
            self._data = self._load()

    def get_chunk(self, chunk_filename: str) -> Tuple:
        data = load_scannet_chunk(chunk_filename)
        volume = SparseVolumeView()

        data.locations
        sdf = np.ones(shape, dtype=np.float32) * np.inf
        sdf[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = sdf
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
