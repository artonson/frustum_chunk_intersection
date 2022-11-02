import glob
import sys
import os
from typing import Mapping, List, Tuple

import imageio
import numpy as np
from tqdm import tqdm

from src.datasets.large_scale_indoor import LargeScaleIndoorDataPaths
from src.datasets.scannet.reader import load_scannet_chunk, load_scannet_sdf
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, SceneVolume)
from src.datasets.matterport3d.reader import load_sdf
from src.geometry.volume_view import SparseVolumeView, VolumeView


def split_chunkvolume_filename(s):
    # f'{self.scene_id}_room{self.room_id}__{self.type}__{self.chunk_id}.sdf'
    # "scene0106_02__cmp__17.sdf"
    # f'{self.scene_id}_{self.room_id}__{self.type}__{self.chunk_id}.sdf'
    s = os.path.basename(s)
    s, sdf = os.path.splitext(s)
    scene_room, tp, chunk_id = s.split('__')
    scene_id, room_id = scene_room.split('_')
    return scene_id, room_id, tp, chunk_id


class ScannetDataPaths(LargeScaleIndoorDataPaths):
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'scannet_re'
    INTRINSICS_DIR = 'intrinsic'
    EXTRINSICS_DIR = 'pose'
    RGB_DIR = 'color'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'scannet_chunk_128'
    FULL_VOLUMES_DIR = 'scannet_sem3dlabel_nyu40'

    def __init__(
            self,
            data_root: str,
            scene_id: str,
            room_id: str,
            chunk_id: str = '*',
            type_id: str = 'cmp',
            load: bool = False,
            verbose: bool = False,
    ):
        # need that set before call to get_chunk_ids
        self.room_id = room_id
        self.type_id = type_id
        super().__init__(
            data_root=data_root,
            scene_id=scene_id,
            chunk_id=chunk_id,
            verbose=verbose)
        self.resize_rgb_to_depth = True

        self._data = VoxelChunkData()
        if load:
            self._data = self._load()

    def get_chunk_ids_by_wildcard(self) -> List[str]:
        wildcard = self.get_chunk_filename('*')
        chunk_filenames = glob.glob(wildcard)
        chunk_ids = [split_chunkvolume_filename(fn)[-1] for fn in chunk_filenames]
        return chunk_ids

    def get_extrinsics(self, extrinsics_filename: str) -> np.array:
        extrinsics = np.loadtxt(extrinsics_filename)
        return extrinsics

    def get_intrinsics(self, intrinsics_filename) -> np.array:
        intrinsics = np.loadtxt(intrinsics_filename)
        return intrinsics

    def get_extrinsics_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR,
            f'{self.scene_id}_{self.room_id}',
            self.EXTRINSICS_DIR, f'{camera_id}.txt')
        if self.verbose:
            print(calib_filename)
        return calib_filename

    def get_intrinsics_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR,
            f'{self.scene_id}_{self.room_id}',
            self.INTRINSICS_DIR, f'intrinsic_depth.txt')
        if self.verbose:
            print(calib_filename)
        return calib_filename

    def get_rgb_filename(self, camera_id):
        rgb_filename = os.path.join(
            self.data_root, self.IMAGES_DIR,
            f'{self.scene_id}_{self.room_id}',
            self.RGB_DIR, f'{camera_id}.jpg')
        if self.verbose:
            print(rgb_filename)
        return rgb_filename

    def get_depth_filename(self, camera_id):
        depth_filename = os.path.join(
            self.data_root, self.IMAGES_DIR,
            f'{self.scene_id}_{self.room_id}',
            self.DEPTH_DIR, f'{camera_id}.png')
        if self.verbose:
            print(depth_filename)
        return depth_filename

    def get_chunk_filename(self, chunk_id):
        chunk_filename = f'{self.scene_id}_{self.room_id}__{self.type_id}__{chunk_id}.sdf'
        chunk_filename = os.path.join(
            self.data_root, self.CHUNK_VOLUMES_DIR, chunk_filename)
        if self.verbose:
            print(chunk_filename)
        return chunk_filename

    def get_chunk(self, chunk_filename: str) -> VolumeView:
        data = load_scannet_chunk(chunk_filename)
        volume = SparseVolumeView(
            sdf=data.sdfs,
            transform=data.chunk_world2grid,
            known=None,
            colors=data.colors,
            zyx_grid_=data.locations,
            shape=(data.dimz, data.dimy, data.dimx), )
        return volume

    def get_scene_filename(self, scene_id: str) -> str:
        sdf_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_{self.room_id}-sparse-sem-color.npy')
        if self.verbose:
            print(sdf_filename)
        return sdf_filename

    def get_scene(self, scene_filename) -> VolumeView:
        data = load_scannet_sdf(scene_filename, load_sparse=True)
        final_world2grid = data.world2grid @ data.Tori @ data.Taxisalign
        volume = SparseVolumeView(
            sdf=data.sdfs,
            transform=final_world2grid,
            known=None,
            colors=data.colors,
            zyx_grid_=data.locations,
            shape=(data.dimz, data.dimy, data.dimx),)
        return volume
