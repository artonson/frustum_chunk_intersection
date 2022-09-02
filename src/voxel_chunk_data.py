import glob
from collections import defaultdict
import os
from typing import Mapping

import numpy as np

from src.frustum import CameraFrustum
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, FullVolume)


def is_visible(
        chunk_volume: ChunkVolume,
        camera_view: CameraView,
        mode: str = 'all',
):
    """Compute per-voxel visibility mask, then reduce."""
    points = ...  # generate points based on chunk_volume
    frustum = CameraFrustum(camera_view.extrinsics, camera_view.intrinsics)
    mask = frustum.is_visible(points)
    if mode == 'all':
        return np.sum(mask) == len(points)
    elif mode == 'any':
        return np.sum(mask) > 0
    else:
        raise ValueError()


def split_chunkvolume_filename(s):
    # f'{self.scene_id}_room{self.room_id}__{self.type}__{self.chunk_id}.sdf'
    s = os.path.basename(s)
    s, sdf = os.path.splitext(s)
    scene_room, tp, chunk_id = s.split('__')
    scene_id, room_id = scene_room.split('_')
    room_id = room_id[4:]
    return scene_id, room_id, tp, chunk_id


class VoxelDataPaths:
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'images'
    CALIB_DIR = 'camera'
    RGB_DIR = 'color'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'data-geo-color'
    FULL_VOLUMES_DIR = 'mp_sdf_2cm_input'

    def __init__(
            self,
            data_root: str,
            scene_id: str,
            room_id: str,
            chunk_id: str = '*',
            load: bool = False,
    ):
        self.data_root = data_root
        self.scene_id = scene_id
        self.room_id = room_id
        self.type = 'cmp'

        if chunk_id == '*':
            wildcard = self.get_chunk_filename('*')
            chunk_filenames = glob.glob(wildcard)
            self.chunk_ids = [split_chunkvolume_filename(fn)[-1] for fn in chunk_filenames]
        else:
            self.chunk_ids = [chunk_id]

        self._data = None
        if load:
            self._data = self._load()

    def get_calib_filename(self, camera_id):
        calib_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.CALIB_DIR, f'{camera_id}.txt')
        return calib_filename

    def get_rgb_filename(self, camera_id):
        rgb_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.RGB_DIR, f'{camera_id}.jpg')
        return rgb_filename

    def get_depth_filename(self, camera_id):
        depth_filename = os.path.join(
            self.data_root, self.IMAGES_DIR, self.scene_id,
            self.DEPTH_DIR, f'{camera_id}.png')
        return depth_filename

    def get_cameras_dataframe_filename(self, chunk_id):
        df_filename = os.path.join(self.data_root, self.DATA_FRAMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{self.type}__{chunk_id}.txt')
        return df_filename

    def get_chunk_filename(self, chunk_id):
        chunk_filename = f'{self.scene_id}_room{self.room_id}__{self.type}__{chunk_id}.sdf'
        chunk_filename = os.path.join(
            self.data_root, self.CHUNK_VOLUMES_DIR, chunk_filename)
        return chunk_filename

    def get_full_filenames(self):
        sdf_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.sdf')
        rgb_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.colors')
        return sdf_filename, rgb_filename

    def _load(self) -> VoxelChunkData:
        # camera IDs
        camera_ids_by_chunk = {}
        for chunk_id in self.chunk_ids:
            ids = np.loadtxt(
                self.get_cameras_dataframe_filename(chunk_id), dtype=np.int_)
            camera_ids_by_chunk[chunk_id] = ids[ids != -1]
        camera_ids = np.unique(np.concatenate(list(camera_ids_by_chunk.values())))

        # camera views (ext, int, rgb, d)
        camera_views = {camera_id: CameraView.from_paths(self, camera_id)
                        for camera_id in camera_ids}

        # chunks
        chunk_volumes = [
            ChunkVolume.from_paths(self, chunk_id)
            for chunk_id in self.chunk_ids]

        # the SDF of the entire room (might be memory intensive)
        full_volume = FullVolume.from_paths(self)

        return VoxelChunkData(
            camera_ids=camera_ids_by_chunk,
            camera_views=camera_views,
            chunk_volumes=chunk_volumes,
            full_volumes=full_volume)

    def load(self):
        self._data = self._load()

    def compute_voxel_visibility(self) -> Mapping[int, int]:
        visibility = defaultdict(list)
        for chunk_volume in self._data.chunk_volumes:
            for camera_view in self._data.camera_views:
                if is_visible(chunk_volume, camera_view):
                    visibility[chunk_volume.id].append(camera_view.id)
        return visibility

    @property
    def full_volume(self):
        return self._data.full_volumes

    @property
    def camera_views(self):
        return self._data.camera_views

    @property
    def chunk_volumes(self):
        return self._data.chunk_volumes

    @property
    def camera_ids(self):
        return self._data.camera_ids
