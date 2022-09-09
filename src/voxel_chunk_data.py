import glob
import sys
from collections import defaultdict
import os
from typing import Mapping, Tuple, List

import numpy as np
import trimesh.transformations as tt
from tqdm import tqdm

from src.camera_pose import CameraPose
from src.objects import (
    ChunkVolume, CameraView, VoxelChunkData, FullVolume)


def compute_fraction_voxels_in_view(
        chunk_volume: ChunkVolume,
        camera_view: CameraView,
        frame_size: Tuple[int, int] = None,
):
    """Compute fraction of points visible inside the view."""
    points = chunk_volume.voxels_xyz  # generate points based on chunk_volume
    projected = tt.transform_points(
        CameraPose(camera_view.extrinsics).world_to_camera(points),
        camera_view.intrinsics)
    if None is not frame_size:
        width, height = frame_size
    else:
        width = 2 * camera_view.intrinsics[0, 2]
        height = 2 * camera_view.intrinsics[1, 2]
    mask = (projected[:, 0] > 0) & (projected[:, 0] < width) \
           & (projected[:, 1] > 0) & (projected[:, 1] < height) \
           & (projected[:, 2] > 0)
    num_inside_points = np.sum(mask)
    return num_inside_points / len(points)


def is_visible(
        chunk_volume: ChunkVolume,
        camera_view: CameraView,
        fraction: float = 0.8,
        frame_size: Tuple[int, int] = None,
):
    fraction_inside = compute_fraction_voxels_in_view(
        chunk_volume,
        camera_view,
        frame_size,)
    return fraction_inside >= fraction


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
            type_id: str = 'cmp',
            load: bool = False,
            fraction: float = 0.8,
            verbose: bool = False,
    ):
        self.data_root = data_root
        self.scene_id = scene_id
        self.room_id = room_id
        self.type_id = type_id
        self.fraction = fraction
        self.verbose = verbose

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

    def get_full_filenames(self):
        sdf_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.sdf')
        rgb_filename = os.path.join(self.data_root, self.FULL_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.colors')
        if self.verbose:
            print(sdf_filename) ; print((rgb_filename))
        return sdf_filename, rgb_filename

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
        if None is camera_ids:
            # load all cameras
            wildcard = self.get_calib_filename('*')
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

        # the SDF of the entire room (might be memory intensive)
        if self.verbose:
            print('Loading full volumes')
        full_volume = None
        if os.path.exists(os.path.join(self.data_root, self.FULL_VOLUMES_DIR)):
            try:
                full_volume = FullVolume.from_paths(self)
            except Exception as e:
                print(f'Cannot read full volumes: {str(e)}', file=sys.stderr)

        return VoxelChunkData(
            camera_ids=camera_ids_by_chunk,
            camera_views=camera_views,
            chunk_volumes=chunk_volumes,
            full_volumes=full_volume)

    def load(self):
        self._data = self._load()

    def compute_voxel_visibility(self) -> Mapping[int, List[int]]:
        visibility = defaultdict(list)
        for chunk_volume in self._data.chunk_volumes:
            iterable = self._data.camera_views.values()
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                if is_visible(chunk_volume, camera_view, fraction=self.fraction):
                    visibility[chunk_volume.id].append(camera_view.id)
        return visibility

    def compute_fraction_voxels_in_view(self) -> Mapping[int, Mapping[int, float]]:
        visibility = defaultdict(lambda: defaultdict(float))
        for chunk_volume in self._data.chunk_volumes:
            iterable = self._data.camera_views.values()
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                fraction = compute_fraction_voxels_in_view(
                    chunk_volume, camera_view)
                visibility[chunk_volume.id][camera_view.id] = fraction
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
