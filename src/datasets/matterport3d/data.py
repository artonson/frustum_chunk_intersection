import glob
import os
import sys
from typing import List

import numpy as np

from src.datasets.large_scale_indoor import LargeScaleIndoorDataPaths
from src.datasets.matterport3d.reader import load_sdf
from src.geometry.volume_view import DenseVolumeView, VolumeView, SparseVolumeView
from src.objects import VoxelChunkData


def split_chunkvolume_filename(s):
    # f'{self.scene_id}_room{self.room_id}__{self.type}__{self.chunk_id}.sdf'
    s = os.path.basename(s)
    s, sdf = os.path.splitext(s)
    scene_room, tp, chunk_id = s.split('__')
    scene_id, room_id = scene_room.split('_')
    room_id = room_id[4:]
    return scene_id, room_id, tp, chunk_id


class Matterport3dDataPaths(LargeScaleIndoorDataPaths):
    DATA_FRAMES_DIR = 'data-frames'
    IMAGES_DIR = 'images'
    EXTRINSICS_DIR = 'camera'
    INTRINSICS_DIR = 'camera'
    RGB_DIR = 'color'
    DEPTH_DIR = 'depth'
    CHUNK_VOLUMES_DIR = 'data-geo-color-128'
    SCENE_VOLUMES_DIR = 'mp_sdf_2cm_target'
    CHUNK_VERSION = '2'  # 1 for older 64 chunks, 2 for newer 128 chunks

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

        self._data = VoxelChunkData()
        if load:
            self._data = self._load()

    def get_chunk_ids_by_wildcard(self) -> List[str]:
        wildcard = self.get_chunk_filename('*')
        chunk_filenames = glob.glob(wildcard)
        chunk_ids = [split_chunkvolume_filename(fn)[-1] for fn in chunk_filenames]
        return chunk_ids

    def get_cameras_dataframe_filename(self, chunk_id):
        df_filename = os.path.join(
            self.data_root, self.DATA_FRAMES_DIR,
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

    def get_scene_filename(self, scene_id: str) -> str:
        sdf_filename = os.path.join(
            self.data_root, self.SCENE_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.sdf')
        rgb_filename = os.path.join(
            self.data_root, self.SCENE_VOLUMES_DIR,
            f'{self.scene_id}_room{self.room_id}__{0}__.colors')
        if self.verbose:
            print(sdf_filename) ; print((rgb_filename))
        return sdf_filename, rgb_filename

    def get_extrinsics(self, extrinsics_filename: str) -> np.array:
        camera_params = np.loadtxt(extrinsics_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]
        return extrinsics

    def get_intrinsics(self, intrinsics_filename) -> np.array:
        camera_params = np.loadtxt(intrinsics_filename)
        extrinsics, intrinsics = camera_params[:4], camera_params[4:]
        return intrinsics

    def get_chunk(self, chunk_filename: str) -> VolumeView:
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
            assert None is not self.scene_volume, \
                f'you must first load full volume in version {self.CHUNK_VERSION}'
            scene_transform = self.scene_volume.volume.transform
            scene_translation = scene_transform[:3, 3]
            chunk_translation = chunk_transform[:3, 3]
            relative_translation = chunk_translation - scene_translation
            corrected_chunk_origin = scene_translation - relative_translation
            chunk_transform[:3, 3] = corrected_chunk_origin

        volume = DenseVolumeView(sdf, chunk_transform, known, colors)
        return volume

    def get_scene(self, scene_filename) -> VolumeView:
        sdf_filename, rgb_filename = scene_filename  # hack
        (zyx_grid, sdf), shape_zyx, transform, known, colors = load_sdf(
            file=sdf_filename,
            load_sparse=True,
            load_known=False,
            load_colors=True,
            color_file=rgb_filename)
        volume = SparseVolumeView(
            sdf, transform, known, colors,
            zyx_grid_=zyx_grid,
            shape=shape_zyx)
        return volume

    def get_dataframes(self):
        camera_ids_by_chunk = None
        if os.path.exists(os.path.join(self.data_root, self.DATA_FRAMES_DIR)):
            try:
                camera_ids_by_chunk = {}
                for chunk_id in self.chunk_ids:
                    ids = np.loadtxt(
                        self.get_cameras_dataframe_filename(chunk_id), dtype=np.int_)
                    camera_ids_by_chunk[chunk_id] = ids[ids != -1]
                self._data.camera_ids = np.unique(np.concatenate(list(camera_ids_by_chunk.values())))
            except Exception as e:
                print(f'Cannot read voxel-chunk correspondences: {str(e)}',
                      file=sys.stderr)
                self._data.camera_ids = camera_ids_by_chunk
