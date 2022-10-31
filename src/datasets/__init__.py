from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple

import numpy as np

from src.objects import VoxelChunkData


class DataPaths(ABC):
    """
    Base class for loading voxelized datasets
    obtained via depth fusion from RGBD sequences.
    """

    def __init__(
        self,
        verbose: bool = False,
    ):
        self.verbose = verbose

    @abstractmethod
    def get_extrinsics_filename(self, camera_id: str) -> str: ...

    @abstractmethod
    def get_intrinsics_filename(self, camera_id: str) -> str: ...

    @abstractmethod
    def get_rgb_filename(self, camera_id: str) -> str: ...

    @abstractmethod
    def get_depth_filename(self, camera_id: str) -> str: ...

    @abstractmethod
    def get_extrinsics(self, extrinsics_filename: str) -> np.array: ...

    @abstractmethod
    def get_intrinsics(self, intrinsics_filename) -> np.array: ...

    @abstractmethod
    def get_depth(self, depth_filename: str) -> np.array: ...

    @abstractmethod
    def get_rgb(self, rgb_filename: str) -> np.array: ...

    @abstractmethod
    def get_chunk_filename(self, chunk_id: str) -> str: ...

    @abstractmethod
    def get_chunk(self, chunk_filename: str) -> Tuple: ...
    # returns: sdf, transform, known, colors

    @abstractmethod
    def get_scene_filename(self, scene_id: str) -> str: ...

    @abstractmethod
    def get_scene(self, scene_filename: str) -> Tuple: ...
    # returns: sdf, transform, known, colors

    @abstractmethod
    def _load(self) -> VoxelChunkData: ...

    def load(self):
        self._data = self._load()

    @property
    def full_volume(self): return self._data.full_volume

    @property
    def camera_views(self): return self._data.camera_views

    @property
    def chunk_volumes(self): return self._data.chunk_volumes

    @property
    def camera_ids(self): return self._data.camera_ids

    def compute_voxel_visibility(self) -> Mapping[int, List[int]]:
        visibility = defaultdict(list)
        for chunk_volume in self._data.chunk_volumes:
            iterable = self._data.camera_views.values()
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                if is_visible(chunk_volume, camera_view,
                              fraction=self.fraction, max_distance_thr=self.max_distance_thr):
                    visibility[chunk_volume.id].append(camera_view.id)
        return visibility

    def compute_fraction_of_view_in_chunk(self, camera_ids_to_check=None) -> Mapping[int, Mapping[int, float]]:
        visibility = defaultdict(lambda: defaultdict(float))
        for chunk_volume in self._data.chunk_volumes:
            if None is camera_ids_to_check:
                camera_ids_to_check = self.camera_views.keys()
            camera_ids_to_check = set(camera_ids_to_check)
            iterable = [self.camera_views[camera_id] for camera_id in camera_ids_to_check]
            if self.verbose:
                iterable = tqdm(iterable)
            for camera_view in iterable:
                fraction = compute_fraction_of_view_in_chunk(
                    chunk_volume, camera_view, max_distance_thr=self.max_distance_thr)
                visibility[chunk_volume.id][camera_view.id] = fraction
        return visibility

