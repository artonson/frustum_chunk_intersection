from abc import ABC, abstractmethod
from typing import List, Mapping

import numpy as np

from src.objects import VoxelChunkData, SceneVolume, CameraView, ChunkVolume
from src.geometry.volume_view import VolumeView


class DataPaths(ABC):
    """
    Base interface class for loading voxelized datasets
    obtained via depth fusion from RGBD sequences.
    """

    def __init__(
            self,
            verbose: bool = False,
    ):
        self.verbose = verbose
        self._data = None

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
    def get_chunk(self, chunk_filename: str) -> VolumeView: ...

    @abstractmethod
    def get_scene_filename(self, scene_id: str) -> str: ...

    @abstractmethod
    def get_scene(self, scene_filename: str) -> VolumeView: ...

    @abstractmethod
    def _load(self) -> VoxelChunkData: ...

    def load(self):
        self._data = self._load()

    @property
    def scene_volume(self) -> SceneVolume:
        return self._data.scene_volume

    @property
    def camera_views(self) -> Mapping[str, CameraView]:
        return self._data.camera_views

    @property
    def chunk_volumes(self) -> List[ChunkVolume]:
        return self._data.chunk_volumes

    @property
    def camera_ids(self) -> List[int]:
        return self._data.camera_ids
