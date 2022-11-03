from enum import Enum

from src.datasets.matterport3d.data import Matterport3dDataPaths
from src.datasets.scannet.data import ScannetDataPaths


class DatasetType(Enum):
    MATTERPORT3D = 'matterport3d'
    SCANNET = 'scannet'


DATASET_BY_TYPE = {
    DatasetType.MATTERPORT3D.value: Matterport3dDataPaths,
    DatasetType.SCANNET.value: ScannetDataPaths,
}

DATASET_TYPES = [v.value for v in DatasetType]
