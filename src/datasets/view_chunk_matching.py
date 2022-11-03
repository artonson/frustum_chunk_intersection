import numpy as np
import trimesh.transformations as tt
from scipy.spatial import cKDTree

from src.geometry.camera_pose import CameraPose
from src.geometry.projection import unproject_rgbd
from src.objects import ChunkVolume, CameraView


class ChunkAssociation:
    """
    Holds K-D tree.
    """
    def __init__(self, chunk_volume):
        # generate points based on chunk_volume
        self.chunk_points = chunk_volume.volume.xyz_world
        self.tree = cKDTree(self.chunk_points)

    def compute_overlap(
            self,
            camera_view: CameraView,
            max_distance_thr: float = 0.02,
            min_depth: float = 1e-2,
            max_depth: float = 6.0,
    ):
        # # the previous version below was projecting the points
        # # into the view, then counting how many chunk's points
        # # are visible in the view's frustum
        # we're using it as a pre-check for speed.
        unprojected_pc = CameraPose(camera_view.extrinsics) \
            .world_to_camera(self.chunk_points)
        depth = unprojected_pc[:, 2].copy()
        unprojected_pc /= np.atleast_2d(depth).T
        uv = tt.transform_points(unprojected_pc, camera_view.intrinsics)
        height, width = camera_view.depth.shape
        mask = (uv[:, 0] > 0) & (uv[:, 0] < width) \
               & (uv[:, 1] > 0) & (uv[:, 1] < height) \
               & (depth > min_depth)
        num_inside_points = np.sum(mask)
        if num_inside_points == 0:
            return 0.

        view_points = unproject_rgbd(camera_view).depth
        view_point_to_chunk_distances, _ = self.tree.query(
            view_points, k=1, distance_upper_bound=max_distance_thr)
        return np.sum(view_point_to_chunk_distances < np.inf) / \
               len(view_point_to_chunk_distances)


def compute_fraction_of_view_in_chunk(
        chunk_volume: ChunkVolume,
        camera_view: CameraView,
        max_distance_thr: float = 0.02,
):
    """Compute fraction of points visible inside the view.

    For this, we unproject depth from the input view into
    the world frame, obtaining a 3D point cloud,
    query the closest points from the chunk, and
    count the fraction of the view's points that have
    neighbours in this chunk within a tolerance.
    """
    # generate points based on chunk_volume
    chunk_points = chunk_volume.volume.xyz_world

    # # the previous version below was projecting the points
    # # into the view, then counting how many chunk's points
    # # are visible in the view's frustum
    # we're using it as a pre-check for speed.
    min_depth: float = 1e-2
    max_depth: float = 6.0
    unprojected_pc = CameraPose(camera_view.extrinsics)\
        .world_to_camera(chunk_points)
    depth = unprojected_pc[:, 2].copy()
    unprojected_pc /= np.atleast_2d(depth).T
    uv = tt.transform_points(unprojected_pc, camera_view.intrinsics)
    height, width = camera_view.depth.shape
    mask = (uv[:, 0] > 0) & (uv[:, 0] < width) \
           & (uv[:, 1] > 0) & (uv[:, 1] < height) \
           & (depth > min_depth)
    num_inside_points = np.sum(mask)
    if num_inside_points == 0:
        return 0.

    view_points = unproject_rgbd(camera_view).depth
    view_point_to_chunk_distances, _ = cKDTree(chunk_points).query(
        view_points, k=1, distance_upper_bound=max_distance_thr)
    return np.sum(view_point_to_chunk_distances < np.inf) / \
           len(view_point_to_chunk_distances)


def is_visible(
        chunk_volume: ChunkVolume,
        camera_view: CameraView,
        fraction: float = 0.8,
        max_distance_thr: float = 0.02,
):
    fraction_inside = compute_fraction_of_view_in_chunk(
        chunk_volume,
        camera_view,
        max_distance_thr=max_distance_thr)
    return fraction_inside >= fraction
