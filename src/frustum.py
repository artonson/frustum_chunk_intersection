import numpy as np

from src.camera_pose import CameraPose


class CameraFrustum:
    """Provide fast checks about what is visible in the camera.

    CameraFrustum uses a pinhole intrinsic model.

    This is an implementation of the algorithm from this
    StackOverflow issue:
    https://math.stackexchange.com/questions/4144827/determine-if-a-point-is-in-a-cameras-field-of-view-3d
    """
    def __init__(self, camera_to_world_4x4: np.ndarray, intrinsics: np.ndarray):
        self._extrinsics = CameraPose(camera_to_world_4x4)
        self._intrinsics = np.copy(intrinsics)

    def is_visible(self, points: np.ndarray) -> np.ndarray:
        """Given an [n, 3] array of 3D points, return their
        visibility mask in the frustum."""
        # work in the reference frame where the origin is (0, 0, 0)
        points = points - self._extrinsics.frame_origin
        cx, cy, cz = self._extrinsics.frame_axes

        # compute z coordinate of each point (in camera frame)
        # as a coordinate along camera Z axis
        z_coord = points @ cz
        z_idx = np.where(z_coord > 0)[0]

        # the projection plane is offset by focal distance
        # from the center of projection in the direction of Z axis
        Kf = self._intrinsics[0]
        f = Kf[0, 0]
        # compute uv coordinates relative to center of projection plane
        uv = f * points[z_idx] / z_coord[z_idx, np.newaxis] - f * cz
        u = uv @ cx
        v = uv @ cy

        # obtain sensor half-size in m
        Ks = self._intrinsics[1]
        shw = 1. / Ks[0, 0] * Ks[0, 2]
        shh = 1. / Ks[1, 1] * Ks[1, 2]
        u_idx = np.where((u > -shw).astype(bool) * (u < shw).astype(bool))[0]
        v_idx = np.where(
            (v[u_idx] > -shh).astype(bool) *
            (v[u_idx] < shh).astype(bool))[0]

        mask = np.zeros(len(points), dtype=bool)
        mask[z_idx[u_idx[v_idx]]] = True
        return mask
