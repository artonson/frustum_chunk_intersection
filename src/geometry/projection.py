import numpy as np
import trimesh.transformations as tt

from src.geometry.camera_pose import CameraPose


def project_rgbd(camera_view, points, colors):
    """Given a scene with 3D points and RGB colors, and
    a camera view with depth, rgb, intrinsics, and extrinsics,
    compute an output camera view with depth, rgb, intrinsics, and extrinsics,
    by projecting the 3D points to camera frame."""
    unprojected_pc = CameraPose(camera_view.extrinsics).world_to_camera(points)
    depth = unprojected_pc[:, 2].copy()
    unprojected_pc /= np.atleast_2d(depth).T
    uv = tt.transform_points(unprojected_pc, camera_view.intrinsics)
    height, width = camera_view.depth.shape
    mask = (uv[:, 0] > 0) & (uv[:, 0] < width) \
           & (uv[:, 1] > 0) & (uv[:, 1] < height) \
           & (depth > 1e-2)
    uv_int = np.floor(uv[mask]).astype(np.int_)
    out_depth = np.zeros_like(camera_view.depth, dtype=np.float32)
    out_depth[uv_int[:, 1], uv_int[:, 0]] = depth[mask]
    out_color = np.zeros_like(camera_view.rgb, dtype=np.int_)
    out_color[uv_int[:, 1], uv_int[:, 0]] = colors[mask]

    from copy import deepcopy
    output_view = deepcopy(camera_view)
    output_view.rgb = out_color
    output_view.depth = out_depth
    return output_view


def unproject_rgbd(camera_view: 'CameraView'):
    """Given a camera view with depth, rgb, intrinsics, and extrinsics,
    compute a camera view with point cloud defined in global frame,
    rgb, intrinsics, and extrinsics."""
    pixels = camera_view.depth
    height, width = pixels.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height))
    image_integers = np.stack((
        i.ravel(),
        j.ravel(),
        np.ones_like(i).ravel()
    )).T  # [n, 3]
    image_integers = image_integers.astype(np.float32)
    depth_integers = pixels.ravel()
    image_integers = image_integers[depth_integers != 0]
    colors = camera_view.rgb.reshape((-1, 3))[depth_integers != 0]
    depth = depth_integers[depth_integers != 0].astype(np.float32) / 1000
    unprojected_depth = tt.transform_points(
        image_integers,
        np.linalg.inv(camera_view.intrinsics))
    unprojected_depth *= np.atleast_2d(depth).T
    unprojected_pc = CameraPose(camera_view.extrinsics).camera_to_world(unprojected_depth)

    from copy import deepcopy
    output_view = deepcopy(camera_view)
    output_view.rgb = colors
    output_view.depth = unprojected_pc
    return output_view


__all__ = [
    'project_rgbd',
    'unproject_rgbd',
]
