from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union

import k3d
import k3d.helpers
import k3d.colormaps.matplotlib_color_maps
import numpy as np
import trimesh
import trimesh.transformations as tt

from src.camera_pose import CameraPose
from src.colors import rgb_to_packed_colors


@dataclass
class Plottable:
    @abstractmethod
    def plot(self, k3d_plot):
        pass


@dataclass
class TrimeshPlottable(Plottable):
    mesh: trimesh.base.Trimesh
    mesh_colors: bool = False  # is set, get from mesh.visual
    face_color: int = 0xbbbbbb
    wireframe: bool = False
    opacity: float = 1.0
    plot_vert: bool = False
    vert_psize: float = 1.0
    vert_color: int = 0x666666
    vert_shader: str = 'flat'

    def plot(self, k3d_plot):
        if None is self.mesh:
            return

        mesh_args = dict(
            vertices=self.mesh.vertices,
            indices=self.mesh.faces,
            color=self.face_color,
            wireframe=self.wireframe,
            opacity=self.opacity)

        if len(self.mesh.visual.vertex_colors) > 0:
            assert self.mesh.visual.vertex_colors.shape == (len(self.mesh.vertices), 4)
            mesh_args.pop('color')
            r, g, b, a = np.split(self.mesh.visual.vertex_colors, 4, axis=1)
            mesh_args['colors'] = rgb_to_packed_colors(r.ravel(), g.ravel(), b.ravel())

        k3d_plot += k3d.mesh(**mesh_args)

        if self.plot_vert:
            vert_args = dict(
                positions=self.mesh.vertices,
                point_size=self.vert_psize,
                color=self.vert_color,
                shader=self.vert_shader)

            k3d_plot += k3d.points(**vert_args)


@dataclass
class PointsPlottable(Plottable):
    points: np.array
    point_color: int = 0xbbbbbb
    point_colors: List = None  # if set, attrs will be ignored
    point_shader: str = 'flat'
    point_opacity: float = 1.0
    point_size: float = 1.0
    point_opacities: List = None
    cmap: Union[str, np.ndarray] = 'jet'
    range_attr: Tuple[float, float] = ()
    attrs: List[float] = None  # if set, map this to colors using colormap

    def plot(self, k3d_plot):
        if None is self.points:
            return
        args = dict(
            positions=self.points,
            point_size=self.point_size,
            color=self.point_color,
            shader=self.point_shader)

        if None is not self.point_colors:
            args.pop('color')
            args['colors'] = self.point_colors

        elif None is not self.attrs:
            args.pop('color')
            assert None is not self.cmap, 'colormap must be specified'
            if isinstance(self.cmap, str):
                self.cmap = getattr(k3d.colormaps.matplotlib_color_maps, self.cmap)
            point_colors = k3d.helpers.map_colors(
                self.attrs, self.cmap, self.range_attr,
            ).astype(np.uint32)
            args['colors'] = point_colors

        k3d_plot += k3d.points(**args)


@dataclass
class VoxelsPlottable(Plottable):
    voxels: np.array
    opacity: float = 1.0
    color_map: Union[str, np.ndarray] = 'jet'
    wireframe: bool = False
    outlines: bool = False
    outlines_color: int = 0
    bounds: Tuple = None
    order: str = 'xyz'

    def plot(self, k3d_plot):
        if None is self.voxels:
            return
        voxels = self.voxels if self.order == 'xyz' else np.transpose(self.voxels, (2, 1, 0))
        args = dict(
            voxels=voxels,
            wireframe=self.wireframe,
            outlines=self.outlines,
            opacity=self.opacity,
            bounds=self.bounds)
        assert None is not self.color_map, 'colormap must be specified'
        if isinstance(self.color_map, str):
            self.color_map = getattr(k3d.colormaps.matplotlib_color_maps, self.color_map)
        color_map = k3d.helpers.map_colors(
            np.arange(0, 255), self.color_map, (0, 255)
        ).astype(np.uint32)
        args['color_map'] = color_map

        k3d_plot += k3d.voxels(**args)


@dataclass
class VectorsPlottable(Plottable):
    origins: np.array
    vectors: np.array
    color: int = 0xbbbbbb
    colors: List = None  # if set, attrs will be ignored
    use_head: bool = False
    head_size: float = 1.0
    line_width: float = 0.01
    cmap: Union[str, np.ndarray] = 'jet'
    range_attr: Tuple[float, float] = ()
    attrs: List[float] = None  # if set, map this to colors using colormap

    def plot(self, k3d_plot):
        if None is self.origins or None is self.vectors:
            return
        args = dict(
            origins=self.origins,
            vectors=self.vectors,
            use_head=self.use_head,
            head_size=self.head_size,
            line_width=self.line_width,
            color=self.color)

        if None is not self.colors:
            args.pop('color')
            args['colors'] = self.colors

        elif None is not self.attrs:
            args.pop('color')
            assert None is not self.cmap, 'colormap must be specified'
            if isinstance(self.cmap, str):
                self.cmap = getattr(k3d.colormaps.matplotlib_color_maps, self.cmap)
            colors = k3d.helpers.map_colors(
                self.attrs, self.cmap, self.range_attr,
            ).astype(np.uint32)
            args['colors'] = [(c, c) for c in colors]

        k3d_plot += k3d.vectors(**args)


@dataclass
class CameraPlottable(Plottable):
    camera_pose: CameraPose
    line_length: float = 1.0
    line_width: float = 1.0
    use_head: bool = False
    head_size: float = 1.0

    x_color = 0xff0000
    y_color = 0x00ff00
    z_color = 0x0000ff
    name: str = None

    def plot(self, k3d_plot):
        camera_center = np.array([
            self.camera_pose.frame_origin,
            self.camera_pose.frame_origin,
            self.camera_pose.frame_origin])

        camera_frame = np.array(
            [self.camera_pose.frame_axes]) * self.line_length

        vectors = k3d.vectors(
            camera_center,
            camera_frame,
            use_head=self.use_head,
            head_size=self.head_size,
            line_width=self.line_width,
            colors=[
                self.x_color, self.x_color,
                self.y_color, self.y_color,
                self.z_color, self.z_color], 
            name=self.name)
        k3d_plot += vectors


@dataclass
class CameraFrustumPlottable(Plottable):
    camera_pose: CameraPose
    image_size: np.ndarray  # in pixels
    focal_length: float = None  # in mm
    sensor_size: np.ndarray = None  # in mm
    principal_point: np.ndarray = None  # in pixels
    intrinsics: np.ndarray = None  # if set, replaces previous params
    line_length: float = 1.0
    face_color: int = 0xbbbbbb
    wireframe: bool = False
    opacity: float = 1.0
    name: str = None

    def plot(self, k3d_plot):
        # compute coordinates of image corners in camera
        # reference frame
        ix, iy = self.image_size
        if None is not self.intrinsics:
            image_corners = np.array([
                [ 0,  0, 1],
                [ix,  0, 1],
                [ix, iy, 1],
                [ 0, iy, 1],
            ])
            image_corners = tt.transform_points(
                image_corners,
                np.linalg.inv(self.intrinsics))
            image_corners *= self.line_length
        else:
            sx, sy = self.sensor_size / self.image_size  # mm/px
            px, py = self.principal_point
            f = self.focal_length[0]
            # the following is in screen coordinates
            # (image plane, Y up, X right, origin = principal point)
            l = -px * sx  # real-world X of the left side of the screen
            r = (ix - px) * sx  # real-world X of the right size of the screen
            b = -py * sy  # real-world Y of the bottom size of the screen
            t = (iy - py) * sy  # real-world Y of the top size of the screen
            image_corners = np.array([
                [l, b, f],
                [r, b, f],
                [r, t, f],
                [l, t, f],
            ]) * self.line_length

        image_corners = self.camera_pose.camera_to_world(image_corners)

        vertices = np.concatenate(
            (np.atleast_2d(self.camera_pose.frame_origin), image_corners),
            axis=0)
        faces = np.array([
            [2, 1, 0],
            [3, 2, 0],
            [4, 3, 0],
            [1, 4, 0],
            [2, 4, 1],
            [2, 3, 4],
        ])
        mesh_args = dict(
            vertices=vertices,
            indices=faces,
            color=self.face_color,
            wireframe=self.wireframe,
            opacity=self.opacity,
            name=self.name)

        k3d_plot += k3d.mesh(**mesh_args)


@dataclass
class VolumePlottable(Plottable):
    volume: np.array
    color_map: str = 'jet'
    order: str = 'xyz'
    interpolation: bool = False
    model_matrix: np.array = None

    def plot(self, k3d_plot):
        if None is self.volume:
            return
        volume = self.volume if self.order == 'xyz' else np.transpose(
            self.volume, (2, 1, 0))
        args = dict(
            volume=volume,
            interpolation=self.interpolation,
            model_matrix=self.model_matrix
        )
        assert None is not self.color_map, 'colormap must be specified'
        #         if isinstance(self.color_map, str):
        #             self.color_map = getattr(k3d.colormaps.matplotlib_color_maps, self.color_map)
        #         color_map = k3d.helpers.map_colors(
        #             np.arange(0, 255), self.color_map, (0, 255)
        #         ).astype(np.uint32)
        #         args['color_map'] = self.color_map

        k3d_plot += k3d.volume(**args)


def display_3d(
        *plot_args,
        k3d_plot = None,
        display=True,
        **k3d_plot_kwargs,
):
    plot = k3d_plot or k3d.plot(**k3d_plot_kwargs)

    for plottable in plot_args:
        plottable.plot(plot)

    if display:
        plot.display()

    return plot
