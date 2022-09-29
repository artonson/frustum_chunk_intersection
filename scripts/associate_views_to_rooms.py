#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

__this_dir__ = os.path.dirname(os.path.realpath(__file__))
__dir__ = os.path.normpath(os.path.join(__this_dir__, '..'))
sys.path[1:1] = [__dir__]

from src.argparse import PathType
from src.voxel_chunk_data import VoxelDataPaths


def main(options):
    if options.verbose:
        print('Loading data')

    VoxelDataPaths.CHUNK_VOLUMES_DIR = 'none'  # don't load chunks
    VoxelDataPaths.RGB_DIR = 'none'  # don't load RGB
    VoxelDataPaths.DEPTH_DIR = 'none'  # don't load depth
    paths = VoxelDataPaths(
        data_root=options.data_dir,
        scene_id=options.scene_id,
        room_id=options.room_id,
        type_id='none',
        chunk_id='none',
        verbose=options.verbose)

    paths.load()

    full_volume = paths.full_volume
    full_volume.plot_sdf_thr = options.sdf_thr
    bbox = np.vstack(
        (np.min(full_volume.voxels_xyz, axis=0),
         np.max(full_volume.voxels_xyz, axis=0)))

    if options.verbose:
        print('Computing room-view association')
    camera_views_in_room = []
    for camera_id, camera_view in paths.camera_views.items():
        camera_t = camera_view.extrinsics[:3, 3]
        if (camera_t > bbox[0]).all() and (camera_t < bbox[1]).all():
            camera_views_in_room.append(camera_id)

    if options.verbose:
        print('Saving outputs')
    os.makedirs(options.output_dir, exist_ok=True)
    output_filename = os.path.join(
        options.output_dir,
        f'{options.scene_id}_room{options.room_id}.txt')
    with open(output_filename, 'w') as f:
        f.write('\n'.join(sorted(camera_views_in_room)))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-d', '--data-dir',
        dest='data_dir',
        type=PathType(exists=True, type='dir', dash_ok=False),
        required=True,
        help='path to root of the data tree.')
    parser.add_argument(
        '-s', '--scene',
        dest='scene_id',
        type=str,
        required=True,
        help='name of the scene to load.')
    parser.add_argument(
        '-r', '--room',
        dest='room_id',
        type=str,
        required=True,
        help='name of the room to load.')

    parser.add_argument(
        '-t', '--sdf-thr',
        dest='sdf_thr',
        type=float,
        default=0.01,
        help='when computing overlap between view and voxels, '
             'consider voxels with SDF smaller than this to be '
             'surface voxels.')

    parser.add_argument(
        '-o', '--output-dir',
        dest='output_dir',
        type=PathType(exists=None, type='dir', dash_ok=False),
        required=True,
        help='path to root of the output.')

    parser.add_argument(
        '-v', '--verbose',
        dest='verbose',
        action='store_true',
        default=False,
        required=False,
        help='be verbose')
    return parser.parse_args()


if __name__ == '__main__':
    options = parse_args()
    main(options)
