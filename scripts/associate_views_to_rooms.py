#!/usr/bin/env python3

import argparse
import os
import sys

import numpy as np

__this_dir__ = os.path.dirname(os.path.realpath(__file__))
__dir__ = os.path.normpath(os.path.join(__this_dir__, '..'))
sys.path[1:1] = [__dir__]

from src.datasets import DatasetType, DATASET_BY_TYPE, DATASET_TYPES
from src.utils.argparse import PathType


def main(options):
    if options.verbose:
        print('Loading data')

    dataset_class = DATASET_BY_TYPE[options.data_type]

    dataset_class.CHUNK_VOLUMES_DIR = 'none'  # don't load chunks
    dataset_class.RGB_DIR = 'none'  # don't load RGB
    dataset_class.DEPTH_DIR = 'none'  # don't load depth
    paths = dataset_class(
        data_root=options.data_dir,
        scene_id=options.scene_id,
        room_id=options.room_id,
        type_id='none',
        chunk_id='none',
        verbose=options.verbose)

    paths.load()

    scene_volume = paths.scene_volume
    scene_volume.plot_sdf_thr = options.sdf_thr
    bbox = np.vstack(
        (np.min(scene_volume.volume.xyz_world, axis=0),
         np.max(scene_volume.volume.xyz_world, axis=0)))

    if options.verbose:
        print('Computing room-view association')
    camera_views_in_room = []
    for camera_id, camera_view in paths.camera_views.items():
        camera_t = camera_view.extrinsics[:3, 3]
        if (camera_t > bbox[0]).all() and (camera_t < bbox[1]).all():
            camera_views_in_room.append(camera_id)

    if options.verbose:
        print(f'Saving outputs: found {len(camera_views_in_room)}')
    os.makedirs(options.output_dir, exist_ok=True)
    output_filename = os.path.join(
        options.output_dir,
        f'{options.scene_id}_room{options.room_id}.txt')
    with open(output_filename, 'w') as f:
        camera_views_in_room = sorted(
            camera_views_in_room, key=lambda id_: int(id_))
        f.write('\n'.join(camera_views_in_room))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-y', '--data-type',
        dest='data_type',
        choices=DATASET_TYPES,
        default=DatasetType.MATTERPORT3D.value,
        help='dataset structure to presuppose. ')

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
