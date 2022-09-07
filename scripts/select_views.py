#!/usr/bin/env python3

import argparse
import os
import sys

__this_dir__ = os.path.dirname(os.path.realpath(__file__))
__dir__ = os.path.normpath(os.path.join(__this_dir__, '..'))
sys.path[1:1] = [__dir__]

from src.argparse import PathType
from src.voxel_chunk_data import VoxelDataPaths


def main(options):
    if options.verbose:
        print('Loading data')

    paths = VoxelDataPaths(
        data_root=options.data_dir,
        scene_id=options.scene_id,
        room_id=options.room_id,
        type_id=options.type_id,
        chunk_id=options.chunk_id,
        fraction=options.overlap_fraction)

    paths.load()

    for volume in paths.chunk_volumes:
        # used in determining which voxels represent surface
        volume.plot_sdf_thr = options.sdf_thr

    if options.verbose:
        print('Computing chunk-voxel visibility')
    if options.output_fraction:
        visibility_map = paths.compute_fraction_voxels_in_view()
    else:
        visibility_map = paths.compute_voxel_visibility()

    if options.verbose:
        print('Saving outputs')
    os.makedirs(options.output_dir, exist_ok=True)
    for chunk_id, camera_ids in visibility_map.items():
        output_filename = os.path.join(
            options.output_dir,
            f'{options.scene_id}_room{options.room_id}__{options.type_id}__{chunk_id}.txt')
        with open(output_filename, 'w') as f:
            if options.output_fraction:
                f.write('\n'.join([
                    f'{id} {value:.4f}' for id, value in camera_ids.items()]))
            else:
                f.write('\n'.join([str(id) for id in camera_ids]))


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
        '-c', '--chunk',
        dest='chunk_id',
        type=str,
        required=True,
        help='name of the chunk to load.')
    parser.add_argument(
        '-tp', '--type',
        dest='type_id',
        type=str,
        required=True,
        help='name of the chunk to load [cmp, inc].')

    parser.add_argument(
        '-l', '--overlap',
        dest='overlap_fraction',
        type=float,
        default=0.6,
        help='when computing overlap between view and voxels, '
             'this fraction of surface voxels needs to be visible.')

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
        '-f', '--output-fraction',
        dest='output_fraction',
        action='store_true',
        default=False,
        required=False,
        help='if set, outputs the fraction of voxels inside '
             'each camera view.')

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
