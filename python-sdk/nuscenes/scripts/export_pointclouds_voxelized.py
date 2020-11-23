# nuScenes dev-kit.
# Code written by Holger Caesar, 2018.

"""
Export fused pointclouds of a scene to a Wavefront OBJ file.
This pointcloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
"""

import argparse
import os
import os.path as osp
import random
from collections import Counter

import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud


def export_scene_pointcloud(nusc: NuScenes,
                            out_path: str,
                            scene_token: str,
                            channel: str = 'LIDAR_TOP',
                            min_dist: float = 2.0,
                            max_dist: float = 50.0,
                            verbose: bool = True) -> None:
    """
    Export fused point clouds of a scene to a Wavefront OBJ file.
    This pointcloud can be viewed in your favorite 3D rendering tool, e.g. Meshlab or Maya.
    :param nusc: NuScenes instance.
    :param out_path: Output path to write the pointcloud to.
    :param scene_token: Unique identifier of scene to render.
    :param channel: Channel to render.
    :param min_dist: Minimum distance to ego vehicle below which points are dropped.
    :param max_dist: Maximum distance to ego vehicle above which points are dropped.
    :param verbose: Whether to print messages to stdout.
    """
    # Settings.
    voxel_size = 0.2

    # Check inputs.
    valid_channels = ['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT', 'RADAR_BACK_LEFT',
                      'RADAR_BACK_RIGHT']
    camera_channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    assert channel in valid_channels, 'Input channel {} not valid.'.format(channel)

    # Get records from DB.
    scene_rec = nusc.get('scene', scene_token)
    start_sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
    sd_rec = nusc.get('sample_data', start_sample_rec['data'][channel])

    # Make list of frames
    cur_sd_rec = sd_rec
    sd_tokens = []
    while cur_sd_rec['next'] != '':
        cur_sd_rec = nusc.get('sample_data', cur_sd_rec['next'])
        sd_tokens.append(cur_sd_rec['token'])

    all_voxel_inds = []
    for i, sd_token in tqdm(enumerate(sd_tokens)):
        if verbose:
            print('Processing {}'.format(sd_rec['filename']))
        sd_rec = nusc.get('sample_data', sd_token)
        lidar_token = sd_rec['token']
        lidar_rec = nusc.get('sample_data', lidar_token)
        pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, lidar_rec['filename']))

        # Points live in their own reference frame. So they need to be transformed to the global frame.
        # First step: transform the point cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', lidar_rec['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Optional Filter by distance to remove the ego vehicle.
        dists_origin = np.sqrt(np.sum(pc.points[:3, :] ** 2, axis=0))
        keep = np.logical_and(min_dist <= dists_origin, dists_origin <= max_dist)
        pc.points = pc.points[:, keep]
        if verbose:
            print('Distance filter: Keeping %d of %d points...' % (keep.sum(), len(keep)))

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', lidar_rec['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Voxelize
        voxel_inds = np.floor(pc.points[:3, :].T / voxel_size).astype(np.int32)
        all_voxel_inds.append(voxel_inds)

    all_voxel_inds = np.concatenate(all_voxel_inds)
    all_voxel_list = list(map(tuple, all_voxel_inds))
    all_voxel_counts = Counter(all_voxel_list)
    all_voxel_set = set(all_voxel_list)
    all_voxel_length = all_voxel_inds.max(axis=0) - all_voxel_inds.min(axis=0)
    print('Scene: ', scene_rec['name'])
    print('Lidar points: ', len(all_voxel_list))
    print('Sparse voxels: ', len(all_voxel_set))
    sparse_ge3_ratio = np.mean(np.array(list(all_voxel_counts.values())) > 2)
    print('Sparse voxels with >=3 points: ', sparse_ge3_ratio)
    print('Voxel length: ', all_voxel_length)
    print('Non-sparse voxels:', np.prod(all_voxel_length))
    print('Points per sparse voxels: ', len(all_voxel_list) / len(all_voxel_set))
    print()


if __name__ == '__main__':
    # Read input parameters
    parser = argparse.ArgumentParser(description='Export a scene in Wavefront point cloud format.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out_dir', default='~/nuscenes-visualization/pointclouds', type=str, help='Output folder')
    parser.add_argument('--verbose', default=0, type=int, help='Whether to print outputs to stdout')

    args = parser.parse_args()
    out_dir = os.path.expanduser(args.out_dir)
    verbose = bool(args.verbose)

    # Extract pointcloud for the specified scene
    nusc = NuScenes()

    random.seed(42)

    # Subsample scenes.
    scene_tokens = [s['token'] for s in nusc.scene]
    random.shuffle(scene_tokens)
    scene_tokens = scene_tokens[:10]

    for scene_token in scene_tokens:
        export_scene_pointcloud(nusc, '', scene_token, channel='LIDAR_TOP', verbose=verbose)
