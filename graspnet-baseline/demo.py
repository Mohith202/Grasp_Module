""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    intrinsic = meta['intrinsic_matrix']
    factor_depth = meta['factor_depth']

    # generate cloud
    camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:50]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)


# Object list from scene 0000 to 0009
object_list = [0, 2, 5, 7, 8, 9, 11, 14, 15, 17, 18, 20, 21, 22, 26, 29, 30, 34, 36, 37, 38, 40, 41, 43, 44, 46, 48, 51, 52, 56, 57, 58, 60, 61, 62, 63, 66, 69, 70]

# object name with respect to their scene id
obj_dict= {7: {'obj_name': '025_mug.ply', 'obj_path': 'models/025_mug.ply'}, 17: {'obj_name': '018_plum.ply', 'obj_path': 'models/018_plum.ply'}, 18: {'obj_name': '032_knife.ply', 'obj_path': 'models/032_knife.ply'}, 26: {'obj_name': '072-d_toy_airplane.ply', 'obj_path': 'models/072-d_toy_airplane.ply'}, 30: {'obj_name': '072-j_toy_airplane.ply', 'obj_path': 'models/072-j_toy_airplane.ply'}, 37: {'obj_name': 'nzskincare_mouth_rinse.ply', 'obj_path': 'models/nzskincare_mouth_rinse.ply'}, 38: {'obj_name': 'dabao_sod.ply', 'obj_path': 'models/dabao_sod.ply'}, 51: {'obj_name': 'large_elephant.ply', 'obj_path': 'models/large_elephant.ply'}, 58: {'obj_name': 'darlie_box.ply', 'obj_path': 'models/darlie_box.ply'}, 61: {'obj_name': 'dabao_facewash.ply', 'obj_path': 'models/dabao_facewash.ply'}, 63: {'obj_name': 'head_shoulders_supreme.ply', 'obj_path': 'models/head_shoulders_supreme.ply'}, 0: {'obj_name': '003_cracker_box.ply', 'obj_path': 'models/003_cracker_box.ply'}, 5: {'obj_name': '011_banana.ply', 'obj_path': 'models/011_banana.ply'}, 46: {'obj_name': 'dish.ply', 'obj_path': 'models/dish.ply'}, 14: {'obj_name': '015_peach.ply', 'obj_path': 'models/015_peach.ply'}, 15: {'obj_name': '016_pear.ply', 'obj_path': 'models/016_pear.ply'}, 20: {'obj_name': '044_flat_screwdriver.ply', 'obj_path': 'models/044_flat_screwdriver.ply'}, 48: {'obj_name': 'camel.ply', 'obj_path': 'models/camel.ply'}, 66: {'obj_name': 'head_shoulders_care.ply', 'obj_path': 'models/head_shoulders_care.ply'}, 70: {'obj_name': 'tape.ply', 'obj_path': 'models/tape.ply'}, 60: {'obj_name': 'black_mouse.ply', 'obj_path': 'models/black_mouse.ply'}, 43: {'obj_name': 'baoke_marker.ply', 'obj_path': 'models/baoke_marker.ply'}, 52: {'obj_name': 'rhinocero.ply', 'obj_path': 'models/rhinocero.ply'}, 41: {'obj_name': 'darlie_toothpaste.ply', 'obj_path': 'models/darlie_toothpaste.ply'}, 2: {'obj_name': '005_tomato_soup_can.ply', 'obj_path': 'models/005_tomato_soup_can.ply'}, 21: {'obj_name': '057_racquetball.ply', 'obj_path': 'models/057_racquetball.ply'}, 44: {'obj_name': 'hosjam.ply', 'obj_path': 'models/hosjam.ply'}, 62: {'obj_name': 'pantene.ply', 'obj_path': 'models/pantene.ply'}, 22: {'obj_name': '065-b_cups.ply', 'obj_path': 'models/065-b_cups.ply'}, 8: {'obj_name': '035_power_drill.ply', 'obj_path': 'models/035_power_drill.ply'}, 9: {'obj_name': '037_scissors.ply', 'obj_path': 'models/037_scissors.ply'}, 11: {'obj_name': '012_strawberry.ply', 'obj_path': 'models/012_strawberry.ply'}, 29: {'obj_name': '072-i_toy_airplane.ply', 'obj_path': 'models/072-i_toy_airplane.ply'}, 34: {'obj_name': 'sum37_secret_repair.ply', 'obj_path': 'models/sum37_secret_repair.ply'}, 36: {'obj_name': 'dabao_wash_soup.ply', 'obj_path': 'models/dabao_wash_soup.ply'}, 40: {'obj_name': 'kispa_cleanser.ply', 'obj_path': 'models/kispa_cleanser.ply'}, 56: {'obj_name': 'gorilla.ply', 'obj_path': 'models/gorilla.ply'}, 57: {'obj_name': 'weiquan.ply', 'obj_path': 'models/weiquan.ply'}, 69: {'obj_name': 'hippo.ply', 'obj_path': 'models/hippo.ply'}}
