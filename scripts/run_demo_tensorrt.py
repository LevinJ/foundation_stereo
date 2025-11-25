import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore')  # TODO

import argparse
import cv2
import glob
import imageio
import logging
import os
import numpy as np
from typing import List
import copy

import omegaconf
import onnxruntime as ort
import open3d as o3d
import torch
import yaml
import time
from onnx_tensorrt import tensorrt_engine
import tensorrt as trt

import sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import *
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder
from  scripts.zed_preprocess import ZedPreprocessor

    

def preprocess(image_path, args):
    input_image = imageio.imread(image_path)
    if args.height and args.width:
      input_image = cv2.resize(input_image, (args.width, args.height))
    resized_image = torch.as_tensor(input_image.copy()).float()[None].permute(0,3,1,2).contiguous()
    return resized_image, input_image


def get_onnx_model(args):
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model = ort.InferenceSession(args.pretrained, sess_options=session_options, providers=['CUDAExecutionProvider'])
    return model


def get_engine_model(args):
    with open(args.pretrained, 'rb') as file:
        engine_data = file.read()
    engine = trt.Runtime(trt.Logger(trt.Logger.WARNING)).deserialize_cuda_engine(engine_data)
    engine = tensorrt_engine.Engine(engine)
    return engine


def inference(left_img_path: str, right_img_path: str, model, args: argparse.Namespace):
    # left_img, input_left = preprocess(left_img_path, args)
    # right_img, _ = preprocess(right_img_path, args)
    zp = ZedPreprocessor( scale=0.5, only_road = True)
    left_img, input_left = zp.prepare(left_img_path)
    right_img, _ = zp.prepare(right_img_path)

    for _ in range(10):
      torch.cuda.synchronize()
      start_time = time.time()
      if args.pretrained.endswith('.onnx'):
          left_disp = model.run(None, {'left': left_img.numpy(), 'right': right_img.numpy()})[0]
      else:
          left_disp = model.run([left_img.numpy(), right_img.numpy()])[0]
      torch.cuda.synchronize()
      end_time = time.time()
      logging.info(f'Inference time: {end_time - start_time:.3f} seconds')

    left_disp = left_disp.squeeze()  # HxW

    # vis = vis_disparity(left_disp)
    # vis = np.concatenate([input_left, vis], axis=1)
    # imageio.imwrite(os.path.join(args.save_path, 'visual', left_img_path.split('/')[-1]), vis)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(input_left)
    ax[0].set_title('Input Left')
    ax[0].axis('off')
    ax[1].imshow(left_disp)
    ax[1].set_title('Disparity Visualization')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
   

    if args.pc:
        save_path = left_img_path.split('/')[-1].split('.')[0] + '.ply'
        baseline = 0.120  # in meters
        doffs = 0
        K = zp.updated_K().copy()
        depth = K[0,0]*baseline/(left_disp + doffs)
        

        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1,3), input_left.reshape(-1,3))
        keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
        keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
        pcd = pcd.select_by_index(keep_ids)
        # Save the filtered point cloud to the script's temp folder
        script_dir = os.path.dirname(os.path.realpath(__file__))
        temp_dir = os.path.join(script_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_ply_path = os.path.join(temp_dir, "cloud_tensorrt.ply")
        o3d.io.write_point_cloud(temp_ply_path, pcd)
        logging.info(f"Filtered point cloud saved to {temp_ply_path}")
        o3d.visualization.draw_geometries([pcd])



def parse_args() -> omegaconf.OmegaConf:
    parser = argparse.ArgumentParser(description='Stereo 2025')
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # File options
    parser.add_argument('--left_img', '-l', help='Path to left image.')
    parser.add_argument('--right_img', '-r',  help='Path to right image.')
    parser.add_argument('--save_path', '-s', default=f'{code_dir}/../output', help='Path to save results.')
    parser.add_argument('--pretrained', default='2024-12-13-23-51-11/model_best_bp2.pth', help='Path to pretrained model')

    # Inference options
    parser.add_argument('--height', type=int, default=448, help='Image height')
    parser.add_argument('--width', type=int, default=672, help='Image width')
    parser.add_argument('--pc', action='store_true', help='Save point cloud')
    parser.add_argument('--z_far', default=100, type=float, help='max depth to clip in point cloud')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    args.left_img = "/media/levin/DATA/nerf/new_es8/stereo/250610/colored_l/00000005.png"
    args.right_img = "/media/levin/DATA/nerf/new_es8/stereo/250610/colored_r/00000005.png"
    args.pretrained = "/media/levin/DATA/checkpoints/foundationstereo/foundation_small_288_960_disp64_6_fp16.engine"
    # args.height = 288
    # args.width = 960
    args.pc = True
    args.z_far = 100

    #for pruned model
    # args.pretrained = "/media/levin/DATA/checkpoints/foundationstereo/foundation_small_96_320_disp64_fp16.engine"
    # args.pretrained = "/media/levin/DATA/checkpoints/foundationstereo/foundation_small_288_960_disp64_0_fp16.engine"
    # args.height = 96
    # args.width = 320

    os.makedirs(args.save_path, exist_ok=True)
    paths = ['continuous/disparity', 'visual', 'denoised_cloud', 'cloud']
    for p in paths:
        os.makedirs(os.path.join(args.save_path, p), exist_ok=True)

    assert os.path.isfile(args.pretrained), f'Pretrained model {args.pretrained} not found'
    logging.info('Pretrained model loaded from %s', args.pretrained)
    set_seed(0)
    if args.pretrained.endswith('.onnx'):
        model = get_onnx_model(args)
    elif args.pretrained.endswith('.engine') or args.pretrained.endswith('.plan'):
        model = get_engine_model(args)
    else:
        assert False, f'Unknown model format {args.pretrained}.'

    inference(args.left_img, args.right_img, model, args)

if __name__ == '__main__':
    main()