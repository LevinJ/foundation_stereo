# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
sys.path.append('/home/levin/workspace/temp/FoundationStereo/scripts')
sys.path.append('/home/levin/workspace/temp/FoundationStereo')
from core.foundation_stereo import *
from Utils import *
from core.utils.utils import InputPadder
from omegaconf import OmegaConf
import os
import sys
import time  # Add this import at the top of the file
from typing import Optional, Tuple
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from  scripts.zed_preprocess import ZedPreprocessor

class InferFS(object):
    def __init__(self):
        return
    def init_model(self):
        set_logging_format()
        set_seed(0)
        torch.autograd.set_grad_enabled(False)

        # ckpt_dir = '/media/levin/DATA/checkpoints/foundationstereo/23-51-11/model_best_bp2.pth'
        # cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/23-51-11/cfg.yaml'


        # ckpt_dir = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/model_best_bp2.pth'
        # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug/ckpt/checkpoint_epoch_20.pth'
        # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug5/ckpt/checkpoint_epoch_999.pth'
        # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug/ckpt/checkpoint_epoch_999.pth'
        # args.ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_scale_0.25_8000/checkpoint_epoch_999.pth'
        # ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_0.5/checkpoint_epoch_500.pth'
        ckpt_dir = '/home/levin/workspace/temp/FoundationStereo/output/ZedDataset/FoundationStereo/fstereo_zed/debug_0.5_50k/checkpoint_epoch_199.pth'
        cfg_file = '/media/levin/DATA/checkpoints/foundationstereo/11-33-40/cfg.yaml'
        cfg = OmegaConf.load(cfg_file)
        if 'vit_size' not in cfg:
            cfg['vit_size'] = 'vitl'
        # for k in args.__dict__:
        #     cfg[k] = args.__dict__[k]
        args = OmegaConf.create(cfg)
        logging.info(f"Using pretrained model from {ckpt_dir}")
        args.vit_size = 'vits'
        args.valid_iters = 4
        args.max_disp = 64

        model = FoundationStereo(args)

        ckpt = torch.load(ckpt_dir)
        global_step = ckpt.get('global_step', 'N/A')
        logging.info(
            f"ckpt global_step:{global_step}, epoch:{ckpt['epoch']}")
        model_state = None
        if 'model' in ckpt:
            model_state = ckpt['model']
        else:
            model_state = ckpt['model_state']
        model.load_state_dict(model_state)

        model.cuda()
        model.eval()
        self.model = model
        self.args = args
        return
    def run_inference(self, left_img_path, right_img_path, zp = ZedPreprocessor(scale=0.5, only_road= False)):
        if not hasattr(self, 'model'):
            self.init_model()
        self.preprocess(zp, left_img_path, right_img_path)
        model = self.model
        args = self.args
        img0 = self.img0
        img1 = self.img1

        for i in range(1):
            start_time = time.time()  # Start timing
            with torch.amp.autocast('cuda', enabled=True):
                disp = model.forward(
                    img0, img1, iters=args.valid_iters, test_mode=True)
                
            end_time = time.time()  # End timing

            # Print the duration
            print(f"Running duration: {end_time - start_time:.2f} seconds")

        H, W = self.img0_ori.shape[:2]
        disp = zp.disp_unpadder(disp)
        disp = disp.data.cpu().numpy().reshape(H, W)
        

        
        yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(
            disp.shape[1]), indexing='ij')
        us_right = xx-disp
        invalid = (us_right < 0) | (disp <= 0)
        disp[invalid] = 0

        
        self.zp = zp
        return disp
    def preprocess(self, zp, left_img_path, right_img_path):
        img0, img0_ori= zp.prepare(left_img_path)
        img1, img1_ori = zp.prepare(right_img_path)
        # H, W = img0.shape[2], img0.shape[3]
        img0 = img0.cuda()
        img1 = img1.cuda()
        self.img0 = img0 
        self.img1 = img1
        self.img0_ori = img0_ori
        self.img1_ori = img1_ori
        return 
    def vis_cloud(self, disp):
        zp = self.zp
        img0_ori = self.img0_ori
        out_dir  = './output'
        z_far = 100
        denoise_cloud = False
        denoise_nb_points = 30
        denoise_radius = 0.03

        K = zp.updated_K().copy()
        baseline = zp.get_baseline()
        depth = K[0, 0]*baseline/disp
        # np.save(f'{args.out_dir}/depth_meter.npy', depth)
        import matplotlib.pyplot as plt

        # Display the RGB, disparity, and depth images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].set_title("RGB Image")
        axes[0].imshow(img0_ori)
        axes[0].axis("off")

        im1 = axes[1].imshow(disp, cmap='plasma')
        axes[1].set_title("Disparity Image")
        axes[1].axis("off")
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Disparity (pixels)")

        im2 = axes[2].imshow(depth)
        axes[2].set_title("Depth Image")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (meters)")

        plt.tight_layout()
        # plt.savefig(f'{args.out_dir}/rgb_and_depth.png')
        plt.show()
        process_and_visualize_point_cloud(
            depth, K, img0_ori, out_dir, z_far, denoise_cloud, denoise_nb_points, denoise_radius)

        return
if __name__ == "__main__":
    obj = InferFS()
    # folder = '/media/levin/DATA/nerf/new_es8/stereo/20250702'
    folder = '/media/levin/DATA/nerf/new_es8/stereo/20251119/8'
    #eval
    # file_name = '1751438147.4760577679.png'
    # file_name = '1751438168.9847328663.png'
    #train
    # file_name ='1751438145.9038047791.png'
    # file_name ='1751438153.062338829.png'

    file_name = '1763538671.3129007816.png'
    left_file = f"{folder}/left_images/{file_name}"
    right_file = f"{folder}/right_images/{file_name}"

    zp = ZedPreprocessor(scale=0.5, only_road= False)
    disp = obj.run_inference(left_file, right_file, zp=zp)
    obj.vis_cloud(disp)