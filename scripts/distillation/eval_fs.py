# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import time  # Add this import at the top of the file
from typing import Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')

from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


class ZedDataset(Dataset):
    def __init__(self, config_txt, root_dir):
        self.samples = []
        self.root_dir = root_dir
        with open(config_txt, 'r') as f:
            for line in f:
                left, right, disp = line.strip().split()
                self.samples.append({
                    'left': os.path.join(root_dir, left),
                    'right': os.path.join(root_dir, right),
                    'disp': os.path.join(root_dir, disp)
                })
        return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        left_img = np.array(Image.open(sample['left']).convert('RGB'))
        right_img = np.array(Image.open(sample['right']).convert('RGB'))
        disp_gt = np.load(sample['disp'])
        # Convert to torch tensors
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float()
        right_img = torch.from_numpy(
            right_img).permute(2, 0, 1).float()
        disp_gt = torch.from_numpy(disp_gt).float()[None]
        return left_img, right_img, disp_gt




if __name__ == "__main__":
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ckpt_dir', default=f'{code_dir}/../../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
    parser.add_argument('--valid_iters', type=int, default=32,
                        help='number of flow-field updates during forward pass')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)

    ckpt_dir = args.ckpt_dir
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
    if 'vit_size' not in cfg:
        cfg['vit_size'] = 'vitl'
    for k in args.__dict__:
        cfg[k] = args.__dict__[k]
    args = OmegaConf.create(cfg)
    logging.info(f"args:\n{args}")
    logging.info(f"Using pretrained model from {ckpt_dir}")

    model = FoundationStereo(args)

    ckpt = torch.load(ckpt_dir)
    logging.info(
        f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
    model.load_state_dict(ckpt['model'])

    model.cuda()
    model.eval()


    # ZedDataset for evaluation

    # Prepare dataset and dataloader
    config_txt = '/home/levin/workspace/temp/OpenStereo/data/ZED/zed_250601.txt'
    root_dir = '/media/levin/DATA/nerf/new_es8/stereo/'
    dataset = ZedDataset(config_txt, root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate EPE metric
    total_epe = 0.0
    count = 0
    for left_img, right_img, disp_gt in dataloader:
        # Move to GPU
        left_img = left_img.cuda()
        right_img = right_img.cuda()
        disp_gt = disp_gt.cuda()
        # Model inference
        with torch.no_grad():
            # Pad input if needed
            padder = InputPadder(left_img.shape, divis_by=32, force_square=False)
            left_img_p, right_img_p = padder.pad(left_img, right_img)
            pred_disp = model.forward(left_img_p, right_img_p, iters=args.valid_iters, test_mode=True)
            # pred_disp = model(left_img_p, right_img_p)[0].squeeze().cpu()
            # Remove padding if applied
            pred_disp = padder.unpad(pred_disp.float())
        # Compute EPE
        mask = (disp_gt > 0)
        epe = torch.mean(torch.abs(pred_disp - disp_gt)[mask]).item()
        total_epe += epe
        count += 1
        print(f"Sample {count}: EPE = {epe}")
    mean_epe = total_epe / count if count > 0 else 0.0
    print(f"Mean EPE over dataset: {mean_epe}")
