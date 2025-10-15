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
from torchvision import transforms

# Define transform pipeline with padding
# class DivisiblePad:
#     def __init__(self, divis_by=32, force_square=False):
#         self.divis_by = divis_by
#         self.force_square = force_square
#     def __call__(self, sample):
#         left_img = sample['left']
#         right_img = sample['right']
#         disp_img = sample['disp']
#         padder = InputPadder(left_img.shape, divis_by=self.divis_by, force_square=self.force_square)
#         left_img_p, right_img_p = padder.pad(left_img, right_img)
#         # Pad disp_img with zeros to match padded shape
#         _, _, h, w = left_img_p.shape
#         disp_shape = disp_img.shape
#         disp_padded = torch.zeros((1, h, w), dtype=disp_img.dtype, device=disp_img.device)
#         # Place original disp_img in top-left corner
#         disp_padded[:, :disp_shape[1], :disp_shape[2]] = disp_img
#         return {
#             'left': left_img_p,
#             'right': right_img_p,
#             'disp': disp_padded,
#             'padder': padder
#         }
class TransposeImage(object):
    def __init__(self):
        return

    def __call__(self, sample):
        sample['left'] = sample['left'].transpose((2, 0, 1))
        sample['right'] = sample['right'].transpose((2, 0, 1))
        return sample
class ToTensor(object):
    def __init__(self):
        return

    def __call__(self, sample):
        for k in sample.keys():
            if isinstance(sample[k], np.ndarray):
                if k == 'super_pixel_label':
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.int32)
                elif k in ['occ_mask', 'occ_mask_2']:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.bool)
                else:
                    sample[k] = torch.from_numpy(sample[k].copy()).to(torch.float32)
        return sample
class DivisiblePad(object):
    def __init__(self, divis_by=32, mode='round'):
        self.by = divis_by
        self.mode = mode

    def __call__(self, sample):
        h, w = sample['left'].shape[:2]
        if h % self.by != 0:
            pad_h = self.by - h % self.by
        else:
            pad_h = 0
        if w % self.by != 0:
            pad_w = self.by - w % self.by
        else:
            pad_w = 0

        if self.mode == 'round':
            pad_top = pad_h // 2
            pad_right = pad_w // 2
            pad_bottom = pad_h - (pad_h // 2)
            pad_left = pad_w - (pad_w // 2)
        elif self.mode == 'tr':
            pad_top = pad_h
            pad_right = pad_w
            pad_bottom = 0
            pad_left = 0
        else:
            raise Exception('no DivisiblePad mode')

        # apply pad for left, right, disp image, and occ mask
        for k in sample.keys():
            if k in ['left', 'right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                sample[k] = np.pad(sample[k], pad_width, 'edge')

            elif k in ['disp', 'disp_right', 'occ_mask', 'occ_mask_right']:
                pad_width = np.array([[pad_top, pad_bottom], [pad_left, pad_right]])
                sample[k] = np.pad(sample[k], pad_width, 'constant', constant_values=0)

        sample['pad'] = [pad_top, pad_right, pad_bottom, pad_left]
        return sample
class ZedDataset(Dataset):
    def __init__(self, config_txt, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
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
        sample_info = self.samples[idx]
        left_img = np.array(Image.open(sample_info['left']).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(sample_info['right']).convert('RGB'), dtype=np.float32)
        disp_img = np.load(sample_info['disp']).astype(np.float32) 

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.transform:
            sample = self.transform(sample)
        return sample




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
    

    # 2. Define Transformations
    data_transforms = transforms.Compose([
        DivisiblePad(divis_by=32, mode='round'),
        TransposeImage(),
        ToTensor(),
    ])
    dataset = ZedDataset(config_txt, root_dir, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Evaluate EPE metric
    total_epe = 0.0
    count = 0
    for batch in dataloader:
        sample = batch
        left_img = sample['left'].cuda()
        right_img = sample['right'].cuda()
        disp_gt = sample['disp'].cuda()
        # padder = sample.get('padder', None)
        # Model inference
        with torch.no_grad():
            pred_disp = model.forward(left_img, right_img, iters=args.valid_iters, test_mode=True)
            # Remove padding if applied
            # if padder is not None:
            #     pred_disp = padder.unpad(pred_disp.float())
        # Compute EPE
        mask = (disp_gt > 0)
        epe = torch.mean(torch.abs(pred_disp.squeeze(1) - disp_gt)[mask]).item()
        total_epe += epe
        count += 1
        print(f"Sample {count}: EPE = {epe}")
    mean_epe = total_epe / count if count > 0 else 0.0
    print(f"Mean EPE over dataset: {mean_epe}")
