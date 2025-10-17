import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset

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
        # Load the disparity images
        if sample_info['disp'].endswith('.npy'):
            disp_img = np.load(sample_info['disp']).astype(np.float32)
        else:
            disp_img = np.array(Image.open(sample_info['disp']), dtype=np.float32) / 256.0
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        if self.transform:
            sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = sample_info['left']
        return sample
