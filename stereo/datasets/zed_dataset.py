import os
from turtle import left
import torch.utils.data as torch_data
import numpy as np
from PIL import Image
from .dataset_template import DatasetTemplate
from stereo.utils.common_utils import get_pos_fullres


class ZedDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        self.use_noc = self.data_info.get('USE_NOC', False)
        if hasattr(self.data_info, 'RETURN_POS'):
            self.retrun_pos = self.data_info.RETURN_POS
        else:
            self.retrun_pos = False
    def crop_image(self, img):
        crop_start_height = 556
        crop_end_height = 940
        crop_start_width = 126
        crop_end_width = 1406
        return img[crop_start_height:crop_end_height, crop_start_width:crop_end_width]

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        if len(full_paths) != 3:
            #in case we have no gt (distillation mode)
            left_img_path, right_img_path = full_paths
        else:
            left_img_path, right_img_path, disp_img_path = full_paths
        if self.use_noc:
            disp_img_path = disp_img_path.replace('disp_occ', 'disp_noc')
        # image
        left_img = np.array(Image.open(left_img_path).convert('RGB'), dtype=np.float32)
        right_img = np.array(Image.open(right_img_path).convert('RGB'), dtype=np.float32)
        # disp
        if len(full_paths) != 3:
            disp_img = np.zeros((left_img.shape[0], left_img.shape[1]), dtype=np.float32)
        else:
            disp_img = np.load(disp_img_path)
        # crop = 1000
        # left_img = left_img[:crop, :, :]
        # right_img = right_img[:crop, :, :]
        # disp_img = disp_img[:crop, :]
        left_img = self.crop_image(left_img)
        right_img = self.crop_image(right_img)
        disp_img = self.crop_image(disp_img)
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
        }
        
        if self.return_right_disp:
            disp_img_right_path = disp_img_path.replace('c_0', 'c_1')
            disp_img_right = np.array(Image.open(disp_img_right_path), dtype=np.float32) / 256.0
            sample['disp_right'] = disp_img_right

        if self.retrun_pos and self.mode == 'training':
            sample['pos'] = get_pos_fullres(800, sample['left'].shape[1], sample['left'].shape[0])

        sample = self.transform(sample)
        sample['index'] = idx
        sample['name'] = left_img_path

        return sample
