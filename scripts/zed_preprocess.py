
import numpy as np
import imageio
import cv2
import torch
from core.utils.utils import InputPadder

class ZedPreprocessor(object):
    def __init__(self, scale=0.5, only_road = True):
        self.K = np.array([
            1049.68408203125, 0.0, 998.2841796875,
            0.0, 1049.68408203125, 589.4127197265625,
            0.0, 0.0, 1.0
        ]).reshape(3, 3)
        self.ori_input_height = 1080
        self.ori_input_width = 1920
        self.crop_start_height = 556+60
        self.crop_end_height = 940+60
        self.crop_start_width = 126
        self.crop_end_width = 1406
        self.img_scale = scale
        self.only_road = only_road
        # self.img_scale = 0.25
    def prepare_img_padder(self, image_path):
        img0 = imageio.imread(image_path, pilmode="RGB")
        img0_orig = img0.copy()
        
        crop = 1000
        img0 =  img0[:crop]
        img0_orig = img0_orig[:crop]

        img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(img0.shape, divis_by=32, force_square=False)
        img0 = padder.pad(img0)[0]
        self.padder = padder
        return img0, img0_orig
    def disp_unpadder(self,disp):
        if self.only_road:
            return disp 
        
        return self.padder.unpad(disp.float())
    def prepare(self, image_path):
        if not self.only_road:
            return self.prepare_img_padder(image_path)
        # Load image
        input_image = imageio.imread(image_path)
        # Crop image using constructor attributes
        cropped_image = input_image[
            self.crop_start_height:self.crop_end_height,
            self.crop_start_width:self.crop_end_width
        ]
        # Resize image using img_scale
        new_height = int(cropped_image.shape[0] * self.img_scale)
        new_width = int(cropped_image.shape[1] * self.img_scale)
        resized_image = cv2.resize(cropped_image, (new_width, new_height))
        # Convert to tensor and reshape as in preprocess
        tensor_image = torch.as_tensor(resized_image.copy()).float()[None].permute(0, 3, 1, 2).contiguous()
        return tensor_image, resized_image
    def prepare_disp(self, disp):
        # Crop disparity using constructor attributes
        cropped_disp = disp[
            self.crop_start_height:self.crop_end_height,
            self.crop_start_width:self.crop_end_width
        ]
        # Resize disparity using img_scale
        new_height = int(cropped_disp.shape[0] * self.img_scale)
        new_width = int(cropped_disp.shape[1] * self.img_scale)
        resized_disp = cv2.resize(cropped_disp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # Multiply disparity values by scale factor
        resized_disp = resized_disp * self.img_scale
        return resized_disp
    def updated_K(self):
        if not self.only_road:
            return self.K
        # Compute crop offsets
        crop_h0 = self.crop_start_height
        crop_w0 = self.crop_start_width
        # Compute scale
        scale = self.img_scale
        # Copy original K
        K = self.K.copy()
        # Update principal point for crop
        K[0, 2] -= crop_w0
        K[1, 2] -= crop_h0
        # Scale focal length and principal point
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K
    def get_baseline(self):
        return 0.120  # in meters  
    def calc_depth(self, K, disparity_map):
        # Compute depth from disparity map
        baseline = self.get_baseline()
        depth_map = (K[0, 0] * baseline) / (disparity_map + 1e-8)  # Add small value to avoid division by zero
        return depth_map 