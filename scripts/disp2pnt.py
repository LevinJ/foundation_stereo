"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 05 2025
*  File : disp2pnt.py
******************************************* -->

"""
import dis
import cv2


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Utils import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from scripts.zed_preprocess import ZedPreprocessor

class App(object):
    def resize_disparity(self, disp, scale):
        """
        Resize a disparity image by a given scale factor and multiply by scale.
        Args:
            disp (np.ndarray): Disparity image (2D or 3D array).
            scale (float or int): Scale factor. If int, interpreted as percentage (e.g., 2 means 2x larger).
        Returns:
            np.ndarray: Resized and scaled disparity image.
        """
        if isinstance(scale, int):
            scale_factor = float(scale)
        else:
            scale_factor = scale
        new_height = int(disp.shape[0] * scale_factor)
        new_width = int(disp.shape[1] * scale_factor)
        resized_disp = cv2.resize(disp, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        # Multiply disparity values by scale factor
        resized_disp = resized_disp * scale_factor
        return resized_disp
    def __init__(self):
        return

    def resize_image(self, img, scale):
        """
        Resize the input image by the given scale factor.
        Args:
            img (np.ndarray): Input image.
            scale (float or int): Scale factor. If int, interpreted as percentage (e.g., 3 means 3x larger).
        Returns:
            np.ndarray: Resized image.
        """
        if isinstance(scale, int):
            scale_factor = float(scale)
        else:
            scale_factor = scale
        new_height = int(img.shape[0] * scale_factor)
        new_width = int(img.shape[1] * scale_factor)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return resized_img
    
    def run(self):
        z_far = 100.0
        out_dir = './output'
        # scale = 1.0
        # intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"
        # file1 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_student_pred_0.npy'
        # file1 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_gt_0.npy'
        # orig_img_path = '/media/levin/DATA/nerf/new_es8/stereo/20250702/left_images/1751438147.4760577679.png'
        # file1 = '/home/levin/workspace/temp/FoundationStereo/scripts/temp/rescale/disp.npy'
        # orig_img_path = '/home/levin/workspace/temp/FoundationStereo/scripts/temp/rescale/rbg.png'

        # file_name = '1751438147.4760577679'
        file_name ='1751438145.9038047791'
        file1 = f'/media/levin/DATA/nerf/new_es8/stereo/20250702/disp/{file_name}.npy'
        orig_img_path = f'/media/levin/DATA/nerf/new_es8/stereo/20250702/left_images/{file_name}.png'
        # Load the original image (as RGB)
        # orig_img = cv2.imread(orig_img_path)
        # orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Resize the original image by a given scale
        # orig_img = self.resize_image(orig_img, 3)

        # Load the disparity images
        if file1.endswith('.npy'):
            disp = np.load(file1)
        else:
            disp = np.array(Image.open(file1), dtype=np.float32) / 256.0
        # disp = self.resize_disparity(disp, 1/3)
        zp = ZedPreprocessor(scale=0.5)
        disp = zp.prepare_disp(disp)
        _, orig_img= zp.prepare(orig_img_path)
        K = zp.updated_K().copy()
        baseline = zp.get_baseline()
        depth = K[0, 0]*baseline/disp

        # Display the RGB, disparity, and depth images side by side
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].set_title("RGB Image")
        axes[0].imshow(orig_img)
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
        # plt.savefig(f'{out_dir}/rgb_disp_depth.png')
        plt.show()


        process_and_visualize_point_cloud(
            depth, K, orig_img, out_dir, z_far, False, None, None)

        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
