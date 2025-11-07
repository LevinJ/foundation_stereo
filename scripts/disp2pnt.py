"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 05 2025
*  File : disp2pnt.py
******************************************* -->

"""
import cv2


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from Utils import *

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')


class App(object):
    def __init__(self):
        return
    def run(self):
        z_far = 100.0
        out_dir = './output/temp'
        scale = 1.0
        intrinsic_file = "/media/levin/DATA/nerf/new_es8/stereo_20250331/K_Zed.txt"
        # file1 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_student_pred_0.npy'
        file1 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_gt_0.npy'
        orig_img_path = '/media/levin/DATA/nerf/new_es8/stereo/20250702/left_images/1751438147.4760577679.png'

        # Load the original image (as RGB)
        orig_img = cv2.imread(orig_img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Load the disparity images
        if file1.endswith('.npy'):
            disp = np.load(file1)
        else:
            disp = np.array(Image.open(file1), dtype=np.float32) / 256.0

        disp = disp[12:-12,: ]
        orig_img = orig_img[:1000,...]

        with open(intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = np.array(list(map(float, lines[0].rstrip().split()))).astype(
                np.float32).reshape(3, 3)
            baseline = float(lines[1])
        K[:2] *= scale
        depth = K[0, 0]*baseline/disp


        process_and_visualize_point_cloud(
            depth, K, orig_img, out_dir, z_far, False, None, None)

        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
