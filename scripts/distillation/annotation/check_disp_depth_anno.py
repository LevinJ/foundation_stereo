import os
import random
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from traitlets import default

class AnnotationViewer:
    def __init__(self, config_txt, root_dir):
        self.root_dir = Path(root_dir)
        self.annotations = self.load_annotations(config_txt)

    def load_annotations(self, config_txt):
        annotations = []
        with open(config_txt, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    left_rel, right_rel, disp_rel = parts
                    annotations.append({
                        'left': self.root_dir / left_rel,
                        'right': self.root_dir / right_rel,
                        'disp': self.root_dir / disp_rel
                    })
        return annotations

    def pick_random(self, default_idx=None):
        if default_idx is not None:
            idx = default_idx
        else:
            idx = random.randint(0, len(self.annotations) - 1)
        print(f"Selected annotation index: {idx}")
        return self.annotations[idx]

    def show(self, annotation, K=None, baseline=None):
        print(f"Left image path: {annotation['left']}")
        left_img = np.array(Image.open(annotation['left']).convert('RGB'))
        disp_img = np.load(annotation['disp'])
        assert left_img.shape[0] == disp_img.shape[0] and left_img.shape[1] == disp_img.shape[1], \
            f"Image and disparity shape mismatch: left_img {left_img.shape}, disp_img {disp_img.shape}"

        # 1. Calculate depth image
        focal_length = K[0, 0]
        depth_img = np.zeros_like(disp_img, dtype=np.float32)
        valid_disp = disp_img > 2
        depth_img[valid_disp] = focal_length * baseline / disp_img[valid_disp]
        # depth_img[depth_img > 100] = 0  # Cap depth values to 100 meters    

        # 2. Generate point cloud and save
        import open3d as o3d
        H, W = disp_img.shape
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        # Only use valid depth
        mask = valid_disp & (depth_img > 0)
        x = xx[mask]
        y = yy[mask]
        z = depth_img[mask]
        # Backproject to 3D
        X = (x - K[0, 2]) * z / K[0, 0]
        Y = (y - K[1, 2]) * z / K[1, 1]
        Z = z
        points = np.stack([X, Y, Z], axis=1)
        colors = left_img[mask]
        colors = colors / 255.0
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # Save to temp folder
        temp_dir = Path(__file__).parent / 'temp'
        temp_dir.mkdir(exist_ok=True)
        pcd_path = temp_dir / 'cloud.ply'
        o3d.io.write_point_cloud(str(pcd_path), pcd)

        # 3. Display images and point cloud
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.title('Left Image')
        plt.imshow(left_img)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Disparity')
        plt.imshow(disp_img, cmap='plasma')
        plt.colorbar(label='Disparity')
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('Depth')
        plt.imshow(depth_img, cmap='viridis')
        plt.colorbar(label='Depth (m)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

        print(f"Point cloud saved to {pcd_path}")
        o3d.visualization.draw_geometries([pcd], window_name='Point Cloud')

if __name__ == "__main__":
    default_idx = 13  # Set to an integer index to pick a specific annotation
    default_idx = None
    config_txt = '/media/levin/DATA/nerf/new_es8/stereo/20250702/picked_images_eval_anno.txt'
    root_dir = '/media/levin/DATA/nerf/new_es8/stereo/20250702/'
    # default_idx = 0  # Set to an integer index to pick a specific annotation
    # config_txt = '/home/levin/workspace/temp/OpenStereo/data/ZED/zed_250601.txt'
    # root_dir = '/media/levin/DATA/nerf/new_es8/stereo/'
    viewer = AnnotationViewer(config_txt, root_dir)
    annotation = viewer.pick_random(default_idx=default_idx)
    K = np.array([
        [1049.68408203125, 0.0, 998.2841796875],
        [0.0, 1049.68408203125, 589.4127197265625],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    baseline = 0.120
    viewer.show(annotation, K=K, baseline=baseline)
