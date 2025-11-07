import cv2


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# # Paths to the disparity .npy files
# file1 = '/media/levin/DATA/nerf/public_depth/kitti12/training/disp_occ/000000_10.png'
# file2 = '/media/levin/DATA/nerf/public_depth/kitti12/training/disp/000000_10.npy'


# # Path to the original image
# orig_img_path = '/media/levin/DATA/nerf/public_depth/kitti12/training/colored_0/000000_10.png'

# # Paths to the disparity .npy files
# file1 = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp/00000006.npy'
# file2 = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp_test/00000006.npy'


# # Path to the original image
# orig_img_path = '/media/levin/DATA/nerf/new_es8/stereo/250610/left_images/00000006.png'

# Paths to the disparity .npy files
# file1 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_gt_0.npy'
# file2 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_teacher_pred_0.npy'
# file2 = '/home/levin/workspace/temp/FoundationStereo/output/temp/disp_student_pred_0.npy'

file1 = '/media/levin/DATA/nerf/new_es8/stereo/20250702/disp/1751438147.4760577679_test.npy'
file2 = '/media/levin/DATA/nerf/new_es8/stereo/20250702/disp/1751438147.4760577679.npy'



# Path to the original image
orig_img_path = '/media/levin/DATA/nerf/new_es8/stereo/20250702/left_images/1751438147.4760577679.png'

# Load the original image (as RGB)
orig_img = cv2.imread(orig_img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# Load the disparity images
if file1.endswith('.npy'):
    disp1 = np.load(file1)
else:
    disp1 = np.array(Image.open(file1), dtype=np.float32) / 256.0

disp2 = np.load(file2)


# disp1 = disp1[12:-12,: ]
# disp2 = disp2[12:-12,: ]
# orig_img = orig_img[:1000,...]
# crop = 1000
# disp1[crop:, ...] = 0
# disp2[crop:, ...] = 0
# orig_img[crop:, ...] = 0

# disp1[:800, ...] = 0
mask = (disp1 >  0)




assert disp1.shape == disp2.shape, "Disparity images must have identical shapes."
# Compute End-Point Error (EPE)
epe = np.mean(np.abs(disp1 - disp2)[mask])
print(f"EPE (End-Point Error) between the two disparity images: {epe:.4f}")

def compute_D1_metric(gt_disp, pred_disp, threshold=3, percent=0.05):
    """
    Computes the D1 stereo depth evaluation metric.
    D1: Percentage of pixels whose disparity error is larger than 3 pixels and 5% of the ground-truth disparity.
    Args:
        gt_disp (np.ndarray): Ground-truth disparity map.
        pred_disp (np.ndarray): Predicted disparity map.
        threshold (float): Absolute error threshold (default: 3 pixels).
        percent (float): Relative error threshold (default: 5%).
    Returns:
        float: D1 error percentage.
    """
    valid_mask = gt_disp > 0
    abs_err = np.abs(gt_disp - pred_disp)
    rel_err = abs_err > threshold
    rel_percent = abs_err > (percent * np.abs(gt_disp))
    error_mask = valid_mask & rel_err & rel_percent
    if np.sum(valid_mask) == 0:
        return 0.0
    D1 = 100.0 * np.sum(error_mask) / np.sum(valid_mask)
    return D1

D1_error = compute_D1_metric(disp1, disp2)
print(f"D1 error (percentage of bad pixels): {D1_error:.4f}%")
# Compare the arrays
if np.array_equal(disp1[mask], disp2[mask]):
    print("The two disparity images are the same.")
else:
    print("The two disparity images are different.")
    

plt.figure(figsize=(12, 10))

# Show original image
plt.subplot(2, 2, 1)
plt.imshow(orig_img)
plt.title('Original Image')
plt.axis('off')

# Show disparity image 1
plt.subplot(2, 2, 2)
plt.imshow(disp1, cmap='plasma')
plt.title('Disparity Image 1')
plt.colorbar()
plt.axis('off')

# Show disparity image 2
plt.subplot(2, 2, 3)
plt.imshow(disp2, cmap='plasma')
plt.title('Disparity Image 2')
plt.colorbar()
plt.axis('off')

# Show absolute difference
plt.subplot(2, 2, 4)
diff = np.abs(disp1 - disp2)
diff[~mask] = 0  # Mask out invalid regions
plt.imshow(diff, cmap='viridis')
plt.title('Absolute Difference')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()