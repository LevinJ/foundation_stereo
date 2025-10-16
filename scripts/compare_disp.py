import cv2


import numpy as np
import matplotlib.pyplot as plt

# Paths to the disparity .npy files
file1 = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp/00000006.npy'
file2 = '/media/levin/DATA/nerf/new_es8/stereo/250610/disp/00000006_2.npy'

# Path to the original image
orig_img_path = '/media/levin/DATA/nerf/new_es8/stereo/250610/colored_l/00000006.png'

# Load the original image (as RGB)
orig_img = cv2.imread(orig_img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

# Load the disparity images
disp1 = np.load(file1)
disp2 = np.load(file2)

# mask = disp1 <= 0
# disp2[mask] = 0

# crop = 0
# disp1[:crop, :] = 0
# disp2[:crop, :] = 0
# orig_img[:crop, :, :] = 0

assert disp1.shape == disp2.shape, "Disparity images must have identical shapes."
# Compute End-Point Error (EPE)
epe = np.mean(np.abs(disp1 - disp2))
print(f"EPE (End-Point Error) between the two disparity images: {epe}")


# Compare the arrays
if np.array_equal(disp1, disp2):
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
plt.imshow(diff, cmap='viridis')
plt.title('Absolute Difference')
plt.colorbar()
plt.axis('off')

plt.tight_layout()
plt.show()